# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded geometry proposals, common validation, new UV and CPU Cycles transfer.

Optional CuMesh geometry proposals never bypass common surface/component gates.
Only Blender's bundled NumPy is used. The source GLB is never modified.
"""
import argparse
from array import array
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import bpy
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

sys.path.insert(0,str(Path(__file__).resolve().parent))
from production_projection import exact_face_position_index, exact_surviving_source_face, _proxy_paint_batch
from production_projection import low_triangle_cache, LOW_TRIANGLE_CACHE_BYTES, native_owner_batch_plan, decode_native_owner_batch
from production_projection import SourceFaceMismatch, Q1SelfGeometricIneligible, q1_self_interiors, q1_self_uv
from production_preflight import map_grid_plan
from conservative_repair import _split_glb, _write_glb, _zero_area
from production_projection import admit_oriented_auxiliary, AUXILIARY_CHANGE_DTYPE, AUXILIARY_FALLBACK_ROWS, OrientedSourceRejected, ORIENTED_RECORD_DTYPE, ORIENTED_STRATEGIES, ORIENTED_METADATA_RESERVE, oriented_resource_bytes, oriented_provenance_schema, oriented_numeric_record, oriented_position_records, source_shading_cache, source_shading_normal, planned_base_shading_normals, source_face_polygon_values, decode_source_face_witness, geometric_world_normals, original_source_face_ids, oriented_source, position_sampler_change_reason, position_refinement_rows, _auxiliary_source_sampler, _AuxiliaryProposalRejected, decode_native_owners, sample_world, position_domain, proxy_triangle, object_normals_to_final, source_object_world, target_normal_errors, PNG8_WORLD_ERROR_BOUND, MAX_PROXY_SAMPLES, MAX_ATLAS_PIXELS
from production_uv import validate_uv_triangles, float32_uv_area, separate_float32_collapse, separate_packed_overlap, reconcile_uv_vertices, align_unseeded_charts, UVRasterSeedError, UVTriangleOverlapError, native_chart_seed_report, regular_chart_clearance, face_recovery_part_indices, chart_bilinear_support, UVFilterSupportError
from production_materials import new_map, pixel_array, save_map, set_proxy, finish_material, source_proxies, orm_proxy, transfer_channels, finalize_baked_channel, geometric_ao_rows
from production_topology import stage_duplicate_faces, components, partition_interface_edges, partition_interface_vertices, partition_interfaces_preserved
from production_tangents import TangentProjectionZero, singleton_cyclic_choice, plan_tangents, frame_repair_metrics, normal_conversion_samples, convert_normal_samples, pad_normal, predict_serialization, restore_serialized_glb_frames, triangle_samples


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def note(stage, **details):
    print('MESH_FINISHING '+json.dumps(dict(stage=stage,**details)),flush=True)


def progress_marker(config,stage):
    nonce=config.get('_progress_nonce')
    if nonce is not None:print(f'MESH_FINISHING_PROGRESS {nonce} {stage}',flush=True)


def write_json_atomic(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def write_completion_metrics(path, result):
    """Bound the completion envelope; retain large diagnostics beside it.

    The runner never follows the relative detail pointer. Candidate/source,
    cancellation and budget decisions still use the ordinary sentinel fields.
    """
    limit=4*1024**2
    if len((json.dumps(result,indent=2)+'\n').encode('utf-8'))<=limit:
        write_json_atomic(path,result);return
    # Both names are derived locally, never from report-supplied paths.
    details=path.with_name(path.stem+'-details.json')
    if path.parent.resolve()!=path.parent or any(p.is_symlink() or p.exists() for p in (details,details.with_suffix(details.suffix+'.tmp'))):
        raise ValueError('Completion detail destination must be a fresh physical job-local path')
    write_json_atomic(details,result)
    def scalars(value,names):
        return {key:value[key] for key in names if key in value and type(value[key]) in (str,int,float,bool,type(None))}
    summary=scalars(result,('schema_version','status','operation','source_sha256','candidate_sha256','blender_version','device','elapsed_seconds','fidelity_notice'))
    summary['timing']=scalars(result.get('timing',{}),('source_import_and_geometry_seconds','CPU_finalization_including_UV_PBR_export_reimport_seconds'))
    summary['validation']=scalars(result.get('validation',{}),('geometry','appearance','modly_ui'))
    if 'geometry' in result:
        summary['geometry']=scalars(result['geometry'],('status','mesh_instances','native_triangles'))
    if 'source_preparation' in result:
        summary['source_preparation']=scalars(result['source_preparation'],('operation','raw_triangles','prepared_triangles','removed_exact_zero_area_triangles','imported_triangles','partitioned_triangles'))
    if 'simplification' in result:
        summary['simplification']=scalars(result['simplification'],('source_triangles','triangles','triangle_budget','budget_met','output_changed','uv_changed','operation','requested_backend','geometry_backend_scope','protected_components','protected_triangles','mutable_source_triangles','mutable_triangle_budget','budget_floor_reason'))
    summary['bakes']=[]
    for bake in result.get('bakes',[]):
        item=scalars(bake,('method','complete','occupied_texels','hit_texels','miss_texels','filled_texels','coverage_filled_texels','sample_count_scope','final_material'))
        item['sampling_grids']=[scalars(grid,('resolution','occupied_texels','hit_texels','miss_texels','filled_texels','coverage_filled_texels','recovered_texels','complete')) for grid in bake.get('sampling_grids',[])]
        summary['bakes'].append(item)
    if 'reimport' in result:
        summary['reimport']=scalars(result['reimport'],('triangles','mesh_objects'))
        comparisons=result['reimport'].get('sampled_geometry',{})
        summary['reimport']['sampled_geometry_valid']=bool(comparisons) and all(value.get('valid') is True for value in comparisons.values())
        summary['reimport'].update(scalars(result['reimport'],('effective_geometry_valid',)))
        if 'effective_geometry' in result['reimport']:summary['reimport']['effective_geometry_methods']=sorted({v['method'] for v in result['reimport']['effective_geometry'].values()})
    summary['attempts']=[]
    for attempt in result.get('attempts',[]):
        item=scalars(attempt,('status','reason','directory','failure_path','occupied_texels','hit_texels','miss_texels'))
        if 'simplification' in attempt and isinstance(attempt['simplification'],dict):
            item['simplification']=scalars(attempt['simplification'],('triangles','triangle_budget','budget_met'))
        summary['attempts'].append(item)
    summary['attempt_count']=len(summary['attempts'])
    summary['rejected_attempt_count']=sum(item['status']=='rejected' for item in summary['attempts'])
    summary['metrics_transport']=dict(mode='bounded_summary',details_path=details.name,details_sha256=digest(details),details_bytes=details.stat().st_size,
        scope='Selected completion scalars only; full original metrics in the atomic job-local detail file; no automatic dereference')
    if len((json.dumps(summary,indent=2)+'\n').encode('utf-8'))>limit:raise ValueError('Selected completion metrics still exceed the unchanged transport limit')
    write_json_atomic(path,summary)


def arrays(mesh):
    mesh.calc_loop_triangles()
    vertices=np.empty((len(mesh.vertices),3),np.float32)
    faces=np.empty((len(mesh.loop_triangles),3),np.int32)
    loops=np.empty_like(faces)
    mesh.vertices.foreach_get('co',vertices.ravel())
    mesh.loop_triangles.foreach_get('vertices',faces.ravel())
    mesh.loop_triangles.foreach_get('loops',loops.ravel())
    if not np.isfinite(vertices).all():raise ValueError('Nonfinite source/candidate geometry')
    return vertices,faces,loops


def make_object(name, vertices, faces, matrix):
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(vertices.tolist(),[],faces.tolist());mesh.update()
    obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj)
    # Avoid an Euler roundtrip of the inherited authored rotation (notably pi).
    obj.rotation_mode='QUATERNION'
    obj.matrix_world=matrix.copy()
    return obj


def remove(obj):
    mesh=obj.data;bpy.data.objects.remove(obj,do_unlink=True)
    if not mesh.users:bpy.data.meshes.remove(mesh)


def surface(obj, *, retain_serialized_world=False):
    vertices,faces,_=arrays(obj.data)
    matrix=np.array(obj.matrix_world,dtype=np.float64)
    world=vertices.astype(np.float64)@matrix[:3,:3].T+matrix[:3,3]
    if not len(faces):raise ValueError('Empty candidate')
    from reconstruction_fidelity import surface_points
    points=surface_points(world,faces)
    lower=world.min(axis=0);upper=world.max(axis=0)
    result=dict(triangles=len(faces),bounds_min=lower.tolist(),bounds_max=upper.tolist(),
                diagonal=float(np.linalg.norm(upper-lower)),samples=points,
                bvh=BVHTree.FromPolygons(world.tolist(),faces.tolist(),all_triangles=True),
                matrix_world_float64=matrix.copy(),
                stencil_faces=np.linspace(0,len(faces)-1,min(2048,len(faces)),dtype=int),
                stencil_local_corners=vertices[faces[np.linspace(0,len(faces)-1,min(2048,len(faces)),dtype=int)]].copy())

    if retain_serialized_world:result.update(serialized_world_vertices=world,serialized_faces=faces)
    return result


def compare(source, candidate, config):
    cap=min(config['max_error_world'],config['max_error_relative']*source['diagonal'])
    def maximum(points,bvh,*,forward_witness=False):
        result=0.;face=None;eligible=True;above=0
        stencil=source.get('stencil_faces',());count=len(stencil)
        for index,point in enumerate(points):
            hit=bvh.find_nearest(Vector(point))
            if hit[0] is None or not math.isfinite(hit[3]):raise ValueError('Nearest surface query failed')
            result=max(result,float(hit[3]))
            if forward_witness and hit[3]>cap:
                above+=1
                if not count or index>=7*count:eligible=False
                else:
                    current=int(stencil[index%count])
                    if face is not None and face!=current:eligible=False
                    face=current
        return (result,face,above,eligible) if forward_witness else result
    forward,face,above,eligible=maximum(source['samples'],candidate['bvh'],forward_witness=True)
    backward=maximum(candidate['samples'],source['bvh'])
    bounds=max(abs(a-b) for key in ('bounds_min','bounds_max') for a,b in zip(source[key],candidate[key]))
    result=dict(valid=max(forward,backward,bounds)<=cap,error_cap_world=cap,
                source_to_low_max_sample_world=forward,low_to_source_max_sample_world=backward,
                bounds_shift_world=bounds,source_samples=len(source['samples']),low_samples=len(candidate['samples']),
                semantics='Deterministic bidirectional sampled surface distance, not exact Hausdorff or a visual pass')
    if (config.get('simplification_backend') in ('cumesh_install_if_needed','cumesh_existing') and
        eligible and above and face is not None and backward<=cap and bounds<=cap):
        ordinal=int(np.flatnonzero(source['stencil_faces']==face)[0])
        corners=source['stencil_local_corners'][ordinal]
        result['source_triangle_witness']=dict(face=face,above_cap_samples=above,
            ordered_local_float32_corner_bits=corners.view(np.uint32).tolist(),
            source_matrix_world_float64=source['matrix_world_float64'].tolist(),
            scope='One ordinary deterministic stencil face; no axial/bounds/reverse failure')
    return result


def validate_source_triangle_counts(raw, imported, partitioned, *, removed_exact_zero_area_triangles=0):
    prepared=raw-removed_exact_zero_area_triangles
    note('source_preparation',raw_triangles=raw,prepared_triangles=prepared,imported_triangles=imported,
         partitioned_triangles=partitioned,removed_exact_zero_area_triangles=removed_exact_zero_area_triangles)
    if imported!=prepared:raise ValueError(f'Blender import changed source triangle count: raw={raw}, imported={imported}, partitioned={partitioned}, prepared={prepared}')
    if partitioned!=imported:raise ValueError(f'Source partition changed triangle count: raw={raw}, imported={imported}, partitioned={partitioned}')


def source_parts(originals, plan, protected, *, preserve_geometry=False, geometry_only=False, protect_blender_interfaces=False):
    """Keep source corners/material graphs intact; normalize only the LOW copy."""
    result=[]
    for original in originals:
        interface_parts=[] if protect_blender_interfaces else None
        vertices,faces,loops=arrays(original.data)
        uv=None
        if not geometry_only:
            uv=np.empty((len(original.data.loops),2),np.float32)
            original.data.uv_layers.active.data.foreach_get('uv',uv.ravel())
        normals=np.empty((len(original.data.loops),3),np.float32)
        original.data.corner_normals.foreach_get('vector',normals.ravel())
        if (uv is not None and not np.isfinite(uv).all()) or not np.isfinite(normals).all():raise ValueError('Nonfinite source UV/normals')
        material_ids=np.zeros(len(faces),np.int32) if geometry_only else np.array([original.data.polygons[t.polygon_index].material_index for t in original.data.loop_triangles])
        for slot in sorted(set(material_ids)):
            material=None if geometry_only else original.data.materials[int(slot)]
            if not geometry_only and material is None:raise ValueError('Missing source material')
            # Preflight binds material document to the importer-generated material name.
            source_index=None if geometry_only else material.get('mesh_finishing_material_index')
            if not geometry_only and source_index is None:raise ValueError('Imported source material identity is ambiguous')
            selected=np.flatnonzero(material_ids==slot)
            groups=components(vertices,faces[selected])
            small=[g for g in groups if len(g)<=protected]
            mutable=[g for g in groups if len(g)>protected]
            # Preserve small disconnected pieces together, unchanged, without creating
            # thousands of Blender objects. Their source face lists remain one-to-one.
            batches=[(g,False,[len(g)]) for g in mutable]
            original_component_ids={id(group):index for index,group in enumerate(groups)}
            batch_component_ids=[[original_component_ids[id(group)]] for group in mutable]
            if small:
                batches.append((np.concatenate(small),True,list(map(len,small))))
                batch_component_ids.append([original_component_ids[id(group)] for group in small])
            if len(batches)>256 or len(result)+len(batches)>1024:raise ValueError('Component/object partition exceeds the bounded Blender part count')
            for ordinal,(indices,is_small,sizes) in enumerate(batches):
                fi=selected[indices];rows,mapping=np.unique(faces[fi],return_inverse=True)
                high=make_object(f'HIGH_{original.name}_{slot}_{ordinal}',vertices[rows],mapping.reshape(-1,3),original.matrix_world)
                if uv is not None:
                    layer=high.data.uv_layers.new(name=original.data.uv_layers.active.name)
                    layer.data.foreach_set('uv',uv[loops[fi]].ravel())
                for face in high.data.polygons:face.use_smooth=True
                high.data.normals_split_custom_set(normals[loops[fi]].reshape(-1,3).tolist())
                if material is not None:high.data.materials.append(material)
                if is_small:
                    low=high.copy();low.data=high.data.copy();bpy.context.collection.objects.link(low)
                else:
                    v,inv=np.unique(vertices[rows],axis=0,return_inverse=True)
                    low=make_object('LOW',v,inv[mapping].reshape(-1,3),original.matrix_world)
                    for face in low.data.polygons:face.use_smooth=True
                    if preserve_geometry:low.data.normals_split_custom_set(normals[loops[fi]].reshape(-1,3).tolist())
                low.name=f'LOW_{original.name}_{slot}_{ordinal}';low.data.materials.clear()
                ends=np.r_[0,np.cumsum(sizes)]
                result.append(dict(high=high,low=low,object=original.name,material=source_index,
                                   _source_lineage=(high.as_pointer(),low.as_pointer(),original.name,source_index,tuple(batch_component_ids[ordinal]),tuple((int(ends[k]),int(ends[k+1]),int(cid)) for k,cid in enumerate(batch_component_ids[ordinal]))),
                                   protected=is_small,component_sizes=sizes,source_triangles=len(fi),
                                   source_component_ids=batch_component_ids[ordinal],
                                   source_component_face_ranges=[[int(ends[k]),int(ends[k+1]),int(cid)] for k,cid in enumerate(batch_component_ids[ordinal])],
                                   source_component_domain='original_geometry_census_before_batching' if geometry_only else 'original_material_census_before_batching'))
                if interface_parts is not None:interface_parts.append((result[-1],fi))
        if interface_parts is not None and len(interface_parts)>1:
            started=time.monotonic();owners=np.full(len(faces),-1,np.int32)
            for index,(_,fi) in enumerate(interface_parts):owners[fi]=index
            interfaces=partition_interface_edges(vertices,faces,owners);anchored=0;edge_count=0
            for index,(part,_) in enumerate(interface_parts):
                edges=interfaces.get(index)
                if part['protected'] or edges is None:continue
                low=part['low'];local_vertices,_,_=arrays(low.data)
                part['_partition_interface_edges']=edges
                part['_partition_interface_vertices']=partition_interface_vertices(local_vertices,edges)
                part['_partition_interface_vertex_count']=len(local_vertices)
                part['_partition_interface_edge_sha256']=hashlib.sha256(edges.tobytes()).hexdigest()
                anchored+=1;edge_count+=len(edges)
            note('partition_interfaces',object=original.name,parts=anchored,part_edge_occurrences=edge_count,
                 local_incidence=1,elapsed_seconds=time.monotonic()-started)
        original.hide_render=True;original.hide_viewport=True
    return result


def simplification_ratios(config, total, *, protected_triangles=0):
    mutable=total-protected_triangles
    target=max(0,config['triangle_budget']-protected_triangles)
    # No positive mutable allowance: keep the existing conservative schedule.
    # Never treat ratio zero as permission to delete geometry.
    ratio=min(1.,target/mutable) if mutable>0 and target>0 else min(1.,config['triangle_budget']/total)
    return sorted({ratio,min(1.,max(ratio*2,math.sqrt(ratio))),1.})


def _pristine_low_identity(low, mesh):
    """Geometry/smooth/matrix identity of the HIGH-derived, pre-UV LOW input."""
    vertices,faces,_=arrays(mesh);smooth=np.empty(len(mesh.polygons),bool)
    mesh.polygons.foreach_get('use_smooth',smooth)
    h=hashlib.sha256()
    for data in (vertices,faces,smooth,np.asarray(low.matrix_world,np.float64)):h.update(data.tobytes())
    return h.hexdigest()


_CUMESH_RUNTIME = None


def _cumesh_output(path, report_path, input_sha, source_counts, target):
    """Admit only complete, typed, hashed helper results; never a stale checkpoint."""
    import zipfile
    if report_path.stat().st_size>1024**2:raise ValueError('CuMesh report exceeds bounded admission')
    report=json.loads(report_path.read_text())
    if (report.get('input_sha256')!=input_sha or report.get('output_sha256')!=digest(path)
        or (report.get('source_vertices'),report.get('source_faces'))!=source_counts
        or report.get('target_faces')!=target):raise ValueError('CuMesh helper result identity differs; no Blender fallback')
    if report.get('status')=='candidate_rejected':
        rejected=report.get('rejected_step') or {};counts=rejected.get('read_counts')
        if (report.get('candidate_rejected') is not True or report.get('usable_mesh') is not False
            or report.get('output_role')!='diagnostic_last_valid_checkpoint_only'
            or rejected.get('reason_code') not in ('EMPTY_GEOMETRY','ZERO_AREA_TRIANGLE')
            or report.get('error')!=rejected.get('reason_code') or rejected.get('phase')!='measured_geometry'
            or rejected.get('classification')!='measured_geometric_candidate_rejected'
            or not isinstance(counts,list) or len(counts)!=2 or any(type(n)is not int or n<0 for n in counts)
            or rejected.get('native_counts')!=counts):raise ValueError('Unknown CuMesh rejection/fatal state; no checkpoint admission')
        vinfo=rejected.get('arrays',{}).get('vertices',{});finfo=rejected.get('arrays',{}).get('faces',{})
        if (vinfo.get('dtype')!='float32' or finfo.get('dtype')!='int32' or vinfo.get('shape')!=[counts[0],3]
            or finfo.get('shape')!=[counts[1],3] or vinfo.get('finite') is not True or finfo.get('finite') is not True
            or counts[0]>6000000 or counts[1]>source_counts[1]
            or (rejected['reason_code']=='EMPTY_GEOMETRY')!=(0 in counts)
            or (counts[1]>0 and (finfo.get('index_min') is None or finfo.get('index_max') is None or not 0<=finfo['index_min']<=finfo['index_max']<counts[0]))):raise ValueError('CuMesh rejection diagnostics contain structural/unknown errors')
        return None,None,report  # Observed failed proposal, never load/promote its older checkpoint.
    if (report.get('status') not in ('completed','budget_unmet') or report.get('error') is not None
        or report.get('usable_mesh') is not True or report.get('output_role')!='validated_geometry_proposal'):
        raise ValueError('CuMesh helper failed or returned only a checkpoint; no Blender fallback')
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist()
        if (len(entries)!=2 or {e.filename for e in entries}!={'vertices.npy','faces.npy'}
            or any(e.file_size>96*1024**2 for e in entries) or sum(e.file_size for e in entries)>128*1024**2):raise ValueError('CuMesh output archive resource/layout refusal')
    with np.load(path,allow_pickle=False,max_header_size=10000) as saved:vertices,faces=saved['vertices'],saved['faces']
    if (vertices.dtype!=np.float32 or faces.dtype!=np.int32 or vertices.ndim!=2 or vertices.shape[1:]!=(3,)
        or faces.ndim!=2 or faces.shape[1:]!=(3,) or not 0<len(vertices)<=6000000 or not 0<len(faces)<=source_counts[1]
        or not np.isfinite(vertices).all() or faces.min()<0 or faces.max()>=len(vertices)
        or report.get('final_vertices')!=len(vertices) or report.get('final_faces')!=len(faces)
        or report.get('budget_met')!=(len(faces)<=target)
        or (report['status']=='completed')!=(len(faces)<=target)):raise ValueError('CuMesh output dtype/count/index/status refusal')
    for start in range(0,len(faces),65536):
        triangle=vertices[faces[start:start+65536]].astype(np.float64)
        if np.any(np.linalg.norm(np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]),axis=1)==0):raise ValueError('CuMesh emitted zero-area geometry')
    return vertices,faces,report


def _cumesh_bind_witness(part, pristine, witness, identity):
    """Bind one HIGH stencil face to the SAME original LOW ordinal/corners."""
    ids=part.get('source_component_ids',())
    if len(ids)!=1 or type(ids[0])is not int:return None
    if not np.array_equal(np.asarray(part['low'].matrix_world,np.float64),np.asarray(witness.get('source_matrix_world_float64'),np.float64)):return None
    vertices,faces,_=arrays(pristine);face=witness.get('face')
    if type(face)is not int or not 0<=face<len(faces):return None
    corners=vertices[faces[face]].copy().view(np.uint32)
    if corners.tolist()!=witness.get('ordered_local_float32_corner_bits'):return None
    return dict(witness,source_component_ordinal=ids[0],object=part['object'],material=part['material'],
                source_pristine_geometry_sha256=identity,source_face_ordinal=face,
                index_space='Pristine HIGH-derived LOW triangulated ordinal; exact ordered local F32 corners')


def _cumesh_frozen_input(vertices, faces, witness):
    """Only change the three indexed rows of one authenticated source face."""
    face=witness['source_face_ordinal']
    if (vertices.dtype!=np.float32 or faces.dtype!=np.int32 or type(face)is not int or not 0<=face<len(faces)
        or len(vertices)+3>6000000 or len(np.unique(faces[face]))!=3):raise ValueError('Protected CuMesh face admission failed')
    corners=vertices[faces[face]].copy()
    if corners.view(np.uint32).tolist()!=witness['ordered_local_float32_corner_bits']:raise ValueError('Protected CuMesh face identity differs')
    result=np.concatenate((vertices,corners));indices=faces.copy()
    indices[face]=np.arange(len(vertices),len(vertices)+3,dtype=np.int32)
    # Untouched rows are copied; duplicated rows contain the exact original bits.
    return result,indices


def _cumesh_frozen_retained(vertices, faces, witness):
    corners=np.asarray(witness['ordered_local_float32_corner_bits'],dtype=np.uint32);count=0
    for start in range(0,len(faces),65536):
        count+=int(np.count_nonzero(np.all(vertices[faces[start:start+65536]].view(np.uint32)==corners,axis=(1,2))))
        if count>1:return False
    return count==1


def _cumesh_reduction(pristine, value, config, *, protected_face=None, target=None):
    """Owned GPU request; no process-group change or implicit fallback."""
    import os
    import subprocess
    import tempfile
    import cumesh_runtime as runtime  # stdlib manager only; Blender default never imports it
    import cumesh_simplify as gpu_helper  # stdlib control plane, not Torch
    global _CUMESH_RUNTIME
    started=time.monotonic();deadline=config['_geometry_deadline'];remaining=deadline-started
    if remaining<1:raise TimeoutError('Geometry job deadline exhausted before CuMesh')
    vertices,faces,_=arrays(pristine)
    if not 0<len(vertices)<=6000000 or not 0<len(faces)<=2000000:raise ValueError('CuMesh input mesh admission exceeded')
    nominal_target=max(1,math.floor(len(faces)*value))
    if protected_face is not None:vertices,faces=_cumesh_frozen_input(vertices,faces,protected_face)
    if target is None:target=max(1,math.floor(len(faces)*value))
    if type(target)is not int or not 1<=target<=len(faces):raise ValueError('Invalid exact CuMesh trial target')
    provision_seconds=0.
    if _CUMESH_RUNTIME is None:
        provision_start=time.monotonic()
        python=runtime.ensure_runtime(allow_install=config.get('simplification_backend')=='cumesh_install_if_needed',timeout_seconds=min(3600,int(remaining)))
        _CUMESH_RUNTIME=(python,python.parent.parent.parent)
        provision_seconds=time.monotonic()-provision_start
    python,recipe_root=_CUMESH_RUNTIME
    directory=Path(config['_geometry_directory']);directory.mkdir(parents=True,exist_ok=True)
    trial=Path(tempfile.mkdtemp(prefix='trial-',dir=directory));input_path=trial/'input.npz';output=trial/'output.npz';report_path=trial/'result.json'
    np.savez(input_path,vertices=vertices,faces=faces);input_sha=digest(input_path)
    remaining=deadline-time.monotonic()
    if remaining<1:raise TimeoutError('Geometry job deadline exhausted after runtime verification')
    seconds=max(1,min(600,int(remaining)));helper=Path(__file__).with_name('cumesh_simplify.py')
    command=[str(python),'-I','-B',str(helper),'--input',str(input_path),'--output',str(output),'--report',str(report_path),
             '--target',str(target),'--max-steps','128','--seconds',str(seconds),'--parent-pid',str(os.getpid())]
    session=config.get('_cumesh_session')
    invocation=dict(command=command,expected_parent_pid=os.getpid(),inherited_pgid=os.getpgrp(),external_timeout_seconds=min(remaining,seconds+10))
    if session is not None:invocation['command_scope']='Equivalent one-shot request; actual child launch recorded in stage_session'
    write_json_atomic(trial/'invocation.json',invocation)
    child_start=time.monotonic()
    oom_message=f'CuMesh reported an out-of-memory error. Free memory or select Blender explicitly; no Blender fallback; see {trial}'
    try:
        if session is not None:
            child=session.run(command,env=runtime._environment(recipe_root),timeout=min(deadline-time.monotonic(),seconds+10),log_path=trial/'helper.log')
        else:
            with (trial/'helper.log').open('wb') as log:
                child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=runtime._environment(recipe_root),timeout=min(remaining,seconds+10))
    except subprocess.TimeoutExpired:
        write_json_atomic(trial/'terminal.json',dict(status='timeout',elapsed_seconds=time.monotonic()-child_start))
        raise RuntimeError(f'CuMesh helper deadline exceeded; no Blender fallback; see {trial}')
    except Exception as error:
        failure=dict(status='failed',error_class=type(error).__name__,elapsed_seconds=time.monotonic()-child_start)
        if isinstance(error,gpu_helper.CuMeshOutOfMemory):
            if session is not None and gpu_helper.bound_oom_report(report_path,input_sha,parent_pid=os.getpid(),
                    helper_pid=session.invocation['pid'],request=session.invocation['request_index']):
                write_json_atomic(trial/'terminal.json',dict(failure,error_code=gpu_helper.TORCH_OOM))
                raise RuntimeError(oom_message) from error
            write_json_atomic(trial/'terminal.json',failure)
            raise RuntimeError(f'CuMesh failure receipt identity differs; no Blender fallback; see {trial}') from error
        write_json_atomic(trial/'terminal.json',failure)
        raise
    finally:
        if session is not None:
            invocation['stage_session']=dict(session.invocation)
            write_json_atomic(trial/'invocation.json',invocation)
    child_seconds=time.monotonic()-child_start
    terminal=dict(returncode=child.returncode,elapsed_seconds=child_seconds,scope='request completed; stage child still owned' if session is not None else 'one-shot process exit')
    if session is None and child.returncode==1 and gpu_helper.bound_oom_report(report_path,input_sha,parent_pid=os.getpid()):
        write_json_atomic(trial/'terminal.json',dict(terminal,status='failed',error_code=gpu_helper.TORCH_OOM))
        raise RuntimeError(oom_message)
    write_json_atomic(trial/'terminal.json',terminal)
    if child.returncode:raise RuntimeError(f'CuMesh GPU helper failed ({child.returncode}); no Blender fallback; see {trial}')
    if digest(input_path)!=input_sha:raise ValueError('CuMesh changed pristine input archive')
    vertices,faces,report=_cumesh_output(output,report_path,input_sha,(len(vertices),len(faces)),target)
    # Native helper elapsed includes imports/device setup; do not mislabel it kernel-only time.
    return vertices,faces,dict(nominal_ratio=value,nominal_ratio_target=nominal_target,
        actual_target=target,protected_target_override=protected_face is not None,
        runtime_provision_or_verify_seconds=provision_seconds,child_wall_seconds=child_seconds,
        child_startup_and_supervision_overhead_seconds=max(0.,child_seconds-report['elapsed_seconds']),
        child_timing_scope=('Stage request wall includes private transport; first request includes process startup; report includes per-request runtime checks and geometry' if session is not None else 'Report elapsed includes imports/device initialization and GPU geometry; overhead is process launch/report exit only'),
        total_geometry_proposal_seconds=time.monotonic()-started,report=str(report_path),native=report)


def _evaluate_pristine_reduction(part, pristine, source, value, config, identity, *, protected_face=None, target=None):
    """One pristine proposal, followed by identical common gates for both backends."""
    started=time.monotonic();low=part['low'];selected=low.data;matrix=np.asarray(low.matrix_world,np.float64).copy()
    if _pristine_low_identity(low,pristine)!=identity:raise ValueError('Pristine LOW input changed before a geometry trial')
    trial=pristine.copy();returned=False;modifier_name=None;gpu=None;temporary_group=None;interfaces=None
    temporary_group_name=None;prior_groups=None
    backend=config.get('simplification_backend','blender')
    try:
        low.data=trial
        if value<1 and backend in ('cumesh_install_if_needed','cumesh_existing'):
            if protected_face is None:vertices,faces,gpu=_cumesh_reduction(pristine,value,config)
            else:vertices,faces,gpu=_cumesh_reduction(pristine,value,config,protected_face=protected_face,target=target)
            if protected_face is not None:
                retained=vertices is not None and _cumesh_frozen_retained(vertices,faces,protected_face)
                gpu['protected_source_triangle']=dict(protected_face,retained_exactly_once=bool(retained),requested_triangles=target)
                if vertices is not None and not retained:
                    if not np.array_equal(matrix,np.asarray(low.matrix_world,np.float64)) or _pristine_low_identity(low,pristine)!=identity:raise ValueError('Rejected protection changed pristine LOW or matrix')
                    return None,dict(ratio=value,triangles=len(faces),valid=False,component_retained=False,has_candidate_mesh=False,
                        rejection='protected_source_triangle_not_retained',requested_backend=backend,effective_backend='cumesh_proposal_rejected',
                        source_pristine_geometry_sha256=identity,source_triangles=part['source_triangles'],matrix_unchanged=True,
                        common_geometry_validation='NOT_RUN_NO_PROTECTED_FACE',GPU_geometry=gpu,trial_elapsed_seconds=time.monotonic()-started)
            if vertices is None:
                if not np.array_equal(matrix,np.asarray(low.matrix_world,np.float64)) or _pristine_low_identity(low,pristine)!=identity:raise ValueError('Rejected proposal changed pristine LOW or matrix')
                native=gpu['native'];rejected=native['rejected_step']
                return None,dict(ratio=value,triangles=rejected['read_counts'][1],valid=False,component_retained=False,
                    has_candidate_mesh=False,rejection=rejected['reason_code'],common_geometry_validation='NOT_RUN_NO_USABLE_MESH',
                    source_pristine_geometry_sha256=identity,source_triangles=part['source_triangles'],matrix_unchanged=True,
                    requested_backend=backend,effective_backend='cumesh_proposal_rejected',GPU_geometry=gpu,
                    triangle_count_basis='Observed rejected native readback, not an exportable candidate',trial_elapsed_seconds=time.monotonic()-started)
            trial.clear_geometry();trial.from_pydata(vertices.tolist(),[],faces.tolist());trial.update()
            for face in trial.polygons:face.use_smooth=True
        elif value<1:
            if backend!='blender':raise ValueError('Unsupported simplification backend')
            interfaces=part.get('_partition_interface_edges')
            if interfaces is not None:
                if part['_partition_interface_identity']!=identity:raise ValueError('Partition interface pristine binding changed')
                prior_groups=tuple(group.name for group in low.vertex_groups)
                temporary_group=low.vertex_groups.new(name='MF_DecimationAllowed')
                temporary_group_name=str(temporary_group.name)
                if temporary_group_name in prior_groups:raise ValueError('Temporary Decimate group name is not unique')
                temporary_group.add(list(range(part['_partition_interface_vertex_count'])),1.,'REPLACE')
                temporary_group.add(part['_partition_interface_vertices'].tolist(),0.,'REPLACE')
            bpy.ops.object.select_all(action='DESELECT');low.select_set(True);bpy.context.view_layer.objects.active=low
            mod=low.modifiers.new('Conservative Decimate','DECIMATE');mod.ratio=value;mod.use_collapse_triangulate=True
            modifier_name=mod.name
            if temporary_group_name is not None:
                mod.vertex_group=temporary_group_name;mod.vertex_group_factor=1.;mod.invert_vertex_group=False
            bpy.ops.object.modifier_apply(modifier=modifier_name)
        common_started=time.monotonic()
        vertices,faces,_=arrays(low.data);intact=bool(len(faces) and len(components(vertices,faces))==1)
        if len(faces):
            current=surface(low);comparison=compare(source,current,config)
            record=dict(ratio=value,triangles=current['triangles'],component_retained=intact,**comparison)
        else:
            record=dict(ratio=value,triangles=0,component_retained=False,valid=False,rejection='empty_decimate_candidate')
        if interfaces is not None:
            preserved=bool(len(faces) and partition_interfaces_preserved(vertices,faces,interfaces))
            record['partition_interface']=dict(edges=len(interfaces),anchors=len(part['_partition_interface_vertices']),
                edge_sha256=part['_partition_interface_edge_sha256'],required_local_incidence=1,preserved=preserved,
                scope='Exact whole edges shared by distinct partitions of one authored object; not original external openings')
            if not preserved:record.update(valid=False,rejection='partition_interface_geometry_or_incidence_changed')
        if not np.array_equal(matrix,np.asarray(low.matrix_world,np.float64)):raise ValueError('Geometry proposal changed authored LOW matrix')
        if _pristine_low_identity(low,pristine)!=identity:raise ValueError('Geometry trial mutated pristine LOW input')
        record.update(source_pristine_geometry_sha256=identity,source_triangles=part['source_triangles'],
                      triangle_count_basis='Actual triangulated exportable mesh',matrix_unchanged=True,trial_elapsed_seconds=time.monotonic()-started)
        record.update(requested_backend=backend,effective_backend='source_retained' if value==1 else 'cumesh' if gpu is not None else 'blender')
        if gpu is not None:
            witness=record.pop('source_triangle_witness',None)
            if witness is not None and intact and not record['valid']:
                bound=_cumesh_bind_witness(part,pristine,witness,identity)
                if bound is not None:record['source_triangle_witness']=bound
            record['GPU_geometry']=gpu
            record['CPU_common_geometry_validation_seconds']=time.monotonic()-common_started
        candidate=low.data;returned=True
        return candidate,record
    finally:
        cleanup_complete=False
        try:
            try:
                if modifier_name is not None:
                    remaining=low.modifiers.get(modifier_name)
                    if remaining is not None:low.modifiers.remove(remaining)
            finally:
                if prior_groups is not None:
                    current=tuple(group.name for group in low.vertex_groups)
                    if tuple(name for name in current if name!=temporary_group_name)!=prior_groups:
                        raise ValueError('Temporary Decimate cleanup found changed unrelated vertex groups')
                    # modifier_apply can invalidate the earlier RNA wrapper.
                    owned=low.vertex_groups.get(temporary_group_name) if temporary_group_name is not None else None
                    if owned is not None:low.vertex_groups.remove(owned)
                    if tuple(group.name for group in low.vertex_groups)!=prior_groups:
                        raise ValueError('Temporary Decimate cleanup did not restore vertex group labels/order')
            cleanup_complete=True
        finally:
            candidate=low.data;low.data=selected
            if (not returned or not cleanup_complete) and candidate!=selected and not candidate.users:bpy.data.meshes.remove(candidate)
            if trial!=candidate and not trial.users:bpy.data.meshes.remove(trial)


def copy_retained_source_mesh(part):
    """Keep exact authored render splits/Ns without another custom-N encoding.

    Called only after the pristine source-retained admission below. LOW keeps
    its object, matrix, materials and UV ownership; HIGH is never modified.
    """
    high=part['high'];low=part['low'];source=high.data;old=low.data
    hv,hf,hl=arrays(source);lv,lf,ll=arrays(old)
    matrix=np.asarray(low.matrix_world,np.float64).copy();high_before=mesh_digest(high)
    if matrix.tobytes()!=np.asarray(high.matrix_world,np.float64).tobytes() or len(hf)!=len(lf):raise ValueError('Retained render copy geometry/matrix differs')
    for first in range(0,len(hf),65536):
        if hv[hf[first:first+65536]].tobytes()!=lv[lf[first:first+65536]].tobytes():raise ValueError('Retained render copy ordered geometry differs')
    source_n=np.empty((len(source.loops),3),np.float32);source.corner_normals.foreach_get('vector',source_n.ravel())
    seams=np.empty(len(old.edges),bool);old.edges.foreach_get('use_seam',seams)
    edge_ids=np.empty(len(old.loops),np.int32);old.loops.foreach_get('edge_index',edge_ids)
    corner_seams=seams[edge_ids[ll]]
    uv=[];active=old.uv_layers.active.name if old.uv_layers.active else None
    for layer in old.uv_layers:
        values=np.empty((len(old.loops),2),np.float32);layer.data.foreach_get('uv',values.ravel())
        uv.append((layer.name,layer.active_render,values[ll].copy()))
    material_ids=np.empty(len(old.polygons),np.int32);old.polygons.foreach_get('material_index',material_ids)
    smooth=np.empty(len(old.polygons),bool);old.polygons.foreach_get('use_smooth',smooth)
    clone=source.copy()
    try:
        # Copy the LOW-owned UV domain, not HIGH's original texture coordinates.
        clone.materials.clear()
        for material in old.materials:clone.materials.append(material)
        clone.polygons.foreach_set('material_index',material_ids)
        actual_smooth=np.empty_like(smooth);clone.polygons.foreach_get('use_smooth',actual_smooth)
        if actual_smooth.tobytes()!=smooth.tobytes():raise ValueError('Retained render copy smooth policy differs')
        for layer in list(clone.uv_layers):clone.uv_layers.remove(layer)
        for name,render,values in uv:
            layer=clone.uv_layers.new(name=name);mapped=np.empty((len(clone.loops),2),np.float32);mapped[hl]=values
            layer.data.foreach_set('uv',mapped.ravel());layer.active_render=render
        if active is not None:clone.uv_layers.active_index=list(clone.uv_layers.keys()).index(active)
        edges=np.empty(len(clone.loops),np.int32);clone.loops.foreach_get('edge_index',edges)
        mapped_seams=np.zeros(len(clone.edges),bool);np.maximum.at(mapped_seams,edges[hl].ravel(),corner_seams.ravel())
        clone.edges.foreach_set('use_seam',mapped_seams)
        v,f,loops=arrays(clone);n=np.empty_like(source_n);clone.corner_normals.foreach_get('vector',n.ravel())
        if any(a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes() for a,b in ((v,hv),(f,hf),(loops,hl),(n,source_n))):raise ValueError('Retained deep HIGH copy changed exact geometry/loops/RNA Ns')
        for layer,(_,_,values) in zip(clone.uv_layers,uv):
            actual=np.empty((len(clone.loops),2),np.float32);layer.data.foreach_get('uv',actual.ravel())
            if actual[hl].tobytes()!=values.tobytes():raise ValueError('Retained render copy changed LOW UV corners')
        actual_seams=np.empty(len(clone.edges),bool);clone.edges.foreach_get('use_seam',actual_seams)
        if actual_seams[edges[hl]].tobytes()!=corner_seams.tobytes():raise ValueError('Retained render copy changed LOW seam corners')
        if mesh_digest(high)!=high_before or np.asarray(low.matrix_world,np.float64).tobytes()!=matrix.tobytes():raise ValueError('Retained render copy changed source/object transform')
        low.data=clone
        if not old.users:bpy.data.meshes.remove(old)
    except Exception:
        if low.data==clone:low.data=old
        if not clone.users:bpy.data.meshes.remove(clone)
        raise
    return dict(method='Deep actual HIGH mesh copy; no custom-normal re-encoding',old_LOW_vertices=len(lv),LOW_vertices=len(hv),
                triangles=len(hf),HIGH_geometry_loops_and_RNA_Ns_byteexact=True,LOW_object_and_matrix_preserved=True,
                LOW_material_and_UV_ownership_preserved=True,render_indexing_changed=True)


def preserve_source_retained_normals(parts, config):
    """Copy actual HIGH corner Ns only for a proven no-backend pristine choice.

    Custom-normal storage is quantized: planned frames must read the real LOW
    RNA afterwards. This never changes the shader NORMAL transfer/roundtrip ABI.
    """
    result=dict(applied_parts=0,copied_corners=0,parts=[],unit_limit=1e-4,storage_angular_limit_radians=1e-3,
                scope='Actual HIGH base corner Ns to authentic source_retained LOW only; not normal-map/shader equivalence')
    if config.get('operation') not in ('simplify-mesh','optimize-rebake'):return result
    for index,part in enumerate(parts):
        chosen=part.get('chosen',{})
        if part['protected'] or chosen.get('effective_backend')!='source_retained':continue
        low=part['low'];high=part['high'];lm=low.data;hm=high.data
        lineage=(high.as_pointer(),low.as_pointer(),part['object'],part['material'],tuple(part['source_component_ids']),tuple(map(tuple,part['source_component_face_ranges'])))
        if (part.get('_source_lineage')!=lineage or not hm.materials or hm.materials[0].get('mesh_finishing_material_index')!=part['material']
                or chosen.get('ratio')!=1. or not chosen.get('valid') or not chosen.get('component_retained')
                or chosen.get('source_triangles')!=part['source_triangles'] or chosen.get('triangles')!=part['source_triangles']
                or chosen.get('source_pristine_geometry_sha256')!=_pristine_low_identity(low,lm)):
            raise ValueError('Retained normal copy lacks genuine pristine/source-part/material/component provenance')
        hv,hf,hl=arrays(hm);lv,lf,ll=arrays(lm);H=np.asarray(high.matrix_world,np.float64);L=np.asarray(low.matrix_world,np.float64)
        if H.tobytes()!=L.tobytes() or not np.isfinite(H).all() or np.linalg.det(H[:3,:3])==0 or len(hf)!=len(lf) or not 0<len(hf)<=2_000_000:
            raise ValueError('Retained normal copy source/LOW geometry or matrix domain differs')
        original_source_face_ids(part['source_component_face_ranges'],part['source_component_ids'],len(hf))
        if (not np.array_equal(np.sort(hl.ravel()),np.arange(len(hm.loops))) or not np.array_equal(np.sort(ll.ravel()),np.arange(len(lm.loops)))):
            raise ValueError('Retained normal copy requires one-to-one triangle corners')
        for first in range(0,len(hf),65536):
            if hv[hf[first:first+65536]].tobytes()!=lv[lf[first:first+65536]].tobytes():raise ValueError('Retained normal copy ordered source/LOW corners differ')
        source=np.empty((len(hm.loops),3),np.float32);hm.corner_normals.foreach_get('vector',source.ravel())
        requested=np.empty((len(lm.loops),3),np.float32);requested[ll.ravel()]=source[hl.ravel()]
        lengths=np.linalg.norm(requested.astype(float),axis=1)
        if not np.isfinite(requested).all() or np.any(np.abs(lengths-1)>result['unit_limit']):raise ValueError('Retained HIGH RNA normal is not finite/unit')
        before=_pristine_low_identity(low,lm);high_before=mesh_digest(high)
        uv_before=uv_loop_data(lm).tobytes() if lm.uv_layers.active else None
        seams=np.empty(len(lm.edges),bool);lm.edges.foreach_get('use_seam',seams)
        # Preserve the actual HIGH indexed/sharp fan domain lost by LOW's
        # coordinate canonicalization. This is not all-sharp or flat shading.
        def sharp(mesh):
            value=np.zeros(len(mesh.edges),bool);attribute=mesh.attributes.get('sharp_edge')
            if attribute:attribute.data.foreach_get('value',value)
            return value
        he=np.empty(len(hm.loops),np.int32);le=np.empty(len(lm.loops),np.int32)
        hm.loops.foreach_get('edge_index',he);lm.loops.foreach_get('edge_index',le)
        source_edges=he[hl].ravel();low_edges=le[ll].ravel()
        if source_edges.min()<0 or source_edges.max()>=len(hm.edges) or low_edges.min()<0 or low_edges.max()>=len(lm.edges):raise ValueError('Invalid retained normal fan edge domain')
        incidence=np.bincount(low_edges,minlength=len(lm.edges));high_incidence=np.bincount(source_edges,minlength=len(hm.edges))
        minimum=np.full(len(lm.edges),len(hm.edges),np.int32);maximum=np.full(len(lm.edges),-1,np.int32)
        np.minimum.at(minimum,low_edges,source_edges);np.maximum.at(maximum,low_edges,source_edges)
        source_sharp=sharp(hm);existing_sharp=sharp(lm);continuous=np.zeros(len(lm.edges),bool)
        paired=np.flatnonzero((incidence==2)&(minimum==maximum))
        continuous[paired]=(high_incidence[minimum[paired]]==2)&~source_sharp[minimum[paired]]
        required_sharp=existing_sharp|((incidence>=2)&~continuous)
        added_sharp=int((required_sharp&~existing_sharp).sum())
        if added_sharp:
            attribute=lm.attributes.get('sharp_edge') or lm.attributes.new(name='sharp_edge',type='BOOLEAN',domain='EDGE')
            attribute.data.foreach_set('value',required_sharp);lm.update()
        lm.normals_split_custom_set(requested.tolist())
        if sharp(lm).tobytes()!=required_sharp.tobytes():raise ValueError('Retained normal setter changed the required source fan flags')
        stored=np.empty_like(requested);lm.corner_normals.foreach_get('vector',stored.ravel());lengths=np.linalg.norm(stored.astype(float),axis=1)
        if not np.isfinite(stored).all() or np.any(np.abs(lengths-1)>result['unit_limit']):raise ValueError('Stored retained LOW normal is not finite/unit')
        inverse=np.linalg.inv(L[:3,:3]);maximum=world_maximum=0.
        for first in range(0,len(stored),65536):
            a=requested[first:first+65536].astype(float);b=stored[first:first+65536].astype(float)
            for world in (False,True):
                x=a@inverse if world else a;y=b@inverse if world else b
                nx=np.linalg.norm(x,axis=1);ny=np.linalg.norm(y,axis=1)
                if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(nx).all() or not np.isfinite(ny).all() or np.any(nx<=0) or np.any(ny<=0):raise ValueError('Invalid retained normal transform')
                x/=nx[:,None];y/=ny[:,None]
                angle=float(np.arctan2(np.linalg.norm(np.cross(x,y),axis=1),np.einsum('ij,ij->i',x,y)).max(initial=0))
                if world:world_maximum=max(world_maximum,angle)
                else:maximum=max(maximum,angle)
        after_seams=np.empty_like(seams);lm.edges.foreach_get('use_seam',after_seams)
        if (_pristine_low_identity(low,lm)!=before or mesh_digest(high)!=high_before or seams.tobytes()!=after_seams.tobytes()
                or (uv_loop_data(lm).tobytes() if lm.uv_layers.active else None)!=uv_before):
            raise ValueError('Retained normal copy changed geometry/UV/seams/matrix/HIGH')
        copied=None
        if max(maximum,world_maximum)>result['storage_angular_limit_radians']:
            note('retained_normal_storage_copy',part=index,components=part['source_component_ids'],maximum_object_angular_error_radians=maximum,
                 maximum_WORLD_angular_error_radians=world_maximum,storage_angular_limit_radians=result['storage_angular_limit_radians'])
            copied=copy_retained_source_mesh(part)
            copied['rejected_compact_object_angular_error_radians']=maximum;copied['rejected_compact_WORLD_angular_error_radians']=world_maximum
            copied['discarded_compact_fan_cuts']=added_sharp
            lm=low.data;stored=np.empty_like(source);lm.corner_normals.foreach_get('vector',stored.ravel());requested=source
            if stored.tobytes()!=source.tobytes():raise ValueError('Retained HIGH-copy LOW RNA normal differs after assignment')
            maximum=world_maximum=0.
        result['applied_parts']+=1;result['copied_corners']+=len(stored)
        result['parts'].append(dict(part=index,object=part['object'],material=part['material'],components=part['source_component_ids'],
            chosen_ratio=chosen['ratio'],effective_backend=chosen['effective_backend'],source_pristine_geometry_sha256=chosen['source_pristine_geometry_sha256'],corners=len(stored),
            source_RNA_mapped_through_triangle_loops=True,actual_stored_LOW_authority=True,normal_bytes_equal=stored.tobytes()==requested.tobytes(),
            intentional_source_fan_cuts=added_sharp if copied is None else 0,LOW_edges=len(lm.edges),existing_LOW_sharp_preserved=copied is None,post_setter_sharp_flags_byteexact=copied is None,
            fan_policy='Only LOW multi-incident edges lacking the same nonsharp indexed HIGH edge pair are cut; original LOW sharp flags retained' if copied is None else 'Original HIGH render splits/sharp fans copied exactly; failed compact encoding discarded',
            retained_render_copy=copied,
            maximum_object_angular_error_radians=maximum,maximum_WORLD_angular_error_radians=world_maximum))
    return result


def rebalance_triangle_budget(parts, budget, evaluate, *, minimum_ratios=None, tested_ratios=None, capacity_bracketing=False):
    """Bounded allocation controller; geometry callback retains only admissible trials.

    Ratio intervals only propose finite trials. No error monotonicity or global
    infeasibility is inferred; actual validated triangle counts decide progress.
    """
    started=time.monotonic();floors=minimum_ratios or {};attempts={};lower={};blocked={}
    tested={i:set(values) for i,values in (tested_ratios or {}).items()}
    actual=sum(p['chosen']['triangles'] for p in parts);initial=actual
    report=dict(method='Deterministic largest-selected-allocation first, stable part-index ties',
                max_extra_evaluations=32,max_extra_evaluations_per_part=4,extra_evaluations=0,
                initial_triangles=initial,initial_deficit=max(0,initial-budget),triangle_budget=budget,trials=[],
                source_policy='Every geometry evaluation from SAME pristine HIGH-derived LOW, never selected LOD',
                refinement='Finite tested-ratio interval heuristic, no monotonic-error assumption or infeasibility proof')
    for index,part in enumerate(parts):
        floor=floors.get(index,0.)
        if not math.isfinite(floor) or not 0<=floor<=1:raise ValueError('Invalid conservative geometry retry floor')
        if type(part['chosen']['triangles']) is not int or part['chosen']['triangles']<1:raise ValueError('Invalid selected actual triangle count')
        if part['chosen']['ratio']<floor:raise ValueError('Initial selection is below conservative geometry retry floor')
    while actual>budget and report['extra_evaluations']<32:
        eligible=[i for i,p in enumerate(parts) if not p['protected'] and i not in blocked and
                  attempts.get(i,0)<4 and p['chosen']['triangles']>1 and p['chosen']['ratio']>floors.get(i,0.)]
        if not eligible:break
        index=min(eligible,key=lambda i:(-parts[i]['chosen']['triangles'],i));part=parts[index]
        previous=dict(part['chosen']);before_count=previous['triangles'];upper=float(previous['ratio'])
        deficit=actual-budget;requested=max(1,before_count-deficit)
        bracket_refinement=index in lower
        # Only meaningful remaining allocation uses a geometric bracket; unit clamps stay arithmetic.
        value=(math.sqrt(lower[index]*upper) if capacity_bracketing and requested>1 else (lower[index]+upper)/2) if bracket_refinement else upper*requested/before_count
        value=max(float(floors.get(index,0.)),value)
        history=tested.setdefault(index,set())
        if value in history:value=(value+upper)/2  # one bounded proposal adjustment, not a search loop
        if not math.isfinite(value) or not floors.get(index,0.)<=value<upper or value<=0 or value in history:
            blocked[index]='Repeated ratio, rounding or retry floor prevents a new smaller proposal';continue
        history.add(value);attempts[index]=attempts.get(index,0)+1;report['extra_evaluations']+=1
        record=evaluate(index,value,requested,before_count,bracket_refinement=bracket_refinement)
        if type(record['triangles']) is not int or record['triangles']<0 or record['ratio']!=value:
            raise ValueError('Geometry trial returned invalid actual count/ratio')
        accepted=bool(record['valid'] and record['component_retained'] and 0<record['triangles']<before_count)
        if record['accepted']!=accepted:raise ValueError('Native trial retention differs from strict smaller-valid allocation rule')
        if accepted:
            # Keep the last rejection as an interior search hint, never a floor.
            part['chosen']=dict(record,selection_stage='global_rebalance',initial_selected_ratio=previous.get('initial_selected_ratio',previous['ratio']))
        else:
            lower[index]=value
        actual=sum(p['chosen']['triangles'] for p in parts)
        report['trials'].append(dict(record,part=index,object=part['object'],material=part['material'],
            requested_triangles=requested,bracket_refinement=bracket_refinement,
            nominal_source_ratio_triangles=max(1,math.floor(part['source_triangles']*value)),
            previous_triangles=before_count,minimum_retry_ratio=floors.get(index,0.),
            deficit_before=deficit,deficit_after=max(0,actual-budget),
            progress='accepted_strictly_smaller' if accepted else 'rejected_geometry_or_no_actual_reduction'))
    met=actual<=budget
    report.update(triangles=actual,budget_met=met,remaining_deficit=max(0,actual-budget),
                  stop_reason='budget_met' if met else 'global_evaluation_limit' if report['extra_evaluations']==32 else 'no_eligible_part_after_floors_per_part_limits_or_no_progress',
                  global_evaluation_limit_reached=report['extra_evaluations']==32,
                  per_part_limit_count=sum(n==4 for n in attempts.values()),blocked_parts=blocked,
                  search_exhausted_without_budget=not met,infeasibility_proven=False,elapsed_seconds=time.monotonic()-started)
    return report


def _source_retained_retry_binding(part, identity, config, total, ratios, count):
    """Same-job part/config binding; identity includes pristine P/F/smooth/matrix."""
    return (identity,count,total,tuple(ratios),config.get('simplification_backend'),
            config.get('triangle_budget'),config.get('max_error_relative'),config.get('max_error_world'),
            part['object'],part['material'],part['source_triangles'],tuple(part['component_sizes']),
            tuple(part['source_component_ids']),tuple(map(tuple,part['source_component_face_ranges'])),
            part['source_component_domain'])


class NoTriangleReduction(ValueError):
    """The selected, valid geometry retained every input triangle."""


def simplify(parts, config, total, *, minimum_ratios=None, retained_source_retry=None):
    protected_triangles=sum(p['source_triangles'] for p in parts if p['protected'])
    mutable=total-protected_triangles;budget=config['triangle_budget']
    allocation=dict(protected_triangles=protected_triangles,mutable_source_triangles=mutable,
                    mutable_triangle_budget=max(0,budget-protected_triangles),
                    budget_floor_reason=('protected_triangles_exceed_budget' if protected_triangles>budget else
                        'protected_triangles_exhaust_budget' if protected_triangles==budget and mutable>0 else None))
    note('triangle_budget_allocation',**allocation)
    ratios=simplification_ratios(config,total,protected_triangles=protected_triangles)
    trials=[];pristines={};identities={};tested={};witnesses={};protected_attempted=set();succeeded=False;session=None
    # Metadata only, at most the existing1024 parts; no retained mesh/array cache.
    reuse=(retained_source_retry if config.get('simplification_backend') in ('cumesh_install_if_needed','cumesh_existing') and len(parts)<=1024 else None)
    if reuse is not None and minimum_ratios is None:reuse.clear()
    try:
        if config.get('simplification_backend') in ('cumesh_install_if_needed','cumesh_existing'):
            from cumesh_simplify import CuMeshSession
            if '_cumesh_session' in config:raise RuntimeError('Nested CuMesh simplification stage is not permitted')
            session=CuMeshSession();config['_cumesh_session']=session
        for index,part in enumerate(parts):
            low=part['low'];high=part['high']
            if part['protected']:
                actual=len(arrays(low.data)[1])
                if actual!=part['source_triangles']:raise ValueError('Protected triangle count changed')
                part['chosen']=dict(triangles=actual,protected_components=len(part['component_sizes']),ratio=1.)
                continue
            pristine=low.data.copy();pristines[index]=pristine;identities[index]=_pristine_low_identity(low,pristine)
            if '_partition_interface_edges' in part:part['_partition_interface_identity']=identities[index]
            source=surface(high);chosen=None;records=[];tested[index]=set()
            entry=reuse.pop(index,None) if reuse is not None and minimum_ratios is not None and index not in minimum_ratios else None
            if (entry is not None and entry.get('binding')==_source_retained_retry_binding(part,identities[index],config,total,ratios,len(parts))
                and entry.get('selection')==(1.,part['source_triangles'],'source_retained') and entry.get('tested')==tuple(ratios)):
                witness=entry.get('witness')
                rebound=None if witness is None else _cumesh_bind_witness(part,pristine,witness,identities[index])
                if witness is None or rebound is not None:
                    candidate=None
                    try:
                        candidate,record=_evaluate_pristine_reduction(part,pristine,source,1.,config,identities[index])
                        records.append(record)
                        if (candidate is not None and record['valid'] and record['component_retained'] and
                            (record['ratio'],record['triangles'],record['effective_backend'])==entry['selection'] and
                            _pristine_low_identity(low,candidate)==identities[index]):
                            record['selection_stage']='same_job_source_retained_retry'
                            record['source_retained_retry']=dict(previous_geometry_directory=entry['geometry_directory'],
                                previous_part=index,previous_tested_ratios=list(entry['tested']),
                                scope='Pristine ratio1 only; current common gates rechecked; prior timings not replayed')
                            old=low.data;low.data=candidate;candidate=None
                            if not old.users:bpy.data.meshes.remove(old)
                            chosen=record;tested[index].update(entry['tested'])
                            if rebound is not None:witnesses[index]=rebound
                    finally:
                        if candidate is not None and not candidate.users:bpy.data.meshes.remove(candidate)
            for value in (() if chosen is not None else ratios):
                if value<(minimum_ratios or {}).get(index,0.):continue
                tested[index].add(value);candidate=None
                try:
                    candidate,record=_evaluate_pristine_reduction(part,pristine,source,value,config,identities[index]);records.append(record)
                    if record.get('source_triangle_witness') is not None:witnesses.setdefault(index,record['source_triangle_witness'])
                    if record['valid'] and record['component_retained']:
                        record['selection_stage']='initial_component_schedule'
                        old=low.data;low.data=candidate;candidate=None
                        if not old.users:bpy.data.meshes.remove(old)
                        chosen=record;break
                finally:
                    if candidate is not None and not candidate.users:bpy.data.meshes.remove(candidate)
            if chosen is None:raise ValueError('No component candidate passes the geometry gate')
            part['chosen']=chosen
            trials.append(dict(part=index,object=part['object'],material=part['material'],candidates=records))
            note('component_reduced',object=part['object'],triangles=chosen['triangles'])
        source=None;cached_index=None;cached_source=None
        def evaluate_extra(index,value,requested,before_count,*,bracket_refinement):
            nonlocal cached_index,cached_source
            part=parts[index];low=part['low'];candidate=None
            if cached_index!=index:cached_source=surface(part['high']);cached_index=index
            note('global_rebalance_trial_started',part=index,ratio=value,requested_triangles=requested,
                 source_pristine_geometry_sha256=identities[index])
            try:
                witness=witnesses.get(index)
                floor=(minimum_ratios or {}).get(index,0.)
                protected_target=max(1,math.floor(part['source_triangles']*value)) if bracket_refinement else requested
                use_protected=(witness is not None and index not in protected_attempted and
                    config.get('simplification_backend') in ('cumesh_install_if_needed','cumesh_existing') and
                    type(requested)is int and math.ceil(part['source_triangles']*floor)<=requested<before_count and
                    math.ceil(part['source_triangles']*floor)<=protected_target<before_count and
                    witness['source_pristine_geometry_sha256']==identities[index])
                if use_protected:
                    protected_attempted.add(index)  # Consumes this ordinary global/per-part evaluation slot.
                    candidate,record=_evaluate_pristine_reduction(part,pristines[index],cached_source,value,config,identities[index],
                        protected_face=witness,target=protected_target)
                else:candidate,record=_evaluate_pristine_reduction(part,pristines[index],cached_source,value,config,identities[index])
                if record.get('source_triangle_witness') is not None:witnesses.setdefault(index,record['source_triangle_witness'])
                accepted=bool(record['valid'] and record['component_retained'] and 0<record['triangles']<before_count)
                if accepted:
                    old=low.data;low.data=candidate;candidate=None
                    if not old.users:bpy.data.meshes.remove(old)
                record['accepted']=accepted
                note('global_rebalance_trial_completed',part=index,requested_triangles=requested,**record)
                return record
            finally:
                if candidate is not None and not candidate.users:bpy.data.meshes.remove(candidate)
        rebalance=rebalance_triangle_budget(parts,budget,evaluate_extra,minimum_ratios=minimum_ratios,tested_ratios=tested,
                                            capacity_bracketing=config.get('simplification_backend','blender')=='blender')
        cached_source=None
        actual=sum(len(arrays(p['low'].data)[1]) for p in parts)
        if actual!=rebalance['triangles']:raise ValueError('Rebalance retained counts differ from actual exportable triangles')
        if actual>=total:raise NoTriangleReduction('No genuine triangle reduction achieved; original preserved, no reduced success')
        result=dict(source_triangles=total,triangles=actual,triangle_budget=budget,budget_met=actual<=budget,
                    output_changed=True,trials=trials,rebalance=rebalance,
                    requested_backend=config.get('simplification_backend','blender'),
                    geometry_backend_scope='GPU proposals only when explicitly selected; common validation and UV/Cycles remain CPU',
                    protected_components=sum(len(p['component_sizes']) for p in parts if p['protected']),**allocation)
        if reuse is not None:
            reuse.clear()
            if minimum_ratios is None:
                for index,part in enumerate(parts):
                    chosen=part['chosen']
                    if (not part['protected'] and chosen.get('effective_backend')=='source_retained' and
                        chosen['ratio']==1. and chosen['triangles']==part['source_triangles'] and tuple(sorted(tested[index]))==tuple(ratios)):
                        reuse[index]=dict(binding=_source_retained_retry_binding(part,identities[index],config,total,ratios,len(parts)),
                            selection=(1.,chosen['triangles'],'source_retained'),tested=tuple(ratios),witness=witnesses.get(index),
                            geometry_directory=config['_geometry_directory'])
        succeeded=True
        return result
    finally:
        try:
            if session is not None:
                config.pop('_cumesh_session',None)
                session.close(check=succeeded)
        finally:
            for index,pristine in pristines.items():
                if not succeeded:
                    # Failed simplification owns only disposable LOW meshes. Restore
                    # that input, free selected/transient data, never delete HIGH/objects.
                    low=parts[index]['low'];old=low.data;low.data=pristine
                    if old!=pristine and not old.users:bpy.data.meshes.remove(old)
                    parts[index].pop('chosen',None)
                if not pristine.users:bpy.data.meshes.remove(pristine)

def uv_loop_data(mesh):
    data=np.empty((len(mesh.loops),2),np.float32)
    mesh.uv_layers.active.data.foreach_get('uv',data.ravel())
    return data


def uv_vertex_equivalence(low, *, include_charts=False, position_keys=False):
    from production_topology import uv_corner_equivalence
    vertices,faces,loops=arrays(low.data);uv=uv_loop_data(low.data)[loops]
    barriers=np.fromiter((low.data.edges[loop.edge_index].use_seam for loop in low.data.loops),dtype=bool,count=len(low.data.loops))[loops]
    inverse,charts=uv_corner_equivalence(vertices,faces,uv,barriers,position_keys=position_keys)
    identifiers=np.empty(len(low.data.loops),np.int32);identifiers[loops.ravel()]=inverse
    return (identifiers,charts) if include_charts else identifiers


def normalize_uv_density(lows, layouts, resolution):
    """Uniform world-area density, evaluated once per logical UV vertex.

    Keep chart seams and aspect ratios. Float64 area/transform arithmetic avoids
    Blender's valence-dependent float32 corner averaging; only the final unique
    UV vertices are rounded to storage precision, then broadcast to their loops.
    """
    states=[];total_world=0.;total_uv=0.
    for low,(identifiers,charts) in zip(lows,layouts):
        vertices,faces,loops=arrays(low.data);uv=uv_loop_data(low.data).astype(np.float64)
        matrix=np.asarray(low.matrix_world,dtype=np.float64)
        world=vertices.astype(np.float64)@matrix[:3,:3].T+matrix[:3,3]
        triangles=world[faces];texcoords=uv[loops]
        world_areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)*.5
        edges=texcoords[:,1:]-texcoords[:,:1]
        uv_areas=np.abs(edges[:,0,0]*edges[:,1,1]-edges[:,0,1]*edges[:,1,0])*.5
        area_world=np.bincount(charts,weights=world_areas);area_uv=np.bincount(charts,weights=uv_areas)
        if not np.isfinite(area_world).all() or not np.isfinite(area_uv).all() or np.any(area_world<=0) or np.any(area_uv<=0):
            raise ValueError('UV density normalization requires finite positive chart areas')
        _,representatives=np.unique(identifiers,return_index=True)
        unique_uv=uv[representatives]
        if not np.array_equal(unique_uv[identifiers],uv):raise ValueError('Pre-normalization logical UV vertices differ')
        loop_charts=np.empty(len(uv),np.int32);loop_charts[loops.ravel()]=np.repeat(charts,3)
        vertex_charts=loop_charts[representatives]
        if not np.array_equal(vertex_charts[identifiers],loop_charts):raise ValueError('Logical UV vertex crosses a chart seam')
        lower=np.full((len(area_uv),2),np.inf);upper=-lower.copy()
        np.minimum.at(lower,vertex_charts,unique_uv);np.maximum.at(upper,vertex_charts,unique_uv)
        states.append((low,identifiers,vertex_charts,unique_uv,(lower+upper)*.5,area_world,area_uv))
        total_world+=float(area_world.sum());total_uv+=float(area_uv.sum())
    total_ratio=total_world/total_uv;maximum_rounding=0.;minimum_scale=math.inf;maximum_scale=0.;logical_vertices=0;chart_count=0
    for low,identifiers,charts,uv,centres,area_world,area_uv in states:
        scale=np.sqrt((area_world/area_uv)/total_ratio)
        transformed=(uv-centres[charts])*scale[charts,None]+centres[charts]
        stored=transformed.astype(np.float32)
        if not np.isfinite(stored).all():raise ValueError('UV density normalization produced nonfinite coordinates')
        low.data.uv_layers.active.data.foreach_set('uv',stored[identifiers].ravel())
        maximum_rounding=max(maximum_rounding,float(np.max(np.abs(stored.astype(np.float64)-transformed))))
        minimum_scale=min(minimum_scale,float(scale.min()));maximum_scale=max(maximum_scale,float(scale.max()))
        logical_vertices+=len(uv);chart_count+=len(scale)
    return dict(method='Float64 world/UV-area uniform chart scaling; one float32 rounding per logical UV vertex',
                charts=chart_count,logical_uv_vertices=logical_vertices,minimum_chart_scale=minimum_scale,maximum_chart_scale=maximum_scale,
                max_float32_rounding_uv=maximum_rounding,max_float32_rounding_texels=maximum_rounding*resolution,
                shared_vertex_uv_exact=True)


def restore_uv_equivalence(lows, identifiers):
    changed=0;maximum=0.;ulps=0
    for low,groups in zip(lows,identifiers):
        fixed,stats=reconcile_uv_vertices(uv_loop_data(low.data),groups)
        if stats['changed_uv_values']:low.data.uv_layers.active.data.foreach_set('uv',fixed.ravel())
        changed+=stats['changed_uv_values'];maximum=max(maximum,stats['max_uv_displacement']);ulps=max(ulps,stats['max_ulps'])
    return dict(changed_uv_values=changed,max_uv_displacement=maximum,max_ulps=ulps)


def correct_final_uv_representation(lows, references):
    corrected=0;maximum=0.;maximum_ulps=0;paired_charts=0;total_faces=0
    for low,reference in zip(lows,references):
        mesh=low.data;vertices,faces,loops=arrays(mesh);data=uv_loop_data(mesh)
        total_faces+=len(faces)
        triangles=data[loops]
        collapsed=np.fromiter((not float32_uv_area(uv) for uv in triangles),dtype=bool,count=len(triangles))
        if not collapsed.any():continue
        edge_rows=np.fromiter((loop.edge_index for loop in mesh.loops),dtype=np.int32,count=len(mesh.loops))
        edge_faces=edge_rows[loops];incidence=np.bincount(edge_rows,minlength=len(mesh.edges))
        for index in np.flatnonzero(collapsed):
            movable=(0,1,2);shared=None
            for edge in edge_faces[index]:
                edge=int(edge)
                if mesh.edges[edge].use_seam or incidence[edge]==1:continue
                pair=np.flatnonzero(np.any(edge_faces==edge,axis=1))
                if len(pair)!=2 or not collapsed[pair].all():raise ValueError('Collapsed UV is not an isolated one/two-face chart')
                for member in pair:
                    for adjacent in edge_faces[member]:
                        adjacent=int(adjacent)
                        if mesh.edges[adjacent].use_seam or incidence[adjacent]==1:continue
                        neighbors=np.flatnonzero(np.any(edge_faces==adjacent,axis=1))
                        if not np.isin(neighbors,pair).all():raise ValueError('Collapsed UV chart exceeds two faces')
                if shared is not None:raise ValueError('Collapsed two-face chart has multiple shared edges')
                shared=edge
                for vertex in mesh.edges[edge].vertices:
                    corners=[int(np.flatnonzero(faces[member]==vertex)[0]) for member in pair]
                    if not np.array_equal(triangles[pair[0],corners[0]],triangles[pair[1],corners[1]]):raise ValueError('Collapsed chart shared UV edge is not continuous')
                movable=tuple(int(c) for c in np.flatnonzero(~np.isin(faces[index],mesh.edges[edge].vertices)))
                if len(movable)!=1:raise ValueError('Collapsed chart has no unique nonshared corner')
                if index==pair.min():paired_charts+=1
            uv=triangles[index]
            if _zero_area(*(tuple(float(v) for v in vertices[row]) for row in faces[index])):raise ValueError('Final UV correction cannot change a zero-area 3D triangle')
            fixed,delta,steps=separate_float32_collapse(uv,reference[loops[index]],movable_corners=movable)
            data[loops[index]]=fixed;corrected+=1;maximum=max(maximum,delta);maximum_ulps=max(maximum_ulps,steps)
        mesh.uv_layers.active.data.foreach_set('uv',data.ravel())
        if not np.array_equal(uv_loop_data(mesh)[loops[~collapsed]],triangles[~collapsed]):raise ValueError('UV quantization changed an otherwise valid face')
    return dict(corrected_faces=corrected,max_uv_displacement=maximum,max_ulps=maximum_ulps,
                corrected_two_face_charts=paired_charts,split_uv_shared_edges=0,unchanged_valid_faces=total_faces-corrected,
                shared_edge_uv_preserved=True,preoperator_orientation_preserved=True)



def raster_chart_seed_phase(lows, layouts, resolution, *, repair):
    states=[];seed_sets=[];failure_details=[];regular_details=[];success_details=[];regular_global=[];repair_faces=[];offset=0
    report=dict(charts=0,seeded_charts=0,orphan_charts=0,aligned_charts=0,regularized_charts=0,minimum_floor_charts=0,seeded_floor_exceptions=0,unsupported_charts=0,
                minimum_singleton_inradius_texels=.001,max_displacement_texels=0.,max_regular_displacement_texels=0.,max_rounding_error_texels=0.,details=[],detail_limit=32,
                criterion='Pinned binary32 raster with strictly interior barycentrics; native occupancy confirmation required',
                UV_density_exception='Generated singleton below0.001px inradius, including already seeded, or failed interior alignment may use the existing regular r0.25px chart; supplied UV unchanged',
                appearance='NOT_VISUALLY_REVIEWED; a chart seed does not prove filtering or HIGH coverage')
    for low,(_,charts) in zip(lows,layouts):
        _,_,loops=arrays(low.data)
        corrected,seeds,stats=align_unseeded_charts(uv_loop_data(low.data),loops,charts,resolution,repair=repair)
        regular_faces=stats.pop('_regular_faces',[]);repair_faces.append(stats.pop('_unsupported_faces',[]))
        regular_global.extend(offset+int(face) for face in regular_faces);offset+=len(loops)
        states.append((low,corrected,stats,loops));seed_sets.append(seeds)
        for key in ('charts','seeded_charts','orphan_charts','aligned_charts','regularized_charts','minimum_floor_charts','seeded_floor_exceptions','unsupported_charts'):report[key]+=stats.get(key,0)
        for key in ('max_displacement_texels','max_regular_displacement_texels','max_rounding_error_texels'):report[key]=max(report[key],stats.get(key,0.))
        for item in stats['details']:
            destination=failure_details if 'unsupported_reason' in item else regular_details if 'UV_density_exception' in item else success_details
            if len(destination)<32:destination.append(dict(low=low.name,**item))
    if not report['unsupported_charts'] and regular_global:
        triangles=np.concatenate([corrected[loops] for _,corrected,_,loops in states])
        try:clearance=regular_chart_clearance(triangles,np.asarray(regular_global,np.int64),resolution)
        except ValueError as error:
            clearance=dict(valid=False,error=str(error),details=[])
        report['regular_chart_clearance']=clearance
        failed=[item for item in clearance['details'] if not item['valid']]
        if not clearance['valid']:
            report['unsupported_charts']+=len(failed) if failed else len(regular_global)
            for item in (failed or [dict(global_face=int(i)) for i in regular_global])[:32]:
                failure_details.append(dict(**item,unsupported_reason=clearance.get('error','Regular chart clearance must be strictly greater than3px')))
        if not clearance['valid']:
            affected=np.asarray([item['global_face'] for item in failed] if failed else regular_global,np.int64);first=0
            for i,(_,_,_,loops) in enumerate(states):
                local=affected[(affected>=first)&(affected<first+len(loops))]-first
                repair_faces[i]=np.union1d(repair_faces[i],local).astype(np.int64).tolist();first+=len(loops)
        for item,proof in zip(regular_details,clearance['details']):item['regular_clearance']=proof
    report['details']=(failure_details+regular_details+success_details)[:32]
    report['details_omitted']=max(0,report['orphan_charts']+report['seeded_floor_exceptions']-len(report['details']))
    report['unsupported_details_omitted']=max(0,report['unsupported_charts']-len(failure_details))
    report['detail_policy']='Unsupported charts, then UV density exceptions, then translations across the entire atlas'
    if report['unsupported_charts']:
        error=UVRasterSeedError(report);error.generated_repair_faces=repair_faces;raise error
    for low,corrected,stats,_ in states:
        if stats['aligned_charts'] or stats.get('regularized_charts',0):low.data.uv_layers.active.data.foreach_set('uv',corrected.ravel())
    return report,seed_sets

def generated_overlap_repair_faces(lows, selected):
    """Add only exact same-LOW/current-island pair witnesses before ONE repack."""
    triangles=[];offsets=[0]
    for low in lows:
        _,_,loops=arrays(low.data)
        triangles.append(uv_loop_data(low.data)[loops]);offsets.append(offsets[-1]+len(loops))
    # Witness collection retains the validator's triangle/work/ambiguity caps.
    # Advisory discovery never replaces the mandatory strict final validators.
    try:scan=validate_uv_triangles(np.concatenate(triangles),collect_overlaps=True,collect_overlap_pairs=True)
    except ValueError as error:
        return selected,dict(advisory_not_used=True,reason=str(error),added_faces=0)
    extra=[[] for _ in lows];charts={};accepted=0
    for first,second in scan['overlap_face_pairs']:
        owner=int(np.searchsorted(offsets,first,side='right')-1)
        if owner!=int(np.searchsorted(offsets,second,side='right')-1):continue
        if owner not in charts:charts[owner]=uv_vertex_equivalence(lows[owner],include_charts=True)[1]
        a,b=first-offsets[owner],second-offsets[owner]
        if charts[owner][a]!=charts[owner][b]:continue
        extra[owner].extend((a,b));accepted+=1
    extended=[np.union1d(ids,added).astype(np.int64) for ids,added in zip(selected,extra)]
    return extended,dict(pair_witnesses=len(scan['overlap_face_pairs']),same_LOW_island_pair_witnesses=accepted,
                         added_faces=sum(len(np.setdiff1d(after,before)) for before,after in zip(selected,extended)),
                         maximum_pair_witnesses=32,maximum_witness_faces=32)


def regularize_generated_uv_faces(lows, references, selected, resolution):
    """One pre-freeze generated-UV exception; geometry and healthy UVs stay exact.

    Isolate only selected faces, then repack ONCE. This is not a coverage claim:
    final SAT, chart seeds, padding support, TBN and native HIGH gates still apply.
    """
    if len(selected)!=len(lows) or not 1<=resolution<=8192:raise ValueError('Invalid generated UV repair inputs')
    selected,overlap_repair=generated_overlap_repair_faces(lows,selected)
    count=0;requested=0
    for low,reference,ordinals in zip(lows,references,selected):
        mesh=low.data;v,f,loops=arrays(mesh);uv=uv_loop_data(mesh);before=uv.copy()
        pieces=[np.asarray(ordinals,np.int64)];requested+=len(pieces[0])
        # The ONE repack can rotate/requantize other narrow faces. Batch all
        # faces below the already-established generated UV floor, not merely
        # the first failure. Chunked scratch is bounded independently of F.
        for start in range(0,len(f),65536):
            t=uv[loops[start:start+65536]].astype(float)*resolution
            perimeter=np.linalg.norm(np.roll(t,-1,axis=1)-t,axis=2).sum(axis=1)
            twice=np.abs(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]))
            radius=np.divide(twice,perimeter,out=np.zeros(len(t)),where=perimeter>0)
            pieces.append(start+np.flatnonzero(radius<.001))
        ids=np.unique(np.concatenate(pieces))
        if ids.size==0:continue
        if ids.min()<0 or ids.max()>=len(f) or len(f)>2_000_000:raise ValueError('Generated UV repair face admission exceeded')
        matrix=np.asarray(low.matrix_world,np.float64).tobytes();normals=np.empty((len(mesh.corner_normals),3),np.float32);mesh.corner_normals.foreach_get('vector',normals.ravel())
        for face in ids:
            ref=reference[loops[face]];sign=float32_uv_area(ref)
            if not sign or not np.isfinite(ref).all() or _zero_area(*(tuple(float(x) for x in v[row]) for row in f[face])):
                raise ValueError('Generated UV repair requires valid prepack orientation and nonzero 3D geometry')
            center=np.clip(uv[loops[face]].astype(float).mean(axis=0)*resolution,.5,resolution-.5)
            triangle=(center+np.array([[0.,.5],[-math.sqrt(3)*.25,-.25],[math.sqrt(3)*.25,-.25]]))/resolution
            if sign<0:triangle=triangle[[0,2,1]]
            uv[loops[face]]=triangle.astype(np.float32)
            for loop in loops[face]:mesh.edges[mesh.loops[int(loop)].edge_index].use_seam=True
        healthy=np.ones(len(f),bool);healthy[ids]=False
        if before[loops[healthy]].tobytes()!=uv[loops[healthy]].tobytes():raise ValueError('Generated UV repair changed unselected UV corners')
        mesh.uv_layers.active.data.foreach_set('uv',uv.ravel())
        after_v,after_f,after_loops=arrays(mesh);after_n=np.empty_like(normals);mesh.corner_normals.foreach_get('vector',after_n.ravel())
        if any(a.tobytes()!=b.tobytes() for a,b in [(v,after_v),(f,after_f),(loops,after_loops),(normals,after_n)]) or np.asarray(low.matrix_world,np.float64).tobytes()!=matrix:
            raise ValueError('Generated UV repair changed geometry/corner normals/matrix')
        count+=len(ids)
    if not count:raise ValueError('Generated UV repair has no affected faces')
    return dict(faces=count,originally_requested_faces=requested,exact_same_island_overlap_repair=overlap_repair,minimum_UV_inradius_texels=.001,policy='Affected and existing-floor-risk valid generated faces isolated as regular r0.25px triangles before ONE additional same-margin pack',UV_density_shape_exception=True,geometry_corner_normals_matrix_byteexact=True,unselected_corners_byteexact_before_repack=True)


def uv_pack_invariant(low):
    """Physical/corner/sharp/seam state; UV coordinates alone may be repacked."""
    mesh=low.data;h=hashlib.sha256(_pristine_low_identity(low,mesh).encode())
    _,_,loops=arrays(mesh);h.update(loops.tobytes())
    normals=np.empty((len(mesh.loops),3),np.float32);mesh.corner_normals.foreach_get('vector',normals.ravel());h.update(normals.tobytes())
    seams=np.empty(len(mesh.edges),bool);mesh.edges.foreach_get('use_seam',seams);h.update(seams.tobytes())
    sharp=np.zeros(len(mesh.edges),bool);attribute=mesh.attributes.get('sharp_edge')
    if attribute:attribute.data.foreach_get('value',sharp)
    h.update(sharp.tobytes());return h.hexdigest()


def recover_generated_uv_overlap(lows, layouts, references, seeds, resolution, error):
    """Only called after the microchart repack has consumed the extra-pack slot."""
    if type(error) is not UVTriangleOverlapError or sum(len(low.data.loops) for low in lows)>6_000_000:return None
    current=[uv_vertex_equivalence(low,include_charts=True) for low in lows]
    if len(current)!=len(layouts) or len(seeds)!=len(lows):return None
    if any(not np.array_equal(a,b) for old,new in zip(layouts,current) for a,b in zip(old,new)):return None
    states=[];triangles=[];owners=[];offset=0
    for pi,low in enumerate(lows):
        _,_,loops=arrays(low.data);uv=uv_loop_data(low.data)
        states.append((uv,loops));triangles.append(uv[loops])
        owners.extend((pi,face-offset) for face in error.faces if offset<=face<offset+len(loops))
        offset+=len(loops)
    if offset>2_000_000 or len(owners)!=2 or owners[0][0]!=owners[1][0]:return None
    pi=owners[0][0];pair=[face for _,face in owners];groups,charts=current[pi]
    if charts[pair[0]]!=charts[pair[1]]:return None
    local=np.flatnonzero(charts==charts[pair[0]])
    if len(local)>32:return None
    uv,loops=states[pi];counts=np.bincount(groups)
    movable=counts[groups[loops[pair]]]==1
    first=sum(len(data[1]) for data in states[:pi])
    result=separate_packed_overlap(np.concatenate(triangles),references[pi][loops[local]],local+first,error.faces,movable,resolution)
    if result is None:return None
    face,triangle,report,validation=result;face-=first
    invariants=[uv_pack_invariant(low) for low in lows]
    original=uv[loops[face]].copy()
    try:
        uv[loops[face]]=triangle;lows[pi].data.uv_layers.active.data.foreach_set('uv',uv.ravel())
        after=[uv_vertex_equivalence(low,include_charts=True) for low in lows]
        if (any(not np.array_equal(a,b) for old,new in zip(current,after) for a,b in zip(old,new))
                or any(uv_loop_data(low.data).tobytes()!=data[0].tobytes() for low,data in zip(lows,states))
                or [uv_pack_invariant(low) for low in lows]!=invariants):
            raise ValueError('UV scalar correction changed undeclared geometry/N/seams/sharp/matrix/UV ownership')
        _,checked=raster_chart_seed_phase(lows,after,resolution,repair=False)
        if len(checked)!=len(seeds) or any(not np.array_equal(a,b) for a,b in zip(seeds,checked)):
            raise ValueError('UV scalar correction changed existing chart raster seeds')
    except BaseException:
        uv[loops[face]]=original;lows[pi].data.uv_layers.active.data.foreach_set('uv',uv.ravel())
        raise
    report.update(part=pi,local_face=face,chart_ownership_unchanged=True,read_only_seed_validation=True,
                  geometry_corner_normals_seams_sharp_matrix_unchanged=True)
    return validation,report


def apply_generated_atlas_extent(lows, atlas_packing):
    """Only explicit compact packing transforms generated post-pack UVs."""
    if type(atlas_packing) is not str or atlas_packing not in ('full','compact'):
        raise ValueError('atlas_packing must be full or compact')
    if atlas_packing=='full':return
    for low in lows:
        uv=uv_loop_data(low.data)
        uv=np.float32(.125)+np.float32(.75)*uv
        low.data.uv_layers.active.data.foreach_set('uv',uv.ravel())


def final_uv(lows, resolution, *, atlas_packing='full'):
    if type(atlas_packing) is not str or atlas_packing not in ('full','compact'):
        raise ValueError('atlas_packing must be full or compact')
    extent=.75 if atlas_packing=='compact' else 1.
    pack_margin=2/resolution if atlas_packing=='full' else 2/(resolution*extent)
    bpy.ops.object.select_all(action='DESELECT')
    for low in lows:
        low.select_set(True)
        for layer in list(low.data.uv_layers):low.data.uv_layers.remove(layer)
        low.data.uv_layers.new(name='MeshFinishingUV')
    bpy.context.view_layer.objects.active=lows[0]
    bpy.context.tool_settings.mesh_select_mode=(False,False,True)
    bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT')
    # Projection seeds chart boundaries but can overlap inside a curved chart.
    bpy.ops.uv.smart_project(angle_limit=math.radians(66),island_margin=0,correct_aspect=False,scale_to_bounds=True)
    bpy.ops.uv.seams_from_islands(mark_seams=True,mark_sharp=False)
    bpy.ops.mesh.select_all(action='DESELECT');bpy.ops.object.mode_set(mode='OBJECT')
    scan=validate_destination_uv(lows,collect_overlaps=True);bad=np.asarray(scan.pop('overlap_faces'),dtype=np.int64)
    offset=0
    for low in lows:
        mesh=low.data
        local=bad[np.searchsorted(bad,offset):np.searchsorted(bad,offset+len(mesh.loop_triangles))]-offset
        for ordinal in local:
            face=mesh.polygons[mesh.loop_triangles[int(ordinal)].polygon_index]
            if len(face.loop_indices)!=3:raise ValueError('Targeted UV cuts require triangle polygons')
            face.select=True
            for loop in face.loop_indices:mesh.edges[mesh.loops[loop].edge_index].use_seam=True
        offset+=len(mesh.loop_triangles)
    bpy.ops.object.mode_set(mode='EDIT')
    selected=sum(low.data.total_face_sel for low in lows)
    if selected!=len(bad):raise ValueError(f'Targeted UV face selection mismatch: selected={selected}, expected={len(bad)}')
    if len(bad):bpy.ops.uv.unwrap(method='ANGLE_BASED',fill_holes=True,correct_aspect=False,margin=0)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    layouts=[uv_vertex_equivalence(low,include_charts=True) for low in lows]
    equivalence=[identifiers for identifiers,_ in layouts]
    references=[uv_loop_data(low.data) for low in lows]
    normalization=normalize_uv_density(lows,layouts,resolution)
    if sum(len(low.data.loops) for low in lows)>2_000_000*3:raise ValueError('Normalized prepack UV snapshot exceeds the existing triangle domain')
    prepack=[uv_loop_data(low.data) for low in lows]
    invariants=[uv_pack_invariant(low) for low in lows]
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.pack_islands(rotate=True,scale=True,merge_overlap=False,margin_method='FRACTION',margin=pack_margin)
    bpy.ops.object.mode_set(mode='OBJECT')
    packed_equivalence=restore_uv_equivalence(lows,equivalence)
    apply_generated_atlas_extent(lows,atlas_packing)
    repair_report=dict(extra_packs=0,maximum_extra_packs=1,faces=0)
    try:
        representability=correct_final_uv_representation(lows,references)
        raster,seeds=raster_chart_seed_phase(lows,layouts,resolution,repair=True)
    except ValueError as error:
        if isinstance(error,UVRasterSeedError):repair_selected=getattr(error,'generated_repair_faces',None)
        else:
            repair_selected=[]
            for low in lows:
                _,_,loops=arrays(low.data);uv=uv_loop_data(low.data)
                repair_selected.append(np.flatnonzero([float32_uv_area(t)==0 for t in uv[loops]]))
        if repair_selected is None or not any(len(ids) for ids in repair_selected):raise
        initial_refusal=str(error)
        repair_report=regularize_generated_uv_faces(lows,references,repair_selected,resolution)
        # New seams are real chart boundaries; no post-pack floor is hidden as
        # a <=0.5px translation or as recovered source detail.
        layouts=[uv_vertex_equivalence(low,include_charts=True) for low in lows]
        equivalence=[identifiers for identifiers,_ in layouts];references=[uv_loop_data(low.data) for low in lows]
        invariants=[uv_pack_invariant(low) for low in lows]
        bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT');bpy.ops.uv.select_all(action='SELECT')
        bpy.ops.uv.pack_islands(rotate=True,scale=True,merge_overlap=False,margin_method='FRACTION',margin=pack_margin)
        bpy.ops.object.mode_set(mode='OBJECT')
        packed_equivalence=restore_uv_equivalence(lows,equivalence)
        apply_generated_atlas_extent(lows,atlas_packing)
        representability=correct_final_uv_representation(lows,references)
        raster,seeds=raster_chart_seed_phase(lows,layouts,resolution,repair=True)
        repair_report.update(extra_packs=1,maximum_extra_packs=1,initial_refusal=initial_refusal)

    alternate=dict(attempted=False,primary_shape='CONCAVE',maximum_extra_packs=1)
    try:validation=validate_destination_uv(lows)
    except UVTriangleOverlapError as error:
        # The microchart repack and this alternate share ONE extra native pack.
        # Neither a generic UV refusal nor supplied UV can enter this branch.
        if repair_report['extra_packs']:
            if [uv_pack_invariant(low) for low in lows]!=invariants:raise ValueError('Microchart packing changed geometry/normals/seams/sharp/matrix')
            recovered=recover_generated_uv_overlap(lows,layouts,references,seeds,resolution,error)
            if recovered is None:raise
            validation,proof=recovered;repair_report['packed_scalar_correction']=proof
        else:
            if [uv_pack_invariant(low) for low in lows]!=invariants:raise ValueError('Generated UV packing changed geometry/normals/seams/sharp/matrix')
            for low,values in zip(lows,prepack):low.data.uv_layers.active.data.foreach_set('uv',values.ravel())
            bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT');bpy.ops.uv.select_all(action='SELECT')
            bpy.ops.uv.pack_islands(rotate=True,scale=True,merge_overlap=False,margin_method='FRACTION',margin=pack_margin,shape_method='CONVEX')
            bpy.ops.object.mode_set(mode='OBJECT')
            packed_equivalence=restore_uv_equivalence(lows,equivalence)
            apply_generated_atlas_extent(lows,atlas_packing)
            representability=correct_final_uv_representation(lows,references)
            raster,seeds=raster_chart_seed_phase(lows,layouts,resolution,repair=True)
            validation=validate_destination_uv(lows)
            if [uv_pack_invariant(low) for low in lows]!=invariants:raise ValueError('CONVEX UV alternate changed geometry/normals/seams/sharp/matrix')
            repair_report.update(extra_packs=1,extra_pack_reason='exact_triangle_overlap_CONVEX')
            alternate.update(attempted=True,shape='CONVEX',initial_overlap_faces=list(error.faces),restored_normalized_prepack_UV=True)

    return dict(method='SmartProject chart seeds; bounded invalid-face seams; selected-face ANGLE_BASED; shared packing',
                cut_faces=len(bad),unwrapped_faces=selected,seed_scan=scan,
                density_normalization=normalization,
                atlas_packing=atlas_packing,packing_extent=extent,
                pack_margin_method='FRACTION',pack_margin_argument_texels=2 if atlas_packing=='full' else 2/extent,pack_gap_texels_target=4,bilinear_support_radius_per_axis=1,
                conservative_two_chart_gap_texels=2*math.sqrt(2),post_alignment_filter_support='Mandatory per-grid chart support admission; no blind radius increase',
                normalization_uv_equivalence=dict(shared_vertex_uv_exact=True),packing_uv_equivalence=packed_equivalence,
                representability=representability,generated_quantization_repair=repair_report,packing_alternate=alternate,raster_seed_alignment=raster,_raster_seed_indices=seeds,
                _destination_uv_validation=validation)


def validate_destination_uv(lows, *, collect_overlaps=False):
    triangles=[]
    for low in lows:
        mesh=low.data
        if not mesh.uv_layers.active:raise ValueError('Destination has no final UV layer')
        uv=np.empty((len(mesh.loops),2),np.float32);mesh.uv_layers.active.data.foreach_get('uv',uv.ravel())
        mesh.calc_loop_triangles()
        loop_ids=np.empty((len(mesh.loop_triangles),3),np.int32);mesh.loop_triangles.foreach_get('loops',loop_ids.ravel())
        triangles.append(uv[loop_ids])
    result=validate_uv_triangles(np.concatenate(triangles),collect_overlaps=collect_overlaps)
    return result


def shader(high, material_doc):
    original=high.data.materials[0];source=original.copy();source.name='Source transfer proxy'
    high.data.materials[0]=source;source.use_nodes=True;source.use_backface_culling=False
    bsdf=next((n for n in source.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
    if bsdf is None:raise ValueError('Imported PBR graph has no Principled surface')
    output=next(n for n in source.node_tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output)
    proxies=source_proxies(source,bsdf,material_doc)
    proxies['orm']=orm_proxy(source,proxies)
    # Alpha is data, not a projection mask. Restore this opaque original normal
    # graph for the tangent-normal bake; no custom tangent reference engine.
    for link in list(bsdf.inputs['Alpha'].links):source.node_tree.links.remove(link)
    bsdf.inputs['Alpha'].default_value=1
    return source,bsdf,output,proxies


def face_ray_cages(parts, config):
    """Disposable explicit cages: face order matches LOW; LOW data is untouched."""
    records=[]
    for index,part in enumerate(parts):
        if part['protected'] or config['cage_extrusion_relative']==0:
            records.append(dict(part=index,policy='Protected or zero-extrusion projection unchanged'));continue
        low=part['low'];before=mesh_digest(low);high_before=mesh_digest(part['high'])
        vertices,faces,_=arrays(low.data);matrix=np.asarray(low.matrix_world,dtype=np.float64)
        triangles=vertices[faces].astype(np.float64)@matrix[:3,:3].T+matrix[:3,3]
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);lengths=np.linalg.norm(normals,axis=1)
        if not np.isfinite(lengths).all() or np.any(lengths==0):raise ValueError('Face-ray cage requires finite nonzero 3D triangles')
        normals/=lengths[:,None]
        diagonal=max((low.matrix_world@Vector(a)-low.matrix_world@Vector(b)).length for a in low.bound_box for b in low.bound_box)
        extrusion=diagonal*config['cage_extrusion_relative']
        displaced=triangles+normals[:,None,:]*extrusion
        local=(displaced-matrix[:3,3])@np.linalg.inv(matrix[:3,:3]).T
        indices=np.arange(faces.size,dtype=np.int32).reshape(-1,3)
        cage=make_object('CAGE_'+low.name,local.reshape(-1,3).astype(np.float32),indices,low.matrix_world)
        cage.select_set(False)
        if len(cage.data.polygons)!=len(low.data.polygons) or len(cage.data.loops)!=len(low.data.loops):raise ValueError('Cage face/corner correspondence differs from LOW')
        cv,cf,_=arrays(cage.data)
        if not np.array_equal(cf,indices):raise ValueError('Cage triangle order changed')
        actual=cv[cf].astype(np.float64)@matrix[:3,:3].T+matrix[:3,3]
        if mesh_digest(low)!=before or mesh_digest(part['high'])!=high_before:raise ValueError('Cage creation changed source or finalized LOW attributes')
        part['projection_cage']=cage
        records.append(dict(part=index,policy='Explicit own-LOW face-normal cage; original LOW shading/UV/storage retained',
            cage_faces=len(cf),cage_corners=len(cage.data.loops),cage_vertices=len(cv),low_vertices=len(low.data.vertices),
            extrusion_world=extrusion,max_ray_distance_world=diagonal*config['projection_distance_relative'],
            maximum_cage_float32_position_error_world=float(np.linalg.norm(actual-displaced,axis=2).max()),
            original_low_geometry_uv_corner_normals_tbn_inputs_unchanged=True,source_high_unchanged=True,
            export_policy='Disposable unselected cage; export selects authored LOW objects only'))
    return records


def bake_image(high, low, target, node, image, config, *, cage=None, occupancy=False, normal=False, clear=True, margin=0):
    bpy.ops.object.select_all(action='DESELECT');low.select_set(True)
    if not occupancy:high.select_set(True)
    bpy.context.view_layer.objects.active=low;node.image=image;target.node_tree.nodes.active=node
    settings=bpy.context.scene.render.bake
    settings.use_selected_to_active=not occupancy;settings.use_clear=clear;settings.margin=margin
    settings.use_cage=not occupancy and cage is not None;settings.cage_object=cage if not occupancy else None
    diagonal=max((low.matrix_world@Vector(a)-low.matrix_world@Vector(b)).length for a in low.bound_box for b in low.bound_box)
    settings.cage_extrusion=diagonal*config['cage_extrusion_relative'];settings.max_ray_distance=diagonal*config['projection_distance_relative']
    result=bpy.ops.object.bake(type='NORMAL' if normal else 'EMIT',**({'normal_space':'TANGENT','normal_r':'POS_X','normal_g':'POS_Y','normal_b':'POS_Z'} if normal else {}))
    if result!={'FINISHED'}:raise ValueError('Cycles bake did not finish')


def pixels(image):
    result=np.empty(len(image.pixels),np.float32);image.pixels.foreach_get(result)
    if not np.isfinite(result).all():raise ValueError('Nonfinite baked pixels')
    return result.reshape(-1,4)


def composite_recovery(primary, alternate, recovery):
    """Binary selection only; never replace an already-valid primary texel."""
    if primary.shape!=alternate.shape or recovery.shape!=(len(primary),) or recovery.dtype!=np.bool_:
        raise ValueError('PBR recovery mask/image shape differs')
    result=primary.copy();result[recovery]=alternate[recovery]
    return result


def initialize_tangents(lows):
    """Explicit one-time mutation, after final UV and before preservation freeze."""
    for low in lows:
        mesh=low.data
        if not mesh.uv_layers.active:raise ValueError('Cannot initialise tangents without final UV')
        mesh.calc_tangents(uvmap=mesh.uv_layers.active.name)


def tbn_snapshot(lows, directory, label):
    """Read stored frame data only; never repair/recompute a preservation capture."""
    paths=[]
    for index,low in enumerate(lows):
        mesh=low.data
        n=np.empty((len(mesh.loops),3),np.float32);t=np.empty_like(n);sign=np.empty(len(mesh.loops),np.float32)
        mesh.corner_normals.foreach_get('vector',n.ravel());mesh.loops.foreach_get('tangent',t.ravel());mesh.loops.foreach_get('bitangent_sign',sign)
        # Pinned RNA returns zero T/sign when CD_MLOOPTANGENT is absent.
        # Refuse absent/invalidated data instead of regenerating or restoring it.
        if not len(t) or not np.isfinite(t).all() or not np.isfinite(sign).all() or not np.isin(sign,(-1,1)).all():
            raise ValueError('Stored tangent data missing or invalidated; preservation capture will not regenerate it')
        loop_vertices=np.empty(len(mesh.loops),np.int32);mesh.loops.foreach_get('vertex_index',loop_vertices)
        vertices,faces,triangle_loops=arrays(mesh);path=directory/f'tbn-{label}-{index:03d}.npz'
        smooth=np.empty(len(mesh.polygons),bool);face_n=np.empty((len(mesh.polygons),3),np.float32)
        mesh.polygons.foreach_get('use_smooth',smooth);mesh.polygons.foreach_get('normal',face_n.ravel())
        tri_faces=np.empty(len(faces),np.int32);mesh.loop_triangles.foreach_get('polygon_index',tri_faces)
        np.savez_compressed(path,n=n,t=t,sign=sign,uv=uv_loop_data(mesh),loop_vertices=loop_vertices,
                            vertices=vertices,faces=faces,triangle_loops=triangle_loops,matrix=np.asarray(low.matrix_world,dtype=np.float64),
                            smooth=smooth[tri_faces],face_n=face_n[tri_faces])
        paths.append(str(path))
    return paths


def tbn_metrics(before, after):
    """Predeclared functional bounds; derived tangent SHA differences are not gates."""
    if any(before[k].shape!=after[k].shape for k in before.keys()):return dict(valid=False,stored_inputs_exact=False,error='TBN snapshot shapes differ')
    exact=all(before[k].dtype==after[k].dtype and before[k].tobytes()==after[k].tobytes()
              for k in ('n','uv','loop_vertices','vertices','faces','triangle_loops','matrix'))
    values=[np.asarray(snapshot[k],np.float64) for snapshot in (before,after) for k in ('n','t')]
    signs=[np.asarray(snapshot['sign']) for snapshot in (before,after)]
    finite=all(np.isfinite(v).all() for v in values+signs)
    lengths=[np.linalg.norm(v,axis=1) for v in values]
    nonzero=all(np.all(v>0) for v in lengths)
    sign_valid=all(np.isin(v,(-1,1)).all() for v in signs);sign_changes=int(np.count_nonzero(signs[0]!=signs[1]))
    result=dict(stored_inputs_exact=exact,finite=bool(finite),nonzero=bool(nonzero),signs_exactly_plus_minus_one=bool(sign_valid),sign_changes=sign_changes)
    if not finite or not nonzero:
        bad=np.zeros(len(signs[0]),bool)
        for v,length in zip(values,lengths):bad|=~np.isfinite(v).all(axis=1)|(length==0)
        for sign in signs:bad|=~np.isfinite(sign)
        result['worst_invalid_loop']=int(np.flatnonzero(bad)[0])
        result.update(valid=False,error='Nonfinite or zero tangent-frame vectors');return result
    nb,tb,na,ta=values
    bb=signs[0][:,None]*np.cross(nb,tb);ba=signs[1][:,None]*np.cross(na,ta)
    if np.any(np.linalg.norm(bb,axis=1)==0) or np.any(np.linalg.norm(ba,axis=1)==0):
        result.update(valid=False,error='Degenerate tangent-frame cross product');return result
    def angles(a,b):
        a=a/np.linalg.norm(a,axis=1)[:,None];b=b/np.linalg.norm(b,axis=1)[:,None]
        return np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.einsum('ij,ij->i',a,b)))
    tangent_angles=angles(tb,ta)
    bitangent_angles=angles(bb,ba)
    norm_errors=np.stack([np.abs(v-1) for v in lengths]);dots=np.stack([np.abs(np.einsum('ij,ij->i',n,t)) for n,t in ((nb,tb),(na,ta))])
    for name,values in [('norm_error',norm_errors),('abs_dot_nt',dots),('tangent_angle_degrees',tangent_angles),('bitangent_angle_degrees',bitangent_angles)]:
        flat=int(np.argmax(values));location=np.unravel_index(flat,values.shape)
        result['max_'+name]=float(values[location]);result['worst_'+name+'_loop']=int(location[-1])
    result['valid']=bool(exact and sign_valid and sign_changes==0 and result['max_norm_error']<=1e-4 and result['max_abs_dot_nt']<=1e-4
                         and result['max_tangent_angle_degrees']<=.01 and result['max_bitangent_angle_degrees']<=.01)
    return result


def initial_generated_tangent_reassignment(parts, config, directory, paths, generation, validation, error):
    """One singleton cyclic choice, only after initial typed exact-zero refusal.

    No packing, frame setter or coverage retry. Ordinary plan/serialization and
    native occupancy/HIGH coverage still run on the resulting live snapshots.
    """
    if config['operation']!='optimize-rebake':return paths,validation
    index=getattr(error,'initial_before_cage_part',None)
    if (type(error) is not TangentProjectionZero or type(index) is not int or not 0<=index<len(paths)
            or getattr(error,'initial_before_cage_paths',None)!=tuple(paths)):
        raise ValueError('Tangent cyclic reassignment requires a marked initial before-cage refusal')
    raw=[]
    for pi,path in enumerate(paths):
        with np.load(path,allow_pickle=False) as saved:frame={k:saved[k] for k in saved.files}
        raw.append(frame)
        if pi!=index:
            try:plan_tangents(frame)
            except TangentProjectionZero:raise ValueError('Tangent cyclic reassignment requires one affected atlas part') from None
    lows=[p['low'] for p in parts]
    if len({low.data.as_pointer() for low in lows})!=len(lows):raise ValueError('Tangent cyclic LOW mesh data aliases')
    layouts=[uv_vertex_equivalence(low,include_charts=True) for low in lows]
    choice=singleton_cyclic_choice(raw[index],error,layouts[index][1])
    invariants=[uv_pack_invariant(low) for low in lows]
    loops=raw[index]['triangle_loops'][choice['face']]
    expected=raw[index]['uv'].copy();expected[loops]=expected[loops[choice['permutation']]]
    proof=dict(status='selected',part=index,**choice)
    report_path=directory/'tbn-cyclic-reassignment.json';write_json_atomic(report_path,proof)
    mesh=lows[index].data;mesh.uv_layers.active.data.foreach_set('uv',expected.ravel())
    mesh.free_tangents();mesh.calc_tangents(uvmap=mesh.uv_layers.active.name)
    updated=tbn_snapshot(lows,directory,'before-cage-cyclic')
    if [uv_pack_invariant(low) for low in lows]!=invariants:raise ValueError('Tangent cyclic reassignment changed geometry/N/seams/sharp/matrix')
    outside=np.ones(len(expected),bool);outside[loops]=False;drift=dict(valid=True,changed_loops=0)
    for pi,path in enumerate(updated):
        with np.load(path,allow_pickle=False) as saved:post={k:saved[k] for k in saved.files}
        before=raw[pi]
        if set(before)!=set(post):raise ValueError('Tangent cyclic snapshot fields differ')
        for key in before:
            if pi==index and key in ('uv','t'):continue
            if before[key].dtype!=post[key].dtype or before[key].shape!=post[key].shape or before[key].tobytes()!=post[key].tobytes():
                raise ValueError(f'Tangent cyclic reassignment changed undeclared part {pi}/{key}')
        if pi!=index:continue
        if post['uv'].dtype!=expected.dtype or post['uv'].tobytes()!=expected.tobytes():raise ValueError('Tangent cyclic UV differs from the one declared permutation')
        changed=np.flatnonzero(outside&np.any(before['t'].view(np.uint32)!=post['t'].view(np.uint32),axis=1))
        if len(changed):
            def subset(frame):
                return {k:(v[changed] if k in ('n','t','sign','uv','loop_vertices') else v) for k,v in frame.items()}
            drift=tbn_metrics(subset(before),subset(post));drift['changed_loops']=len(changed)
            for key in list(drift):
                if key.startswith('worst_') and key.endswith('_loop'):drift[key]=int(changed[drift[key]])
            drift['loop_index_scope']='Original part loop indices'
            if not drift['valid']:raise ValueError('Tangent cyclic outside-frame functional gate failed: '+str(drift))
    # Re-census actual Blender seams/opposite barriers, then freeze fresh seeds
    # at EVERY map grid. repair=False cannot translate, regularize or repack UV.
    current=[uv_vertex_equivalence(low,include_charts=True) for low in lows];offset=0
    for part,(_,before),(_,after) in zip(parts,layouts,current):
        if not np.array_equal(before,after) or not np.array_equal(after+offset,part['uv_filter_chart_ids']):
            raise ValueError('Tangent cyclic reassignment changed atlas chart ownership')
        offset+=int(after.max())+1
    validation=validate_destination_uv(lows)
    for grid in generation['map_grids']:
        size=grid['resolution'];raster,seeds=raster_chart_seed_phase(lows,current,size,repair=False)
        if len(seeds)!=len(parts):raise ValueError('Tangent cyclic seed provenance is incomplete')
        generation['footprints_by_resolution'][str(size)]=raster
        if size==generation['packing_resolution']:generation['raster_seed_alignment']=raster
        for part,indices in zip(parts,seeds):part['uv_raster_seeds_by_resolution'][size]=indices
    proof.update(status='validated_before_transfer',outside_frame=drift,modifications=1,maximum_modifications=1,
                 charts_unchanged=True,UV_region_unchanged=True,native_occupancy_and_HIGH_coverage='PENDING')
    generation['tangent_cyclic_reassignment']=proof
    write_json_atomic(report_path,proof);write_json_atomic(directory/'uv-raster-seeds.json',generation)
    return updated,validation


COVERAGE_DETAIL_LIMIT=32


def coverage_miss_facts(parts, coordinates, resolution):
    facts=[dict(x=x,y=y,candidates=[]) for x,y in coordinates[:COVERAGE_DETAIL_LIMIT]]
    for part_index,part in enumerate(parts):
        low=part['low'];vertices,faces,loops=arrays(low.data);uv=uv_loop_data(low.data)[loops]
        lower=uv.min(axis=1);upper=uv.max(axis=1)
        for fact in facts:
            start=np.array([fact['x'],fact['y']])/resolution;end=start+1/resolution
            candidates=np.flatnonzero(np.all(upper>=start,axis=1)&np.all(lower<=end,axis=1))
            for index in candidates[:16]:
                fact['candidates'].append(dict(part=part_index,high=part['high'].name,low=low.name,
                    object=part['object'],material=part['material'],triangle=int(index),
                    uv_corners=uv[index].tolist(),local_positions=vertices[faces[index]].tolist()))
            if len(candidates)>16:fact['candidates_truncated']=True
    return facts


def coverage_details(parts, missing, resolution, directory, label):
    indices=np.flatnonzero(missing)
    mask_path=directory/(label+'-missing-mask.npy')
    np.save(mask_path,missing.reshape(resolution,resolution))
    coordinates=[(int(i%resolution),int(i//resolution)) for i in indices[:COVERAGE_DETAIL_LIMIT]]
    return dict(missing=coverage_miss_facts(parts,coordinates,resolution),missing_mask=str(mask_path),
                missing_detail_limit=COVERAGE_DETAIL_LIMIT,missing_detail_count=len(coordinates),
                missing_detail_omitted=max(0,len(indices)-len(coordinates)))


class CandidateRejected(ValueError):
    def __init__(self, reason, path, details):
        super().__init__(details['error'])
        self.reason=reason;self.path=path;self.details=details
        self.simplification=None;self.atlas_group=None;self.minimum_ratios={};self.ratio_changes=[]


def final_rejection_guidance(error, config, *, retry_no_reduction=False):
    """Advice for recorded final refusals only; never changes retry or evidence."""
    if not isinstance(error,CandidateRejected) or type(config) is not dict or type(error.details) is not dict:return ''
    operation=config.get('operation')
    if type(operation) is not str or operation not in ('optimize-rebake','uv-repack-rebake','rebake-onto-mesh'):return ''
    detail=error.details.get('error')
    if type(error.reason) is not str or type(detail) is not str:return ''
    foreign=error.reason=='oriented_correspondence' and detail=='foreign_Q0_provenance'
    explanation=' Texture transfer matched a different component or material and was refused.' if foreign else ''
    if retry_no_reduction:
        return (explanation+' The conservative retry could not reduce the input within the validation limits. '
                'No output was produced. Try the original dense mesh or a higher triangle target.')
    if foreign:return explanation+' No output was produced.'
    cause=error.__cause__
    if operation=='optimize-rebake' and error.reason=='generated_uv' and cause is not None and type(cause) is UVRasterSeedError:
        from production_preflight import MAX_TRIANGLES, RESOLUTION_PRESETS
        packing=error.details.get('packing_resolution');stage=error.details.get('stage')
        grids=error.details.get('map_grids');raster=error.details.get('raster_seed_alignment')
        if (type(packing) is not int or packing not in RESOLUTION_PRESETS or type(stage) is not str
            or type(grids) is not list or not 2<=len(grids)<=len(RESOLUTION_PRESETS)
            or type(raster) is not dict or raster is not cause.report or detail!=str(cause)):return ''
        unsupported=raster.get('unsupported_charts')
        if type(unsupported) is not int or not 1<=unsupported<=MAX_TRIANGLES:return ''
        sizes=set()
        for grid in grids:
            size=grid.get('resolution') if type(grid) is dict else None
            if type(size) is not int or size not in RESOLUTION_PRESETS or size in sizes:return ''
            sizes.add(size)
        if min(sizes)!=packing:return ''
        for size in sizes:
            if size>packing and stage==f'UV chart footprint at {size}px':
                return (f' Generated UV charts failed the {size}px map grid. No output was produced. '
                        f'Try a single resolution for all active maps ({packing}px or lower). '
                        'Success is not guaranteed; validation limits are unchanged.')
        return ''
    if error.reason=='oriented_correspondence' and detail=='native_SOURCEFACE_shared_row_budget':
        if operation=='rebake-onto-mesh':
            return (' No output was produced. A lower active map resolution with the same supplied UVs may be tried, '
                    'but is not guaranteed to pass; UVs and validation limits remain unchanged.')
        packing=config.get('atlas_packing','full')
        if type(packing) is not str:return ''
        if packing=='full':
            return (' No output was produced. Try Compact atlas packing (lower texel density) or a lower active map resolution; '
                    'neither guarantees success. Validation limits are unchanged.')
        if packing=='compact':
            return (' No output was produced. Compact atlas packing is already selected; try a lower active map resolution. '
                    'Success is not guaranteed; validation limits are unchanged.')
    if error.reason=='nearest_correspondence':
        prefix='Nearest own-HIGH correspondence failed for '
        suffix=' paint samples; original position guards unchanged'
        if not detail.startswith(prefix) or not detail.endswith(suffix):return ''
        count=detail[len(prefix):-len(suffix)]
        if not count.isascii() or not count.isdecimal() or count.startswith('0'):return ''
        target=config.get('target_path')
        if operation=='uv-repack-rebake' and not (type(target) is str and target):
            return ' Correspondence validation stopped without output. Validation limits are unchanged.'
        suggestion=('Try less simplification (a higher triangle budget)' if operation=='optimize-rebake'
                    else 'Try starting with a higher-triangle LOW')
        return (' Correspondence validation stopped without output. '+suggestion+
                '; success is not guaranteed and validation limits are unchanged.')
    return ''


class ProjectionCoverageError(CandidateRejected):
    def __init__(self, coverage, path):
        super().__init__('projection_coverage',path,dict(error=f"Projection missed {coverage['miss_texels']} occupied texels; no widening or fill"))
        self.coverage=coverage


class NativeTBNRejected(CandidateRejected):
    def __init__(self, path, preflight):
        super().__init__('generated_native_tbn',path,dict(stage='pre_pbr_native_tbn',
            error='Planned tangent frame failed before PBR',failed_parts=[p['part'] for p in preflight if not p['valid']]))


def coverage_retry_ratios(parts, group, config, total):
    """Advance one existing candidate for the whole failed atlas group.

    Diagnostic samples are bounded, not exhaustive piece ownership. Retrying
    the group is conservative; protected pieces and original geometry stay fixed.
    """
    group_ids={id(part) for part in group};minimum={};changes=[]
    protected_triangles=sum(p['source_triangles'] for p in parts if p['protected'])
    ratios=simplification_ratios(config,total,protected_triangles=protected_triangles)
    for index,part in enumerate(parts):
        if id(part) not in group_ids or part['protected']:continue
        previous=part['chosen']['ratio']
        following=next((value for value in ratios if value>previous),None)
        if following is not None:
            minimum[index]=following
            changes.append(dict(part=index,object=part['object'],material=part['material'],
                previous_ratio=previous,minimum_retry_ratio=following,previous_triangles=part['chosen']['triangles']))
    return minimum,changes


def prepare_destination_uv(parts, config, directory):
    """Only generated Optimize UV rejection may advance a geometry candidate."""
    lows=[p['low'] for p in parts];stage='generated_uv';generation={};supplied_layouts=None
    material=config['plan']['materials'][parts[0]['material']]
    grids=map_grid_plan(config,transfer_channels(material));minimum=min(g['resolution'] for g in grids)
    try:
        stage=f'UV generation/footprint at {minimum}px'
        if config['operation']!='rebake-onto-mesh':
            packing=config.get('atlas_packing','full')
            generation=(final_uv(lows,minimum) if packing=='full'
                        else final_uv(lows,minimum,atlas_packing=packing))
            validation=generation.pop('_destination_uv_validation')
        else:
            supplied_layouts=[uv_vertex_equivalence(low,include_charts=True,position_keys=True) for low in lows]
            raster,seeds=raster_chart_seed_phase(lows,supplied_layouts,minimum,repair=False)
            generation=dict(method='Preserved supplied UV',raster_seed_alignment=raster,_raster_seed_indices=seeds)
            stage='destination_uv_validation';validation=validate_destination_uv(lows)
        seed_sets=generation.pop('_raster_seed_indices',None)
        if seed_sets is None or len(seed_sets)!=len(parts):raise ValueError('Missing complete UV chart seed provenance')
        for part,seeds in zip(parts,seed_sets):
            part['uv_raster_seeds_by_resolution']={minimum:seeds}
        generation['packing_resolution']=minimum
        generation['map_grids']=grids
        generation['footprints_by_resolution']={str(minimum):generation['raster_seed_alignment']}
        # Final UV is immutable: a different paint phase may not reuse a seed index
        # from the minimum grid, nor silently translate/repack for another map.
        for grid in grids:
            size=grid['resolution']
            if size==minimum:continue
            stage=f'UV chart footprint at {size}px'
            layouts=supplied_layouts if supplied_layouts is not None else [uv_vertex_equivalence(low,include_charts=True) for low in lows]
            raster,seeds=raster_chart_seed_phase(lows,layouts,size,repair=False)
            generation['footprints_by_resolution'][str(size)]=raster
            for part,indices in zip(parts,seeds):part['uv_raster_seeds_by_resolution'][size]=indices
        chart_offset=0;generation['filter_face_chart_provenance']=[]
        for pi,part in enumerate(parts):
            charts=supplied_layouts[pi][1] if supplied_layouts is not None else uv_vertex_equivalence(part['low'],include_charts=True)[1]
            part['uv_filter_chart_ids']=charts+chart_offset
            path=directory/f'uv-filter-face-charts-{pi:03d}.npy';np.save(path,part['uv_filter_chart_ids'])
            part['uv_filter_face_charts_path']=str(path)
            generation['filter_face_chart_provenance'].append(dict(part=pi,path=str(path),first_chart=chart_offset,charts=int(charts.max())+1,faces=len(charts)))
            chart_offset+=int(charts.max())+1
        write_json_atomic(directory/'uv-raster-seeds.json',generation)
    except ValueError as error:
        details=dict(stage=stage,error=str(error),packing_resolution=minimum,map_grids=grids);path=directory/'uv-rejection.json'
        raster=getattr(error,'report',generation.get('raster_seed_alignment'))
        if raster is not None:details['raster_seed_alignment']=raster
        write_json_atomic(path,details)
        hint=''
        if config['operation']=='rebake-onto-mesh' and isinstance(error,UVRasterSeedError) and isinstance(raster,dict) and raster.get('unsupported_charts',0)>0:
            hint=' Supplied UVs were unchanged. Keep a map resolution already validated for this atlas, or explicitly choose UV / Repack + Rebake to generate new UVs.'
        if config['operation']!='optimize-rebake':raise ValueError(f'{stage}: {error}; see {path}{hint}') from error
        raise CandidateRejected('generated_uv',path,details) from error
    return generation,validation


def interior_source_aim(nearest, source_triangle, low_point, tolerance):
    """One ULP-bounded source-face aim, only for noncoplanar boundary misses."""
    triangle=np.asarray(source_triangle,np.float64);nearest=np.asarray(nearest,np.float64)
    edges=triangle[1:]-triangle[0]
    values,_,rank,_=np.linalg.lstsq(edges.T,nearest-triangle[0],rcond=None)
    if rank!=2 or not np.isfinite(values).all():return None
    bary=np.r_[1-values.sum(),values]
    boundary_limit=64*np.finfo(np.float32).eps
    if bary.min()>boundary_limit or bary.min()<-boundary_limit:return None
    normal=np.cross(edges[0],edges[1]);length=np.linalg.norm(normal)
    if not math.isfinite(length) or length==0:return None
    plane_distance=abs(float((np.asarray(low_point)-triangle[0])@(normal/length)))
    # A ray through a point on the same plane but outside the face cannot
    # intersect that face. Such UV-raster extrapolation is NOT an aim problem.
    if plane_distance<=tolerance:return None
    bary=np.maximum(bary,0);bary/=bary.sum();projected=bary@triangle;center=triangle.mean(axis=0)
    distance=float(np.linalg.norm(center-projected))
    ulp=float(np.abs(np.spacing(triangle.astype(np.float32))).max())
    step=min(tolerance/4,8*ulp)
    if distance==0 or not math.isfinite(step) or step<=0:return None
    fraction=min(.25,step/distance);interior_bary=(1-fraction)*bary+fraction/3
    aim=interior_bary@triangle
    if not np.isfinite(aim).all() or np.linalg.norm(aim-nearest)>tolerance/2:return None
    return dict(point=aim.tolist(),source_boundary_barycentric=bary.tolist(),source_interior_barycentric=interior_bary.tolist(),
                step_limit_world=step,source_vertex_ulp_world=ulp,low_distance_to_source_plane_world=plane_distance)



def f32_bary_points(triangle, bary):
    """Three binary32 interpolation orders; no geometric/UV input mutation."""
    triangle=np.asarray(triangle,np.float32);bary=np.asarray(bary,np.float32)
    if triangle.shape!=(3,3) or bary.shape!=(3,) or not np.isfinite(triangle).all() or not np.isfinite(bary).all():
        raise ValueError('Invalid float32 cage interpolation input')
    return [np.add(np.add(np.multiply(triangle[i],bary[i]),np.multiply(triangle[j],bary[j])),np.multiply(triangle[k],bary[k]))
            for i,j,k in ((0,1,2),(0,2,1),(1,2,0))]


def protected_original_components(part, vertices, faces, matrix, hv, hf, hm):
    """Bind original source census labels only to unchanged protected streams."""
    if not part.get('protected') or part.get('source_component_domain')!='original_material_census_before_batching':
        return None,dict(guard='missing_original_protected_identity')
    if (vertices.dtype!=np.float32 or hv.dtype!=np.float32 or faces.dtype!=np.int32 or hf.dtype!=np.int32
            or vertices.shape!=hv.shape or faces.shape!=hf.shape or vertices.tobytes()!=hv.tobytes() or faces.tobytes()!=hf.tobytes()):
        return None,dict(guard='protected_HIGH_LOW_raw_geometry_identity_failed')
    if matrix.shape!=(4,4) or hm.shape!=(4,4) or matrix.tobytes()!=hm.tobytes() or not np.array_equal(matrix,np.eye(4)):
        return None,dict(guard='protected_HIGH_LOW_identity_matrix_failed')
    ids=np.asarray(part.get('source_component_ids',[]));sizes=np.asarray(part.get('component_sizes',[]))
    if (ids.ndim!=1 or sizes.shape!=ids.shape or not len(ids) or not np.issubdtype(ids.dtype,np.integer)
            or not np.issubdtype(sizes.dtype,np.integer) or np.any(ids<0) or len(np.unique(ids))!=len(ids)
            or np.any(sizes<=0) or int(sizes.sum())!=len(hf)):
        return None,dict(guard='missing_or_invalid_original_component_ranges')
    # Never recensus a filtered protected stream: removing other parts can merge components.
    return np.repeat(ids.astype(np.int64),sizes.astype(np.int64)),None


def axis_ulp_origin(part, vertices, faces, matrix, hv, hf, hm, hw, bvh, item, extrusion, old_cage_triangle):
    """One protected collapsed-origin precision floor; NOT an empty-ball claim."""
    low_triangle=vertices[faces[item['face']]];bary=np.asarray(item['barycentric'],np.float32)
    low_points=f32_bary_points(low_triangle,bary);old_points=f32_bary_points(old_cage_triangle,bary)
    if not all(a.tobytes()==b.tobytes() for a,b in zip(low_points,old_points)):
        return None,dict(guard='axis_floor_requires_all_three_F32_origins_collapsed')
    labels,rejection=protected_original_components(part,vertices,faces,matrix,hv,hf,hm)
    if rejection is not None:return None,rejection
    if not np.all(bary>0) or not np.all(bary<1):return None,dict(guard='axis_floor_requires_strict_interior_barycentric')
    source_face=int(item['nearest_source_face']);face=int(item['face'])
    if not 0<=source_face<len(hf) or labels[source_face]!=labels[face]:
        return None,dict(guard='nearest_source_original_component_mismatch')
    source_triangle=hw[hf[source_face]];normal=np.cross(source_triangle[1]-source_triangle[0],source_triangle[2]-source_triangle[0]);length=float(np.linalg.norm(normal))
    if not math.isfinite(length) or length<=0:return None,dict(guard='invalid_source_geometric_normal')
    normal/=length;axis=int(np.argmax(np.abs(normal)));sign=1 if normal[axis]>0 else -1
    point=np.asarray(item['low'],np.float64);nearest=np.asarray(item['nearest'],np.float64);guard=float(item['position_tolerance'])
    ulp=float(abs(np.spacing(np.float32(point[axis]))));offset=8*ulp;cap=min(float(extrusion),guard/4)
    if not math.isfinite(offset) or not 0<offset<=cap:return None,dict(guard='axis_8ULP_floor_exceeds_existing_caps',offset_world=offset if math.isfinite(offset) else None,cap_world=cap)
    vector=np.zeros(3);vector[axis]=sign*offset;stored=(low_triangle.astype(np.float64)+vector).astype(np.float32)
    if not np.isfinite(stored).all() or float(np.linalg.norm(stored.astype(np.float64)-low_triangle,axis=1).max())>cap:
        return None,dict(guard='stored_axis_floor_exceeds_existing_caps',cap_world=cap)
    component=int(labels[source_face]);other_faces=np.flatnonzero(labels!=component);other_distance=None;other_face=None;other_component=None
    if len(other_faces):
        other=BVHTree.FromPolygons(hw.tolist(),hf[other_faces].tolist(),all_triangles=True)
        location,_,index,distance=other.find_nearest(Vector(nearest))
        if location is None or index is None or not 0<=int(index)<len(other_faces) or distance is None or not math.isfinite(distance) or distance<=2*guard:
            return None,dict(guard='other_original_component_clearance_failed',clearance_world=float(distance) if distance is not None and math.isfinite(distance) else None,required_clearance_world=2*guard,scope='selected_HIGH_part_original_components')
        other_distance=float(distance);other_face=int(other_faces[int(index)]);other_component=int(labels[other_face])
    starts=np.r_[0,np.cumsum(np.asarray(part['component_sizes'],np.int64))];slot=int(np.flatnonzero(np.asarray(part['source_component_ids'])==component)[0]);rays=[]
    for order,(low,origin) in enumerate(zip(low_points,f32_bary_points(stored,bary))):
        delta=low.astype(np.float64)-origin.astype(np.float64);distance=float(np.linalg.norm(delta))
        if not math.isfinite(distance) or not 0<distance<=cap:return None,dict(guard='axis_floor_F32_origin_or_direction_invalid',order=order,origin=origin.tolist(),low=low.tolist(),origin_distance=distance if math.isfinite(distance) else None,cap_world=cap)
        direction=delta/distance
        hit,_,hit_face,hit_distance=bvh.ray_cast(Vector(origin),Vector(direction),float(item['ray_cap']))
        error=float(np.linalg.norm(np.asarray(hit)-nearest)) if hit is not None else None
        if hit is None or hit_face is None or not 0<=int(hit_face)<len(hf) or int(labels[int(hit_face)])!=component or not math.isfinite(error) or error>guard:
            return None,dict(guard='axis_floor_first_hit_or_original_component_failed',order=order,first_hit=np.asarray(hit,np.float64).tolist() if hit is not None and np.isfinite(np.asarray(hit)).all() else None,first_source_face=int(hit_face) if hit_face is not None else None,position_error=error if error is not None and math.isfinite(error) else None,position_guard=guard)
        rays.append(dict(order=order,origin=origin.tolist(),low=low.tolist(),direction=direction.tolist(),origin_distance=distance,first_hit=np.asarray(hit,np.float64).tolist(),first_source_face=int(hit_face),position_error=error))
    return stored,dict(strategy='axis_ulp_origin',axis=axis,outward_sign=sign,coordinate_ULP_world=ulp,offset_world=offset,offset_cap_world=cap,
                       low_cage_F32_collapsed_all_three=True,protected_HIGH_LOW_raw_geometry_identity=True,identity_matrix=True,
                       original_component_id=component,original_component_face_range=[int(starts[slot]),int(starts[slot+1])],
                       component_scope='selected_HIGH_part_original_components',other_component_clearance_world=other_distance,
                       other_source_face=other_face,other_component_id=other_component,no_other_components=not bool(len(other_faces)),
                       required_other_component_clearance_world=2*guard,position_guard=guard,position_guard_unchanged=True,
                       empty_ball_claim=False,F32_rays=rays)

def local_projection_cages(parts, missing, resolution, config):
    """One bounded source-directed trial; no LOW or source attributes are edited."""
    indices=np.flatnonzero(missing);records=[]
    if len(indices)>COVERAGE_DETAIL_LIMIT:
        return [],dict(policy='Local trial limited to 32 still-missing texels',skipped_texels=len(indices))
    owners={int(i):[] for i in indices}
    for pi,part in enumerate(parts):
        vertices,faces,loops=arrays(part['low'].data);uv=uv_loop_data(part['low'].data)[loops]
        lower=uv.min(axis=1);upper=uv.max(axis=1)
        for index in indices:
            xy=np.array([index%resolution,index//resolution])/resolution
            candidates=np.flatnonzero(np.all(upper>=xy-1/resolution,axis=1)&np.all(lower<=xy+2/resolution,axis=1))
            for face in candidates:
                for pixel,bary in triangle_samples(uv[face],resolution):
                    if pixel==index:owners[int(index)].append((pi,int(face),bary))
    if any(len(v)!=1 for v in owners.values()):
        return [],dict(policy='Local projection requires unique pinned raster ownership',owners={str(k):len(v) for k,v in owners.items()})
    for pi,part in enumerate(parts):
        requested=[(index,owner[0][1],owner[0][2]) for index,owner in owners.items() if owner[0][0]==pi]
        if not requested:continue
        low=part['low'];high=part['high'];before=mesh_digest(low);high_before=mesh_digest(high)
        vertices,faces,_=arrays(low.data);matrix=np.asarray(low.matrix_world,np.float64)
        triangles=vertices[faces].astype(np.float64)@matrix[:3,:3].T+matrix[:3,3]
        hv,hf,_=arrays(high.data);hm=np.asarray(high.matrix_world,np.float64)
        hw=hv.astype(np.float64)@hm[:3,:3].T+hm[:3,3]
        bvh=BVHTree.FromPolygons(hw.tolist(),hf.tolist(),all_triangles=True)
        diagonal=max((low.matrix_world@Vector(a)-low.matrix_world@Vector(b)).length for a in low.bound_box for b in low.bound_box)
        extrusion=diagonal*config['cage_extrusion_relative'];ray_cap=diagonal*config['projection_distance_relative']
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);length=np.linalg.norm(normals,axis=1)
        if np.any(length==0) or not np.isfinite(normals).all():raise ValueError('Invalid local cage face')
        displaced=triangles+extrusion*(normals/length[:,None])[:,None,:];planned=[];used=set();eligible=True;rejection={}
        for index,face,bary in requested:
            if face in used:eligible=False;rejection=dict(guard='multiple_missing_samples_per_face',pixel=int(index),face=face);break
            used.add(face);point=bary.astype(np.float64)@triangles[face]
            nearest,_,source_face,delta=bvh.find_nearest(Vector(point))
            if nearest is None or not math.isfinite(delta) or not 0<delta<ray_cap or extrusion<=0:
                eligible=False;rejection=dict(guard='nearest_distance_or_cap',pixel=int(index),face=face,nearest_distance=float(delta) if delta is not None else None,ray_cap=ray_cap,maximum_extrusion=extrusion);break
            nearest=np.asarray(nearest,np.float64);direction=nearest-point;direction/=np.linalg.norm(direction)
            displaced[face]=triangles[face]-min(extrusion,delta/2)*direction
            scale=max(1.,float(np.abs(hw).max()),float(np.ptp(hw,axis=0).max()))
            tolerance=max(delta*.01,16*np.finfo(np.float32).eps*scale)
            planned.append(dict(pixel=int(index),part=pi,face=face,barycentric=bary.tolist(),low=point.tolist(),nearest=nearest.tolist(),
                                nearest_source_face=int(source_face),nearest_distance=float(delta),position_tolerance=float(tolerance),ray_cap=float(ray_cap)))
        if not eligible:
            records.append(dict(part=pi,eligible=False,reason='Zero/invalid distance, cap, or multiple missing samples on one face',rejection=rejection));continue
        local=(displaced-matrix[:3,3])@np.linalg.inv(matrix[:3,:3]).T
        corner_faces=np.arange(faces.size,dtype=np.int32).reshape(-1,3)
        cage=make_object('LOCAL_CAGE_'+low.name,local.reshape(-1,3).astype(np.float32),corner_faces,low.matrix_world);cage.select_set(False)
        cv,cf,_=arrays(cage.data)
        if not np.array_equal(cf,corner_faces):raise ValueError('Local cage triangle order changed')
        actual=cv[cf].astype(np.float64)@matrix[:3,:3].T+matrix[:3,3]
        for item in planned:
            point=np.asarray(item['low']);bary=np.asarray(item['barycentric']);origin=bary@actual[item['face']]
            low_F32=f32_bary_points(triangles[item['face']],bary)
            old_F32=f32_bary_points(actual[item['face']],bary)
            if all(a.tobytes()==b.tobytes() for a,b in zip(low_F32,old_F32)):
                adjusted,precision=axis_ulp_origin(part,vertices,faces,matrix,hv,hf,hm,hw,bvh,item,extrusion,actual[item['face']])
                if adjusted is None:
                    eligible=False;rejection=dict(pixel=item['pixel'],face=item['face'],**precision);break
                for corner,coordinate in zip(cf[item['face']],adjusted):cage.data.vertices[int(corner)].co=coordinate
                cage.data.update();cv,cf,_=arrays(cage.data)
                actual[item['face']]=cv[cf[item['face']]].astype(np.float64)@matrix[:3,:3].T+matrix[:3,3]
                if actual[item['face']].astype(np.float32).tobytes()!=adjusted.tobytes():
                    eligible=False;rejection=dict(guard='stored_axis_cage_coordinates_changed',pixel=item['pixel'],face=item['face']);break
                origin=bary@actual[item['face']];distance=float(np.linalg.norm(origin-point))
                first=precision['F32_rays'][0]
                item.update(strategy='axis_ulp_origin',precision_floor=precision,origin=origin.tolist(),origin_distance=distance,
                            analytic_first_source_face=first['first_source_face'],analytic_first_hit=first['first_hit'])
                continue
            distance=float(np.linalg.norm(origin-point))
            if not 0<distance<item['nearest_distance']:
                eligible=False;rejection=dict(guard='float32_origin_not_inside_empty_ball',pixel=item['pixel'],face=item['face'],origin=origin.tolist(),origin_distance=distance,nearest_distance=item['nearest_distance']);break
            direction=(point-origin)/distance
            hit,_,source_face,_=bvh.ray_cast(Vector(origin),Vector(direction),item['ray_cap'])
            if hit is None or np.linalg.norm(np.asarray(hit)-item['nearest'])>item['position_tolerance']:
                aim=interior_source_aim(item['nearest'],hw[hf[item['nearest_source_face']]],point,item['position_tolerance'])
                if aim is None:
                    eligible=False;rejection=dict(guard='first_hit_miss' if hit is None else 'first_hit_far',pixel=item['pixel'],face=item['face'],low=point.tolist(),origin=origin.tolist(),nearest=item['nearest'],nearest_source_face=item['nearest_source_face'],first_hit=list(hit) if hit is not None else None,first_hit_error=float(np.linalg.norm(np.asarray(hit)-item['nearest'])) if hit is not None else None,position_tolerance=item['position_tolerance'],interior_aim_eligible=False);break
                initial_hit=list(hit) if hit is not None else None
                aim_direction=np.asarray(aim['point'])-point;aim_direction/=np.linalg.norm(aim_direction)
                adjusted=triangles[item['face']]-min(extrusion,item['nearest_distance']/2)*aim_direction
                adjusted_local=((adjusted-matrix[:3,3])@np.linalg.inv(matrix[:3,:3]).T).astype(np.float32)
                for corner,coordinate in zip(cf[item['face']],adjusted_local):cage.data.vertices[int(corner)].co=coordinate
                cage.data.update();cv,cf,_=arrays(cage.data)
                actual[item['face']]=cv[cf[item['face']]].astype(np.float64)@matrix[:3,:3].T+matrix[:3,3]
                origin=bary@actual[item['face']];distance=float(np.linalg.norm(origin-point))
                if not 0<distance<item['nearest_distance']:
                    eligible=False;rejection=dict(guard='interior_aim_origin_not_inside_empty_ball',pixel=item['pixel'],face=item['face'],origin_distance=distance,nearest_distance=item['nearest_distance'],interior_aim=aim);break
                direction=(point-origin)/distance
                hit,_,source_face,_=bvh.ray_cast(Vector(origin),Vector(direction),item['ray_cap'])
                if hit is None or np.linalg.norm(np.asarray(hit)-item['nearest'])>item['position_tolerance']:
                    eligible=False;rejection=dict(guard='interior_aim_first_hit_miss' if hit is None else 'interior_aim_first_hit_far',pixel=item['pixel'],face=item['face'],first_hit=list(hit) if hit is not None else None,nearest=item['nearest'],position_tolerance=item['position_tolerance'],interior_aim=aim);break
                item.update(interior_aim=aim,initial_rejected_hit=initial_hit)
            item.update(origin=origin.tolist(),origin_distance=distance,analytic_first_source_face=int(source_face),analytic_first_hit=list(hit))
        if mesh_digest(low)!=before or mesh_digest(high)!=high_before:raise ValueError('Local cage modified HIGH or finalized LOW')
        if eligible:part['local_projection_cage']=cage;records.extend(planned)
        else:remove(cage);records.append(dict(part=pi,eligible=False,reason='Float32 origin or first-hit guard failed',rejection=rejection))
    bounds=[]
    for part in parts:
        if not part.get('local_projection_cage'):continue
        v,_,_=arrays(part['high'].data);m=np.asarray(part['high'].matrix_world,np.float64);bounds.append(v.astype(np.float64)@m[:3,:3].T+m[:3,3])
    if not bounds:return [],dict(records=records,policy='No eligible local cage; existing strict coverage rejection retained')
    lower=np.min([v.min(axis=0) for v in bounds],axis=0);upper=np.max([v.max(axis=0) for v in bounds],axis=0)
    span=np.maximum(upper-lower,np.finfo(np.float32).eps)
    return [r for r in records if 'pixel' in r],dict(records=records,position_lower=lower.tolist(),position_span=span.tolist(),
        encoding='RGB=0.25+0.5*(worldPosition-lower)/span; clear black is not a hit',
        position_guard='distance(nativeHit, nearestOwnHIGH)<=max(nearestDistance*0.01,16*float32eps*worldScale)',
        source=('Only corresponding own HIGH selected; axis_ulp_origin is a protected F32-collapsed precision floor, not a delta/2 empty-ball claim; original caps and position guard unchanged'
                if any(record.get('strategy')=='axis_ulp_origin' for record in records) else
                'Only corresponding own HIGH selected; one shortened cage trial; all caps unchanged'))


def local_position_hit(color, nearest, lower, span, tolerance):
    color=np.asarray(color,np.float64);point=np.asarray(lower)+(color-.25)*2*np.asarray(span)
    encoded=bool(np.isfinite(color).all() and np.all(color>=.25-1e-6) and np.all(color<=.75+1e-6))
    error=float(np.linalg.norm(point-nearest))
    return dict(native_encoded_hit=encoded,native_position=point.tolist(),native_position_error=error,
                native_position_accepted=bool(encoded and math.isfinite(error) and error<=tolerance))


def position_proxy(source, output, lower, span):
    nodes=source.node_tree.nodes;links=source.node_tree.links
    geometry=nodes.new('ShaderNodeNewGeometry');subtract=nodes.new('ShaderNodeVectorMath');subtract.operation='SUBTRACT'
    links.new(geometry.outputs['Position'],subtract.inputs[0]);subtract.inputs[1].default_value=lower
    scale=nodes.new('ShaderNodeVectorMath');scale.operation='MULTIPLY';links.new(subtract.outputs['Vector'],scale.inputs[0])
    scale.inputs[1].default_value=(.5/np.asarray(span)).tolist()
    offset=nodes.new('ShaderNodeVectorMath');offset.operation='ADD';links.new(scale.outputs['Vector'],offset.inputs[0]);offset.inputs[1].default_value=(.25,.25,.25)
    set_proxy(source,output,offset.outputs['Vector'])


def prepare_transfer(parts, material_doc, config, directory):
    resolution=config['resolution'];lows=[p['low'] for p in parts]
    sources=[shader(p['high'],material_doc) for p in parts]
    target=bpy.data.materials.new('Bake Target');target.use_nodes=True
    for low in lows:low.data.materials.clear();low.data.materials.append(target)
    node=target.node_tree.nodes.new('ShaderNodeTexImage')
    target_output=target.node_tree.nodes.get('Material Output')
    set_proxy(target,target_output,(1,1,1,1))
    def bake_all(image, *, occupancy=False, normal=False, margin=0, face_cage=False, local_cage=False, face_part_indices=None):
        selected=[p for p in parts if p.get('local_projection_cage')] if local_cage else parts
        if face_cage and not local_cage and face_part_indices is not None and margin==0:
            selected=[parts[index] for index in face_part_indices]
        for index,part in enumerate(selected):
            bake_image(part['high'],part['low'],target,node,image,config,cage=part.get('local_projection_cage') if local_cage else part.get('projection_cage') if face_cage else None,
                       occupancy=occupancy,normal=normal,clear=index==0,margin=margin)
        return len(selected)
    # Both masks use the actual Blender rasterizer, with zero margin. Native
    # padding is never counted as recovered source information.
    image=new_map('Destination occupancy',resolution,'Non-Color');bake_all(image,occupancy=True)
    occupancy=pixels(image)[:,0]>.9;occupancy_path=save_map(image,directory,'occupancy-self');bpy.data.images.remove(image)
    if not occupancy.any():raise ValueError('Target UV has no occupied pixels')
    if any('uv_raster_seeds' not in part for part in parts):raise ValueError('Missing native UV chart seed provenance')
    raster=native_chart_seed_report([p['uv_raster_seeds'] for p in parts],occupancy)
    write_json_atomic(directory/'raster-seed-native.json',raster)
    if not raster['valid']:raise ValueError(f"Native occupancy lost {raster['unsupported_charts']} genuine UV chart seeds")
    for source,bsdf,output,proxies in sources:set_proxy(source,output,proxies['coverage'])
    image=new_map('Source coverage',resolution,'Non-Color');bake_all(image)
    primary=pixels(image)[:,0]>.9
    coverage_path=save_map(image,directory,'coverage');bpy.data.images.remove(image)
    primary_missing=occupancy&~primary;primary_details=coverage_details(parts,primary_missing,resolution,directory,'primary')
    face=np.zeros_like(primary);face_path=None
    if primary_missing.any() and any(p.get('projection_cage') is not None for p in parts):
        image=new_map('Face cage coverage',resolution,'Non-Color');bake_all(image,face_cage=True)
        face=pixels(image)[:,0]>.9;face_path=save_map(image,directory,'face-coverage');bpy.data.images.remove(image)
    face_recovery=primary_missing&face;hit=primary|face_recovery;still_missing=occupancy&~hit
    local_recovery=np.zeros_like(primary);local_report=dict(policy='Not needed')
    if still_missing.any():
        local_plan,local_report=local_projection_cages(parts,still_missing,resolution,config)
        if local_plan:
            for part,(source,bsdf,output,proxies) in zip(parts,sources):
                if part.get('local_projection_cage'):position_proxy(source,output,local_report['position_lower'],local_report['position_span'])
            image=new_map('Local source position coverage',resolution,'Non-Color',float_buffer=True);bake_all(image,local_cage=True)
            data=pixels(image);lower=np.asarray(local_report['position_lower']);span=np.asarray(local_report['position_span'])
            for item in local_plan:
                item.update(local_position_hit(data[item['pixel'],:3],np.asarray(item['nearest']),lower,span,item['position_tolerance']))
                local_recovery[item['pixel']]=item['native_position_accepted']
            local_report['samples']=local_plan;bpy.data.images.remove(image)
            for source,bsdf,output,proxies in sources:set_proxy(source,output,proxies['coverage'])
        write_json_atomic(directory/'local-projection.json',local_report)
    local_recovery&=still_missing
    recovery=face_recovery|local_recovery;recovery.setflags(write=False);hit=primary|recovery;missing=occupancy&~hit
    masks={}
    for label,mask in [('occupancy',occupancy),('primary-hit',primary),('face-hit',face),('face-recovery',face_recovery),('local-recovery',local_recovery),('recovery',recovery),('union-hit',hit)]:
        path=directory/(label+'-mask.npy');np.save(path,mask.reshape(resolution,resolution));masks[label]=str(path)
    details=coverage_details(parts,missing,resolution,directory,'union')
    image=new_map('Coverage union mask',resolution,'Non-Color');mask_pixels=np.ones((len(hit),4),np.float32)
    mask_pixels[:,:3]=hit[:,None];image.pixels.foreach_set(mask_pixels.ravel());image.update()
    union_path=save_map(image,directory,'union-coverage');bpy.data.images.remove(image);del mask_pixels
    coverage=dict(raster_chart_seeds=raster,occupied_texels=int(occupancy.sum()),hit_texels=int((occupancy&hit).sum()),
        miss_texels=int(missing.sum()),filled_texels=0,fallback_texels=int(recovery.sum()),recovered_texels=int(recovery.sum()),
        face_recovered_texels=int(face_recovery.sum()),local_recovered_texels=int(local_recovery.sum()),
        face_recovery_mask=masks['face-recovery'],local_recovery_mask=masks['local-recovery'],local_projection=local_report,
        face_recovery_mask_sha256=hashlib.sha256(face_recovery.tobytes()).hexdigest(),local_recovery_mask_sha256=hashlib.sha256(local_recovery.tobytes()).hexdigest(),
        primary_miss_texels=int(primary_missing.sum()),face_miss_texels=int((occupancy&~face).sum()) if face_path else None,
        face_lost_primary_texels_not_applied=int((occupancy&primary&~face).sum()) if face_path else 0,primary_hits_discarded=0,
        recovery_mask=masks['recovery'],recovery_mask_sha256=hashlib.sha256(recovery.tobytes()).hexdigest(),masks=masks,primary_details=primary_details,
        projection_configs=dict(primary='Implicit Blender vertex-normal extrusion',recovery='Explicit own-part face-normal cage',
                                cage_extrusion_relative=config['cage_extrusion_relative'],projection_distance_relative=config['projection_distance_relative']),
        coordinate_origin='bottom-left',
        occupancy_method='Cycles LOW-to-SELF opaque white emission, selected_to_active=False, margin=0',
        coverage_method='Native primary, face recovery, then one guarded local source-position cage trial; immutable missing masks; margin=0',
        analytic_role='UV pixel-rectangle candidates are diagnostics only, never occupancy authority',
        occupancy_map=str(occupancy_path),coverage_map=str(union_path),primary_coverage_map=str(coverage_path),
        face_coverage_map=str(face_path) if face_path else None,**details)
    write_json_atomic(directory/'coverage.json',coverage)
    if coverage['miss_texels']:raise ProjectionCoverageError(coverage,directory/'coverage.json')
    return target,node,sources,bake_all,coverage


def exact_source_arrays_match(raw, vertices, faces, loops, normals, matrix):
    """Exact owner identity, with only the currently qualified normal gauge."""
    if not np.array_equal(matrix,np.eye(4)):return False
    fields=dict(vertices=vertices,faces=faces,triangle_loops=loops,n=normals,matrix=matrix)
    return all(value.dtype==raw[key].dtype and value.shape==raw[key].shape and value.tobytes()==raw[key].tobytes()
               and np.isfinite(value).all() for key,value in fields.items())


def remove_source_self(part):
    clone=part.pop('source_self',None);material=part.pop('source_self_material',None)
    if clone is not None:remove(clone)
    if material is not None and not material.users:bpy.data.materials.remove(material)
    for key in ('source_self_node','source_self_target_UV','source_self_source_UV','source_self_UV_sha','source_self_digest'):part.pop(key,None)


def prepare_source_self(part,raw,source_state,material_doc,hv,hf,hloops,hm):
    """Private full-source clone, never a face filter or independent-LOW route."""
    if not part.get('protected'):return None,'not_protected'
    source,bsdf,_,_=source_state
    if material_doc.get('normalTexture') is not None or bsdf.inputs['Normal'].is_linked:return None,'perturbed_source_not_qualified'
    high=part['high'];low=part['low']
    lineage=(high.as_pointer(),low.as_pointer(),part['object'],part['material'],tuple(part['source_component_ids']),tuple(tuple(x) for x in part['source_component_face_ranges']))
    if (part.get('_source_lineage')!=lineage or part.get('source_component_domain')!='original_material_census_before_batching'
        or source.get('mesh_finishing_material_index')!=part['material']):return None,'source_lineage_not_proven'
    actual=np.empty_like(raw['n']);high.data.corner_normals.foreach_get('vector',actual.ravel())
    if not exact_source_arrays_match(raw,hv,hf,hloops,actual,hm):return None,'source_arrays_or_identity_gauge_not_proven'
    lv,lf,ll=arrays(low.data);low.data.corner_normals.foreach_get('vector',actual.ravel())
    if not exact_source_arrays_match(raw,lv,lf,ll,actual,np.asarray(low.matrix_world,np.float64)) or uv_loop_data(low.data).tobytes()!=raw['uv'].tobytes():return None,'live_LOW_identity_not_proven'
    if len(high.data.uv_layers)!=1:return None,'source_UV_layout_not_qualified'
    source_uv=high.data.uv_layers.active.name;target_uv='__mesh_finishing_target_UV'
    if source_uv==target_uv:return None,'reserved_target_UV_collision'
    for texture in source.node_tree.nodes:
        if texture.type!='TEX_IMAGE':continue
        for link in texture.inputs['Vector'].links:
            if not (link.from_node.type=='UVMAP' and link.from_node.uv_map in ('',source_uv)):
                return None,'source_texture_coordinates_not_qualified'
    clone=high.copy();clone.data=high.data.copy();bpy.context.collection.objects.link(clone);part['source_self']=clone
    try:
        state=shader(clone,material_doc);material=state[0];part['source_self_material']=material
        source_layer=clone.data.uv_layers[source_uv];original_uv=np.empty_like(raw['uv']);source_layer.data.foreach_get('uv',original_uv.ravel())
        layer=clone.data.uv_layers.new(name=target_uv)
        if layer.name!=target_uv:raise ValueError('Exact-source target UV name changed')
        layer.data.foreach_set('uv',raw['uv'].ravel());source_layer=clone.data.uv_layers[source_uv];source_layer.active_render=True;clone.data.uv_layers.active_index=len(clone.data.uv_layers)-1
        uvnode=material.node_tree.nodes.new('ShaderNodeUVMap');uvnode.uv_map=source_uv
        for texture in list(material.node_tree.nodes):
            if texture.type=='TEX_IMAGE':
                for link in list(texture.inputs['Vector'].links):material.node_tree.links.remove(link)
                material.node_tree.links.new(uvnode.outputs['UV'],texture.inputs['Vector'])
        part['source_self_node']=material.node_tree.nodes.new('ShaderNodeTexImage');part['source_self_target_UV']=target_uv;part['source_self_source_UV']=source_uv
        bpy.context.view_layer.update();v,f,loops=arrays(clone.data);clone.data.corner_normals.foreach_get('vector',actual.ravel())
        if not exact_source_arrays_match(raw,v,f,loops,actual,np.asarray(clone.matrix_world,np.float64)) or uv_loop_data(clone.data).tobytes()!=raw['uv'].tobytes():raise ValueError('Exact-source clone changed geometry/Ns/target UV')
        check=np.empty_like(original_uv);source_layer.data.foreach_get('uv',check.ravel())
        if check.tobytes()!=original_uv.tobytes():raise ValueError('Exact-source original UV changed')
        part['source_self_UV_sha']=hashlib.sha256(original_uv.tobytes()).hexdigest();part['source_self_digest']=mesh_digest(clone)
        return state,'protected_live_exact_unperturbed_identity'
    except Exception:
        remove_source_self(part);raise


def bake_source_part(part,target,node,image,*,normal=False,clear=True):
    if 'source_self' not in part:return bake_sample_proxy(part,target,node,image,normal=normal,clear=clear)
    clone=part['source_self'];material=part['source_self_material'];texture=part['source_self_node']
    bpy.ops.object.select_all(action='DESELECT');clone.hide_render=False;clone.hide_set(False);clone.select_set(True);bpy.context.view_layer.objects.active=clone
    texture.image=image;material.node_tree.nodes.active=texture
    settings=bpy.context.scene.render.bake;settings.use_selected_to_active=False;settings.use_cage=False;settings.cage_object=None
    settings.use_clear=clear;settings.margin=0;settings.cage_extrusion=0.;settings.max_ray_distance=0.;settings.target='IMAGE_TEXTURES'
    result=bpy.ops.object.bake(type='NORMAL' if normal else 'EMIT',uv_layer=part['source_self_target_UV'],**({'normal_space':'OBJECT','normal_r':'POS_X','normal_g':'POS_Y','normal_b':'POS_Z'} if normal else {}))
    if result!={'FINISHED'}:raise ValueError('Exact-source SELF bake did not finish')
    uv=np.empty((len(clone.data.loops),2),np.float32);clone.data.uv_layers[part['source_self_source_UV']].data.foreach_get('uv',uv.ravel())
    if mesh_digest(clone)!=part['source_self_digest'] or hashlib.sha256(uv.tobytes()).hexdigest()!=part['source_self_UV_sha']:raise ValueError('SELF bake changed frozen source clone fields')
    if 'source_self_triangles_sha' in part:
        _,faces,_=arrays(clone.data)
        if hashlib.sha256(faces.tobytes()).hexdigest()!=part['source_self_triangles_sha'] or not np.array_equal(np.asarray(clone.matrix_world),np.eye(4)):
            raise ValueError('Sparse Q1 SELF changed frozen triangles/identity matrix')


def remove_q1_face_self(part):
    context=part.pop('q1_face_self',None)
    if context is not None:remove_source_self(context)


def prepare_q1_face_self(part,source_state,material_doc,error,samples,context,config,directory,index):
    """Explicit sparse SELF context after a valid ray POSITION / refused FACE.

    The full original HIGH remains intact. This is face-directed shader
    evaluation, not a visibility or ray-equivalence certificate.
    """
    budget=config.get('_q1_face_self_budget');high=part['high'];source,bsdf,_,_=source_state
    rows=error.rows;cache=context.get('shading');resolution=config['resolution']
    if (budget is None or budget['rows']+len(rows)>32 or budget['parts']>=4 or cache is None
        or part.get('q1_face_self') is not None or 'source_self' in part or material_doc.get('normalTexture') is not None
        or bsdf.inputs['Normal'].is_linked or not np.array_equal(np.asarray(high.matrix_world),np.eye(4))
        or not np.array_equal(samples['LOW_matrix'],np.eye(4))):raise ValueError('Sparse Q1 SELF scope/resource/gauge unsupported')
    lineage=(high.as_pointer(),part['low'].as_pointer(),part['object'],part['material'],tuple(part['source_component_ids']),tuple(tuple(x) for x in part['source_component_face_ranges']))
    if (part.get('_source_lineage')!=lineage or part.get('source_component_domain')!='original_material_census_before_batching'
        or source.get('mesh_finishing_material_index')!=part['material'] or high.modifiers or len(high.data.uv_layers)!=1):raise ValueError('Sparse Q1 SELF original source lineage/UV unsupported')
    source_uv=high.data.uv_layers.active.name;target_uv='__mesh_finishing_Q1_SELF_UV'
    if not source_uv or source_uv==target_uv:raise ValueError('Sparse Q1 SELF named source UV unsupported')
    for texture in source.node_tree.nodes:
        if texture.type=='TEX_IMAGE' and any(not (link.from_node.type=='UVMAP' and link.from_node.uv_map in ('',source_uv)) for link in texture.inputs['Vector'].links):raise ValueError('Sparse Q1 SELF source texture coordinates unsupported')
    estimate=len(cache['faces'])*3*64
    if budget['clone_bytes']+estimate>256*1024**2 or estimate+resolution*resolution*16*3>512*1024**2:raise ValueError('Sparse Q1 SELF bounded buffer/clone admission exceeded')
    bary,details=q1_self_interiors(error,samples['recovered_rows'],samples['nearest'],samples['guards'],samples['LOW_points'],samples['ray_cap'],context,samples['LOW_Ns'])
    indices=samples['indices'][rows];faces=error.proposed;guard=samples['guards'][rows];points=samples['nearest'][rows]
    target,predicted=q1_self_uv(cache['world'][cache['faces'][faces]],bary,indices,resolution,points,guard)
    budget.update(rows=budget['rows']+len(rows),parts=budget['parts']+1,clone_bytes=budget['clone_bytes']+estimate)
    proof=directory/f'q1-face-self-{index:03d}';proof.mkdir();sparse={};image=None
    record=dict(method='Explicit face-directed full-HIGH SELF, not nearest-ray-equivalent',rows=details,indices=indices.tolist(),samples=len(rows),
        original_guards=guard.tolist(),original_ray_cap=samples['ray_cap'],CPU_predicted_position_errors=predicted,source_UV=source_uv,target_UV=target_uv,
        budget=dict(budget),resource_scope='Bounded array/clone estimate; native Blender allocations remain cooperative, not a strict RSS guarantee',status='pending')
    try:
        clone=high.copy();sparse['source_self']=clone;clone.data=high.data.copy();bpy.context.collection.objects.link(clone)
        state=shader(clone,material_doc);material=state[0];sparse['source_self_material']=material
        original_uv=np.empty((len(clone.data.loops),2),np.float32);clone.data.uv_layers[source_uv].data.foreach_get('uv',original_uv.ravel())
        source_check=np.empty_like(original_uv);high.data.uv_layers[source_uv].data.foreach_get('uv',source_check.ravel())
        if source_check.tobytes()!=original_uv.tobytes():raise ValueError('Sparse Q1 SELF clone changed named source UV')
        uv=np.full_like(original_uv,-4);uv[cache['loops'][faces]]=target
        layer=clone.data.uv_layers.new(name=target_uv)
        if layer.name!=target_uv:raise ValueError('Sparse Q1 SELF target UV name collision')
        layer.data.foreach_set('uv',uv.ravel());clone.data.uv_layers[source_uv].active_render=True;clone.data.uv_layers.active_index=len(clone.data.uv_layers)-1
        uvnode=material.node_tree.nodes.new('ShaderNodeUVMap');uvnode.uv_map=source_uv
        for texture in list(material.node_tree.nodes):
            if texture.type=='TEX_IMAGE':
                for link in list(texture.inputs['Vector'].links):material.node_tree.links.remove(link)
                material.node_tree.links.new(uvnode.outputs['UV'],texture.inputs['Vector'])
        sparse.update(source_self_node=material.node_tree.nodes.new('ShaderNodeTexImage'),source_self_target_UV=target_uv,source_self_source_UV=source_uv,
            source_self_UV_sha=hashlib.sha256(original_uv.tobytes()).hexdigest(),indices=indices,record=record)
        bpy.context.view_layer.update();v,f,l=arrays(clone.data);n=np.empty_like(cache['corner_n']);clone.data.corner_normals.foreach_get('vector',n.ravel())
        if (not np.array_equal(v.astype(float),cache['world']) or f.tobytes()!=cache['faces'].tobytes() or l.tobytes()!=cache['loops'].tobytes()
            or n.tobytes()!=cache['corner_n'].tobytes() or not np.array_equal(np.asarray(clone.matrix_world),np.eye(4))):raise ValueError('Sparse Q1 SELF clone changed original geometry/Ns/matrix')
        sparse['source_self_digest']=mesh_digest(clone);sparse['source_self_triangles_sha']=hashlib.sha256(f.tobytes()).hexdigest();lower,span,_=position_domain(cache['world'])
        position_proxy(material,state[2],lower,span);image=new_map('Sparse Q1 SELF POSITION',resolution,'Non-Color',float_buffer=True)
        bake_source_part(sparse,None,None,image);data=pixels(image);np.save(proof/'POSITION-rgba.npy',data)
        observed=lower+(data[indices,:3].astype(float)-.25)*2*span;errors=np.linalg.norm(observed-points,axis=1)
        nonclear=np.flatnonzero(np.any(data[:,:3]!=0,axis=1));encoded=np.flatnonzero(np.all((data[:,:3]>=.25-1e-6)&(data[:,:3]<=.75+1e-6),axis=1))
        record.update(native_POSITION=str(proof/'POSITION-rgba.npy'),maximum_POSITION_error=float(errors.max()),native_POSITION_passes=1)
        if (not np.isfinite(data).all() or not np.array_equal(nonclear,np.sort(indices)) or not np.array_equal(encoded,np.sort(indices))
            or not np.isfinite(errors).all() or np.any(errors>guard) or np.any(np.linalg.norm(observed-samples['LOW_points'][rows],axis=1)>samples['ray_cap'])):raise ValueError('Sparse Q1 SELF native POSITION set/original guards failed')
        bpy.data.images.remove(image);image=None;del data
        record['native_SOURCEFACE']={}
        native_source_face_witness(sparse,state,None,sparse['source_self_node'],indices,np.arange(len(indices),dtype=np.int32),faces,context['components'],resolution,proof,record['native_SOURCEFACE'],index)
        record.update(status='native_verified',channels=[]);write_json_atomic(proof/'proof.json',record)
        sparse['proof_path']=proof/'proof.json';return sparse,state
    except Exception as error:
        record.update(status='refused',error=str(error));write_json_atomic(proof/'proof.json',record);remove_source_self(sparse);raise
    finally:
        if image is not None:bpy.data.images.remove(image)


def bake_q1_face_overlay(part,target,node,image,*,normal=False):
    sparse=part.get('q1_face_self')
    if sparse is None:return
    scratch=new_map('Proven sparse Q1 SELF shader',image.size[0],image.colorspace_settings.name,float_buffer=image.is_float)
    try:
        bake_source_part(sparse,target,node,scratch,normal=normal);values=pixels(scratch);indices=sparse['indices']
        if not np.isfinite(values).all():raise ValueError('Nonfinite sparse Q1 SELF shader field')
        data=pixels(image);data[indices]=values[indices];image.pixels.foreach_set(data.ravel());image.update()
        sparse['record']['channels'].append('native_NORMAL' if normal else 'native_EMIT');write_json_atomic(sparse['proof_path'],sparse['record'])
    finally:bpy.data.images.remove(scratch)


def bake_sample_proxy(part, target, node, image, *, normal=False, clear=True):
    """Native own-HIGH shader bake, identity proxy, original final-LOW caps."""
    proxy=part['sample_proxy'];cage=part['sample_cage'];high=part['high']
    bpy.ops.object.select_all(action='DESELECT');proxy.select_set(True);high.select_set(True)
    bpy.context.view_layer.objects.active=proxy;node.image=image;target.node_tree.nodes.active=node
    settings=bpy.context.scene.render.bake;settings.use_selected_to_active=True;settings.use_clear=clear
    settings.margin=0;settings.use_cage=True;settings.cage_object=cage
    settings.cage_extrusion=part['sample_caps']['extrusion'];settings.max_ray_distance=part['sample_caps']['ray_cap']
    result=bpy.ops.object.bake(type='NORMAL' if normal else 'EMIT',**({'normal_space':'OBJECT','normal_r':'POS_X','normal_g':'POS_Y','normal_b':'POS_Z'} if normal else {}))
    if result!={'FINISHED'}:raise ValueError('Nearest-sample native bake did not finish')


def native_source_face_witness(part, source_state, target, node, indices, recovered_rows, proposed_faces, components, resolution, directory, record, part_index):
    """One full-own-HIGH FACE observation after final POSITION, before PBR admission."""
    source,_,output,_=source_state;observed=part['source_self'] if 'source_self' in part else part['high'];mesh=observed.data;name='__modly_source_face'
    nodes=source.node_tree.nodes;links=source.node_tree.links;surface=output.inputs['Surface']
    def graph_state():
        defaults=tuple((n.as_pointer(),tuple((i.name,repr(tuple(i.default_value) if hasattr(i.default_value,'__len__') else i.default_value)) for i in n.inputs if hasattr(i,'default_value'))) for n in nodes)
        return defaults,sorted((l.from_socket.as_pointer(),l.to_socket.as_pointer()) for l in links)
    attributes=tuple((a.name,a.domain,a.data_type) for a in mesh.attributes);before=mesh_digest(observed);graph_before=graph_state()
    old_links=[(l.from_socket,l.to_socket) for l in surface.links];old_active=nodes.active;old_image=node.image
    temporary=[];created=False;image=None
    record.update(status='pending',source_face_count=len(components),raw_field=str(directory/f'source-face-{part_index:03d}-rgba.npy'),
                  calls=1,full_own_HIGH=True,after_final_POSITION=True,coincident_hit_stability='NOT_QUALIFIED')
    try:
        if name in mesh.attributes:raise ValueError('Reserved SOURCEFACE attribute already exists')
        mesh.calc_loop_triangles();polygon_ids=np.empty(len(mesh.loop_triangles),np.int32);mesh.loop_triangles.foreach_get('polygon_index',polygon_ids)
        values=source_face_polygon_values(polygon_ids,len(mesh.polygons))
        if len(values)!=len(components):raise ValueError('SOURCEFACE triangle count differs from frozen original domain')
        attribute=mesh.attributes.new(name=name,type='FLOAT',domain='FACE');created=True;attribute.data.foreach_set('value',values)
        attribute_node=nodes.new('ShaderNodeAttribute');temporary.append(attribute_node);attribute_node.attribute_type='GEOMETRY';attribute_node.attribute_name=name
        emission=nodes.new('ShaderNodeEmission');temporary.append(emission);emission.inputs['Strength'].default_value=1.
        links.new(attribute_node.outputs['Fac'],emission.inputs['Color'])
        for link in list(surface.links):links.remove(link)
        links.new(emission.outputs['Emission'],surface)
        image=new_map('Native own HIGH SOURCEFACE',resolution,'Non-Color',float_buffer=True)
        try:bake_source_part(part,target,node,image)
        finally:
            # Persist the actual full field even if bake/decode fails; never synthesize IDs.
            data=np.empty((resolution*resolution,4),np.float32);image.pixels.foreach_get(data.ravel())
            np.save(record['raw_field'],data);record['raw_sha256']=digest(record['raw_field'])
        record.update(decode_source_face_witness(data,indices,len(components),recovered_rows,proposed_faces,components),status='native_verified')
    except Exception as error:
        record.update(status='refused',error=str(error));raise
    finally:
        for temporary_node in reversed(temporary):nodes.remove(temporary_node)
        for link in list(surface.links):links.remove(link)
        for origin,destination in old_links:links.new(origin,destination)
        nodes.active=old_active;node.image=old_image
        if created:mesh.attributes.remove(mesh.attributes[name])
        if image is not None:bpy.data.images.remove(image)
        if tuple((a.name,a.domain,a.data_type) for a in mesh.attributes)!=attributes or mesh_digest(observed)!=before or graph_state()!=graph_before:
            raise ValueError('SOURCEFACE observation failed to restore SOURCE attributes/material graph/defaults')


def verify_source_self(part,source_state,target,node,raw,owners,selected,resolution,directory,index,config):
    """All-owner native context/position proof; no nearest-ray/Q1 claims."""
    indices=owners['indices'][selected];faces=owners['faces'][selected];weights=owners['weights'][selected]
    hv,hf,_=arrays(part['high'].data);lower,span,domain=position_domain(hv.astype(float))
    guard=16*float(np.finfo(np.float32).eps)*domain['world_scale']
    diagonal=max((part['low'].matrix_world@Vector(a)-part['low'].matrix_world@Vector(b)).length for a in part['low'].bound_box for b in part['low'].bound_box)
    ray_cap=diagonal*config['projection_distance_relative']
    record=dict(part=index,method='Exact protected source SELF at original owner interpolants; no geometry ray',paint_samples=len(indices),proxy_faces=0,
                source_component_ids=part['source_component_ids'],source_component_domain=part['source_component_domain'],
                oracle='Runtime exact source geometry/corner Ns/matrix lineage; native FACE equals independent LOW owner',
                original_guard_rule='Unchanged16*eps32*worldScale precision floor; never larger than nearest-ray g0',
                guard=guard,original_LOW_ray_cap=ray_cap,cage_offset=0.,normal_gauge='Identity OBJECT is WORLD; unperturbed only',
                source_UV=part['source_self_source_UV'],target_UV=part['source_self_target_UV'],native_POSITION_passes=1)
    points=np.empty((len(indices),3),np.float64)
    for j,(face,bary) in enumerate(zip(faces,weights)):points[j]=sample_world(raw,int(face),bary)
    source,_,output,_=source_state;position_proxy(source,output,lower,span)
    image=new_map('Exact source SELF POSITION',resolution,'Non-Color',float_buffer=True)
    path=directory/f'exact-source-SELF-{index:03d}-samples.npz'
    try:
        bake_source_part(part,target,node,image);data=np.empty((resolution*resolution,4),np.float32);image.pixels.foreach_get(data.ravel());observed=data[indices].copy()
        nonclear=np.flatnonzero(np.any(data[:,:3]!=0,axis=1));encoded=np.flatnonzero(np.all((data[:,:3]>=.25-1e-6)&(data[:,:3]<=.75+1e-6),axis=1))
        positions=lower+(observed[:,:3].astype(float)-.25)*2*span;errors=np.linalg.norm(positions-points,axis=1)
        np.savez_compressed(path,indices=indices,owner_faces=faces,weights=weights,original_points=points,native_POSITION=observed,positions=positions,errors=errors,guard=np.float64(guard))
        record.update(samples=str(path),samples_sha256=digest(path),maximum_native_position_error=float(errors.max(initial=0)),
                      native_sample_set_exact=bool(np.array_equal(nonclear,np.sort(indices)) and np.array_equal(encoded,np.sort(indices))))
        if not np.isfinite(data).all() or not record['native_sample_set_exact'] or not np.isfinite(errors).all() or np.any(errors>guard) or np.any(errors>ray_cap):
            raise ValueError('Exact-source SELF POSITION failed independent owner set/original precision or LOW cap')
    except Exception as error:
        if 'data' in locals():
            failed=directory/f'exact-source-SELF-{index:03d}-failed-POSITION.npy';np.save(failed,data);record['failed_POSITION_raw']=str(failed)
        record['failure']=str(error);write_json_atomic(directory/f'exact-source-SELF-{index:03d}.json',record);raise
    finally:bpy.data.images.remove(image)
    del data
    components=original_source_face_ids(part['source_component_face_ranges'],part['source_component_ids'],len(hf));face_record={};record['native_SOURCEFACE']=face_record
    try:native_source_face_witness(part,source_state,target,node,indices,np.arange(len(indices),dtype=np.int32),faces,components,resolution,directory,face_record,index)
    finally:
        if 'recovered_rows' in face_record:face_record['verified_owner_rows']=face_record.pop('recovered_rows')
        if 'full_proxy_samples' in face_record:face_record['full_SELF_samples']=face_record.pop('full_proxy_samples')
        face_record['method']='SELF original face identity, not Q1 recovery or geometry-ray tie'
        write_json_atomic(directory/f'exact-source-SELF-{index:03d}.json',record)
    record.update(accepted=len(indices),failed=0,filled_texels=0,recovered_texels=0)
    write_json_atomic(directory/f'exact-source-SELF-{index:03d}.json',record)
    return record


def _check_native_owner_inputs(low, raw, pi, *, lineage=False):
    mesh=low.data;vertices,faces,loops=arrays(mesh)
    n=np.empty((len(mesh.loops),3),np.float32);t=np.empty_like(n);sign=np.empty(len(mesh.loops),np.float32)
    mesh.corner_normals.foreach_get('vector',n.ravel());mesh.loops.foreach_get('tangent',t.ravel());mesh.loops.foreach_get('bitangent_sign',sign)
    actual=dict(vertices=vertices,faces=faces,triangle_loops=loops,uv=uv_loop_data(mesh),n=n,t=t,sign=sign,matrix=np.asarray(low.matrix_world,np.float64))
    if lineage:
        loop_vertices=np.empty(len(mesh.loops),np.int32);mesh.loops.foreach_get('vertex_index',loop_vertices)
        smooth=np.empty(len(mesh.polygons),bool);face_n=np.empty((len(mesh.polygons),3),np.float32)
        mesh.polygons.foreach_get('use_smooth',smooth);mesh.polygons.foreach_get('normal',face_n.ravel())
        polygon_ids=np.empty(len(mesh.loop_triangles),np.int32);mesh.loop_triangles.foreach_get('polygon_index',polygon_ids)
        actual.update(loop_vertices=loop_vertices,smooth=smooth[polygon_ids],face_n=face_n[polygon_ids])
    changed=[key for key,value in actual.items() if value.dtype!=raw[key].dtype or value.shape!=raw[key].shape or value.tobytes()!=raw[key].tobytes()]
    if changed:
        raise ValueError(f'Native SELF ownership changed frozen LOW inputs: part={pi}; fields={changed}')


def _bake_native_owner_batch(parts, raw_frames, plan, target, node, image, config, directory, counts, record):
    """One temporary geometry-only SELF bake; failures never rerun individually."""
    obj=None;mesh=None;rgba=None;before={};names=('__modly_owner_face','__modly_owner_w0','__modly_owner_w1')
    try:
        vertices=[];faces=[];uv=[];ids=[];smooth=[];vertex_offset=0;offsets=np.r_[0,np.cumsum(counts,dtype=np.int64)]
        for pi in plan['parts']:
            low=parts[pi]['low'];raw=raw_frames[pi];_check_native_owner_inputs(low,raw,pi,lineage=True);before[pi]=mesh_digest(low)
            if any(name in low.data.attributes for name in names):raise ValueError('Reserved native ownership attribute already exists')
            low.data.calc_loop_triangles();polygons=np.empty(len(low.data.loop_triangles),np.int32);low.data.loop_triangles.foreach_get('polygon_index',polygons)
            if len(low.data.polygons)!=counts[pi] or not np.array_equal(np.sort(polygons),np.arange(counts[pi])):raise ValueError('Native ownership requires one-to-one triangulated polygons and corners')
            vertices.append(raw['vertices']);faces.append(raw['faces']+vertex_offset);vertex_offset+=len(raw['vertices'])
            uv.append(raw['uv'][raw['triangle_loops']]);smooth.append(raw['smooth']);ids.append(np.arange(offsets[pi]+1,offsets[pi+1]+1,dtype=np.float32))
        mesh=bpy.data.meshes.new('Temporary grouped native SELF');mesh.from_pydata(np.concatenate(vertices).tolist(),[],np.concatenate(faces).tolist());mesh.update()
        obj=bpy.data.objects.new('Temporary grouped native SELF',mesh);bpy.context.collection.objects.link(obj);obj.rotation_mode='QUATERNION';obj.matrix_world=Matrix(raw_frames[plan['parts'][0]]['matrix'])
        mesh.uv_layers.new(name='MeshFinishingUV');mesh.uv_layers.active.data.foreach_set('uv',np.concatenate(uv).ravel());mesh.polygons.foreach_set('use_smooth',np.concatenate(smooth));mesh.materials.append(target)
        actual_vertices,actual_faces,loops=arrays(mesh)
        if (actual_vertices.tobytes()!=np.concatenate(vertices).tobytes() or actual_faces.tobytes()!=np.concatenate(faces).tobytes() or uv_loop_data(mesh)[loops].tobytes()!=np.concatenate(uv).tobytes()
                or np.asarray(obj.matrix_world,np.float64).tobytes()!=raw_frames[plan['parts'][0]]['matrix'].tobytes()):
            record.update(advisory_not_used=True,reason='Temporary SELF representation is not byte-exact; independent lane retained')
            return None,{}
        w0=np.zeros(len(mesh.loops),np.float32);w1=w0.copy();w0[loops[:,0]]=1;w1[loops[:,1]]=1
        for name,domain,values in zip(names,('FACE','CORNER','CORNER'),(np.concatenate(ids),w0,w1)):
            attribute=mesh.attributes.new(name=name,type='FLOAT',domain=domain);attribute.data.foreach_set('value',values)
        record['native_calls']=1
        try:bake_image(obj,obj,target,node,image,config,occupancy=True,clear=True,margin=0)
        finally:
            rgba=np.empty((config['resolution']**2,4),np.float32);image.pixels.foreach_get(rgba.ravel())
        rows=decode_native_owner_batch(rgba,counts,config['resolution'],plan)
        return rgba,rows
    except Exception:
        if rgba is not None:
            nonzero=np.flatnonzero(np.any(rgba[:,:3]!=0,axis=1)|~np.isfinite(rgba[:,3]))
            np.savez_compressed(directory/'native-owner-batch-failed.npz',indices=nonzero,rgba=rgba[nonzero],image_shape=np.array([config['resolution'],config['resolution'],4],np.int32))
        raise
    finally:
        if obj is not None:remove(obj)
        elif mesh is not None and not mesh.users:bpy.data.meshes.remove(mesh)
        for pi,expected in before.items():
            _check_native_owner_inputs(parts[pi]['low'],raw_frames[pi],pi,lineage=True)
            if mesh_digest(parts[pi]['low'])!=expected:raise ValueError('Grouped SELF changed original frozen LOW')


def _native_owner_shader(target):
    names=('__modly_owner_face','__modly_owner_w0','__modly_owner_w1')
    nodes=target.node_tree.nodes;links=target.node_tree.links;output=nodes.get('Material Output');temporary=[]
    emission=nodes.new('ShaderNodeEmission');emission.inputs['Strength'].default_value=1.;temporary.append(emission)
    combine=nodes.new('ShaderNodeCombineXYZ');temporary.append(combine)
    negate=nodes.new('ShaderNodeMath');negate.operation='MULTIPLY';negate.use_clamp=False;negate.inputs[1].default_value=-1.;temporary.append(negate)
    for name,channel in zip(names,('X','Y',None)):
        attribute=nodes.new('ShaderNodeAttribute');attribute.attribute_type='GEOMETRY';attribute.attribute_name=name;temporary.append(attribute)
        if channel:links.new(attribute.outputs['Fac'],combine.inputs[channel])
        else:links.new(attribute.outputs['Fac'],negate.inputs[0]);links.new(negate.outputs[0],combine.inputs['Z'])
    links.new(combine.outputs['Vector'],emission.inputs['Color'])
    for link in list(output.inputs['Surface'].links):links.remove(link)
    links.new(emission.outputs['Emission'],output.inputs['Surface'])
    return temporary


def native_low_owners(parts, raw_frames, target, node, config, directory):
    """Independent native SELF ownership, optionally one proved disjoint CPU group.

    Every unresolved part is still cleared/baked independently; all observations
    retain own-part decoding and cross-part collision checks. Exact UV-interior
    SAT and all later POSITION/FACE/PBR gates remain unchanged.
    """
    resolution=config['resolution'];counts=[len(raw['faces']) for raw in raw_frames]
    if len(parts)!=len(raw_frames) or sum(counts)>=2**24:raise ValueError('Native owner parts/FLOAT ID range unsupported')
    if resolution*resolution>MAX_ATLAS_PIXELS:raise ValueError('Native owner atlas resource bound exceeded')
    names=('__modly_owner_face','__modly_owner_w0','__modly_owner_w1')
    nodes=target.node_tree.nodes;output=nodes.get('Material Output');temporary=_native_owner_shader(target)
    image=new_map('Native final LOW owner and corner weights',resolution,'Non-Color',float_buffer=True)
    occupancy=np.zeros(resolution*resolution,bool);streams=[];records=[];offset=0
    metadata=dict(codec='R=global flat FACE triangle ID+1 (positive geometry coverage); G=w0; B=-w1; w2=fl(fl(1-w0)-w1)',
                  face_counts=counts,coordinate_origin='bottom-left',alpha_role='Ignored: SELF EMIT alpha is not coverage',
                  ownership='Native SELF FACE/CORNER shader observations; CPU paint ownership not used',
                  shared_edges='Within-part native owner is authoritative; cross-part occupied pixels rejected',
                  native_settings=dict(float_buffer=True,colorspace='Non-Color',use_selected_to_active=False,use_cage=False,margin=0,clear_each_part=True),parts=records)
    batch_pixels=None;batch_rows={}
    try:
        scene=bpy.context.scene
        plan=native_owner_batch_plan(raw_frames,resolution,cpu=scene.render.engine=='CYCLES' and scene.cycles.device=='CPU',margin=0)
        if plan is not None and len({parts[pi]['object'] for pi in plan['parts']})==1:
            metadata['native_batch']=dict(plan['facts'],attempted=True,completed=False,native_calls=0)
            batch_pixels,batch_rows=_bake_native_owner_batch(parts,raw_frames,plan,target,node,image,config,directory,counts,metadata['native_batch'])
            metadata['native_batch']['completed']=batch_pixels is not None
        plan=None # Release the O(atlas) proof mask before individual decoding.
        for pi,(part,raw) in enumerate(zip(parts,raw_frames)):
            low=part['low'];mesh=low.data;created=[];before=mesh_digest(low);attribute_names=set(mesh.attributes.keys())
            check_inputs=lambda:_check_native_owner_inputs(low,raw,pi)
            check_inputs()
            try:
                if any(name in attribute_names for name in names):raise ValueError('Reserved native ownership attribute already exists')
                mesh.calc_loop_triangles();polygon_ids=np.empty(len(mesh.loop_triangles),np.int32);mesh.loop_triangles.foreach_get('polygon_index',polygon_ids)
                if len(mesh.polygons)!=counts[pi] or not np.array_equal(np.sort(polygon_ids),np.arange(len(mesh.polygons))) or not np.array_equal(np.sort(raw['triangle_loops'].ravel()),np.arange(len(mesh.loops))):
                    raise ValueError('Native ownership requires one-to-one triangulated polygons and corners')
                ids=np.empty(len(mesh.polygons),np.float32);ids[polygon_ids]=np.arange(offset+1,offset+counts[pi]+1,dtype=np.float32)
                w0=np.zeros(len(mesh.loops),np.float32);w1=w0.copy();w0[raw['triangle_loops'][:,0]]=1;w1[raw['triangle_loops'][:,1]]=1
                for name,domain,values in zip(names,('FACE','CORNER','CORNER'),(ids,w0,w1)):
                    attribute=mesh.attributes.new(name=name,type='FLOAT',domain=domain);created.append(attribute.name);attribute.data.foreach_set('value',values)
                record=dict(part=pi,global_FACE_ID_range=[offset+1,offset+counts[pi]],raw_fields=str(directory/f'native-owner-{pi:03d}.npz'))
                records.append(record);write_json_atomic(directory/'native-owners.json',metadata)
                try:
                    if pi not in batch_rows:bake_image(part['high'],low,target,node,image,config,occupancy=True,clear=True,margin=0)
                finally:
                    # Required raw RGB fields survive bake/decode failures, including
                    # nonfinite values. Outside these indices RGB is exactly zero;
                    # alpha is not used by any ownership or occupancy decision.
                    if pi in batch_rows:
                        # The complete group was validated before attribution;
                        # no malformed/foreign observation is silently cleared.
                        rgba=np.zeros((resolution*resolution,4),np.float32);rows=batch_rows[pi];rgba[rows]=batch_pixels[rows]
                    else:
                        rgba=np.empty((resolution*resolution,4),np.float32);image.pixels.foreach_get(rgba.ravel())
                    nonzero=np.flatnonzero(np.any(rgba[:,:3]!=0,axis=1)|~np.isfinite(rgba[:,3]))
                    np.savez_compressed(record['raw_fields'],indices=nonzero,rgba=rgba[nonzero],image_shape=np.array([resolution,resolution,4],np.int32))
                    record['raw_fields_sha256']=digest(record['raw_fields']);write_json_atomic(directory/'native-owners.json',metadata)
                decoded=decode_native_owners(rgba,counts,resolution,part=pi);indices=decoded['indices']
                overlap=indices[occupancy[indices]]
                if len(overlap):raise ValueError(f'Ambiguous native ownership across own-HIGH parts: part={pi} count={len(overlap)} first={overlap[:8].tolist()}')
                if int(occupancy.sum())+len(indices)>MAX_PROXY_SAMPLES:raise ValueError('Native owner total sample resource bound exceeded')
                occupancy[indices]=True;streams.append(decoded)
                record.update(occupied_texels=len(indices),negative_weight_samples=decoded['negative_weight_samples'],maximum_unit_sum_residual=decoded['maximum_unit_sum_residual'],signed_transport_observed=decoded['signed_transport_observed'],HDR_FACE_ID_observed=decoded['HDR_FACE_ID_observed'])
                offset+=counts[pi]
            finally:
                for name in reversed(created):mesh.attributes.remove(mesh.attributes[name])
                if set(mesh.attributes.keys())!=attribute_names or mesh_digest(low)!=before:raise ValueError('Temporary native ownership attributes leaked or LOW changed')
                check_inputs()
        if not occupancy.any():raise ValueError('Target UV has no native LOW occupied pixels')
        owners={key:np.concatenate([stream[key] for stream in streams]) for key in ('indices','parts','faces','weights')}
        order=np.argsort(owners['indices']);owners={key:value[order] for key,value in owners.items()}
        np.savez_compressed(directory/'native-owners-decoded.npz',**owners)
        metadata.update(complete=True,occupied_texels=int(occupancy.sum()),decoded_fields=str(directory/'native-owners-decoded.npz'))
        write_json_atomic(directory/'native-owners.json',metadata)
        return owners,occupancy,metadata
    except Exception as exc:
        metadata.update(complete=False,failure=str(exc));write_json_atomic(directory/'native-owners.json',metadata);raise
    finally:
        for temporary_node in reversed(temporary):nodes.remove(temporary_node)
        set_proxy(target,output,(1,1,1,1));bpy.data.images.remove(image)


def record_floor_refinement(report, refinement, *, part, pixel, face):
    """Count construction admission only, never recovered shader/PBR coverage."""
    if refinement is None:return
    counts=report['construction_refinement'];counts['attempted']+=1
    counts['admitted' if refinement['admitted'] else 'unsupported']+=1
    for key in ('transitions','cells_tested','quantizer_evaluations'):counts[key]+=int(refinement[key])
    if len(counts['details'])<counts['detail_limit']:
        counts['details'].append(dict(part=int(part),pixel=int(pixel),LOW_face=int(face),**refinement))


def present_vertex_q0_row(*,native_set_exact,accepted,errors,guards,low_to_nearest,low_to_native_hit,ray_cap,
                          strategies,initial_auxiliary,recovered_rows,exact_q0_rows,nearest,source_faces,world,high_faces,
                          attempted_samples,present_attempted,proof_rows,proof_bytes,proof_limit):
    """One present nominal compatible Q0 only; missing-only policy is separate."""
    if not native_set_exact:return None,'native_sample_set_not_exact'
    if np.count_nonzero(~accepted)!=1:return None,'not_one_present_rejection'
    if attempted_samples>=32 or present_attempted:return None,'grid_refinement_budget_exceeded'
    j=int(np.argmin(accepted))
    if strategies[j] or initial_auxiliary[j] or j in recovered_rows or j in exact_q0_rows:
        return None,'not_nominal_compatible_Q0'
    if (not np.isfinite([errors[j],guards[j],low_to_nearest[j],low_to_native_hit[j],ray_cap]).all()
        or not 0<guards[j]<errors[j] or min(low_to_nearest[j],low_to_native_hit[j])<0
        or max(low_to_nearest[j],low_to_native_hit[j])>ray_cap):return None,'original_Q_or_LOW_caps_ineligible'
    if not np.any(np.all(world[high_faces[int(source_faces[j])]]==nearest[j],axis=1)):
        return None,'Q0_not_exact_own_HIGH_vertex'
    if proof_rows>=65536 or proof_bytes+4>proof_limit:return None,'native_SOURCEFACE_or_provenance_budget'
    return j,'eligible_present_vertex_Q0'


def present_vertex_q0_auxiliary(point,nearest,triangle,low_triangle,index,resolution,extrusion,ray_cap,guard,first_hit,expected_face,budget):
    """Strongest existing construction, then three complete-own-HIGH queries.

    Debited to existing fallback work; no alternate search after a failed hit.
    These CPU witnesses never replace all-sample native POSITION or SOURCEFACE.
    """
    if budget['rows']>=budget['max_rows'] or budget['rays']+3>budget['max_rays'] or budget['work_seconds']>=budget['max_seconds']:
        raise _AuxiliaryProposalRejected('present_Q0_work_budget','Existing auxiliary work budget exhausted')
    budget['rows']+=1;started=time.monotonic()
    try:
        V,C,U,fact=_auxiliary_source_sampler(point,nearest,triangle,low_triangle,index,resolution,extrusion,ray_cap,guard,strongest=True)
        bary=np.asarray(fact['raster_bary'],np.float32);checks=[]
        for a,b,c in ((0,1,2),(0,2,1),(1,2,0)):
            if budget['work_seconds']+time.monotonic()-started>=budget['max_seconds']:
                raise _AuxiliaryProposalRejected('present_Q0_work_budget','Existing auxiliary work time exhausted')
            interp=lambda x:np.add(np.add(x[a]*bary[a],x[b]*bary[b]),x[c]*bary[c]).astype(float)
            sample=interp(V);origin=interp(C);delta=sample-origin;length=float(np.linalg.norm(delta))
            if not np.isfinite(length) or not 0<length<=min(extrusion,guard/4):
                raise _AuxiliaryProposalRejected('present_Q0_origin_cap','Stored auxiliary separation exceeds original caps')
            budget['rays']+=1
            hit,_,face,distance=first_hit(origin,delta/length,ray_cap)
            if face is None:
                if hit is not None or distance is not None:raise ValueError('Inconsistent own-HIGH ray miss')
                raise _AuxiliaryProposalRejected('present_Q0_first_hit','Complete own-HIGH ray missed')
            hit=np.asarray(hit,float)
            if (hit.shape!=(3,) or not np.isfinite(hit).all() or not isinstance(face,(int,np.integer)) or face<0
                or distance is None or not np.isfinite(distance) or distance<0):raise ValueError('Invalid own-HIGH first-hit result')
            error=float(np.linalg.norm(hit-nearest));low_distance=float(np.linalg.norm(hit-point))
            if face!=expected_face or error>guard or low_distance>ray_cap or distance>ray_cap:
                raise _AuxiliaryProposalRejected('present_Q0_first_hit','Complete own-HIGH first hit differs from original face/Q/LOW guards')
            checks.append(dict(face=int(face),Q_error=error,LOW_to_hit=low_distance))
        if budget['work_seconds']+time.monotonic()-started>=budget['max_seconds']:
            raise _AuxiliaryProposalRejected('present_Q0_work_budget','Existing auxiliary work time exhausted')
        fact['present_vertex_Q0_first_hits']=dict(faces=[c['face'] for c in checks],checks=checks,native_verification='PENDING_AT_CONSTRUCTION')
        return V,C,U,fact
    finally:budget['work_seconds']+=time.monotonic()-started


def _exact_serialized_surface(source, candidate):
    """Final reimport only: exact F64 oriented triangles, multiplicity and AABB.

    No hash/tolerance/F32 projection. Cyclic starts, face order and vertex/index
    splits alone are ignored. Full keys are sorted only after sampled rejection.
    """
    proof=dict(equal=False,method='exact_world_F64_oriented_triangle_multiset_and_AABB',triangles=source['triangles'])
    pairs=[]
    for surface in (source,candidate):
        v=np.asarray(surface.get('serialized_world_vertices'));f=np.asarray(surface.get('serialized_faces'));m=np.asarray(surface.get('matrix_world_float64'))
        if (v.dtype!=np.float64 or v.ndim!=2 or v.shape[1]!=3 or not 0<len(v)<=6_000_000 or not np.isfinite(v).all()
            or f.ndim!=2 or f.shape[1]!=3 or not np.issubdtype(f.dtype,np.integer) or not 0<len(f)<=2_000_000
            or f.min()<0 or f.max()>=len(v) or m.dtype!=np.float64 or m.shape!=(4,4) or not np.isfinite(m).all()):
            proof['reason']='unsupported_or_nonfinite_exact_surface';return proof
        pairs.append((v,f,m))
    (av,af,am),(bv,bf,bm)=pairs
    proof['triangle_counts_equal']=len(af)==len(bf)==source['triangles']==candidate['triangles']
    proof['matrix_equal']=am.tobytes()==bm.tobytes()
    proof['AABB_equal']=(av.min(axis=0).tobytes()==bv.min(axis=0).tobytes() and av.max(axis=0).tobytes()==bv.max(axis=0).tobytes())
    if not all(proof[k] for k in ('triangle_counts_equal','matrix_equal','AABB_equal')):
        proof['reason']='count_matrix_or_AABB_differs';return proof
    def keys(v,f):
        # Same finite2M SOURCE face/6M vertex domain; at most144MB per full key
        # array, plus fixed65536-face scratch. Reuse already-computed world arrays.
        dtype=np.dtype([(str(i),np.uint64) for i in range(9)]);out=np.empty(len(f),dtype)
        for start in range(0,len(f),65536):
            triangles=v[f[start:start+65536]]
            area=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            if not np.isfinite(area).all() or np.any(np.all(area==0,axis=1)):return None
            bits=triangles.view(np.uint64)
            cycles=np.stack([bits,np.roll(bits,1,axis=1),np.roll(bits,2,axis=1)],axis=1).reshape(-1,3,9)
            first=np.lexsort(tuple(cycles[:,:,i] for i in range(8,-1,-1)),axis=1)[:,0]
            chosen=np.ascontiguousarray(cycles[np.arange(len(cycles)),first])
            out[start:start+len(triangles)]=chosen.view(dtype).ravel()
        out.sort();return out
    a=keys(av,af);b=keys(bv,bf)
    if a is None or b is None:proof['reason']='degenerate_or_nonfinite_area';return proof
    proof['equal']=bool(np.array_equal(a,b));proof['reason']='exact_equal' if proof['equal'] else 'oriented_triangle_multiset_differs'
    return proof


def _serialized_surface_validation(source, candidate, sampled):
    """Keep raw sampled verdict unchanged; exact exception is final-only."""
    if sampled['valid']:
        return dict(effective_valid=True,sampled_valid=True,method='sampled_surface_bounds',exact_world_surface=None)
    proof=_exact_serialized_surface(source,candidate)
    return dict(effective_valid=proof['equal'],sampled_valid=False,method=proof['method'] if proof['equal'] else 'failed_sampled_and_exact',exact_world_surface=proof)


def _sourceface_row_prepass(selected, nearest, low_points, guards, source_faces, classify, *, eligible, native_f32_normals=False):
    """Necessary shared-row rejection only; a complete cache is not acceptance.

    Existing dense buffers hold P/Q/g/FACE. Extra native N/d/compatibility is
    33 bytes per owner, or21 for explicitly checked native mathutils.Vector
    binary32 normals, plus4096 bytes reserved for fixed array metadata.
    The64MiB job-local cap never expands the SOURCE/query/provenance budgets.
    """
    dtype=np.dtype([('normal',np.float32 if native_f32_normals else np.float64,(3,)),('distance',np.float64),('compatible',np.bool_)])
    required=len(selected)*dtype.itemsize+4096;cap=64*1024**2
    facts=dict(enabled=False,scratch_bytes=0,required_scratch_bytes=required,scratch_cap_bytes=cap,
        classified_samples=0,incompatible_rows=0,shared_part_limit=65536,shared_row_limit_exceeded=False,
        complete=False,constructed_samples=0,native_SOURCEFACE_verified=0,
        scope='Necessary shared-row count only; every incompatible Q0 needs one row, including exact-surviving Q0. No Q1/exact admission or native verification.')
    if not eligible or len(selected)<=65536 or required>cap:return None,facts
    scratch=np.empty(len(selected),dtype);facts.update(enabled=True,scratch_bytes=required)
    for j,row in enumerate(selected):
        point,where,normal,face,distance,guard,compatible=classify(j,row)
        if native_f32_normals and type(normal) is not Vector:raise ValueError('Native prepass normal cache requires mathutils.Vector')
        low_points[j]=point;nearest[j]=where;guards[j]=guard;source_faces[j]=face
        scratch[j]=(normal,distance,compatible);facts['classified_samples']=j+1
        if not compatible:
            facts['incompatible_rows']+=1
            if facts['incompatible_rows']>65536:
                facts['shared_row_limit_exceeded']=True;return scratch,facts
    facts['complete']=True
    return scratch,facts


def oriented_construction_rejection(error, report, directory, construction_started, part, resolution):
    """Retain an unfinished part's wall time separately from completed parts."""
    report['timing']['nearest_proxy_construction_partial']=dict(
        seconds=time.monotonic()-construction_started,part=part,complete=False,
        scope='In-progress part only; not added to completed-part nearest_proxy_construction_seconds')
    reason=str(error);details=dict(error=reason)
    if reason=='oriented_query_deadline':
        budget=report['oriented_SOURCE']['budget']
        details.update(reason=reason,error=(
            f"{reason}: INTERNAL oriented query-work budget of {budget['max_seconds']:g} seconds exceeded "
            f"(observed {budget['work_seconds']:.6f} query-work seconds) at {resolution}x{resolution}px. "
            'Choose a lower resolution previously validated for this input; a longer outer timeout does not change this internal budget.'))
    rejected=CandidateRejected('oriented_correspondence',directory/'nearest-correspondence.json',details)
    rejected.correspondence_failure=reason
    return rejected


def prepare_nearest_transfer(parts, material_doc, config, directory, raw_frames, planned_frames, *, sources=None):
    correspondence_started=time.monotonic()
    resolution=config['resolution']
    if sources is None:sources=[shader(p['high'],material_doc) for p in parts]
    bake_sources=list(sources)
    spaces=[item.identifier for item in bpy.ops.object.bake.get_rna_type().properties['normal_space'].enum_items]
    if 'OBJECT' not in spaces:raise ValueError('Native OBJECT normal space required for nearest-sample transfer')
    target=bpy.data.materials.new('Bake Target');target.use_nodes=True
    for part in parts:part['low'].data.materials.clear();part['low'].data.materials.append(target)
    node=target.node_tree.nodes.new('ShaderNodeTexImage');set_proxy(target,target.node_tree.nodes.get('Material Output'),(1,1,1,1))
    ownership_started=time.monotonic()
    owners,occupancy,ownership=native_low_owners(parts,raw_frames,target,node,config,directory)
    ownership_seconds=time.monotonic()-ownership_started
    if any(resolution not in p.get('uv_raster_seeds_by_resolution',{}) for p in parts):raise ValueError(f'Missing UV chart footprint provenance at {resolution}px')
    raster=native_chart_seed_report([p['uv_raster_seeds_by_resolution'][resolution] for p in parts],occupancy);write_json_atomic(directory/'raster-seed-native.json',raster)
    if not raster['valid']:raise ValueError('Native LOW occupancy lost generated chart seeds')
    np.save(directory/'occupancy-mask.npy',occupancy.reshape(resolution,resolution));occupancy.setflags(write=False)
    report=dict(method='Native own-HIGH POSITION/FACE before PBR: rays or proven exact protected source SELF',
                resolution=resolution,raster_chart_seeds=raster,occupancy_method='Independent final LOW native SELF FACE/CORNER EMIT, positive FACE ID, margin0',native_ownership=ownership,
                occupied_texels=int(occupancy.sum()),hit_texels=0,miss_texels=0,pending_texels=int(occupancy.sum()),complete=False,filled_texels=0,
                recovered_texels=0,coordinate_origin='bottom-left',parts=[],OBJECT_normal_space=True,visual_qualification='NOT_RUN',
                shader_footprint='Native isolated microproxy footprint; not old PRIMARY/FACE texture-derivative equivalence',
                direct_edge_sampler='Valid nominal DIRECT retained by default. A strict three-order nearest-face witness is only an auxiliary selection heuristic, not native acceptance. First admissible auxiliary Aim-centered normal sampler from source-coordinate inset1..8ULPs may be selected; each Aim within original Q guard/8 and all original LOW caps; rejected auxiliary proposal retains nominal DIRECT bytes, not PBR fallback/recovery; original LOW-relative user caps and original Q native POSITION guard retained; final LOW/TBN unchanged; no empty-ball claim in auxiliary mode',
                short_origin='delta<=original_guard/8, zero, or F32-collapsed: nearest-HIGH-point anchored geometric-normal 3*guard/32 floor (near-distance ideal offset<=7*guard/32, reserving guard/32 before F32 roundoff), sign=-1 iff dot(normalized geometric normal,Q-LOW)<0 else+1, actual LOW-to-cage<=guard/4, no empty-ball claim; otherwise min(original_extrusion,delta/2); only cage-dependent floor rejection enables explicit actual-F32-cell refinement within nominal floor, max256 coordinate transitions; construction admission, not recovered PBR',
                precision_floor_reason_counts={},reason_counted_samples=0,reason_counts_complete=False,
                construction_refinement=dict(attempted=0,admitted=0,unsupported=0,transitions=0,cells_tested=0,quantizer_evaluations=0,details=[],detail_limit=32,complete=False,scope='Nominal LOW-centred constructions only; excludes auxiliary source-sampler attempts, including discarded auxiliary refinement records; not PBR coverage or recovery'),
                resource_caps=dict(max_samples=MAX_PROXY_SAMPLES,max_atlas_pixels=MAX_ATLAS_PIXELS),
                masks=dict(occupancy=str(directory/'occupancy-mask.npy')),occupancy_map=str(directory/'occupancy-mask.npy'))
    report['paint_raster_batch_admitted_rows']=0
    report['timing']=dict(native_ownership_and_readback_seconds=ownership_seconds,
        nearest_proxy_construction_seconds=0.,native_POSITION_FACE_and_validation_seconds=0.,
        scope='Observed wall phases; construction/POSITION_FACE counters cover ray parts, SELF remains in total; native calls include CPU Blender/operator overhead, not GPU kernel time; budgets unchanged')
    sampler_refinement=dict(maximum_samples_per_grid=32,attempted_samples=0,admitted_changes=0,parts_rebaked=0,initial_missing_samples=0,present_vertex_Q0_attempted=0,initial_present_rejected_samples=0,present_vertex_Q0_FACE_verified=0,meaning='Temporary sampler construction only, not texture filling/recovered detail')
    report['native_POSITION_sampler_refinement']=sampler_refinement
    oriented_budget=dict(max_samples=65536,max_queries_per_sample=32,max_hits_per_sample=65536,max_returned_hits=2_000_000,max_seconds=30.,max_provenance_bytes=16*1024**2,
                         samples=0,proposed=0,native_FACE_verified=0,range_queries=0,returned_hits=0,provenance_bytes=ORIENTED_METADATA_RESERVE,work_seconds=0.,numeric_records=0,compact_parts=0,human_detail_count=0,human_detail_bytes=0)
    report['oriented_SOURCE']=dict(policy='Q0 compatible unchanged; otherwise bounded full-own-HIGH first compatible range window, original component/material/caps. Unperturbed uses authored/planned-base Ns; admitted glTF tangent normal maps use strictly positive geometric orientation only. Native SOURCEFACE after final POSITION mandatory',
        numerical_guard='g0 is unchanged from original unrestricted Q0 distance; native POSITION verifies the explicitly proposed Q1, not Q1-Q0',
        normal_convention='NPZ/part orientation_basis0: unperturbed RNA Ns vs planned-base Ns; basis1: geometric world N only for admitted perturbed source. Effective perturbed NORMAL is evaluated exclusively by the original native shader. Not a physical visibility certificate',
        oracle_scope='BVH API F32 strict-radius candidate set, NOT exact F64 nearest/closed-ball certificate; SOURCEFACE/POSITION/PBR independent calls, exact coincident-hit stability NOT_QUALIFIED',
        query_resource_scope='Full API list materializes before Python length/time checks; source2M cap bounds source domain only, not strict native per-query allocation/time',
        budget=oriented_budget,Q0_compatible_samples=0,parts=[],compact_provenance=oriented_provenance_schema())
    report['construction_statistics_scope']='Initial nominal construction; post-POSITION sampler changes have separate native_POSITION_sampler_refinement counts'
    auxiliary_work=dict(max_rows=AUXILIARY_FALLBACK_ROWS,max_rays=500000,max_seconds=30.,rows=0,rays=0,work_seconds=0.,fast_samples=0,fast_rays=0,seeded_samples=0,unchanged=0,changed=0,refused=0,height_seeds=0,cell_transition_charge=0,interrupted_cell_allowances=0,refusal_reasons={})
    report['oriented_auxiliary_first_hit']=dict(budget=auxiliary_work,scope='Every admitted recovered Q1; three full-own-HIGH F32-order ray checks. Optional present exact-vertex Q0 construction and three queries debit this same fallback row/ray/time budget. Fast Q1 checks excluded from fallback row/ray/time budget. Finite nine incenter Aims,32 dyadic heights each, at most ONE256-cell refinement per fallback row. Exhausted preferred construction may seed this same budgeted search from Q1, without an invented ULP margin. CPU BVH witnesses do not replace native POSITION/SOURCEFACE/PBR.',work_limit_scope='Cooperative checks before candidate/ray work; does not interrupt an executing native BVH call',cell_accounting='Exact returned transitions; interrupted cell calls conservatively charge their ONE256 allowance, not measured transitions',exhaustion='Original sampler retained for unchanged authoritative native gates; not filled/recovered detail')
    report['exact_SOURCE_SELF']=dict(admission=[],parts=0,paint_samples=0,FACE_verified=0,scope='Only live exact protected copies; identity OBJECT/WORLD gauge; unperturbed supported named-source-UV graphs. No nearest/Q1/recovered-pixel claim.')
    exact_q0_report=report['exact_surviving_Q0']=dict(proposed_rows=0,native_FACE_verified=0,max_rows=65536,shared_SOURCEFACE_rows_per_part_limit=65536,
        scope='Unperturbed shading-negative Q0 only: live original part lineage, byte-exact matrix/cyclic positions, unique unoriented own-HIGH position match. Not SELF; POSITION and exact SOURCEFACE mandatory.')
    report['ray_sampler_expected_samples']=len(owners['indices'])
    report['shader_footprint']='Ray parts use isolated microproxies; exact-source SELF uses original source UV with explicit target UV. No general derivative/mipmap/fidelity guarantee.'
    report['construction_refinement']['scope']+='; excludes exact-source SELF'
    hit=np.zeros_like(occupancy);save_current_provenance=None
    try:
        report.update(native_owner_samples=len(owners['indices']),source_sample_set_exactly_native_LOW=True,proxy_sample_set_exactly_native_LOW=True)
        for pi,(part,source_state) in enumerate(zip(parts,sources)):
            selected=np.flatnonzero(owners['parts']==pi);indices=owners['indices'][selected];part['sample_rows']=selected
            if not len(indices):continue
            before=mesh_digest(part['low']);high_before=mesh_digest(part['high']);raw=raw_frames[pi]
            if not np.array_equal(np.asarray(part['low'].matrix_world,np.float64),raw['matrix']):raise ValueError('Final LOW matrix differs from frozen sample frame')
            hv,hf,hloops=arrays(part['high'].data);hm=np.asarray(part['high'].matrix_world,np.float64)
            self_state,self_reason=prepare_source_self(part,raw,source_state,material_doc,hv,hf,hloops,hm)
            report['exact_SOURCE_SELF']['admission'].append(dict(part=pi,selected=self_state is not None,reason=self_reason))
            if self_state is not None:
                bake_sources[pi]=self_state;report['proxy_sample_set_exactly_native_LOW']=False
                record=verify_source_self(part,self_state,target,node,raw,owners,selected,resolution,directory,pi,config)
                if mesh_digest(part['low'])!=before or mesh_digest(part['high'])!=high_before:raise ValueError('Exact-source SELF changed original HIGH/final LOW')
                report['parts'].append(record);hit[indices]=True
                report['exact_SOURCE_SELF']['parts']+=1;report['exact_SOURCE_SELF']['paint_samples']+=len(indices);report['exact_SOURCE_SELF']['FACE_verified']+=len(indices)
                report['ray_sampler_expected_samples']-=len(indices)
                report['reason_counts_complete']=report['reason_counted_samples']==report['ray_sampler_expected_samples'];report['construction_refinement']['complete']=report['reason_counts_complete']
                report.update(hit_texels=int(hit.sum()),pending_texels=int(occupancy.sum())-sum(x['paint_samples'] for x in report['parts']))
                write_json_atomic(directory/'nearest-correspondence.json',report)
                continue
            world=hv.astype(float)@hm[:3,:3].T+hm[:3,3];bvh=BVHTree.FromPolygons(world.tolist(),hf.tolist(),all_triangles=True)
            cache_started=time.monotonic()
            if len(hf)>2_000_000:raise ValueError('Oriented SOURCE face resource cap exceeded')
            high_normals=geometric_world_normals(world,hf,hm)
            low_world=raw['vertices'].astype(float)@raw['matrix'][:3,:3].T+raw['matrix'][:3,3]
            low_normals=geometric_world_normals(low_world,raw['faces'],raw['matrix']);del low_world
            component_ids=original_source_face_ids(part.get('source_component_face_ranges',[]),part['source_component_ids'],len(hf))
            protected_ranges=False
            if len(part['source_component_ids'])>1 and part['protected'] and len(raw['faces'])==len(hf) and raw['matrix'].tobytes()==hm.tobytes():
                protected_ranges=all(raw['vertices'][raw['faces'][k:k+65536]].tobytes()==hv[hf[k:k+65536]].tobytes() for k in range(0,len(hf),65536))
            source,source_bsdf,_,_=source_state
            unperturbed=material_doc.get('normalTexture') is None and not source_bsdf.inputs['Normal'].is_linked
            normal_links=source_bsdf.inputs['Normal'].links
            geometric_recovery_qualified=(material_doc.get('normalTexture') is not None and len(normal_links)==1
                and normal_links[0].from_node.type=='NORMAL_MAP' and normal_links[0].from_node.space=='TANGENT'
                and normal_links[0].from_socket.name=='Normal')
            shading=None;low_shading=None;source_n=None
            if unperturbed:
                source_n=np.empty((len(part['high'].data.loops),3),np.float32)
                part['high'].data.corner_normals.foreach_get('vector',source_n.ravel())
                shading=source_shading_cache(world,hf,hloops,source_n,hm)
                low_shading=planned_base_shading_normals(planned_frames[pi],owners['faces'][selected],owners['weights'][selected])
            source_context=dict(faces=hf,normals=high_normals,components=component_ids,materials=np.full(len(hf),part['material'],np.int32),shading=shading,
                                geometric_recovery_qualified=geometric_recovery_qualified)
            lineage=part.get('_source_lineage')
            exact_lineage=(lineage==(part['high'].as_pointer(),part['low'].as_pointer(),part['object'],part['material'],tuple(part['source_component_ids']),
                tuple(tuple(row)for row in part['source_component_face_ranges'])))
            exact_index=None
            def source_first_hit(origin,direction,cap):
                result=bvh.ray_cast(Vector(origin),Vector(direction),cap)
                if result[2] is not None and not 0<=int(result[2])<len(hf):raise ValueError('Invalid full-own-HIGH first-hit face index')
                return result
            oracle_path=directory/f'nearest-samples-{pi:03d}.npz'
            if oracle_path.exists():raise ValueError('Consumed oriented-source provenance path')
            oracle_record=dict(part=pi,material=part['material'],schema_version=2,changed=0,provenance=str(oracle_path),oriented_array='oriented',detail=[],detail_limit=32,details_truncated=False,
                               human_scope='First32 globally over this sampling grid; numeric rows are complete',
                               normal_cache_seconds=time.monotonic()-cache_started,source_faces=len(hf),
                               original_component_ranges=part['source_component_face_ranges'],acceptance='Q1 and exact-surviving Q0 proposals only until native POSITION and SOURCEFACE verification',
                               classifier_method=('HIGH_unperturbed_RNA_Ns_vs_LOW_plannedBase_Ns' if unperturbed else
                                   'HIGH_geometric_N_vs_LOW_geometric_N; perturbed normal exclusively from native shader' if geometric_recovery_qualified else
                                   'Perturbed SOURCE: compatible legacy geometry path only; recovery NOT_QUALIFIED'),
                               orientation_basis=0 if unperturbed else 1,geometric_recovery_qualified=geometric_recovery_qualified,
                               SOURCE_normalTexture=material_doc.get('normalTexture') is not None,SOURCE_BSDF_Normal_linked=bool(source_bsdf.inputs['Normal'].is_linked),
                               appearance_cache_bytes=shading['additional_bytes']+low_shading.nbytes if unperturbed else 0)
            report['oriented_SOURCE']['parts'].append(oracle_record)
            lower,span,domain=position_domain(world)
            diagonal=max((part['low'].matrix_world@Vector(a)-part['low'].matrix_world@Vector(b)).length for a in part['low'].bound_box for b in part['low'].bound_box)
            extrusion=diagonal*config['cage_extrusion_relative'];ray_cap=diagonal*config['projection_distance_relative'];part['sample_caps']=dict(extrusion=extrusion,ray_cap=ray_cap)
            count=len(indices)
            nearest=np.empty((count,3),np.float64);low_points=np.empty_like(nearest);guards=np.empty(count,np.float64);source_faces=np.empty(count,np.int32);strategies=np.empty(count,np.uint8)
            initial_auxiliary=np.zeros(count,bool)
            oriented_buffer=None;oriented_count=0;constructed_count=0
            exact_q0_rows=np.empty(min(count,exact_q0_report['max_rows']),np.int32);exact_q0_count=0;present_face_row=None
            oracle_record['exact_Q0']=dict(proposed_rows=0,source_lineage_valid=bool(exact_lineage),index_bytes=0,index_seconds=0.,
                sparse_array='exact_Q0_rows',binding='Constructed sample-row IDs into the same NPZ dense indices/nearest/source_faces/guards; not Q1 or texel filling')
            auxiliary_changes=np.empty(AUXILIARY_FALLBACK_ROWS,AUXILIARY_CHANGE_DTYPE);auxiliary_change_count=0
            def remember_auxiliary_change(sample_row,fact,stage):
                nonlocal auxiliary_change_count
                change=fact.get('own_HIGH_first_hit') or {}
                if change.get('status')!='changed':return
                if auxiliary_change_count>=AUXILIARY_FALLBACK_ROWS:raise ValueError('Sparse auxiliary change record budget exceeded')
                auxiliary_changes[auxiliary_change_count]=(sample_row,stage,change['Aim'],change['height'],change['method']);auxiliary_change_count+=1
            oriented_budget['compact_parts']+=1
            oriented_budget['provenance_bytes']=oriented_resource_bytes(oriented_budget['numeric_records'],oriented_budget['compact_parts'],oriented_budget['human_detail_bytes'])+oriented_budget.get('exact_Q0_bytes',0)+oriented_budget.get('present_Q0_bytes',0)
            if oriented_budget['provenance_bytes']>oriented_budget['max_provenance_bytes']:raise ValueError('Compact oriented headers exceed unchanged16MiB')
            def check_compact_metadata():
                descriptors=[{k:v for k,v in item.items() if k!='detail'} for item in report['oriented_SOURCE']['parts']]
                size=len(json.dumps(dict(schema=report['oriented_SOURCE']['compact_provenance'],parts=descriptors),separators=(',',':')).encode())
                if size>ORIENTED_METADATA_RESERVE:raise ValueError('Compact oriented schema/descriptors exceed1MiB reserve')
                oriented_budget['metadata_bytes_actual']=size
            def checkpoint_nearest():
                if oracle_path.exists():return
                sparse=oriented_buffer[:oriented_count] if oriented_buffer is not None else np.empty(0,ORIENTED_RECORD_DTYPE)
                c=constructed_count;temporary=oracle_path.with_suffix('.tmp.npz')
                np.savez_compressed(temporary,indices=indices[:c],faces=owners['faces'][selected[:c]],weights=owners['weights'][selected[:c]],
                    LOW_points=low_points[:c],nearest=nearest[:c],source_faces=source_faces[:c],guards=guards[:c],precision_floor=strategies[:c],oriented=sparse,
                    orientation_basis=np.uint8(0 if unperturbed else 1),exact_Q0_rows=exact_q0_rows[:exact_q0_count][exact_q0_rows[:exact_q0_count]<c],
                    oriented_sampler_changes=auxiliary_changes[:auxiliary_change_count][auxiliary_changes[:auxiliary_change_count]['stage']==0])
                temporary.replace(oracle_path)
                oracle_record.update(provenance_sha256=digest(oracle_path),full_record_count=oriented_count,dense_prefix_samples=c,sampler_construction_complete=c==count,initial_sampler_changes=int((auxiliary_changes[:auxiliary_change_count]['stage']==0).sum()))
                check_compact_metadata()
            save_current_provenance=checkpoint_nearest
            detail=[];maximum_error=0.;floor_count=0;reason_counts={};auxiliary_count=0;proposal_rejections=dict(count=0,reasons={},details=[],detail_limit=32);margin_candidates=dict(attempted=0,selected=0,rejected=0,details=[],detail_limit=32)
            construction_started=time.monotonic()
            low_triangles=low_triangle_cache(raw)
            oracle_record['LOW_triangle_cache']=dict(enabled=low_triangles is not None,faces=len(raw['faces']),
                accounted_bytes=low_triangles['accounted_bytes'] if low_triangles is not None else 0,cap_bytes=LOW_TRIANGLE_CACHE_BYTES,
                semantics='Exact old per-face F64 expressions; no barycentric/world transform reassociation; current frozen part/grid only')
            def classify_Q0(j,row):
                face=int(owners['faces'][row]);bary=owners['weights'][row]
                point=(sample_world(raw,face,bary) if low_triangles is None else
                       sample_world(raw,face,bary,local_triangle=low_triangles['local'][face]))
                where,normal,source_face,distance=bvh.find_nearest(Vector(point))
                if where is None:raise ValueError(f'No nearest own HIGH part={pi} pixel={int(indices[j])}')
                guard=max(float(distance)*.01,16*float(np.finfo(np.float32).eps)*domain['world_scale'])
                try:
                    reference=low_shading[j] if unperturbed else low_normals[face]
                    compatible=(float(source_shading_normal(shading,int(source_face),where)[0]@reference)>0 if unperturbed
                                else float(high_normals[int(source_face)]@low_normals[face])>0)
                except ValueError as error:
                    report['oriented_SOURCE']['unsupported_sample']=dict(part=pi,pixel=int(indices[j]),LOW_face=face,source_face=int(source_face),reason=str(error))
                    write_json_atomic(directory/'nearest-correspondence.json',report)
                    raise CandidateRejected('oriented_correspondence',directory/'nearest-correspondence.json',dict(error=str(error))) from error
                return point,where,normal,source_face,distance,guard,compatible
            classification_cache,prepass=_sourceface_row_prepass(selected,nearest,low_points,guards,source_faces,classify_Q0,
                eligible=unperturbed and len(part['source_component_ids'])==1,native_f32_normals=True)
            oracle_record['SOURCEFACE_row_prepass']=prepass
            if prepass['shared_row_limit_exceeded']:
                # No current-part proxy or numerical provenance has been constructed.
                # Do not checkpoint uninitialized dense buffers as if they were Q rows.
                save_current_provenance=None
                oracle_record.update(provenance=None,full_record_count=0,dense_prefix_samples=0,sampler_construction_complete=False)
                report['proxy_sample_set_exactly_native_LOW']=False
                row=selected[prepass['classified_samples']-1]
                report['oriented_SOURCE']['unsupported_sample']=dict(part=pi,pixel=int(indices[prepass['classified_samples']-1]),
                    LOW_face=int(owners['faces'][row]),reason='native_SOURCEFACE_shared_row_budget',
                    minimum_shared_rows=prepass['incompatible_rows'],shared_part_limit=65536,constructed_samples=0,Q1_rows=0,exact_Q0_rows=0)
                report['timing']['nearest_proxy_construction_seconds']+=time.monotonic()-construction_started
                raise CandidateRejected('oriented_correspondence',directory/'nearest-correspondence.json',dict(error='native_SOURCEFACE_shared_row_budget'))
            vertices=np.empty((count*3,3),np.float32);cages=vertices.copy();uv=np.empty((count*3,2),np.float32)
            paint_batch=[];paint_batch_start=0;paint_sample=None
            for j,row in enumerate(selected):
                face=int(owners['faces'][row]);bary=owners['weights'][row]
                if classification_cache is None:
                    point,where,normal,source_face,distance,guard,compatible=classify_Q0(j,row)
                else:
                    point=low_points[j];where=Vector(nearest[j]);normal=Vector(classification_cache['normal'][j])
                    source_face=int(source_faces[j]);distance=float(classification_cache['distance'][j]);guard=float(guards[j]);compatible=bool(classification_cache['compatible'][j])
                tri=(raw['vertices'][raw['faces'][face]].astype(float)@raw['matrix'][:3,:3].T+raw['matrix'][:3,3]
                     if low_triangles is None else low_triangles['world'][face])
                original_distance=None;selection=None
                reference=low_shading[j] if unperturbed else low_normals[face]
                if not compatible:
                    try:
                        if oriented_count+exact_q0_count>=65536:
                            raise OrientedSourceRejected('native_SOURCEFACE_shared_row_budget',dict(Q1_rows=oriented_count,exact_Q0_rows=exact_q0_count,shared_part_limit=65536))
                        if len(part['source_component_ids'])==1:expected_component=int(part['source_component_ids'][0])
                        elif protected_ranges:expected_component=int(component_ids[face])
                        else:raise OrientedSourceRejected('unproved_original_LOW_component',{})
                        exact_admitted=False
                        if unperturbed and exact_lineage and raw['matrix'].tobytes()==hm.tobytes() and exact_q0_report['proposed_rows']<exact_q0_report['max_rows']:
                            if exact_index is None:
                                tick=time.monotonic();exact_index=exact_face_position_index(hv,hf)
                                oracle_record['exact_Q0'].update(index_bytes=exact_index.nbytes,index_seconds=time.monotonic()-tick)
                            charged=oriented_resource_bytes(oriented_budget['numeric_records'],oriented_budget['compact_parts'],oriented_budget['human_detail_bytes'])+oriented_budget.get('exact_Q0_bytes',0)+oriented_budget.get('present_Q0_bytes',0)+4
                            exact_admitted=(charged<=oriented_budget['max_provenance_bytes'] and exact_surviving_source_face(exact_index,hv,hf,raw['vertices'][raw['faces'][face]],
                                raw['matrix'],hm,int(source_face),expected_component,part['material'],component_ids,source_context['materials']))
                            if exact_admitted:
                                exact_q0_rows[exact_q0_count]=j;exact_q0_count+=1;exact_q0_report['proposed_rows']+=1
                                oracle_record['exact_Q0']['proposed_rows']+=1;oriented_budget['exact_Q0_bytes']=oriented_budget.get('exact_Q0_bytes',0)+4;oriented_budget['provenance_bytes']=charged
                        if not exact_admitted:
                            original_distance=float(distance)
                            where,normal,source_face,selection=oriented_source(point,reference,where,normal,int(source_face),original_distance,guard,ray_cap,
                                expected_component,part['material'],source_context,
                                lambda p,d:bvh.find_nearest_range(Vector(p),d),oriented_budget,low_geometric_normal=low_normals[face])
                            selection.update(pixel=int(indices[j]),LOW_face=face,LOW_point=point.tolist(),g0=guard)
                    except OrientedSourceRejected as error:
                        report['oriented_SOURCE']['unsupported_sample']=dict(part=pi,pixel=int(indices[j]),LOW_face=face,**error.facts)
                        rejected=oriented_construction_rejection(error,report,directory,construction_started,pi,resolution)
                        write_json_atomic(directory/'nearest-correspondence.json',report)
                        raise rejected from error
                else:report['oriented_SOURCE']['Q0_compatible_samples']+=1
                extra={} if original_distance is None else dict(original_nearest_distance=original_distance,first_hit=source_first_hit,expected_source_face=int(source_face),auxiliary_budget=auxiliary_work,allow_q1_seed=True)
                if low_triangles is not None:extra['low_basis']=low_triangles['basis'][face]
                # Initial Q0 only; Q1, auxiliaries and POSITION retries stay scalar.
                paint_sample=None
                if original_distance is None:
                    if not paint_batch or not paint_batch_start<=j<paint_batch_start+len(paint_batch):
                        # Release the previous block before allocating its replacement.
                        paint_batch=None;paint_sample=None;paint_batch_start=j
                        paint_batch=_proxy_paint_batch(indices[j:j+2048],resolution)
                    paint_sample=paint_batch[j-paint_batch_start]
                try:v,c,u,fact=proxy_triangle(point,np.asarray(where),np.asarray(normal),tri,int(indices[j]),resolution,extrusion,ray_cap,guard,zero_distance=distance==0.,source_triangle=world[hf[int(source_face)]],_paint_sample=paint_sample,**extra)
                except ValueError as exc:
                    refinement=getattr(exc,'refinement',None);record_floor_refinement(report,refinement,part=pi,pixel=indices[j],face=face)
                    report['unsupported_sample']=dict(floor_refinement=refinement,part=pi,pixel=int(indices[j]),LOW_face=face,barycentric=bary.tolist(),LOW_point=point.tolist(),nearest=list(where),source_face=int(source_face),nearest_distance=float(np.linalg.norm(np.asarray(where)-point)),original_nearest_distance=float(distance),oriented_selection=selection,source_geometric_normal=list(normal),original_guard=guard,extrusion=extrusion,ray_cap=ray_cap,reason=str(exc))
                    raise
                if paint_sample is not None:report['paint_raster_batch_admitted_rows']+=1
                remember_auxiliary_change(j,fact,0)
                vertices[j*3:j*3+3]=v;cages[j*3:j*3+3]=c;uv[j*3:j*3+3]=u;nearest[j]=where;low_points[j]=point;guards[j]=guard;source_faces[j]=source_face
                strategies[j]=fact['strategy'].endswith('precision_floor');floor_count+=int(strategies[j]);maximum_error=max(maximum_error,fact['maximum_center_error']);constructed_count=j+1
                if selection is not None:
                    selection['sampler']=dict(strategy=fact['strategy'],preferred_auxiliary=fact.get('oriented_preferred_auxiliary'),
                        center_reference=fact['center_reference'],empty_ball_claim=fact['empty_ball_claim'],texel_fill=False)
                    human_bytes=0;human_detail=selection
                    if oriented_budget['human_detail_count']<32:
                        human_bytes=len(json.dumps(selection,separators=(',',':')).encode())
                        if human_bytes>8192:
                            human_detail=dict(omitted=True,reason='Optional human detail exceeds8KiB; complete numeric record retained',original_size_bytes=human_bytes,
                                numeric_reference=dict(schema_version=2,array='oriented',oriented_row=oriented_count,sample_row=j,owner_row=int(row)))
                            human_bytes=len(json.dumps(human_detail,separators=(',',':')).encode())
                    charged=oriented_resource_bytes(oriented_budget['numeric_records']+1,oriented_budget['compact_parts'],oriented_budget['human_detail_bytes']+human_bytes)+oriented_budget.get('exact_Q0_bytes',0)+oriented_budget.get('present_Q0_bytes',0)
                    if charged>oriented_budget['max_provenance_bytes']:raise ValueError('Compact oriented numeric provenance exceeds unchanged16MiB')
                    if oriented_buffer is None:oriented_buffer=np.empty(65536,ORIENTED_RECORD_DTYPE)
                    oriented_buffer[oriented_count]=oriented_numeric_record(selection,j,int(row),fact);oriented_count+=1
                    oriented_budget['numeric_records']+=1;oriented_budget['provenance_bytes']=charged
                    oracle_record['changed']=oriented_count;oracle_record['full_record_count']=oriented_count
                    if human_bytes:
                        oracle_record['detail'].append(human_detail);oriented_budget['human_detail_count']+=1;oriented_budget['human_detail_bytes']+=human_bytes
                    oracle_record['details_truncated']=oriented_count>len(oracle_record['detail'])
                record_floor_refinement(report,fact['floor_refinement'],part=pi,pixel=indices[j],face=face)
                initial_auxiliary[j]=fact['auxiliary_sampler'] is not None;auxiliary_count+=int(initial_auxiliary[j])
                margins=fact['auxiliary_margin_candidates']
                if margins is not None:
                    for key in ('attempted','selected','rejected'):margin_candidates[key]+=margins[key]
                    if len(margin_candidates['details'])<32:margin_candidates['details'].append(dict(pixel=int(indices[j]),**margins))
                rejected=fact['auxiliary_proposal_rejected']
                if rejected is not None:
                    proposal_rejections['count']+=1;reason=rejected['reason'];proposal_rejections['reasons'][reason]=proposal_rejections['reasons'].get(reason,0)+1
                    if len(proposal_rejections['details'])<32:proposal_rejections['details'].append(dict(pixel=int(indices[j]),**rejected))
                precision_reason=fact['precision_floor_reason'];report['reason_counted_samples']+=1
                if precision_reason is not None:
                    reason_counts[precision_reason]=reason_counts.get(precision_reason,0)+1
                    report['precision_floor_reason_counts'][precision_reason]=report['precision_floor_reason_counts'].get(precision_reason,0)+1
                if len(detail)<32:detail.append(dict(pixel=int(indices[j]),LOW_face=face,nearest_source_face=int(source_face),guard=guard,**fact))
            faces=np.arange(count*3,dtype=np.int32).reshape(-1,3)
            proxy=make_object('Nearest sample proxies '+str(pi),vertices,faces,Matrix.Identity(4))
            part['sample_proxy']=proxy;proxy.data.uv_layers.new(name='NativePaintProxy');proxy.data.uv_layers.active.data.foreach_set('uv',uv.ravel());proxy.data.materials.append(target)
            cage=make_object('Nearest sample cages '+str(pi),cages,faces,Matrix.Identity(4));part['sample_cage']=cage;cage.select_set(False)
            pv,pf,_=arrays(proxy.data);cv,cf,_=arrays(cage.data)
            if not np.array_equal(pv,vertices) or not np.array_equal(pf,faces) or not np.array_equal(cv,cages) or not np.array_equal(cf,faces) or not np.array_equal(uv_loop_data(proxy.data),uv):raise ValueError('Native proxy/cage storage changed')
            checkpoint_nearest()
            del low_triangles,tri,extra,classification_cache
            report['timing']['nearest_proxy_construction_seconds']+=time.monotonic()-construction_started
            sparse=oriented_buffer[:oriented_count] if oriented_buffer is not None else np.empty(0,ORIENTED_RECORD_DTYPE)
            recovered_rows=sparse['sample_row'].astype(np.int32)
            final_oriented_strategy=sparse['initial_strategy'].copy();final_oriented_margin=sparse['initial_margin_ULPs'].copy()
            def capture_oriented_POSITION():
                if not len(sparse):return
                path=directory/f'nearest-samples-{pi:03d}-oriented-POSITION.npz';temporary=path.with_suffix('.tmp.npz')
                retained=oriented_position_records(sparse,data,indices,positions,errors,low_to_nearest,low_to_native_hit,accepted,final_oriented_strategy,final_oriented_margin)
                np.savez_compressed(temporary,oriented_POSITION=retained,oriented_sampler_changes=auxiliary_changes[:auxiliary_change_count][auxiliary_changes[:auxiliary_change_count]['stage']==1]);temporary.replace(path)
                oracle_record['POSITION']=dict(path=str(path),sha256=digest(path),array='oriented_POSITION',full_record_count=len(retained),native_pass=record['native_POSITION_passes'],
                    meaning='Actual final POSITION readback before FACE; encoded/accepted flags distinguish refused data from physical hits',POSITION_retry_sampler_changes=int((auxiliary_changes[:auxiliary_change_count]['stage']==1).sum()))
                check_compact_metadata()

            record=dict(auxiliary_margin_candidates=margin_candidates,auxiliary_proposals_rejected=proposal_rejections,auxiliary_source_sampler_count=auxiliary_count,part=pi,paint_samples=count,proxy_faces=count,precision_floor_samples=floor_count,precision_floor_reason_counts=reason_counts,maximum_proxy_error_to_declared_center=maximum_error,
                        domain=domain,encoding_lower=lower.tolist(),encoding_span=span.tolist(),caps=part['sample_caps'],details=detail,detail_limit=32,
                        source_component_domain=part['source_component_domain'],source_component_ids=part['source_component_ids'],source_face_index_space='Own full HIGH part-local triangles; no source filtering',
                        nearest_array_semantics='Selected Q0-compatible or explicitly oriented Q1 target; complete Q0/d0/face/query/Ns provenance is the numeric oriented array, not first32 human details; guards remain original g0',
                        oriented_source_selection=dict(schema_version=2,report_part=pi,reference='oriented_SOURCE.parts',provenance=str(oracle_path)))
            if sum(reason_counts.values())!=floor_count:raise ValueError('Precision-floor reason counts differ from constructed samples')
            report['reason_counts_complete']=report['reason_counted_samples']==report['ray_sampler_expected_samples'];report['construction_refinement']['complete']=report['reason_counts_complete']
            record.update(sampler_statistics_scope='Initial nominal construction; native POSITION sampler changes are reported separately',final_sampler_counts=dict(auxiliary_source_sampler_count=auxiliary_count,precision_floor_samples=floor_count),native_POSITION_passes=1)
            report['parts'].append(record);write_json_atomic(directory/'nearest-correspondence.json',report)
            proof_started=time.monotonic()
            source,bsdf,output,proxies=source_state;position_proxy(source,output,lower,span)
            image=new_map('Native nearest POSITION '+str(pi),resolution,'Non-Color',float_buffer=True)
            try:
                bake_sample_proxy(part,target,node,image);data=pixels(image);rgb=data[indices,:3].astype(float)
                nonclear=np.flatnonzero(np.any(data[:,:3]!=0,axis=1));encoded_all=np.flatnonzero(np.all((data[:,:3]>=.25-1e-6)&(data[:,:3]<=.75+1e-6),axis=1))
                native_set_exact=np.array_equal(nonclear,np.sort(indices)) and np.array_equal(encoded_all,np.sort(indices))
                positions=lower+(rgb-.25)*2*span;errors=np.linalg.norm(positions-nearest,axis=1)
                low_to_nearest=np.linalg.norm(nearest-low_points,axis=1);low_to_native_hit=np.linalg.norm(positions-low_points,axis=1)
                accepted=(low_to_nearest<=ray_cap)&np.isfinite(low_to_native_hit)&(low_to_native_hit<=ray_cap)&np.isfinite(errors)&(errors<=guards)&np.all((rgb>=.25-1e-6)&(rgb<=.75+1e-6),axis=1)
                present_row=None
                if native_set_exact and not np.all(accepted):
                    present_row,present_eligibility=present_vertex_q0_row(native_set_exact=native_set_exact,accepted=accepted,errors=errors,guards=guards,
                        low_to_nearest=low_to_nearest,low_to_native_hit=low_to_native_hit,ray_cap=ray_cap,strategies=strategies,initial_auxiliary=initial_auxiliary,
                        recovered_rows=recovered_rows,exact_q0_rows=exact_q0_rows[:exact_q0_count],nearest=nearest,source_faces=source_faces,world=world,high_faces=hf,
                        attempted_samples=sampler_refinement['attempted_samples'],present_attempted=sampler_refinement['present_vertex_Q0_attempted'],
                        proof_rows=oriented_count+exact_q0_count,proof_bytes=oriented_budget['provenance_bytes'],proof_limit=oriented_budget['max_provenance_bytes'])
                    record['present_POSITION_rejection']=dict(samples=int((~accepted).sum()),eligibility=present_eligibility,selected_row=present_row)
                    sampler_refinement['initial_present_rejected_samples']+=int((~accepted).sum())
                if not native_set_exact or present_row is not None:
                    if not native_set_exact:
                        expected_mask=np.zeros_like(occupancy);expected_mask[indices]=True
                        nonclear_mask=np.zeros_like(occupancy);nonclear_mask[nonclear]=True
                        encoded_mask=np.zeros_like(occupancy);encoded_mask[encoded_all]=True
                        mismatch_mask=(expected_mask!=nonclear_mask)|(expected_mask!=encoded_mask)
                        record['native_POSITION_set_masks']={}
                        for name,mask in [('expected',expected_mask),('nonclear',nonclear_mask),('encoded',encoded_mask),('mismatch',mismatch_mask)]:
                            path=directory/f'position-{pi:03d}-{name}-mask.npy';np.save(path,mask.reshape(resolution,resolution));record['native_POSITION_set_masks'][name]=str(path)
                        record['native_POSITION_set_counts']=dict(expected=count,nonclear=len(nonclear),encoded=len(encoded_all),mismatch=int(mismatch_mask.sum()))
                        record['POSITION_mask_semantics']='Nonclear POSITION combines target raster and source-ray hit; not an independent LOW occupancy measurement'
                        record['initial_native_POSITION_set_counts']=dict(record['native_POSITION_set_counts'])
                        record['native_POSITION_set_masks_scope']='Initial failed pass, preserved unchanged after any sampler refinement'
                    else:
                        record['initial_native_POSITION_set_counts']=dict(expected=count,nonclear=len(nonclear),encoded=len(encoded_all),mismatch=0)
                    initial_path=directory/f'position-{pi:03d}-initial-rgba.npy';np.save(initial_path,data)
                    record['initial_native_POSITION_rgba']=str(initial_path)
                    if present_row is None:
                        rows,eligibility=position_refinement_rows(indices,nonclear_mask,encoded_mask,accepted,sampler_refinement['attempted_samples'])
                        sampler_refinement['initial_missing_samples']+=len(rows)
                    else:
                        rows=np.array([present_row],np.int32);eligibility=present_eligibility
                        oriented_budget['present_Q0_bytes']=oriented_budget.get('present_Q0_bytes',0)+4;oriented_budget['provenance_bytes']+=4
                        oracle_record['present_vertex_Q0']=dict(sample_row=present_row,pixel=int(indices[present_row]),source_face=int(source_faces[present_row]),
                            initial_Q_error=float(errors[present_row]),original_guard=float(guards[present_row]),binding='Same immutable dense NPZ row; separate post-POSITION sampler archive; native SOURCEFACE mandatory, never SELF')
                        check_compact_metadata()
                    refinement=dict(status='refused',eligibility=eligibility,case='missing' if present_row is None else 'present_vertex_Q0',initial_missing_samples=len(rows) if present_row is None else 0,initial_present_rejected_samples=0 if present_row is None else 1,attempted_samples=0,proposed_changes=0,admitted_changes=0,extra_POSITION_passes=0,details=[])
                    record['native_POSITION_sampler_refinement']=refinement;changes=[]
                    if eligibility in ('eligible','eligible_present_vertex_Q0'):
                        for j in rows:
                            sampler_refinement['attempted_samples']+=1;refinement['attempted_samples']+=1
                            face=int(owners['faces'][selected[j]])
                            triangle=raw['vertices'][raw['faces'][face]].astype(float)@raw['matrix'][:3,:3].T+raw['matrix'][:3,3]
                            try:
                                slot=int(np.searchsorted(recovered_rows,j))
                                if present_row is not None:
                                    sampler_refinement['present_vertex_Q0_attempted']+=1
                                    V,C,U,fact=present_vertex_q0_auxiliary(low_points[j],nearest[j],world[hf[int(source_faces[j])]],triangle,int(indices[j]),resolution,extrusion,ray_cap,float(guards[j]),source_first_hit,int(source_faces[j]),auxiliary_work)
                                elif slot<len(recovered_rows) and recovered_rows[slot]==j:
                                    V,C,U,fact=proxy_triangle(low_points[j],nearest[j],high_normals[int(source_faces[j])],triangle,int(indices[j]),resolution,extrusion,ray_cap,float(guards[j]),
                                        source_triangle=world[hf[int(source_faces[j])]],original_nearest_distance=float(sparse[slot]['d0']),first_hit=source_first_hit,
                                        expected_source_face=int(source_faces[j]),auxiliary_budget=auxiliary_work,allow_q1_seed=True)
                                    if (fact.get('own_HIGH_first_hit') or {}).get('status') not in ('changed','unchanged_admitted'):
                                        raise _AuxiliaryProposalRejected('first_hit_not_admitted','POSITION retry cannot replace its sampler with unchecked auxiliary geometry')
                                else:V,C,U,fact=_auxiliary_source_sampler(low_points[j],nearest[j],world[hf[int(source_faces[j])]],triangle,int(indices[j]),resolution,extrusion,ray_cap,float(guards[j]),strongest=True)
                            except _AuxiliaryProposalRejected as error:
                                refinement['refusal']=dict(pixel=int(indices[j]),proposal=error.proposal);break
                            first=int(j)*3;last=first+3
                            refusal=position_sampler_change_reason(vertices[first:last],cages[first:last],uv[first:last],V,C,U)
                            if refusal:
                                refinement['refusal']=dict(pixel=int(indices[j]),reason=refusal);break
                            changes.append((int(j),V,C,fact))
                            refinement['details'].append(dict(pixel=int(indices[j]),LOW_face=face,nearest_source_face=int(source_faces[j]),original_guard=float(guards[j]),
                                initial_sampler='auxiliary' if initial_auxiliary[j] else 'precision_floor' if strategies[j] else 'DIRECT',candidate=fact))
                        refinement['proposed_changes']=len(changes)
                        if len(changes)==len(rows):
                            archive=directory/f'position-{pi:03d}-sampler-changes.npz'
                            np.savez(archive,**({'present_vertex_Q0_rows':rows} if present_row is not None else {}),indices=indices[rows],baseline_vertices=np.array([vertices[j*3:j*3+3] for j,_,_,_ in changes]),
                                     baseline_cages=np.array([cages[j*3:j*3+3] for j,_,_,_ in changes]),vertices=np.array([V for _,V,_,_ in changes]),cages=np.array([C for _,_,C,_ in changes]),uv=uv.reshape(-1,3,2)[rows])
                            if present_row is not None:present_face_row=present_row
                            for j,V,C,_ in changes:
                                for vertex,coordinate in zip(pf[j],V):proxy.data.vertices[int(vertex)].co=coordinate
                                for vertex,coordinate in zip(cf[j],C):cage.data.vertices[int(vertex)].co=coordinate
                            proxy.data.update();cage.data.update();bpy.context.view_layer.update()
                            if uv_loop_data(proxy.data).tobytes()!=uv.tobytes():raise ValueError('Sampler refinement changed immutable proxy UV')
                            for j,V,C,_ in changes:
                                if (np.array([proxy.data.vertices[int(k)].co for k in pf[j]],np.float32).tobytes()!=V.tobytes()
                                    or np.array([cage.data.vertices[int(k)].co for k in cf[j]],np.float32).tobytes()!=C.tobytes()):raise ValueError('Sampler refinement storage changed proposed geometry')
                            for changed_row,_,_,changed_fact in changes:
                                remember_auxiliary_change(changed_row,changed_fact,1)
                                slot=int(np.searchsorted(recovered_rows,changed_row))
                                if slot<len(recovered_rows) and recovered_rows[slot]==changed_row:
                                    final_oriented_strategy[slot]=ORIENTED_STRATEGIES.index(changed_fact['strategy'])
                                    final_oriented_margin[slot]=(changed_fact.get('auxiliary_margin_candidates') or {}).get('selected_margin_ULPs') or 0
                            record['final_sampler_counts']=dict(auxiliary_source_sampler_count=auxiliary_count+int((~initial_auxiliary[rows]).sum()),precision_floor_samples=floor_count-int(strategies[rows].sum()))
                            strategies[rows]=0
                            refinement.update(status='constructed',admitted_changes=len(changes),geometry_archive=str(archive),extra_POSITION_passes=1)
                            sampler_refinement['admitted_changes']+=len(changes);sampler_refinement['parts_rebaked']+=1;record['native_POSITION_passes']=2
                            write_json_atomic(directory/'nearest-correspondence.json',report)
                            bake_sample_proxy(part,target,node,image);data=pixels(image);rgb=data[indices,:3].astype(float)
                            refined_path=directory/f'position-{pi:03d}-refined-rgba.npy';np.save(refined_path,data);refinement['native_POSITION_rgba']=str(refined_path)
                            nonclear=np.flatnonzero(np.any(data[:,:3]!=0,axis=1));encoded_all=np.flatnonzero(np.all((data[:,:3]>=.25-1e-6)&(data[:,:3]<=.75+1e-6),axis=1))
                            native_set_exact=np.array_equal(nonclear,np.sort(indices)) and np.array_equal(encoded_all,np.sort(indices))
                            record['native_POSITION_set_counts']=dict(expected=count,nonclear=len(nonclear),encoded=len(encoded_all),mismatch=0 if native_set_exact else int(np.union1d(np.setxor1d(indices,nonclear,assume_unique=True),np.setxor1d(indices,encoded_all,assume_unique=True)).size))
                            positions=lower+(rgb-.25)*2*span;errors=np.linalg.norm(positions-nearest,axis=1)
                            low_to_nearest=np.linalg.norm(nearest-low_points,axis=1);low_to_native_hit=np.linalg.norm(positions-low_points,axis=1)
                            accepted=(low_to_nearest<=ray_cap)&np.isfinite(low_to_native_hit)&(low_to_native_hit<=ray_cap)&np.isfinite(errors)&(errors<=guards)&np.all((rgb>=.25-1e-6)&(rgb<=.75+1e-6),axis=1)
                            refinement.update(status='native_verified' if native_set_exact and np.all(accepted) else 'native_refused',final_sample_set_exact=bool(native_set_exact),
                                final_counts=dict(expected=count,nonclear=len(nonclear),encoded=len(encoded_all)),final_original_guard_failures=int((~accepted).sum()),maximum_native_Q_error=float(errors.max()),
                                refined_sample_checks=[dict(pixel=int(indices[j]),native_position=positions[j].tolist(),selected_Q=nearest[j].tolist(),Q_error=float(errors[j]),original_guard=float(guards[j]),LOW_to_native_hit=float(low_to_native_hit[j]),original_ray_cap=ray_cap) for j in rows])
                    write_json_atomic(directory/'nearest-correspondence.json',report)
                    if not native_set_exact or not np.all(accepted):
                        capture_oriented_POSITION()
                        error='Native proxy POSITION raster differs from own sample set' if not native_set_exact else 'Native proxy POSITION failed original Q/LOW guards'
                        raise ValueError(f'{error}: part={pi} expected={count} nonclear={len(nonclear)} encoded={len(encoded_all)}; refinement={refinement["status"]}')
                capture_oriented_POSITION()
                face_rows=np.union1d(recovered_rows,exact_q0_rows[:exact_q0_count]).astype(np.int32)
                if present_face_row is not None:face_rows=np.union1d(face_rows,np.array([present_face_row],np.int32))
                if len(face_rows) and native_set_exact and np.all(accepted):
                    self_FACE_count=0
                    record['native_SOURCEFACE']=dict(status='pending',Q1_rows=len(recovered_rows),exact_Q0_rows=exact_q0_count,present_vertex_Q0_rows=int(present_face_row is not None))
                    write_json_atomic(directory/'nearest-correspondence.json',report)
                    try:native_source_face_witness(part,source_state,target,node,indices,face_rows,source_faces,component_ids,resolution,directory,record['native_SOURCEFACE'],pi)
                    except SourceFaceMismatch as mismatch:
                        if present_face_row is not None and present_face_row in mismatch.rows:
                            raise SourceFaceMismatch('Native SOURCEFACE differs from present exact-vertex Q0; SELF is forbidden for this row',mismatch.rows,mismatch.observed,mismatch.proposed) from mismatch
                        try:
                            sparse_part,sparse_state=prepare_q1_face_self(part,source_state,material_doc,mismatch,
                                dict(recovered_rows=recovered_rows,indices=indices,nearest=nearest,guards=guards,LOW_points=low_points,ray_cap=ray_cap,LOW_Ns=low_shading,LOW_matrix=raw['matrix']),
                                source_context,config,directory,pi)
                        except Q1SelfGeometricIneligible as error:
                            if config['operation']!='optimize-rebake' or part['protected']:raise
                            record['native_SOURCEFACE']['SELF_geometric_rejection']=dict(error=str(error),rows=mismatch.rows.tolist(),
                                observed=mismatch.observed.tolist(),proposed=mismatch.proposed.tolist())
                            # Existing failure cleanup persists this report before the one alternate LOW.
                            raise CandidateRejected('q1_self_geometry',directory/'nearest-correspondence.json',dict(error=str(error))) from error
                        part['q1_face_self']=sparse_part;bake_sources.append(sparse_state)
                        self_FACE_count=len(mismatch.rows)
                        report['explicit_Q1_FACE_SELF_samples']=report.get('explicit_Q1_FACE_SELF_samples',0)+len(mismatch.rows)
                        record['native_SOURCEFACE']['explicit_sparse_SELF']=dict(samples=len(mismatch.rows),proof=str(sparse_part['proof_path']),
                            scope='Original ray FACE refusal retained; only these rows use separately proven native SELF POSITION/FACE/shader, not ray equivalence')
                    finally:record['native_SOURCEFACE']['recovered_rows']=len(recovered_rows)
                    oriented_budget['native_FACE_verified']+=len(recovered_rows)-self_FACE_count
                    oriented_budget['native_SELF_FACE_verified']=oriented_budget.get('native_SELF_FACE_verified',0)+self_FACE_count
                    oriented_budget['native_FACE_verified_total']=oriented_budget.get('native_FACE_verified_total',0)+len(recovered_rows)
                    exact_q0_report['native_FACE_verified']+=exact_q0_count
                    if present_face_row is not None:
                        sampler_refinement['present_vertex_Q0_FACE_verified']+=1
                        record['native_SOURCEFACE']['present_vertex_Q0_verified']=1
                hit[indices]=accepted;record.update(native_sample_set_exact=True,accepted=int(accepted.sum()),failed=int((~accepted).sum()),maximum_native_position_error=float(errors.max()),
                    maximum_LOW_to_nearest_distance=float(low_to_nearest.max()),maximum_LOW_to_native_hit_distance=float(low_to_native_hit.max()),original_LOW_ray_cap=ray_cap,
                    failure_details=[dict(pixel=int(indices[j]),LOW_face=int(owners['faces'][selected[j]]),nearest=nearest[j].tolist(),native_position=positions[j].tolist(),error=float(errors[j]),original_guard=float(guards[j]),LOW_point=low_points[j].tolist(),LOW_to_nearest_distance=float(low_to_nearest[j]),LOW_to_native_hit_distance=float(low_to_native_hit[j]),original_ray_cap=ray_cap,precision_floor=bool(strategies[j])) for j in np.flatnonzero(~accepted)[:32]])
            finally:bpy.data.images.remove(image)
            if mesh_digest(part['low'])!=before or mesh_digest(part['high'])!=high_before or not np.array_equal(np.asarray(part['high'].matrix_world,np.float64),hm) or not np.array_equal(np.asarray(part['low'].matrix_world,np.float64),raw['matrix']):raise ValueError('Nearest proxies modified HIGH/final LOW attributes or matrices')
            del bvh,vertices,cages,uv,nearest,guards,source_faces,strategies,data,world,low_points,low_to_nearest,low_to_native_hit,initial_auxiliary,high_normals,low_normals,component_ids,source_context,shading,low_shading,source_n,oriented_buffer,sparse,recovered_rows,final_oriented_strategy,final_oriented_margin,auxiliary_changes
            save_current_provenance=None
            report.update(hit_texels=int(hit.sum()),miss_texels=sum(r.get('failed',0) for r in report['parts']),pending_texels=int(occupancy.sum())-sum(r.get('paint_samples',0) for r in report['parts']))
            write_json_atomic(directory/'nearest-correspondence.json',report)
            if record['failed']:raise CandidateRejected('nearest_correspondence',directory/'nearest-correspondence.json',dict(error=f"Nearest own-HIGH correspondence failed for {record['failed']} paint samples; original position guards unchanged"))
            report['timing']['native_POSITION_FACE_and_validation_seconds']+=time.monotonic()-proof_started
            note('nearest_projection_part_verified',part=pi,texels=count,precision_floor_samples=floor_count)
        if not np.array_equal(hit,occupancy):raise ValueError('Nearest correspondence does not exactly cover independent native LOW occupancy')
        np.save(directory/'nearest-hit-mask.npy',hit.reshape(resolution,resolution));hit.setflags(write=False)
        report['masks']['nearest_hit']=str(directory/'nearest-hit-mask.npy');report['hit_texels']=int(hit.sum());report['miss_texels']=0;report['pending_texels']=0;report['complete']=True
        report['timing']['total_correspondence_seconds']=time.monotonic()-correspondence_started
        write_json_atomic(directory/'nearest-correspondence.json',report);write_json_atomic(directory/'coverage.json',report)
    except Exception as exc:
        if save_current_provenance is not None:save_current_provenance()
        report['failure']=getattr(exc,'correspondence_failure',str(exc));write_json_atomic(directory/'nearest-correspondence.json',report)
        for part in parts:
            for key in ('sample_proxy','sample_cage'):
                if key in part:remove(part.pop(key))
            remove_source_self(part)
            remove_q1_face_self(part)
        raise
    def bake_all(image, *, normal=False, margin=0):
        if margin!=0:raise ValueError('Nearest native shader bakes require margin0')
        active=[p for p in parts if p.get('sample_proxy') or p.get('source_self')]
        for i,part in enumerate(active):
            bake_source_part(part,target,node,image,normal=normal,clear=i==0)
            bake_q1_face_overlay(part,target,node,image,normal=normal)
        return len(active)
    return target,node,bake_sources,bake_all,report,owners


def normal_png8_readback(path,native_rgba,owners,target_frames):
    header=Path(path).read_bytes()[:26]
    if header[:8]!=b'\x89PNG\r\n\x1a\n' or header[24]!=8:raise ValueError('Target normal map is not actual PNG8')
    image=bpy.data.images.load(str(path),check_existing=False);image.colorspace_settings.name='Non-Color';image.alpha_mode='CHANNEL_PACKED'
    try:observed=pixels(image)[owners['indices']]
    finally:bpy.data.images.remove(image)
    np.save(Path(path).parent/'normal-png8-readback.npy',observed)
    parts=[];failures=[]
    for pi,target in enumerate(target_frames):
        selected=np.flatnonzero(owners['parts']==pi)
        if not len(selected):continue
        errors,_=target_normal_errors(observed[selected,:3],source_object_world(native_rgba[selected]),target,owners['faces'][selected],owners['weights'][selected])
        bad=np.flatnonzero(errors>PNG8_WORLD_ERROR_BOUND)
        parts.append(dict(part=pi,samples=len(selected),failed=len(bad),maximum_error=float(errors.max(initial=0))))
        for j in bad[:max(0,32-len(failures))]:
            row=selected[j];failures.append(dict(part=pi,pixel=int(owners['indices'][row]),face=int(owners['faces'][row]),error=float(errors[j])))
    return dict(valid=not any(p['failed'] for p in parts),bit_depth=8,actual_file_readback=True,source_WORLD_field_reused=True,world_error_bound=float(PNG8_WORLD_ERROR_BOUND),not_RGB32_1e6_claim=True,parts=parts,failures=failures,maximum_error=max((p['maximum_error'] for p in parts),default=0.))


def apply_target_normal_attributes(low,raw,target,affected_faces):
    """Export-only N split support; serialized frozen plan remains N/T authority."""
    before_v,before_f,_=arrays(low.data);before_uv=uv_loop_data(low.data);matrix=np.asarray(low.matrix_world,float)
    if not np.array_equal(before_v,raw['vertices']) or not np.array_equal(before_f,raw['faces']) or not np.array_equal(before_uv,raw['uv']) or not np.array_equal(matrix,raw['matrix']):raise ValueError('LOW geometry differs before explicit target-frame application')
    if len(affected_faces):low.data.normals_split_custom_set(target['n'].tolist())
    after_v,after_f,_=arrays(low.data)
    if before_v.tobytes()!=after_v.tobytes() or before_f.tobytes()!=after_f.tobytes() or before_uv.tobytes()!=uv_loop_data(low.data).tobytes() or matrix.tobytes()!=np.asarray(low.matrix_world,float).tobytes():raise ValueError('Target-frame application changed LOW geometry/UV/matrix')
    return dict(affected_faces=list(affected_faces),positions_faces_UV_matrix_byteexact=True,N_T_authority='Final serialized frozen Three target frame; intermediate Blender N storage may quantize and Mikk T may regenerate',no_SOURCE_normal_resampling=True)


def transfer(parts, material_doc, config, directory, *, tbn_before, ao_owner_lows=None):
    resolution=config['resolution'];lows=[p['low'] for p in parts]
    # Repair and gate invalid native frames before even the coverage bakes.
    raw_frames=[];planned_frames=[];repair_loops=[];planned_paths=[];preflight=[]
    for index,path in enumerate(tbn_before):
        with np.load(path) as saved:raw={k:saved[k] for k in saved.files}
        try:planned,changed=plan_tangents(raw)
        except TangentProjectionZero as error:
            error.initial_before_cage_part=index;error.initial_before_cage_paths=tuple(tbn_before)
            raise
        metrics=tbn_metrics(planned,planned)
        planned_path=directory/f'tbn-planned-{index:03d}.npz';np.savez_compressed(planned_path,**planned)
        raw_frames.append(raw);planned_frames.append(planned);repair_loops.append(changed);planned_paths.append(str(planned_path))
        preflight.append(dict(part=index,repaired_loops=changed.tolist(),planned_snapshot=str(planned_path),
                              repair=frame_repair_metrics(raw,planned,changed),**metrics))
    write_json_atomic(directory/'tbn-preflight.json',dict(valid=all(p['valid'] for p in preflight),parts=preflight))
    if not all(p['valid'] for p in preflight):
        if config['operation']=='optimize-rebake':raise NativeTBNRejected(directory/'tbn-preflight.json',preflight)
        raise ValueError('Planned tangent frame failed before PBR')
    try:serialization,serialization_metrics=predict_serialization(raw_frames,planned_frames)
    except ValueError as exc:
        write_json_atomic(directory/'tbn-serialization-preflight.json',dict(valid=False,error=str(exc)));raise
    serialization_path=directory/'tbn-serialization-plan.npz';np.savez_compressed(serialization_path,**serialization)
    write_json_atomic(directory/'tbn-serialization-preflight.json',dict(valid=True,**serialization_metrics))
    del serialization
    padding_report=dict(policy='Native PBR margin0 then complete geometric SAME-chart bilinear support from ORIGINAL valid samples',
        guarantee='Base-level bilinear only, REPEAT; not mip/anisotropic/appearance',
        coverage_filled_texels=0,recovered_texels=0,channels={},visual_qualification='NOT_RUN')
    target_frames=list(planned_frames);target_paths=[];target_repairs=[]
    paths={};channels=transfer_channels(material_doc);grids=map_grid_plan(config,channels)
    grid_reports=[];map_reports={};sources=[shader(p['high'],material_doc) for p in parts]
    q1_face_self_budget=dict(rows=0,parts=0,clone_bytes=0)
    strength=sources[0][3]['emission_strength']
    if any(item[3]['emission_strength']!=strength for item in sources):raise ValueError('Source material emission strengths diverged')
    for grid in grids:
        try:
            resolution=grid['resolution'];grid_directory=directory/f'grid-{resolution}'
            grid_directory.mkdir()
            # No geometry/frame changes are permitted between map-group samplings.
            snapshots=tbn_snapshot(lows,grid_directory,'before-grid')
            for raw,path in zip(raw_frames,snapshots):
                with np.load(path) as saved:
                    if any(raw[k].dtype!=saved[k].dtype or raw[k].shape!=saved[k].shape or raw[k].tobytes()!=saved[k].tobytes() for k in raw):raise ValueError('LOW geometry/UV/N/T changed before map grid')
            progress_marker(config,'correspondence')
            bake_target,node,grid_sources,bake_all,coverage,owners=prepare_nearest_transfer(parts,material_doc,dict(config,resolution=resolution,_q1_face_self_budget=q1_face_self_budget),grid_directory,raw_frames,planned_frames,sources=sources)
            coverage['channels']=grid['channels'];grid_reports.append(coverage)
            write_json_atomic(grid_directory/'coverage.json',coverage)
            occupancy=np.load(coverage['masks']['occupancy']).ravel();occupancy.setflags(write=False)
            try:
                padding_plan=chart_bilinear_support(raw_frames,[p['uv_filter_chart_ids'] for p in parts],owners,occupancy,resolution)
            except UVFilterSupportError as error:
                path=grid_directory/'bilinear-support.json';write_json_atomic(path,error.report)
                if config['operation']=='optimize-rebake':raise CandidateRejected('bilinear_filter_support',path,dict(error=str(error))) from error
                raise ValueError(f'UV bilinear support refused at {resolution}px; supplied UV unchanged; see {path}') from error
            np.save(grid_directory/'bilinear-required-charts.npy',padding_plan['required_charts'])
            np.save(grid_directory/'bilinear-original-seed-origins.npy',padding_plan['origins'])
            support=dict(padding_plan['report'],face_chart_provenance=[dict(part=pi,path=p['uv_filter_face_charts_path']) for pi,p in enumerate(parts)],original_native_occupancy=coverage['masks']['occupancy'],original_Q_correspondence_complete=coverage['complete'],required_charts=str(grid_directory/'bilinear-required-charts.npy'),original_seed_origins=str(grid_directory/'bilinear-original-seed-origins.npy'))
            write_json_atomic(grid_directory/'bilinear-support.json',support)
            coverage['bilinear_support']=support;write_json_atomic(grid_directory/'coverage.json',coverage)
            coverage['timing']['PBR_channels']={}
            progress_marker(config,'bake')
            for label in grid['channels']:
                generated_ao=None;ao_report=None
                if label=='orm':
                    ao_report=dict(ao_source='authored' if material_doc.get('occlusionTexture') is not None else 'neutral')
                    if config.get('ao_mode','transfer')=='generate_if_missing' and material_doc.get('occlusionTexture') is None:
                        generated_ao,ao_report=geometric_ao_rows(lows,ao_owner_lows,occupancy,resolution)
                image=new_map('Baked '+label,resolution,'sRGB' if label in ('base','emission') else 'Non-Color',float_buffer=label=='normal')
                for source,bsdf,output,proxies in grid_sources:
                    if label=='normal':
                        for link in list(output.inputs['Surface'].links):source.node_tree.links.remove(link)
                        source.node_tree.links.new(bsdf.outputs['BSDF'],output.inputs['Surface'])
                    else:set_proxy(source,output,proxies[label])
                native_bake_started=time.monotonic()
                bake_all(image,normal=label=='normal',margin=0)
                native_bake_seconds=time.monotonic()-native_bake_started
                map_write_started=time.monotonic()
                data=pixels(image)
                if generated_ao is not None:
                    cursor=0
                    for start in range(0,len(occupancy),65536):
                        stop=min(start+65536,len(occupancy));selected=occupancy[start:stop]
                        count=int(np.count_nonzero(selected))
                        data[start:stop,0][selected]=generated_ao[cursor:cursor+count];cursor+=count
                    if cursor!=len(generated_ao):raise ValueError('Geometric AO composition row count differs')
                    del generated_ao
                    generated_ao=True
                if label=='normal':
                    native_normal_values=data[owners['indices']].copy();np.save(directory/'normal-conversion-input.npy',native_normal_values)
                    measurements=[]
                    for pi,part in enumerate(parts):
                        selected=part['sample_rows']
                        if len(selected):
                            measured,target=object_normals_to_final(data,owners['indices'][selected],planned_frames[pi],owners['faces'][selected],owners['weights'][selected]);measurements.append(dict(part=pi,**measured));target_frames[pi]=target
                    target_repairs=[dict(part=pi,affected_faces=next((m['conditioning_repair']['affected_faces'] for m in measurements if m['part']==pi),[])) for pi in range(len(parts))]
                    for pi,target in enumerate(target_frames):
                        path=directory/f'tbn-target-{pi:03d}.npz';np.savez_compressed(path,**target);target_paths.append(str(path))
                    target_serialization,target_serialization_metrics=predict_serialization(raw_frames,target_frames);serialization_path=directory/'tbn-target-serialization-plan.npz';np.savez_compressed(serialization_path,**target_serialization)
                    write_json_atomic(directory/'tbn-target-serialization-preflight.json',dict(valid=True,**target_serialization_metrics));del target_serialization
                    conversion=dict(valid=True,parts=measurements,target_frame_repairs=target_repairs,normal_space='Cached native WORLD OBJECT -> explicitly frozen Three0.171 target ABI; source read bias only, unbiased glTF RGB',converted_texels=len(owners['indices']),world_roundtrip_limit=1e-6,
                        max_normalized_world_roundtrip_residual=max((m['max_normalized_world_roundtrip_residual'] for m in measurements),default=0.))
                    write_json_atomic(directory/'normal-conversion.json',conversion)
                if label=='alpha':
                    base=bpy.data.images.load(str(paths['base']),check_existing=False);base.alpha_mode='CHANNEL_PACKED'
                    bp,channel_padding=finalize_baked_channel(label,data,occupancy,resolution,material_doc,padding_plan=padding_plan,base_rgba=pixels(base))
                    base.pixels.foreach_set(bp.ravel());base.update();save_map(base,directory,'base-rgba');bpy.data.images.remove(base);del bp
                else:
                    data,channel_padding=finalize_baked_channel(label,data,occupancy,resolution,material_doc,padding_plan=padding_plan,generated_ao=generated_ao is True)
                    image.pixels.foreach_set(data.ravel());image.update()
                    if label=='normal':
                        stored=new_map('Unbiased glTF normal PNG8',resolution,'Non-Color',float_buffer=False);stored.pixels.foreach_set(data.ravel());stored.update()
                        paths[label]=save_map(stored,directory,label);bpy.data.images.remove(stored)
                        readback=normal_png8_readback(paths[label],native_normal_values,owners,target_frames);conversion['PNG8_readback']=readback;conversion['valid']=readback['valid']
                        for measured in conversion['parts']:
                            actual=next(p for p in readback['parts'] if p['part']==measured['part']);measured['PNG8_actual_readback']='PASS' if not actual['failed'] else 'FAIL';measured['PNG8_actual_max_residual']=actual['maximum_error']
                        write_json_atomic(directory/'normal-png8-readback.json',readback);write_json_atomic(directory/'normal-conversion.json',conversion)
                        if not readback['valid']:raise ValueError('Actual PNG8 target normal exceeds quantization-derived world envelope')
                        del native_normal_values
                    else:paths[label]=save_map(image,directory,'base-rgba' if label=='base' else label)
                channel_padding['resolution']=resolution
                padding_report['channels'][label]=channel_padding
                map_reports[label]=dict(resolution=resolution,width=resolution,height=resolution,coverage_report=str(grid_directory/'coverage.json'),occupied_texels=coverage['occupied_texels'],hit_texels=coverage['hit_texels'],miss_texels=0,filled_texels=0)
                if label=='orm':map_reports[label]['ambient_occlusion']=ao_report
                with Path(paths['base' if label=='alpha' else label]).open('rb') as saved_map:header=saved_map.read(24)
                if header[:8]!=b'\x89PNG\r\n\x1a\n' or int.from_bytes(header[16:20],'big')!=resolution or int.from_bytes(header[20:24],'big')!=resolution:raise ValueError('Saved map dimensions differ from native grid')
                write_json_atomic(directory/'post-composite-padding.json',padding_report)
                if label=='normal':
                    conversion['padded_unoccupied_texels']=channel_padding['padded_unoccupied_texels']
                    conversion['coverage_filled_texels']=0
                    write_json_atomic(directory/'normal-conversion.json',conversion)
                if label=='normal':
                    tbn_after=tbn_snapshot(lows,directory,'after-normal');measurements=[]
                    for index,(before_path,after_path) in enumerate(zip(tbn_before,tbn_after)):
                        with np.load(after_path) as saved:after_raw={k:saved[k] for k in saved.files}
                        after,after_changed=plan_tangents(after_raw);np.savez_compressed(directory/f'tbn-planned-after-normal-{index:03d}.npz',**after)
                        metrics=tbn_metrics(planned_frames[index],after)
                        fields=('n','t','sign','uv','loop_vertices','vertices','faces','triangle_loops','matrix','smooth','face_n')
                        raw_exact=all(raw_frames[index][k].dtype==after_raw[k].dtype and raw_frames[index][k].shape==after_raw[k].shape and raw_frames[index][k].tobytes()==after_raw[k].tobytes() for k in fields)
                        metrics.update(raw_stored_inputs_exact=raw_exact,repair_before=preflight[index]['repair'],
                                       repair_after=frame_repair_metrics(after_raw,after,after_changed),
                                       repaired_loop_identity_exact=bool(np.array_equal(repair_loops[index],after_changed)))
                        metrics['valid']=metrics['valid'] and raw_exact and metrics['repaired_loop_identity_exact']
                        measurements.append(dict(part=index,low=lows[index].name,before_snapshot=before_path,after_snapshot=after_path,**metrics))
                    tbn_report=dict(valid=all(m['valid'] for m in measurements),scope='Native LOW unchanged through source bake; explicit final Three target N/T repair is separate and serialized afterward',policy=dict(max_norm_error=1e-4,max_abs_dot_nt=1e-4,max_direction_change_degrees=.01,sign_changes=0,stored_inputs='exact',coordinate_space='Mesh-local N/T with frozen object matrix'),parts=measurements)
                    write_json_atomic(directory/'tbn-metrics.json',tbn_report)
                    if not tbn_report['valid']:raise ValueError(f'Functional tangent frame gate failed; see {directory / "tbn-metrics.json"}')
                del data;bpy.data.images.remove(image)
                coverage['timing']['PBR_channels'][label]=dict(native_Cycles_and_operator_seconds=native_bake_seconds,
                    map_postprocess_write_validation_seconds=time.monotonic()-map_write_started)
                note('channel_baked',object=parts[0]['object'],material=parts[0]['material'],channel=label,
                     resolution=resolution,nearest_correspondence_texels=coverage['hit_texels'],coverage_filled_texels=0)
            write_json_atomic(grid_directory/'coverage.json',coverage)
            for part in parts:
                for key in ('sample_proxy','sample_cage'):
                    if key in part:remove(part.pop(key))
                part.pop('sample_rows',None);part.pop('sample_caps',None)
            for low in lows:low.data.materials.clear()
            bpy.data.materials.remove(bake_target)
            del owners,occupancy,bake_all,padding_plan,grid_sources
            gc.collect()
        finally:
            for part in parts:
                remove_source_self(part);remove_q1_face_self(part)
    coverage=dict(method='Independent native LOW ownership and verified own-source POSITION/FACE per grid; ray and exact-source SELF contexts reported separately',
        complete=all(g['complete'] for g in grid_reports),sampling_grids=grid_reports,
        occupied_texels=sum(g['occupied_texels'] for g in grid_reports),hit_texels=sum(g['hit_texels'] for g in grid_reports),
        miss_texels=0,filled_texels=0,coverage_filled_texels=0,
        sample_count_scope='Sum over distinct grids, not unique surface points or resized texels',map_groups=map_reports)
    write_json_atomic(directory/'coverage.json',coverage)
    images={key:bpy.data.images.load(str(path),check_existing=False) for key,path in paths.items()}
    for key,image in images.items():image.colorspace_settings.name='sRGB' if key in ('base','emission') else 'Non-Color';image.alpha_mode='CHANNEL_PACKED'
    final=finish_material(lows[0],images,images['base'],images['orm'],material_doc,strength)
    final.name=f'MF_{parts[0]["material"]}_{directory.name}'
    for part in parts:
        low=part['low'];low.data.materials.clear();low.data.materials.append(final)
        part['final_material']=final.name;remove(part['high'])
        for key in ('sample_proxy','sample_cage'):
            if key in part:remove(part.pop(key))
        part.pop('sample_rows',None);part.pop('sample_caps',None)
        if part.get('projection_cage') is not None:remove(part.pop('projection_cage'))
        if part.get('local_projection_cage') is not None:remove(part.pop('local_projection_cage'))
    for image in list(bpy.data.images):
        if not image.users:bpy.data.images.remove(image)
    gc.collect()
    return dict(coverage,post_composite_padding=padding_report,maps={k:str(v) for k,v in paths.items()},tbn=tbn_report,channel_plan=channels,tbn_planned=planned_paths,tbn_target=target_paths,target_frame_repairs=target_repairs,tbn_serialization_plan=str(serialization_path),final_material=final.name)


def import_tagged(path, target, directory, *, staging=None, prepare=False):
    # Only material names in this disposable import copy are changed, so slots
    # can be bound without relying on Blender's automatic .001 suffixes.
    doc,binary=_split_glb(path)
    doc,binary,audit=stage_duplicate_faces(doc,binary,prepare=prepare)
    if staging is not None:staging.update(audit)
    note('lossless_import_staging',target=target,**audit)
    prefix='MF_TARGET_' if target else 'MF_SOURCE_'
    for index,material in enumerate(doc.get('materials',[])):material['name']=prefix+str(index)
    stage=directory/('target-import.glb' if target else 'source-import.glb')
    _write_glb(stage,doc,binary)
    before=set(bpy.data.objects);bpy.ops.import_scene.gltf(filepath=str(stage),merge_vertices=False)
    objects=[o for o in set(bpy.data.objects)-before if o.type=='MESH']
    for obj in objects:
        for material in obj.data.materials:
            if material is not None:
                if not material.name.startswith(prefix):raise ValueError('Imported material identity is ambiguous')
                material['mesh_finishing_material_index']=int(material.name[len(prefix):])
                if material.use_nodes:
                    for node in material.node_tree.nodes:
                        if node.type=='TEX_IMAGE':
                            image=node.image
                            # Packed glTF images decode lazily when size is read.
                            size=tuple(image.size) if image is not None else ()
                            if len(size)!=2 or min(size)<=0:raise ValueError('Blender could not decode an embedded source image')
                            if max(size)>8192:raise ValueError('Source images above 8192 pixels are unsupported')
                            if not image.has_data or image.channels<=0 or len(image.pixels)!=size[0]*size[1]*image.channels:raise ValueError('Decoded source image has incomplete pixels')
                            if not all(math.isfinite(float(v)) for v in image.pixels[:image.channels]):raise ValueError('Decoded source image has nonfinite pixels')
    return objects


def supplied_target(parts, originals, config, directory):
    # Explicitly bounded one-to-one component pairing. No global geometric
    # certificate: every paired component still passes sampled geometry gates.
    from mathutils.kdtree import KDTree
    from production_topology import split_exact_opposite_singletons
    for original in originals:original.name='SOURCE_ORIGINAL_'+original.name
    targets=import_tagged(Path(config['target_path']),True,directory)
    if sorted(o.name for o in targets)!=sorted(config['plan']['objects']):raise ValueError('High/low authored object names must match')
    source_groups={}
    for part in parts:source_groups.setdefault((part['object'],part['material']),[]).append(part)
    seen=set();comparisons=[]
    for target in targets:
        vertices,faces,loops=arrays(target.data)
        material_ids=np.array([target.data.polygons[t.polygon_index].material_index for t in target.data.loop_triangles])
        for slot in sorted(set(material_ids)):
            keys=[k for k in source_groups if k[0]==target.name]
            if len(keys)==1:key=keys[0]
            else:
                material=target.data.materials[int(slot)] if len(target.data.materials)>slot else None
                index=material.get('mesh_finishing_material_index') if material else None
                docs=config['target_plan']['materials']
                name=docs[index].get('name') if index is not None else None
                keys=[k for k in keys if config['plan']['materials'][k[1]].get('name')==name]
                if len(keys)!=1:raise ValueError('Multi-material targets must preserve distinct source material names')
                key=keys[0]
            if key in seen:raise ValueError('Multiple target material slots map to one source material group')
            seen.add(key);sources=source_groups[key];selected=np.flatnonzero(material_ids==slot)
            groups=components(vertices,faces[selected])
            original_count=len(groups);exact_owners={}
            if len(groups)!=len(sources):
                singletons=[]
                for i,part in enumerate(sources):
                    if len(part['high'].data.polygons)!=1:continue
                    v,f,_=arrays(part['high'].data)
                    if len(f)==1:singletons.append((i,v[f[0]],np.array(part['high'].matrix_world,np.float64)))
                groups,exact_owners=split_exact_opposite_singletons(vertices,faces[selected],groups,singletons,
                    np.array(target.matrix_world,np.float64),len(sources))
            if len(groups)!=len(sources):raise ValueError(f'Supplied target changed geometric component count: source={len(sources)}, target={original_count}, recovered={len(groups)}')
            if len(groups)>256:raise ValueError('Supplied target exceeds 256 components per material')
            reserved=set(exact_owners.values())
            kd=KDTree(len(sources)-len(reserved)) if len(reserved)<len(sources) else None
            source_bounds={}
            for i,part in enumerate(sources):
                if i in reserved:continue
                bound=surface(part['high']);kd.insert(Vector([(a+b)/2 for a,b in zip(bound['bounds_min'],bound['bounds_max'])]),i)
                # Retain only existing bounds/cap, never samples, BVHs or world vertices.
                source_bounds[i]=(tuple(bound['bounds_min']),tuple(bound['bounds_max']),
                                  min(config['max_error_world'],config['max_error_relative']*bound['diagonal']))
            if kd is not None:kd.balance()
            assigned=set()
            for group_index,group in enumerate(groups):
                fi=selected[group];rows,mapping=np.unique(faces[fi],return_inverse=True)
                low=make_object('SUPPLIED_LOW',vertices[rows],mapping.reshape(-1,3),target.matrix_world)
                if not target.data.uv_layers and config['operation']=='rebake-onto-mesh':raise ValueError('Rebake onto Mesh requires final target UV; use UV / Repack + Rebake to generate it')
                if target.data.uv_layers:
                    uv=np.empty((len(target.data.loops),2),np.float32);target.data.uv_layers.active.data.foreach_get('uv',uv.ravel())
                    layer=low.data.uv_layers.new(name='MeshFinishingUV');layer.data.foreach_set('uv',uv[loops[fi]].ravel())
                normals=np.empty((len(target.data.loops),3),np.float32);target.data.corner_normals.foreach_get('vector',normals.ravel())
                for face in low.data.polygons:face.use_smooth=True
                low.data.normals_split_custom_set(normals[loops[fi]].reshape(-1,3).tolist())
                observed=surface(low);center=Vector([(a+b)/2 for a,b in zip(observed['bounds_min'],observed['bounds_max'])])
                fallback=None
                if group_index in exact_owners:index=exact_owners[group_index]
                else:
                    if kd is None:raise ValueError('Supplied component correspondence is ambiguous')
                    _,index,_=kd.find(center)
                    def bounds_limit(index):
                        lower,upper,cap=source_bounds[index]
                        shift=max(abs(a-b) for bound,field in ((lower,'bounds_min'),(upper,'bounds_max')) for a,b in zip(bound,observed[field]))
                        return shift,cap
                    shift,cap=bounds_limit(index)
                    if index in assigned or shift>cap:
                        # Include assigned sources: removing them could manufacture
                        # uniqueness for geometrically ambiguous target components.
                        eligible=[]
                        for i in range(len(sources)):
                            if i in reserved:continue
                            source_shift,source_cap=bounds_limit(i)
                            if source_shift<=source_cap:eligible.append(i)
                        if len(eligible)==1:
                            previous=index;index=eligible[0]
                            fallback=dict(target_component=int(group_index),previous_source_component=int(previous),
                                          source_component=int(index),criterion='unique_source_specific_AABB_cap',
                                          trigger='already_assigned' if previous in assigned else 'bounds_limit')
                if index in assigned:raise ValueError('Supplied component correspondence is ambiguous')
                assigned.add(index);part=sources[index]
                comparison=compare(surface(part['high']),observed,config)
                if not comparison['valid']:
                    note('supplied_target_geometry_rejected',object=key[0],material=int(key[1]),source_component=int(index),**comparison)
                    cap=comparison['error_cap_world']
                    metrics=(('source-to-target','source_to_low_max_sample_world'),
                             ('target-to-source','low_to_source_max_sample_world'),
                             ('bounds shift','bounds_shift_world'))
                    failed=', '.join(f'{label}={comparison[field]:.12g}' for label,field in metrics if comparison[field]>cap)
                    raise ValueError(f'Supplied target exceeds geometry error/bounds limits: {failed} '
                                     f'(limit={cap:.12g} world units). Use a target closer to the source surface.')
                if fallback is not None:
                    note('supplied_target_unique_bounds_fallback',object=key[0],material=int(key[1]),
                         bounds_shift_world=comparison['bounds_shift_world'],error_cap_world=comparison['error_cap_world'],**fallback)
                remove(part['low']);part['low']=low;part['chosen']=dict(triangles=observed['triangles'],**comparison)
                comparisons.append(comparison)
            if assigned!=set(range(len(sources))):raise ValueError('Supplied component correspondence is ambiguous')
        remove(target)
    if seen!=set(source_groups):raise ValueError('Supplied target lost a source material group')
    count=sum(len(p['low'].data.polygons) for p in parts)
    return dict(source_triangles=config['plan']['triangles'],triangles=count,triangle_budget=None,
                budget_met=None,output_changed=count<config['plan']['triangles'],
                operation='supplied_target_transfer_not_simplification',comparisons=comparisons)


def join_authored_objects(parts):
    groups={}
    for part in parts:groups.setdefault(part['object'],[]).append(part['low'])
    result=[]
    for name,objects in groups.items():
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects:obj.select_set(True)
        bpy.context.view_layer.objects.active=objects[0]
        if len(objects)>1:bpy.ops.object.join()
        obj=objects[0];obj.name=name;obj.data.name=name;result.append(obj)
    return result


def mesh_digest(obj):
    mesh=obj.data;mesh.calc_loop_triangles();h=hashlib.sha256()
    for collection,field,width,dtype in [(mesh.vertices,'co',3,np.float32),(mesh.loops,'vertex_index',1,np.int32),
                                         (mesh.corner_normals,'vector',3,np.float32),(mesh.uv_layers.active.data,'uv',2,np.float32)]:
        data=np.empty(len(collection)*width,dtype=dtype);collection.foreach_get(field,data);h.update(data.tobytes())
    return h.hexdigest()


def export(path, objects, materials=True, *, normals=True):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:obj.hide_viewport=False;obj.hide_render=False;obj.select_set(True)
    bpy.context.view_layer.objects.active=objects[0]
    # Geometry-only reconstruction must retain indexed topology, not split
    # vertices for flat corner normals. All existing callers keep old options.
    normal_options={} if normals else dict(export_normals=False)
    bpy.ops.export_scene.gltf(filepath=str(path),export_format='GLB',use_selection=True,
                              export_materials='EXPORT' if materials else 'NONE',export_image_format='AUTO',
                              export_tangents=materials,export_animations=False,export_skins=False,
                              export_cameras=False,export_lights=False,export_vertex_color='NONE',**normal_options)
    if not path.is_file() or not path.stat().st_size:raise ValueError('GLB export failed')


def process_attempt(args, config, directory, *, minimum_ratios=None, retained_source_retry=None):
    started=time.monotonic();source=Path(args.source);candidate=Path(args.candidate)
    if digest(source)!=args.sha256:raise ValueError('Preserved source hash changed')
    if config['operation']!='simplify-mesh':progress_marker(config,'geometry')
    if config.get('ao_mode','transfer')=='generate_if_missing':
        if any(material.get('alphaMode','OPAQUE')!='OPAQUE' for material in config['plan']['materials']):
            raise ValueError('Generated geometric AO requires opaque source materials; MASK/BLEND are unsupported. Use transfer AO.')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    config=dict(config,_geometry_directory=str(directory/'gpu-geometry'))
    source_staging={}
    originals=sorted(import_tagged(source,False,directory,staging=source_staging,prepare=True),key=lambda o:o.name)
    imported_triangles=sum(len(arrays(o.data)[1]) for o in originals)
    note('source_imported',raw_triangles=config['plan']['triangles'],imported_triangles=imported_triangles)
    if sorted(o.name for o in originals)!=sorted(config['plan']['objects']):raise ValueError('Imported object identity differs from glTF')
    if any(o.modifiers or not np.isfinite(np.array(o.matrix_world)).all() or abs(o.matrix_world.to_3x3().determinant())<1e-15 for o in originals):raise ValueError('Unsupported modifiers or singular/nonfinite transform')
    supplied=config['operation']=='rebake-onto-mesh' or (config['operation']=='uv-repack-rebake' and bool(config.get('target_path')))
    repack=config['operation']=='uv-repack-rebake'
    parts=source_parts(originals,config['plan'],0 if supplied else config['protected_component_triangles'],
        preserve_geometry=repack,geometry_only=config.get('geometry_only',False),
        protect_blender_interfaces=not supplied and not repack and config.get('simplification_backend','blender')=='blender')
    # Freeze geometry identities only after new HIGH/LOW transforms are evaluated.
    bpy.context.view_layer.update()
    actual_source=sum(p['source_triangles'] for p in parts)
    preparation=source_staging['preparation']
    validate_source_triangle_counts(config['plan']['triangles'],imported_triangles,actual_source,
        removed_exact_zero_area_triangles=preparation['removed_exact_zero_area_triangles'])
    note('source_partitioned',triangles=actual_source,parts=len(parts))
    if supplied:geometry=supplied_target(parts,originals,config,directory)
    elif repack:
        geometry=dict(source_triangles=actual_source,triangles=actual_source,triangle_budget=None,
                      budget_met=None,output_changed=False,uv_changed=True,operation='same_geometry_new_uv_rebake')
    else:
        geometry=simplify(parts,config,actual_source,minimum_ratios=minimum_ratios,retained_source_retry=retained_source_retry)
        for index,part in enumerate(parts):
            chosen=part['chosen']
            note('component_final_selection',part=index,object=part['object'],material=part['material'],components=part['source_component_ids'],
                 ratio=chosen['ratio'],effective_backend=chosen.get('effective_backend','protected_source_copy'),triangles=chosen['triangles'],
                 source_pristine_geometry_sha256=chosen.get('source_pristine_geometry_sha256'))
        geometry['source_retained_normals']=(dict(applied_parts=0,copied_corners=0,scope='Geometry-only input has no source appearance contract')
                                            if config.get('geometry_only') else preserve_source_retained_normals(parts,config))
        write_json_atomic(directory/'source-retained-normals.json',geometry['source_retained_normals'])
    geometry_finished=time.monotonic()
    for original in originals:remove(original)
    lows=[p['low'] for p in parts]
    if config['operation']=='simplify-mesh':
        for part in parts:remove(part['high'])
        lows=join_authored_objects(parts)
        for low in lows:
            for layer in list(low.data.uv_layers):low.data.uv_layers.remove(layer)
            low.data.materials.clear()
        export(candidate,lows,materials=False);bakes=[]
    else:
        scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=1
        scene.render.threads_mode='FIXED';scene.render.threads=max(1,min(8,int(config.get('cpu_threads',2))));scene.render.bake.target='IMAGE_TEXTURES'
        progress_marker(config,'uv')
        bakes=[];atlas_groups={}
        for part in parts:atlas_groups.setdefault((part['object'],part['material']),[]).append(part)
        for index,group in enumerate(atlas_groups.values()):
            group_lows=[p['low'] for p in group]
            out=directory/'maps'/f'atlas-{index:03d}';out.mkdir(parents=True)
            try:
                uv_generation,uv_validation=prepare_destination_uv(group,config,out)
                # Resolve assigned object TRS before freezing frames; later matrix checks stay byte-exact.
                bpy.context.view_layer.update()
                initialize_tangents(group_lows)
                tbn_before=tbn_snapshot(group_lows,out,'before-cage')
                frozen=[mesh_digest(low) for low in group_lows]
                reassigned=False
                try:baked=transfer(group,config['plan']['materials'][group[0]['material']],config,out,tbn_before=tbn_before,ao_owner_lows=([p['low'] for p in parts if p['object']==group[0]['object']] if config.get('ao_mode','transfer')=='generate_if_missing' else None))
                except TangentProjectionZero as error:
                    part_index=getattr(error,'initial_before_cage_part',None)
                    if (config['operation']!='optimize-rebake' or type(error) is not TangentProjectionZero
                            or type(part_index) is not int or not 0<=part_index<len(group)
                            or getattr(error,'initial_before_cage_paths',None)!=tuple(tbn_before)):raise
                    tbn_before,uv_validation=initial_generated_tangent_reassignment(group,config,out,tbn_before,uv_generation,uv_validation,error)
                    frozen=[mesh_digest(low) for low in group_lows];reassigned=True
                if reassigned:
                    # Outside the catch: a second or later-phase refusal cannot retry.
                    baked=transfer(group,config['plan']['materials'][group[0]['material']],config,out,tbn_before=tbn_before,ao_owner_lows=([p['low'] for p in parts if p['object']==group[0]['object']] if config.get('ao_mode','transfer')=='generate_if_missing' else None))
            except CandidateRejected as exc:
                if config['operation']=='optimize-rebake':
                    exc.minimum_ratios,exc.ratio_changes=coverage_retry_ratios(parts,group,config,actual_source)
                exc.simplification=geometry
                exc.atlas_group=dict(index=index,object=group[0]['object'],material=group[0]['material'])
                raise
            baked['destination_uv']=uv_validation;baked['uv_generation']=uv_generation;bakes.append(baked)
            if [mesh_digest(low) for low in group_lows]!=frozen:raise ValueError('Bake changed final geometry, normals or UV')
            baked['target_attribute_application']=[]
            for pi,part in enumerate(group):
                with np.load(tbn_before[pi]) as saved:raw={k:saved[k] for k in saved.files}
                with np.load(baked['tbn_target'][pi]) as saved:target={k:saved[k] for k in saved.files}
                baked['target_attribute_application'].append(apply_target_normal_attributes(part['low'],raw,target,baked['target_frame_repairs'][pi]['affected_faces']))
        lows=join_authored_objects(parts)
        export(candidate,lows)
        serialized_tbn=restore_serialized_glb_frames(candidate,{b['final_material']:b['tbn_serialization_plan'] for b in bakes})
        write_json_atomic(directory/'tbn-serialized.json',serialized_tbn)
        doc,binary=_split_glb(candidate)
        by_name={p['final_material']:config['plan']['materials'][p['material']] for p in parts}
        for material in doc.get('materials',[]):
            original=by_name.get(material.get('name'))
            if original is None:raise ValueError('Exported unexpected material')
            material['alphaMode']=original.get('alphaMode','OPAQUE');material['doubleSided']=original.get('doubleSided',False)
            if material['alphaMode']=='MASK':material['alphaCutoff']=original.get('alphaCutoff',.5)
        stage=candidate.with_suffix('.semantic.glb');_write_glb(stage,doc,binary);stage.replace(candidate)
    if bakes:progress_marker(config,'validation')
    # Fresh import of the actual serialized GLB, not merely the modifier scene.
    export_surfaces={o.name:surface(o,retain_serialized_world=True) for o in lows}
    expected_objects=len(lows);expected_triangles=geometry['triangles']
    bpy.ops.wm.read_factory_settings(use_empty=True)
    reimport_directory=directory/'final-import';reimport_directory.mkdir()
    reimport_staging={};imported=import_tagged(candidate,False,reimport_directory,staging=reimport_staging)
    count=sum(len(arrays(o.data)[1]) for o in imported)
    if count!=expected_triangles or len(imported)!=expected_objects or {o.name for o in imported}!=set(export_surfaces):raise ValueError('Final GLB reimport changed authored objects/triangle counts')
    export_comparison={};export_validation={}
    for obj in imported:
        final_surface=surface(obj,retain_serialized_world=True)
        export_comparison[obj.name]=compare(export_surfaces[obj.name],final_surface,dict(config,max_error_relative=1e-5,max_error_world=1e-5))
        export_validation[obj.name]=_serialized_surface_validation(export_surfaces[obj.name],final_surface,export_comparison[obj.name])
    effective_reimport_valid=all(v['effective_valid'] for v in export_validation.values())
    write_json_atomic(directory/'serialized-reimport-validation.json',dict(scope='Final serialized geometry only; not simplification fidelity or material acceptance',
        sampled_geometry=export_comparison,effective_geometry=export_validation,effective_geometry_valid=effective_reimport_valid))
    if not effective_reimport_valid:raise ValueError('Serialized GLB failed sampled and exact world-surface preservation')
    if bakes:
        for obj in imported:
            if not obj.data.uv_layers or not obj.data.materials:raise ValueError('Final GLB lost UV/materials')
        doc,final_binary=_split_glb(candidate)
        for mesh in doc.get('meshes',[]):
            for primitive in mesh['primitives']:
                if not {'POSITION','NORMAL','TEXCOORD_0','TANGENT'} <= set(primitive['attributes']):raise ValueError('Final GLB lost normal/UV/tangent attributes')
                if type(primitive.get('material')) is not int or not 0<=primitive['material']<len(doc['materials']):raise ValueError('Final GLB lost primitive material assignment')
        for texture in doc.get('textures',[]):
            index=texture.get('source')
            if type(index) is not int or not 0<=index<len(doc.get('images',[])):raise ValueError('Final texture image reference is invalid')
            sampler=doc.get('samplers',[])[texture['sampler']] if 'sampler' in texture else {}
            if sampler.get('wrapS',10497)!=10497 or sampler.get('wrapT',10497)!=10497:raise ValueError('Exported sampler differs from validated REPEAT bilinear support')
            image=doc['images'][index]
            if image.get('mimeType')!='image/png' or 'bufferView' not in image:raise ValueError('Final maps must be embedded PNG images')
        for material in doc['materials']:
            original=by_name.get(material.get('name'))
            if original is None or material.get('alphaMode','OPAQUE')!=original.get('alphaMode','OPAQUE') or material.get('doubleSided',False)!=original.get('doubleSided',False) or (original.get('alphaMode')=='MASK' and material.get('alphaCutoff',.5)!=original.get('alphaCutoff',.5)):raise ValueError('Serialized GLB changed material/alpha semantics')
            if not all(key in material.get('pbrMetallicRoughness',{}) for key in ('baseColorTexture','metallicRoughnessTexture')) or 'normalTexture' not in material or 'occlusionTexture' not in material:raise ValueError('Final GLB lost required PBR bindings')
            bindings=dict(base=material['pbrMetallicRoughness']['baseColorTexture'],orm=material['pbrMetallicRoughness']['metallicRoughnessTexture'],normal=material['normalTexture'])
            if 'emissiveTexture' in material:bindings['emission']=material['emissiveTexture']
            if material['occlusionTexture']['index']!=bindings['orm']['index']:raise ValueError('Final AO and roughness/metalness no longer share packed ORM')
            for label,binding in bindings.items():
                size=next(g['resolution'] for g in map_grid_plan(config,[label]))
                image=doc['images'][doc['textures'][binding['index']]['source']]
                view=doc['bufferViews'][image['bufferView']];start=view.get('byteOffset',0)
                header=final_binary[start:start+24]
                if header[:8]!=b'\x89PNG\r\n\x1a\n' or int.from_bytes(header[16:20],'big')!=size or int.from_bytes(header[20:24],'big')!=size:raise ValueError(f'Exported {label} dimensions differ from native {size}px grid')

    if digest(source)!=args.sha256:raise ValueError('Original source changed during processing')
    if supplied and digest(config['target_path'])!=config['target_sha256']:raise ValueError('Preserved target changed during processing')
    result=dict(schema_version='2.0.0',status='completed',candidate_sha256=digest(candidate),
                blender_version=bpy.app.version_string,device='CPU UV/PBR; CuMesh GPU geometry' if any(p.get('chosen',{}).get('effective_backend')=='cumesh' for p in parts) else 'CPU',
                timing=dict(source_import_and_geometry_seconds=geometry_finished-started,CPU_finalization_including_UV_PBR_export_reimport_seconds=time.monotonic()-geometry_finished),elapsed_seconds=time.monotonic()-started,
                source_preparation=dict(preparation,imported_triangles=imported_triangles,
                    partitioned_triangles=actual_source,import_staging=source_staging),
                simplification=geometry,bakes=bakes,reimport=dict(triangles=count,mesh_objects=len(imported),sampled_geometry=export_comparison,effective_geometry=export_validation,effective_geometry_valid=effective_reimport_valid,import_staging=reimport_staging),
                validation=dict(geometry='sampled',appearance='NOT_VISUALLY_REVIEWED',modly_ui='NOT_RUN'))
    return result


def cleanup_completed_import_staging(directory):
    """Best-effort disposal of fixed import copies, only after completion commits.

    Refuse redirected files/parents; this is not a concurrent path-race defense.
    Rejected attempts and every other diagnostic remain untouched.
    """
    counts=dict(removed_files=0,removed_bytes=0,skipped_files=0,errors=0)
    try:
        if directory.resolve()!=directory or not directory.is_dir():
            counts['skipped_files']=3;return counts
    except (OSError,RuntimeError):
        counts['errors']=1;return counts
    for relative in ('source-import.glb','target-import.glb','final-import/source-import.glb'):
        path=directory/relative
        try:
            if path.parent.resolve()!=path.parent or path.is_symlink():
                counts['skipped_files']+=1;continue
            if not path.is_file():continue
            size=path.stat().st_size
            path.unlink()
            counts['removed_files']+=1;counts['removed_bytes']+=size
        except (OSError,RuntimeError):
            counts['errors']+=1
    return counts


def main():
    parser=argparse.ArgumentParser()
    for name in ('source','sha256','config','candidate','sentinel'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);config=json.loads(Path(args.config).read_text())
    root=Path(args.sentinel).parent
    config['_geometry_deadline']=time.monotonic()+config['timeout_seconds']
    if config.get('operation') in ('reconstruct-wtivo','reconstruct-cumesh-dc'):
        from wtivo_reconstruct import reconstruct
        result=reconstruct(Path(args.source),args.sha256,root,config,bpy=bpy,
                           import_meshes=import_tagged,mesh_arrays=arrays,make_mesh=make_object,export_meshes=export)
        write_completion_metrics(Path(args.sentinel),result)
        note('reconstruction_completed',triangles=result['reconstruction']['triangles'],appearance='INVALIDATED')
        return
    if config.get('operation')=='inspect-geometry':
        from inspect_native_geometry import inspect_native_geometry
        geometry=inspect_native_geometry(Path(args.source),args.sha256,root,bpy=bpy)
        write_completion_metrics(Path(args.sentinel),dict(status='completed',operation='inspect-geometry',source_sha256=args.sha256,geometry=geometry))
        note('inspection_completed',mesh_instances=geometry['mesh_instances'],triangles=geometry['native_triangles'])
        return
    attempts=[];minimum_ratios=None;first_rejection=None
    retained_source_retry=({} if config.get('operation')=='optimize-rebake' and
        config.get('simplification_backend') in ('cumesh_install_if_needed','cumesh_existing') else None)
    for index in range(2):
        directory=root/f'attempt-{index:03d}';directory.mkdir()
        try:result=process_attempt(args,config,directory,minimum_ratios=minimum_ratios,retained_source_retry=retained_source_retry)
        except CandidateRejected as exc:
            rejected=dict(status='rejected',reason=exc.reason,directory=str(directory),
                failure_path=str(exc.path),details=exc.details,atlas_group=exc.atlas_group,
                simplification=exc.simplification,ratio_changes=exc.ratio_changes,
                retry_scope='All mutable parts of the failed atlas group; not inferred from bounded diagnostics')
            if isinstance(exc,ProjectionCoverageError):
                rejected.update(coverage_path=str(exc.path),occupied_texels=exc.coverage['occupied_texels'],
                                hit_texels=exc.coverage['hit_texels'],miss_texels=exc.coverage['miss_texels'])
            write_json_atomic(directory/'attempt-result.json',rejected);attempts.append(rejected)
            if index or not exc.minimum_ratios:
                exc._final_guidance=final_rejection_guidance(exc,config)
                raise
            first_rejection=exc;minimum_ratios=exc.minimum_ratios
            note('conservative_candidate_retry',reason=exc.reason,parts=len(exc.ratio_changes),previous_attempt=str(directory))
        except NoTriangleReduction as exc:
            if first_rejection is None:raise
            write_json_atomic(directory/'attempt-result.json',dict(status='rejected',
                              reason='no_triangle_reduction',details=dict(error=str(exc))))
            first_rejection._final_guidance=final_rejection_guidance(first_rejection,config,retry_no_reduction=True)
            raise first_rejection from exc
        else:
            attempts.append(dict(status='completed',directory=str(directory)))
            result['attempts']=attempts
            write_completion_metrics(Path(args.sentinel),result)
            note('import_staging_cleanup',**cleanup_completed_import_staging(directory))
            note('completed',triangles=result['simplification']['triangles'],
                 budget_met=result['simplification']['budget_met'])
            return


def worker_entry():
    terminal=Path(sys.argv[sys.argv.index('--sentinel')+1]).with_name('worker-terminal.json')
    try:main()
    except BaseException as exc:
        message=f'{type(exc).__name__}: {exc}'
        guidance=getattr(exc,'_final_guidance','') if isinstance(exc,CandidateRejected) else ''
        if type(guidance) is str and len(guidance)<=300:message+=guidance
        write_json_atomic(terminal,dict(status='error',error=message[:500]))
        raise
    else:write_json_atomic(terminal,dict(status='completed'))


if __name__=='__main__':worker_entry()
