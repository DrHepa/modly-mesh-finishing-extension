# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only original/imported geometry diagnostics; not repair or validity."""
from pathlib import Path
from glb_inspector import GLBReader, inspect_glb

MAX_TRIANGLES=1_250_000
MAX_RENDER_ROWS=2_000_000
MAX_OBJECTS=64
MAX_BOUNDARY_GRAPH_EDGES=65_536


def plan_geometry_inspection(path, inspection=None):
    """Refuse unsupported input before Blender import, including inactive scopes."""
    path=Path(path)
    if not path.is_file() or path.suffix.lower()!='.glb' or path.stat().st_size>512*1024**2:
        raise ValueError('Geometry inspection requires a GLB file no larger than 512 MiB')
    inspection=inspect_glb(path) if inspection is None else inspection
    def require(ok,message):
        if not ok:raise ValueError('Geometry inspection: '+message)
    with GLBReader(path) as reader:
        doc=reader.doc
        require(not inspection['findings'],'structural findings must be resolved before native import')
        require(not any(doc.get(key) for key in ('animations','skins','cameras')),'animations, skins and cameras are unsupported')
        require(len(doc.get('buffers',[]))==1 and 'uri' not in doc['buffers'][0],'embedded GLB buffer required')
        require(set(doc.get('extensionsUsed',[]))<= {'EXT_texture_webp','KHR_materials_emissive_strength'},'unsupported glTF extension')
        require(set(doc.get('extensionsRequired',[]))<=set(doc.get('extensionsUsed',[])),'invalid required extensions')
        require(len(doc.get('scenes',[]))==1 and doc.get('scene')==0,'one explicit default scene required')
        nodes=doc.get('nodes',[]);seen=set();stack=list(doc['scenes'][0].get('nodes',[]))
        require(len(nodes)<=256,'scene exceeds the 256 node work limit')
        while stack:
            index=stack.pop()
            require(type(index)is int and 0<=index<len(nodes) and index not in seen,'invalid or repeated scene node')
            seen.add(index);node=nodes[index]
            require(not set(node)-{'name','mesh','children','matrix','translation','rotation','scale'},'unsupported node semantics')
            require(not ('matrix' in node and any(key in node for key in ('translation','rotation','scale'))),'matrix and TRS may not coexist')
            stack.extend(node.get('children',[]))
        require(len(seen)==len(nodes),'inactive nodes are unsupported (no silent omission)')
        mesh_nodes=[(i,n['mesh']) for i,n in enumerate(nodes) if 'mesh' in n]
        require(0<len(mesh_nodes)<=MAX_OBJECTS,'one to 64 mesh instances required')
        require(all(type(m)is int and 0<=m<len(doc.get('meshes',[])) for _,m in mesh_nodes),'invalid mesh instance')
        require({m for _,m in mesh_nodes}==set(range(len(doc.get('meshes',[])))),'unused meshes are unsupported')
        counts=inspection['counts']
        require(0<counts['active_scene_render_triangles']<=MAX_TRIANGLES,'active triangles exceed the 1250000 work limit or are empty')
        require(counts['active_scene_render_vertices']<=MAX_RENDER_ROWS,'active render rows exceed the 2000000 work limit')
        for mesh in doc['meshes']:
            require(not set(mesh)-{'name','primitives','extras'} and not mesh.get('extras'),'morph/mesh extensions or metadata are unsupported')
            require(bool(mesh['primitives']),'empty meshes are unsupported')
            for prim in mesh['primitives']:
                require(not set(prim)-{'attributes','indices','material','mode'},'unsupported primitive semantics')
                require(prim.get('mode',4)==4 and type(prim.get('indices'))is int,'indexed TRIANGLES required; other primitives are not omitted')
                attrs=prim['attributes']
                require('POSITION' in attrs and set(attrs)<= {'POSITION','NORMAL','TANGENT','TEXCOORD_0'},'unsupported vertex attributes')
                for semantic,index in attrs.items():
                    a=doc['accessors'][index];layout=reader.accessor_layout(index)
                    require(a.get('componentType')==5126 and a.get('type')=={'POSITION':'VEC3','NORMAL':'VEC3','TANGENT':'VEC4','TEXCOORD_0':'VEC2'}[semantic] and not a.get('normalized') and not layout[5],'dense float32 vertex attributes required')
                layout=reader.accessor_layout(prim['indices'])
                require(not layout[5] and layout[1]%3==0 and not doc['accessors'][prim['indices']].get('normalized'),'dense complete triangle indices required')
        for image in doc.get('images',[]):
            require('uri' not in image and type(image.get('bufferView'))is int and image.get('mimeType')in ('image/png','image/jpeg','image/webp'),'embedded PNG/JPEG/WebP images required')
            index=image['bufferView'];require(0<=index<len(doc.get('bufferViews',[])),'invalid embedded image bufferView')
            view=doc['bufferViews'][index];offset=view.get('byteOffset',0);length=view.get('byteLength')
            require(view.get('buffer')==0 and type(offset)is int and type(length)is int and offset>=0 and length>0 and offset+length<=doc['buffers'][0]['byteLength'],'embedded image bounds are invalid')
        return dict(mesh_nodes=[dict(node=i,mesh=m,name=nodes[i].get('name'),import_name=f'MF_INSPECT_NODE_{i:04d}',
                        original_triangles=sum(row['render_triangles'] for row in inspection['primitives'] if row['mesh']==m),
                        original_render_rows=sum(row['render_vertices'] for row in inspection['primitives'] if row['mesh']==m)) for i,m in mesh_nodes],
                    original_active_triangles=counts['active_scene_render_triangles'],original_active_render_rows=counts['active_scene_render_vertices'])


def _boundary_graph_summary(edges):
    """Summarize bounded, unique nonzero directed one-incidence edges only."""
    graph={}
    for a,b in edges:
        a,b=int(a),int(b)
        graph.setdefault(a,[]).append((b,1));graph.setdefault(b,[]).append((a,-1))
    seen=set();sizes=[];closed=chains=branched=coherent=0
    for start in graph:
        if start in seen:continue
        stack=[start];seen.add(start);twice_edges=endpoints=0;cycle=oriented=True;branch=False
        while stack:
            vertex=stack.pop();links=graph[vertex];degree=len(links);twice_edges+=degree
            endpoints+=degree==1;cycle&=degree==2;branch|=degree>2
            oriented&=degree==2 and sum(sign for _,sign in links)==0
            for neighbor,_ in links:
                if neighbor not in seen:seen.add(neighbor);stack.append(neighbor)
        sizes.append(twice_edges//2)
        if branch:branched+=1
        elif cycle:closed+=1;coherent+=oriented
        elif endpoints==2:chains+=1
    return dict(status='MEASURED',boundary_edges=len(edges),edge_limit=MAX_BOUNDARY_GRAPH_EDGES,
        components=len(sizes),closed_contours=closed,open_chains=chains,branched_components=branched,
        coherently_oriented_closed_contours=coherent,component_edges_min=min(sizes,default=0),component_edges_max=max(sizes,default=0),
        semantics='Exact-position one-incidence edge graph within this mesh/object. Closed contours have degree two everywhere; coherent cycles have one incoming and one outgoing edge per vertex. Graph classification does not establish a repairable hole, intended opening, area, intersections or physical validity.')


def geometric_topology(vertices, faces, *, include_boundary_graph=False):
    """Exact-position, per-object incidence; nonmanifold edges join all incidents."""
    import numpy as np
    vertices=np.asarray(vertices,dtype=np.float64);faces=np.asarray(faces,dtype=np.int32)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or not np.isfinite(vertices).all() or (faces.size and (faces.min()<0 or faces.max()>=len(vertices))):
        raise ValueError('Invalid native topology arrays')
    if len(faces)>MAX_TRIANGLES or len(vertices)>MAX_RENDER_ROWS:raise ValueError('Native topology exceeds admitted work limits')
    if not len(faces):
        result=dict(status='MEASURED',triangles=0,components=0,boundary_edges=0,nonmanifold_edges=0,orientation_conflict_edges=0,degenerate_edges=0,numerical_local_zero_area_triangles=0)
        if include_boundary_graph:result['boundary_graph']=_boundary_graph_summary(())
        return result
    _,inverse=np.unique(vertices,axis=0,return_inverse=True);canonical=inverse[faces]
    directed=np.stack((canonical,np.roll(canonical,-1,axis=1)),axis=-1).reshape(-1,2)
    face_ids=np.repeat(np.arange(len(faces),dtype=np.int32),3)
    nonzero=directed[:,0]!=directed[:,1];degenerate_edges=int(np.count_nonzero(~nonzero))
    directed=directed[nonzero];face_ids=face_ids[nonzero];keys=np.sort(directed,axis=1)
    order=np.lexsort((keys[:,1],keys[:,0]));keys=keys[order];directed=directed[order];face_ids=face_ids[order]
    starts=np.flatnonzero(np.r_[True,np.any(keys[1:]!=keys[:-1],axis=1)]) if len(keys) else np.empty(0,dtype=int)
    lengths=np.diff(np.r_[starts,len(keys)])
    parent=np.arange(len(faces),dtype=np.int32)
    def root(index):
        while parent[index]!=index:parent[index]=parent[parent[index]];index=parent[index]
        return int(index)
    for start,length in zip(starts[lengths>1],lengths[lengths>1]):
        first=root(int(face_ids[start]))
        for face in face_ids[start+1:start+length]:parent[root(int(face))]=first
    roots=np.fromiter((root(i) for i in range(len(faces))),dtype=np.int32,count=len(faces));_,sizes=np.unique(roots,return_counts=True)
    paired=starts[lengths==2]
    distinct=face_ids[paired]!=face_ids[paired+1]
    conflicts=distinct & np.all(directed[paired]==directed[paired+1],axis=1)
    a,b,c=(vertices[faces[:,i]] for i in range(3))
    result=dict(status='MEASURED',triangles=len(faces),components=len(sizes),component_triangles_min=int(sizes.min()),component_triangles_max=int(sizes.max()),
                unique_positions=int(inverse.max())+1,unique_nondegenerate_edges=len(starts),boundary_edges=int(np.count_nonzero(lengths==1)),
                nonmanifold_edges=int(np.count_nonzero(lengths>2)),orientation_conflict_edges=int(np.count_nonzero(conflicts)),
                self_incident_two_use_edges=int(np.count_nonzero(~distinct)),degenerate_edges=degenerate_edges,
                numerical_local_zero_area_triangles=int(np.count_nonzero(np.all(np.cross(b-a,c-a)==0,axis=1))),
                semantics='Exact-position complete-edge face incidence within this object only; point contacts do not connect. Nonmanifold edges join all incident faces. Zero-length edges are counted but do not connect faces; orientation conflicts count two distinct incidents with equal direction, not global orientability. Local zero-area counts use float64 cross-product equality and can round nondegenerate triangles to zero; they are not exact degeneracy proofs.')
    if include_boundary_graph:
        count=result['boundary_edges']
        result['boundary_graph']=(_boundary_graph_summary(directed[starts[lengths==1]]) if count<=MAX_BOUNDARY_GRAPH_EDGES else
            dict(status='NOT_EVALUATED_RESOURCE_CAP',boundary_edges=count,edge_limit=MAX_BOUNDARY_GRAPH_EDGES))
    return result


def original_accessor_topology(path, *, validated_plan=None):
    """Read every original mesh definition; never expand/weld its instances.

    Native callers reuse the same-input plan_geometry_inspection result. Direct
    CPU callers omit it to run that admission first. Actual counts and unsigned
    index ranges are checked independently before allocation/int32 narrowing.
    """
    path=Path(path)
    if not path.is_file() or path.suffix.lower()!='.glb' or path.stat().st_size>512*1024**2:
        raise ValueError('Original topology requires an admitted GLB no larger than512MiB')
    plan=plan_geometry_inspection(path) if validated_plan is None else validated_plan
    from inspect_source_uv import _array
    with GLBReader(path) as reader:
        doc=reader.doc;references=plan['mesh_nodes']
        actual={(i,n['mesh']) for i,n in enumerate(doc['nodes']) if 'mesh' in n}
        if ({(r['node'],r['mesh']) for r in references}!=actual or len(references)!=len(actual)
            or not 0<len(references)<=MAX_OBJECTS or {r['mesh'] for r in references}!=set(range(len(doc['meshes'])))):
            raise ValueError('Original topology plan omits or changes mesh instances')
        specs=[];active_triangles=0;active_rows=0
        for mesh_id,mesh in enumerate(doc['meshes']):
            rows=0;triangles=0
            for primitive in mesh['primitives']:
                if primitive.get('mode',4)!=4 or type(primitive.get('indices'))is not int:
                    raise ValueError('Original topology requires all indexed TRIANGLES primitives')
                for index,shape,kinds in ((primitive['attributes']['POSITION'],'VEC3',(5126,)),(primitive['indices'],'SCALAR',(5121,5123,5125))):
                    accessor=doc['accessors'][index];layout=reader.accessor_layout(index)
                    if (accessor['type']!=shape or accessor['componentType']not in kinds or accessor.get('normalized',False)is not False or layout[5] or layout[0]is None):
                        raise ValueError('Original topology requires supported dense accessors')
                    if shape=='VEC3':rows+=layout[1]
                    else:
                        if layout[1]%3:raise ValueError('Original topology has incomplete triangle indices')
                        triangles+=layout[1]//3
            instances=[r['node'] for r in references if r['mesh']==mesh_id]
            active_triangles+=triangles*len(instances);active_rows+=rows*len(instances)
            if active_triangles>MAX_TRIANGLES or active_rows>MAX_RENDER_ROWS:
                raise ValueError('Original topology exceeds admitted triangle/render-row work limits')
            if any(r['original_triangles']!=triangles or r['original_render_rows']!=rows for r in references if r['mesh']==mesh_id):
                raise ValueError('Original topology counts differ from the validated plan')
            specs.append((mesh_id,rows,triangles,instances))
        import numpy as np
        meshes=[]
        for mesh_id,rows,triangles,instances in specs:
            vertices=np.empty((rows,3),np.float64);faces=np.empty((triangles,3),np.int32);vertex_offset=0;face_offset=0
            mesh=doc['meshes'][mesh_id]
            for primitive in mesh['primitives']:
                positions=_array(reader,primitive['attributes']['POSITION'],'VEC3',(5126,),np)
                indices=_array(reader,primitive['indices'],'SCALAR',(5121,5123,5125),np).reshape(-1,3)
                if not np.isfinite(positions).all() or (indices.size and np.any(indices>=len(positions))):
                    raise ValueError('Original topology has nonfinite positions or out-of-range unsigned indices')
                vertices[vertex_offset:vertex_offset+len(positions)]=positions
                faces[face_offset:face_offset+len(indices)]=indices.astype(np.int32)+vertex_offset
                vertex_offset+=len(positions);face_offset+=len(indices)
            meshes.append(dict(mesh=mesh_id,name=mesh.get('name'),instance_nodes=instances,primitives=len(mesh['primitives']),
                render_rows=rows,triangles=triangles,geometric_topology=geometric_topology(vertices,faces,include_boundary_graph=True)))
        return dict(schema_version='1.1.0',status='MEASURED',mesh_definitions=len(meshes),meshes=meshes,
            limits=dict(active_triangles=MAX_TRIANGLES,active_render_rows=MAX_RENDER_ROWS,mesh_instances=MAX_OBJECTS),
            scope='Complete original POSITION/index accessor topology per unique authored mesh in local coordinates, across its material primitives; instances reference mesh records, never cross-mesh/object/instance welding.',
            semantics='All faces and POSITION rows are counted, including zero-area faces, attribute splits and unused rows. Repeated accessor rows count per primitive as in structural render counts. Exact-position incidence is diagnostic only; no manifoldness, overlap, volume, repair or validity certificate. Numerical zero-area limitations are unchanged.')


def repair_guidance(source_topology, imported_topology):
    """Interpret existing per-object counts; never admit or perform a repair."""
    def measured(topology, key):
        value=topology.get(key) if isinstance(topology,dict) and topology.get('status')=='MEASURED' else None
        return value if type(value) is int and value>=0 else None
    source_boundary=measured(source_topology,'boundary_edges');imported_boundary=measured(imported_topology,'boundary_edges')
    source_nonmanifold=measured(source_topology,'nonmanifold_edges');imported_nonmanifold=measured(imported_topology,'nonmanifold_edges')
    complete=all(v is not None for v in (source_boundary,imported_boundary,source_nonmanifold,imported_nonmanifold))
    if not complete:
        assessment='UNDETERMINED'
        next_step='Measurements are incomplete; do not infer source closure or select local repair from imported metrics.'
    elif source_boundary==0 and imported_boundary>0:
        assessment='IMPORT_ONLY_BOUNDARIES'
        next_step='Do not treat import-only boundaries as source holes or local-reconstruction candidates; compare original and imported geometry first.'
    elif source_boundary>0:
        assessment='SOURCE_BOUNDARIES_OBSERVED'
        next_step='Manually review source contours for intentional openings before considering local repair; a contour is not automatically a hole.'
    else:
        assessment='NO_BOUNDARIES_OBSERVED'
        next_step='This census does not indicate a source boundary-loop repair; review other geometry issues separately.'
    nonmanifold=('OBSERVED' if any(v is not None and v>0 for v in (source_nonmanifold,imported_nonmanifold)) else
                 'NOT_OBSERVED' if source_nonmanifold==imported_nonmanifold==0 else 'UNKNOWN')
    return dict(status='MEASURED_COMPARISON' if complete else 'INCOMPLETE_MEASUREMENTS',boundary_assessment=assessment,
        source_boundary_edges=source_boundary,imported_boundary_edges=imported_boundary,
        source_nonmanifold_edges=source_nonmanifold,imported_nonmanifold_edges=imported_nonmanifold,
        nonmanifold_issue=nonmanifold,next_step=next_step,
        nonmanifold_guidance='Non-manifold edges are a separate issue; local boundary-loop repair is not a general fix.',
        watertightness='NOT_EVALUATED',local_repair_admission='NOT_ASSESSED')


def inspect_native_geometry(source, source_sha256, directory, *, bpy):
    """Import a disposable name-tagged copy; never save, repair or export a mesh."""
    import numpy as np
    from conservative_repair import _split_glb, _write_glb
    from production_runner import digest
    if digest(source)!=source_sha256:raise ValueError('Preserved source changed before inspection')
    plan=plan_geometry_inspection(source)
    original_topology=original_accessor_topology(source,validated_plan=plan)
    doc,binary=_split_glb(source)
    for i,node in enumerate(doc['nodes']):node['name']=f'MF_INSPECT_NODE_{i:04d}'
    staging=directory/'inspection-import.glb';_write_glb(staging,doc,binary)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(staging),merge_vertices=False)
    objects=[obj for obj in bpy.context.scene.objects if obj.type=='MESH']
    expected={row['import_name']:row for row in plan['mesh_nodes']}
    if {obj.name for obj in objects}!=set(expected) or len(objects)!=len(expected):raise ValueError('Native import changed mesh-instance identity; no partial scene diagnosis returned')
    source_by_mesh={row['mesh']:row.get('geometric_topology') for row in original_topology['meshes']} if original_topology.get('status')=='MEASURED' else {}
    reports=[];depsgraph=bpy.context.evaluated_depsgraph_get()
    for obj in sorted(objects,key=lambda value:value.name):
        evaluated=obj.evaluated_get(depsgraph);mesh=evaluated.to_mesh(preserve_all_data_layers=True,depsgraph=depsgraph)
        try:
            mesh.calc_loop_triangles();vertices=np.empty((len(mesh.vertices),3),np.float64);faces=np.empty((len(mesh.loop_triangles),3),np.int32)
            mesh.vertices.foreach_get('co',vertices.ravel());mesh.loop_triangles.foreach_get('vertices',faces.ravel())
            matrix=np.asarray(obj.matrix_world,dtype=np.float64);world=vertices@matrix[:3,:3].T+matrix[:3,3]
            if not len(vertices) or not np.isfinite(world).all() or not np.isfinite(matrix).all():raise ValueError('Empty/nonfinite imported mesh or transform')
            a,b,c=(world[faces[:,i]] for i in range(3))
            topology=geometric_topology(vertices,faces,include_boundary_graph=True)
            reports.append(dict(**expected[obj.name],native_render_rows=len(vertices),native_triangles=len(faces),native_polygons=len(mesh.polygons),
                authored_render_rows=len(obj.data.vertices),authored_polygons=len(obj.data.polygons),
                importer_triangle_delta=len(faces)-expected[obj.name]['original_triangles'],importer_render_row_delta=len(vertices)-expected[obj.name]['original_render_rows'],
                matrix_world=matrix.tolist(),matrix_local=np.asarray(obj.matrix_local,dtype=np.float64).tolist(),
                world_bounds_min=world.min(axis=0).tolist(),world_bounds_max=world.max(axis=0).tolist(),
                numerical_world_zero_area_triangles=int(np.count_nonzero(np.all(np.cross(b-a,c-a)==0,axis=1))),
                geometric_topology=topology,repair_guidance=repair_guidance(source_by_mesh.get(expected[obj.name]['mesh']),topology)))
        finally:evaluated.to_mesh_clear()
    if digest(source)!=source_sha256:raise ValueError('Preserved source changed during inspection')
    triangles=sum(row['native_triangles'] for row in reports);rows=sum(row['native_render_rows'] for row in reports)
    return dict(schema_version='1.2.0',status='PARTIAL_MEASUREMENTS',scope='Blender imported/evaluated static mesh objects, instance-expanded; exact-position topology separately per object, never across objects',
        blender_version=bpy.app.version_string,mesh_instances=len(reports),unique_native_mesh_datablocks=len({obj.data.as_pointer() for obj in objects}),objects=reports,
        original_structural_counts=plan,original_accessor_topology=original_topology,native_triangles=triangles,native_render_rows=rows,
        import_staging=dict(path=str(staging),sha256=digest(staging),operation='Disposable node-name tags only; original BIN bytes copied without geometry preparation'),
        importer_triangle_delta=triangles-plan['original_active_triangles'],importer_render_row_delta=rows-plan['original_active_render_rows'],
        importer_changes_policy='Imported object metrics describe imported geometry only; the original accessor census is separate. Blender may change duplicates, degenerates or render-row storage. No original topology preservation is inferred.',
        numerical_zero_area_limitation='Local and world zero-area counts use float64 cross-product equality; rounding can classify nondegenerate triangles as zero. Not an exact degeneracy proof.',
        not_evaluated=[dict(check=check,status='NOT_EVALUATED') for check in ('cross_object_contacts_or_components','self_intersections','internal_surfaces','vertex_manifoldness','watertight_volume','global_orientability','animation_or_deformation_behavior','visual_quality','UV_or_material_quality')])
