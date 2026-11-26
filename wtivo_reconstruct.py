# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared explicit geometry-only reconstruction; no automatic backend or appearance."""
import json
import math
import os
from pathlib import Path
import subprocess
import time

MAX_FILE_BYTES=512*1024**2
MAX_LOG_BYTES=16*1024**2
MAX_COMPONENT_JOBS=128
MAX_VERTICES=2_000_000
MAX_FACES=1_250_000
MAX_DC_METRICS_BYTES=64*1024
WITNESS_RAW_ACCOUNTING='charged cumulative raw geometry, including retained closed source components under the existing budget route; not final triangles or GPU memory'


def write_json(path,value):
    stage=path.with_suffix('.tmp');stage.write_text(json.dumps(value,indent=2)+'\n');stage.replace(path)


def surface_provenance(candidate_sha256,objects,reimport,repair_mode,blender_version):
    """Report geometry provenance only after exact oriented reimport validation.

    Ordinals describe the fresh Blender triangle order of this exact artifact,
    not glTF accessor/index values or identities portable through later edits.
    Metadata inconsistencies withhold ranges, never alter geometry admission.
    """
    source={obj['name']:obj for obj in objects};records=[]
    for imported in reimport:
        count=imported['triangles'];ranges=[];local=repair_mode=='local_holes'
        if local:
            kept=source[imported['object']]['local_geometry_preservation']
            original=kept['original_faces'];generated=kept['new_faces']
            valid=original>=0 and generated>=0 and original+generated==count
            if valid:
                if original:ranges.append(dict(start=0,end=original,classification='retained_original_geometry_prefix'))
                if generated:ranges.append(dict(start=original,end=count,classification='synthetic_appended_geometry',source_face_identity='UNMATCHED_SYNTHETIC'))
        else:
            original=0;generated=count;valid=count>=0
            if count:ranges.append(dict(start=0,end=count,classification='generated_geometry',source_face_identity='UNVERIFIED_NOT_PROVEN_UNMATCHED'))
        row=dict(object=imported['object'],triangles=count,retained_original_geometry_faces=original,generated_faces=generated,
                 coverage_validated=valid,ordinal_ranges=ranges)
        if local:row['original_geometry_sha256']=kept['original_geometry_sha256']
        records.append(row)
    return dict(schema_version=1,artifact=dict(format='GLB',sha256=candidate_sha256),
        artifact_identity_policy='Exact SHA only; matrices and ordered oriented triangle corners verified by fresh Blender reimport before this record',
        index_space=dict(base=0,object_local=True,unit='triangulated ordinal on fresh Blender reimport',range_encoding='half-open [start,end)',
                         blender_version=blender_version,portability='Invalid after Simplify/edit/re-export; not a raw glTF index/accessor claim'),
        source_face_policy='Local prefix retains original geometry only; synthetic suffix has no authored source face. Volume source identity is unverified, not proven unmatched.',
        appearance_policy='Geometry-only artifact has no authored UV/materials; even retained prefix does not promise authored appearance. A later reference transfer is separate.',
        coverage_validated=all(row['coverage_validated'] for row in records),objects=records)


def _dyadic_f32(vertices,exponent,check,remaining_bytes):
    """Change units only when every F32 coordinate has an exact bitwise inverse."""
    import numpy as np
    check()
    if vertices.dtype!=np.float32 or vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices):raise ValueError('WTiVo dyadic units require nonempty F32 positions')
    required=vertices.nbytes+min(len(vertices),4096)*3*9+4096
    if remaining_bytes<required:raise MemoryError('WTiVo dyadic units exceed remaining host working-byte budget')
    result=np.empty_like(vertices);check()
    for start in range(0,len(vertices),4096):
        check();original=vertices[start:start+4096];block=result[start:start+4096]
        with np.errstate(over='raise',invalid='raise',under='ignore'):
            np.ldexp(original,exponent,out=block)
            inverse=np.ldexp(block,-exponent)
        if not np.isfinite(block).all() or not np.array_equal(inverse.view(np.uint32),original.view(np.uint32)):
            raise ValueError('WTiVo dyadic units are not finite bit-exact F32 reversible')
        check()
    return result


def wtivo_native_units(vertices,check,remaining_bytes):
    """Bound physical-area graph capacities without changing small component units."""
    import numpy as np
    check()
    if vertices.dtype!=np.float32 or vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices):raise ValueError('WTiVo native units require nonempty F32 positions')
    span=vertices.max(0).astype(np.float64)-vertices.min(0).astype(np.float64)
    diagonal=float(np.linalg.norm(span))
    if not np.isfinite(diagonal):raise ValueError('WTiVo native units have nonfinite bounds')
    record=dict(applied=False,exponent=0,scale=1.,source_diagonal_local=diagonal,
                scope='Executed native WTiVo component units only; no translation, geometry repair or fidelity guarantee')
    if diagonal<=2.:return vertices,record
    _,exponent=math.frexp(diagonal)
    scaled=_dyadic_f32(vertices,-exponent,check,remaining_bytes)
    scaled_span=scaled.max(0).astype(np.float64)-scaled.min(0).astype(np.float64)
    # Match the actual pinned _build_sparse_udf floor, not merely a nonzero span.
    if np.any(scaled_span<=1e-20):raise ValueError('WTiVo dyadic native axis span is below the native floor')
    native_diagonal=float(np.linalg.norm(scaled_span))
    if not .5<=native_diagonal<1.:raise ValueError('WTiVo dyadic native diagonal is outside [.5,1)')
    record.update(applied=True,exponent=exponent,scale=math.ldexp(1.,-exponent),native_diagonal_local=native_diagonal,
                  inverse_policy='Exact inverse power-of-two and bitwise F32 roundtrip before output geometry gates')
    check();return scaled,record


def read_native_ply(path,face_limit,*,backend='WTiVo'):
    """Admit the pinned Trimesh binary PLY layout, not an arbitrary mesh parser."""
    import numpy as np
    path=Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size>MAX_FILE_BYTES:raise ValueError(f'Missing/oversized {backend} PLY')
    with path.open('rb') as stream:
        lines=[];size=0
        for _ in range(32):
            line=stream.readline(1025);size+=len(line)
            if not line.endswith(b'\n') or len(line)>1024 or size>8192:raise ValueError(f'Invalid {backend} PLY header')
            text=line.decode('ascii').strip()
            if not text.startswith('comment '):lines.append(text)
            if text=='end_header':break
        else:raise ValueError(f'{backend} PLY header limit exceeded')
        if len(lines)!=9 or lines[:2]!=['ply','format binary_little_endian 1.0'] or lines[3:6]!=['property float x','property float y','property float z'] or lines[7:]!=['property list uchar int vertex_indices','end_header']:
            raise ValueError(f'Unsupported {backend} PLY layout')
        try:
            vertex_label,vertex_kind,nv=lines[2].split();face_label,face_kind,nf=lines[6].split()
            nv=int(nv);nf=int(nf)
        except (ValueError,TypeError) as error:raise ValueError(f'Invalid {backend} PLY counts') from error
        if (vertex_label,vertex_kind,face_label,face_kind)!=('element','vertex','element','face') or not 4<=nv<=MAX_VERTICES or not 4<=nf<=min(MAX_FACES,face_limit):raise ValueError(f'{backend} output exceeds vertex/triangle budget or is empty')
        expected=nv*12+nf*13
        if path.stat().st_size-stream.tell()!=expected:raise ValueError(f'{backend} PLY payload/count mismatch')
        vertices=np.frombuffer(stream.read(nv*12),dtype='<f4').reshape(nv,3).copy()
        face_dtype=np.dtype([('count','u1'),('indices','<i4',(3,))]);rows=np.frombuffer(stream.read(nf*13),dtype=face_dtype)
        if not np.all(rows['count']==3):raise ValueError(f'{backend} output contains non-triangle faces')
        faces=rows['indices'].astype(np.int32,copy=True)
    return vertices,faces


def validate_output(vertices,faces,*,backend='WTiVo'):
    """Independent closed oriented manifold admission; no self-intersection proof."""
    from cumesh_simplify import indexed_topology
    from inspect_native_geometry import geometric_topology
    indexed=indexed_topology(vertices,faces)
    geometric=geometric_topology(vertices,faces)
    if indexed['boundary_edges'] or any(geometric[key] for key in ('boundary_edges','nonmanifold_edges','orientation_conflict_edges','degenerate_edges','numerical_local_zero_area_triangles')):
        raise ValueError(f'{backend} output is not a closed nondegenerate oriented manifold')
    return dict(indexed=indexed,geometric=geometric,self_intersections='NOT_EVALUATED',physical_or_visual_validity='NOT_QUALIFIED')


def group_resident_bytes(group):
    """Conservative Linux RSS sum including the supervising Blender, not VRAM."""
    if not sys_linux():raise RuntimeError('WTiVo host-memory supervision currently requires Linux /proc')
    total=0;entries=list(Path('/proc').iterdir())
    if len(entries)>65536:raise RuntimeError('Process memory census exceeds its work bound')
    for entry in entries:
        if not entry.name.isdecimal():continue
        try:fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
        except (FileNotFoundError,ProcessLookupError):continue
        if int(fields[2])==group:total+=max(0,int(fields[21]))*os.sysconf('SC_PAGE_SIZE')
    return total


def sys_linux():
    import sys
    return sys.platform.startswith('linux') and os.name=='posix'


def read_dc_metrics(path):
    """Retain actual bounded child instrumentation, never substitute for PLY gates."""
    from production_runner import digest
    if not path.is_file() or path.is_symlink() or not 0<path.stat().st_size<=MAX_DC_METRICS_BYTES:raise ValueError('Missing/oversized CuMesh DC stage metrics')
    def reject_nonfinite(value):raise ValueError('Nonfinite CuMesh DC stage metrics: '+value)
    value=json.loads(path.read_text(),parse_constant=reject_nonfinite)
    if not isinstance(value,dict):raise ValueError('Invalid CuMesh DC stage metrics')
    return dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size,reported=value,
                scope='Actual owned child instrumentation; CPU Blender BVH distances and native CUDA contours, not GPU distance or kernel-only wall time; common geometry gates remain authoritative')


def dc_roi_backend(runtime,lock):
    """Load only the resolved owned version's pinned CPU witness contract."""
    import hashlib,types
    from production_runner import digest
    capability=lock.get('witness_roi',{})
    if type(capability.get('schema'))is not int or capability['schema']!=1 or capability.get('cli')!='--witness-roi <absolute-json>':
        raise RuntimeError('Owned CuMesh DC has no qualified local ROI ABI; explicitly configure the current optional runtime')
    script=Path(runtime['script']);source=script.parent;helper=source/'dc_roi_witness.py'
    if script.name!='dc_driver.py' or script.is_symlink() or helper.is_symlink() or digest(script)!=lock['source_files']['dc_driver.py']:
        raise ValueError('DC ROI driver differs from the current owned source lock')
    data=helper.read_bytes()
    if len(data)>65536 or hashlib.sha256(data).hexdigest()!=lock['source_files']['dc_roi_witness.py']:raise ValueError('DC ROI contract source hash differs')
    module=types.ModuleType('owned_dc_roi_contract');module.__file__=str(helper)
    exec(compile(data,str(helper),'exec'),module.__dict__)
    return module


def admit_dc_roi(helper,v,f,request,frame,roi_path,input_path,output,native,rv,rf,check,deadline):
    """Authenticate an open witness, never substitute it for the closed output."""
    import hashlib,zipfile
    from fractions import Fraction
    import numpy as np
    from production_runner import digest
    from wtivo_local_holes import physical_topology,geometry_hash,_exact
    check();wrapped=native['adapter_stage_metrics'];stage=wrapped['reported'];metrics=Path(wrapped['path'])
    if digest(metrics)!=wrapped['sha256'] or json.loads(metrics.read_text())!=stage:raise ValueError('DC ROI persisted metrics changed')
    if json.loads(roi_path.read_text())!=request:raise ValueError('DC ROI serialized request/frame differs')
    unsigned=dict(frame);claimed=unsigned.pop('sha256')
    if hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=claimed:raise ValueError('DC ROI frame digest differs')
    if frame['request']!=request or frame['source_geometry_sha256']!=geometry_hash(v,f):raise ValueError('DC ROI frame belongs to another source')
    loaded=helper.load_roi(roi_path,frame['resolution'])
    binding=helper.bind_source(loaded,v,f,deadline=deadline)
    if (stage.get('status')!='witness_proposal' or stage.get('scope')!='local_hole_witness_only' or stage.get('policy')!='retain_all_unsigned_shells'
            or stage.get('resolution')!=frame['resolution'] or stage.get('band')!=1 or stage.get('source_vertices')!=len(v) or stage.get('source_triangles')!=len(f) or stage.get('source_preserved')is not True
            or stage.get('native_sha256')!=native['native_sha256'] or stage.get('input_sha256')!=digest(input_path)
            or stage.get('roi_sha256')!=digest(roi_path) or stage.get('output_sha256')!=digest(output) or stage.get('output_bytes')!=output.stat().st_size
            or stage.get('roi_source_binding')!=binding or stage.get('protected_slab_voxels')!=2 or stage.get('boundary_guard_voxels')!=8):
        raise ValueError('DC ROI native/OBJ/PLY/source binding identity differs')
    grid=stage['native'];pitch=request['extent']/stage['resolution']
    if (grid.get('witness_only')is not True or grid.get('roi_center')!=np.asarray(request['center'],np.float32).tolist()
            or grid.get('roi_extent')!=request['extent'] or grid.get('voxel')!=pitch or grid.get('iso')!=pitch
            or grid.get('field_iso_f32')!=float(np.float32(pitch)) or stage['output'].get('vertices')!=len(rv) or stage['output'].get('triangles')!=len(rf)):
        raise ValueError('DC ROI effective grid/voxel/output identity differs')
    # Verify the actual full-source BVH payload, not just its reported row count.
    source=metrics.parent/'distance-worker/source.npz';ready=stage['distance_ready'];worker=stage['worker_result']
    if ready.get('source_vertices')!=len(v) or ready.get('source_faces')!=len(f) or ready.get('source_sha256')!=digest(source) or worker.get('status')!='CLOSED' or worker.get('source_vertices')!=len(v) or worker.get('source_faces')!=len(f):raise ValueError('DC ROI full-source BVH identity differs')
    if source.is_symlink()or source.stat().st_size>v.nbytes+f.nbytes+4096:raise ValueError('DC ROI source payload is not bounded')
    with zipfile.ZipFile(source)as archive:
        expected={'vertices':v,'faces':f}
        if set(archive.namelist())!={x+'.npy'for x in expected}:raise ValueError('DC ROI source payload members differ')
        for key,array in expected.items():
            entry=archive.getinfo(key+'.npy')
            if entry.compress_type!=zipfile.ZIP_STORED or entry.file_size>array.nbytes+512:raise ValueError('DC ROI source payload size differs')
            with archive.open(entry)as stream:
                if np.lib.format.read_magic(stream)!=(1,0):raise ValueError('DC ROI source array version differs')
                shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
                if shape!=array.shape or dtype!=array.dtype or fortran:raise ValueError('DC ROI source array header differs')
    with np.load(source,allow_pickle=False)as arrays:
        if arrays['vertices'].tobytes()!=v.tobytes()or arrays['faces'].tobytes()!=f.tobytes():raise ValueError('DC ROI BVH source rows/faces differ')
    if digest(source)!=ready['source_sha256']:raise ValueError('DC ROI source payload changed')
    topology=physical_topology(rv,rf,closed=False);edges=np.stack((rf,np.roll(rf,-1,axis=1)),axis=-1).reshape(-1,2)
    unique,count=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True);boundary=unique[count==1]
    certificate=stage['output'].get('boundary_certificate',{})
    if not 0<len(boundary)<=32768 or len(boundary)!=stage['output'].get('boundary_edges') or certificate.get('boundary_outside_certificate')is not True or certificate.get('guard_voxels')!=8:raise ValueError('DC ROI open boundary certificate differs')
    # Independently recheck ALL segments against the full footprint extrusion;
    # native slot ownership was checked by the pinned child, not inferred here.
    center=_exact(request['center']);direction=_exact(request['normal']);p=Fraction(pitch)
    radius=max(sum(abs(a-b)for a,b in zip(_exact(point),center))for point in request['protected_polygon'])+2*p*sum(abs(a)for a in direction);guard=radius+8*p;minimum=None
    for i,(a,b)in enumerate(boundary):
        if i%128==0:check()
        x=tuple(a-b for a,b in zip(_exact(rv[a]),center));d=tuple(a-b for a,b in zip(_exact(rv[b]),_exact(rv[a])));den=sum(a*a for a in d)
        if den<=0:raise ValueError('DC ROI boundary segment collapsed')
        t=max(0,min(1,-sum(a*b for a,b in zip(x,d))/den));distance=sum((a+t*b)**2 for a,b in zip(x,d))
        if distance<=guard**2:raise ValueError('DC ROI boundary intersects the protected full slab plus margin')
        minimum=distance if minimum is None else min(minimum,distance)
    if certificate.get('enclosing_radius')!=float(radius)or certificate.get('minimum_segment_distance')!=math.sqrt(float(minimum)):raise ValueError('DC ROI persisted separation certificate differs')
    check();return dict(raw_topology=topology,roi_source_binding=binding,boundary_certificate=certificate,
        input_sha256=digest(input_path),roi_sha256=digest(roi_path),output_sha256=digest(output),native_geometry_sha256=geometry_hash(rv,rf),
        identity_scope='Bounded open full-source native witness only; not a closed proposal or final GLB')


def run_native(input_path,output_path,directory,config,runtime):
    """Child remains in the common production runner's owned process group."""
    from production_runner import digest,parent_death_command
    dc=config.get('operation')=='reconstruct-cumesh-dc';backend='CuMesh DC' if dc else 'WTiVo'
    if not sys_linux():raise RuntimeError(f'{backend} process-group memory supervision is not qualified on this platform')
    python=Path(runtime['python']);script=Path(runtime['script']);env=runtime['env']
    if not python.is_absolute() or not python.is_file() or not script.is_absolute() or not script.is_file() or script.suffix!='.py':raise ValueError(f'Owned {backend} runtime returned invalid paths')
    if not isinstance(env,dict) or len(env)>256 or any(not isinstance(k,str) or not isinstance(v,str) for k,v in env.items()) or len(json.dumps(env).encode())>128*1024:raise ValueError(f'Owned {backend} runtime returned invalid environment')
    if dc:
        native_hash=runtime.get('native_sha256');blender=Path(config['_blender_binary_path']);seconds=min(180,int(config['_geometry_deadline']-time.monotonic()))
        if not isinstance(native_hash,str) or len(native_hash)!=64 or any(c not in '0123456789abcdef' for c in native_hash):raise ValueError('Owned CuMesh DC native hash is invalid')
        if not blender.is_absolute() or not blender.is_file():raise ValueError('Actual caller Blender binary is invalid')
        if seconds<1:raise TimeoutError('CuMesh DC has no remaining native deadline')
        metrics_path=directory/'metrics.json'
        command=[str(python),'-I','-B',str(script),'--input',str(input_path),'--output',str(output_path),
            '--metrics',str(metrics_path),'--resolution',str(config['resolution']),'--band','1','--seconds',str(seconds),
            '--blender',str(blender),'--parent-pid',str(os.getpid()),'--threads',str(config['cpu_threads']),
            '--native-sha256',native_hash,'--retain-all']
        if '_dc_witness_roi'in config:
            roi_path=Path(config['_dc_witness_roi'])
            if not roi_path.is_absolute()or roi_path.is_symlink()or not roi_path.is_file()or roi_path.stat().st_size>16384:raise ValueError('Invalid bounded DC witness ROI path')
            roi_hash=digest(roi_path);command+=['--witness-roi',str(roi_path)]
    else:
        command=[str(python),'-I','-B',str(script),'--input',str(input_path),'--output',str(output_path),
            '--input-res',str(config['input_resolution']),'--final-res',str(config['final_resolution']),
            '--proxy-points',str(config['proxy_points']),'--threads',str(config['cpu_threads']),
            '--max_rounds',str(config['max_rounds']),'--faithc_component_mode','keep_all',
            '--thick_band_voxels','3','--thin_band_voxels','3','--thin_iso_vox','0',
            '--gpupr_local_steps','8','--global_relabel_period','1024']
    # Blender's sys.executable is Blender, so use the resolved owned runtime Python.
    launch_command=command if dc else parent_death_command(command,python=python)
    source_hash=digest(input_path);group=os.getpgrp();log_path=directory/'native.log';maximum_rss=0;started=time.monotonic()
    record=dict(command=command,launch_command=launch_command,
        parent_death_guard='Existing DC driver guard' if dc else 'Linux direct child SIGKILL before exec',
        process_group=group,keep_all=True,component_pruning=False,status='starting',
        memory_limit_bytes=config['memory_budget_mib']*1024**2,memory_scope='Conservative RSS sum of owned Blender process group; shared pages may be counted twice; GPU VRAM is not included',
        resource_checks='Cooperative 0.1s checks; no strict native per-step memory/allocation guarantee')
    if dc:record.update(backend='CuMesh DC',distance_backend='Blender CPU BVH',contour_backend='Owned native CUDA connected contours',
                        unsigned_shell_policy='Retain all inner and outer offset sheets; not a signed solid',native_sha256=native_hash,
                        resolution_scope='128 has one verified local-hole repair case; resource admission is not general native/quality qualification')
    if dc and '_dc_witness_roi'in config:record.update(roi_sha256=roi_hash,unsigned_shell_policy='Open bounded full-source witness; not closed reconstruction or final GLB')
    write_json(directory/'invocation.json',record)
    # No new session/group: reap only our direct child on errors, never our group
    # leader. The common runner retains descendant/cancellation group cleanup.
    with log_path.open('xb') as log:
        child=subprocess.Popen(launch_command,cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            record.update(status='running',pid=child.pid);write_json(directory/'invocation.json',record)
            while True:
                if time.monotonic()>=config['_geometry_deadline']:raise TimeoutError(f'{backend} total reconstruction deadline exceeded')
                if os.getpgid(child.pid)!=group:raise RuntimeError(f'{backend} escaped its owned production process group')
                maximum_rss=max(maximum_rss,group_resident_bytes(group))
                if maximum_rss>record['memory_limit_bytes']:raise MemoryError(f'{backend} owned process-group host RSS exceeded the declared budget')
                if log_path.stat().st_size>MAX_LOG_BYTES or (output_path.exists() and output_path.stat().st_size>MAX_FILE_BYTES):raise ValueError(f'{backend} output/log exceeded resource limit')
                if dc and metrics_path.exists() and metrics_path.stat().st_size>MAX_DC_METRICS_BYTES:raise ValueError('CuMesh DC stage metrics exceeded resource limit')
                code=child.poll()
                if code is not None:break
                time.sleep(.1)
            if code:raise RuntimeError(f'{backend} native command failed ({code}); see {log_path}')
            if digest(input_path)!=source_hash:raise ValueError(f'{backend} changed its geometry input')
            if dc and '_dc_witness_roi'in config and digest(roi_path)!=roi_hash:raise ValueError('DC witness changed its ROI request')
            if dc:record['adapter_stage_metrics']=read_dc_metrics(metrics_path)
            record['status']='completed'
        except BaseException as error:
            record.update(status='failed',error=f'{type(error).__name__}: {error}')
            cleanup_started=time.monotonic()
            cleanup=dict(scope='Only this direct owned Popen child; outer runner retains descendant group fallback',
                         status='started',terminate_wait_seconds=3,kill_wait_seconds=5)
            record['direct_child_cleanup']=cleanup
            try:
                if child.poll() is None:
                    try:
                        child.terminate();cleanup['terminate_requested']=True
                    except ProcessLookupError:cleanup['terminate_missing']=True
                try:
                    code=child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        child.kill();cleanup['kill_requested']=True
                    except ProcessLookupError:cleanup['kill_missing']=True
                    code=child.wait(timeout=5)
                cleanup.update(status='reaped',returncode=code)
            except BaseException as cleanup_error:
                cleanup.update(status='failed',error=f'{type(cleanup_error).__name__}: {cleanup_error}',
                               returncode=child.returncode)
            finally:cleanup['elapsed_seconds']=time.monotonic()-cleanup_started
            raise
        finally:
            record.update(elapsed_seconds=time.monotonic()-started,maximum_group_RSS_bytes=maximum_rss,returncode=child.returncode,
                          native_log=str(log_path),native_log_bytes=log_path.stat().st_size)
            write_json(directory/'invocation.json',record)
    return record


def write_obj(path,vertices,faces,deadline):
    with path.open('x') as stream:
        for i,v in enumerate(vertices):
            if i%4096==0 and time.monotonic()>=deadline:raise TimeoutError('Reconstruction preparation deadline exceeded')
            stream.write('v %.17g %.17g %.17g\n'%tuple(v))
        for i,f in enumerate(faces):
            if i%4096==0 and time.monotonic()>=deadline:raise TimeoutError('Reconstruction preparation deadline exceeded')
            stream.write('f %d %d %d\n'%tuple(int(k)+1 for k in f))
    if path.stat().st_size>MAX_FILE_BYTES:raise ValueError('WTiVo OBJ exceeds input byte limit')


def source_component_selection(vertices,faces,components,matrix,percent,check):
    """Explicit source-only selection per object; zero is an area-free keep-all."""
    import hashlib
    import numpy as np
    if type(percent) not in (int,float) or not math.isfinite(percent) or not 0<=percent<=5:raise ValueError('Invalid source component area percentage')
    records=[dict(component_index=i,source_face_count=len(group),source_face_index_bounds=[min(group),max(group)],
        source_face_ids_sha256=hashlib.sha256(np.asarray(group,np.int32).tobytes()).hexdigest(),retained=True,
        world_surface_area=None,threshold_world_area=None,largest_world_area=None,threshold_percent=percent,
        area_scope='NOT_COMPUTED: zero disables source component filtering') for i,group in enumerate(components)]
    if percent==0:return records
    linear=np.asarray(matrix,np.float64)[:3,:3]
    if linear.shape!=(3,3) or not np.isfinite(linear).all():raise ValueError('Invalid source area transform')
    areas=[]
    for group in components:
        blocks=[]
        for start in range(0,len(group),4096):
            check();tri=np.asarray(vertices[faces[np.asarray(group[start:start+4096],np.int32)]],np.float64)
            with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                cross=np.cross((tri[:,1]-tri[:,0])@linear.T,(tri[:,2]-tri[:,0])@linear.T)
                scale=np.max(np.abs(cross),axis=1)
                area=.5*scale*np.sqrt(np.sum((cross/scale[:,None])**2,axis=1))
            if not np.isfinite(area).all() or np.any(area<=0):raise ValueError('Source component area is zero, nonfinite or unproved')
            blocks.append(math.fsum(area.tolist()))
        value=math.fsum(blocks)
        if not math.isfinite(value) or value<=0:raise ValueError('Source component area is zero, nonfinite or unproved')
        areas.append(value)
    if not areas:raise ValueError('No source component area')
    largest=max(areas);threshold=largest*(percent/100.)
    if not math.isfinite(threshold) or threshold<=0:raise ValueError('Source component area threshold is unproved')
    for record,area in zip(records,areas):
        record.update(world_surface_area=area,threshold_world_area=threshold,largest_world_area=largest,retained=not area<threshold,
            area_scope='Float64 transformed edge cross-products and stable sums; translation excluded; strict below per-object largest, not a noise classifier')
    check();return records


def reconstruct(source,source_sha256,directory,config,*,bpy,import_meshes,mesh_arrays,make_mesh,export_meshes):
    import numpy as np
    from mathutils import Matrix
    from geometric_edge_components import analyze_geometric_edges
    from conservative_repair import _split_glb, _write_glb
    from inspect_native_geometry import geometric_topology
    from production_runner import digest
    from reconstruction_fidelity import component as fidelity_component,parent_bounds,notice as fidelity_notice
    operation=config.get('operation','reconstruct-wtivo')
    if operation not in ('reconstruct-wtivo','reconstruct-cumesh-dc'):raise ValueError('Unsupported reconstruction operation')
    dc=operation=='reconstruct-cumesh-dc';backend='CuMesh DC' if dc else 'WTiVo';prefix='cumesh-dc' if dc else 'wtivo'
    local=config.get('repair_mode','volume')=='local_holes'
    if type(config.get('repair_mode','volume'))is not str or config.get('repair_mode','volume') not in ('volume','local_holes'):raise ValueError('Invalid reconstruction repair_mode')
    if local and config.get('drop_below_largest_area_percent',0.)!=0:raise ValueError('Local holes requires drop0')
    if dc and local and config.get('shell_policy','keep_all')!='keep_all':raise ValueError('DC local holes requires keep_all; verified_outer is volume-only')
    final_validate=validate_output
    if local:
        from wtivo_local_holes import prepare,finish,physical_topology,geometry_hash,dc_voxel,witness_crop,dc_witness_roi,DCContactRejected,_dc_world_voxel
        final_validate=lambda v,f,**kwargs:physical_topology(v,f)
    if dc:
        from cumesh_dc_runtime import resolve_runtime
        config=dict(config,_blender_binary_path=str(Path(bpy.app.binary_path).resolve(strict=True)))
    else:from wtivo_runtime import resolve_runtime
    started=time.monotonic();candidate=directory/'candidate.glb';report_path=directory/(prefix+'-reconstruction.json')
    if digest(source)!=source_sha256:raise ValueError(f'Preserved source changed before {backend}')
    runtime=resolve_runtime(extension=Path(__file__).resolve().parent)
    provenance=runtime.get('provenance',{})
    if len(json.dumps(provenance).encode())>65536:raise ValueError(f'{backend} runtime provenance exceeds its bound')
    report=dict(status='running',operation=operation,source_sha256=source_sha256,runtime=provenance,
        appearance='INVALIDATED: geometry-only output has no UV/materials; new surfaces have no inferred HIGH correspondence',
        repair_mode=config.get('repair_mode','volume'),
        policy=(('Local clean convex small boundary loops only; closed components retained; actual CuMesh unsigned straddling pairs witness approximate synthetic midpoints, no fallback' if dc else 'Local clean convex small boundary loops only; closed source components retained; actual WTiVo witnesses bound appended geometry, no fallback') if local else
                ('Unsigned band=1; keep_all default. Explicit verified_outer removes only a proven nested generated inward sheet per closed outward source component, or refuses. Not signed-solid reconstruction'
                 if dc else 'Selected exact-position shared-edge source components are processed separately with native keep_all; no automatic Optimize fallback')),
        source_component_policy='Zero area percentage retains every source component without area computation. Opt-in drops only components strictly below the per-object largest world surface area percentage, before native processing; never classifies noise or drops an object/largest/tie. Native keep_all describes each selected staged component, not this explicit source policy.',
        maximum_component_jobs=MAX_COMPONENT_JOBS,parameters={k:v for k,v in config.items() if not k.startswith('_') and k!='plan'},objects=[],components=[])
    write_json(report_path,report)
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True);staging={}
        # This route intentionally discards appearance. Avoid decoding unrelated
        # textures merely to extract geometry, while preserving all P/F/nodes.
        doc,binary=_split_glb(source)
        for mesh in doc['meshes']:
            for primitive in mesh['primitives']:
                primitive['attributes']={'POSITION':primitive['attributes']['POSITION']};primitive.pop('material',None)
        for key in ('materials','textures','images','samplers'):doc.pop(key,None)
        geometry_source=directory/'geometry-source.glb';_write_glb(geometry_source,doc,binary)
        originals=sorted(import_meshes(geometry_source,False,directory,staging=staging,prepare=False),key=lambda o:o.name)
        if sorted(o.name for o in originals)!=sorted(config['plan']['objects']):raise ValueError(f'{backend} import changed object namespace')
        imported=sum(len(mesh_arrays(o.data)[1]) for o in originals)
        if imported!=config['plan']['triangles']:raise ValueError(f'{backend} import changed triangle count; diagnose/repair the original explicitly first')
        outputs=[];total_faces=0;total_vertices=0;jobs=0;native_jobs=0;raw_total_faces=0;raw_total_vertices=0;shell_work={};fidelity_work={}
        def check_resources():
            if time.monotonic()>=config['_geometry_deadline']:raise TimeoutError(f'{backend} total reconstruction deadline exceeded')
            if group_resident_bytes(os.getpgrp())>config['memory_budget_mib']*1024**2:raise MemoryError('Reconstruction host RSS budget exceeded')
        check_resources()
        context=None;context_offsets={}
        if local:
            # Include every original object/component in patch contact checks;
            # no welding or contacts with unrelated components are allowed.
            context_parts=[];context_offset=0
            for obj in originals:
                cv,cf,_=mesh_arrays(obj.data);cm=np.asarray(obj.matrix_world,np.float64)
                world=np.asarray(cv,np.float64)@cm[:3,:3].T+cm[:3,3]
                if not np.isfinite(world).all():raise ValueError('WTiVo local source context is nonfinite')
                context_offsets[obj.name]=context_offset;context_offset+=len(cf)
                context_parts.append(world[cf]);check_resources()
            context=np.concatenate(context_parts);check_resources()
        for original in originals:
            vertices,faces,_=mesh_arrays(original.data);matrix=np.asarray(original.matrix_world,np.float64)
            if original.modifiers or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3,:3]))<1e-15:raise ValueError('Unsupported reconstruction transform/modifier')
            before=geometric_topology(vertices,faces);groups=analyze_geometric_edges(vertices,faces.tolist())
            if groups['assessment']=='not_assessed':raise ValueError(f'{backend} source component work bound exceeded')
            components=groups['components'];jobs+=len(components)
            if jobs>MAX_COMPONENT_JOBS:raise ValueError(f'{backend} source exceeds128 component jobs; no component is silently dropped')
            selection=source_component_selection(vertices,faces,components,matrix,config.get('drop_below_largest_area_percent',0.),check_resources)
            entry=dict(name=original.name,matrix_world=matrix.tolist(),before=before,source_components=len(components),source_component_selection=selection);report['objects'].append(entry);write_json(report_path,report)
            combined_v=[];combined_f=[];offset=0;component_fidelity=[]
            local_vertices=[];local_faces=[];local_vertex_count=0;local_face_count=0;component_vertex_count=0
            for group,decision in zip(components,selection):
                check_resources()
                if not decision['retained']:continue
                part=directory/f'{prefix}-{len(report["components"]):03d}';part.mkdir()
                selected=faces[np.asarray(group,np.int32)];rows,inverse=np.unique(selected,return_inverse=True)
                input_v=vertices[rows];input_f=inverse.astype(np.int32).reshape(-1,3);input_path=part/'input.obj';output=part/'output.ply'
                local_plan=prepare(input_v,input_f,matrix,check_resources) if local else None
                if dc and local:
                    # Unit voxel tests numerical frame admission only, before native work.
                    for loop in local_plan['loops']:_dc_world_voxel({'local_voxel':1.},matrix,np.asarray(loop['normal']))
                retained_closed=local and not local_plan['loops']
                native_input_v=input_v;native_units=None
                if not dc and not retained_closed:
                    native_input_v,native_units=wtivo_native_units(input_v,check_resources,
                        config['memory_budget_mib']*1024**2-group_resident_bytes(os.getpgrp()))
                    decision['native_units']=native_units;write_json(report_path,report)
                write_obj(input_path,native_input_v,input_f,config['_geometry_deadline']);del native_input_v
                if retained_closed:
                    native=dict(status='retained_closed_without_native',keep_all=True,component_pruning=False)
                    out_v,out_f=input_v.copy(),input_f.copy()
                else:
                    if local and native_jobs>=MAX_COMPONENT_JOBS:raise ValueError(f'{backend} local native job budget exceeded')
                    native=run_native(input_path,output,part,config,runtime)
                    native_jobs+=1
                    out_v,out_f=read_native_ply(output,config['max_output_triangles']-raw_total_faces,backend=backend)
                    if native_units is not None and native_units['applied']:
                        out_v=_dyadic_f32(out_v,native_units['exponent'],check_resources,
                            config['memory_budget_mib']*1024**2-group_resident_bytes(os.getpgrp()))
                outer=dc and not local and config.get('shell_policy','keep_all')=='verified_outer'
                if outer:
                    from dc_shell_policy import proof_capacity,verified_outer
                    check_resources()
                    proof_capacity(len(input_f),len(out_f),len(input_v),len(out_v),remaining_bytes=config['memory_budget_mib']*1024**2-group_resident_bytes(os.getpgrp()))
                topology=(final_validate if retained_closed else validate_output)(out_v,out_f,backend=backend);raw_total_faces+=len(out_f);raw_total_vertices+=len(out_v);check_resources()
                if raw_total_faces>config['max_output_triangles'] or raw_total_vertices>MAX_VERTICES:raise ValueError(f'{backend} aggregate raw output exceeds declared work/size budget')
                shell_record=None;raw_topology=topology;local_record=None
                if local:
                    witness={}
                    if dc:
                        witness['witness_policy']='dc_unsigned_midpoint'
                        if not retained_closed:
                            stage=native['adapter_stage_metrics']['reported']
                            if stage.get('input_sha256')!=digest(input_path) or stage.get('output_sha256')!=digest(output):raise ValueError('DC local native part hashes differ')
                            witness['native_voxel']=dc_voxel(input_v,input_f,config['resolution'],stage)
                            attempted=set()
                            def refine_dc_witness(loop_index,loop,rejection):
                                nonlocal native_jobs,raw_total_faces,raw_total_vertices
                                check_resources()
                                if not isinstance(rejection,DCContactRejected):raise ValueError('DC ROI requires a contact-only rejection')
                                if loop_index in attempted:raise RuntimeError('DC local witness refinement repeated')
                                if native_jobs>=MAX_COMPONENT_JOBS or config['max_output_triangles']-raw_total_faces<4 or MAX_VERTICES-raw_total_vertices<4:raise ValueError('DC local refinement aggregate native budget exhausted')
                                attempted.add(loop_index);refined=dict(loop_index=loop_index,status='preparing',trigger_class=type(rejection).__name__,whole_component_contact_rejection=str(rejection))
                                decision.setdefault('local_witness_refinements',[]).append(refined)
                                def save_dc_work():
                                    report['local_witness_work']=dict(native_calls=native_jobs,maximum_native_calls=MAX_COMPONENT_JOBS,
                                        raw_triangles=raw_total_faces,raw_vertices=raw_total_vertices,
                                        maximum_raw_triangles=config['max_output_triangles'],maximum_raw_vertices=MAX_VERTICES,raw_accounting=WITNESS_RAW_ACCOUNTING)
                                    write_json(report_path,report)
                                save_dc_work()
                                try:
                                    lock=json.loads((Path(__file__).resolve().parent/'resources/cumesh-dc/runtime-lock.json').read_text())
                                    helper=dc_roi_backend(runtime,lock)
                                    request,roi_voxel,frame=dc_witness_roi(input_v,input_f,matrix,loop,config['resolution'],check_resources)
                                    refined.update(frame=frame,native_voxel=roi_voxel,effective_input_resolution=config['resolution'],effective_final_resolution=config['resolution'])
                                    folder=part/f'local-witness-{loop_index:03d}';folder.mkdir();roi_path=folder/'roi.json';roi_output=folder/'output.ply';write_json(roi_path,request)
                                    refined.update(input_sha256=digest(input_path),input_path=str(input_path),roi_sha256=digest(roi_path),output_path=str(roi_output),status='native_running')
                                    native_jobs+=1;save_dc_work()
                                    print(json.dumps(dict(event='cumesh_dc_local_witness_refinement',loop=loop_index,resolution=config['resolution'],source='full_original_component')),flush=True)
                                    refined['native']=run_native(input_path,roi_output,folder,dict(config,_dc_witness_roi=str(roi_path)),runtime)
                                    rv,rf=read_native_ply(roi_output,config['max_output_triangles']-raw_total_faces,backend=backend)
                                    raw_total_faces+=len(rf);raw_total_vertices+=len(rv);save_dc_work()
                                    if raw_total_vertices>MAX_VERTICES:raise ValueError('DC local refinement aggregate vertex budget exceeded')
                                    check_resources()
                                    remaining=config['memory_budget_mib']*1024**2-group_resident_bytes(os.getpgrp())
                                    if remaining<1024*(len(rv)+len(rf))+2*(input_v.nbytes+input_f.nbytes):raise MemoryError('DC local witness admission headroom exceeded')
                                    refined.update(admit_dc_roi(helper,input_v,input_f,request,frame,roi_path,input_path,roi_output,refined['native'],rv,rf,check_resources,config['_geometry_deadline']))
                                    check_resources()
                                    if config['memory_budget_mib']*1024**2-group_resident_bytes(os.getpgrp())<24*len(rv)+72*len(rf):raise MemoryError('DC witness world triangle headroom exceeded')
                                    triangles=(rv.astype(np.float64)@matrix[:3,:3].T+matrix[:3,3])[rf];del rv,rf
                                    refined['status']='native_open_witness_ready';save_dc_work();return triangles,refined
                                except BaseException as error:
                                    refined.update(status='failed',error=f'{type(error).__name__}: {error}');save_dc_work();raise
                            witness['refine']=refine_dc_witness
                    elif not retained_closed:
                        attempted=set()
                        def refine_witness(loop_index,loop,rejection):
                            nonlocal native_jobs,raw_total_faces,raw_total_vertices
                            check_resources()
                            if loop_index in attempted:raise RuntimeError('WTiVo local witness refinement repeated')
                            if native_jobs>=MAX_COMPONENT_JOBS or config['max_output_triangles']-raw_total_faces<4 or MAX_VERTICES-raw_total_vertices<4:raise ValueError('WTiVo local refinement aggregate native budget exhausted')
                            attempted.add(loop_index);stage=dict(loop_index=loop_index,status='preparing',trigger_class=type(rejection).__name__,whole_component_contact_rejection=str(rejection))
                            decision.setdefault('local_witness_refinements',[]).append(stage)
                            def save_work():
                                report['local_witness_work']=dict(native_calls=native_jobs,maximum_native_calls=MAX_COMPONENT_JOBS,
                                    raw_triangles=raw_total_faces,raw_vertices=raw_total_vertices,
                                    maximum_raw_triangles=config['max_output_triangles'],maximum_raw_vertices=MAX_VERTICES,raw_accounting=WITNESS_RAW_ACCOUNTING)
                                write_json(report_path,report)
                            last_check=0.
                            def check_crop():
                                nonlocal last_check
                                now=time.monotonic()
                                if now>=config['_geometry_deadline']:raise TimeoutError('WTiVo local witness total deadline exceeded')
                                if now-last_check>=.1:check_resources();last_check=now
                            save_work()
                            try:
                                cv,cf,crop=witness_crop(input_v,input_f,matrix,loop,config['input_resolution'],check_crop,
                                    face_limit=MAX_FACES,remaining_bytes=config['memory_budget_mib']*1024**2-group_resident_bytes(os.getpgrp()))
                                stage['source_crop']=crop;check_resources()
                                cv,units=wtivo_native_units(cv,check_resources,config['memory_budget_mib']*1024**2-group_resident_bytes(os.getpgrp()))
                                stage['native_units']=units;folder=part/f'local-witness-{loop_index:03d}';folder.mkdir();roi_input=folder/'input.obj';roi_output=folder/'output.ply'
                                write_obj(roi_input,cv,cf,config['_geometry_deadline'])
                                # Verify the real serialized F32 corners and exact face order, not only builder arrays.
                                with roi_input.open() as stream:
                                    for row in cv:
                                        check_crop();tokens=stream.readline().split()
                                        if tokens[:1]!=['v'] or np.asarray(tokens[1:],np.float32).tobytes()!=row.tobytes():raise ValueError('WTiVo witness OBJ positions differ')
                                    for row in cf:
                                        check_crop();tokens=stream.readline().split()
                                        if tokens!=['f',*(str(int(x)+1) for x in row)]:raise ValueError('WTiVo witness OBJ face order differs')
                                    if stream.read():raise ValueError('WTiVo witness OBJ has unexpected data')
                                del cv,cf
                                stage.update(input_sha256=digest(roi_input),input_path=str(roi_input),output_path=str(roi_output),status='native_running')
                                native_jobs+=1;save_work()
                                roi_config=dict(config,input_resolution=crop['input_resolution'])
                                stage.update(effective_input_resolution=roi_config['input_resolution'],effective_final_resolution=roi_config['final_resolution']);save_work()
                                print(json.dumps(dict(event='wtivo_local_witness_refinement',loop=loop_index,input_resolution=roi_config['input_resolution'],final_resolution=roi_config['final_resolution'])),flush=True)
                                stage['native']=run_native(roi_input,roi_output,folder,roi_config,runtime)
                                rv,rf=read_native_ply(roi_output,config['max_output_triangles']-raw_total_faces)
                                raw_total_faces+=len(rf);raw_total_vertices+=len(rv);save_work()
                                if raw_total_vertices>MAX_VERTICES:raise ValueError('WTiVo local refinement aggregate vertex budget exceeded')
                                if units['applied']:rv=_dyadic_f32(rv,units['exponent'],check_resources,config['memory_budget_mib']*1024**2-group_resident_bytes(os.getpgrp()))
                                stage.update(output_sha256=digest(roi_output),native_geometry_sha256=geometry_hash(rv,rf),raw_topology=validate_output(rv,rf))
                                check_resources()
                                if config['memory_budget_mib']*1024**2-group_resident_bytes(os.getpgrp())<24*len(rv)+72*len(rf):raise MemoryError('WTiVo witness world triangle headroom exceeded')
                                triangles=(rv.astype(np.float64)+np.asarray(crop['center_world']))[rf];del rv,rf
                                stage['status']='native_witness_ready';save_work();check_resources();return triangles,stage
                            except BaseException as error:
                                stage.update(status='failed',error=f'{type(error).__name__}: {error}');save_work();raise
                        witness['refine']=refine_witness
                    out_v,out_f,local_record=finish(input_v,input_f,matrix,local_plan,out_v,out_f,check_resources,
                        context=context,owned_faces=[context_offsets[original.name]+int(i) for i in group],**witness)
                    topology=final_validate(out_v,out_f,backend=backend);check_resources()
                    if len(out_f)>len(input_f):
                        patch_world=np.asarray(out_v,np.float64)@matrix[:3,:3].T+matrix[:3,3]
                        context=np.concatenate((context,patch_world[out_f[len(input_f):]]));check_resources()
                if outer:
                    # Only the proof's hot callbacks throttle /proc scans; every
                    # callback still checks the actual total deadline.
                    last_proof_rss_check=0.
                    def check_proof(force=False):
                        nonlocal last_proof_rss_check
                        now=time.monotonic()
                        if now>=config['_geometry_deadline']:raise TimeoutError(f'{backend} total reconstruction deadline exceeded')
                        if force or now-last_proof_rss_check>=.1:
                            check_resources();last_proof_rss_check=time.monotonic()
                    check_proof(True)
                    try:
                        out_v,out_f,shell_record=verified_outer(input_v,input_f,out_v,out_f,work=shell_work,check=check_proof,
                            remaining_bytes=config['memory_budget_mib']*1024**2-group_resident_bytes(os.getpgrp()))
                    finally:check_proof(True)
                    topology=validate_output(out_v,out_f,backend=backend);check_resources()
                fidelity=fidelity_component(input_v,input_f,out_v,out_f,matrix,check_resources,fidelity_work,local=local_record,
                    remaining_bytes=max(0,config['memory_budget_mib']*1024**2-group_resident_bytes(os.getpgrp())))
                component_fidelity.append(fidelity)
                total_faces+=len(out_f);total_vertices+=len(out_v)
                if total_faces>config['max_output_triangles'] or total_vertices>MAX_VERTICES:raise ValueError(f'{backend} aggregate final output exceeds declared budget')
                if local:
                    # Preserve the entire object's original rows and triangles,
                    # not merely its compact per-component corner geometry.
                    if out_v[:len(input_v)].tobytes()!=input_v.tobytes() or out_f[:len(input_f)].tobytes()!=input_f.tobytes():raise ValueError('WTiVo local component changed source geometry')
                    extra_v=out_v[len(input_v):];extra_f=out_f[len(input_f):]
                    first_vertex=len(vertices)+local_vertex_count;first_face=len(faces)+local_face_count
                    mapping=np.r_[rows,np.arange(first_vertex,first_vertex+len(extra_v),dtype=np.int32)]
                    local_vertices.append(extra_v);local_faces.append(mapping[extra_f].astype(np.int32))
                    local_record.update(object_new_vertex_ids_before_serialization=list(range(first_vertex,first_vertex+len(extra_v))),
                        object_new_face_ids_before_serialization=list(range(first_face,first_face+len(extra_f))))
                    local_vertex_count+=len(extra_v);local_face_count+=len(extra_f);component_vertex_count+=len(out_v)
                else:
                    combined_v.append(out_v);combined_f.append(out_f+offset);offset+=len(out_v)
                import hashlib
                report['components'].append(dict(object=original.name,source_face_count=len(group),source_face_index_bounds=[min(group),max(group)],
                    source_face_ids_sha256=hashlib.sha256(np.asarray(group,np.int32).tobytes()).hexdigest(),
                    input_sha256=digest(input_path),output_sha256=None if retained_closed else digest(output),native=native,after=topology,
                    raw_after=raw_topology if shell_record is not None or local else None,shell_selection=shell_record,local_holes=local_record,
                    fidelity=fidelity,correspondence='SOURCE_GEOMETRY_RETAINED_PATCH_HAS_NO_APPEARANCE_IDENTITY' if local else 'NEW_SURFACE_NO_FACE_OR_UV_IDENTITY',
                    **({'native_units':native_units} if native_units is not None else {})));write_json(report_path,report)
            name=original.name;original.name=('CUMESH_DC_ORIGINAL_' if dc else 'WTIVO_ORIGINAL_')+name
            if local:
                combined_v=[vertices,*local_vertices];combined_f=[faces,*local_faces]
                total_vertices+=len(vertices)+local_vertex_count-component_vertex_count
                if total_vertices>MAX_VERTICES:raise ValueError('WTiVo local aggregate vertex budget exceeded')
                entry['local_geometry_preservation']=dict(original_vertices=len(vertices),original_faces=len(faces),
                    original_geometry_sha256=geometry_hash(vertices,faces),
                    original_rows_and_triangles_are_byteexact_prefix=True,new_vertices=local_vertex_count,new_faces=local_face_count)
            result=make_mesh(name,np.concatenate(combined_v),np.concatenate(combined_f),Matrix(matrix.tolist()));outputs.append(result)
            if result.name!=name or np.asarray(result.matrix_world,np.float64).tobytes()!=matrix.tobytes():raise ValueError(f'{backend} changed output object namespace/matrix')
            if result.data.materials or result.data.uv_layers:raise ValueError(f'{backend} output must not claim source UV/materials')
            entry['after']=final_validate(*mesh_arrays(result.data)[:2],backend=backend)
            entry['fidelity_bounds']=parent_bounds(component_fidelity);check_resources()
        export_meshes(candidate,outputs,materials=False,normals=False)
        if candidate.stat().st_size>MAX_FILE_BYTES:raise ValueError(f'{backend} GLB exceeds output byte limit')
        # Validate the real serialized artifact in a fresh scene in this worker.
        expected={o.name:(np.asarray(o.matrix_world,np.float64),mesh_arrays(o.data)[:2]) for o in outputs}
        bpy.ops.wm.read_factory_settings(use_empty=True);final_dir=directory/'reconstructed-reimport';final_dir.mkdir()
        imported_objects=import_meshes(candidate,False,final_dir)
        if {o.name for o in imported_objects}!=set(expected) or len(imported_objects)!=len(expected):raise ValueError(f'{backend} GLB reimport changed objects')
        reimport=[]
        for obj in imported_objects:
            v,f,_=mesh_arrays(obj.data);matrix=np.asarray(obj.matrix_world,np.float64);old_matrix,(ov,of)=expected[obj.name]
            if matrix.tobytes()!=old_matrix.tobytes() or not np.array_equal(v[f],ov[of]):raise ValueError(f'{backend} GLB reimport changed matrices/oriented corners')
            if obj.data.uv_layers or obj.data.materials:raise ValueError(f'{backend} GLB acquired fabricated appearance attributes')
            reimport.append(dict(object=obj.name,triangles=len(f),topology=final_validate(v,f,backend=backend)));check_resources()
        if digest(source)!=source_sha256:raise ValueError('Original source changed during reconstruction')
        candidate_sha256=digest(candidate)
        if 'local_witness_work' in report:
            report['local_witness_work'].update(native_calls=native_jobs,raw_triangles=raw_total_faces,raw_vertices=raw_total_vertices,raw_accounting=WITNESS_RAW_ACCOUNTING)
        report.update(status='completed',triangles=total_faces,vertices=total_vertices,raw_triangles=raw_total_faces,raw_vertices=raw_total_vertices,
                      component_jobs=native_jobs,source_component_count=jobs,dropped_source_components=sum(not row['retained'] for obj in report['objects'] for row in obj['source_component_selection']),source_import=staging,
                      geometry_import=dict(path=str(geometry_source),sha256=digest(geometry_source),scope='All positions/indices/nodes; UV/material/image bindings deliberately excluded, no source geometry preparation'),
                      hierarchy='Mesh object names and world matrices retained; non-mesh hierarchy is not reproduced',
                      reimport=reimport,surface_provenance=surface_provenance(candidate_sha256,report['objects'],reimport,config.get('repair_mode','volume'),bpy.app.version_string),
                      fidelity_notice=fidelity_notice([c['fidelity'] for c in report['components']]),elapsed_seconds=time.monotonic()-started)
        write_json(report_path,report)
        return dict(schema_version='2.0.0',status='completed',operation=operation,source_sha256=source_sha256,
            candidate_sha256=candidate_sha256,blender_version=bpy.app.version_string,
            device=((('Owned CuMesh DC CUDA unsigned-pair witness + CPU synthetic local patch; Blender geometry export' if native_jobs else 'No native/CUDA witness: closed source geometry retained; Blender geometry export') if local else 'Blender CPU BVH unsigned distances; owned CuMesh native CUDA connected contours; Blender geometry export') if dc
                    else (('Owned WTiVo CUDA witness + bounded local source-preserving patch; Blender geometry export' if native_jobs else 'No native/CUDA witness: closed source geometry retained; Blender geometry export') if local else 'Owned optional WTiVo CUDA reconstruction; Blender geometry export')),
            elapsed_seconds=time.monotonic()-started,reconstruction=report,fidelity_notice=report['fidelity_notice'],bakes=[],
            validation=dict(geometry='Closed oriented manifold numeric gates; self-intersections/physical validity NOT_EVALUATED',appearance='INVALIDATED',modly_ui='NOT_RUN'))
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');write_json(report_path,report);raise
