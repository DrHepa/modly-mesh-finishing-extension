# SPDX-License-Identifier: GPL-3.0-or-later
"""Modly one-line PROCESS adapter for the ordinary Blender pipeline."""
from datetime import datetime, timezone
from contextlib import contextmanager, suppress
import io
import json
import os
from pathlib import Path
import shutil
import select
import signal
import socket
import stat
import sys
import threading
import uuid

from glb_inspector import GLBReader, inspect_glb
from conservative_repair import plan_repair, stage_candidate, verify_candidate
from geometry_runner import find_blender
from production_preflight import NODES, parameters, plan_source, plan_geometry_source
from production_runner import run_production, digest

CANCEL_EVENT=threading.Event()
_STDOUT_SUPERVISOR_FD=None
_STDOUT_PROGRESS_SUFFIX=''


def emit(value):
    global _STDOUT_PROGRESS_SUFFIX
    line=json.dumps(value)
    prefix,_STDOUT_PROGRESS_SUFFIX=_STDOUT_PROGRESS_SUFFIX,''
    try:print(prefix+line,flush=True)
    except (BrokenPipeError,ConnectionResetError):
        CANCEL_EVENT.set();raise


def cancelled(*_):CANCEL_EVENT.set()


def _stdout_supervisor_fd():
    if sys.platform!='linux':return None
    try:fd=sys.stdout.fileno()
    except (AttributeError,io.UnsupportedOperation):return None
    mode=os.fstat(fd).st_mode
    return fd if stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode) else None


def _stdout_disconnected(fd):
    # No probe writes: writable readiness alone never means host loss.
    try:
        poll=select.poll();poll.register(fd,select.POLLHUP|select.POLLERR)
        return any(flags&(select.POLLHUP|select.POLLERR|select.POLLNVAL) for _,flags in poll.poll(0))
    except (OSError,ValueError):return True


def _progress_frame(value):
    return (json.dumps(value)+'\n').encode('ascii')


class _StreamProgress:
    """Keep only an already-started frame, never block while Blender is owned."""
    def __init__(self,connection):
        self.connection=connection;self.pending=b''

    def _try_send(self,data):
        try:return self.connection.send(data,socket.MSG_DONTWAIT|socket.MSG_NOSIGNAL)
        except OSError:return 0

    def report(self,value):
        if self.pending:
            self.pending=self.pending[self._try_send(self.pending):]
            if self.pending:return
        frame=_progress_frame(value)
        if len(frame)>256:return
        sent=self._try_send(frame)
        if sent:self.pending=frame[sent:]
        # A wholly unsent optional event is dropped rather than queued.

@contextmanager
def _native_progress_sink():
    """Best-effort Linux FIFO/Unix-stream progress without changing stdout flags."""
    global _STDOUT_PROGRESS_SUFFIX
    fd=_STDOUT_SUPERVISOR_FD;writer=None;pipe_buf=0;connection=None;stream=None
    try:
        if sys.platform=='linux' and fd is not None:
            mode=os.fstat(fd).st_mode
            if stat.S_ISFIFO(mode):
                # Reopen, not dup: an independent description owns O_NONBLOCK.
                writer=os.open(f'/proc/self/fd/{fd}',os.O_WRONLY|os.O_NONBLOCK)
                pipe_buf=os.fpathconf(writer,'PC_PIPE_BUF')
            elif stat.S_ISSOCK(mode) and os.get_blocking(fd) and socket.getdefaulttimeout() is None:
                # Socketpair used by Node spawn stdio; per-send nonblocking,
                # never O_NONBLOCK on the shared open file description.
                sockfd=os.dup(fd)
                try:connection=socket.socket(fileno=sockfd)
                except OSError:
                    with suppress(OSError):os.close(sockfd)
                    raise
                if connection.family==socket.AF_UNIX and connection.getsockopt(socket.SOL_SOCKET,socket.SO_TYPE)==socket.SOCK_STREAM:
                    stream=_StreamProgress(connection)
    except (OSError,ValueError):
        pass
    try:
        def report_fifo(value):
            data=_progress_frame(value)
            if len(data)<=pipe_buf:
                with suppress(OSError):os.write(writer,data)
        yield stream.report if stream is not None else report_fifo if writer is not None and pipe_buf>0 else None
    finally:
        # No I/O to stdout here, even when ownership cleanup itself fails.
        # The next ordinary NDJSON write completes any already-started line.
        if stream is not None and stream.pending:
            _STDOUT_PROGRESS_SUFFIX=stream.pending.decode('ascii')
        if connection is not None:connection.close()
        if writer is not None:os.close(writer)


@contextmanager
def _watch_stdout_supervisor():
    global _STDOUT_SUPERVISOR_FD
    previous=_STDOUT_SUPERVISOR_FD;fd=_stdout_supervisor_fd()
    stopped=threading.Event();thread=None;_STDOUT_SUPERVISOR_FD=fd
    def watch():
        while not stopped.wait(.1):
            if _stdout_disconnected(fd):CANCEL_EVENT.set();return
    try:
        if fd is not None:
            thread=threading.Thread(target=watch,name='Mesh Finishing stdout supervisor',daemon=True);thread.start()
        yield
    finally:
        stopped.set()
        if thread is not None and thread.is_alive():thread.join(.5)
        _STDOUT_SUPERVISOR_FD=previous


def write_json(path,value):
    stage=path.with_suffix(path.suffix+'.tmp');stage.write_text(json.dumps(value,indent=2)+'\n');stage.replace(path)


def _budget_unmet_notice(report):
    if report.get('status')!='completed_budget_unmet' or report.get('node_id') not in ('optimize-rebake','simplify-mesh'):return None
    worker=report.get('worker');after=report.get('after')
    simplification=worker.get('simplification') if isinstance(worker,dict) else None
    counts=after.get('counts') if isinstance(after,dict) else None
    if not isinstance(simplification,dict) or not isinstance(counts,dict) or simplification.get('budget_met') is not False:return None
    requested=simplification.get('triangle_budget');actual=counts.get('active_scene_render_triangles')
    if type(requested)is not int or type(actual)is not int or not 0<requested<actual:return None
    return f'Mesh Finishing: kept {actual} triangles; target was {requested}. Quality limits were not relaxed.'


def _checked_directories(*paths):
    for path in paths:
        if path.resolve()!=path:raise ValueError('Mesh Finishing directories may not contain symlinks')
    for path in paths:
        path.mkdir(parents=True,exist_ok=True)
        if path.resolve()!=path:raise ValueError('Mesh Finishing directories may not contain symlinks')


def process(payload, *, keep_diagnostics=None):
    if not isinstance(payload,dict):raise ValueError('Expected an object payload')
    data=payload.get('input') or {};params=payload.get('params') or {}
    if not isinstance(data,dict) or not isinstance(params,dict):raise ValueError('Input and params must be objects')
    node=data.get('nodeId') or payload.get('nodeId')
    if node not in NODES:raise ValueError('Unsupported Mesh Finishing nodeId')
    workspace=payload.get('workspaceDir');raw=data.get('filePath')
    if not isinstance(workspace,str) or not workspace or not isinstance(raw,str):raise ValueError('workspaceDir and input.filePath are required')
    workspace=Path(workspace).resolve();source=Path(raw).resolve(strict=True)
    if not source.is_file() or source.suffix.lower()!='.glb':raise ValueError('A GLB input is required')
    if source.stat().st_size>512*1024**2:raise ValueError('GLB input exceeds 512 MiB')
    config=parameters(params,node)
    if keep_diagnostics is None:
        keep_diagnostics=os.environ.get('MODLY_MESH_FINISHING_KEEP_DIAGNOSTICS')=='1'
    output_root=workspace/'Workflows'/'MeshFinishing'/'outputs'
    # Neither this private scratch tree nor a diagnostic opt-in is in a
    # Modly Library scan root (Workflows or Exports).
    scratch_root=workspace/'.mesh-finishing'/'jobs'
    _checked_directories(output_root,scratch_root)
    job=scratch_root/uuid.uuid4().hex;job.mkdir(mode=0o700)
    try:
        return _process_job(source,node,config,job,output_root,keep_diagnostics)
    finally:
        if not keep_diagnostics:
            with suppress(OSError):shutil.rmtree(job)


def _process_job(source,node,config,job,output_root,keep_diagnostics):
    preserved=job/'source.glb';shutil.copyfile(source,preserved);original_hash=digest(preserved)
    if digest(source)!=original_hash:raise ValueError('Source changed while preserving it')
    report=dict(schema_version='2.0.0',node_id=node,started_at=datetime.now(timezone.utc).isoformat(),
                source=dict(path=str(source),preserved=str(preserved),sha256=original_hash),parameters=config,
                validation=dict(visual='NOT_RUN',modly_ui='NOT_RUN'),status='running')
    sidecar=job/'report.json' if keep_diagnostics else None
    if sidecar is not None:write_json(sidecar,report)
    staged=job/'candidate.glb';produced=False;published=False;owned_output_dir=None
    try:
        if CANCEL_EVENT.is_set():raise ValueError('Processing cancelled')
        emit(dict(type='progress',percent=5,label='Original preserved'))
        inspection=inspect_glb(preserved);report['before']=inspection
        if node=='inspect-mesh':
            from inspect_source_uv import inspect_source_uv
            report['original_uv']=inspect_source_uv(preserved,inspection,check=lambda: check_cancel())
            if config['analysis_mode']=='geometry':
                from inspect_native_geometry import plan_geometry_inspection
                plan_geometry_inspection(preserved,inspection)
                emit(dict(type='progress',percent=15,label='Inspecting imported geometry without modifying the source'))
                metrics,_=run_production(find_blender(),preserved,original_hash,job,dict(config,operation='inspect-geometry'),
                                         cancel_event=CANCEL_EVENT,timeout_seconds=config['timeout_seconds'])
                report['worker']=metrics;report['geometry']=metrics['geometry'];check_cancel()
            result=preserved;report['status']='inspection_completed'
        elif node=='conservative-repair':
            plan=plan_repair(preserved,allow_emissive_strength=True);report['repair_plan']=plan
            if plan['removed_zero_area_triangles']:
                result=job/'repaired.glb';stage_candidate(preserved,staged,plan)
                report['repair_validation']=verify_candidate(preserved,staged,plan);produced=True
                report['status']='repair_completed'
            else:result=preserved;report['status']='repair_noop'
            report['appearance_policy']='Only exactly zero-area triangle indices are removed; no distance weld, hole filling or remeshing'
        elif node in ('reconstruct-wtivo','reconstruct-cumesh-dc'):
            config={**config,**dict(operation=node,plan=plan_geometry_source(preserved))}
            if node=='reconstruct-cumesh-dc':
                from cumesh_dc_runtime import resolve_runtime
            else:from wtivo_runtime import resolve_runtime
            runtime=resolve_runtime(extension=Path(__file__).resolve().parent)
            report['optional_runtime']=runtime['provenance']
            backend='CuMesh DC' if node=='reconstruct-cumesh-dc' else 'WTiVo'
            emit(dict(type='progress',percent=15,label=(f'Checking for local boundary loops with optional {backend}' if config.get('repair_mode')=='local_holes'
                 else f'Reconstructing geometry with optional {backend}; source appearance is invalidated')))
            metrics,_=run_production(find_blender(),preserved,original_hash,job,config,
                                     cancel_event=CANCEL_EVENT,timeout_seconds=config['timeout_seconds'])
            report['worker']=metrics;check_cancel()
            reconstruction=metrics.get('reconstruction')
            components=reconstruction.get('components') if isinstance(reconstruction,dict) else None
            noop=(config.get('repair_mode')=='local_holes' and metrics.get('status')=='completed'
                  and isinstance(reconstruction,dict) and reconstruction.get('status')=='completed'
                  and reconstruction.get('repair_mode')=='local_holes'
                  and type(reconstruction.get('component_jobs')) is int and reconstruction['component_jobs']==0
                  and type(reconstruction.get('dropped_source_components')) is int and reconstruction['dropped_source_components']==0
                  and isinstance(components,list) and bool(components)
                  and type(reconstruction.get('source_component_count')) is int and reconstruction['source_component_count']==len(components)
                  and all(isinstance(component,dict) and isinstance(component.get('native'),dict)
                          and component['native'].get('status')=='retained_closed_without_native'
                          and isinstance(component.get('local_holes'),dict)
                          and component['local_holes'].get('loops')==[] for component in components))
            if noop:
                staged.unlink()  # Only the validated, job-owned geometry-only scratch candidate.
                result=preserved;report['status']='reconstruction_noop'
                report['worker_scope']='The internal geometry-only candidate was validated then discarded. Worker INVALIDATED appearance/provenance describes that scratch candidate, not the returned original GLB.'
                report['appearance_policy']='Original GLB preserved byte-exact; no local holes or dropped source components. Authored UV/PBR preserved.'
            else:
                result=job/'reconstructed.glb';produced=True
                if metrics.get('fidelity_notice'):emit(dict(type='log',message=metrics['fidelity_notice']))
                report['status']='geometry_reconstructed'
                report['appearance_policy']='Geometry only; UV/PBR invalidated. Preserve the original textured HIGH for an explicit later transfer. New surfaces have no inferred source appearance.'
        else:
            supplied=node=='rebake-onto-mesh' or (node=='uv-repack-rebake' and bool(config.get('high_poly_path')))
            geometry_only=False
            if node=='simplify-mesh':
                with GLBReader(preserved) as reader:
                    primitives=[primitive for mesh in reader.doc.get('meshes',[]) for primitive in mesh.get('primitives',[])]
                    no_uv=all('TEXCOORD_0' not in primitive['attributes'] for primitive in primitives)
                    plain_materials=all(isinstance(material,dict)
                        and set(material)<={'name','pbrMetallicRoughness','emissiveFactor','alphaMode','alphaCutoff','doubleSided'}
                        and isinstance(material.get('pbrMetallicRoughness',{}),dict)
                        and set(material.get('pbrMetallicRoughness',{}))<={'baseColorFactor','metallicFactor','roughnessFactor'}
                        for material in reader.doc.get('materials',[]))
                    geometry_only=no_uv and (all('material' not in primitive for primitive in primitives)
                        or (plain_materials and all(reader.doc.get(key,[])==[] for key in ('textures','images','samplers'))))
            plan=plan_geometry_source(preserved) if geometry_only else plan_source(preserved,target=supplied,require_final_uv=node=='rebake-onto-mesh',
                allow_generated_target_normals=node=='uv-repack-rebake' and supplied,inspection=inspection)
            config={**config,**dict(operation=node,plan=plan,geometry_only=geometry_only)}
            active=inspection['counts']['active_scene_render_triangles']
            noop=node=='simplify-mesh' and type(active)is int and 0<active<=config['triangle_budget']
            if noop:
                if inspection['findings']:raise ValueError('Structural findings must be resolved before simplification no-op')
                if type(plan['triangles'])is not int or plan['triangles']!=active:
                    raise ValueError('Simplification plan differs from active scene triangle count')
                result=preserved;report['status']='simplification_noop'
                report['simplification']=dict(source_triangles=active,triangles=active,triangle_budget=config['triangle_budget'],
                    budget_met=True,output_changed=False,requested_backend=config['simplification_backend'],
                    backend_execution='NOT_RUN',native_geometry='NOT_RUN')
                report['appearance_policy']='Original GLB preserved byte-exact; authored appearance and attributes unchanged. No native geometry qualification.'
            else:
                if supplied:
                    high=Path(config['high_poly_path']).resolve(strict=True)
                    if high==source or not high.is_file() or high.suffix.lower()!='.glb':raise ValueError('High source and low target must be distinct GLBs')
                    if high.stat().st_size>512*1024**2:raise ValueError('High source exceeds 512 MiB')
                    high_copy=job/'high-source.glb';shutil.copyfile(high,high_copy)
                    high_hash=digest(high_copy)
                    if digest(high)!=high_hash:raise ValueError('Original HIGH changed while preserving it')
                    report['appearance_source']=dict(path=str(high),preserved=str(high_copy),sha256=high_hash)
                    config.update(target_path=str(preserved),target_sha256=original_hash,
                                  target_plan=plan,plan=plan_source(high_copy))
                    preserved_high=high_copy;worker_hash=digest(high_copy)
                else:preserved_high=preserved;worker_hash=original_hash
                emit(dict(type='progress',percent=15,label=('Reducing mesh geometry; no UV or texture transfer' if node=='simplify-mesh'
                    else 'Preparing final UV and transferring PBR' if node in ('rebake-onto-mesh','uv-repack-rebake') else 'Reducing geometry, preparing UV and transferring PBR')))
                if node=='simplify-mesh':
                    metrics,_=run_production(find_blender(),preserved_high,worker_hash,job,config,
                                             cancel_event=CANCEL_EVENT,timeout_seconds=config['timeout_seconds'])
                else:
                    with _native_progress_sink() as progress:
                        metrics,_=run_production(find_blender(),preserved_high,worker_hash,job,config,
                                                 cancel_event=CANCEL_EVENT,timeout_seconds=config['timeout_seconds'],progress=progress)
                report['worker']=metrics;check_cancel()
                result=job/('simplified.glb' if node=='simplify-mesh' else 'optimized_rebaked.glb')
                produced=True
                report['status']=('completed_budget_unmet' if node in ('simplify-mesh','optimize-rebake')
                                  and not metrics['simplification']['budget_met'] else 'completed')
                report['appearance_policy']=('Geometry only; source UV/PBR are invalidated; rebake before use' if node=='simplify-mesh'
                                             else 'PBR transferred by CPU Cycles; visual and Modly UI qualification remain separate')
        output=staged if produced else result
        report['after']=inspection if not produced and output==preserved else inspect_glb(output)
        if produced:
            counts=report['after']['counts']
            if counts['primitives_with_out_of_range_indices']:raise ValueError('Produced GLB has out-of-range triangle indices')
            if counts['nonfinite_position_vertices']:raise ValueError('Produced GLB has nonfinite POSITION values')
            if not counts['active_scene_render_triangles']:raise ValueError('Produced GLB has no active scene triangles')
        output_hash=digest(output)
        if digest(source)!=original_hash or (output_hash if output==preserved else digest(preserved))!=original_hash:raise ValueError('Original source changed')
        if report.get('appearance_source'):
            appearance=report['appearance_source']
            if digest(appearance['path'])!=appearance['sha256'] or digest(appearance['preserved'])!=appearance['sha256']:
                raise ValueError('Original appearance HIGH changed')
        # Keep the preserved copy when diagnostics are explicitly requested.
        # In a normal no-op, promote that exact copy without another disk write.
        if output==preserved and keep_diagnostics:
            shutil.copyfile(preserved,staged)
            if digest(staged)!=output_hash:raise ValueError('Output copy changed before publication')
            output=staged
        _checked_directories(output_root)
        destination=output_root/job.name
        destination.mkdir(mode=0o700)
        owned_output_dir=destination
        result=destination/result.name
        # Last cancellation boundary. Only one validated, complete file is
        # published, atomically. A signal after commit does not undo success.
        check_cancel()
        output.replace(result);published=True
        report['result']=dict(path=str(result),sha256=output_hash)
        if sidecar is not None:
            write_json(sidecar,report)
            emit(dict(type='log',message='Mesh Finishing report: '+str(sidecar)))
        elif node=='inspect-mesh':
            count=inspection['counts']['active_scene_render_triangles']
            emit(dict(type='log',message=f"Mesh Finishing: inspected {count} triangles; {len(inspection['findings'])} structural findings. Source unchanged."))
        notice=_budget_unmet_notice(report)
        if notice is not None:emit(dict(type='log',message=notice))
        return result,sidecar
    except Exception as error:
        report.pop('result',None)
        if published:
            if produced:result.replace(output)
            else:result.unlink()
        if owned_output_dir is not None:
            with suppress(OSError):owned_output_dir.rmdir()
        if sidecar is not None:
            report.update(status='cancelled' if CANCEL_EVENT.is_set() else 'failed',error=str(error))
            write_json(sidecar,report)
        raise


def check_cancel():
    if _STDOUT_SUPERVISOR_FD is not None and _stdout_disconnected(_STDOUT_SUPERVISOR_FD):CANCEL_EVENT.set()
    if CANCEL_EVENT.is_set():raise ValueError('Processing cancelled')


def main():
    signal.signal(signal.SIGINT,cancelled);signal.signal(signal.SIGTERM,cancelled)
    try:
        with _watch_stdout_supervisor():
            check_cancel()
            line=sys.stdin.readline()
            if not line:raise ValueError('Missing Modly PROCESS payload')
            check_cancel()
            output,_=process(json.loads(line))
            emit(dict(type='progress',percent=100,label='Processing complete'))
            emit(dict(type='done',result=dict(filePath=str(output))))
        return 0
    except Exception as error:
        try:emit(dict(type='error',message=str(error)))
        except (BrokenPipeError,ConnectionResetError):
            try:sys.stdout.close()  # Avoid a second broken flush during interpreter shutdown.
            except OSError:pass
            try:print('PROCESS output supervisor disconnected; processing cancelled',file=sys.stderr,flush=True)
            except OSError:pass
        return 1


if __name__=='__main__':raise SystemExit(main())
