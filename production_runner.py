# SPDX-License-Identifier: GPL-3.0-or-later
"""One isolated Blender process with bounded lifetime and owned logs."""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from geometry_runner import GeometryError, GeometryCancelled, _terminate_tree, _posix_child_exited_without_reaping

SHUTDOWN_GRACE_SECONDS=3

# These are ordered milestone positions, not fractions of runtime or ETAs.
_PROGRESS_MARKERS={
    b'geometry':(25,'Geometry preparation started'),
    b'uv':(45,'Destination UV preparation started'),
    b'correspondence':(65,'High-poly correspondence started'),
    b'bake':(85,'Texture baking started'),
    b'validation':(95,'Export validation started'),
}
_PROGRESS_READ_BYTES=64*1024


class _ProgressReader:
    """Read at most one bounded chunk per poll without consuming the raw log."""
    def __init__(self,log,emit,nonce):
        self.log=log;self.emit=emit;self.pending=b'';self.discard=False;self.percent=0
        prefix=f'MESH_FINISHING_PROGRESS {nonce} '.encode('ascii')
        self.markers={prefix+k:v for k,v in _PROGRESS_MARKERS.items()}
        self.max_line_bytes=max(map(len,self.markers))+1  # Allow CRLF.

    def poll(self):
        try:block=self.log.read(_PROGRESS_READ_BYTES)
        except OSError:return
        if not block:return
        lines=(self.pending+block).split(b'\n')
        pending=lines.pop()
        # An oversized line's suffix must never become a standalone marker.
        if self.discard and lines:lines=lines[1:];self.discard=False
        self.discard=self.discard or len(pending)>self.max_line_bytes
        self.pending=b'' if self.discard else pending
        for line in lines:
            mark=self.markers.get(line.removesuffix(b'\r'))
            if mark and mark[0]>self.percent:
                self.percent=mark[0]
                self.emit(dict(type='progress',percent=mark[0],label=mark[1]))


# Run after exec into an isolated Python child, never in unsafe preexec_fn.
_LINUX_PARENT_DEATH_EXEC = """import ctypes,os,signal,sys
parent=int(sys.argv[1])
if parent<=1 or os.getppid()!=parent:
    raise SystemExit('Owning parent absent before native exec')
if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)!=0:
    raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG failed')
if os.getppid()!=parent:
    raise SystemExit('Owning parent changed before native exec')
os.execv(sys.argv[2],sys.argv[2:])
"""


def parent_death_command(command, *, python=None):
    """Linux direct-child death guard; exec preserves the owned PID and PGID."""
    command=list(map(str,command))
    if sys.platform!='linux':return command
    interpreter=Path(sys.executable if python is None else python)
    if not interpreter.is_absolute() or not interpreter.is_file():
        raise GeometryError('Parent-death launcher requires the verified absolute Python interpreter')
    if not command or not Path(command[0]).is_absolute():
        raise GeometryError('Parent-death launcher requires the validated absolute native executable')
    return [str(interpreter),'-I','-B','-c',_LINUX_PARENT_DEATH_EXEC,str(os.getpid()),*command]


def read_terminal(path):
    if not path.is_file():return None
    if path.stat().st_size>4096:raise GeometryError('Worker terminal sentinel exceeds its size limit')
    result=json.loads(path.read_text())
    if not isinstance(result,dict) or result.get('status') not in ('completed','error'):raise GeometryError('Invalid worker terminal sentinel')
    return result


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def cpu_threads():
    try:available=len(os.sched_getaffinity(0))
    except (AttributeError,OSError):available=os.cpu_count() or 1
    return max(1,min(8,available))


def run_production(blender, source, source_sha256, job_dir, config, *, cancel_event, timeout_seconds, progress=None):
    if os.name=='posix' and not all(hasattr(os,name) for name in ('waitid','WNOWAIT')):
        raise GeometryError('Safe Blender process-group cleanup requires POSIX waitid/WNOWAIT')
    config=dict(config,cpu_threads=cpu_threads())
    if progress is not None:config['_progress_nonce']=secrets.token_hex(16)
    config_path=job_dir/'config.json';config_path.write_text(json.dumps(config,indent=2)+'\n')
    candidate=job_dir/'candidate.glb';sentinel=job_dir/'worker-result.json';terminal=job_dir/'worker-terminal.json'
    command=[str(blender),'--background','--factory-startup','--disable-autoexec','-noaudio','--threads',str(config['cpu_threads']),
             '--python-exit-code','1','--python',str(Path(__file__).with_name('blender_production_worker.py')),
             '--','--source',str(source),'--sha256',source_sha256,'--config',str(config_path),
             '--candidate',str(candidate),'--sentinel',str(sentinel)]
    launch_command=parent_death_command(command)
    windows_owner=None
    if os.name=='nt':
        from windows_job import WindowsJobOwner, windows_job_command
        launch_command=windows_job_command(command)
    start=time.monotonic();child=None;failure=None;terminal_result=None;terminal_at=None;shutdown_forced=False
    invocation=dict(command=command,launch_command=launch_command,
                    parent_death_guard=('Windows KILL_ON_JOB_CLOSE after gated assignment; PID is CPython shim' if os.name=='nt' else 'Linux direct child SIGKILL before exec' if sys.platform=='linux' else 'NOT_SUPPORTED_ON_THIS_PLATFORM'),
                    timeout_seconds=timeout_seconds,status='starting',cpu_threads=config['cpu_threads'],
                    shutdown_grace_seconds=SHUTDOWN_GRACE_SECONDS,shutdown_forced=False,shutdown_method=None)
    path=job_dir/'invocation.json'
    log_path=job_dir/'blender.log'
    with log_path.open('wb') as log,ExitStack() as stack:
        relay=None
        if progress is not None:
            try:relay=_ProgressReader(stack.enter_context(log_path.open('rb',buffering=0)),progress,config['_progress_nonce'])
            except OSError:pass  # Optional progress must not affect native admission.
        try:
            if os.name=='nt':
                windows_owner=WindowsJobOwner()
                child=windows_owner.start(launch_command,log)
            else:
                child=subprocess.Popen(launch_command,stdout=log,stderr=subprocess.STDOUT,
                                       start_new_session=os.name=='posix',creationflags=0)
            invocation.update(status='running',pid=child.pid);path.write_text(json.dumps(invocation,indent=2)+'\n')
            while not (_posix_child_exited_without_reaping(child.pid) if os.name=='posix' else child.poll() is not None):
                if cancel_event.is_set():failure=GeometryCancelled('Processing cancelled');break
                if time.monotonic()-start>timeout_seconds:failure=GeometryError('Blender operation timed out');break
                if relay is not None:relay.poll()
                if terminal_result is None:terminal_result=read_terminal(terminal)
                if terminal_result is not None:
                    if terminal_at is None:terminal_at=time.monotonic()
                    if time.monotonic()-terminal_at>=SHUTDOWN_GRACE_SECONDS:shutdown_forced=True;break
                time.sleep(.2)
        except Exception as exc:failure=exc
        finally:
            if windows_owner is not None:
                # Close even if the shim already exited: descendants must not
                # outlive the PROCESS-owned job. Never use numeric Windows PID.
                child=windows_owner.child
                try:windows_owner.close()
                except Exception as exc:failure=failure or exc
            elif child is not None:
                # POSIX observation above retains the leader's PID. The group
                # is signalled while that identity is still reserved, then the
                # existing helper reaps it. Never poll()/wait() before killpg.
                _terminate_tree(child)
            try:
                if terminal_result is None:terminal_result=read_terminal(terminal)
                if terminal_result and terminal_result['status']=='error':failure=failure or GeometryError(f'Blender worker failed: {terminal_result.get("error","unspecified error")}')
            except Exception as exc:failure=failure or exc
            if child is not None and child.returncode and not (shutdown_forced and child.returncode==(-signal.SIGKILL if os.name=='posix' else 1) and terminal_result and terminal_result['status']=='completed'):
                failure=failure or GeometryError(f'Blender failed (exit {child.returncode}); see {job_dir / "blender.log"}')
            # Fast workers may exit between polls. Drain one chunk after
            # cleanup only for exit 0 and a successful terminal sentinel.
            if relay is not None and not failure and child is not None and child.returncode==0 and not cancel_event.is_set() and terminal_result and terminal_result['status']=='completed':
                relay.poll()
            invocation.update(status='failed' if failure or child is None else 'finished_forced_shutdown' if shutdown_forced else 'finished',
                              shutdown_forced=shutdown_forced,
                              shutdown_method=('SIGKILL_process_group' if os.name=='posix' else 'TerminateJobObject') if shutdown_forced else None,
                              returncode=child.returncode if child else None,elapsed_seconds=time.monotonic()-start)
            path.write_text(json.dumps(invocation,indent=2)+'\n')
    if failure:raise failure
    if not terminal_result or terminal_result['status']!='completed':raise GeometryError('Worker did not return a successful terminal sentinel')
    if cancel_event.is_set():raise GeometryCancelled('Processing cancelled')
    if not sentinel.is_file() or sentinel.stat().st_size>4*1024**2:raise GeometryError('Worker did not return bounded completion metrics')
    metrics=json.loads(sentinel.read_text())
    if metrics.get('status')!='completed':raise GeometryError('Worker did not complete')
    if config.get('operation')=='inspect-geometry':
        if metrics.get('operation')!='inspect-geometry' or metrics.get('source_sha256')!=source_sha256 or not isinstance(metrics.get('geometry'),dict) or metrics['geometry'].get('status')!='PARTIAL_MEASUREMENTS' or candidate.exists():
            raise GeometryError('Inspection completion/source identity mismatch or unexpected candidate output')
    elif not candidate.is_file() or metrics.get('candidate_sha256')!=digest(candidate):raise GeometryError('Worker completion/output identity mismatch')
    if digest(source)!=source_sha256:raise GeometryError('Preserved source changed')
    metrics['shutdown']={key:invocation[key] for key in ('returncode','shutdown_forced','shutdown_method','shutdown_grace_seconds')}
    return metrics,[]
