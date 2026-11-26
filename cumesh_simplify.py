# SPDX-License-Identifier: GPL-3.0-or-later
"""Owned GPU NPZ simplifier. Imports Torch only inside an explicit command."""
import argparse
import hashlib
import ctypes
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import resource
import sys
import tempfile
import time
import zipfile

MAX_VERTICES = 6_000_000
MAX_FACES = 2_000_000
TORCH_OOM = 'TORCH_OUT_OF_MEMORY'


class CuMeshOutOfMemory(RuntimeError):
    # Private typed terminal signal from the owned helper, never a proposal.
    pass


def is_torch_oom(error, torch):
    if isinstance(error,NativeStepError):error=error.original_error
    kinds=(getattr(torch,'OutOfMemoryError',None),getattr(getattr(torch,'cuda',None),'OutOfMemoryError',None))
    return any(isinstance(kind,type) and isinstance(error,kind) for kind in kinds)


def bound_oom_report(path, input_sha, *, parent_pid, helper_pid=None, request=None):
    # Admission reads only the owned atomic failure report, never the checkpoint.
    try:
        if path.is_symlink() or not path.is_file():return False
        with path.open('rb') as stream:data=stream.read(1024**2+1)
        if len(data)>1024**2:return False
        r=json.loads(data)
    except (OSError,ValueError,RecursionError):
        return False
    return (isinstance(r,dict) and r.get('status')=='failed' and r.get('error_code')==TORCH_OOM
        and r.get('usable_mesh') is False and r.get('output_role')=='diagnostic_input_or_checkpoint_only'
        and r.get('input_sha256')==input_sha and type(r.get('parent_pid')) is int and r['parent_pid']==parent_pid
        and type(r.get('pid')) is int and r['pid']>1 and r['pid']!=parent_pid
        and (r.get('stage_request') is None if request is None else
            type(r.get('stage_request')) is int and r['stage_request']==request and r['pid']==helper_pid))


class MeshValidationError(ValueError):
    def __init__(self, code):
        self.code=code
        self.geometric=code in ('EMPTY_GEOMETRY','ZERO_AREA_TRIANGLE')
        super().__init__(code)


class NativeStepError(RuntimeError):
    def __init__(self, phase, error, counts, vertices, faces):
        self.phase,self.error_class,self.counts=phase,type(error).__name__,counts
        self.vertices,self.faces,self.original_error=vertices,faces,error
        super().__init__(str(error))


def validate_mesh(vertices, faces, *, reported=None, previous_faces=None):
    import numpy as np
    if not isinstance(vertices,np.ndarray) or not isinstance(faces,np.ndarray):raise MeshValidationError('INVALID_ARRAY_CONTAINER')
    if vertices.dtype!=np.float32 or faces.dtype!=np.int32:raise MeshValidationError('INVALID_DTYPE')
    if vertices.ndim!=2 or faces.ndim!=2 or vertices.shape[1:]!=(3,) or faces.shape[1:]!=(3,):raise MeshValidationError('INVALID_SHAPE')
    if len(vertices)>MAX_VERTICES or len(faces)>MAX_FACES:raise MeshValidationError('RESOURCE_LIMIT')
    if not np.isfinite(vertices).all():raise MeshValidationError('NONFINITE_VERTICES')
    if faces.size and (faces.min()<0 or faces.max()>=len(vertices)):raise MeshValidationError('INVALID_INDICES')
    if previous_faces is not None:
        if not isinstance(reported,(tuple,list)) or len(reported)!=2 or any(type(n)is not int for n in reported) or tuple(reported)!=(len(vertices),len(faces)):raise MeshValidationError('NATIVE_COUNT_MISMATCH')
        if len(faces)>previous_faces:raise MeshValidationError('NATIVE_FACE_COUNT_INCREASE')
    if not len(vertices) or not len(faces):raise MeshValidationError('EMPTY_GEOMETRY')
    for start in range(0,len(faces),65536):
        triangle=vertices[faces[start:start+65536]].astype(float)
        if np.any(np.linalg.norm(np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]),axis=1)==0):raise MeshValidationError('ZERO_AREA_TRIANGLE')


class TopologyValidationError(ValueError):
    def __init__(self, code, stats):
        self.code,self.stats=code,stats
        super().__init__(code)


def face_duplicates(f,np):
    keys=np.sort(f,axis=1);order=np.lexsort((keys[:,2],keys[:,1],keys[:,0]));keys=keys[order]
    starts=np.flatnonzero(np.r_[True,np.any(keys[1:]!=keys[:-1],axis=1)]);sizes=np.diff(np.r_[starts,len(f)])
    parity=(f[:,0]>f[:,1])^(f[:,0]>f[:,2])^(f[:,1]>f[:,2]);positive=np.add.reduceat(parity[order].astype(np.int64),starts)
    return dict(groups=int(np.count_nonzero(sizes>1)),opposed_pairs=int(np.sum(positive*(sizes-positive))))

def corner_fans(f,left,right,np):
    # Corner nodes and every work array are bounded by 3F, never V*fixed-valence.
    a=np.r_[left,left//3*3+(left%3+1)%3];b=np.r_[right//3*3+(right%3+1)%3,right]
    flat=f.ravel();parent=np.arange(len(flat),dtype=np.int32)
    if not np.array_equal(flat[a],flat[b]):raise TopologyValidationError('INVALID_LINK_ENDPOINTS',{})
    for rounds in range(64):
        for _ in range(64):
            compressed=parent[parent]
            if np.array_equal(compressed,parent):break
            parent=compressed
        else:raise TopologyValidationError('LINK_COMPRESSION_RESOURCE_LIMIT',{})
        ra,rb=parent[a],parent[b]
        if np.array_equal(ra,rb):break
        np.minimum.at(parent,np.maximum(ra,rb),np.minimum(ra,rb))
    else:raise TopologyValidationError('LINK_UNION_RESOURCE_LIMIT',{})
    order=np.argsort(flat,kind='stable');vertices=flat[order];roots=parent[order]
    bad=(vertices[1:]==vertices[:-1])&(roots[1:]!=roots[:-1]);bad_vertices=np.unique(vertices[1:][bad])
    return dict(bad_vertex_fans=len(bad_vertices),first_bad_vertices=bad_vertices[:8].tolist(),union_rounds=rounds+1,
        referenced_vertices=int(np.count_nonzero(np.r_[True,vertices[1:]!=vertices[:-1]])),
        semantics='One connected corner fan per referenced vertex; together with simplicial faces/oriented edge incidence<=2 implies a path or cycle')


def indexed_topology(vertices, faces):
    """Independent oriented simplicial manifold admission; no geometric welding."""
    import numpy as np
    validate_mesh(vertices,faces)
    directed=np.stack((faces,np.roll(faces,-1,axis=1)),axis=-1).reshape(-1,2);keys=np.sort(directed,axis=1)
    order=np.lexsort((keys[:,1],keys[:,0]));keys=keys[order]
    starts=np.flatnonzero(np.r_[True,np.any(keys[1:]!=keys[:-1],axis=1)]);sizes=np.diff(np.r_[starts,len(order)])
    pairs=starts[sizes==2];left,right=order[pairs],order[pairs+1]
    stats=dict(vertices=len(vertices),faces=len(faces),edges=len(starts),manifold_pairs=len(pairs),
        boundary_edges=int(np.count_nonzero(sizes==1)),nonmanifold_edges=int(np.count_nonzero(sizes>2)),
        orientation_conflicts=int(np.count_nonzero(~np.all(directed[left]==directed[right,::-1],axis=1))),indexed_duplicate_faces=face_duplicates(faces,np))
    if stats['indexed_duplicate_faces']['groups']:raise TopologyValidationError('INDEXED_DUPLICATE_FACE',stats)
    if stats['nonmanifold_edges'] or stats['orientation_conflicts']:raise TopologyValidationError('EDGE_INCIDENCE_OR_ORIENTATION',stats)
    stats['vertex_links']=corner_fans(faces,left,right,np)
    if stats['vertex_links']['bad_vertex_fans']:raise TopologyValidationError('DISCONNECTED_VERTEX_LINK',stats)
    boundary=directed[order[starts[sizes==1]]];bits=vertices[boundary].copy().view(np.uint32).reshape(-1,6)
    bits=bits[np.lexsort(tuple(bits[:,i] for i in range(5,-1,-1)))];stats['boundary_corner_multiset_sha256']=hashlib.sha256(bits.tobytes()).hexdigest()
    incident=vertices[faces[np.any(np.isin(faces,np.unique(boundary)),axis=1)]].copy().view(np.uint32).reshape(-1,9)
    incident=incident[np.lexsort(tuple(incident[:,i] for i in range(8,-1,-1)))];stats['boundary_incident_triangles']=dict(count=len(incident),ordered_corner_multiset_sha256=hashlib.sha256(incident.tobytes()).hexdigest())
    stats['Euler']=stats['vertex_links']['referenced_vertices']-stats['edges']+len(faces)
    return stats


def prepare_topology(vertices, faces, repair, facts):
    """One exact representation split only when indexed admission requires it."""
    import numpy as np
    try:
        facts.update(initial=indexed_topology(vertices,faces),representation_repaired=False)
        return vertices,faces,facts['initial']
    except TopologyValidationError as error:
        facts.update(initial=error.stats,initial_rejection=error.code,representation_repaired=False)
        if error.code not in ('INDEXED_DUPLICATE_FACE','EDGE_INCIDENCE_OR_ORIENTATION','DISCONNECTED_VERTEX_LINK'):raise
        if not error.stats['manifold_pairs']:raise TopologyValidationError('REPRESENTATION_REPAIR_REQUIRES_M_POSITIVE',error.stats)
    if 3*len(faces)>MAX_VERTICES:raise TopologyValidationError('REPRESENTATION_REPAIR_RESOURCE_LIMIT',facts['initial'])
    prepared_v,prepared_f=repair()  # Native repair clears its own caches; never re-init this instance.
    validate_mesh(prepared_v,prepared_f)
    facts['native_prepared_counts']=[len(prepared_v),len(prepared_f)]
    if len(prepared_f)!=len(faces):raise TopologyValidationError('REPRESENTATION_FACE_COUNT_CHANGED',facts)
    for start in range(0,len(faces),65536):
        if not np.array_equal(prepared_v[prepared_f[start:start+65536]].view(np.uint32),vertices[faces[start:start+65536]].view(np.uint32)):
            raise TopologyValidationError('REPRESENTATION_ORDERED_CORNERS_CHANGED',facts)
    prepared=indexed_topology(prepared_v,prepared_f)
    facts.update(representation_repaired=True,ordered_corner_geometry_bitexact=True,prepared=prepared)
    return prepared_v,prepared_f,prepared


def final_topology(vertices, faces, prepared):
    stats=indexed_topology(vertices,faces)
    if stats['Euler']!=prepared['Euler']:raise TopologyValidationError('EULER_CHANGED',stats)
    for key in ('boundary_edges','boundary_corner_multiset_sha256','boundary_incident_triangles'):
        if stats[key]!=prepared[key]:raise TopologyValidationError('BOUNDARY_COLLAR_GEOMETRY_CHANGED',stats)
    return stats


def step_observation(vertices, faces, counts):
    """Bounded diagnostic reductions only; empty/unknown arrays never use min/max."""
    import numpy as np
    arrays={}
    for name,value in (('vertices',vertices),('faces',faces)):
        info=dict(container=type(value).__name__,dtype=None,shape=None,finite=None,index_min=None,index_max=None)
        if isinstance(value,np.ndarray):
            info.update(dtype=str(value.dtype),shape=list(value.shape))
            if value.size<=max(MAX_VERTICES,MAX_FACES)*3 and value.dtype.kind in 'fiu':
                info['finite']=bool(np.isfinite(value).all())
                if name=='faces' and value.size and value.dtype.kind in 'iu':info.update(index_min=int(value.min()),index_max=int(value.max()))
        arrays[name]=info
    native=list(counts) if isinstance(counts,(list,tuple)) and len(counts)==2 and all(type(n)is int for n in counts) else dict(invalid_container=type(counts).__name__)
    return dict(native_counts=native,read_counts=[len(a) if isinstance(a,np.ndarray) and a.ndim else None for a in (vertices,faces)],arrays=arrays)


def load_mesh(path):
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != 2 or {x.filename for x in entries} != {'vertices.npy', 'faces.npy'}: raise ValueError('NPZ must contain exactly vertices.npy and faces.npy')
        if any(x.file_size > 96*1024**2 for x in entries) or sum(x.file_size for x in entries) > 128*1024**2: raise ValueError('NPZ uncompressed resource admission exceeded')
    with np.load(path, allow_pickle=False, max_header_size=10000) as mesh:
        vertices, faces = mesh['vertices'], mesh['faces']
    validate_mesh(vertices, faces)
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces)


def bounded_steps(vertices, faces, target, step, *, max_steps, deadline, clock=time.monotonic, checkpoint=None, smoke=False, diagnostic=None, source_counts=None, is_oom=None):
    """Bound upstream's step primitive; never invoke its unbounded simplify()."""
    validate_mesh(vertices, faces)
    if type(target) is not int or not 0 < target <= MAX_FACES or type(max_steps) is not int or not 1 <= max_steps <= 128: raise ValueError('Invalid target or step admission')
    source_v,source_f=(len(vertices),len(faces)) if source_counts is None else source_counts
    if type(source_v) is not int or not 0<source_v<=MAX_VERTICES or source_f!=len(faces):raise ValueError('Original source counts differ from exact prepared face geometry')
    best_v, best_f = vertices.copy(), faces.copy(); previous = len(faces); threshold = 1e3 if smoke else 1e-8; stagnant = 0
    facts = []; stop = 'step_limit'; failure = None; rejected_step=None
    for ordinal in range(max_steps):
        if len(best_f) <= target: stop = 'budget_met'; break
        if clock() >= deadline: stop = 'deadline'; break
        v=f=reported=None
        try:
            v,f,reported=step(threshold)
            validate_mesh(v,f,reported=reported,previous_faces=previous)
        except Exception as error:
            geometric=isinstance(error,MeshValidationError) and error.geometric
            stop='geometric_candidate_rejected' if geometric else 'invalid_native_step';failure=str(error)
            phase='measured_geometry' if geometric else 'native_count_supervision' if isinstance(error,MeshValidationError) and error.code.startswith('NATIVE_') else 'readback_validation'
            if isinstance(error,NativeStepError):v,f,reported=error.vertices,error.faces,error.counts;phase=error.phase
            elif not isinstance(error,MeshValidationError):phase='native_step_or_readback'
            rejected_step=dict(ordinal=ordinal+1,threshold=threshold,reason_code=error.code if isinstance(error,MeshValidationError) else 'NATIVE_EXECUTION_ERROR',
                error_class=getattr(error,'error_class',type(error).__name__),phase=phase,classification='measured_geometric_candidate_rejected' if geometric else 'fatal_native_or_structural_error',
                **step_observation(v,f,reported))
            if diagnostic is not None:diagnostic(rejected_step)  # Persist before returning/exiting; never a checkpoint write.
            if is_oom is not None and is_oom(error):raise
            break
        removed = previous-len(f); facts.append(dict(step=ordinal+1, threshold=threshold, vertices=len(v), faces=len(f)))
        if len(f) < len(best_f):
            best_v, best_f = v.copy(), f.copy()
            if checkpoint is not None: checkpoint(best_v, best_f, list(facts))
        stagnant = stagnant+1 if removed == 0 and threshold == 1e3 else 0
        if removed/max(previous, 1) < .01: threshold = min(1e3, threshold*10)
        previous = len(f)
        if len(best_f) <= target: stop = 'budget_met'; break
        if clock() >= deadline: stop = 'deadline'; break
        if stagnant >= 8: stop = 'no_progress'; break
    return best_v, best_f, dict(stop_reason=stop, error=failure, candidate_rejected=stop=='geometric_candidate_rejected',rejected_step=rejected_step,steps=facts, attempted_steps=len(facts)+(failure is not None),
        budget_met=len(best_f) <= target, output_changed=len(best_f) < len(faces), source_vertices=source_v, source_faces=source_f, native_input_vertices=len(vertices), native_input_faces=len(faces), final_vertices=len(best_v), final_faces=len(best_f))


def terminal_policy(facts):
    """Shared actual CLI write/status/exit policy: an empty message is still an error."""
    has_error=facts['error'] is not None
    return dict(status='candidate_rejected' if facts['candidate_rejected'] else 'failed' if has_error else 'completed' if facts['budget_met'] else 'budget_unmet',
        usable_mesh=not has_error,output_role='diagnostic_last_valid_checkpoint_only' if has_error else 'validated_geometry_proposal',
        fatal_error=has_error and not facts['candidate_rejected'])


def parent_death_guard(expected_parent):
    if sys.platform != 'linux': raise RuntimeError('GPU helper currently requires Linux')
    if type(expected_parent) is not int or expected_parent <= 1 or os.getppid() != expected_parent: raise RuntimeError('Expected direct owning parent is absent; no native work permitted')
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0: raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG failed')
    if os.getppid() != expected_parent: raise RuntimeError('Owning parent changed while installing death signal')


def atomic_npz(path, vertices, faces):
    import numpy as np
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez(stream, vertices=vertices, faces=faces); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def smoke_mesh():
    import numpy as np
    vertices = np.array([(x/8, y/8, .05*np.sin(x*.5)*np.cos(y*.5)) for y in range(9) for x in range(9)], np.float32)
    faces = []
    for y in range(8):
        for x in range(8):
            a = y*9+x; faces.extend(((a,a+1,a+9), (a+1,a+10,a+9)))
    return vertices, np.array(faces, np.int32)


# Private Linux packet channel, never stdout (native libraries may print there).
_PACKET_BYTES = 65536


class CuMeshSession:
    """One lazy direct child owned exclusively by one simplify() call."""
    def __init__(self):
        self.child = self.channel = None
        self.closed = False
        self.sequence = 0
        self.deadline = None
        self.invocation = {}

    def run(self, command, *, env, timeout, log_path):
        if self.closed: raise RuntimeError('CuMesh stage is closed; no replay permitted')
        deadline = self.deadline = time.monotonic()+timeout
        try:
            if not 0<timeout<=610: raise TimeoutError('CuMesh request deadline exhausted or invalid')
            prefix, argv = list(command[:4]), list(command[4:])
            if self.child is None:
                self.prefix, self.environment = prefix, dict(env)
                self.log_path = str(log_path)
                self.channel, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                launch = prefix+['--serve-fd', str(peer.fileno()), '--parent-pid', str(os.getpid())]
                try:
                    with open(log_path, 'xb') as log:
                        self.child = subprocess.Popen(launch, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, env=env, pass_fds=(peer.fileno(),))
                finally:
                    peer.close()
                self.invocation = dict(command=launch, pid=self.child.pid, parent_pid=os.getpid(),
                    inherited_pgid=os.getpgrp(), log=self.log_path, scope='one direct child per simplify call')
            if prefix != self.prefix or env != self.environment:
                raise RuntimeError('CuMesh stage runtime identity changed')
            self.sequence += 1
            self.invocation['request_index'] = self.sequence
            packet = json.dumps(dict(version=1, request=self.sequence, argv=argv)).encode('utf-8')
            if len(packet)>_PACKET_BYTES: raise ValueError('CuMesh request exceeds private channel bound')
            remaining = deadline-time.monotonic()
            if remaining<=0: raise TimeoutError('CuMesh request deadline exhausted')
            self.channel.settimeout(remaining)
            if self.channel.send(packet) != len(packet): raise RuntimeError('Incomplete CuMesh request')
            remaining = deadline-time.monotonic()
            if remaining<=0: raise TimeoutError('CuMesh request deadline exhausted')
            self.channel.settimeout(remaining)
            response, _, flags, _ = self.channel.recvmsg(4096)
            if not response or flags & socket.MSG_TRUNC: raise RuntimeError('CuMesh response missing or oversized')
            reply = json.loads(response)
            if time.monotonic()>deadline: raise TimeoutError('CuMesh response exceeded request deadline')
            if (isinstance(reply,dict) and type(reply.get('version')) is int and type(reply.get('request')) is int
                and reply==dict(version=1,request=self.sequence,status='failed',error_code=TORCH_OOM)):
                raise CuMeshOutOfMemory()
            if (not isinstance(reply, dict) or type(reply.get('version')) is not int or
                type(reply.get('request')) is not int or reply != dict(version=1, request=self.sequence, status='completed')):
                raise RuntimeError('CuMesh request failed or response identity differs; see '+self.log_path)
            return subprocess.CompletedProcess(command, 0)
        except (TimeoutError, BlockingIOError) as error:
            self.close(check=False)
            raise subprocess.TimeoutExpired(command, timeout) from error
        except BaseException:
            self.close(check=False)
            raise

    def close(self, *, check=True):
        if self.closed: return
        self.closed = True
        primary = sys.exc_info()[1]
        try:
            try:
                if self.channel is not None: self.channel.close()
            finally:
                if self.child is not None:
                    # The unreaped Popen lease owns this PID, never the inherited PGID.
                    grace = min(2., max(0., self.deadline-time.monotonic())) if self.deadline is not None else 2.
                    try:
                        code = self.child.wait(timeout=grace if check else 0.)
                    except subprocess.TimeoutExpired:
                        self.child.terminate()
                        try: code = self.child.wait(timeout=.5)
                        except subprocess.TimeoutExpired:
                            self.child.kill()
                            code = self.child.wait(timeout=.5)
                    self.invocation['cleanup'] = dict(status='reaped', returncode=code)
                    if check and code: raise RuntimeError('CuMesh stage child failed during shutdown')
        except Exception as error:
            unreaped = self.child is not None and self.child.returncode is None
            self.invocation['cleanup'] = dict(status='unreaped_direct_child' if unreaped else 'cleanup_failed',
                pid=self.child.pid if self.child is not None else None,
                returncode=self.child.returncode if self.child is not None else None, error_class=type(error).__name__,
                escalation='Owning production supervisor must reap its authenticated process group; no replay or fallback')
            message = 'CuMesh stage cleanup failed; see stage_session.cleanup; outer production group cleanup required'
            if primary is not None: primary.add_note(message)
            try: print(json.dumps(dict(event='cumesh_stage_cleanup_failed', **self.invocation['cleanup'])), file=sys.stderr, flush=True)
            except Exception: pass  # The primary failure and invocation still retain the escalation.
            if check: raise RuntimeError(message) from error


def serve(fd, expected_parent):
    parent_death_guard(expected_parent)
    if not sys.flags.isolated: raise RuntimeError('Owned GPU helper requires Python -I')
    with socket.socket(fileno=fd) as channel:
        sequence = 1
        while True:
            packet, _, flags, _ = channel.recvmsg(_PACKET_BYTES)
            if not packet: return
            if flags & socket.MSG_TRUNC: raise ValueError('Oversized CuMesh request')
            request = json.loads(packet)
            argv = request.get('argv')
            if (request.keys() != {'version', 'request', 'argv'} or type(request['version']) is not int or
                request['version'] != 1 or type(request['request']) is not int or request['request'] != sequence or
                not isinstance(argv, list) or not 1<=len(argv)<=64 or
                any(not isinstance(x, str) or len(x)>16384 or '\x00' in x for x in argv) or
                argv.count('--parent-pid') != 1 or argv[-2:] != ['--parent-pid', str(expected_parent)]):
                raise ValueError('CuMesh private request admission failed')
            # A new main frame and fresh _C.CuMesh, with ALL original guards per request.
            # Any exception exits this child; the owner never replays a request.
            try:main(argv, initialize_interop=sequence==1, stage_request=sequence)
            except CuMeshOutOfMemory:
                channel.sendall(json.dumps(dict(version=1,request=sequence,status='failed',error_code=TORCH_OOM)).encode('utf-8'))
                raise
            channel.sendall(json.dumps(dict(version=1, request=sequence, status='completed')).encode('utf-8'))
            sequence += 1


def main(argv=None, *, initialize_interop=True, stage_request=None):
    parser = argparse.ArgumentParser(); source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path); source.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--target', type=int, required=True); parser.add_argument('--max-steps', type=int, default=128); parser.add_argument('--seconds', type=int, default=60)
    args = parser.parse_args(argv)
    parent_death_guard(args.parent_pid)
    if not sys.flags.isolated: raise RuntimeError('Owned GPU helper must be launched with Python -I')
    spec = importlib.util.spec_from_file_location('_owned_cumesh_runtime', Path(__file__).with_name('cumesh_runtime.py'))
    runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
    atomic_json, file_sha = runtime.atomic_json, runtime.file_sha
    if not 1 <= args.target <= MAX_FACES or not 1 <= args.max_steps <= 128: raise ValueError('Target/step admission exceeded')
    if not 1 <= args.seconds <= 600: raise ValueError('GPU command deadline must be 1..600 seconds')
    if args.output.exists() or args.report.exists() or args.output.resolve() == args.report.resolve(): raise ValueError('Output/report must be distinct new owned paths')
    if args.input and args.input.resolve() in (args.output.resolve(), args.report.resolve()): raise ValueError('Original NPZ may not be overwritten')
    started = time.monotonic(); report = dict(status='started', backend='CuMesh core simplify_step', pid=os.getpid(), parent_pid=os.getppid(), pgid=os.getpgrp(), timeout_scope='Cooperative step checks; owning caller must enforce an external deadline', usable_mesh=False, output_role='diagnostic_input_or_checkpoint_only')
    if stage_request is not None:
        report.update(stage_request=stage_request, process_scope='Reused stage child; RSS is lifetime peak, not per-request peak')
    atomic_json(args.report, report)
    torch=None
    try:
        import numpy as np
        lock = json.loads((Path(__file__).resolve().parent/'resources/cumesh/runtime-lock.json').read_text())
        for spec in lock['wheels']:
            if importlib.metadata.version(spec['name']) != spec['version']: raise RuntimeError('Owned runtime version differs: '+spec['name'])
        if sys.version_info[:3] != (3,12,16): raise RuntimeError('Owned helper requires pinned CPython3.12.16')
        input_sha = file_sha(args.input) if args.input else None
        vertices, faces = smoke_mesh() if args.smoke else load_mesh(args.input)
        atomic_npz(args.output, vertices, faces)
        report.update(status='input_checkpoint_not_final', source_vertices=len(vertices), source_faces=len(faces), final_faces=len(faces), input_sha256=input_sha, output_sha256=file_sha(args.output))
        atomic_json(args.report, report)
        import torch
        from cumesh import _C
        torch.set_num_threads(max(1,min(8,len(os.sched_getaffinity(0)))))
        if initialize_interop: torch.set_num_interop_threads(1)
        if not torch.cuda.is_available() or not str(torch.version.cuda).startswith('13.'): raise RuntimeError('CUDA13 runtime/device unavailable; no CPU fallback')
        if torch.cuda.get_device_capability(0) != (12,1) or 'GB10' not in torch.cuda.get_device_name(0): raise RuntimeError('Only GB10 SM121 lane is currently admitted')
        core = next(x for x in lock['wheels'] if x['name'] == 'cumesh')
        if file_sha(Path(_C.__file__)) != core['native_extension_sha256']: raise RuntimeError('Loaded CuMesh native binary differs')
        prefix = Path(sys.prefix).resolve()
        origins = dict(torch=str(Path(torch.__file__).resolve()), cumesh=str(Path(sys.modules['cumesh'].__file__).resolve()), native_core=str(Path(_C.__file__).resolve()), numpy=str(Path(np.__file__).resolve()))
        if prefix.name != 'venv' or not all(Path(path).is_relative_to(prefix) for path in origins.values()): raise RuntimeError('GPU libraries are not from the owned venv')
        torch.cuda.set_device(0)
        if not bool(torch._C._GLIBCXX_USE_CXX11_ABI): raise RuntimeError('CuMesh requires the locked CXX11 ABI')
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        mesh=None;preparation={};report['topology_preparation']=preparation
        def repair():
            nonlocal mesh
            mesh=_C.CuMesh();mesh.init(torch.from_numpy(vertices).cuda().contiguous(),torch.from_numpy(faces).cuda().contiguous())
            mesh.repair_non_manifold_edges();torch.cuda.synchronize();pv,pf=mesh.read()
            return pv.cpu().numpy(),pf.cpu().numpy()
        report['phase']='indexed_input_admission_or_exact_representation_split'
        native_v,native_f,prepared=prepare_topology(vertices,faces,repair,preparation)
        report.update(native_input_vertices=len(native_v),native_input_faces=len(native_f));atomic_json(args.report,report)
        if mesh is None:
            mesh=_C.CuMesh();mesh.init(torch.from_numpy(native_v).cuda().contiguous(),torch.from_numpy(native_f).cuda().contiguous())
        def step(threshold):
            phase='native_simplify_step';counts=cpu_v=cpu_f=None
            try:
                counts=mesh.simplify_step(1e-2,1e-3,threshold,False)
                phase='CUDA_synchronize';torch.cuda.synchronize()
                phase='native_read';v,f=mesh.read()
                phase='vertices_CPU_readback';cpu_v=v.cpu().numpy()
                phase='faces_CPU_readback';cpu_f=f.cpu().numpy()
                return cpu_v,cpu_f,counts
            except Exception as error:raise NativeStepError(phase,error,counts,cpu_v,cpu_f) from error
        def checkpoint(v, f, steps):
            atomic_npz(args.output, v, f)
            report.update(status='last_valid_checkpoint_not_final', final_vertices=len(v), final_faces=len(f), steps=steps, output_sha256=file_sha(args.output))
            atomic_json(args.report, report)
        def diagnostic(failure):
            atomic_json(args.report.with_name(args.report.stem+'.rejected-step.json'),failure)
            report['rejected_step']=failure;atomic_json(args.report,report)
        v,f,facts=bounded_steps(native_v,native_f,args.target,step,max_steps=args.max_steps,deadline=started+args.seconds,checkpoint=checkpoint,smoke=args.smoke,diagnostic=diagnostic,source_counts=(len(vertices),len(faces)),is_oom=lambda error:is_torch_oom(error,torch))
        if args.input and file_sha(args.input) != input_sha: raise RuntimeError('Original NPZ changed during GPU execution')
        policy=terminal_policy(facts)
        if policy['usable_mesh']:
            report['phase']='independent_final_indexed_topology'
            report['final_topology']=final_topology(v,f,prepared)
            atomic_npz(args.output,v,f)
        report.update(facts,status=policy['status'],usable_mesh=policy['usable_mesh'],output_role=policy['output_role'],
            output=str(args.output), input_sha256=input_sha, output_sha256=file_sha(args.output), native_extension=str(Path(_C.__file__).resolve()),
            cpu_threads=torch.get_num_threads(), library_origins=origins, python=sys.version, python_executable=sys.executable, torch=torch.__version__, CUDA=torch.version.cuda, device=torch.cuda.get_device_name(0), compute_capability=list(torch.cuda.get_device_capability(0)),
            target_faces=args.target, scheduler=dict(lambda_edge_length=.01, lambda_skinny=.001, initial_threshold=1e3 if args.smoke else 1e-8, threshold_cap=1e3, maximum_steps=args.max_steps, threshold_is_geometric_error_bound=False),
            elapsed_seconds=time.monotonic()-started, torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(), torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, process_memory_scope='Linux process peak RSS; not total GPU/unified memory',
            memory_scope='Torch allocator only; excludes CuMesh raw cudaMalloc and total device/system memory', geometry_scope='Every-step typed/count/finite/nonzero-area; independent initial/final indexed manifold, boundary collar and Euler admission; exact representation only, no collision/quality guarantee. Parent original surface/component/material/error caps remain mandatory')
        atomic_json(args.report, report)
        if policy['fatal_error']:raise RuntimeError(facts['error'])
        print(json.dumps(dict(status=report['status'], report=str(args.report))), flush=True)
    except Exception as error:
        if isinstance(error,TopologyValidationError):report['topology_failure']=dict(code=error.code,stats=error.stats)
        oom=is_torch_oom(error,torch);code=getattr(error,'code',None)
        report.update(status='failed',error=str(error),error_class=type(error).__name__,error_code=TORCH_OOM if oom else None if code==TORCH_OOM else code,usable_mesh=False,output_role='diagnostic_input_or_checkpoint_only',elapsed_seconds=time.monotonic()-started)
        atomic_json(args.report, report)
        if oom:raise CuMeshOutOfMemory() from error
        raise


if __name__ == '__main__':
    if sys.argv[1:2] == ['--serve-fd']:
        parser = argparse.ArgumentParser()
        parser.add_argument('--serve-fd', type=int, required=True)
        parser.add_argument('--parent-pid', type=int, required=True)
        args = parser.parse_args()
        serve(args.serve_fd, args.parent_pid)
    else:
        main()
