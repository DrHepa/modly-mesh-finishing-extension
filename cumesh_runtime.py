# SPDX-License-Identifier: GPL-3.0-or-later
"""Explicitly requested, extension-owned optional CuMesh runtime. No import-time I/O."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def atomic_json(path, value, *, owned_root=None):
    path = Path(path)
    if owned_root is not None: _owned(path, owned_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if owned_root is not None: _owned(path, owned_root)
    fd, name = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def _inside(path, root):
    if not path.resolve().is_relative_to(root.resolve()): raise RuntimeError('Refusing symlink redirection of owned path: '+str(path))
    return path


def _owned(path, root):
    """Check against a previously verified canonical anchor, never a cache alias."""
    root = Path(root); path = Path(path)
    if root.resolve() != root.absolute(): raise RuntimeError('Owned anchor was redirected; suspicious path preserved: '+str(root))
    return _inside(path, root)


def _owned_tree(path, root):
    _owned(path, root)
    if path.exists():
        for count, child in enumerate(path.rglob('*')):
            if count >= 200000: raise RuntimeError('Owned tree exceeds symlink admission scan limit')
            _owned(child, root)


def _new_smoke_attempt(root):
    _owned(root, root)
    return _owned(Path(tempfile.mkdtemp(prefix='smoke-attempt-', dir=root)), root)


def _platform_check():
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise RuntimeError('Optional CuMesh supports only the prepared Linux AArch64 GB10/CUDA13 lane; Blender core is unaffected')
    name, version = platform.libc_ver()
    if name != 'glibc' or tuple(map(int, version.split('.')[:2])) < (2, 28): raise RuntimeError('CuMesh requires glibc >=2.28')
    if not hasattr(tarfile, 'data_filter'): raise RuntimeError('Caller lacks secure tar data-filter support; no bootstrap extraction permitted')


def recipe(extension=None):
    extension = Path(extension or Path(__file__).resolve().parent).resolve()
    resources = extension/'resources/cumesh'
    lock = json.loads((resources/'runtime-lock.json').read_text())
    bootstrap = json.loads((resources/'python-bootstrap.json').read_text())
    identity = dict(lock=lock, bootstrap=bootstrap, manager=file_sha(extension/'cumesh_runtime.py'), helper=file_sha(extension/'cumesh_simplify.py'))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]
    return extension, resources, lock, bootstrap, fingerprint


def _https(url):
    value = urllib.parse.urlsplit(url)
    if value.scheme != 'https' or not value.hostname or value.username or value.password: raise ValueError('Only credential-free HTTPS artifacts are allowed')
    return url


class _HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        _https(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def artifact(spec, resources, cache, deadline, *, owned_root):
    filename = spec['filename']
    if Path(filename).name != filename or filename in ('.', '..') or '\\' in filename: raise ValueError('Unsafe artifact filename')
    if 'bundled' in spec:
        path = _inside(resources/spec['bundled'], resources)
        if not path.is_file() or path.stat().st_size != spec['size_bytes'] or file_sha(path) != spec['sha256']: raise RuntimeError('Bundled core wheel identity differs')
        return path
    _https(spec['url']); _owned(cache, owned_root)
    directory = _owned(cache/spec['sha256'], owned_root); directory.mkdir(parents=True, exist_ok=True)
    path = _owned(directory/filename, owned_root)
    if path.exists():
        if path.stat().st_size != spec['size_bytes'] or file_sha(path) != spec['sha256']: raise RuntimeError('Cached artifact is corrupt; preserved for inspection: '+str(path))
        return path
    if time.monotonic() >= deadline: raise TimeoutError('Optional runtime download deadline exceeded')
    fd, name = tempfile.mkstemp(prefix='.download-', dir=directory)
    try:
        h = hashlib.sha256(); count = 0
        with os.fdopen(fd, 'wb') as stream, urllib.request.build_opener(_HTTPSRedirect()).open(spec['url'], timeout=30) as response:
            _https(response.geturl())
            while True:
                if time.monotonic() >= deadline: raise TimeoutError('Optional runtime download deadline exceeded')
                block = response.read(1024*1024)
                if not block: break
                count += len(block)
                if count > spec['size_bytes']: raise RuntimeError('Artifact exceeds locked size')
                h.update(block); stream.write(block)
            stream.flush(); os.fsync(stream.fileno())
        if count != spec['size_bytes'] or h.hexdigest() != spec['sha256']: raise RuntimeError('Downloaded artifact hash/size differs')
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)
    return path


def extract_python(archive, root):
    """Fresh destination only; bounded data-filter extraction, retaining all licenses."""
    _owned(root, root)
    if (root/'python').exists(): raise RuntimeError('Incomplete bootstrap retained; inspect/remove only this owned recipe before retry: '+str(root))
    with tarfile.open(archive, 'r:gz') as stream:
        members = []; total = 0
        for member in stream:
            total += member.size; members.append(member)
            if len(members) > 40000 or total > 1024**3: raise RuntimeError('Python archive exceeds extraction admission')
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0] != 'python': raise RuntimeError('Unsafe Python archive path')
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()): raise RuntimeError('Unsupported Python archive entry')
        stream.extractall(root, members=members, filter='data')


def _licenses(root):
    values = []
    for p in root.rglob('*'):
        if p.is_file() and p.name.upper().startswith(('LICENSE', 'COPYING')):
            _inside(p, root); values.append(dict(path=str(p.relative_to(root)), sha256=file_sha(p)))
    if not values: raise RuntimeError('No actual bundled license texts found; runtime will not be marked ready')
    return values


def _environment(root):
    _owned_tree(root/'cache', root)
    for child in ('cache/torch', 'cache/cuda', 'cache/triton'): _owned(root/child, root)
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(('PIP_','LD_')): env.pop(name)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUDA_HOME', 'CUDA_PATH'): env.pop(name, None)
    env.update(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='8', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_CONFIG_FILE=os.devnull,
               XDG_CACHE_HOME=str(root/'cache'), TORCH_HOME=str(root/'cache/torch'), CUDA_CACHE_PATH=str(root/'cache/cuda'), TRITON_CACHE_DIR=str(root/'cache/triton'))
    return env


def _call(argv, root, label, deadline, env, check=True, *, log_directory=None):
    remaining = deadline-time.monotonic()
    if remaining <= 0: raise TimeoutError('Optional runtime deadline exceeded')
    log = _owned((log_directory or root)/(label+'.log'), root)
    log.parent.mkdir(parents=True, exist_ok=True); _owned(log, root)
    with log.open('wb') as stream:
        result = subprocess.run(list(map(str, argv)), stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=remaining)
    if check and result.returncode: raise RuntimeError(f'{label} failed ({result.returncode}); see {log}')
    return result


def _ready(root, fingerprint, core):
    marker = _owned(root/'ready.json', root); python = root/'venv/bin/python'
    _inside(python, root)
    if not marker.exists(): return None
    _owned_tree(root/'venv', root)
    value = json.loads(marker.read_text())
    binary = _inside(root/value['core_relative_path'], root)
    if (value.get('fingerprint') != fingerprint or not python.is_file() or not binary.is_file()
        or file_sha(python) != value['python_sha256'] or file_sha(binary) != core['native_extension_sha256']):
        raise RuntimeError('Owned readiness identity changed; core Blender remains available')
    _inside(python, root)
    return python


def _repair_cusparselt_tag(root, python, wheel, spec, issue, deadline, env):
    """One exact published sbsa-tag defect; two before-images define recoverable states."""
    import base64
    import csv
    import io
    import zipfile
    if (platform.system()!='Linux' or platform.machine()!='aarch64' or spec['name']!='nvidia-cusparselt-cu13'
        or spec['version']!='0.8.1' or issue['package']!=spec['name'] or issue['version']!=spec['version']
        or file_sha(wheel)!=spec['sha256'] or spec['sha256']!=issue['wheel_sha256']):raise RuntimeError('Unknown cuSPARSELt repair identity/platform')
    site=root/'venv/lib/python3.12/site-packages';info='nvidia_cusparselt_cu13-0.8.1.dist-info'
    wn=info+'/WHEEL';rn=info+'/RECORD';transaction=_owned(root/'cusparselt-tag-repair',root)
    _owned_tree(transaction,root);_owned_tree(site/'nvidia/cusparselt',root);_owned_tree(site/info,root)
    with zipfile.ZipFile(wheel) as archive:
        original=archive.read(wn);original_record=archive.read(rn);names=set(archive.namelist())
        if (len(archive.infolist())!=7 or file_sha(site/info/'METADATA')!=issue['metadata_sha256']
            or hashlib.sha256(original).hexdigest()!=issue['internal_WHEEL_sha256']
            or hashlib.sha256(original_record).hexdigest()!=issue['internal_RECORD_sha256']):raise RuntimeError('Original cuSPARSELt metadata differs')
    changed=original.replace(b'Tag: py3-none-manylinux2014_sbsa\n',b'Tag: py3-none-manylinux2014_aarch64\n')
    if changed==original or hashlib.sha256(changed).hexdigest()!=issue['repaired_WHEEL_sha256']:raise RuntimeError('Unexpected cuSPARSELt tag edit')
    before_wheel=transaction/'WHEEL.before';before_record=transaction/'RECORD.before'
    current_wheel=(site/wn).read_bytes();current_record=(site/rn).read_bytes()
    if before_wheel.exists() and before_wheel.read_bytes()!=original:raise RuntimeError('Corrupt repair before-image WHEEL')
    if not before_record.exists() and current_wheel!=original:raise RuntimeError('Repaired tag without prior RECORD before-image')
    baseline=before_record.read_bytes() if before_record.exists() else current_record
    parse=lambda data:list(csv.reader(io.StringIO(data.decode(),newline='')))
    rows=parse(baseline);original_rows=parse(original_record)
    if any(len(row)!=3 for row in rows+original_rows):raise RuntimeError('Invalid cuSPARSELt RECORD row')
    entries={row[0]:row[1:] for row in rows};origin_entries={row[0]:row[1:] for row in original_rows}
    extras={info+'/'+name for name in ('INSTALLER','REQUESTED','direct_url.json')}
    if (len(entries)!=len(rows) or len(origin_entries)!=len(original_rows) or set(origin_entries)!=names
        or set(entries)!=names|extras or entries.get(rn)!=['','']):raise RuntimeError('Unknown cuSPARSELt RECORD members')
    files={str(p.relative_to(site)) for parent in (site/'nvidia/cusparselt',site/info) for p in parent.rglob('*') if p.is_file()}
    if files!=set(entries):raise RuntimeError('Unrecorded or missing cuSPARSELt payload')
    for name,(encoded,size) in entries.items():
        path=_owned(site/name,root)
        if name==rn:continue
        if name in origin_entries and [encoded,size]!=origin_entries[name]:raise RuntimeError('Original RECORD payload identity differs')
        actual=original if name==wn else path.read_bytes() if name in extras else None
        got=hashlib.sha256(actual).digest() if actual is not None else bytes.fromhex(file_sha(path))
        count=len(actual) if actual is not None else path.stat().st_size
        if encoded!='sha256='+base64.urlsafe_b64encode(got).decode().rstrip('=') or size!=str(count):raise RuntimeError('Installed cuSPARSELt payload/RECORD differs: '+name)
    direct=json.loads((site/info/'direct_url.json').read_text())
    if ((site/info/'INSTALLER').read_bytes()!=b'pip\n' or (site/info/'REQUESTED').read_bytes()!=b''
        or direct!={'archive_info':{'hash':'sha256='+spec['sha256'],'hashes':{'sha256':spec['sha256']}},'url':wheel.resolve().as_uri()}):raise RuntimeError('Unknown pip provenance for repair')
    new_row=wn+',sha256='+base64.urlsafe_b64encode(hashlib.sha256(changed).digest()).decode().rstrip('=')+','+str(len(changed))
    record_lines=baseline.splitlines(keepends=True);matched=[i for i,line in enumerate(record_lines) if line.startswith((wn+',').encode())]
    if len(matched)!=1:raise RuntimeError('Ambiguous WHEEL RECORD row')
    index=matched[0];ending=b'\r\n' if record_lines[index].endswith(b'\r\n') else b'\n' if record_lines[index].endswith(b'\n') else b''
    record_lines[index]=new_row.encode()+ending;repaired_record=b''.join(record_lines)
    if current_wheel not in (original,changed) or current_record not in (baseline,repaired_record):raise RuntimeError('Unknown interrupted cuSPARSELt repair state')
    native=_owned(site/issue['native_relative_path'],root)
    with native.open('rb') as stream:header=stream.read(20)
    if (file_sha(native)!=issue['native_sha256'] or header[:6]!=b'\x7fELF\x02\x01'
        or int.from_bytes(header[18:20],'little')!=183):raise RuntimeError('cuSPARSELt native hash/AArch64 ELF mismatch')
    cwd=_owned(root/'cusparselt-loader-empty',root);cwd.mkdir(exist_ok=True)
    if any(cwd.iterdir()):raise RuntimeError('Owned loader cwd must be empty (native RUNPATH includes cwd)')
    _owned(python,root)
    loader_env={k:v for k,v in env.items() if not k.startswith('LD_')}
    code="import ctypes,os,sys,pathlib; assert sys.version_info[:3]==(3,12,16); assert pathlib.Path(sys.prefix).resolve()==pathlib.Path(sys.argv[2]).resolve(); ctypes.CDLL(sys.argv[1],mode=os.RTLD_NOW|os.RTLD_LOCAL)"
    remaining=min(30.,deadline-time.monotonic())
    if remaining<=0:raise TimeoutError('cuSPARSELt verification deadline exceeded')
    with _owned(root/'cusparselt-native-load.log',root).open('wb') as log:
        loaded=subprocess.run([str(python),'-I','-B','-c',code,str(native),str(root/'venv')],cwd=cwd,env=loader_env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
    if loaded.returncode:raise RuntimeError('Owned isolated cuSPARSELt CDLL load failed; metadata untouched')
    def write(path,data):
        _owned(path,root);fd,temp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
            os.replace(temp,path)
        finally:
            if os.path.exists(temp):os.unlink(temp)
    transaction.mkdir(exist_ok=True)
    for path,data in ((before_wheel,original),(before_record,baseline)):
        if path.exists():
            if path.read_bytes()!=data:raise RuntimeError('Corrupt cuSPARSELt repair before-image')
        else:write(path,data)
    write(site/wn,changed);write(site/rn,repaired_record)
    atomic_json(transaction/'result.json',dict(package=spec['name'],version=spec['version'],original_wheel_sha256=spec['sha256'],
        before_RECORD_sha256=hashlib.sha256(baseline).hexdigest(),after_RECORD_sha256=hashlib.sha256(repaired_record).hexdigest(),
        before_WHEEL_sha256=issue['internal_WHEEL_sha256'],after_WHEEL_sha256=issue['repaired_WHEEL_sha256'],native_load_returncode=0,
        scope='Only known Tag and its RECORD row changed; original archive and all other row bytes preserved'),owned_root=root)


def ensure_runtime(*, allow_install=False, extension=None, timeout_seconds=3600):
    """Return owned venv Python; provisioning requires explicit consent, never setup/import."""
    extension, resources, lock, bootstrap, fingerprint = recipe(extension)
    _platform_check()
    tools = extension/'tools/cumesh'; root = tools/fingerprint
    tools = _inside(tools, extension).resolve(); root = tools/fingerprint
    _owned(root, tools)
    core = next(w for w in lock['wheels'] if w['name'] == 'cumesh')
    ready = _ready(root, fingerprint, core)
    if ready: return ready
    if not allow_install: raise RuntimeError('CuMesh tool is not installed. Explicitly enable optional tool installation; Blender core requires no GPU tool')
    if type(timeout_seconds) is not int or not 60 <= timeout_seconds <= 7200: raise ValueError('Runtime installation timeout must be 60..7200 seconds')
    import fcntl
    tools.mkdir(parents=True, exist_ok=True); _inside(tools, extension)
    if (tools/'provision.lock').is_symlink(): raise RuntimeError('Provision lock must not be a symlink')
    with _owned(tools/'provision.lock', tools).open('a+b') as guard:
        try: fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another process owns CuMesh provisioning; no concurrent installation started')
        ready = _ready(root, fingerprint, core)
        if ready: return ready
        root = _owned(root, tools); root.mkdir(exist_ok=True); _owned(root, tools)
        deadline = time.monotonic()+timeout_seconds; env = _environment(root); cache = tools/'downloads'
        try:
            if shutil.disk_usage(tools).free < 3*sum(w['size_bytes'] for w in lock['wheels'])+512*1024**2: raise RuntimeError('Insufficient free disk for locked wheels, extraction and private installation')
            archive = artifact(bootstrap['archive'], resources, cache, deadline, owned_root=tools)
            complete = _owned(root/'python-extracted.json', root)
            if not complete.exists():
                extract_python(archive, root)
                atomic_json(complete, dict(archive_sha256=bootstrap['archive']['sha256'], licenses=_licenses(root/'python')), owned_root=root)
            elif json.loads(complete.read_text())['archive_sha256'] != bootstrap['archive']['sha256']: raise RuntimeError('Bootstrap marker identity differs')
            _owned_tree(root/'python', root)
            base = _inside(root/bootstrap['interpreter'], root)
            if not base.is_file(): raise RuntimeError('Expected owned CPython entrypoint missing')
            identity = "import sys; assert sys.version_info[:3]==(3,12,16); assert sys.implementation.cache_tag=='cpython-312'"
            _call([base, '-I', '-B', '-c', identity], root, 'python-identity', deadline, env)
            venv = _owned(root/'venv', root); python = venv/'bin/python'
            _owned_tree(venv, root)
            if not venv.exists(): _call([base, '-I', '-B', '-m', 'venv', str(venv)], root, 'venv', deadline, env)
            _owned_tree(venv, root)
            config = (venv/'pyvenv.cfg').read_text()
            if 'include-system-site-packages = false' not in config: raise RuntimeError('Owned venv must not expose system packages')
            _inside(python, root)
            scope = "import sys,pathlib; assert pathlib.Path(sys.prefix).resolve()==pathlib.Path(sys.argv[1]).resolve(); assert pathlib.Path(sys.base_prefix).resolve()==pathlib.Path(sys.argv[2]).resolve()"
            _call([python, '-I', '-B', '-c', identity+'; '+scope, venv, root/'python'], root, 'venv-identity', deadline, env)
            wheels = [artifact(spec, resources, cache, deadline, owned_root=tools) for spec in lock['wheels']]
            requirements = _owned(root/'requirements.lock', root)
            pip_cache = _owned(tools/'pip-cache', tools); _owned_tree(pip_cache, tools)
            _owned_tree(venv, root)
            requirements.write_text(''.join(path.resolve().as_uri()+' --hash=sha256:'+spec['sha256']+'\n' for path, spec in zip(wheels, lock['wheels'])))
            _call([python, '-I', '-B', '-m', 'pip', '--isolated', '--disable-pip-version-check', 'install', '--no-index', '--no-deps', '--require-hashes', '--cache-dir', pip_cache, '-r', requirements], root, 'locked-install', deadline, env)
            checked = _call([python, '-I', '-B', '-m', 'pip', '--isolated', 'check'], root, 'pip-check', deadline, env, check=False)
            issue=lock['known_installation_issue']
            if checked.returncode and (root/'pip-check.log').read_text().strip()!=issue['expected_pip_check_line']:
                raise RuntimeError('Strict pip check failed with unknown/related conflicts; no metadata repair or ready marker')
            if checked.returncode or (root/'cusparselt-tag-repair').exists():
                index=next(i for i,w in enumerate(lock['wheels']) if w['name']==issue['package'])
                _repair_cusparselt_tag(root,python,wheels[index],lock['wheels'][index],issue,deadline,env)
                _call([python,'-I','-B','-m','pip','--isolated','check'],root,'pip-check-after-exact-repair',deadline,env)
            smoke_dir = _new_smoke_attempt(root)
            _call([python, '-I', '-B', extension/'cumesh_simplify.py', '--smoke', '--parent-pid', str(os.getpid()), '--output', smoke_dir/'mesh.npz', '--report', smoke_dir/'result.json', '--target', '64', '--max-steps', '1', '--seconds', '30'], root, 'native-smoke', min(deadline, time.monotonic()+60), env, log_directory=smoke_dir)
            smoke = json.loads((smoke_dir/'result.json').read_text())
            if smoke.get('error') or smoke['source_faces'] != 128 or smoke['source_vertices'] != 81 or not 0 < smoke['final_faces'] < 128: raise RuntimeError('Owned81-vertex/128-triangle GPU smoke did not prove actual reduction')
            binary = _inside(Path(smoke['native_extension']), venv)
            if file_sha(binary) != core['native_extension_sha256']: raise RuntimeError('Installed native core hash differs')
            atomic_json(root/'ready.json', dict(fingerprint=fingerprint, python_sha256=file_sha(python), core_relative_path=str(binary.relative_to(root)),
                        smoke_attempt=str(smoke_dir.relative_to(root)), smoke=smoke, licenses=dict(python=_licenses(root/'python'), installed=_licenses(venv/'lib/python3.12/site-packages'))), owned_root=root)
            return python
        except Exception as error:
            atomic_json(root/'failed.json', dict(fingerprint=fingerprint, error=str(error), ready=False), owned_root=root); raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--allow-install', action='store_true'); parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args(); print(json.dumps(dict(python=str(ensure_runtime(allow_install=args.allow_install, timeout_seconds=args.timeout)))))


if __name__ == '__main__': main()
