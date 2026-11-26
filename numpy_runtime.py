# SPDX-License-Identifier: GPL-3.0-or-later
"""Pinned host-only NumPy setup; pure imports and scalar admission stay stdlib.

Blender owns its bundled NumPy. This module never installs into Blender or
silently changes interpreter/site-packages. Runtime guards do not install.
"""
import hashlib,json,os,re,stat,subprocess,tempfile,uuid
from pathlib import Path

VERSION='1.26.4'
ROOT=Path(__file__).resolve().parent
HINT='Original-UV diagnostics require NumPy 1.26.4. Run this extension setup.py (see README Installation) to provision its private venv; local-folder linking does not run setup. Do not enable system-site-packages or switch interpreters.'
MAX_PIP_DIAGNOSTIC_CHARS=2048


def private_venv_paths(ext_dir):
    """Admit physical environment directories before executing Python or pip.

    Resolve directories, not the interpreter file: ordinary CPython venvs
    legitimately link their Python executable to the base interpreter.
    """
    directory=Path(ext_dir)/'venv'
    scripts=directory/('Scripts'if os.name=='nt'else'bin')
    for path in (directory,scripts):
        expected=Path(os.path.abspath(path))
        if (path.is_symlink() or path.resolve()!=expected or
                (path.exists() and not path.is_dir())):
            raise RuntimeError('Private environment requires physical venv and bin/Scripts directories; preserve linked or redirected directories and choose a physical extension environment')
    return directory,scripts/('python.exe'if os.name=='nt'else'python')


def _pip_diagnostic(value):
    """Bound captured output and remove URLs/labelled credentials before display."""
    if isinstance(value,bytes):value=value.decode('utf-8',errors='replace')
    if not isinstance(value,str):return ''
    value=''.join(c if c.isprintable()or c in '\n\t'else ' 'for c in value)
    value=re.sub(r'(?i)\b[a-z][a-z0-9+.-]*://\S+','<url>',value)
    value=re.sub(r'''(?im)\b([a-z0-9_-]*(?:password|passwd|token|secret|api[_-]?key)|(?:proxy[-_])?authorization)["']?\s*[:=][^\n]*''',r'\1=<redacted>',value)
    if len(value)<=MAX_PIP_DIAGNOSTIC_CHARS:return value
    marker='\n... [truncated] ...\n'
    available=MAX_PIP_DIAGNOSTIC_CHARS-len(marker);head=available//2;tail=available-head
    return value[:head]+marker+value[-tail:]


def _run_pip(command,*,env,action):
    try:return subprocess.run(command,check=True,text=True,capture_output=True,env=env)
    except subprocess.CalledProcessError as error:
        detail=f'Required private NumPy pip {action} failed (exit {error.returncode}). Resolve the reported pip error and rerun setup.'
        for label,value in (('stdout',error.stdout),('stderr',error.stderr)):
            captured=_pip_diagnostic(value)
            if captured:detail+='\n'+label+':\n'+captured
        # The original exception embeds its full command; never expose it.
        raise RuntimeError(detail)from None


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def require_host_numpy():
    try:import numpy
    except (ImportError,OSError)as error:raise RuntimeError(HINT)from error
    if numpy.__version__!=VERSION:raise RuntimeError(HINT+' Found '+str(numpy.__version__)+'.')
    return numpy


def load_lock():
    lock=json.loads((ROOT/'numpy_runtime_lock.json').read_text())
    if lock.get('version')!=VERSION or len(lock.get('wheels',[]))!=6:raise RuntimeError('Invalid required NumPy wheel lock')
    for row in lock['wheels']:
        if not row['url'].startswith('https://files.pythonhosted.org/')or not row['url'].endswith('/'+row['filename'])or \
                len(row['sha256'])!=64 or any(c not in '0123456789abcdef'for c in row['sha256'])or not row['filename'].endswith('.whl'):
            raise RuntimeError('Invalid required NumPy wheel/hash pin')
    return lock


def select_wheel(identity,lock=None):
    lock=load_lock()if lock is None else lock;system=identity.get('os');arch=identity.get('arch','').lower()
    arch={'amd64':'x86_64','arm64':'aarch64'}.get(arch,arch)
    supported=identity.get('implementation')=='CPython'and identity.get('minor')in('3.11','3.12')and identity.get('bits',64)==64
    if system=='linux':
        try:glibc=tuple(int(v)for v in identity['glibc'].split('.')[:2])
        except (KeyError,ValueError,AttributeError):glibc=()
        supported=supported and arch in('x86_64','aarch64')and identity.get('libc')=='glibc'and glibc>=(2,17)
    else:supported=supported and system=='win32'and arch=='x86_64'
    rows=[row for row in lock['wheels']if(row['minor'],row['os'],row['arch'])==(identity.get('minor'),system,arch)]
    if not supported or len(rows)!=1:raise RuntimeError('Required NumPy supports CPython 3.11/3.12 Windows x64 or Linux glibc>=2.17 x64/aarch64 only; no source build or interpreter fallback is supported.')
    return rows[0]


IDENTITY_CODE="""import json,platform,struct,sys
l,v=platform.libc_ver()
print(json.dumps(dict(implementation=platform.python_implementation(),minor='%s.%s'%sys.version_info[:2],os=sys.platform,arch=platform.machine().lower(),bits=8*struct.calcsize('P'),libc=l,glibc=v,prefix=sys.prefix,base_prefix=sys.base_prefix)))"""
PROBE_CODE="""import hashlib,importlib.metadata,json,sys
from pathlib import Path
import numpy as np
from numpy.core import _multiarray_umath as native
root=Path(sys.argv[1]).resolve()
assert Path(sys.prefix).resolve()==root and sys.prefix!=sys.base_prefix,'Not the private extension venv'
assert np.__version__=='1.26.4','Required NumPy version differs'
module=Path(np.__file__).resolve();core=Path(native.__file__).resolve()
assert module.is_relative_to(root) and core.is_relative_to(root),'NumPy escaped the private venv'
a=np.arange(6,dtype=np.int64).reshape(2,3)
assert a.flags.c_contiguous and (a@a.T).tolist()==[[5,14],[14,50]],'Native NumPy array operation failed'
dist=importlib.metadata.distribution('numpy');license=Path(dist.locate_file('numpy-1.26.4.dist-info/LICENSE.txt')).resolve()
assert license.is_relative_to(root),'NumPy license escaped private venv'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(dict(version=np.__version__,module=str(module),native_module=str(core),native_sha256=sha(core),license_sha256=sha(license),native_operation=[[5,14],[14,50]],prefix=sys.prefix)))"""


def _probe(python,venv_dir,env):
    try:result=subprocess.run([str(python),'-I','-B','-c',PROBE_CODE,str(venv_dir)],check=True,text=True,capture_output=True,env=env)
    except subprocess.CalledProcessError as error:
        captured=_pip_diagnostic(error.stderr).strip()
        detail=f'Required private NumPy native verification failed (exit {error.returncode}).'
        detail+='\nstderr:\n'+captured if captured else ' No stderr was captured.'
        raise RuntimeError(detail)from None
    row=json.loads(result.stdout)
    if row.get('version')!=VERSION or row.get('native_operation')!=[[5,14],[14,50]]:raise RuntimeError('Required private NumPy native verification failed')
    return row


def _stage_locked_wheel(value,row,stage_dir):
    """Copy an explicitly supplied physical wheel; only verified staged bytes reach pip."""
    if type(value)is not str or not value:raise ValueError('numpy_wheel_path must be an explicit absolute wheel file path')
    source=Path(value)
    if not source.is_absolute()or source.name!=row['filename']:
        raise ValueError('numpy_wheel_path must be absolute and match the selected locked wheel filename')
    if source.resolve(strict=True)!=source or source.is_symlink():
        raise ValueError('numpy_wheel_path must identify a physical file without symlink aliases')
    fd=os.open(source,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode)or info.st_size!=row['bytes']:
            raise ValueError('numpy_wheel_path must be a regular file with the selected locked byte count')
        current=source.stat()
        if source.resolve(strict=True)!=source or source.is_symlink()or (current.st_dev,current.st_ino)!=(info.st_dev,info.st_ino):
            raise ValueError('numpy_wheel_path identity changed before staging')
        destination=stage_dir/row['filename'];digest=hashlib.sha256();size=0
        stream=os.fdopen(fd,'rb');fd=None
        with stream,destination.open('xb')as target:
            os.chmod(destination,0o600)
            while True:
                block=stream.read(min(1024**2,row['bytes']-size+1))
                if not block:break
                size+=len(block)
                if size>row['bytes']:raise ValueError('numpy_wheel_path exceeds its locked byte count')
                digest.update(block);target.write(block)
            target.flush();os.fsync(target.fileno())
        if size!=row['bytes']or digest.hexdigest()!=row['sha256']:
            raise ValueError('numpy_wheel_path bytes differ from the selected locked wheel')
        return destination.resolve().as_uri()
    finally:
        if fd is not None:os.close(fd)


def prepare_numpy(ext_dir,venv_python,*,numpy_wheel_path=None):
    """Idempotent pinned private install + actual native probe + strict pip check.

    Receipt is installation provenance, not SOURCE/native/visual authority.
    A prior receipt permits reuse only with the same freshly probed native file.
    numpy_wheel_path is an explicit installer-only input, never auto-discovered.
    The official URL remains wheel identity; a local install report must instead
    identify the verified private staged file URI used for that pip invocation.
    """
    ext_dir=Path(ext_dir);venv_python=Path(venv_python)
    directory,expected_python=private_venv_paths(ext_dir)
    if venv_python!=expected_python:
        raise RuntimeError('NumPy installer requires the exact extension private venv Python')
    cfg=(directory/'pyvenv.cfg').read_text().lower()
    site=[line.partition('=')[2].strip()for line in cfg.splitlines()if line.partition('=')[0].strip()=='include-system-site-packages']
    if site!=['false']:raise RuntimeError('Private venv must disable system-site-packages exactly once; setup will not modify its configuration')
    # Remove inherited pip target/user/index configuration. os.devnull disables
    # all pip config files; -I disables Python user site and PYTHONPATH.
    env={k:v for k,v in os.environ.items()if not k.upper().startswith(('PIP_','PYTHON'))}
    env.update(PIP_CONFIG_FILE=os.devnull,PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_NO_INPUT='1')
    identity=json.loads(subprocess.run([str(venv_python),'-I','-B','-c',IDENTITY_CODE],check=True,text=True,capture_output=True,env=env).stdout)
    if Path(identity.get('prefix','')).resolve()!=directory.resolve()or identity.get('prefix')==identity.get('base_prefix'):
        raise RuntimeError('NumPy installer interpreter is not the extension private venv')
    row=select_wheel(identity);receipt_path=directory/'numpy_runtime_provenance.json';previous=None;probe=None
    try:
        if receipt_path.stat().st_size<=65536:previous=json.loads(receipt_path.read_text())
    except (OSError,ValueError):pass
    try:probe=_probe(venv_python,directory,env)
    except (subprocess.CalledProcessError,ImportError,OSError,ValueError,RuntimeError):pass
    pin={'filename':row['filename'],'url':row['url'],'sha256':row['sha256'],'bytes':row['bytes']}
    reused=type(previous)is dict and previous.get('selected_wheel')==pin and previous.get('native_probe')==probe and probe is not None
    if not reused or numpy_wheel_path is not None:
        with tempfile.TemporaryDirectory(prefix='.numpy-setup-',dir=directory)as staging:
            stage_dir=Path(staging)
            transport_url=row['url']if numpy_wheel_path is None else _stage_locked_wheel(numpy_wheel_path,row,stage_dir)
            if not reused:
                requirements=stage_dir/'requirements.txt'
                requirements.write_text('numpy @ '+transport_url+' --hash=sha256:'+row['sha256']+'\n',encoding='utf-8')
                report_path=stage_dir/'pip-report.json'
                _run_pip([str(venv_python),'-I','-B','-m','pip','install','--no-index','--only-binary=:all:',
                    '--no-deps','--require-hashes','--no-cache-dir','--force-reinstall','--report',str(report_path),'-r',str(requirements)],
                    env=env,action='install')
                if report_path.stat().st_size>1024**2:raise RuntimeError('NumPy pip provenance report exceeds limit')
                installs=json.loads(report_path.read_text()).get('install',[])
                if len(installs)!=1 or installs[0].get('metadata',{}).get('version')!=VERSION or \
                        installs[0].get('download_info',{}).get('url')!=transport_url or \
                        installs[0].get('download_info',{}).get('archive_info',{}).get('hashes',{}).get('sha256')!=row['sha256']:
                    raise RuntimeError('Required NumPy pip wheel/hash provenance mismatch')
    if not reused:
        probe=_probe(venv_python,directory,env)
    if probe.get('license_sha256')not in {v['sha256']for v in row['licenses']}:
        raise RuntimeError('Installed NumPy license differs from the selected official wheel')
    _run_pip([str(venv_python),'-I','-B','-m','pip','check'],env=env,action='check')
    receipt={'schema_version':1,'dependency':'numpy','version':VERSION,'selected_wheel':pin,'native_probe':probe,
        'strict_pip_check':True,'metadata_source':'https://pypi.org/pypi/numpy/1.26.4/json',
        'lock_sha256':file_sha(ROOT/'numpy_runtime_lock.json'),'license_inventory':row['licenses'],
        'scope':'private host dependency only; not Blender, SOURCE, PBR or platform qualification'}
    stage=directory/('.numpy-provenance-'+uuid.uuid4().hex+'.json')
    try:
        with stage.open('x',encoding='utf-8')as stream:stream.write(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
        stage.replace(receipt_path)
    finally:stage.unlink(missing_ok=True)
    return receipt
