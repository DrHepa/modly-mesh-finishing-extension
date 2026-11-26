# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional source-built CuMesh DC. No import-time install, Torch or Blender calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _owned(path,root):
    path,root=Path(path),Path(root)
    if root.absolute()!=root.resolve()or not path.resolve().is_relative_to(root):raise RuntimeError('CuMesh DC owned path redirected; preserved: '+str(path))
    return path


def _atomic(path,value):
    fd,name=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def _recipe(extension=None):
    extension=Path(extension or Path(__file__).resolve().parent).resolve()
    resources=_owned(extension/'resources/cumesh-dc',extension);lock=json.loads((resources/'runtime-lock.json').read_text())
    identity=dict(lock=lock,manager=_sha(extension/'cumesh_dc_runtime.py'))
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24]
    return extension,resources,lock,fingerprint,_owned(extension/'tools/cumesh-dc'/fingerprint,extension)


def _hardware():
    libc,version=platform.libc_ver()
    if platform.system()!='Linux'or platform.machine()!='aarch64'or libc!='glibc'or tuple(map(int,version.split('.')[:2]))<(2,39):raise RuntimeError('CuMesh DC initial source-build lane requires Linux ARM64/glibc2.39+, GB10 sm121; other nodes remain available')
    binary=shutil.which('nvidia-smi',path='/usr/bin:/bin')
    if not binary:raise RuntimeError('Qualified NVIDIA driver is unavailable')
    result=subprocess.run([binary,'--query-gpu=driver_version,name,compute_cap','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,env={'PATH':'/usr/bin:/bin','LANG':'C'})
    rows=[line.split(',')for line in result.stdout.strip().splitlines()]
    if result.returncode or len(rows)!=1 or len(rows[0])!=3 or int(rows[0][0].strip().split('.')[0])<580 or rows[0][1].strip()!='NVIDIA GB10'or rows[0][2].strip()!='12.1':raise RuntimeError('CuMesh DC requires the qualified single GB10 sm121/driver580+ lane; no fallback')
    return dict(device='NVIDIA GB10',compute_capability='12.1',driver=rows[0][0].strip(),glibc=version)


def _shared(extension):
    import cumesh_runtime
    return _owned(Path(cumesh_runtime.ensure_runtime(allow_install=False,extension=extension)).absolute(),extension)


def _env(root):
    env={k:v for k,v in os.environ.items()if not k.startswith(('PYTHON','PYTORCH_','LD_','PIP_','CUDA_','NVCC_','TORCH_','BLENDER_','CUMESH_DC_','CMAKE_'))and k not in ('CC','CXX','CFLAGS','CPPFLAGS','CXXFLAGS','LDFLAGS','CPATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','NVCC','CUDACXX','CUDAHOSTCXX')}
    env.update(PATH='/usr/local/cuda-13.0/bin:/usr/bin:/bin',HOME=str(root),LANG='C.UTF-8',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
               CUDA_HOME='/usr/local/cuda-13.0',CUDA_INC_PATH='/usr/local/cuda-13.0/include/cccl',TORCH_CUDA_ARCH_LIST='12.1+PTX',
               CC='/usr/bin/gcc',CXX='/usr/bin/g++',MAX_JOBS='3',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='8',
               TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),CUDA_CACHE_PATH=str(root/'cache/cuda'),TORCH_EXTENSIONS_DIR=str(root/'cache/extensions'))
    if 'CUDA_VISIBLE_DEVICES'in os.environ:env['CUDA_VISIBLE_DEVICES']=os.environ['CUDA_VISIBLE_DEVICES']
    return env


def _files(root):
    paths=list((root/'source').rglob('*'))+list((root/'lib').glob('*.so'))
    files=[p for p in paths if p.is_file()and '.build'not in p.parts]
    if len(files)>1000:raise RuntimeError('CuMesh DC runtime inventory admission exceeded')
    return {str(p.relative_to(root)):dict(size=p.stat().st_size,sha256=_sha(_owned(p,root)))for p in sorted(files)}


def resolve_runtime(*,extension=None):
    """Only an already built/smoke-qualified owned tool; never bootstrap on resolve."""
    extension,resources,lock,fingerprint,root=_recipe(extension)
    marker=_owned(root/'ready.json',extension)
    if not marker.is_file():raise RuntimeError('Optional CuMesh DC is not prepared. Explicitly run python cumesh_dc_runtime.py configure --allow-build --blender <qualified Blender path>; requires existing owned Torch and local CUDA13/GCC/Ninja')
    hardware=_hardware();python=_shared(extension)
    manifest=_owned(root/'payload.json',root)
    if marker.stat().st_size>16384 or manifest.stat().st_size>1024**2:raise RuntimeError('CuMesh DC readiness/inventory exceeds admission')
    ready=json.loads(marker.read_text())
    if ready.get('fingerprint')!=fingerprint or ready.get('python_sha256')!=_sha(python)or ready.get('payload_sha256')!=_sha(manifest):raise RuntimeError('CuMesh DC owned readiness identity changed')
    if _files(root)!=json.loads(manifest.read_text()):raise RuntimeError('CuMesh DC owned contents changed, including unexpected importable code')
    library=_owned(root/'lib/_h3_dc_contours.cpython-312-aarch64-linux-gnu.so',root)
    if _sha(library)!=ready['native_sha256']:raise RuntimeError('CuMesh DC compiled binary changed')
    import cumesh_runtime
    for directory in ('cache','tmp'):cumesh_runtime._owned_tree(root/directory,root)
    return dict(python=str(python),script=str(root/'source/dc_driver.py'),env=_env(root),native_sha256=ready['native_sha256'],root=str(root),
                provenance=dict(backend='CuMesh CUDA connected contours',distance='separate CPU Blender BVH',fingerprint=fingerprint,
                    source_commit=lock['source_commit'],hardware=hardware,proof=ready['proof'],blender_sha256=lock['blender_sha256'],
                    native_qualified_resolutions=[32,64],resource_admitted_resolutions=[32,64,128],resolution128='NOT_NATIVE_QUALIFIED',
                    shared_torch='read-only owned runtime libraries; no CuMesh simplification calls',self_intersections='NOT_EVALUATED'))


def configure(*,allow_build=False,extension=None,blender=None,timeout_seconds=600):
    """Explicit local compilation only; no network, package installation or ready copying."""
    if not allow_build:raise RuntimeError('CuMesh DC source build requires explicit --allow-build')
    if type(timeout_seconds)is not int or not 60<=timeout_seconds<=600:raise ValueError('CuMesh DC preparation timeout must be60..600s')
    extension,resources,lock,fingerprint,root=_recipe(extension);_hardware();python=_shared(extension)
    blender=Path(blender or '')
    if not blender.is_absolute()or not blender.is_file()or _sha(blender)!=lock['blender_sha256']:raise RuntimeError('Explicit parent-verified qualified Blender binary is required; no arbitrary executable')
    if (root/'ready.json').exists():return resolve_runtime(extension=extension)
    tools=root.parent;tools.mkdir(parents=True,exist_ok=True);_owned(tools,extension)
    import fcntl
    guard_path=_owned(tools/'configure.lock',tools)
    if guard_path.is_symlink():raise RuntimeError('CuMesh DC configure lock must not be a symlink')
    with guard_path.open('a+b')as guard:
        try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another caller owns CuMesh DC configuration')
        if root.exists():raise RuntimeError('Incomplete CuMesh DC root preserved; inspect before explicit removal/new recipe: '+str(root))
        root.mkdir();(root/'tmp').mkdir();(root/'cache/cuda').mkdir(parents=True);deadline=time.monotonic()+timeout_seconds;env=_env(root)
        def run(argv,label,limit,*,cwd=root):
            remaining=min(limit,deadline-time.monotonic())
            if remaining<=0:raise TimeoutError('CuMesh DC configure deadline exceeded')
            with _owned(root/(label+'.log'),root).open('xb')as log:
                result=subprocess.run(list(map(str,argv)),cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
            if result.returncode:raise RuntimeError('CuMesh DC '+label+' failed; see '+str(root/(label+'.log')))
        try:
            for tool in ('g++','gcc','ninja'):
                if not shutil.which(tool,path='/usr/bin:/bin'):raise RuntimeError('Local '+tool+' required; no installation attempted')
            if not Path('/usr/local/cuda-13.0/bin/nvcc').is_file():raise RuntimeError('Local CUDA13 SDK required; no installation attempted')
            if shutil.disk_usage(tools).free<512*1024**2:raise RuntimeError('CuMesh DC source build requires512MiB free')
            archive=_owned(resources/'source.tar',resources)
            if archive.stat().st_size!=lock['source_archive']['size']or _sha(archive)!=lock['source_archive']['sha256']:raise RuntimeError('Bundled CuMesh DC source identity changed')
            source=root/'source';source.mkdir()
            with tarfile.open(archive)as stream:stream.extractall(source,filter='data')
            import cumesh_runtime
            cumesh_runtime._owned_tree(source,root)
            if {str(p.relative_to(source)):_sha(p)for p in source.rglob('*')if p.is_file()}!=lock['source_files']:raise RuntimeError('CuMesh DC source closure differs')
            run([python,'-I','-B',source/'setup.py','build_ext','--build-temp',root/'build','--build-lib',root/'lib'],'native-build',300,cwd=source)
            library=root/'lib/_h3_dc_contours.cpython-312-aarch64-linux-gnu.so'
            if not library.is_file():raise RuntimeError('Expected native ABI artifact missing')
            native_sha=_sha(library)
            run([python,'-I','-B',source/'dc_driver.py','--input',source/'smoke.obj','--output',root/'smoke.ply','--metrics',root/'smoke-metrics.json',
                 '--resolution','32','--band','1','--retain-all','--seconds','180','--threads','8','--parent-pid',os.getpid(),
                 '--blender',blender,'--native-sha256',native_sha,'--capture-field'],'native-smoke',180)
            run([python,'-I','-B',source/'inspect_smoke.py',root],'inspect-smoke',30)
            proof=json.loads((root/'smoke-proof.json').read_text())
            _atomic(root/'payload.json',_files(root))
            _atomic(root/'ready.json',dict(fingerprint=fingerprint,python_sha256=_sha(python),native_sha256=native_sha,payload_sha256=_sha(root/'payload.json'),proof=proof))
        except BaseException as error:
            _atomic(root/'failed.json',dict(ready=False,error_class=type(error).__name__,error=str(error)));raise
    return resolve_runtime(extension=extension)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('configure','resolve'));parser.add_argument('--allow-build',action='store_true');parser.add_argument('--blender');parser.add_argument('--timeout',type=int,default=600);args=parser.parse_args()
    result=configure(allow_build=args.allow_build,blender=args.blender,timeout_seconds=args.timeout)if args.command=='configure'else resolve_runtime()
    print(json.dumps({k:v for k,v in result.items()if k!='env'},indent=2))
