# SPDX-License-Identifier: GPL-3.0-or-later
"""Explicit source-built WTiVo tool; import/resolve never installs anything."""
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
import zipfile


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _owned(path, root):
    path, root = Path(path), Path(root)
    if root.absolute() != root.resolve() or not path.resolve().is_relative_to(root):
        raise RuntimeError('WTiVo owned path redirected; suspicious files preserved: ' + str(path))
    return path


def _json(path, value):
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _recipe(extension=None):
    extension = Path(extension or Path(__file__).resolve().parent).resolve()
    resources = _owned(extension / 'resources/wtivo', extension)
    lock = json.loads((resources / 'runtime-lock.json').read_text())
    identity = dict(lock=lock, manager=_sha(extension / 'wtivo_runtime.py'))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]
    tools = _owned(extension / 'tools/wtivo', extension)
    return extension, resources, lock, fingerprint, _owned(tools / fingerprint, extension)


def _platform():
    libc, version = platform.libc_ver()
    if (platform.system() != 'Linux' or platform.machine() != 'aarch64'
            or libc != 'glibc' or tuple(map(int, version.split('.')[:2])) < (2, 39)):
        raise RuntimeError('Optional WTiVo currently requires Linux ARM64/glibc >=2.39, NVIDIA GB10/CUDA13; core tools remain available')
    executable = shutil.which('nvidia-smi', path='/usr/bin:/bin')
    if not executable:
        raise RuntimeError('WTiVo requires the qualified NVIDIA driver; nvidia-smi unavailable')
    result = subprocess.run([executable, '--query-gpu=driver_version,name,compute_cap', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=5, env={'PATH': '/usr/bin:/bin', 'LANG': 'C'})
    rows = [line.split(',') for line in result.stdout.strip().splitlines()]
    if (result.returncode or len(rows) != 1 or len(rows[0]) != 3
            or int(rows[0][0].strip().split('.')[0]) < 580 or rows[0][1].strip() != 'NVIDIA GB10'
            or rows[0][2].strip() != '12.1'):
        raise RuntimeError('WTiVo lane is qualified only for one NVIDIA GB10 sm121 with driver >=580; no automatic fallback')
    return dict(driver=rows[0][0].strip(), device=rows[0][1].strip(), compute_capability='12.1', glibc=version)


def _shared(extension):
    import cumesh_runtime
    python = Path(cumesh_runtime.ensure_runtime(allow_install=False, extension=extension)).absolute()
    _owned(python, extension)
    return python


def _environment(root):
    # Only this child changes environment. No user/global CUDA, Python or pip edits.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('PYTHON', 'LD_', 'PIP_', 'CUDA_', 'TORCH_', 'WTIVO_', 'CMAKE_'))
           and key not in ('CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'CPATH', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH')}
    env.update(PATH='/usr/local/cuda-13.0/bin:/usr/bin:/bin', HOME=str(root), LANG='C.UTF-8',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='3',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='3', MAX_JOBS='3',
               LD_LIBRARY_PATH=str(root / 'sysroot/usr/lib/aarch64-linux-gnu'),
               XDG_CACHE_HOME=str(root / 'cache'), TORCH_HOME=str(root / 'cache/torch'),
               CUDA_CACHE_PATH=str(root / 'cache/cuda'), TMPDIR=str(root / 'tmp'),
               CUDA_HOME='/usr/local/cuda-13.0', WTIVO_CUDA_HOME='/usr/local/cuda-13.0', WTIVO_CUDA_ARCH='12.1+PTX')
    return env


def _payload(root):
    # Runtime files only: source, Python package, native modules and native libraries.
    paths = list((root / 'source').rglob('*')) + list((root / 'sysroot/usr/lib/aarch64-linux-gnu').glob('*.so*'))
    paths = [p for p in paths if p.is_file() and '.build' not in p.relative_to(root).parts]
    if len(paths) > 10000:
        raise RuntimeError('WTiVo runtime file inventory exceeds admission limit')
    return {str(p.relative_to(root)): dict(sha256=_sha(_owned(p, root)), size=p.stat().st_size) for p in sorted(paths)}


def resolve_runtime(*, extension=None):
    """Return pinned owned python/script/env or an actionable refusal; no provisioning."""
    extension, resources, lock, fingerprint, root = _recipe(extension)
    marker = _owned(root / 'ready.json', extension)
    if not marker.is_file():
        raise RuntimeError('WTiVo optional runtime is not prepared. Explicitly run: python wtivo_runtime.py configure --allow-build (requires existing owned Torch runtime and local GCC/CMake/Ninja/CUDA13 SDK)')
    hardware = _platform()
    python = _shared(extension)
    if marker.stat().st_size > 8192:
        raise RuntimeError('WTiVo readiness marker exceeds admission limit')
    ready = json.loads(marker.read_text())
    manifest = _owned(root / 'payload.json', root)
    if (ready.get('fingerprint') != fingerprint or ready.get('python_sha256') != _sha(python)
            or manifest.stat().st_size > 2 * 1024 * 1024 or _sha(manifest) != ready.get('payload_sha256')):
        raise RuntimeError('WTiVo readiness identity changed; explicitly reconfigure in a fresh recipe root')
    files = json.loads(manifest.read_text())
    if not isinstance(files, dict) or not 1 <= len(files) <= 10000:
        raise RuntimeError('Invalid WTiVo runtime inventory')
    if _payload(root) != files:
        raise RuntimeError('WTiVo runtime contents changed (including unexpected importable files)')
    for name, digest in lock['patched_files'].items():
        if _sha(_owned(root / 'source' / name, root)) != digest:
            raise RuntimeError('WTiVo pinned source identity changed: ' + name)
    # Check writable child directories before returning an environment that uses them.
    import cumesh_runtime
    for name in ('cache', 'tmp'):
        cumesh_runtime._owned_tree(root / name, root)
    return dict(python=str(python), script=str(root / 'source/wtivo.py'), env=_environment(root), root=str(root),
                provenance=dict(backend='WTiVo', fingerprint=fingerprint, source_commit=lock['source_commit'],
                                shared_torch='read-only owned CuMesh runtime libraries; no CuMesh geometry calls',
                                proof=ready['proof'], hardware=hardware, license='GPL-3.0-or-later; OpenVDB10 MPL-2.0'))


def configure(*, allow_build=False, extension=None, timeout_seconds=600):
    """Explicit local source build, never setup/default-node execution or global install."""
    if not allow_build:
        raise RuntimeError('WTiVo source compilation requires explicit --allow-build')
    if type(timeout_seconds) is not int or not 60 <= timeout_seconds <= 1800:
        raise ValueError('WTiVo preparation timeout must be 60..1800 seconds')
    extension, resources, lock, fingerprint, root = _recipe(extension)
    _platform(); python = _shared(extension)
    if (root / 'ready.json').exists():
        return resolve_runtime(extension=extension)
    import cumesh_runtime as shared
    tools = root.parent; tools.mkdir(parents=True, exist_ok=True); _owned(tools, extension)
    import fcntl
    guard_path = _owned(tools / 'configure.lock', tools)
    if guard_path.is_symlink(): raise RuntimeError('WTiVo configure lock must not be a symlink')
    with guard_path.open('a+b') as guard:
        try: fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another WTiVo configuration owns the build lock')
        if root.exists():
            raise RuntimeError('Incomplete WTiVo preparation preserved at ' + str(root) + '; inspect before any explicit removal/retry')
        root.mkdir(); (root / 'tmp').mkdir(); (root / 'cache/cuda').mkdir(parents=True)
        deadline = time.monotonic() + timeout_seconds; env = _environment(root)
        def run(argv, label):
            remaining = deadline - time.monotonic()
            if remaining <= 0: raise TimeoutError('WTiVo explicit preparation deadline exceeded')
            with _owned(root / (label + '.log'), root).open('xb') as stream:
                result = subprocess.run(list(map(str, argv)), cwd=root, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=remaining)
            if result.returncode: raise RuntimeError('WTiVo ' + label + ' failed; see ' + str(root / (label + '.log')))
        try:
            if shutil.disk_usage(tools).free < 1024 ** 3:
                raise RuntimeError('WTiVo build requires at least 1 GiB free in owned tools')
            for spec in lock['bundled']:
                path = _owned(resources / spec['filename'], resources)
                if path.stat().st_size != spec['size_bytes'] or _sha(path) != spec['sha256']:
                    raise RuntimeError('WTiVo bundled resource identity differs: ' + spec['filename'])
            for name in ('cmake', 'ninja', 'g++', 'dpkg-deb', 'patch'):
                if not shutil.which(name, path='/usr/bin:/bin'):
                    raise RuntimeError('WTiVo requires local build tool ' + name + '; no system installation is attempted')
            if not Path('/usr/local/cuda-13.0/bin/nvcc').is_file():
                raise RuntimeError('WTiVo requires locally provisioned CUDA13 SDK; no SDK installation is attempted')
            cache = _owned(tools / 'downloads', tools)
            packages = [shared.artifact(spec, resources, cache, deadline, owned_root=tools) for spec in lock['deb_artifacts']]
            sysroot = root / 'sysroot'; sysroot.mkdir()
            for ordinal, package in enumerate(packages):
                run(['/usr/bin/dpkg-deb', '-x', package, sysroot], 'extract-' + str(ordinal))
            shared._owned_tree(sysroot, root)
            source = root / 'source'; source.mkdir()
            with tarfile.open(resources / 'upstream.tar') as archive:
                archive.extractall(source, filter='data')
            shared._owned_tree(source, root)
            run(['/usr/bin/patch', '-p1', '-i', resources / 'linux-resource.patch', '-d', source], 'source-patch')
            for name, digest in lock['patched_files'].items():
                if _sha(source / name) != digest: raise RuntimeError('Patched source differs: ' + name)
            wheel = shared.artifact(lock['trimesh'], resources, cache, deadline, owned_root=tools)
            with zipfile.ZipFile(wheel) as archive:
                if any(not (name.startswith('trimesh/') or name.startswith('trimesh-4.10.1.dist-info/'))
                       or '..' in Path(name).parts for name in archive.namelist()):
                    raise RuntimeError('Unexpected Trimesh wheel layout')
                archive.extractall(source)
            headers = python.parent.parent / 'lib/python3.12/site-packages/torch/include'
            if _sha(headers / 'pybind11/detail/common.h') != lock['pybind_header_sha256']:
                raise RuntimeError('Shared Torch pybind3 headers differ from qualified recipe')
            prefix = sysroot / 'usr'
            run(['/usr/bin/cmake', '-S', source, '-B', root / 'build', '-G', 'Ninja',
                 '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF',
                 '-DCMAKE_PREFIX_PATH=' + str(prefix),
                 '-DCMAKE_MODULE_PATH=' + str(prefix / 'lib/aarch64-linux-gnu/cmake/OpenVDB'),
                 '-DPython3_EXECUTABLE=' + str(python), '-DWTIVO_PYBIND11_INCLUDE=' + str(headers),
                 '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
                 '-DCMAKE_INSTALL_RPATH=$ORIGIN/../../sysroot/usr/lib/aarch64-linux-gnu',
                 '-DWTIVO_OUTPUT_DIR=' + str(source / 'build')], 'cmake')
            run(['/usr/bin/cmake', '--build', root / 'build', '--target', 'wtivo_core', 'wtivo_vdb', '--parallel', '3'], 'cpu-build')
            run([python, '-I', '-B', source / 'scripts/build_gpupr.py'], 'gpu-build')
            run([python, '-I', '-B', source / 'wtivo.py', '--input', resources / 'smoke.obj', '--output', root / 'smoke.ply',
                 '--proxy-points', '4096', '--input-res', '32', '--final-res', '32', '--threads', '3', '--max_rounds', '4096'], 'native-smoke')
            run([python, '-I', '-B', resources / 'inspect_smoke.py', root], 'inspect-smoke')
            proof = json.loads((root / 'smoke-proof.json').read_text())
            _json(root / 'payload.json', _payload(root))
            _json(root / 'ready.json', dict(fingerprint=fingerprint, python_sha256=_sha(python),
                                           payload_sha256=_sha(root / 'payload.json'), proof=proof))
        except Exception as error:
            _json(root / 'failed.json', dict(ready=False, error=str(error), error_class=type(error).__name__))
            raise
    return resolve_runtime(extension=extension)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['configure', 'resolve'])
    parser.add_argument('--allow-build', action='store_true')
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    runtime = configure(allow_build=args.allow_build, timeout_seconds=args.timeout) if args.command == 'configure' else resolve_runtime()
    print(json.dumps({key: value for key, value in runtime.items() if key != 'env'}, indent=2))
