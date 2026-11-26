# SPDX-License-Identifier: GPL-3.0-or-later
"""Private Python environment for inspection; Blender owns its own dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys
from numpy_runtime import prepare_numpy, private_venv_paths


def setup(payload):
    root=Path(__file__).resolve().parent
    supplied=payload.get('ext_dir') or payload.get('extDir') or str(root)
    if Path(supplied).resolve()!=root:raise ValueError('ext_dir must identify this extension')
    if 'numpy_wheel_path'in payload and (type(payload['numpy_wheel_path'])is not str or not payload['numpy_wheel_path']):
        raise ValueError('numpy_wheel_path must be an explicit absolute wheel file path')
    target,executable=private_venv_paths(root)
    python=payload.get('python_exe') or payload.get('pythonExe') or sys.executable
    version=json.loads(subprocess.run([python,'-c','import json,sys;print(json.dumps(list(sys.version_info[:2])))'],
                                     check=True,text=True,capture_output=True).stdout)
    if version not in ([3,11],[3,12]):raise ValueError('Python 3.11 or 3.12 is required')
    if not executable.is_file():subprocess.run([python,'-m','venv',str(target)],check=True)
    actual=json.loads(subprocess.run([str(executable),'-c','import json,sys;print(json.dumps(list(sys.version_info[:2])))'],
                                    check=True,text=True,capture_output=True).stdout)
    if actual!=version:raise ValueError('Existing private environment uses another Python ABI; preserve it and choose the matching Python for Repair')
    prepare_numpy(root,executable,**({'numpy_wheel_path':payload['numpy_wheel_path']}if 'numpy_wheel_path'in payload else {}))
    print('Private inspection environment ready. Set BLENDER_EXECUTABLE to your Blender executable or add Blender to PATH. No native helpers or Blender add-ons are installed.',flush=True)
    return executable


if __name__=='__main__':
    if len(sys.argv)==2:payload=json.loads(sys.argv[1])
    elif len(sys.argv)>=4:payload=dict(python_exe=sys.argv[1],ext_dir=sys.argv[2])
    else:raise SystemExit('Expected one setup JSON argument or legacy Python/extension arguments')
    setup(payload)
