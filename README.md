# Mesh Finishing for Modly

Mesh Finishing is a PROCESS extension by **DrHepa**. It inspects, repairs and
simplifies GLB meshes, creates UVs and transfers PBR materials using an external
Blender. CuMesh simplification, CuMesh Dual Contouring and WTiVo are optional.
Default setup does not install Blender, CUDA, optional backends or model weights.

## Install

Use the directory configured as Modly's `extensionsDir`. Copy this checkout's
release files into a new physical `mesh-finishing` subdirectory, without
overwriting an existing installation. From that directory run, substituting
your actual absolute Modly Python executable:

```sh
"$MODLY_PYTHON" -c 'import json,subprocess,sys; from pathlib import Path; root=Path.cwd(); subprocess.run([sys.executable,str(root/"setup.py"),json.dumps({"python_exe":sys.executable,"ext_dir":str(root)})],check=True)'
```

It uses a private Python 3.11 or 3.12 environment and verified NumPy 1.26.4.
For offline setup, pass the absolute path of the matching locked wheel as the
`numpy_wheel_path` JSON key. Restart or reload Modly after setup. Install
Blender separately and put it on PATH or set `BLENDER_EXECUTABLE` for Modly to
the actual Blender executable. Blender 5.2 and the Modly 0.5 source build were
tested on Linux ARM64. Native Windows/Linux x64, public GitHub installation
and stock GUI Repair are not qualified. Stock Workflow Stop and live
per-stage labels require host support.

## Use

In Workflows choose **Import**, open a template from [workflows](workflows/),
and select your GLB in its empty input. **Optimize & Rebake** combines reduction,
UV generation and PBR transfer. If a bake declines a layout, try a single
moderate map resolution without relaxing verification. A valid result may
exceed the triangle budget; this is reported separately from failure.

**Inspect Mesh** returns the original bytes and logs a summary. **Conservative
Repair** removes exactly degenerate triangle indices. **Simplify Mesh** retains
a valid already-within-budget input unchanged; a changed output has no UV/PBR.
**UV / Repack + Rebake** generates new UVs. **Rebake onto Mesh** preserves the
LOW's supplied UVs and requires an explicit distinct textured HIGH. The
original is never overwritten. By default, only a validated GLB is published
and private scratch is removed; failed checks publish nothing. Set
`MODLY_MESH_FINISHING_KEEP_DIAGNOSTICS=1` only if you need a full job report.

## Optional backends

- **CuMesh:** select the explicit first-use GPU option or run
  `python cumesh_runtime.py --allow-install`. The native lane is Linux ARM64,
  NVIDIA GB10 (SM121), CUDA 13 and the locked private CPython 3.12.16/PyTorch
  capsule. Installed-only mode never installs a runtime. GPU errors never
  switch back to Blender automatically.
- **[CuMesh Dual Contouring](resources/cumesh-dc/README.md)** and
  **[WTiVo](resources/wtivo/README.md)** are separately configured source-built
  tools using the owned optional Python/Torch runtime. They do not run
  automatically during optimization. Changed reconstruction results invalidate
  UV/PBR; use an intact textured HIGH for any later rebake.

## Credits and licence

- Modly integration: **DrHepa**. Modly is by **Lightning Pixel**.
- External Blender and selected CuMesh, xatlas and WTiVo sources retain their
  attribution. See [Third-party notices](THIRD_PARTY_NOTICES.md).
- The extension is [GPL-3.0-or-later](LICENSE). GPL does not prohibit commercial
  use; distribution remains subject to the applicable terms of the extension
  and each dependency. Optional CUDA/Torch components retain their own terms.
