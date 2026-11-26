# Optional WTiVo source-built runtime

This experimental lane is Linux ARM64/glibc 2.39+, NVIDIA GB10 (sm121), driver
580+, CUDA 13.0 SDK, GCC/CMake/Ninja. It is not a portable binary distribution.
`wtivo_runtime.py configure --allow-build` explicitly downloads the hash-locked
Ubuntu ARM64 development/runtime archives and Trimesh wheel, extracts them only
under this extension's `tools/wtivo`, and compiles the pinned native sources.
No sudo, system package installation or default setup hook is used. It requires
the already-ready owned CuMesh CPython/Torch runtime **as read-only libraries**;
WTiVo never calls CuMesh geometry. Core Blender nodes do not require this tool.

The recipe preserves WTiVo commit `7a0ac7cb24fd0588f2f7a78c787779d760d36f48`
inside `upstream.tar`, with its complete license/attribution files. The patch
adds Linux build plumbing, shared pybind3 headers, owned Eigen-return storage and a validated proxy-point
resource option; omitted proxy count retains upstream's 12-million default.
It also recomputes split-sheet QEF anchors/normals from each sheet's own
crossings and realizes distinct shared-face contour arcs before legacy topology
finalization. These are geometry changes, not merely build adaptations.
This is source-build reproducibility, not a promise of byte-identical compiler
outputs. Locally built payload hashes and one real bounded reconstruction smoke
are recorded before an atomic ready marker; resolve checks those identities.

Licensing: the combined WTiVo/CGAL package is GPL-3.0-or-later, not an unrestricted
MIT-only backend. Its upstream `LICENSE`, `LICENSING.md`, `NOTICE`,
`THIRD_PARTY_NOTICES.md` and `LICENSES/` remain in the source archive/runtime.
The actual pinned **OpenVDB 10 is MPL-2.0**, not the license of later OpenVDB
releases. Ubuntu dependency copyright files stay under `sysroot/usr/share/doc`;
Trimesh 4.10.1 is MIT and retains its wheel license. Torch/CUDA attribution and
license obligations remain those of the separately owned pinned runtime.
The upstream OpenVDB 12 license table remains unchanged; it does not describe
this pinned OpenVDB 10 dependency. Retained sysroot copyright files are outside
the runtime payload hash inventory; the shared Torch `NOTICE` is present but
not separately recorded in its readiness license inventory. These checks add
no legal clearance, CPython 3.11 or Windows qualification.

The native proof does not qualify visual/PBR fidelity, Modly distribution,
all input geometries, 12-million-point workloads or untested platforms. Upstream
Linux CPU memory counters are not available; zero values must not be presented
as measured per-job memory. Configure failures preserve their owned root/logs;
there is no automatic deletion, fallback, environment repair or hidden retry.
