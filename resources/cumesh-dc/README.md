# Optional CuMesh dual-contour runtime

Explicitly prepare with `python cumesh_dc_runtime.py configure --allow-build
--blender <qualified Blender 5.2 ELF>`. Use the actual `bpy.app.binary_path`
executable, not a shell launcher such as `/usr/local/bin/blender`. The same
locked ELF identity is required by configure and the distance worker. The
current lock identifies one locally built Blender 5.2.0 LTS executable from
revision `fbe6228777e7`; see `blender_sha256` in [the runtime lock](runtime-lock.json).
Another executable with the same Blender version is not accepted unless its
bytes match. The pinned Blender binary is not provided here. Without it,
this optional backend cannot be configured by the current recipe; no
qualification for other Blender builds is claimed. No inherited
`LD_LIBRARY_PATH` or launcher environment is needed. This is a local source build, not a
downloaded opaque CLI or an automatic setup hook. It requires the already-ready
extension-owned CPython 3.12.16/Torch 2.12+cu130 runtime (libraries only, no CuMesh
simplification calls), local CUDA 13.0 SDK/GCC/Ninja, and the initially qualified
Linux ARM64/glibc 2.39+/GB10 sm121/driver580+ lane. No network, package installation,
global environment change, model or weight is used by this recipe.

`source.tar` contains the complete pinned MIT CuMesh `d10e54` hash/grid subset,
the qualified connected-contour CUDA extension, its source patch provenance,
and the owned Python driver/CPU Blender BVH worker. Only three required CUDA
units and four bindings are built; no `.bvh`, cuBVH, Warp, research subtree or
old single-vertex contour fallback is imported or built. Preserve the bundled
CuMesh `LICENSE`/Jianfeng XIANG attribution. New project controllers retain
`CONTROLLER-LICENSE` (GPL-3.0-or-later); externally installed Blender and shared
Torch/CUDA keep their own licenses. This is not a blanket distribution clearance.

The internal driver consumes staged position/triangle-only OBJ and writes real
binary little-endian PLY plus bounded metrics. It requires the parent's verified
Blender binary and native digest, direct parent PID, explicit `--retain-all`,
`--band 1`, a dyadic resolution (32/64/128), <=8 CPU threads and <=180 seconds.
Resolution 128 is qualified only for a bounded local-hole repair case.
Other resolution-128 workloads remain unqualified. There is no signed
distance, implicit shell removal, geometry cleanup, UV/material/normal identity
or target triangle-budget guarantee. Unsigned inputs can create inner and outer
closed shells; both are retained. The common parent topology/vertex-fan and
actual GLB reimport gates remain mandatory after this geometry proposal.
Explicit local-hole repair can also request a bounded local witness grid
after a whole-component contact refusal. Its distances still use the full
original component. Open grid-border edges are accepted only as part of
the internal witness when they are proven clear of the protected region;
the final repair still must pass the original closed-mesh checks.

Nearest distance is **CPU** `BVHTree.FromPolygons(all_triangles=True, epsilon=0)`
in one owned headless Blender process per rebuild, with bounded sparse typed
query batches. Contour construction/compaction is native CUDA/Torch. Metrics
separate CPU query work and wrapper wall time; kernel-only timing and global
self-intersections are not claimed. No <=128-face backend limit is imposed;
that bound belongs only to the independent verification reference. Finite source,
array, cell, contour, query, aggregate and deadline limits may explicitly refuse
an unsupported case, never omit faces or default-fill distances.

Configure creates an atomic ready marker only after a real open-mesh CPU-BVH /
CUDA reconstruction and independent closed/oriented/vertex-link/field checks.
Built payload hashes are then verified by `resolve_runtime`, without rebuilding
or installing on import/default Blender execution. Failed roots/logs are kept;
there is no hidden retry, arbitrary CLI registration or ready-marker copying.
One readiness smoke and the prior fixed small cases do not qualify all geometry,
visual/PBR fidelity, large-mesh performance, other platforms or public install.
