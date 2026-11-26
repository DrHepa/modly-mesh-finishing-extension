# Third-party notices

Mesh Finishing's integration and Blender worker are GPL-3.0-or-later.
The external Blender executable and its glTF importer are not bundled.
The glTF-Blender-IO importer credits its authors under Apache-2.0.

## Blender raster arithmetic

`production_tangents.py` adapts Blender's affected-triangle raster arithmetic
from revision `fbe6228777e7`. The modified
implementation restricts scanning to affected triangles and checks coverage
against Blender's native bake results. Preserve the attribution in that file.

- [zbuf.cc](https://github.com/blender/blender/blob/fbe6228777e7/source/blender/render/intern/zbuf.cc):
  © 2001–2002 NaN Holding BV; GPL-2.0-or-later; source SHA-256
  `b8872591b10bbb6150553d7a4c73dabe1acf4a1218f8080a30c7550eca1de1fc`.
- [bake.cc](https://github.com/blender/blender/blob/fbe6228777e7/source/blender/render/intern/bake.cc):
  © 2023 Blender Authors; GPL-2.0-or-later; source SHA-256
  `b3b88befc3edd2ab83d20d7137e7e12858aaf92e5d98cbce7b6a49d4dc597baf`.

## NumPy

The private installer uses the exact official NumPy 1.26.4 wheels in
[the lock](numpy_runtime_lock.json). No NumPy wheel is shipped. Preserve the
[upstream wheel licences](third_party/numpy-1.26.4/licenses/), including the
notices for bundled OpenBLAS, GCC runtime libraries and their exceptions
where present.

## Optional native backends

- **CuMesh core/atlas:** a modified MIT-licensed selection from
  [CuMesh](https://github.com/visualbruno/CuMesh/tree/d10e54c30ddd03d11472c1431693f985501c7966) by Jianfeng XIANG.
  The [corresponding source and build recipe](third_party/cumesh/selected-sources-core-topology2.tar.gz),
  [member identities](third_party/cumesh/build-provenance.json) and [notice
  inventory](third_party/cumesh/CORE_ONLY_NOTICES.txt) accompany the Linux ARM64
  wheel. xatlas retains its [licence](third_party/cumesh/xatlas-LICENSE), and
  embedded Thekla, Fast-BVH, OpenNL and CUDA header notices are retained.
  The topology2 change is described in the notice and patch. No cuBVH source
  or native extension is included. The wheel instantiates headers from CUDA
  CCCL v3.0.1 (CUDA package `cuda-cccl-13-0` 13.0.85-1). The exact upstream
  [CCCL licence](third_party/cumesh/CCCL-v3.0.1-LICENSE.txt) includes the
  Apache-2.0 terms and LLVM exception for libcu++ and the relevant BSD terms.
  Keep that text and the exact CUB copyright notices with the wheel.
- **CuMesh DC:** the [optional build recipe](resources/cumesh-dc/README.md)
  includes its MIT subset source and GPL-3.0-or-later controllers with separate
  licence texts. It uses external Blender for CPU distance queries.
- **WTiVo:** the [optional build recipe](resources/wtivo/README.md) includes
  the linked source selection and upstream notices. The combined WTiVo/CGAL
  work is GPL-3.0-or-later. The pinned OpenVDB 10 is MPL-2.0 and Trimesh 4.10.1
  is MIT. The upstream archive's later OpenVDB 12 table does not describe the
  OpenVDB 10 runtime. The installer retains the locked Ubuntu package notices.

The optional runtime fetches its locked Python, Torch and NVIDIA packages
separately. CUDA SDKs and drivers remain external. Their terms are not replaced
by the extension licence. No redistribution or legal clearance is asserted.
