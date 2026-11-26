# SPDX-License-Identifier: GPL-3.0-or-later
"""Read an actual source-built WTiVo smoke output; no geometry repair/export."""
import hashlib
import gc
import json
from pathlib import Path
import runpy
import sys

root = Path(sys.argv[1]).resolve()
modules = runpy.run_path(str(root / 'source/wtivo.py'), run_name='wtivo_smoke_inspection')
np, torch, trimesh = (modules[key] for key in ('np', 'torch', 'trimesh'))
if (sys.version_info[:3] != (3, 12, 16) or np.__version__ != '2.4.6'
        or torch.__version__ != '2.12.0+cu130' or trimesh.__version__ != '4.10.1'):
    raise RuntimeError('WTiVo shared Python/Torch/NumPy/Trimesh ABI differs from qualified lane')
if (not Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        or not Path(np.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        or not Path(trimesh.__file__).resolve().is_relative_to(root / 'source')):
    raise RuntimeError('WTiVo shared/package import origin is not owned')
paths = {name: Path(modules[key].__file__).resolve() for name, key in
         (('core', 'core'), ('vdb', 'vdb'), ('gpu', 'gpupr'))}
if not all(path.is_relative_to(root / 'source/build') for path in paths.values()):
    raise RuntimeError('WTiVo native module origin is not owned')
# A C-contiguous input requires a temporary for Eigen's column-major Ref.
# The bridge must return its own matrix, not a view into that temporary.
source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [.2, .2, .2]], dtype=np.float64, order='C')
expected = source.copy()
expected_bits = expected.tobytes(order='C')
returned, tets, neighbors = modules['core'].tetrahedralize_neighbors(source, 3)
if (not source.flags.c_contiguous or np.shares_memory(returned, source)
        or (not returned.flags.owndata and type(returned.base).__name__ != 'PyCapsule')
        or returned.dtype != np.float64 or returned.tobytes(order='C') != expected_bits
        or tets.dtype != np.int32 or neighbors.dtype != np.int32
        or tets.shape != (4, 4) or neighbors.shape != tets.shape or tets.min() < 0 or tets.max() >= len(source)):
    raise RuntimeError('WTiVo Eigen bridge failed owned-array/type/index admission')
tet_before, neighbor_before = tets.copy(), neighbors.copy()
source[:] = 9876.; del source; gc.collect()
allocations = [np.full(expected.shape, float(i), dtype=np.float64, order='F') for i in range(8)]
if (returned.tobytes(order='C') != expected_bits or not np.array_equal(tets, tet_before)
        or not np.array_equal(neighbors, neighbor_before)):
    raise RuntimeError('WTiVo Eigen bridge values changed after source lifetime/allocation stress')
empty, empty_tets, empty_neighbors = modules['core'].tetrahedralize_neighbors(np.empty((0, 3), dtype=np.float64), 3)
if (empty.dtype != np.float64 or empty.shape != (0, 3) or empty_tets.shape != (0, 4) or empty_neighbors.shape != (0, 4)
        or (not empty.flags.owndata and type(empty.base).__name__ != 'PyCapsule')):
    raise RuntimeError('WTiVo Eigen bridge empty return does not have owned backing')
mesh = trimesh.load_mesh(root / 'smoke.ply', process=False)
vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
if (vertices.ndim != 2 or faces.ndim != 2 or vertices.shape[1] != 3 or faces.shape[1] != 3
        or not len(vertices) or not len(faces) or not np.isfinite(vertices).all()
        or faces.min() < 0 or faces.max() >= len(vertices)):
    raise RuntimeError('WTiVo actual smoke output invalid')
tri = vertices[faces]
if not (np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) > 0).all():
    raise RuntimeError('WTiVo smoke has degenerate triangles')
edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
balance = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
if not (counts == 2).all() or (balance != 0).any() or len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
    raise RuntimeError('WTiVo smoke output not a closed oriented unique-face triangle mesh')
maps = Path('/proc/self/maps').read_text().splitlines()
vdb = [line.split()[-1] for line in maps if 'libopenvdb.so' in line]
if not vdb or not all(Path(path).resolve().is_relative_to(root / 'sysroot') and 'libopenvdb.so.10.0.1' in path for path in vdb):
    raise RuntimeError('WTiVo did not load the owned OpenVDB10 ABI')
proof = dict(scope='one real open-mesh 4096-point/res32 native smoke, not general quality or platform support',
             vertices=len(vertices), triangles=len(faces), boundary_or_nonmanifold_edges=0,
             orientation_conflicts=0, python='3.12.16', numpy=np.__version__, torch=torch.__version__,
             trimesh=trimesh.__version__, device=torch.cuda.get_device_name(0),
             eigen_bridge=dict(owned_backing=True, C_contiguous_input=True, source_lifetime_stress=True,
                               ordinary_allocations=8, vertex_bits_preserved=True, typed_tetra_geometry_preserved=True,
                               early_return_owned=True),
             output_sha256=hashlib.sha256((root / 'smoke.ply').read_bytes()).hexdigest(),
             process_memory='NOT_MEASURED: upstream CPU counters are Windows-only')
(root / 'smoke-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof))
