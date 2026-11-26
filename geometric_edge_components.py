# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded, exact-coordinate face components within one mesh object.

This is an incidence metric, not a weld or a proof of physical contact. Two
faces connect only when they share a complete geometric edge; a shared point
alone never connects them. Coordinates are compared exactly, with no tolerance.
"""

from __future__ import annotations

import math


MAX_TRIANGLE_CORNERS = 4_000_000
MAX_UNIQUE_EDGES = 3_000_000
MAX_POSITIONS = 4_000_000


def _root(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def analyze_geometric_edges(positions, triangles, *,
                            max_triangle_corners: int = MAX_TRIANGLE_CORNERS,
                            max_unique_edges: int = MAX_UNIQUE_EDGES,
                            max_positions: int = MAX_POSITIONS, check_cancelled=None) -> dict:
    """Return face-index groups and edge incidence, or an explicit unassessed result.

    ``positions`` and ``triangles`` are per-object sequences. Triangle indices
    may differ across UV/material/normal seams while coordinates coincide.
    Work limits are checked before or during allocation; no cross-object or
    tolerance-based matching is performed.
    """
    face_count = len(triangles)
    if face_count * 3 > max_triangle_corners:
        return {"assessment": "not_assessed", "reason": "triangle corner budget exceeded",
                "max_triangle_corners": max_triangle_corners, "triangles": face_count}
    if len(positions) > max_positions:
        return {"assessment": "not_assessed", "reason": "position budget exceeded",
                "max_positions": max_positions, "triangles": face_count}
    coordinate_ids: dict[tuple[float, float, float], int] = {}
    vertex_ids = []
    for row, point in enumerate(positions):
        if check_cancelled is not None and row % 4096 == 0:
            check_cancelled()
        key = tuple(float(value) for value in point)
        if len(key) != 3 or not all(math.isfinite(value) for value in key):
            return {"assessment": "not_assessed", "reason": "non-finite or non-3D position",
                    "triangles": face_count}
        if key not in coordinate_ids:
            coordinate_ids[key] = len(coordinate_ids)
        vertex_ids.append(coordinate_ids[key])
    parent = list(range(face_count))
    # First incident face and use count suffice; subsequent faces union to it.
    edges: dict[tuple[int, int], tuple[int, int]] = {}
    degenerate_edges = 0
    for face_index, triangle in enumerate(triangles):
        if check_cancelled is not None and face_index % 4096 == 0:
            check_cancelled()
        if len(triangle) != 3 or any(not isinstance(index, int) or isinstance(index, bool)
                                      or index < 0 or index >= len(vertex_ids) for index in triangle):
            return {"assessment": "not_assessed", "reason": "invalid triangle indices",
                    "triangles": face_count}
        points = [vertex_ids[index] for index in triangle]
        for offset in range(3):
            first, second = points[offset], points[(offset + 1) % 3]
            if first == second:
                degenerate_edges += 1
                continue
            edge = (min(first, second), max(first, second))
            previous = edges.get(edge)
            if previous is None:
                if len(edges) >= max_unique_edges:
                    return {"assessment": "not_assessed", "reason": "unique edge budget exceeded",
                            "max_unique_edges": max_unique_edges, "triangles": face_count}
                edges[edge] = (face_index, 1)
            else:
                first_face, uses = previous
                first_root, current_root = _root(parent, first_face), _root(parent, face_index)
                if first_root != current_root:
                    parent[current_root] = first_root
                edges[edge] = (first_face, uses + 1)
    groups: dict[int, list[int]] = {}
    for face_index in range(face_count):
        if check_cancelled is not None and face_index % 4096 == 0:
            check_cancelled()
        groups.setdefault(_root(parent, face_index), []).append(face_index)
    ambiguous_edges = sum(uses > 2 for _, uses in edges.values())
    return {"assessment": "ambiguous" if ambiguous_edges or degenerate_edges else "assessed",
            "triangles": face_count, "components": list(groups.values()),
            "component_count": len(groups), "unique_edges": len(edges),
            "boundary_edges": sum(uses == 1 for _, uses in edges.values()),
            "shared_two_face_edges": sum(uses == 2 for _, uses in edges.values()),
            "ambiguous_edges_more_than_two_faces": ambiguous_edges,
            "degenerate_edges": degenerate_edges,
            "component_semantics": "Exact-coordinate shared-edge face incidence within one mesh object; point-only contacts, separate objects, and near-coincident coordinates remain separate."}
