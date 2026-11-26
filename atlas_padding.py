# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic one-island atlas padding from a geometry-derived occupancy mask.

Only the validated low UV island supplies seeds. This copies real occupied
values (including black or zero alpha), never fabricates ray-hit coverage.
The entire bounded <=1024^2 tile is assigned one nearest 4-neighbor seed so
linear/trilinear sampling cannot blend a valid island edge with empty black.
"""

from __future__ import annotations

from array import array


_UNSET = 0xFFFFFFFF


def padding_owners(occupancy: bytearray, resolution: int) -> tuple[array, dict]:
    if not 1 <= resolution <= 1024 or len(occupancy) != resolution * resolution:
        raise ValueError("Atlas padding needs a bounded square occupancy mask")
    count = resolution * resolution
    owners = array("I", [_UNSET]) * count
    distance = array("H", [0]) * count
    queue = array("I")
    for index, occupied in enumerate(occupancy):
        if occupied:
            owners[index] = index
            queue.append(index)
    if not queue:
        raise ValueError("Atlas padding has no validated occupied seed")
    seed_count = len(queue)
    head = 0
    maximum_distance = 0
    while head < len(queue):
        index = queue[head]
        head += 1
        x, y = index % resolution, index // resolution
        next_distance = distance[index] + 1
        for neighbor in ((index - 1 if x else -1),
                         (index + 1 if x < resolution - 1 else -1),
                         (index - resolution if y else -1),
                         (index + resolution if y < resolution - 1 else -1)):
            if neighbor < 0 or owners[neighbor] != _UNSET:
                continue
            owners[neighbor] = owners[index]
            distance[neighbor] = next_distance
            maximum_distance = max(maximum_distance, next_distance)
            queue.append(neighbor)
    if len(queue) != count:
        raise ValueError("Atlas padding did not assign every tile texel")
    return owners, {"algorithm": "deterministic_four_neighbor_nearest_occupied",
                    "scope": "one validated geometric UV island; full bounded 0-1 tile",
                    "seed_texels": seed_count,
                    "padded_texels": count - seed_count,
                    "max_distance_texels": maximum_distance,
                    "source_recovery_texels": 0,
                    "coverage_and_ray_hits_unchanged": True}


def pad_rgba_pixels(values: array, owners: array, occupancy: bytearray) -> int:
    """Change only unoccupied RGBA texels; preserve occupied values bitwise."""
    if values.typecode != "f" or len(values) != len(owners) * 4 or len(owners) != len(occupancy):
        raise ValueError("Atlas padding pixel or owner dimensions differ")
    padded = 0
    for index, owner in enumerate(owners):
        if occupancy[index]:
            if owner != index:
                raise ValueError("Occupied atlas seed changed")
            continue
        if owner == _UNSET or not occupancy[owner]:
            raise ValueError("Atlas padding owner is not an occupied texel")
        target, source = index * 4, owner * 4
        values[target] = values[source]
        values[target + 1] = values[source + 1]
        values[target + 2] = values[source + 2]
        values[target + 3] = values[source + 3]
        padded += 1
    return padded
