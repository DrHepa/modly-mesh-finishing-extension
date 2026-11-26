# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact-zero-area, indexed-TRIANGLES GLB repair with a deliberately narrow gate.

Only index values and the corresponding accessor count/min/max change in the
parsed glTF document; the JSON chunk is reserialized, so its raw bytes are not
identical. All non-index BIN ranges (including image and vertex-attribute
payloads) remain byte-for-byte intact. This module does not round-trip through
Blender's glTF exporter; Blender separately diagnoses and fresh-imports it.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import struct

from glb_inspector import GLBError, GLBReader, inspect_glb, _inspect_webp_references


_COMPONENT_FORMAT = {5121: "B", 5123: "H", 5125: "I"}
_ALLOWED_ATTRIBUTES = {"POSITION", "NORMAL", "TANGENT", "TEXCOORD_0", "TEXCOORD_1", "COLOR_0"}
_MATERIAL_KEYS = {"name", "pbrMetallicRoughness", "normalTexture", "occlusionTexture",
                  "emissiveTexture", "emissiveFactor", "alphaMode", "alphaCutoff", "doubleSided"}
_PBR_KEYS = {"baseColorFactor", "baseColorTexture", "metallicFactor", "roughnessFactor",
             "metallicRoughnessTexture"}
_TEXTURE_INFO_KEYS = {"index", "texCoord", "scale", "strength"}


class RepairRefused(GLBError):
    """A source cannot be changed without weakening the preservation contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RepairRefused(message)


def _array(doc: dict, name: str) -> list:
    result = doc.get(name, [])
    _require(isinstance(result, list), f"glTF {name} must be an array")
    return result


def _check_materials(doc: dict, *, allow_emissive_strength: bool = False,
                     webp_images: frozenset[int] = frozenset()) -> None:
    for material in _array(doc, "materials"):
        allowed_keys = _MATERIAL_KEYS | ({"extensions"} if allow_emissive_strength else set())
        _require(isinstance(material, dict) and not (set(material) - allowed_keys),
                 "Unsupported material property or extension; repair was refused")
        if allow_emissive_strength and "extensions" in material:
            extensions = material["extensions"]
            _require(isinstance(extensions, dict) and
                     set(extensions) == {"KHR_materials_emissive_strength"} and
                     "KHR_materials_emissive_strength" in doc.get("extensionsUsed", []),
                     "Unsupported or undeclared material extension; repair was refused")
            payload = extensions["KHR_materials_emissive_strength"]
            _require(isinstance(payload, dict) and set(payload) == {"emissiveStrength"} and
                     type(payload["emissiveStrength"]) in (int, float) and
                     math.isfinite(payload["emissiveStrength"]) and
                     1 <= payload["emissiveStrength"] <= 1000,
                     "Invalid emissive-strength extension; repair was refused")
        pbr = material.get("pbrMetallicRoughness", {})
        _require(isinstance(pbr, dict) and not (set(pbr) - _PBR_KEYS),
                 "Unsupported PBR material property; repair was refused")
        for texture_info in [*(pbr.get(key) for key in ("baseColorTexture", "metallicRoughnessTexture")),
                             *(material.get(key) for key in ("normalTexture", "occlusionTexture", "emissiveTexture"))]:
            if texture_info is not None:
                _require(isinstance(texture_info, dict) and not (set(texture_info) - _TEXTURE_INFO_KEYS),
                         "Texture transform or unsupported texture property; repair was refused")
        _require(material.get("alphaMode", "OPAQUE") in ("OPAQUE", "MASK", "BLEND"),
                 "Unsupported alpha mode; repair was refused")
    for texture in _array(doc, "textures"):
        _require(isinstance(texture, dict) and not (set(texture) - {"name", "sampler", "source", "extensions"}),
                 "Texture extension or unsupported texture property; repair was refused")
        if "extensions" in texture:
            extensions = texture["extensions"]
            _require(isinstance(extensions, dict) and set(extensions) == {"EXT_texture_webp"}
                     and isinstance(extensions["EXT_texture_webp"], dict)
                     and set(extensions["EXT_texture_webp"]) == {"source"},
                     "Unsupported texture extension payload; repair was refused")
            if "source" in texture:
                source = texture["source"]
                _require(type(source) is int and 0 <= source < len(_array(doc, "images")),
                         "Present WebP fallback source must be a valid image index")
    for image_id, image in enumerate(_array(doc, "images")):
        _require(isinstance(image, dict) and not (set(image) - {"name", "uri", "bufferView", "mimeType"}),
                 "Unsupported image property; repair was refused")
        allowed_mimes = (None, "image/png", "image/jpeg") + (("image/webp",) if image_id in webp_images else ())
        _require(image.get("mimeType") in allowed_mimes,
                 "Only core PNG/JPEG or validated EXT_texture_webp images are supported for repair")
        if "uri" in image:
            prefixes = ("data:image/png;base64,", "data:image/jpeg;base64,")
            if image_id in webp_images:
                prefixes += ("data:image/webp;base64,",)
            _require(isinstance(image["uri"], str) and
                     image["uri"].startswith(prefixes),
                     "Only supported embedded image URIs are supported for repair")
    for sampler in _array(doc, "samplers"):
        _require(isinstance(sampler, dict) and
                 not (set(sampler) - {"name", "magFilter", "minFilter", "wrapS", "wrapT"}),
                 "Unsupported texture sampler semantics; repair was refused")


def _preflight(reader: GLBReader, *, allow_emissive_strength: bool = False) -> list[tuple[int, int, dict, dict, int]]:
    doc = reader.doc
    _require(not (set(doc) - {"asset", "scene", "scenes", "nodes", "meshes", "materials", "textures",
                              "images", "samplers", "accessors", "bufferViews", "buffers", "skins",
                              "animations", "extensionsUsed", "extensionsRequired", "extras"}),
             "Unsupported top-level glTF semantics; repair was refused")
    used, required = doc.get("extensionsUsed", []), doc.get("extensionsRequired", [])
    allowed = {"EXT_texture_webp"} | ({"KHR_materials_emissive_strength"} if allow_emissive_strength else set())
    _require(isinstance(used, list) and isinstance(required, list) and
             all(isinstance(value, str) for value in used + required) and
             len(set(used)) == len(used) and len(set(required)) == len(required) and
             set(used) <= allowed and set(required) <= set(used),
             "glTF extensions are outside the conservative repair contract")
    _require(not _array(doc, "skins") and not _array(doc, "animations"),
             "Skins or animations require a separate preservation gate; repair was refused")
    _require(isinstance(doc.get("scene"), int) and not isinstance(doc.get("scene"), bool)
             and len(_array(doc, "scenes")) == 1 and doc["scene"] == 0,
             "Repair requires one explicit default scene")
    try:
        webp = _inspect_webp_references(reader)
    except GLBError as error:
        raise RepairRefused(str(error)) from error
    _check_materials(doc, allow_emissive_strength=allow_emissive_strength,
                     webp_images=frozenset(item["image"] for item in webp["textures"]))
    meshes = _array(doc, "meshes")
    nodes = _array(doc, "nodes")
    _require(bool(meshes), "Repair requires at least one mesh")
    mesh_uses = Counter()
    for node in nodes:
        _require(isinstance(node, dict) and "skin" not in node and "extensions" not in node,
                 "Node skin or extension is unsupported for repair")
        if "mesh" in node:
            mesh_id = node["mesh"]
            _require(isinstance(mesh_id, int) and not isinstance(mesh_id, bool) and
                     0 <= mesh_id < len(meshes), "Node mesh reference is invalid")
            mesh_uses[mesh_id] += 1
    _require(all(mesh_uses[index] == 1 for index in range(len(meshes))),
             "Every mesh must have exactly one node instance")
    _require(all(isinstance(node, dict) and not (set(node) - {"name", "mesh", "children", "matrix",
                                                                   "translation", "rotation", "scale", "extras"})
                 for node in nodes), "Unsupported node semantics; repair was refused")
    primitives: list[tuple[int, int, dict, dict, int]] = []
    accessors = _array(doc, "accessors")
    for view in _array(doc, "bufferViews"):
        _require(isinstance(view, dict) and not (set(view) - {"buffer", "byteOffset", "byteLength",
                                                               "byteStride", "target", "name"}),
                 "Unsupported bufferView semantics; repair was refused")
    for accessor in accessors:
        _require(isinstance(accessor, dict) and not (set(accessor) - {"bufferView", "byteOffset",
                                                                    "componentType", "normalized", "count",
                                                                    "type", "min", "max", "name"}),
                 "Sparse or unsupported accessor semantics; repair was refused")
    for mesh_id, mesh in enumerate(meshes):
        _require(isinstance(mesh, dict) and not (set(mesh) - {"name", "primitives", "extras"})
                 and isinstance(mesh.get("primitives"), list) and mesh["primitives"],
                 "Unsupported mesh semantics; repair was refused")
        _require("weights" not in mesh, "Morph weights are unsupported for repair")
        for primitive_id, primitive in enumerate(mesh["primitives"]):
            _require(isinstance(primitive, dict) and not (set(primitive) - {"attributes", "indices", "material", "mode"})
                     and "targets" not in primitive, "Morph target or unsupported primitive semantics; repair was refused")
            _require(primitive.get("mode", 4) == 4, "Only indexed TRIANGLES primitives can be repaired")
            attributes = primitive.get("attributes")
            _require(isinstance(attributes, dict) and "POSITION" in attributes
                     and not (set(attributes) - _ALLOWED_ATTRIBUTES),
                     "Unsupported vertex attribute; repair was refused")
            _require("indices" in primitive, "Non-indexed primitive repair is not supported")
            index_id = primitive["indices"]
            _require(isinstance(index_id, int) and not isinstance(index_id, bool) and 0 <= index_id < len(accessors),
                     "Primitive index accessor is invalid")
            index_accessor = accessors[index_id]
            _require(isinstance(index_accessor, dict) and index_accessor.get("componentType") in _COMPONENT_FORMAT
                     and index_accessor.get("type") == "SCALAR" and "sparse" not in index_accessor,
                     "Only dense unsigned index accessors are supported")
            position_id = attributes["POSITION"]
            _require(isinstance(position_id, int) and not isinstance(position_id, bool)
                     and 0 <= position_id < len(accessors), "Primitive POSITION accessor is invalid")
            position_accessor = accessors[position_id]
            _require(isinstance(position_accessor, dict) and position_accessor.get("type") == "VEC3"
                     and position_accessor.get("componentType") == 5126 and "sparse" not in position_accessor,
                     "Only dense float32 positions are supported")
            for attr_name, accessor_id in attributes.items():
                _require(isinstance(accessor_id, int) and not isinstance(accessor_id, bool)
                         and 0 <= accessor_id < len(accessors), f"{attr_name} accessor is invalid")
                _require("sparse" not in accessors[accessor_id],
                         "Sparse vertex attributes are unsupported for repair")
                _, attr_count, _, _, _, _ = reader.accessor_layout(accessor_id)
                _require(attr_count == position_accessor.get("count"), "Vertex attribute counts differ")
            _, index_count, _, _, _, _ = reader.accessor_layout(index_id)
            _require(index_count >= 3 and index_count % 3 == 0, "Index count must contain complete triangles")
            primitives.append((mesh_id, primitive_id, primitive, position_accessor, index_id))
    return primitives


def _positions(reader: GLBReader, accessor_id: int) -> list[tuple[float, float, float]]:
    offset, count, stride, _, _, sparse = reader.accessor_layout(accessor_id)
    _require(offset is not None and not sparse, "Sparse or missing POSITION buffer is unsupported")
    unpack = struct.Struct("<fff").unpack_from
    result = [unpack(reader.data, offset + index * stride) for index in range(count)]
    _require(all(math.isfinite(value) for point in result for value in point),
             "Non-finite positions are unsupported for repair")
    return result


def _zero_area(first: tuple[float, ...], second: tuple[float, ...], third: tuple[float, ...]) -> bool:
    # A float64 cross product is a fast *retention* test, not proof of zero:
    # subtracting a min-subnormal float32 from a max-normal float32 can erase
    # the small coordinate even though the exact triangle has nonzero area.
    if first == second or first == third or second == third:
        return True
    ax, ay, az = (second[i] - first[i] for i in range(3))
    bx, by, bz = (third[i] - first[i] for i in range(3))
    if any(component != 0.0 for component in
           (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)):
        return False
    # Decoded float32 values are represented exactly by Python float. Use exact
    # dyadic rationals only for deletion candidates; no rounded zero can be
    # promoted into a destructive edit.
    a = tuple(Fraction.from_float(second[i]) - Fraction.from_float(first[i]) for i in range(3))
    b = tuple(Fraction.from_float(third[i]) - Fraction.from_float(first[i]) for i in range(3))
    return (a[1] * b[2] == a[2] * b[1] and
            a[2] * b[0] == a[0] * b[2] and
            a[0] * b[1] == a[1] * b[0])


def _split_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    _require(len(raw) >= 28 and struct.unpack_from("<4sII", raw, 0) == (b"glTF", 2, len(raw)),
             "Invalid GLB header")
    json_size, json_kind = struct.unpack_from("<II", raw, 12)
    _require(json_kind == 0x4E4F534A and 20 + json_size + 8 <= len(raw), "Invalid GLB JSON chunk")
    bin_size, bin_kind = struct.unpack_from("<II", raw, 20 + json_size)
    _require(bin_kind == 0x004E4942 and 28 + json_size + bin_size == len(raw),
             "Repair requires exactly one JSON and one BIN chunk")
    return json.loads(raw[20:20 + json_size]), raw[28 + json_size:]


def _write_glb(path: Path, doc: dict, binary: bytes) -> None:
    encoded = json.dumps(doc, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    encoded += b" " * (-len(encoded) % 4)
    blob = (struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary))
            + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
            + struct.pack("<II", len(binary), 0x004E4942) + binary)
    with path.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())


def plan_repair(source: Path, *, allow_emissive_strength: bool = False) -> dict:
    """Plan exact edits without mutating source; reject ambiguity before staging."""
    with GLBReader(source) as reader:
        primitives = _preflight(reader, allow_emissive_strength=allow_emissive_strength)
        doc = reader.doc
        all_accessors = _array(doc, "accessors")
        views = _array(doc, "bufferViews")
        edits = []
        total_before = total_removed = 0
        for mesh_id, primitive_id, primitive, _, index_id in primitives:
            positions = _positions(reader, primitive["attributes"]["POSITION"])
            indices = list(reader.scalar_values(index_id))
            _require(all(0 <= index < len(positions) for index in indices), "Index out of POSITION range")
            survivor_indices = []
            removed = 0
            for start in range(0, len(indices), 3):
                triangle = indices[start:start + 3]
                if _zero_area(*(positions[index] for index in triangle)):
                    removed += 1
                else:
                    survivor_indices.extend(triangle)
            total_before += len(indices) // 3
            total_removed += removed
            if removed:
                _require(survivor_indices, "Repair would remove every face in a primitive; refused to preserve identity")
                accessor = all_accessors[index_id]
                _require(sum(1 for _, _, candidate, _, _ in primitives if candidate["indices"] == index_id) == 1,
                         "Shared index accessor cannot be edited safely")
                view_id = accessor.get("bufferView")
                _require(isinstance(view_id, int) and not isinstance(view_id, bool) and 0 <= view_id < len(views),
                         "Index bufferView is invalid")
                _require(sum(1 for candidate in all_accessors if candidate.get("bufferView") == view_id) == 1,
                         "Shared index bufferView cannot be edited safely")
                _require(not any(image.get("bufferView") == view_id for image in _array(doc, "images")),
                         "Index bufferView is also used by an image; repair was refused")
                view = views[view_id]
                _require("byteStride" not in view,
                         "Strided index bufferView is outside the repair contract")
                start = view.get("byteOffset", 0)
                end = start + view["byteLength"]
                _require(all(other_id == view_id or
                             end <= other.get("byteOffset", 0) or
                             start >= other.get("byteOffset", 0) + other["byteLength"]
                             for other_id, other in enumerate(views)),
                         "Index bufferView overlaps another bufferView")
                _require(set(accessor).issubset({"bufferView", "byteOffset", "componentType", "count", "type", "min", "max", "name"}),
                         "Unsupported index accessor metadata")
                _require(("min" in accessor) == ("max" in accessor),
                         "Incomplete index accessor bounds")
                edits.append({"mesh": mesh_id, "primitive": primitive_id, "accessor": index_id,
                              "survivor_indices": survivor_indices, "before_triangles": len(indices) // 3,
                              "removed_zero_area_triangles": removed})
        inspection = inspect_glb(source)
        _require(inspection["counts"]["active_scene_render_triangles"] == total_before,
                 "Repair plan includes mesh geometry outside the active scene")
        return {"before_triangles": total_before, "removed_zero_area_triangles": total_removed,
                "after_triangles": total_before - total_removed, "edits": edits,
                "operation": "remove_exact_zero_local_area_indexed_triangles"}


def stage_candidate(source: Path, stage: Path, plan: dict) -> None:
    """Write a candidate, then prove only approved index/accessor bytes changed."""
    _require(plan["removed_zero_area_triangles"] > 0, "No repair candidate is needed for a no-op")
    doc, original_binary = _split_glb(source)
    binary = bytearray(original_binary)
    for edit in plan["edits"]:
        accessor = doc["accessors"][edit["accessor"]]
        view = doc["bufferViews"][accessor["bufferView"]]
        fmt = _COMPONENT_FORMAT[accessor["componentType"]]
        stride = struct.calcsize("<" + fmt)
        offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        for index, value in enumerate(edit["survivor_indices"]):
            struct.pack_into("<" + fmt, binary, offset + index * stride, value)
        accessor["count"] = len(edit["survivor_indices"])
        if "min" in accessor:
            accessor["min"] = [min(edit["survivor_indices"])]
            accessor["max"] = [max(edit["survivor_indices"])]
    _write_glb(stage, doc, bytes(binary))
    verify_candidate(source, stage, plan, original_binary)


def verify_candidate(source: Path, candidate: Path, plan: dict, source_binary: bytes | None = None) -> dict:
    source_doc, source_bin = _split_glb(source)
    target_doc, target_bin = _split_glb(candidate)
    if source_binary is not None:
        _require(source_binary == source_bin, "Source changed during candidate staging")
    _require(len(source_bin) == len(target_bin), "Candidate BIN size changed")
    expected_doc = json.loads(json.dumps(source_doc))
    expected_bin = bytearray(source_bin)
    for edit in plan["edits"]:
        accessor = expected_doc["accessors"][edit["accessor"]]
        view = expected_doc["bufferViews"][accessor["bufferView"]]
        fmt = _COMPONENT_FORMAT[accessor["componentType"]]
        stride = struct.calcsize("<" + fmt)
        offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        for index, value in enumerate(edit["survivor_indices"]):
            struct.pack_into("<" + fmt, expected_bin, offset + index * stride, value)
        accessor["count"] = len(edit["survivor_indices"])
        if "min" in accessor:
            accessor["min"] = [min(edit["survivor_indices"])]
            accessor["max"] = [max(edit["survivor_indices"])]
    _require(expected_doc == target_doc and expected_bin == target_bin,
             "Candidate changed geometry, attributes, transforms, material, image, or unrelated GLB bytes")
    before = inspect_glb(source)
    after = inspect_glb(candidate)
    _require(before["counts"]["active_scene_render_triangles"] - after["counts"]["active_scene_render_triangles"]
             == plan["removed_zero_area_triangles"], "Candidate triangle count mismatch")
    _require(not any(item["code"] == "index_out_of_range" for item in after["findings"]),
             "Candidate has out-of-range indices")
    return {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "allowed_document_semantics_and_exact_bin_diff": True,
            "unchanged": ["all_vertex_attribute_accessors_and_bytes", "all_surviving_face_indices_in_original_order",
                          "all_mesh_node_transforms_and_instances", "all_material_and_texture_definitions_semantically",
                          "all_non_index_binary_ranges", "all_primitive_material_assignments"],
            "geometry_invariants": "All surviving triangle indices and referenced vertex attributes remain in original order and unchanged. Removing a zero-area face may change aggregate edge incidence, boundary, or component counts; those are separately observed by Blender, not assumed invariant.",
            "source_render_triangles": before["counts"]["active_scene_render_triangles"],
            "candidate_render_triangles": after["counts"]["active_scene_render_triangles"]}
