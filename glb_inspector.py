# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only glTF 2.0 GLB structural inspection; no geometry is modified."""

from __future__ import annotations

import json
import math
import mmap
from pathlib import Path
import re
import struct
from typing import Any, Iterator


class GLBError(ValueError):
    """Invalid or unsupported self-contained GLB container."""


_COMPONENT_FORMAT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
_COMPONENT_SIZE = {kind: struct.calcsize("<" + fmt) for kind, fmt in _COMPONENT_FORMAT.items()}
_TYPE_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_TRIANGLE_MODES = {4: "TRIANGLES", 5: "TRIANGLE_STRIP", 6: "TRIANGLE_FAN"}
_GEOMETRY_EXTENSIONS = {"KHR_draco_mesh_compression", "EXT_meshopt_compression", "EXT_mesh_gpu_instancing"}
_SUPPORTED_REQUIRED_EXTENSIONS = {"EXT_texture_webp"}


def _array(doc: dict, key: str) -> list:
    value = doc.get(key, [])
    if not isinstance(value, list):
        raise GLBError(f"glTF {key} must be an array")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GLBError(f"{label} must be a nonnegative integer")
    return value


class GLBReader:
    def __init__(self, path: Path):
        if path.stat().st_size == 0:
            raise GLBError("GLB file is empty")
        self.file = path.open("rb")
        try:
            self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            self.doc, self.bin_offset, self.bin_length = self._parse()
        except Exception:
            if hasattr(self, "data"):
                self.data.close()
            self.file.close()
            raise

    def close(self) -> None:
        self.data.close()
        self.file.close()

    def __enter__(self) -> "GLBReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _parse(self) -> tuple[dict, int | None, int]:
        data = self.data
        if len(data) < 20:
            raise GLBError("GLB header or JSON chunk is missing")
        magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
        if magic != b"glTF" or version != 2 or declared_length != len(data):
            raise GLBError("Invalid GLB magic, version, or declared length")
        chunks: list[tuple[int, int, int]] = []
        cursor = 12
        while cursor < len(data):
            if len(data) - cursor < 8:
                raise GLBError("Truncated GLB chunk header")
            length, kind = struct.unpack_from("<II", data, cursor)
            cursor += 8
            if length % 4 or length > len(data) - cursor:
                raise GLBError("Invalid or truncated GLB chunk")
            chunks.append((kind, cursor, length))
            cursor += length
        if not chunks or chunks[0][0] != 0x4E4F534A:
            raise GLBError("First GLB chunk must be JSON")
        if sum(1 for kind, _, _ in chunks if kind == 0x4E4F534A) != 1:
            raise GLBError("GLB must contain exactly one JSON chunk")
        try:
            raw = bytes(data[chunks[0][1]:chunks[0][1] + chunks[0][2]]).rstrip(b" \x00")
            doc = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise GLBError("GLB JSON chunk is invalid") from exc
        if not isinstance(doc, dict) or not isinstance(doc.get("asset"), dict):
            raise GLBError("glTF asset object is missing")
        if doc["asset"].get("version") != "2.0":
            raise GLBError("Only glTF 2.0 is supported")
        required = doc.get("extensionsRequired", [])
        if not isinstance(required, list) or any(not isinstance(name, str) for name in required):
            raise GLBError("glTF extensionsRequired must be an array")
        if set(required) - _SUPPORTED_REQUIRED_EXTENSIONS:
            raise GLBError("Required glTF extensions are not supported by this inspector")
        used = doc.get("extensionsUsed", [])
        if not isinstance(used, list) or any(not isinstance(name, str) for name in used):
            raise GLBError("glTF extensionsUsed must be an array")
        if not set(required).issubset(used):
            raise GLBError("Required glTF extension is missing from extensionsUsed")
        if _GEOMETRY_EXTENSIONS.intersection(used):
            raise GLBError("Geometry-affecting glTF extensions are not supported by this inspector")
        bins = [(offset, length) for kind, offset, length in chunks if kind == 0x004E4942]
        if len(bins) > 1:
            raise GLBError("GLB contains multiple BIN chunks")
        bin_offset, bin_length = bins[0] if bins else (None, 0)
        buffers = _array(doc, "buffers")
        if any(not isinstance(buffer, dict) or "uri" in buffer for buffer in buffers):
            raise GLBError("Only an implicit BIN buffer is supported; buffer URIs, including data URIs, are not decoded")
        if len(buffers) > 1:
            raise GLBError("Self-contained GLB may contain only one implicit BIN buffer")
        if buffers:
            expected = _nonnegative_int(buffers[0].get("byteLength"), "buffer byteLength")
            if bin_offset is None or expected > bin_length or bin_length - expected > 3:
                raise GLBError("BIN chunk does not match declared buffer length")
        elif bin_offset is not None:
            raise GLBError("BIN chunk has no declared buffer")
        for image in _array(doc, "images"):
            if not isinstance(image, dict):
                raise GLBError("Image entry is invalid")
            uri = image.get("uri")
            if uri is not None and (not isinstance(uri, str) or not uri.startswith("data:")):
                raise GLBError("External GLB images are not supported; source must be self-contained")
        return doc, bin_offset, bin_length

    def accessor_layout(self, index: int) -> tuple[int | None, int, int, str, int, bool]:
        accessors = _array(self.doc, "accessors")
        views = _array(self.doc, "bufferViews")
        if index < 0 or index >= len(accessors) or not isinstance(accessors[index], dict):
            raise GLBError("Primitive references an invalid accessor")
        accessor = accessors[index]
        count = _nonnegative_int(accessor.get("count"), "accessor count")
        component_type = accessor.get("componentType")
        value_type = accessor.get("type")
        if component_type not in _COMPONENT_SIZE or value_type not in _TYPE_WIDTH:
            raise GLBError("Accessor has unsupported component or value type")
        element_size = _COMPONENT_SIZE[component_type] * _TYPE_WIDTH[value_type]
        sparse = "sparse" in accessor
        view_index = accessor.get("bufferView")
        if view_index is None:
            if not sparse:
                raise GLBError("Accessor has no bufferView or sparse data")
            return None, count, element_size, _COMPONENT_FORMAT[component_type], component_type, sparse
        if isinstance(view_index, bool) or not isinstance(view_index, int) or not 0 <= view_index < len(views):
            raise GLBError("Accessor references an invalid bufferView")
        view = views[view_index]
        if not isinstance(view, dict) or view.get("buffer") != 0 or self.bin_offset is None:
            raise GLBError("Accessor bufferView is invalid")
        view_offset = _nonnegative_int(view.get("byteOffset", 0), "bufferView byteOffset")
        view_length = _nonnegative_int(view.get("byteLength"), "bufferView byteLength")
        accessor_offset = _nonnegative_int(accessor.get("byteOffset", 0), "accessor byteOffset")
        stride = _nonnegative_int(view.get("byteStride", element_size), "bufferView byteStride")
        if stride < element_size or stride % _COMPONENT_SIZE[component_type]:
            raise GLBError("Accessor stride is invalid")
        last_byte = accessor_offset + ((count - 1) * stride + element_size if count else 0)
        declared_buffer_length = self.doc["buffers"][0]["byteLength"]
        if last_byte > view_length or view_offset + view_length > declared_buffer_length:
            raise GLBError("Accessor exceeds its bufferView or declared buffer")
        if accessor_offset % _COMPONENT_SIZE[component_type]:
            raise GLBError("Accessor offset is misaligned")
        return self.bin_offset + view_offset + accessor_offset, count, stride, _COMPONENT_FORMAT[component_type], component_type, sparse

    def scalar_values(self, index: int) -> Iterator[int]:
        accessor = self.doc["accessors"][index]
        if accessor.get("type") != "SCALAR" or accessor.get("componentType") not in (5121, 5123, 5125):
            raise GLBError("Index accessor must be unsigned SCALAR")
        offset, count, stride, fmt, _, sparse = self.accessor_layout(index)
        if sparse or offset is None:
            raise GLBError("Sparse index accessor is not decoded")
        unpack = struct.Struct("<" + fmt).unpack_from
        for position in range(count):
            yield unpack(self.data, offset + position * stride)[0]


def _triangle_count(mode: int, vertices: int) -> int:
    if mode == 4:
        return vertices // 3
    if mode in (5, 6):
        return max(0, vertices - 2)
    return 0


def _inspect_webp_references(reader: GLBReader) -> dict:
    """Validate WebP reference structure and embedded storage, never image pixels."""
    doc = reader.doc
    used = "EXT_texture_webp" in doc.get("extensionsUsed", [])
    required = "EXT_texture_webp" in doc.get("extensionsRequired", [])
    textures = _array(doc, "textures")
    images = _array(doc, "images")
    views = _array(doc, "bufferViews")
    references: list[dict] = []
    for texture_id, texture in enumerate(textures):
        if not isinstance(texture, dict):
            raise GLBError("Texture entry is invalid")
        extensions = texture.get("extensions", {})
        if not isinstance(extensions, dict):
            raise GLBError("Texture extensions must be an object")
        if "EXT_texture_webp" not in extensions:
            continue
        if not used:
            raise GLBError("WebP texture is not declared in extensionsUsed")
        extension = extensions["EXT_texture_webp"]
        if not isinstance(extension, dict):
            raise GLBError("WebP texture extension must be an object")
        image_id = extension.get("source")
        if isinstance(image_id, bool) or not isinstance(image_id, int) or not 0 <= image_id < len(images):
            raise GLBError("WebP texture source image index is invalid")
        image = images[image_id]
        if not isinstance(image, dict):
            raise GLBError("WebP source image entry is invalid")
        has_view = "bufferView" in image
        has_uri = "uri" in image
        if has_view == has_uri:
            raise GLBError("WebP source image must use exactly one embedded storage form")
        if has_view:
            view_id = image["bufferView"]
            if isinstance(view_id, bool) or not isinstance(view_id, int) or not 0 <= view_id < len(views):
                raise GLBError("WebP image bufferView reference is invalid")
            view = views[view_id]
            if not isinstance(view, dict) or view.get("buffer") != 0 or reader.bin_offset is None:
                raise GLBError("WebP image bufferView must reference the implicit BIN buffer")
            offset = _nonnegative_int(view.get("byteOffset", 0), "WebP image byteOffset")
            length = _nonnegative_int(view.get("byteLength"), "WebP image byteLength")
            if length == 0 or offset + length > doc["buffers"][0]["byteLength"]:
                raise GLBError("WebP image bufferView exceeds the implicit BIN buffer")
            if image.get("mimeType") != "image/webp":
                raise GLBError("Embedded WebP image must declare mimeType image/webp")
            storage = "bufferView"
        else:
            uri = image["uri"]
            prefix = "data:image/webp;base64,"
            encoded = uri[len(prefix):] if isinstance(uri, str) and uri.startswith(prefix) else ""
            if not encoded or len(encoded) % 4 or re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", encoded) is None:
                raise GLBError("WebP image URI must be a nonempty embedded image/webp base64 data URI")
            if image.get("mimeType", "image/webp") != "image/webp":
                raise GLBError("WebP image MIME type is invalid")
            storage = "data_uri"
        fallback = texture.get("source")
        if fallback is None and not required:
            raise GLBError("Optional WebP texture requires a PNG or JPEG fallback image")
        fallback_format = None
        if fallback is not None:
            if isinstance(fallback, bool) or not isinstance(fallback, int) or not 0 <= fallback < len(images):
                raise GLBError("WebP texture fallback image index is invalid")
            fallback_image = images[fallback]
            if not isinstance(fallback_image, dict):
                raise GLBError("WebP texture fallback image entry is invalid")
            fallback_has_view = "bufferView" in fallback_image
            fallback_has_uri = "uri" in fallback_image
            if fallback_has_view == fallback_has_uri:
                raise GLBError("WebP fallback image must use exactly one embedded storage form")
            if fallback_has_view:
                fallback_format = fallback_image.get("mimeType")
                view_id = fallback_image["bufferView"]
                if isinstance(view_id, bool) or not isinstance(view_id, int) or not 0 <= view_id < len(views):
                    raise GLBError("WebP fallback image bufferView reference is invalid")
                view = views[view_id]
                if not isinstance(view, dict) or view.get("buffer") != 0 or reader.bin_offset is None:
                    raise GLBError("WebP fallback image must reference the implicit BIN buffer")
                offset = _nonnegative_int(view.get("byteOffset", 0), "WebP fallback byteOffset")
                length = _nonnegative_int(view.get("byteLength"), "WebP fallback byteLength")
                if length == 0 or offset + length > doc["buffers"][0]["byteLength"]:
                    raise GLBError("WebP fallback image bufferView exceeds the implicit BIN buffer")
            else:
                uri = fallback_image["uri"]
                if not isinstance(uri, str):
                    raise GLBError("WebP fallback image URI is invalid")
                for mime in ("image/png", "image/jpeg"):
                    prefix = f"data:{mime};base64,"
                    if uri.startswith(prefix):
                        encoded = uri[len(prefix):]
                        if encoded and len(encoded) % 4 == 0 and re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", encoded):
                            fallback_format = mime
                        break
                if fallback_image.get("mimeType", fallback_format) != fallback_format:
                    raise GLBError("WebP fallback image MIME type is inconsistent")
            if fallback_format not in ("image/png", "image/jpeg"):
                raise GLBError("WebP fallback image must be PNG or JPEG, not WebP")
        references.append({"texture": texture_id, "image": image_id, "storage": storage,
                           "fallback_image": fallback, "fallback_format": fallback_format})
    if required and not references:
        raise GLBError("Required WebP extension has no texture reference")
    return {"used": used, "required": required, "present": bool(references), "textures": references}


def inspect_glb(path: Path) -> dict:
    with GLBReader(path) as reader:
        doc = reader.doc
        webp = _inspect_webp_references(reader)
        meshes = _array(doc, "meshes")
        nodes = _array(doc, "nodes")
        scenes = _array(doc, "scenes")
        materials = _array(doc, "materials")
        skins = _array(doc, "skins")
        animations = _array(doc, "animations")
        findings: list[dict] = []
        primitives: list[dict] = []
        mesh_triangles: list[int] = []
        mesh_vertices: list[int] = []
        unsupported_modes = 0
        invalid_index_primitives = 0
        sparse_index_primitives = 0
        sparse_attribute_primitives = 0
        nonfinite_positions = 0
        for mesh_id, mesh in enumerate(meshes):
            if not isinstance(mesh, dict) or not isinstance(mesh.get("primitives"), list):
                raise GLBError("Mesh primitives are missing")
            mesh_tri = mesh_vert = 0
            for primitive_id, primitive in enumerate(mesh["primitives"]):
                sparse = False
                if not isinstance(primitive, dict) or not isinstance(primitive.get("attributes"), dict):
                    raise GLBError("Primitive attributes are missing")
                attributes = primitive["attributes"]
                primitive_has_sparse_attribute = False
                attribute_counts: dict[str, int] = {}
                for attribute_name, attribute_index in attributes.items():
                    if not isinstance(attribute_name, str) or isinstance(attribute_index, bool) or not isinstance(attribute_index, int):
                        raise GLBError("Primitive attribute reference is invalid")
                    _, attribute_count, _, _, _, attribute_sparse = reader.accessor_layout(attribute_index)
                    attribute_counts[attribute_name] = attribute_count
                    primitive_has_sparse_attribute |= attribute_sparse
                targets = primitive.get("targets", [])
                if not isinstance(targets, list):
                    raise GLBError("Primitive morph targets must be an array")
                target_counts: list[int] = []
                for target in targets:
                    if not isinstance(target, dict):
                        raise GLBError("Primitive morph target is invalid")
                    for target_index in target.values():
                        if isinstance(target_index, bool) or not isinstance(target_index, int):
                            raise GLBError("Morph target accessor reference is invalid")
                        _, target_count, _, _, _, target_sparse = reader.accessor_layout(target_index)
                        target_counts.append(target_count)
                        primitive_has_sparse_attribute |= target_sparse
                if primitive_has_sparse_attribute:
                    sparse_attribute_primitives += 1
                position_index = attributes.get("POSITION")
                if isinstance(position_index, bool) or not isinstance(position_index, int):
                    raise GLBError("Primitive POSITION accessor is missing")
                pos_accessor = _array(doc, "accessors")[position_index] if 0 <= position_index < len(_array(doc, "accessors")) else None
                if not isinstance(pos_accessor, dict) or pos_accessor.get("type") != "VEC3" or pos_accessor.get("componentType") != 5126:
                    raise GLBError("Primitive POSITION accessor must be float VEC3")
                pos_offset, vertex_count, pos_stride, _, _, pos_sparse = reader.accessor_layout(position_index)
                if any(count != vertex_count for count in attribute_counts.values()) or any(count != vertex_count for count in target_counts):
                    raise GLBError("Primitive attribute or morph target accessor count differs from POSITION count")
                if not pos_sparse and pos_offset is not None:
                    unpack_pos = struct.Struct("<fff").unpack_from
                    for vertex_id in range(vertex_count):
                        if not all(math.isfinite(x) for x in unpack_pos(reader.data, pos_offset + vertex_id * pos_stride)):
                            nonfinite_positions += 1
                else:
                    findings.append({"code": "sparse_position_not_scanned", "mesh": mesh_id, "primitive": primitive_id})
                index_id = primitive.get("indices")
                if index_id is not None:
                    if isinstance(index_id, bool) or not isinstance(index_id, int):
                        raise GLBError("Primitive index accessor is invalid")
                    index_accessor = _array(doc, "accessors")[index_id] if 0 <= index_id < len(_array(doc, "accessors")) else None
                    if not isinstance(index_accessor, dict) or index_accessor.get("type") != "SCALAR" or index_accessor.get("componentType") not in (5121, 5123, 5125):
                        raise GLBError("Primitive index accessor must be unsigned SCALAR")
                    _, draw_count, _, _, _, sparse = reader.accessor_layout(index_id)
                    if sparse:
                        sparse_index_primitives += 1
                    else:
                        invalid_indices = sum(1 for value in reader.scalar_values(index_id) if value >= vertex_count)
                        if invalid_indices:
                            invalid_index_primitives += 1
                            findings.append({"code": "index_out_of_range", "mesh": mesh_id, "primitive": primitive_id, "count": invalid_indices})
                else:
                    draw_count = vertex_count
                mode = primitive.get("mode", 4)
                if isinstance(mode, bool) or not isinstance(mode, int) or mode not in range(7):
                    raise GLBError("Primitive mode is invalid")
                if mode not in _TRIANGLE_MODES:
                    unsupported_modes += 1
                if mode == 4 and draw_count % 3:
                    findings.append({"code": "incomplete_triangle_group", "mesh": mesh_id, "primitive": primitive_id})
                triangle_count = _triangle_count(mode, draw_count)
                mesh_tri += triangle_count
                mesh_vert += vertex_count
                material_id = primitive.get("material")
                if material_id is not None and (isinstance(material_id, bool) or not isinstance(material_id, int) or not 0 <= material_id < len(materials)):
                    findings.append({"code": "invalid_material_reference", "mesh": mesh_id, "primitive": primitive_id})
                primitives.append({
                    "mesh": mesh_id, "primitive": primitive_id,
                    "mode": _TRIANGLE_MODES.get(mode, ["POINTS", "LINES", "LINE_LOOP", "LINE_STRIP"][mode] if mode < 4 else str(mode)),
                    "indexed": index_id is not None, "draw_elements": draw_count,
                    "render_vertices": vertex_count, "render_triangles": triangle_count,
                    "material": material_id, "attributes": sorted(attributes.keys()),
                    "morph_targets": len(targets),
                    "indices_validity": "not_assessed_sparse" if index_id is not None and sparse else ("assessed" if index_id is not None else "not_applicable"),
                    "position_finiteness": "not_assessed_sparse" if pos_sparse else "assessed",
                })
            mesh_triangles.append(mesh_tri)
            mesh_vertices.append(mesh_vert)

        selected_scene = doc.get("scene")
        if selected_scene is not None and (isinstance(selected_scene, bool) or not isinstance(selected_scene, int) or not 0 <= selected_scene < len(scenes)):
            raise GLBError("Default scene reference is invalid")
        active_instances = 0
        active_triangles = 0
        active_vertices = 0
        skinned_instances = 0
        active_scene_nodes: set[int] = set()
        transform_issues = 0
        active_transforms: dict[int, dict] = {}
        if selected_scene is not None:
            scene = scenes[selected_scene]
            if not isinstance(scene, dict) or not isinstance(scene.get("nodes", []), list):
                raise GLBError("Default scene nodes are invalid")
            stack: list[int] = list(scene.get("nodes", []))
            while stack:
                node_id = stack.pop()
                if isinstance(node_id, bool) or not isinstance(node_id, int) or not 0 <= node_id < len(nodes):
                    raise GLBError("Scene references an invalid node")
                if node_id in active_scene_nodes:
                    raise GLBError("Scene graph repeats a node (cycle, duplicate child, or shared parent)")
                node = nodes[node_id]
                if not isinstance(node, dict):
                    raise GLBError("Scene node is invalid")
                active_scene_nodes.add(node_id)
                matrix = node.get("matrix")
                translation = node.get("translation", [0, 0, 0])
                rotation = node.get("rotation", [0, 0, 0, 1])
                scale = node.get("scale", [1, 1, 1])
                valid_shape = (
                    (matrix is None or isinstance(matrix, list) and len(matrix) == 16)
                    and isinstance(translation, list) and len(translation) == 3
                    and isinstance(rotation, list) and len(rotation) == 4
                    and isinstance(scale, list) and len(scale) == 3
                )
                values = (matrix if matrix is not None else translation + rotation + scale) if valid_shape else []
                finite = valid_shape and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values)
                if not finite:
                    transform_issues += 1
                active_transforms[node_id] = {
                    "node": node_id, "matrix": matrix,
                    "translation": None if matrix is not None else translation,
                    "rotation": None if matrix is not None else rotation,
                    "scale": None if matrix is not None else scale,
                    "finite": finite,
                }
                if "mesh" in node:
                    mesh_id = node["mesh"]
                    if isinstance(mesh_id, bool) or not isinstance(mesh_id, int) or not 0 <= mesh_id < len(meshes):
                        raise GLBError("Node references an invalid mesh")
                    active_instances += 1
                    active_triangles += mesh_triangles[mesh_id]
                    active_vertices += mesh_vertices[mesh_id]
                    if "skin" in node:
                        skin_id = node["skin"]
                        if isinstance(skin_id, bool) or not isinstance(skin_id, int) or not 0 <= skin_id < len(skins):
                            findings.append({"code": "invalid_skin_reference", "node": node_id})
                        skinned_instances += 1
                children = node.get("children", [])
                if not isinstance(children, list):
                    raise GLBError("Node children must be an array")
                stack.extend(children)
        if transform_issues:
            findings.append({"code": "nonfinite_or_malformed_transform", "count": transform_issues})
        if nonfinite_positions:
            findings.append({"code": "nonfinite_positions", "count": nonfinite_positions})

        material_rows = []
        for material_id, material in enumerate(materials):
            if not isinstance(material, dict):
                raise GLBError("Material entry is invalid")
            pbr = material.get("pbrMetallicRoughness") or {}
            if not isinstance(pbr, dict):
                raise GLBError("Material PBR entry is invalid")
            material_rows.append({
                "material": material_id, "alpha_mode": material.get("alphaMode", "OPAQUE"),
                "alpha_cutoff": material.get("alphaCutoff", 0.5), "double_sided": material.get("doubleSided", False),
                "textures": {name: isinstance(owner.get(name), dict) for name, owner in (
                    ("baseColorTexture", pbr), ("metallicRoughnessTexture", pbr),
                    ("normalTexture", material), ("occlusionTexture", material), ("emissiveTexture", material)
                )},
                "extensions": sorted((material.get("extensions") or {}).keys()),
            })
        animation_rows = []
        for animation_id, animation in enumerate(animations):
            if not isinstance(animation, dict):
                raise GLBError("Animation entry is invalid")
            channels = animation.get("channels", [])
            if not isinstance(channels, list):
                raise GLBError("Animation channels are invalid")
            targeted_nodes = sorted({channel.get("target", {}).get("node") for channel in channels
                                     if isinstance(channel, dict) and isinstance(channel.get("target"), dict)
                                     and isinstance(channel["target"].get("node"), int)
                                     and not isinstance(channel["target"].get("node"), bool)})
            animation_rows.append({"animation": animation_id, "channels": len(channels),
                                   "samplers": len(animation.get("samplers", [])), "targeted_nodes": targeted_nodes,
                                   "targets_active_scene_nodes": bool(active_scene_nodes.intersection(targeted_nodes))})
        not_assessed = [
            {"check": "geometric_topology", "reason": "Render primitives do not prove welded geometric adjacency, boundaries, or components."},
            {"check": "connected_components", "reason": "No geometry-weld or component analysis is performed."},
            {"check": "degenerate_triangles_and_orientation", "reason": "Triangle area and winding are not evaluated in this slice."},
            {"check": "self_intersections_and_internal_surfaces", "reason": "Requires geometric analysis beyond GLB structure."},
            {"check": "UV_continuity_and_bake_quality", "reason": "Attribute presence is not a UV or appearance validation."},
            {"check": "animated_deformation_and_skin_validity", "reason": "Skin/animation metadata is reported, not evaluated or modified."},
            {"check": "world_space_bounds_and_scale", "reason": "Transforms are checked for finiteness, not composed into bounds."},
        ]
        if sparse_index_primitives:
            not_assessed.append({"check": "sparse_index_values", "reason": "Sparse index overrides are not decoded; render triangle counts still use accessor count."})
        if sparse_attribute_primitives:
            not_assessed.append({"check": "sparse_attribute_values", "reason": "Sparse accessor override buffers and values are not decoded."})
        if selected_scene is None:
            not_assessed.append({"check": "active_scene_instances", "reason": "No default scene is declared."})
        if webp["present"]:
            not_assessed.append({"check": "webp_image_decode_and_visual_fidelity", "reason": "WebP image bytes and rendering are not decoded or evaluated."})
        return {
            "assessment_status": "partial",
            "assessed": {
                "glb_container": True, "accessor_bounds_for_used_primitives": sparse_attribute_primitives == 0 and sparse_index_primitives == 0,
                "render_triangle_counts": True, "active_scene_instances": selected_scene is not None,
                "index_value_ranges": sparse_index_primitives == 0,
                "position_finiteness": not any(p["position_finiteness"] != "assessed" for p in primitives),
                "material_metadata": True, "skin_animation_metadata": True,
                "webp_texture_references": True,
            },
            "not_assessed": not_assessed,
            "counts": {
                "meshes": len(meshes), "nodes": len(nodes), "scenes": len(scenes),
                "primitives": len(primitives), "materials": len(materials),
                "unique_mesh_render_triangles": sum(mesh_triangles),
                "active_scene_mesh_instances": active_instances if selected_scene is not None else None,
                "active_scene_render_triangles": active_triangles if selected_scene is not None else None,
                "active_scene_render_vertices": active_vertices if selected_scene is not None else None,
                "non_triangle_primitives": unsupported_modes,
                "primitives_with_out_of_range_indices": invalid_index_primitives,
                "primitives_with_unassessed_sparse_indices": sparse_index_primitives,
                "nonfinite_position_vertices": nonfinite_positions,
                "skins": len(skins), "active_scene_skinned_mesh_instances": skinned_instances if selected_scene is not None else None,
                "animations": len(animations),
                "webp_textures": len(webp["textures"]),
            },
            "active_scene": selected_scene,
            "active_scene_node_transforms": list(active_transforms.values()) if selected_scene is not None else None,
            "primitives": primitives, "materials": material_rows, "animations": animation_rows,
            "webp": webp,
            "skins": [{"skin": i, "joints": len(skin.get("joints", [])) if isinstance(skin, dict) else None}
                      for i, skin in enumerate(skins)],
            "findings": findings,
        }
