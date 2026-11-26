# SPDX-License-Identifier: GPL-3.0-or-later
"""Small Blender PBR/image primitives extracted from the original rebaker."""
from array import array
from pathlib import Path
import bpy
from atlas_padding import pad_rgba_pixels

def _node(nodes, kind: str, name: str):
    node = nodes.new(kind)
    node.name = name
    return node


def _math(nodes, links, operation: str, first, second, name: str):
    node = _node(nodes, "ShaderNodeMath", name)
    node.operation = operation
    for index, value in enumerate((first, second)):
        if hasattr(value, "is_output"):
            links.new(value, node.inputs[index])
        else:
            node.inputs[index].default_value = value
    return node.outputs[0]


def _multiply_color(nodes, links, color, factor: list[float], name: str):
    if color is None:
        return (*factor, 1)
    node = _node(nodes, "ShaderNodeMixRGB", name)
    node.blend_type = "MULTIPLY"
    node.inputs[0].default_value = 1
    links.new(color, node.inputs[1])
    node.inputs[2].default_value = (*factor, 1)
    return node.outputs[0]


def source_proxies(source, bsdf, material_doc):
    def value(socket):
        if socket.is_linked:return socket.links[0].from_socket
        return socket.default_value[:] if hasattr(socket.default_value,'__len__') else float(socket.default_value)
    proxies={label:value(bsdf.inputs[name]) for label,name in
             [('base','Base Color'),('metal','Metallic'),('rough','Roughness'),('emission','Emission Color')]}
    strength=bsdf.inputs['Emission Strength']
    if strength.is_linked:raise ValueError('Linked imported emission strength is unsupported')
    proxies['emission_strength']=float(strength.default_value)
    if not 0<=proxies['emission_strength']<=1000:raise ValueError('Unsupported imported effective emission strength')
    # Blender 5.2 folds gray emissiveFactor into this strength. Keep the
    # imported effective value exactly once, not merely the JSON extension.
    pbr=material_doc.get('pbrMetallicRoughness',{})
    alpha=pbr.get('baseColorFactor',[1,1,1,1])[3]
    if pbr.get('baseColorTexture') is not None:
        textures=[n for n in source.node_tree.nodes if n.type=='TEX_IMAGE' and n.label=='BASE COLOR']
        if len(textures)!=1:raise ValueError('Imported continuous base alpha is ambiguous')
        # The importer clips MASK alpha before the Principled Alpha socket.
        # Read the original texture alpha and apply its factor BEFORE that clip.
        alpha=_math(source.node_tree.nodes,source.node_tree.links,'MULTIPLY',
                    textures[0].outputs['Alpha'],alpha,'Continuous Source Alpha')
    proxies['alpha']=alpha;proxies['coverage']=(1,1,1,1);proxies['ao']=1.
    if material_doc.get('occlusionTexture') is not None:
        for node in source.node_tree.nodes:
            if node.type=='GROUP' and node.inputs.get('Occlusion'):
                proxies['ao']=value(node.inputs['Occlusion']);break
    return proxies


def orm_proxy(source, proxies):
    """Pack already-imported effective scalar values, without applying factors twice."""
    node=_node(source.node_tree.nodes,'ShaderNodeCombineColor','Source ORM')
    node.mode='RGB'
    for channel,label in [('Red','ao'),('Green','rough'),('Blue','metal')]:
        value=proxies[label]
        if hasattr(value,'is_output'):
            if getattr(value,'type',None) in ('RGBA','VECTOR'):
                separate=_node(source.node_tree.nodes,'ShaderNodeSeparateColor','Source '+label+' Red')
                separate.mode='RGB';source.node_tree.links.new(value,separate.inputs['Color']);value=separate.outputs['Red']
            source.node_tree.links.new(value,node.inputs[channel])
        else:node.inputs[channel].default_value=float(value[0] if hasattr(value,'__len__') else value)
    return node.outputs['Color']


def transfer_channels(material_doc):
    channels=['base']
    if material_doc.get('alphaMode','OPAQUE')!='OPAQUE':channels.append('alpha')
    channels.append('orm')
    if material_doc.get('emissiveTexture') or any(material_doc.get('emissiveFactor',[0,0,0])):channels.append('emission')
    return channels+['normal']


def set_proxy(source, output, value):
    nodes, links = source.node_tree.nodes, source.node_tree.links
    for link in list(links):
        if link.to_node == output and link.to_socket.name == "Surface":
            links.remove(link)
    emission = nodes.get("Rebake Proxy Emission") or _node(nodes, "ShaderNodeEmission", "Rebake Proxy Emission")
    for link in list(links):
        if link.to_node == emission and link.to_socket.name == "Color":
            links.remove(link)
    if hasattr(value, "is_output"):
        links.new(value, emission.inputs["Color"])
    elif isinstance(value, (int, float)):
        emission.inputs["Color"].default_value = (value, value, value, 1)
    else:
        emission.inputs["Color"].default_value = value
    links.new(emission.outputs[0], output.inputs["Surface"])


def new_map(name: str, resolution: int, color: str, *, float_buffer=False):
    image = bpy.data.images.new(name, resolution, resolution, alpha=True, float_buffer=float_buffer)
    image.generated_color = (0, 0, 0, 0)
    image.alpha_mode = "CHANNEL_PACKED"
    image.colorspace_settings.name = color
    return image


def save_map(image, out_dir: Path, label: str) -> Path:
    path = out_dir / (label + ".png")
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    return path


def pixel_array(image) -> array:
    values = array("f", [0.0]) * (image.size[0] * image.size[1] * 4)
    image.pixels.foreach_get(values)
    return values


def pad_image(image, owners: array, occupancy: bytearray) -> int:
    pixels = pixel_array(image)
    padded = pad_rgba_pixels(pixels, owners, occupancy)
    image.pixels.foreach_set(pixels)
    image.update()
    return padded


def finish_material(low, maps: dict, base, orm, material_doc: dict, emissive_strength: float):
    material = bpy.data.materials.new("Rebaked PBR")
    material.use_nodes = True
    low.data.materials.clear()
    low.data.materials.append(material)
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    def texture(name, image):
        tex = _node(nodes, "ShaderNodeTexImage", name)
        tex.image = image
        tex.extension = 'REPEAT'; tex.interpolation = 'Linear'
        return tex
    base_tex = texture("Rebaked Base", base)
    links.new(base_tex.outputs["Color"], bsdf.inputs["Base Color"])
    mode = material_doc.get("alphaMode", "OPAQUE")
    if mode == "BLEND":
        links.new(base_tex.outputs["Alpha"], bsdf.inputs["Alpha"])
        material.surface_render_method = "DITHERED"
    elif mode == "MASK":
        cutoff = material_doc.get("alphaCutoff", .5)
        less = _node(nodes, "ShaderNodeMath", "Alpha Cutoff")
        less.operation = "LESS_THAN"
        less.inputs[1].default_value = cutoff
        links.new(base_tex.outputs["Alpha"], less.inputs[0])
        subtract = _node(nodes, "ShaderNodeMath", "Alpha Mask")
        subtract.operation = "SUBTRACT"
        subtract.inputs[0].default_value = 1
        links.new(less.outputs[0], subtract.inputs[1])
        links.new(subtract.outputs[0], bsdf.inputs["Alpha"])
    packed = texture("Rebaked ORM", orm)
    separate = _node(nodes, "ShaderNodeSeparateColor", "ORM Channels")
    separate.mode = "RGB"
    links.new(packed.outputs["Color"], separate.inputs["Color"])
    links.new(separate.outputs["Green"], bsdf.inputs["Roughness"])
    links.new(separate.outputs["Blue"], bsdf.inputs["Metallic"])
    from io_scene_gltf2.blender.com.material_helpers import create_settings_group
    settings = _node(nodes, "ShaderNodeGroup", "glTF Material Output")
    settings.node_tree = create_settings_group("glTF Material Output")
    links.new(separate.outputs["Red"], settings.inputs["Occlusion"])
    normal_tex = texture("Rebaked Normal", maps["normal"])
    normal = _node(nodes, "ShaderNodeNormalMap", "Rebaked Tangent Normal")
    normal.space = "TANGENT"
    links.new(normal_tex.outputs["Color"], normal.inputs["Color"])
    links.new(normal.outputs["Normal"], bsdf.inputs["Normal"])
    if "emission" in maps:
        emission_tex = texture("Rebaked Emission", maps["emission"])
        links.new(emission_tex.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = emissive_strength
    material.use_backface_culling = not material_doc.get("doubleSided", False)
    return material




def geometric_ao_rows(lows, owner_lows, occupancy, resolution):
    """CPU SELF AO, compact occupied R only; no borrowed HIGH/proxy occluders."""
    import math
    import time
    import numpy as np
    started=time.monotonic();scene=bpy.context.scene
    mask=np.asarray(occupancy)
    if mask.dtype!=np.bool_ or mask.size!=resolution*resolution:
        raise ValueError('Geometric AO requires the immutable native occupancy')
    mask=mask.reshape(-1)
    if not lows or not owner_lows or not set(lows)<=set(owner_lows):
        raise ValueError('Geometric AO requires final LOWs of one authored object')
    objects=list(scene.objects)
    if any(low not in objects or low.type!='MESH' for low in owner_lows):
        raise ValueError('Geometric AO owner is not an active LOW mesh')
    lower=np.full(3,np.inf);upper=np.full(3,-np.inf)
    for low in owner_lows:
        matrix=np.array(low.matrix_world,dtype=np.float64)
        if not len(low.data.vertices):raise ValueError('Empty geometric AO LOW')
        for start in range(0,len(low.data.vertices),65536):
            vertices=np.array([tuple(vertex.co) for vertex in low.data.vertices[start:start+65536]],dtype=np.float64)
            world=vertices@matrix[:3,:3].T+matrix[:3,3]
            if not np.isfinite(world).all():raise ValueError('Nonfinite geometric AO world vertices')
            lower=np.minimum(lower,world.min(axis=0));upper=np.maximum(upper,world.max(axis=0))
    diagonal=math.sqrt(sum((hi-lo)**2 for lo,hi in zip(lower,upper)))
    distance=.05*diagonal
    if not math.isfinite(distance) or distance<=0:raise ValueError('Invalid geometric AO world bounds')
    native_distance=np.float32(distance)
    if float(native_distance)>distance:native_distance=np.nextafter(native_distance,np.float32(0))
    if not math.isfinite(float(native_distance)) or native_distance<=0:
        raise ValueError('Geometric AO distance is not positive finite binary32')
    visibility=[(obj,obj.hide_render,obj.hide_viewport,obj.hide_get(),obj.select_get()) for obj in objects]
    slots=[(low,tuple(low.data.materials)) for low in owner_lows]
    active=bpy.context.view_layer.objects.active
    cycles=scene.cycles;settings=scene.render.bake
    cycle_fields=('device','samples','seed','use_animated_seed','use_denoising')
    bake_fields=('use_selected_to_active','use_cage','cage_object','margin','use_clear','target')
    saved_cycles={name:getattr(cycles,name) for name in cycle_fields}
    saved_bake={name:getattr(settings,name) for name in bake_fields}
    engine=scene.render.engine;material=None;image=None
    try:
        scene.render.engine='CYCLES';cycles.device='CPU';cycles.samples=16
        cycles.seed=0;cycles.use_animated_seed=False;cycles.use_denoising=False
        settings.use_selected_to_active=False;settings.use_cage=False;settings.cage_object=None
        settings.margin=0;settings.use_clear=True;settings.target='IMAGE_TEXTURES'
        owners=set(owner_lows)
        for obj in objects:
            obj.select_set(False);obj.hide_render=obj not in owners
            if obj in owners:obj.hide_viewport=False;obj.hide_set(False)
        material=bpy.data.materials.new('Temporary geometric AO SELF');material.use_nodes=True
        nodes=material.node_tree.nodes;nodes.clear();links=material.node_tree.links
        output=nodes.new('ShaderNodeOutputMaterial');emission=nodes.new('ShaderNodeEmission')
        emission.inputs['Strength'].default_value=1
        combine=nodes.new('ShaderNodeCombineColor');combine.mode='RGB'
        combine.inputs['Green'].default_value=1;combine.inputs['Blue'].default_value=0
        ao=nodes.new('ShaderNodeAmbientOcclusion');ao.samples=16;ao.inside=False;ao.only_local=False
        ao.inputs['Color'].default_value=(1,1,1,1);ao.inputs['Distance'].default_value=float(native_distance)
        effective_distance=float(ao.inputs['Distance'].default_value)
        if not math.isfinite(effective_distance) or not 0<effective_distance<=distance:
            raise ValueError('Geometric AO RNA distance exceeds its exact world cap')
        geometry=nodes.new('ShaderNodeNewGeometry')
        links.new(geometry.outputs['True Normal'],ao.inputs['Normal'])
        links.new(ao.outputs['AO'],combine.inputs['Red'])
        links.new(combine.outputs['Color'],emission.inputs['Color'])
        links.new(emission.outputs['Emission'],output.inputs['Surface'])
        image=new_map('Temporary AO R / occupancy G',resolution,'Non-Color',float_buffer=True)
        image.generated_type='BLANK';image.generated_color=(0,0,0,0)
        target=nodes.new('ShaderNodeTexImage');target.image=image;nodes.active=target
        for low,prior in slots:
            low.data.materials.clear()
            for _ in range(max(1,len(prior))):low.data.materials.append(material)
        for index,low in enumerate(lows):
            low.select_set(True);bpy.context.view_layer.objects.active=low
            settings.use_clear=index==0
            if bpy.ops.object.bake(type='EMIT')!={'FINISHED'}:raise ValueError('Geometric AO self bake did not finish')
            low.select_set(False)
        # Never create a second full RGBA CPU array or retain an AO atlas with ORM.
        values=np.empty(int(np.count_nonzero(mask)),np.float32);cursor=0
        for start in range(0,len(mask),65536):
            stop=min(start+65536,len(mask))
            block=np.asarray(image.pixels[4*start:4*stop],dtype=np.float32).reshape(-1,4)
            occupied=mask[start:stop]
            if not np.isfinite(block).all() or np.any(block<0) or np.any(block>1):
                raise ValueError('Geometric AO has nonfinite or out-of-range native values')
            if (np.any(block[occupied,1]!=1) or np.any(block[~occupied,1]!=0)
                    or np.any(block[:,2]!=0)):
                raise ValueError('Geometric AO native coverage differs from immutable LOW occupancy')
            count=int(np.count_nonzero(occupied))
            values[cursor:cursor+count]=block[occupied,0];cursor+=count
        if cursor!=len(values):raise ValueError('Geometric AO occupied row count differs')
        report=dict(ao_source='generated_final_LOW',resolution=resolution,occupied_texels=cursor,
            native_coverage_exact=True,coverage_filled_texels=0,engine='CYCLES',device='CPU',
            samples=16,seed=0,AO_samples=16,inside=False,only_local=False,
            normal_input='Geometry.True Normal',distance_world=effective_distance,distance_world_cap=distance,
            distance_policy='0.05 * world AABB diagonal of final LOWs of the same authored object',
            owner_low_parts=len(owner_lows),selected_low_parts=len(lows),margin=0,
            elapsed_seconds=time.monotonic()-started)
    finally:
        for low,prior in slots:
            low.data.materials.clear()
            for prior_material in prior:low.data.materials.append(prior_material)
        for name,value in saved_bake.items():setattr(settings,name,value)
        for name,value in saved_cycles.items():setattr(cycles,name,value)
        scene.render.engine=engine
        for obj,render,viewport,hidden,selected in visibility:
            obj.hide_render=render;obj.hide_viewport=viewport;obj.hide_set(hidden);obj.select_set(selected)
        bpy.context.view_layer.objects.active=active
        if material is not None:bpy.data.materials.remove(material)
        if image is not None:bpy.data.images.remove(image)
        if (any(tuple(low.data.materials)!=prior for low,prior in slots)
                or any((obj.hide_render,obj.hide_viewport,obj.hide_get(),obj.select_get())!=(render,viewport,hidden,selected)
                       for obj,render,viewport,hidden,selected in visibility)
                or bpy.context.view_layer.objects.active!=active or scene.render.engine!=engine
                or any(getattr(settings,name)!=value for name,value in saved_bake.items())
                or any(getattr(cycles,name)!=value for name,value in saved_cycles.items())):
            raise ValueError('Geometric AO failed to restore native scene state')
    report['temporary_image_removed_before_ORM']=True;report['state_restored']=True
    return values,report


def finalize_baked_channel(label, data, occupancy, resolution, material_doc, *, padding_plan, base_rgba=None, generated_ao=False):
    """Finalize then extend RGBA from valid occupied samples, never from alpha.

    Whole-chart conservative base-level bilinear support, REPEAT. Copy only
    ORIGINAL source-valid seeds of the SAME chart after RGBA composition.
    No native occupancy fill, mip/anisotropic or full appearance claim.
    """
    import numpy as np
    from production_uv import apply_chart_gutters
    if label not in ('base','alpha','orm','emission','normal'):raise ValueError('Unknown PBR channel')
    if data.shape!=(resolution*resolution,4):raise ValueError('PBR pixel shape differs')
    mask=np.asarray(occupancy)
    if mask.dtype!=np.bool_ or mask.shape not in ((resolution*resolution,),(resolution,resolution)):
        raise ValueError('Padding requires the immutable boolean atlas occupancy')
    report=dict(policy='Post-composite SAME-chart ORIGINAL source-valid RGBA',native_pbr_margin=0,
                padding_region='Required geometric bilinear support outside immutable native occupancy only',
                guarantee='Base-level bilinear only, REPEAT; not mip/anisotropic/fidelity',
                padded_unoccupied_texels=0,coverage_filled_texels=0,alpha_merge_deferred=False,
                intentional_semantics_change='Replaces radius2 global nearest padding; occupied composed RGBA remains byteexact')
    rgba=data
    if label=='base':
        if material_doc.get('alphaMode','OPAQUE')!='OPAQUE':
            report['alpha_merge_deferred']=True
            return rgba,report
        rgba[:,3]=1.
    elif label=='alpha':
        if base_rgba is None or base_rgba.shape!=data.shape:raise ValueError('Alpha requires assembled base RGBA')
        rgba=base_rgba;rgba[:,3]=np.clip(data[:,0],0,1)
    elif label=='orm':
        rgba[:,:3]=np.clip(rgba[:,:3],0,1)
        if generated_ao and material_doc.get('occlusionTexture') is not None:
            raise ValueError('Generated AO cannot replace authored AO')
        if material_doc.get('occlusionTexture') is None and not generated_ao:rgba[:,0]=1.
        if generated_ao:
            occupied=mask.reshape(-1)
            for start in range(0,len(occupied),65536):
                stop=min(start+65536,len(occupied))
                rgba[start:stop,0][~occupied[start:stop]]=1.
        rgba[:,3]=1.
    elif label=='emission':
        # glTF emissiveTexture ignores A; canonicalize before SAME-chart copying.
        rgba[:,3]=1.
    # NORMAL has already been converted into the unchanged frozen target basis.
    # No colour/alpha value determines source validity or geometric coverage.
    report.update(apply_chart_gutters(rgba,mask,resolution,padding_plan))
    if label=='normal':
        # glTF normalTexture ignores A; preserve all occupied/required RGBA.
        occupied=mask.reshape(-1);charts=padding_plan['required_charts']
        for start in range(0,len(occupied),65536):
            stop=min(start+65536,len(occupied));clear=charts[start:stop]<0
            clear[occupied[start:stop]]=False
            rgba[start:stop,3][clear]=0.
    return rgba,report
