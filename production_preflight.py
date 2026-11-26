# SPDX-License-Identifier: GPL-3.0-or-later
"""Explicit static GLB/PBR subset for the ordinary Blender production worker."""
from pathlib import Path
import math
import struct
from glb_inspector import GLBReader

MAX_BYTES = 512 * 1024**2
MAX_TRIANGLES = 2_000_000
MAP_RESOLUTION_KEYS = dict(base='base_resolution', alpha='base_resolution', normal='normal_resolution',
                           orm='orm_resolution', emission='emission_resolution')
RESOLUTION_PRESETS = (256, 512, 1024, 2048, 4096)


def map_grid_plan(config, channels):
    """One native owner/POSITION grid per active size; NORMAL's grid is last."""
    groups = {}
    for channel in channels:
        resolution = config.get(MAP_RESOLUTION_KEYS[channel], 0) or config['resolution']
        groups.setdefault(resolution, []).append(channel)
    return [dict(resolution=size, channels=labels) for size, labels in groups.items()
            if 'normal' not in labels] + [dict(resolution=size, channels=labels)
            for size, labels in groups.items() if 'normal' in labels]
NODES = {'inspect-mesh', 'conservative-repair', 'simplify-mesh', 'optimize-rebake',
         'uv-repack-rebake', 'rebake-onto-mesh', 'reconstruct-wtivo', 'reconstruct-cumesh-dc'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value, low, high, label):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
            f'{label} must be finite in [{low}, {high}]')
    return value


def parameters(params, node):
    if node=='reconstruct-cumesh-dc':
        defaults=dict(resolution=32,memory_budget_mib=8192,max_output_triangles=1_000_000,timeout_seconds=180,shell_policy='keep_all',drop_below_largest_area_percent=0.,repair_mode='volume')
        require(set(params)<=set(defaults),'Unsupported CuMesh DC parameter; band1 is mandatory')
        result={**defaults,**params}
        number(result['drop_below_largest_area_percent'],0,5,'drop_below_largest_area_percent')
        require(result['shell_policy'] in ('keep_all','verified_outer'),'CuMesh DC shell policy must be keep_all or verified_outer')
        require(type(result['repair_mode'])is str and result['repair_mode'] in ('volume','local_holes'),'CuMesh DC repair_mode must be volume or local_holes')
        require(result['repair_mode']!='local_holes' or (result['shell_policy']=='keep_all' and result['drop_below_largest_area_percent']==0),'CuMesh DC local_holes requires keep_all and drop0')
        # Stock selects emit canonical strings; saved numeric workflows stay valid.
        if type(result['resolution'])is str and result['resolution'] in ('32','64','128'):
            result['resolution']=int(result['resolution'])
        require(type(result['resolution'])is int and result['resolution'] in (32,64,128),'CuMesh DC resolution must be 32, 64 or 128; 128 was verified in one local-hole repair case')
        for key,low,high in [('memory_budget_mib',512,65536),('max_output_triangles',4,1_250_000),('timeout_seconds',30,180)]:
            require(type(result[key])is int and low<=result[key]<=high,f'Invalid {key}: integer in [{low}, {high}] required')
        return result
    if node=='reconstruct-wtivo':
        defaults=dict(input_resolution=32,final_resolution=32,proxy_points=4096,max_rounds=4096,
                      memory_budget_mib=8192,max_output_triangles=1_000_000,timeout_seconds=1800,drop_below_largest_area_percent=0.,repair_mode='volume')
        require(set(params)<=set(defaults),'Unsupported WTiVo parameter')
        result={**defaults,**params}
        number(result['drop_below_largest_area_percent'],0,5,'drop_below_largest_area_percent')
        require(type(result['repair_mode'])is str and result['repair_mode'] in ('volume','local_holes'),'WTiVo repair_mode must be volume or local_holes')
        require(result['repair_mode']!='local_holes' or result['drop_below_largest_area_percent']==0,'WTiVo local_holes requires keep-all source components (drop0)')
        for key,low,high in [('input_resolution',2,256),('final_resolution',2,256),('proxy_points',4,1_000_000),
                             ('max_rounds',1,65536),('memory_budget_mib',512,65536),
                             ('max_output_triangles',4,1_250_000),('timeout_seconds',30,14400)]:
            require(type(result[key])is int and low<=result[key]<=high,f'Invalid {key}: integer in [{low}, {high}] required')
        return result
    defaults = dict(max_error_relative=.003, max_error_world=.01,
                    resolution=2048, projection_distance_relative=.01,
                    cage_extrusion_relative=.005, protected_component_triangles=64,
                    timeout_seconds=14400, base_resolution=0, normal_resolution=0,
                    orm_resolution=0, emission_resolution=0)
    if node == 'inspect-mesh':
        require(set(params) <= {'analysis_mode','timeout_seconds'}, 'Unsupported inspection parameter')
        require(params.get('analysis_mode', 'structural') in ('structural', 'geometry'), 'Invalid analysis_mode')
        timeout=params.get('timeout_seconds',600)
        require(type(timeout)is int and 30<=timeout<=3600,'Inspection timeout_seconds must be an integer in [30, 3600]')
        return dict(analysis_mode=params.get('analysis_mode', 'structural'),timeout_seconds=timeout)
    if node in ('simplify-mesh','optimize-rebake'):
        defaults['triangle_budget']=80000
        defaults['simplification_backend']='blender'
    if node in ('optimize-rebake','uv-repack-rebake'):
        defaults['atlas_packing']='full'
    if node in ('optimize-rebake','uv-repack-rebake','rebake-onto-mesh'):
        defaults['ao_mode']='transfer'
    allowed = set(defaults) | ({'high_poly_path'} if node in ('rebake-onto-mesh','uv-repack-rebake') else set())
    require(set(params) <= allowed, 'Unsupported parameter: '+', '.join(sorted(set(params)-allowed)))
    result = {**defaults, **params}
    if node in ('optimize-rebake','uv-repack-rebake','rebake-onto-mesh'):
        require(type(result['ao_mode']) is str and result['ao_mode'] in ('transfer','generate_if_missing'),
                'ao_mode must be transfer or generate_if_missing')
    if node in ('optimize-rebake','uv-repack-rebake'):
        require(type(result['atlas_packing']) is str and result['atlas_packing'] in ('full','compact'),
                'atlas_packing must be full or compact')
    if node in ('optimize-rebake','uv-repack-rebake','rebake-onto-mesh'):
        # Stock Modly numeric selects emit canonical option strings on change.
        for key in ('resolution', *set(MAP_RESOLUTION_KEYS.values())):
            presets=RESOLUTION_PRESETS if key=='resolution' else (0,*RESOLUTION_PRESETS)
            canonical={str(value):value for value in presets}
            if type(result[key]) is str and result[key] in canonical:
                result[key]=canonical[result[key]]
    if node in ('simplify-mesh','optimize-rebake'):
        require(result['simplification_backend'] in ('blender','cumesh_install_if_needed','cumesh_existing'), 'Unsupported simplification_backend')
    if node in ('simplify-mesh','optimize-rebake'):
        require(type(result['triangle_budget']) is int and 1 <= result['triangle_budget'] <= MAX_TRIANGLES, 'Invalid triangle_budget')
    for key, low, high in [('resolution', 256, 4096),
                           ('protected_component_triangles', 0, 10000), ('timeout_seconds', 30, 86400)]:
        require(type(result[key]) is int and low <= result[key] <= high, f'Invalid {key}')
    require(result['resolution'] in RESOLUTION_PRESETS, 'Resolution must be a power-of-two preset')
    for key in set(MAP_RESOLUTION_KEYS.values()):
        require(type(result[key]) is int and result[key] in (0, *RESOLUTION_PRESETS),
                f'{key} must be 0 (inherit resolution) or a power-of-two preset')
    for key, low, high in [('max_error_relative', 1e-7, .1), ('max_error_world', 1e-9, 1000),
                           ('projection_distance_relative', 1e-6, .1), ('cage_extrusion_relative', 0, .1)]:
        number(result[key],low,high,key)
    require(result['cage_extrusion_relative'] <= result['projection_distance_relative'], 'Cage exceeds projection distance')
    if node=='rebake-onto-mesh':
        require(isinstance(result.get('high_poly_path'),str) and result['high_poly_path'], 'high_poly_path is required')
    elif node=='uv-repack-rebake':
        require(isinstance(result.get('high_poly_path',''),str),'high_poly_path must be a string')
    return result


def plan_geometry_source(path):
    """Static geometry-only input; no source UV/PBR or correspondence promise."""
    from inspect_native_geometry import plan_geometry_inspection
    plan=plan_geometry_inspection(path)
    names=[item['name'] for item in plan['mesh_nodes']];meshes=[item['mesh'] for item in plan['mesh_nodes']]
    require(all(isinstance(name,str) and name for name in names) and len(set(names))==len(names),
            'Geometry-only processing requires distinct nonempty mesh object names')
    require(len(set(meshes))==len(meshes),'Geometry-only processing does not support mesh instances')
    return dict(objects=names,triangles=plan['original_active_triangles'],materials=[],geometry_only=True,
                native_plan=plan,appearance='INVALIDATED; no UV/PBR correspondence inferred')


def plan_source(path, *, target=False, require_final_uv=False, allow_generated_target_normals=False, inspection=None):
    path=Path(path)
    require(not allow_generated_target_normals or (target and not require_final_uv),
            'Generated target normals are only supported for UV / Repack + Rebake')
    require(path.is_file() and path.suffix.lower()=='.glb', 'A GLB file is required')
    require(path.stat().st_size <= MAX_BYTES, 'Source exceeds 512 MiB input limit')
    position_checked=(isinstance(inspection,dict) and isinstance(inspection.get('assessed'),dict)
        and inspection['assessed'].get('position_finiteness')is True and isinstance(inspection.get('counts'),dict)
        and type(inspection['counts'].get('nonfinite_position_vertices'))is int
        and inspection['counts']['nonfinite_position_vertices']>=0)
    if allow_generated_target_normals and position_checked:
        require(inspection['counts']['nonfinite_position_vertices']==0,'Nonfinite provided target POSITION')
    with GLBReader(path) as r:
        d=r.doc
        require(not any(d.get(k) for k in ('animations','skins','cameras')), 'Animation, skins and cameras are unsupported')
        require(set(d.get('extensionsUsed',[])) <= {'EXT_texture_webp','KHR_materials_emissive_strength'}, 'Unsupported glTF extension')
        require(set(d.get('extensionsRequired',[])) <= set(d.get('extensionsUsed',[])), 'Invalid required extensions')
        require(len(d.get('buffers',[]))==1 and not d['buffers'][0].get('uri'), 'Only embedded GLB resources are supported')
        objects=[]; mesh_ids=[]; count=0
        for n in d.get('nodes',[]):
            require(not set(n)-{'name','mesh','children','matrix','translation','rotation','scale'}, 'Unsupported node semantics')
            if 'mesh' not in n: continue
            require(type(n['mesh']) is int and 0<=n['mesh']<len(d.get('meshes',[])), 'Invalid mesh node')
            name=n.get('name')
            require(isinstance(name,str) and name and name not in objects, 'Distinct mesh object names are required')
            objects.append(name);mesh_ids.append(n['mesh'])
            mesh=d['meshes'][n['mesh']]
            require(not set(mesh)-{'name','primitives','extras'} and not mesh.get('extras'), 'Morph targets, mesh extensions and nonempty metadata are unsupported')
            for prim in mesh['primitives']:
                require(not set(prim)-{'attributes','indices','material','mode'}, 'Unsupported primitive semantics')
                require(prim.get('mode',4)==4 and type(prim.get('indices')) is int, 'Indexed TRIANGLES are required')
                attrs=prim.get('attributes',{})
                required=({'POSITION'} if allow_generated_target_normals else {'POSITION','NORMAL'}) if target else {'POSITION','NORMAL','TEXCOORD_0'}
                require(required <= set(attrs) <= {'POSITION','NORMAL','TANGENT','TEXCOORD_0'}, 'One UV set, normals and no custom/color attributes are required')
                require('TANGENT' not in attrs or 'NORMAL' in attrs,'TANGENT requires authored NORMAL')
                require(not require_final_uv or 'TEXCOORD_0' in attrs,
                        'Rebake onto Mesh requires final LOW TEXCOORD_0 UV; use UV / Repack + Rebake to generate it')
                layouts={semantic:r.accessor_layout(index) for semantic,index in attrs.items()}
                require(all(not layout[5] for layout in layouts.values()), 'Sparse accessors are unsupported')
                rows=layouts['POSITION'][1]
                for semantic,index in attrs.items():
                    accessor=d['accessors'][index];layout=layouts[semantic]
                    require(accessor.get('componentType')==5126 and accessor.get('type')=={'POSITION':'VEC3','NORMAL':'VEC3','TANGENT':'VEC4','TEXCOORD_0':'VEC2'}[semantic] and not accessor.get('normalized') and layout[1]==rows, 'Unsupported attribute layout')
                    if allow_generated_target_normals:
                        offset,count_rows,stride,_,_,_=layout
                        require(offset is not None,'Dense provided target '+semantic+' is required')
                        if semantic!='POSITION' or not position_checked:
                            unpack=struct.Struct('<'+{'POSITION':'3f','NORMAL':'3f','TANGENT':'4f','TEXCOORD_0':'2f'}[semantic]).unpack_from
                            require(all(math.isfinite(value) for row in range(count_rows) for value in unpack(r.data,offset+row*stride)),
                                    'Nonfinite provided target '+semantic)
                layout=r.accessor_layout(prim['indices'])
                require(d['accessors'][prim['indices']].get('type')=='SCALAR' and layout[4] in (5121,5123,5125) and not layout[5], 'Unsupported index accessor')
                require(all(0<=value<rows for value in r.scalar_values(prim['indices'])), 'Out-of-range vertex indices')
                size=layout[1]
                require(size%3==0, 'Invalid triangle indices');count+=size//3
                require((target and 'material' not in prim) or (type(prim.get('material')) is int and 0<=prim['material']<len(d.get('materials',[]))), 'Explicit valid source materials are required')
        require(objects and len(objects)<=64 and len(set(mesh_ids))==len(mesh_ids), 'Mesh instances or more than 64 objects are unsupported')
        require(len(mesh_ids)==len(d.get('meshes',[])), 'Unused meshes are unsupported')
        require(1<=count<=MAX_TRIANGLES, 'Source triangle count must be 1–2000000')
        require(len(d.get('materials',[]))<=16 and len(d.get('images',[]))<=32, 'Source exceeds 16 materials or 32 images')
        materials=[]
        for material in d.get('materials',[]):
            require(not set(material)-{'name','pbrMetallicRoughness','normalTexture','occlusionTexture','emissiveTexture','emissiveFactor','alphaMode','alphaCutoff','doubleSided','extensions'}, 'Unsupported material feature')
            require(set(material.get('extensions',{})) <= {'KHR_materials_emissive_strength'}, 'Unsupported material extension')
            pbr=material.get('pbrMetallicRoughness',{})
            require(not set(pbr)-{'baseColorFactor','baseColorTexture','metallicFactor','roughnessFactor','metallicRoughnessTexture'}, 'Unsupported PBR model')
            for key,width,default in [('baseColorFactor',4,[1]*4),('emissiveFactor',3,[0]*3)]:
                value=pbr.get(key,default) if key=='baseColorFactor' else material.get(key,default)
                require(isinstance(value,list) and len(value)==width, 'Invalid material factor')
                for v in value:number(v,0,1,key)
            for key in ('metallicFactor','roughnessFactor'):number(pbr.get(key,1),0,1,key)
            require(material.get('alphaMode','OPAQUE') in ('OPAQUE','MASK','BLEND'), 'Invalid alpha mode')
            number(material.get('alphaCutoff',.5),0,1,'alpha cutoff')
            require(type(material.get('doubleSided',False)) is bool, 'Invalid doubleSided')
            require(set(material.get('extensions',{}).get('KHR_materials_emissive_strength',{})) <= {'emissiveStrength'}, 'Unsupported emission extension field')
            strength=material.get('extensions',{}).get('KHR_materials_emissive_strength',{}).get('emissiveStrength',1)
            number(strength,0,1000,'emissive strength')
            slots=[(pbr.get('baseColorTexture'),set()),(pbr.get('metallicRoughnessTexture'),set()),
                   (material.get('normalTexture'),{'scale'}),(material.get('occlusionTexture'),{'strength'}),(material.get('emissiveTexture'),set())]
            for info,extra in slots:
                if info is None:continue
                require(isinstance(info,dict) and not set(info)-({'index','texCoord'}|extra) and info.get('texCoord',0)==0, 'Texture transforms and secondary UVs are unsupported')
                require(type(info.get('index')) is int and 0<=info['index']<len(d.get('textures',[])), 'Invalid texture reference')
                if 'scale' in info:number(info['scale'],0,10,'normal scale')
                if 'strength' in info:
                    number(info['strength'],0,1,'AO strength')
                    require(info['strength']==1, 'Non-default AO strength is not yet supported by the imported-shader transfer')
            materials.append(material)
        for t in d.get('textures',[]):
            require(not set(t)-{'name','source','sampler','extensions'} and set(t.get('extensions',{})) <= {'EXT_texture_webp'}, 'Unsupported texture feature')
            src=t.get('extensions',{}).get('EXT_texture_webp',{}).get('source',t.get('source'))
            require(type(src) is int and 0<=src<len(d.get('images',[])), 'Invalid image reference')
            require('sampler' not in t or type(t['sampler']) is int and 0<=t['sampler']<len(d.get('samplers',[])), 'Invalid sampler index')
        for image in d.get('images',[]):
            require(not set(image)-{'name','bufferView','mimeType'} and image.get('mimeType') in ('image/png','image/jpeg','image/webp') and type(image.get('bufferView')) is int, 'Only embedded PNG/JPEG/WebP images are supported')
        for s in d.get('samplers',[]):
            require(not set(s)-{'name','magFilter','minFilter','wrapS','wrapT'} and s.get('magFilter',9729)==9729 and s.get('minFilter',9987)==9987 and s.get('wrapS',10497)==s.get('wrapT',10497)==10497, 'Only linear repeat source samplers are supported')
        return dict(objects=objects,triangles=count,materials=materials)
