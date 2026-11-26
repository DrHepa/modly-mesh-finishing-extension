# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional read-only original UV census, not topology repair or bake admission.

Unique mesh/primitive/UV-set definitions are isolated. No world/object or
cross-primitive weld is attempted. NumPy is lazy and absence is diagnostic only.
"""
import re
import struct
import time
from glb_inspector import GLBReader

MAX_TRIANGLES=1250000  # Aggregate triangle/UV-set work, not instance expansion.
MAX_ROWS=1500000
MAX_SCOPES=128
MAX_BYTES=1024**3     # Conservative logical scratch admission, not measured RSS.
MAX_SECONDS=60       # Cooperative; each bounded NumPy operation is indivisible.
MAX_IMAGE_BYTES=64*1024**2


class NotEvaluated(ValueError):pass


def _array(reader,index,shape,kind,np):
    a=reader.doc['accessors'][index]
    if a['type']!=shape or a['componentType']not in kind or a.get('normalized',False)is not False:
        raise NotEvaluated('Unsupported accessor type/normalization for exact original values')
    off,n,stride,fmt,component,sparse=reader.accessor_layout(index)
    if sparse or off is None:raise NotEvaluated('Sparse accessor values are not decoded')
    width={'SCALAR':1,'VEC2':2,'VEC3':3}[shape];dtype=np.dtype('<'+{5126:'f4',5125:'u4',5123:'u2',5121:'u1'}[component])
    # Own one compact buffer; never retain mmap views across reader.close().
    return np.ndarray((n,width),dtype=dtype,buffer=reader.data,offset=off,strides=(stride,dtype.itemsize)).copy()


def _census(faces,np,check):
    """Exact full undirected edge incidents, including nonmanifold groups.

    Adapted from the measured RAW census; no Python per-face/edge objects.
    Point-only contact does not connect; no triangle filtering is performed.
    """
    count=len(faces);parents=np.arange(count,dtype=np.uint32)
    def root(i):
        i=int(i)
        while parents[i]!=i:parents[i]=parents[parents[i]];i=int(parents[i])
        return i
    check();edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])).astype(np.uint32);edges.sort(axis=1)
    check();order=np.lexsort((edges[:,1],edges[:,0]));ordered=edges[order]
    starts=np.flatnonzero(np.r_[True,np.any(ordered[1:]!=ordered[:-1],axis=1),True]);del edges,ordered
    joined=0
    for ordinal,(a,b)in enumerate(zip(starts[:-1],starts[1:])):
        if ordinal%4096==0:check()
        group=order[a:b];first=int(group[0]%count)
        for other in group[1:]:
            x,y=root(first),root(int(other%count))
            if x!=y:parents[max(x,y)]=min(x,y);joined+=1
    del order,starts
    roots=np.empty(count,dtype=np.uint32)
    for i in range(count):
        if i%4096==0:check()
        roots[i]=root(i)
    check();_,sizes=np.unique(roots,return_counts=True);check()
    return dict(components=int(len(sizes)),single_triangle_components=int(np.count_nonzero(sizes==1)),
        minimum_faces=int(sizes.min()),maximum_faces=int(sizes.max()),union_links=joined,faces_counted=int(sizes.sum()))


def _images(reader,check):
    """Existing bounded still-WebP header profile, PNG IHDR; no pixel decode."""
    result=[];used=0;doc=reader.doc
    for i,image in enumerate(doc.get('images',[])[:MAX_SCOPES]):
        check();row=dict(image=i,mime=image.get('mimeType'),status='NOT_EVALUATED',pixels_decoded=False)
        try:
            view_id=image.get('bufferView')
            if type(view_id)is not int or not 0<=view_id<len(doc.get('bufferViews',[])):raise NotEvaluated('Embedded bufferView required; data URI metadata not decoded')
            view=doc['bufferViews'][view_id]
            if not isinstance(view,dict):raise NotEvaluated('Image bufferView entry must be an object')
            start=view.get('byteOffset',0);length=view.get('byteLength')
            if view.get('buffer')!=0 or type(start)is not int or type(length)is not int or start<0 or length<24 or reader.bin_offset is None or start+length>doc['buffers'][0]['byteLength']:
                raise NotEvaluated('Invalid embedded image bounds')
            offset=reader.bin_offset+start
            if row['mime']=='image/png':
                head=reader.data[offset:offset+24]
                if head[:8]!=b'\x89PNG\r\n\x1a\n'or head[8:16]!=b'\0\0\0\rIHDR':raise NotEvaluated('PNG IHDR unavailable')
                width,height=struct.unpack_from('>II',head,16)
                if not width or not height:raise NotEvaluated('Invalid PNG header dimensions')
                row['header_profile']='PNG_IHDR_only_not_bitstream_validation'
            elif row['mime']=='image/webp':
                if length>MAX_IMAGE_BYTES-used:raise NotEvaluated('Image metadata encoded-byte resource limit')
                used+=length
                raise NotEvaluated('WebP image dimensions not assessed; UV connectivity is still measured')
                row.update(header_profile='existing_bounded_still_WebP_profile_1_to_4096',ignored_RIFF_tail_bytes=envelope['ignored_tail_bytes'])
                del data,riff
            else:raise NotEvaluated('Image header format not evaluated (no decoder invoked)')
            row.update(status='HEADER_ONLY',width=width,height=height)
        except(ValueError,KeyError,TypeError,MemoryError)as error:row['reason']=str(error)[:256]
        result.append(row)
    return result


def inspect_source_uv(path,inspection,*,check=None,progress=None):
    """Partial diagnostics never replace structural Inspect or geometry checks."""
    started=time.monotonic();deadline=started+MAX_SECONDS
    def checkpoint():
        if check is not None:check()  # Cancellation is never a diagnostic refusal.
        if time.monotonic()>deadline:raise NotEvaluated('UV census cooperative time resource limit')
    result=dict(schema_version='1.0.0',status='PER_PRIMITIVE_ONLY',
        scope='Unique mesh/primitive/UV-set definitions in original local accessor coordinates; not instance-expanded',
        cross_primitive_object_continuity='NOT_EVALUATED',
        semantics='Exact float32 POSITION+UV endpoint equality across full undirected edges; +/-zero equal in temporary keys. No proximity weld, face removal or source writes. Literal render-row charts and POSITION-only edge components are separate; UV discontinuities are not holes or independent objects. Coincident full edges within one primitive may join; these counts do not certify surface topology, overlap, density, bake or visual quality.',
        unique_mesh_triangles=inspection['counts']['unique_mesh_render_triangles'],active_scene_triangles=inspection['counts']['active_scene_render_triangles'],
        triangles_assessed_at_least_one_UV_set=0,triangle_UV_pairs_counted=0,omitted_scopes=0,
        limits=dict(triangle_UV_pairs=MAX_TRIANGLES,rows_per_primitive=MAX_ROWS,logical_scratch_bytes=MAX_BYTES,cooperative_seconds=MAX_SECONDS,scope_rows=MAX_SCOPES),scopes=[],images=[])
    used=0;assessed=set()
    with GLBReader(path)as reader:
        try:result['images']=_images(reader,checkpoint)
        except NotEvaluated as error:result['image_metadata_reason']=str(error)
        result['omitted_images']=max(0,len(reader.doc.get('images',[]))-MAX_SCOPES)
        for p in inspection['primitives']:
            primitive=reader.doc['meshes'][p['mesh']]['primitives'][p['primitive']];attributes=primitive['attributes']
            sets=sorted(k for k in attributes if re.fullmatch(r'TEXCOORD_[0-9]+',k))or[None]
            for uv_set in sets:
                if len(result['scopes'])>=MAX_SCOPES:result['omitted_scopes']+=1;continue
                row=dict(mesh=p['mesh'],primitive=p['primitive'],material=p['material'],UV_set=uv_set,triangles=p['render_triangles'],
                    instance_nodes=[],instance_nodes_omitted=0,status='NOT_EVALUATED')
                result['scopes'].append(row)
                try:
                    checkpoint();n=p['render_triangles'];v=p['render_vertices']
                    for i,node in enumerate(reader.doc.get('nodes',[])):
                        if i%4096==0:checkpoint()
                        if isinstance(node,dict)and node.get('mesh')==p['mesh']:
                            if len(row['instance_nodes'])<64:row['instance_nodes'].append(i)
                            else:row['instance_nodes_omitted']+=1
                    material=p['material']
                    if material is not None and(type(material)is not int or not 0<=material<len(reader.doc.get('materials',[]))):
                        raise NotEvaluated('Invalid material reference; per-material UV scope not established')
                    if uv_set is None:raise NotEvaluated('No UV set; no UV chart count is invented')
                    if primitive.get('mode',4)!=4 or p['draw_elements']%3:raise NotEvaluated('Only complete TRIANGLES are evaluated; strips/fans/other modes remain structural only')
                    if not 0<n or v>MAX_ROWS or used+n>MAX_TRIANGLES or 512*n+256*v+8*1024**2>MAX_BYTES:
                        raise NotEvaluated('UV census triangle/row/logical scratch resource limit')
                    used+=n
                    try:import numpy as np
                    except(ImportError,OSError)as error:raise NotEvaluated('Optional NumPy unavailable; structural Inspect remains supported')from error
                    if progress is not None:progress(f'mesh {p["mesh"]} primitive {p["primitive"]} {uv_set}')
                    pos=_array(reader,attributes['POSITION'],'VEC3',(5126,),np);uv=_array(reader,attributes[uv_set],'VEC2',(5126,),np)
                    if len(pos)!=len(uv)or not np.isfinite(pos).all()or not np.isfinite(uv).all():raise NotEvaluated('Nonfinite or mismatched POSITION/UV data')
                    faces=(_array(reader,primitive['indices'],'SCALAR',(5121,5123,5125),np).reshape(-1,3).astype(np.uint32)
                           if'indices'in primitive else np.arange(v,dtype=np.uint32).reshape(-1,3))
                    if np.any(faces>=v):raise NotEvaluated('Out-of-range original index values')
                    literal=_census(faces,np,checkpoint)
                    checkpoint();keys=np.column_stack((pos,uv)).view(np.uint32);keys[keys==0x80000000]=0
                    _,mapping=np.unique(keys,axis=0,return_inverse=True);del keys
                    stitched=_census(mapping[faces].astype(np.uint32),np,checkpoint);del mapping
                    checkpoint();keys=pos.view(np.uint32).copy();keys[keys==0x80000000]=0
                    _,mapping=np.unique(keys,axis=0,return_inverse=True);del keys
                    geometry=_census(mapping[faces].astype(np.uint32),np,checkpoint)
                    checkpoint();row.update(status='EVALUATED',literal_render_row_charts=literal,position_uv_stitched_charts=stitched,geometry_only_edge_components=geometry)
                    result['triangle_UV_pairs_counted']+=n
                    if(p['mesh'],p['primitive'])not in assessed:result['triangles_assessed_at_least_one_UV_set']+=n;assessed.add((p['mesh'],p['primitive']))
                except(NotEvaluated,MemoryError)as error:row['reason']=str(error)[:256]or'UV census allocation resource refusal'
                finally:pos=uv=faces=keys=mapping=None
    if result['omitted_scopes']or any(x['status']!='EVALUATED'for x in result['scopes']):result['status']='PARTIAL'
    return result
