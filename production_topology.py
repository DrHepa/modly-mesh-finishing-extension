# SPDX-License-Identifier: GPL-3.0-or-later
"""Lossless render-row staging and a face-preserving component census."""
from copy import deepcopy
import numpy as np
from conservative_repair import _zero_area


INDEX_DTYPES={5121:np.dtype('u1'),5123:np.dtype('<u2'),5125:np.dtype('<u4')}
ATTRIBUTE_WIDTHS={'POSITION':3,'NORMAL':3,'TANGENT':4,'TEXCOORD_0':2}


def _rows(doc, binary, index, width, dtype):
    accessor=doc['accessors'][index];view=doc['bufferViews'][accessor['bufferView']]
    size=width*dtype.itemsize;stride=view.get('byteStride',size);count=accessor['count']
    offset=accessor.get('byteOffset',0)
    if accessor.get('sparse') or accessor.get('normalized') or view.get('buffer')!=0 or stride<size or offset+(count-1)*stride+size>view['byteLength']:
        raise ValueError('Unsupported or out-of-range lossless import accessor')
    return np.ndarray((count,width),dtype=dtype,buffer=binary,
                      offset=view.get('byteOffset',0)+offset,strides=(stride,dtype.itemsize)).copy()


def stage_duplicate_faces(original_doc, binary, *, prepare=False):
    """Give repeated index-triple faces identical private corner rows.

    Blender Mesh.validate removes even opposite-winding duplicate index faces.
    Only disposable render storage changes here; every corner attribute and
    triangle order/value survives, and the original BIN prefix stays untouched.
    Explicit preparation additionally removes only exactly zero-area faces.
    """
    doc=deepcopy(original_doc);staged=bytearray(binary);records=[];preparation=[]
    def append(accessor, payload, count, target):
        staged.extend(b'\0'*(-len(staged)%4));offset=len(staged);staged.extend(payload)
        doc['bufferViews'].append(dict(buffer=0,byteOffset=offset,byteLength=len(payload),target=target))
        value=dict(accessor,bufferView=len(doc['bufferViews'])-1,byteOffset=0,count=count)
        doc['accessors'].append(value)
        return len(doc['accessors'])-1
    for mesh_index,mesh in enumerate(doc.get('meshes',[])):
        for primitive_index,primitive in enumerate(mesh['primitives']):
            accessor=doc['accessors'][primitive['indices']];component=accessor['componentType']
            faces=_rows(doc,binary,primitive['indices'],1,INDEX_DTYPES[component]).reshape(-1,3)
            source_rows=np.arange(len(faces));removed=[];before=len(faces)
            if prepare:
                vertices=_rows(doc,binary,primitive['attributes']['POSITION'],3,np.dtype('<f4'))
                a,b,c=(vertices[faces[:,i]].astype(np.float64) for i in range(3))
                candidates=np.flatnonzero(np.all(np.cross(b-a,c-a)==0,axis=1))
                removed=[int(i) for i in candidates if _zero_area(*(tuple(float(v) for v in vertices[index]) for index in faces[i]))]
                keep=np.ones(before,dtype=bool);keep[removed]=False
                faces=faces[keep];source_rows=source_rows[keep]
                if not len(faces):raise ValueError('Conservative source preparation would remove an entire primitive')
                preparation.append(dict(mesh=mesh_index,primitive=primitive_index,raw_triangles=before,
                    prepared_triangles=len(faces),removed_exact_zero_area_faces=removed))
            _,first,inverse=np.unique(np.sort(faces,axis=1),axis=0,return_index=True,return_inverse=True)
            repeated=np.flatnonzero(np.arange(len(faces))!=first[inverse])
            if not len(repeated) and not removed:continue
            attributes=primitive['attributes'];rows=doc['accessors'][attributes['POSITION']]['count']
            corners=faces[repeated].ravel();added=len(corners);maximum=rows+added-1
            if maximum>np.iinfo(np.uint32).max:raise ValueError('Lossless import vertex indices exceed uint32')
            if added and maximum>np.iinfo(INDEX_DTYPES[component]).max:component=5123 if maximum<=65535 else 5125
            faces=faces.astype(INDEX_DTYPES[component],copy=True)
            faces[repeated]=np.arange(rows,rows+added,dtype=INDEX_DTYPES[component]).reshape(-1,3)
            for semantic,index in list(attributes.items()) if added else ():
                source=doc['accessors'][index]
                if semantic not in ATTRIBUTE_WIDTHS or source['componentType']!=5126 or source['count']!=rows:
                    raise ValueError('Unsupported corner attribute for lossless import staging')
                values=_rows(doc,binary,index,ATTRIBUTE_WIDTHS[semantic],np.dtype('<f4'))
                payload=np.concatenate((values,values[corners]),axis=0).tobytes()
                attributes[semantic]=append(source,payload,rows+added,34962)
            index_accessor=dict(accessor,componentType=component,min=[int(faces.min())],max=[int(faces.max())])
            primitive['indices']=append(index_accessor,faces.tobytes(),int(faces.size),34963)
            records.append(dict(mesh=mesh_index,primitive=primitive_index,repeated_faces=repeated.tolist(),
                                repeated_source_faces=source_rows[repeated].tolist(),
                                duplicated_faces=len(repeated),duplicated_vertex_rows=added,
                                original_vertex_rows=rows,staged_vertex_rows=rows+added))
    staged.extend(b'\0'*(-len(staged)%4));doc['buffers'][0]['byteLength']=len(staged)
    audit=dict(operation='lossless_duplicate_render_rows',
        duplicated_faces=sum(r['duplicated_faces'] for r in records),
        duplicated_vertex_rows=sum(r['duplicated_vertex_rows'] for r in records),primitives=records)
    if prepare:audit['preparation']=dict(operation='remove_exact_zero_local_area_indexed_triangles',
        raw_triangles=sum(r['raw_triangles'] for r in preparation),
        prepared_triangles=sum(r['prepared_triangles'] for r in preparation),
        removed_exact_zero_area_triangles=sum(len(r['removed_exact_zero_area_faces']) for r in preparation),primitives=preparation)
    return doc,bytes(staged),audit


def uv_corner_equivalence(vertices, faces, uv, barriers, *, position_keys=False):
    """Logical chart IDs only; never weld mesh rows or change corner attributes."""
    ids=faces.ravel()
    if position_keys:
        _,positions=np.unique(vertices,axis=0,return_inverse=True)
        ids=positions[faces].ravel()
    corners=np.column_stack((ids,uv.reshape(-1,2)))
    groups=components(corners,np.arange(len(corners),dtype=np.int32).reshape(-1,3),edge_barriers=barriers)
    charts=np.empty(len(faces),np.int32)
    for index,group in enumerate(groups):charts[group]=index
    _,inverse=np.unique(np.column_stack((np.repeat(charts,3),corners)),axis=0,return_inverse=True)
    return inverse,charts


def split_exact_opposite_singletons(vertices, faces, groups, singletons, matrix, source_count):
    """Recover census-only pairs within one authored object/material group.

    A third HIGH face on the same support makes all three incident at each
    edge, so it is also a singleton in our prepared census. Repeated oriented
    singleton keys therefore reject ambiguous support without a nearest query.
    """
    if len(groups)>=source_count:return groups,{}
    matrix=np.asarray(matrix)
    if matrix.shape!=(4,4) or matrix.dtype!=np.float64 or not np.isfinite(matrix).all():return groups,{}
    def key(triangle):
        triangle=np.asarray(triangle)
        if triangle.shape!=(3,3) or triangle.dtype!=np.float32 or not np.isfinite(triangle).all():return None
        if not np.any(np.cross(triangle[1].astype(float)-triangle[0],triangle[2].astype(float)-triangle[0])):return None
        rows=tuple(row.tobytes() for row in triangle)
        return min(rows,rows[1:]+rows[:1],rows[2:]+rows[:2])
    source_keys={}
    for index,triangle,source_matrix in singletons:
        source_matrix=np.asarray(source_matrix)
        if source_matrix.shape!=matrix.shape or source_matrix.dtype!=matrix.dtype or source_matrix.tobytes()!=matrix.tobytes():continue
        oriented=key(triangle)
        if oriented is not None:source_keys.setdefault(oriented,[]).append(index)
    recovered=[];owners={};reserved=set()
    for group in groups:
        if len(group)==2:
            first,second=(vertices[faces[index]] for index in group)
            a,b=key(first),key(second)
            if a is not None and a!=b and b==key(first[::-1]):
                matches=[source_keys.get(oriented,[]) for oriented in (a,b)]
                if any(len(match)>1 for match in matches):return groups,{}
                if all(len(match)==1 for match in matches):
                    indices=[match[0] for match in matches]
                    if len(set(indices))!=2 or reserved.intersection(indices):return groups,{}
                    reserved.update(indices)
                    for face,index in zip(group,indices):
                        owners[len(recovered)]=index
                        recovered.append(np.array([face],dtype=group.dtype))
                    continue
        recovered.append(group)
    return (recovered,owners) if len(recovered)==source_count else (groups,{})


def partition_interface_edges(vertices, faces, owners):
    """Local incidence-one whole edges shared by distinct parts of ONE object.

    Call separately per authored object. Original openings and render splits
    within one part are not interfaces. Work/storage are bounded by the existing
    two-million-triangle domain (at most three edge/owner records per face).
    """
    vertices=np.asarray(vertices);faces=np.asarray(faces);owners=np.asarray(owners)
    if (faces.ndim!=2 or faces.shape[1]!=3 or not 0<len(faces)<=2_000_000
            or owners.shape!=(len(faces),) or not np.issubdtype(owners.dtype,np.integer)
            or np.any(owners<0) or np.any(owners>=1024)):
        raise ValueError('Invalid bounded partition interface ownership')
    if len(np.unique(owners))<2:return {}
    positions,inverse=np.unique(vertices,axis=0,return_inverse=True)
    canonical=inverse[faces];edges=np.sort(np.stack((canonical,np.roll(canonical,-1,axis=1)),axis=-1).reshape(-1,2),axis=1)
    valid=edges[:,0]!=edges[:,1]
    records=np.column_stack((edges[valid],np.repeat(owners,3)[valid]))
    records,incidence=np.unique(records,axis=0,return_counts=True)
    starts=np.flatnonzero(np.r_[True,np.any(records[1:,:2]!=records[:-1,:2],axis=1)])
    lengths=np.diff(np.r_[starts,len(records)])
    shared=records[np.repeat(lengths>1,lengths)&(incidence==1)]
    result={}
    if not len(shared):return result
    shared=shared[np.argsort(shared[:,2],kind='stable')]
    owners,starts=np.unique(shared[:,2],return_index=True)
    for owner,start,end in zip(owners,starts,np.r_[starts[1:],len(shared)]):
        values=positions[shared[start:end,:2]].astype(np.float32,copy=True)
        values[values==0]=0;values.setflags(write=False);result[int(owner)]=values
    return result


def _interface_position_keys(values):
    values=np.ascontiguousarray(values,dtype=np.float32).copy();values[values==0]=0
    return values.view(np.dtype([('v'+str(i),'<f4') for i in range(3)])).reshape(-1)


def partition_interface_vertices(vertices, edges):
    """Exact endpoint rows, including any render duplicates; no distance weld."""
    needed=np.unique(_interface_position_keys(edges.reshape(-1,3)))
    keys=_interface_position_keys(vertices)
    if not np.isin(needed,keys).all():raise ValueError('Partition interface endpoint missing from pristine LOW')
    result=np.flatnonzero(np.isin(keys,needed)).astype(np.int32);result.setflags(write=False)
    return result


def partition_interfaces_preserved(vertices, faces, required):
    """Every pinned whole edge must still have exactly ONE local incidence."""
    positions,inverse=np.unique(vertices,axis=0,return_inverse=True)
    keys=_interface_position_keys(positions);needed=_interface_position_keys(required.reshape(-1,3))
    found=np.searchsorted(keys,needed)
    if np.any(found>=len(keys)) or not np.array_equal(keys[found],needed):return False
    wanted=np.sort(found.reshape(-1,2),axis=1)
    canonical=inverse[faces];edges=np.sort(np.stack((canonical,np.roll(canonical,-1,axis=1)),axis=-1).reshape(-1,2),axis=1)
    edges,incidence=np.unique(edges,axis=0,return_counts=True)
    dtype=np.dtype([('a',edges.dtype),('b',edges.dtype)])
    actual=edges.view(dtype).reshape(-1);wanted=np.ascontiguousarray(wanted,dtype=edges.dtype).view(dtype).reshape(-1)
    index=np.searchsorted(actual,wanted)
    return bool(np.all(index<len(actual)) and np.array_equal(actual[index],wanted) and np.all(incidence[index]==1))


def components(vertices, faces, *, edge_barriers=None):
    """Join face IDs only across exactly-two opposite directed incidents.

    Position-equal render seams are canonicalized only for this census. Zero
    edges and nonmanifold edges are barriers; no mutable mesh may drop faces.
    """
    count=len(faces)
    if count>2_000_000:raise ValueError('Component census exceeds the face limit')
    if not count:return []
    _,inverse=np.unique(vertices,axis=0,return_inverse=True)
    canonical=inverse[faces].astype(np.int32,copy=False)
    directed=np.stack((canonical,np.roll(canonical,-1,axis=1)),axis=-1).reshape(-1,2)
    edge_faces=np.repeat(np.arange(count,dtype=np.int32),3)
    valid=directed[:,0]!=directed[:,1]
    if edge_barriers is not None:
        if np.shape(edge_barriers)!=(count,3):raise ValueError('Component edge-barrier shape differs from faces')
        valid&=~np.asarray(edge_barriers,dtype=bool).ravel()
    valid=np.flatnonzero(valid)
    keys=np.sort(directed[valid],axis=1)
    order=np.lexsort((keys[:,1],keys[:,0]));keys=keys[order];incidents=valid[order]
    starts=np.flatnonzero(np.r_[True,np.any(keys[1:]!=keys[:-1],axis=1)]) if len(keys) else np.empty(0,dtype=int)
    lengths=np.diff(np.r_[starts,len(keys)])
    paired=starts[lengths==2];left=incidents[paired];right=incidents[paired+1]
    opposite=np.all(directed[left]==directed[right,::-1],axis=1)&(edge_faces[left]!=edge_faces[right])
    parent=list(range(count));rank=bytearray(count)
    def find(index):
        while parent[index]!=index:
            parent[index]=parent[parent[index]];index=parent[index]
        return index
    for a,b in zip(edge_faces[left[opposite]],edge_faces[right[opposite]]):
        a=find(a);b=find(b)
        if a==b:continue
        if rank[a]<rank[b]:a,b=b,a
        parent[b]=a
        if rank[a]==rank[b]:rank[a]+=1
    roots=np.fromiter((find(i) for i in range(count)),dtype=np.int32,count=count)
    order=np.argsort(roots,kind='stable').astype(np.int32)
    groups=np.split(order,np.flatnonzero(np.diff(roots[order]))+1)
    if sum(map(len,groups))!=count:raise ValueError('Component census lost faces')
    return groups
