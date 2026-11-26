# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded geometry-only local holes with explicit native witness policies.

Exact POSITION identification is diagnostic only: no source rows are welded.
No HIGH geometry, UV or authored appearance is inferred for appended faces.
"""
import hashlib
from fractions import Fraction
import numpy as np

MAX_LOOP_EDGES=32
MAX_LOOPS=16
MAX_EXACT_PAIRS=8192
MAX_RAY_CANDIDATES=256


class LocalContactRejected(ValueError):
    """A contact-only refusal; never an allocation, deadline or native failure."""


class DCContactRejected(ValueError):
    """Only the unsigned-pair contact may request one bounded DC ROI witness."""


def dc_witness_roi(v,f,matrix,loop,resolution,check):
    """Exact native ABI in original F32 input axes, not inverse-world rounding."""
    import json
    check()
    if type(resolution)is not int or resolution not in (32,64,128):raise ValueError('Unsupported DC witness resolution')
    ring=v[np.asarray(loop['raw_pairs'])[:,0]].astype(np.float64)
    center=ring.mean(0);diameter=max(float(np.linalg.norm(a-b))for a in ring for b in ring)
    pitch=.4*diameter
    voxel=dict(local_voxel=pitch,resolution=resolution,source_geometry_sha256=geometry_hash(v,f),
               basis='Full original F32 source ring diameter times0.4; explicit bounded ROI, not whole-component padding')
    world,metric=_dc_world_voxel(voxel,matrix,np.asarray(loop['normal']))
    ray=np.linalg.solve(np.asarray(matrix,np.float64)[:3,:3],np.asarray(loop['normal']));ray/=np.linalg.norm(ray)
    if not np.isfinite(ray).all()or abs(np.linalg.norm(ray)-1)>128*np.finfo(float).eps:raise ValueError('DC witness unit ray is unresolved')
    request=dict(schema=1,scope='local_hole_witness_only',center=center.tolist(),extent=resolution*pitch,
                 protected_polygon=ring.tolist(),normal=ray.tolist(),guard_voxels=8)
    frame=dict(source_geometry_sha256=voxel['source_geometry_sha256'],resolution=resolution,matrix_world=np.asarray(matrix,np.float64).tolist(),request=request,
               world_voxel=world,ray_metric=metric,axes='Original full source input axes; normal is normalized inverse-linear world chart ray, not plane covector')
    frame['sha256']=hashlib.sha256(json.dumps(frame,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    voxel.update(frame_sha256=frame['sha256'],effective_center_f32=np.asarray(center,np.float32).tolist(),roi_extent=request['extent'])
    check();return request,voxel,frame


def witness_crop(v,f,matrix,loop,input_resolution,check,*,face_limit,remaining_bytes):
    """Uncapped +/-4d source crop in translated Blender world axes, witness only."""
    import math
    from itertools import combinations
    check();frame_bytes=64*len(v)+160*min(len(f),4096)+65536
    if remaining_bytes<frame_bytes:raise MemoryError('WTiVo witness crop frame working-byte budget exceeded')
    center=np.asarray(loop['center']);half=4*loop['diameter']
    world=np.asarray(v,np.float64)@matrix[:3,:3].T+matrix[:3,3]-center;check()
    if not np.isfinite(world).all():raise ValueError('WTiVo witness crop frame is nonfinite')
    out=[];owners=[]
    def intersect(a,b,axis,bound):
        if tuple(a)>tuple(b):a,b=b,a
        p=a+(bound-a[axis])/(b[axis]-a[axis])*(b-a);p[axis]=bound;return p
    def candidates():
        for start in range(0,len(f),4096):
            check();block=world[f[start:start+4096]]
            overlap=np.all(block.max(1)>=-half,axis=1)&np.all(block.min(1)<=half,axis=1)
            for row in np.flatnonzero(overlap):yield start+int(row),block[row]
    candidate_faces=sum(1 for _ in candidates());limit=min(7*candidate_faces,face_limit)
    if not candidate_faces:raise ValueError('WTiVo witness crop is empty')
    if limit<4 or remaining_bytes<1024*limit+frame_bytes:raise MemoryError('WTiVo witness crop working-byte budget exceeded')
    for face,triangle in candidates():
        check();poly=[p.copy() for p in triangle]
        for axis in range(3):
            for sign in (-1,1):
                if not poly:break
                nxt=[];bound=sign*half
                for a,b in zip(poly,poly[1:]+poly[:1]):
                    ia=sign*a[axis]<=half;ib=sign*b[axis]<=half
                    if ia:nxt.append(a)
                    if ia!=ib:nxt.append(intersect(a,b,axis,bound))
                poly=[]
                for p in nxt:
                    if not poly or not np.array_equal(p,poly[-1]):poly.append(p)
                if len(poly)>1 and np.array_equal(poly[0],poly[-1]):poly.pop()
                if len(poly)>9:raise ValueError('WTiVo witness crop polygon work bound exceeded')
        for j in range(1,len(poly)-1):
            triangle=np.stack((poly[0],poly[j],poly[j+1]))
            if np.any(np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])):
                if len(out)>=limit:raise ValueError('WTiVo witness crop triangle budget exceeded')
                out.append(triangle);owners.append(face)
    if not out:raise ValueError('WTiVo witness crop is empty')
    check();corners64=np.asarray(out);del out
    corners=corners64.astype(np.float32);cv,inverse=np.unique(corners.reshape(-1,3),axis=0,return_inverse=True);cf=inverse.reshape(-1,3).astype(np.int32);check()
    if not np.isfinite(cv).all() or cv[cf].tobytes()!=corners.tobytes():raise ValueError('WTiVo witness crop F32 oriented corners differ')
    topology=physical_topology(cv,cf,closed=False)
    if topology['indexed']['vertex_links']['bad_vertex_fans'] or topology['indexed']['indexed_duplicate_faces']['groups']:raise ValueError('WTiVo witness crop fan/duplicate admission failed')
    span=cv.max(0).astype(float)-cv.min(0).astype(float)
    if np.any(span<=1e-20):raise ValueError('WTiVo witness crop has a degenerate native axis')
    ring=world[np.asarray(loop['raw_pairs'])[:,0]].astype(np.float32).astype(float)
    if not all(any(np.array_equal(p,row) for row in cv) for p in ring):raise ValueError('WTiVo witness crop changed original hole vertices')
    edges=np.stack((cf,np.roll(cf,-1,axis=1)),axis=-1).reshape(-1,2);_,first,count=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True);boundary=cv[edges[first[count==1]]]
    rawpairs=world[np.asarray(loop['raw_pairs'])].astype(np.float32)
    if sum(any(np.array_equal(pair,row) for row in boundary) for pair in rawpairs)!=len(ring):raise ValueError('WTiVo witness crop changed original hole edges')
    cut=np.any(np.all(np.isclose(np.abs(boundary.astype(float)),half,rtol=0,atol=2*np.spacing(np.float32(half))),axis=1),axis=1)
    if int((~cut).sum())!=len(ring):raise ValueError('WTiVo witness crop has another unresolved source boundary')
    normalized=(ring-cv.min(0).astype(float))/span
    if len(ring)==3:
        perimeter=np.linalg.norm(np.roll(normalized,-1,axis=0)-normalized,axis=1).sum()
        r0=float(np.linalg.norm(np.cross(normalized[1]-normalized[0],normalized[2]-normalized[0]))/perimeter)
    else:
        # A bounded 3-variable Chebyshev-center LP, no dependency or native frame rotation.
        _,_,axes=np.linalg.svd(normalized-normalized.mean(0));chart=(normalized-normalized.mean(0))@axes[:2].T
        edge=np.roll(chart,-1,axis=0)-chart;sign=np.sign(sum(_cross2(chart[i],chart[(i+1)%len(chart)]) for i in range(len(chart))))
        if not sign or any(sign*_cross2(edge[i],edge[(i+1)%len(edge)])<=0 for i in range(len(edge))):raise ValueError('WTiVo normalized witness chart is not strictly convex')
        normals=sign*np.stack((-edge[:,1],edge[:,0]),axis=1)/np.linalg.norm(edge,axis=1)[:,None];rhs=(normals*chart).sum(1);system=np.c_[normals,-np.ones(len(chart))];r0=0.
        for triple in combinations(range(len(chart)),3):
            check()
            try:point=np.linalg.solve(system[list(triple)],rhs[list(triple)])[:2]
            except np.linalg.LinAlgError:continue
            if np.isfinite(point).all() and np.all(point>=chart.min(0)) and np.all(point<=chart.max(0)):r0=max(r0,float((normals@point-rhs).min()))
    if not np.isfinite(r0) or r0<=0:raise ValueError('WTiVo normalized witness inradius is invalid')
    resolution=min(input_resolution,math.floor(.85*(2+1/r0)))
    if resolution<=2:raise ValueError('WTiVo witness input resolution has a degenerate native inset')
    record=dict(center_world=center.tolist(),frame='Translated original Blender world axes; world=native_witness+center_world',half_width=half,
        source_geometry_sha256=geometry_hash(v,f),source_face_ids_sha256=hashlib.sha256(np.asarray(owners,np.int32).tobytes()).hexdigest(),
        contributing_source_faces=len(set(owners)),vertices=len(cv),triangles=len(cf),geometry_sha256=geometry_hash(cv,cf),
        block_AABB_candidate_faces=candidate_faces,estimated_peak_work_bytes=1024*limit+frame_bytes,
        r0=r0,input_resolution=resolution,selection='min(user input resolution,floor(.85*(2+1/r0))) on actual F32 crop bbox-normalized hole chart',
        F64_to_F32_error_max=float(np.linalg.norm(corners.astype(float)-corners64,axis=2).max()),source_hole_edges=len(ring),artificial_cut_edges=int(cut.sum()),caps=False,
        topology=topology,appearance='Witness source subset only; no HIGH, material or authored new-face identity')
    check();return cv,cf,record


def geometry_hash(v,f):
    return hashlib.sha256(b'POSITIONf32/TRIANGLEi32\0'+np.asarray([len(v),len(f)],'<u8').tobytes()+np.asarray(v,'<f4').tobytes()+np.asarray(f,'<i4').tobytes()).hexdigest()


def physical_topology(v,f,*,closed=True):
    from cumesh_simplify import indexed_topology
    from inspect_native_geometry import geometric_topology
    unique,inverse=np.unique(v,axis=0,return_inverse=True)
    indexed=indexed_topology(unique,inverse[f].astype(np.int32))
    geometric=geometric_topology(v,f)
    keys=('nonmanifold_edges','orientation_conflict_edges','degenerate_edges','numerical_local_zero_area_triangles')
    if any(geometric[k] for k in keys) or (closed and geometric['boundary_edges']):
        raise ValueError('WTiVo local holes require clean oriented manifold geometry')
    return dict(indexed=indexed,geometric=geometric,index_scope='Exact POSITION quotient for topology only; original render rows retained, never welded for export',
                self_intersections='Only appended-patch contacts checked; unrelated existing intersections NOT_EVALUATED',physical_or_visual_validity='NOT_QUALIFIED')


def prepare(v,f,matrix,check):
    """Admit every loop before a native call; closed components need no native."""
    check()
    from cumesh_simplify import MeshValidationError
    try:topology=physical_topology(v,f,closed=False)
    except MeshValidationError as error:
        if error.code!='ZERO_AREA_TRIANGLE':raise
        rejected=MeshValidationError(error.code)
        rejected.args=(f'{error}. Use Inspect Mesh; Conservative Repair can remove exact zero-area faces, not every topology defect.',)
        raise rejected from error
    unique,inverse=np.unique(v,axis=0,return_inverse=True);pf=inverse[f]
    directed=np.stack((pf,np.roll(pf,-1,axis=1)),axis=-1).reshape(-1,2)
    raw=np.stack((f,np.roll(f,-1,axis=1)),axis=-1).reshape(-1,2)
    _,first,counts=np.unique(np.sort(directed,axis=1),axis=0,return_index=True,return_counts=True)
    boundary=first[counts==1]
    world=np.asarray(v,np.float64)@matrix[:3,:3].T+matrix[:3,3]
    if not np.isfinite(world).all():raise ValueError('WTiVo local source world coordinates are nonfinite')
    diagonal=float(np.linalg.norm(np.ptp(world,axis=0)))
    record=dict(source_geometry_sha256=geometry_hash(v,f),source_vertices=len(v),source_faces=len(f),loops=[],before=topology)
    if not len(boundary):return record
    if not np.isfinite(diagonal) or diagonal<=0 or len(boundary)>MAX_LOOP_EDGES*MAX_LOOPS:raise ValueError('WTiVo local hole boundary/scale bound exceeded')
    outgoing={};incoming={}
    for row in boundary:
        a,b=map(int,directed[row])
        if a in outgoing or b in incoming:raise ValueError('WTiVo local hole boundary branches or touches')
        outgoing[a]=(b,int(row));incoming[b]=a
    if set(outgoing)!=set(incoming):raise ValueError('WTiVo local hole boundary is not a closed directed loop')
    unseen=set(outgoing)
    while unseen:
        check();cur=min(unseen);seed=cur;order=[];rows=[]
        while cur in unseen:
            unseen.remove(cur);order.append(cur);cur,row=outgoing[cur];rows.append(row)
            if len(order)>MAX_LOOP_EDGES:raise ValueError('WTiVo local hole edge bound exceeded')
        if cur!=seed or len(order)<3 or len(record['loops'])>=MAX_LOOPS:raise ValueError('Unsupported WTiVo local hole loop')
        pairs=raw[rows];ring=world[pairs[:,0]];center=ring.mean(0)
        diameter=float(np.linalg.norm(ring[:,None]-ring[None,:],axis=2).max())
        if not 0<diameter<=.05*diagonal:raise ValueError('WTiVo local hole exceeds5% component diagonal')
        _,singular,axes=np.linalg.svd(ring-center);normal=axes[-1]
        if singular[1]<=64*np.finfo(float).eps*diameter:raise ValueError('WTiVo local hole chart is degenerate')
        adjacent=world[f[np.asarray(rows)//3]];an=np.cross(adjacent[:,1]-adjacent[:,0],adjacent[:,2]-adjacent[:,0]);reference=an.sum(0)
        if np.linalg.norm(reference)==0:raise ValueError('WTiVo local hole orientation is ambiguous')
        if normal@reference<0:normal=-normal
        if np.any((an@normal)/np.linalg.norm(an,axis=1)<.5):raise ValueError('WTiVo local hole collar orientation is incompatible')
        if np.abs((ring-center)@normal).max()>.05*diameter:raise ValueError('WTiVo local hole is not sufficiently planar')
        u=ring[1]-ring[0];u-=normal*(u@normal);u/=np.linalg.norm(u);w=np.cross(normal,u)
        chart=np.stack(((ring-center)@u,(ring-center)@w),axis=1)
        area=sum(_cross2(chart[i],chart[(i+1)%len(chart)]) for i in range(len(chart)))
        sign=np.sign(area)
        if not sign or any(sign*_cross2(chart[(i+1)%len(chart)]-chart[i],chart[(i+2)%len(chart)]-chart[(i+1)%len(chart)])<=0 for i in range(len(chart))):raise ValueError('WTiVo local hole must be strictly convex')
        margin=lambda q:min(sign*_cross2(chart[(i+1)%len(chart)]-chart[i],q-chart[i])/np.linalg.norm(chart[(i+1)%len(chart)]-chart[i]) for i in range(len(chart)))
        radius=float(margin(np.zeros(2)))
        if not radius>0:raise ValueError('WTiVo local hole chart center is unresolved')
        record['loops'].append(dict(raw_pairs=pairs.tolist(),adjacent_faces=(np.asarray(rows)//3).tolist(),center=center.tolist(),normal=normal.tolist(),u=u.tolist(),w=w.tolist(),
            chart=chart.tolist(),sign=float(sign),inradius=radius,diameter=diameter,component_diagonal=diagonal))
    check();return record


def _cross2(a,b):return a[0]*b[1]-a[1]*b[0]
def _sub(a,b):return tuple(x-y for x,y in zip(a,b))
def _dot(a,b):return sum(x*y for x,y in zip(a,b))
def _cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def _exact(a):return tuple(Fraction(float(x)) for x in a)
def _area(t):return _cross(_sub(t[1],t[0]),_sub(t[2],t[0]))


def _ray(tri,p,d):
    e1=_sub(tri[1],tri[0]);e2=_sub(tri[2],tri[0]);h=_cross(d,e2);det=_dot(e1,h)
    if det==0:return None
    s=_sub(p,tri[0]);u=_dot(s,h)/det;q=_cross(s,e1);v=_dot(d,q)/det;t=_dot(e2,q)/det
    if u<0 or v<0 or u+v>1:return None
    return t,tuple(p[i]+t*d[i] for i in range(3))


def _ray_hits(triangles,origin,direction,center,normal,check,*,include_origin=False,normal_parity=1):
    # Conservative expanded AABB ray candidates; exact rational intersections
    # decide contacts, including coincident triangle-edge hits as one point.
    candidates=[]
    for start in range(0,len(triangles),4096):
        check();block=triangles[start:start+4096];lo=block.min(1);hi=block.max(1)
        eps=128*np.finfo(float).eps*np.maximum(1.,np.maximum(np.abs(lo),np.abs(hi)))
        lo-=eps;hi+=eps;near=np.full(len(block),0.);far=np.full(len(block),np.inf)
        for k in range(3):
            if direction[k]==0:
                far[(origin[k]<lo[:,k])|(origin[k]>hi[:,k])]=-1
            else:
                a=(lo[:,k]-origin[k])/direction[k];b=(hi[:,k]-origin[k])/direction[k]
                near=np.maximum(near,np.minimum(a,b));far=np.minimum(far,np.maximum(a,b))
        candidates.extend((start+np.flatnonzero(far>=near)).tolist())
        if len(candidates)>MAX_RAY_CANDIDATES:raise ValueError('WTiVo local contact candidate bound exceeded')
    p,d=_exact(origin),_exact(direction);hits={}
    for i in candidates:
        check();tri=tuple(map(_exact,triangles[i]));n=_area(tri)
        if _dot(n,d)==0 and _dot(n,_sub(p,tri[0]))==0:raise ValueError('WTiVo local contact is coplanar/ambiguous')
        hit=_ray(tri,p,d)
        if hit is None or hit[0]<0 or (hit[0]==0 and not include_origin):continue
        point=np.asarray(tuple(map(float,hit[1])));normal_vector=np.asarray(tuple(map(float,n)));cosine=float(normal_vector@normal/np.linalg.norm(normal_vector));normal_vector/=np.linalg.norm(normal_vector)
        # Only unsigned DC contact normals compensate a reflected linear frame.
        if normal_parity<0:cosine=-cosine;normal_vector=-normal_vector
        hits.setdefault(hit[1],[]).append(dict(face=i,distance=float(hit[0]),point_world=point.tolist(),height=float((point-center)@normal),normal_dot=cosine,normal_world=normal_vector.tolist()))
    return sorted(hits.values(),key=lambda rows:rows[0]['distance'])


def _front_hit(triangles,origin,direction,center,normal,diameter,check):
    try:ordered=_ray_hits(triangles,origin,direction,center,normal,check)
    except ValueError as error:
        if str(error)!='WTiVo local contact is coplanar/ambiguous':raise
        raise LocalContactRejected(str(error)) from error
    local=[rows for rows in ordered if abs(rows[0]['height'])<=diameter/4]
    if len(local)!=1 or not ordered or local[0] is not ordered[0] or any(x['normal_dot']<.5 for x in local[0]):raise LocalContactRejected('WTiVo local contact missing, obstructed, opposed or nonunique')
    return local[0][0]


def dc_voxel(v,f,resolution,stage):
    """Cross-check the actual isolated native part, not a caller-chosen scale."""
    if type(resolution)is not int or resolution not in (32,64,128) or v.dtype!=np.float32 or not np.isfinite(v).all():raise ValueError('DC local voxel source/resolution is invalid')
    if type(stage.get('resolution'))is not int or type(stage.get('band'))not in (int,float) or type(stage.get('source_vertices'))is not int or type(stage.get('source_triangles'))is not int:raise ValueError('DC local native stage metadata types differ')
    if stage.get('status')!='geometry_proposal' or stage.get('resolution')!=resolution or stage.get('band')!=1 or stage.get('policy')!='retain_all_unsigned_shells' or stage.get('source_vertices')!=len(v) or stage.get('source_triangles')!=len(f):raise ValueError('DC local native stage identity differs')
    extent=float(np.max(np.max(v,axis=0)-np.min(v,axis=0)));expected=extent*1.1/resolution
    actual=stage.get('native',{}).get('iso')
    if type(actual)not in (float,int) or not np.isfinite(actual) or actual<=0 or not np.isfinite(expected) or expected<=0 or float(actual).hex()!=expected.hex():raise ValueError('DC local voxel scale differs from the exact native padded source extent')
    return dict(local_voxel=float(actual),padded_local_scale=extent*1.1,resolution=resolution,source_geometry_sha256=geometry_hash(v,f),basis='Exact F32 component max side, native Python padded-scale/resolution and reported band1 iso cross-check')


def _dc_world_voxel(voxel,matrix,normal):
    a=np.asarray(matrix,np.float64)[:3,:3]
    if a.shape!=(3,3) or not np.isfinite(a).all():raise ValueError('DC local ray transform is nonfinite or malformed')
    try:condition=float(np.linalg.cond(a));parity,_=np.linalg.slogdet(a)
    except np.linalg.LinAlgError as error:raise ValueError('DC local ray transform is unresolved') from error
    if parity not in (-1.,1.):raise ValueError('DC local ray transform is singular')
    if not np.isfinite(a).all() or not np.isfinite(condition) or condition>1e6:raise ValueError('DC local ray transform is not numerically provable')
    ray=np.linalg.solve(a,normal);length=float(np.linalg.norm(ray));residual=float(np.linalg.norm(a@ray-normal))
    bound=128*np.finfo(float).eps*max(1.,float(np.linalg.norm(a))*length)
    world=voxel['local_voxel']/length
    if not np.isfinite(ray).all() or not np.isfinite(length) or length<=0 or residual>bound or not np.isfinite(world) or world<=0:raise ValueError('DC local world ray metric is unresolved')
    return world,dict(world_ray_voxel=world,local_ray_length_per_world_unit=length,transform_condition=condition,residual=residual,unsigned_normal_parity=int(parity),
        scope='Local isotropic voxel length converted along the actual world chart-normal ray; not an isotropic world offset or signed surface identity')


def _paired_midpoint(triangles,origin,center,normal,voxel,diameter,check,*,normal_parity=1):
    try:ordered=_ray_hits(triangles,origin,-normal,center,normal,check,include_origin=True,normal_parity=normal_parity)
    except ValueError as error:
        if str(error)!='WTiVo local contact is coplanar/ambiguous':raise
        raise DCContactRejected(str(error)) from error
    contacts=[rows for rows in ordered if abs(rows[0]['height'])<=2*voxel]
    if len(contacts)!=2 or len(ordered)<2 or contacts[0]is not ordered[0] or contacts[1]is not ordered[1]:raise DCContactRejected('DC local unsigned pair is missing, obstructed or ambiguous')
    front,back=contacts;hf=front[0]['height'];hb=back[0]['height']
    if not (.5*voxel<=hf<=1.5*voxel and -1.5*voxel<=hb<=-.5*voxel):raise DCContactRejected('DC local contacts do not bracket the plane in the verified voxel band')
    for rows,sign in ((front,1),(back,-1)):
        if any(not np.isfinite(x['normal_dot']) or sign*x['normal_dot']<.5 or not np.isfinite(x['normal_world']).all() for x in rows):raise DCContactRejected('DC local contact normal is incompatible')
    if any(np.dot(a['normal_world'],b['normal_world'])>-.5 for a in front for b in back):raise DCContactRejected('DC local pair normals do not oppose')
    point=(np.asarray(front[0]['point_world'])+np.asarray(back[0]['point_world']))*.5;height=float((point-center)@normal)
    if abs(height)>min(.25*voxel,diameter/4):raise DCContactRejected('DC local synthetic midpoint is too far from the source plane')
    return dict(point_world=point.tolist(),front=front[0],back=back[0],front_face_ids=[x['face']for x in front],back_face_ids=[x['face']for x in back],
        front_height=hf,back_height=hb,midpoint_plane_height=height,identity='Approximate synthetic midpoint from two actual native contacts; no authored source/HIGH identity')


def _contacts(a,b,allow_shared):
    """Exact new/old or new/new triangle intersection, only shared seams allowed."""
    common=set(a)&set(b) if allow_shared else set()
    if len(common)==3:raise ValueError('WTiVo local patch duplicates a face')
    def permitted(p):
        if p in common:return True
        if len(common)==2:
            x,y=tuple(common);xy=_sub(y,x);xp=_sub(p,x)
            return _cross(xy,xp)==(0,0,0) and 0<=_dot(xp,xy)<=_dot(xy,xy)
        return False
    def coplanar_segment(x,y,t):
        axis=max(range(3),key=lambda k:abs(_area(t)[k]));project=lambda p:tuple(p[k] for k in range(3) if k!=axis)
        xx,yy=project(x),project(y);tt=tuple(map(project,t));orientation=_cross2(_sub(tt[1],tt[0]),_sub(tt[2],tt[0]))
        for point,pp in ((x,xx),(y,yy)):
            sides=[_cross2(_sub(tt[(i+1)%3],tt[i]),_sub(pp,tt[i]))*orientation for i in range(3)]
            if min(sides)>=0 and not permitted(point):raise ValueError('WTiVo local patch overlaps a surface')
        for i in range(3):
            z,w=t[i],t[(i+1)%3];zz,ww=tt[i],tt[(i+1)%3];d=_sub(yy,xx);e=_sub(ww,zz);den=_cross2(d,e)
            if den:
                s=_cross2(_sub(zz,xx),e)/den;r=_cross2(_sub(zz,xx),d)/den
                if 0<=s<=1 and 0<=r<=1 and not permitted(tuple(x[k]+s*(y[k]-x[k]) for k in range(3))):raise ValueError('WTiVo local patch crosses a surface')
            elif _cross2(d,_sub(zz,xx))==0:
                for point,pp in ((x,xx),(y,yy),(z,zz),(w,ww)):
                    if all(min(xx[k],yy[k])<=pp[k]<=max(xx[k],yy[k]) and min(zz[k],ww[k])<=pp[k]<=max(zz[k],ww[k]) for k in range(2)) and not permitted(point):raise ValueError('WTiVo local patch has unshared collinear contact')
    for first,second in ((a,b),(b,a)):
        n=_area(second)
        for k in range(3):
            x,y=first[k],first[(k+1)%3];d=_sub(y,x)
            if _dot(n,d)==0 and _dot(n,_sub(x,second[0]))==0:coplanar_segment(x,y,second);continue
            hit=_ray(second,x,d)
            if hit is not None and 0<=hit[0]<=1 and not permitted(hit[1]):raise ValueError('WTiVo local patch intersects away from its sewn boundary')


def finish(v,f,matrix,plan,native_v,native_f,check,*,context=None,owned_faces=None,witness_policy='oriented_front',native_voxel=None,refine=None):
    """Sew center witnesses to exact source boundary; append only, never fallback."""
    check()
    if witness_policy not in ('oriented_front','dc_unsigned_midpoint'):raise ValueError('Unsupported local hole witness policy')
    if geometry_hash(v,f)!=plan['source_geometry_sha256']:raise ValueError('WTiVo local source changed after admission')
    if not plan['loops']:return v.copy(),f.copy(),dict(plan,mode='retained_closed',new_vertex_ids=[],new_face_ids=[])
    native_world=np.asarray(native_v,np.float64)@matrix[:3,:3].T+matrix[:3,3];native_tri=native_world[native_f]
    inverse=np.linalg.inv(matrix);new=[];added=[];witnesses=[]
    for loop_index,loop in enumerate(plan['loops']):
        check();center=np.asarray(loop['center']);normal=np.asarray(loop['normal']);u=np.asarray(loop['u']);w=np.asarray(loop['w']);diameter=loop['diameter'];radius=loop['inradius'];chart=np.asarray(loop['chart']);sign=loop['sign'];controls=[]
        dc=witness_policy=='dc_unsigned_midpoint'
        if dc:
            if native_voxel is None or native_voxel['source_geometry_sha256']!=plan['source_geometry_sha256']:raise ValueError('DC local voxel belongs to a different source part')
            voxel,metric=_dc_world_voxel(native_voxel,matrix,normal)
        def contacts(triangles):
            controls=[]
            for x,y in ((0.,0.),(.2*radius,0.),(-.2*radius,0.),(0.,.2*radius),(0.,-.2*radius)):
                if dc:
                    hit=_paired_midpoint(triangles,center+x*u+y*w+normal*(2*voxel),center,normal,voxel,diameter,check,normal_parity=metric['unsigned_normal_parity'])
                    if controls and (abs(hit['front_height']-controls[0]['front_height'])>.5*voxel or abs(hit['back_height']-controls[0]['back_height'])>.5*voxel):raise DCContactRejected('DC local contacts do not form one continuous local band')
                else:hit=_front_hit(triangles,center+x*u+y*w+normal*diameter,-normal,center,normal,diameter,check)
                point=np.asarray(hit['point_world']);q=np.array([(point-center)@u,(point-center)@w])
                if min(sign*_cross2(chart[(i+1)%len(chart)]-chart[i],q-chart[i]) for i in range(len(chart)))<=0:raise ValueError('WTiVo local contact lies outside footprint')
                controls.append(hit)
            return controls
        refinement=None
        try:controls=contacts(native_tri)
        except (LocalContactRejected,DCContactRejected) as error:
            if refine is None or (dc and not isinstance(error,DCContactRejected)) or (not dc and not isinstance(error,LocalContactRejected)):raise
            refined_tri,refinement=refine(loop_index,loop,error)
            try:
                if dc:voxel,metric=_dc_world_voxel(refinement['native_voxel'],matrix,normal)
                controls=contacts(refined_tri)
                refinement['contact_status']='five_original_controls_passed'
            except BaseException as failure:
                refinement.update(contact_status='failed',contact_error=f'{type(failure).__name__}: {failure}');raise
            finally:del refined_tri
        point=np.asarray(controls[0]['point_world']);local=(inverse[:3,:3]@point+inverse[:3,3]).astype(np.float32)
        if not np.isfinite(local).all() or any(np.array_equal(local,p) for p in v) or any(np.array_equal(local,p) for p in new):raise ValueError('WTiVo local contact collapses onto existing geometry')
        stored=local.astype(np.float64)@matrix[:3,:3].T+matrix[:3,3]
        q=np.array([(stored-center)@u,(stored-center)@w]);rounding=float(np.linalg.norm(stored-point))
        if rounding>diameter*.01 or abs((stored-center)@normal)>diameter/4 or min(sign*_cross2(chart[(i+1)%len(chart)]-chart[i],q-chart[i]) for i in range(len(chart)))<=0:raise ValueError('WTiVo local stored contact exceeds its chart/rounding bound')
        idx=len(v)+len(new);new.append(local)
        for a,b in loop['raw_pairs']:added.append((b,a,idx))
        if dc:
            if abs((stored-center)@normal)>.25*voxel:raise ValueError('DC local stored midpoint exceeds its voxel plane bound')
            witnesses.append(dict(policy=witness_policy,controls=controls,new_vertex_id=idx,native_voxel=refinement['native_voxel']if refinement is not None else native_voxel,world_voxel_metric=metric,
                synthetic_point_world=point.tolist(),midpoint_to_stored_world_distance=rounding,identity='New approximate geometry, not either native envelope or authored source face',
                **({'refinement':refinement}if refinement is not None else {})))
        else:witnesses.append(dict(controls=controls,new_vertex_id=idx,native_contact_face=controls[0]['face'],native_contact_world=point.tolist(),native_hit_to_stored_world_distance=rounding,
                                  **({'refinement':refinement} if refinement is not None else {})))
    out_v=np.concatenate((v,np.asarray(new,np.float32)));out_f=np.concatenate((f,np.asarray(added,np.int32)))
    if out_v[:len(v)].tobytes()!=v.tobytes() or out_f[:len(f)].tobytes()!=f.tobytes():raise ValueError('WTiVo local patch modified original rows/triangles')
    world=np.asarray(out_v,np.float64)@matrix[:3,:3].T+matrix[:3,3];new_tri=world[np.asarray(added,np.int32)]
    old_tri=world[f] if context is None else context;owned=set(range(len(f))) if owned_faces is None else set(owned_faces);pairs=0
    adjacent=[face for loop in plan['loops'] for face in loop['adjacent_faces']]
    for i,tri in enumerate(new_tri):
        exact=tuple(map(_exact,tri));normal=_area(exact);old=tuple(map(_exact,world[f[adjacent[i]]]))
        if normal==(0,0,0) or _dot(normal,_area(old))<=0:raise ValueError('WTiVo local patch is degenerate or opposes its collar')
        for start in range(0,len(old_tri),4096):
            check();block=old_tri[start:start+4096];ids=np.flatnonzero(np.all(block.max(1)>=tri.min(0),axis=1)&np.all(block.min(1)<=tri.max(0),axis=1));pairs+=len(ids)
            if pairs>MAX_EXACT_PAIRS:raise ValueError('WTiVo local intersection work bound exceeded')
            for j in ids:_contacts(exact,tuple(map(_exact,block[j])),start+int(j) in owned)
        for j in range(i):
            pairs+=1
            if pairs>MAX_EXACT_PAIRS:raise ValueError('WTiVo local intersection work bound exceeded')
            _contacts(exact,tuple(map(_exact,new_tri[j])),True)
    topology=physical_topology(out_v,out_f);check()
    proof=dict(plan,mode='local_holes',new_vertex_ids=list(range(len(v),len(out_v))),new_face_ids=list(range(len(f),len(out_f))),
        new_faces=out_f[len(f):].tolist(),native_geometry_sha256=geometry_hash(native_v,native_f),output_geometry_sha256=geometry_hash(out_v,out_f),original_rows_triangles_byteexact=True,witnesses=witnesses,
        intersection_pairs=pairs,after=topology,new_faces_identity_scope='Component geometry hash + appended pre-serialization triangle indices; new geometry only, no HIGH/UV/material identity')
    if any('refinement' in w for w in witnesses):
        initial=proof.pop('native_geometry_sha256');proof['initial_native_geometry_sha256']=initial
        proof['effective_witnesses']=[dict(loop_index=i,basis='bounded_dc_full_source_roi'if witness_policy=='dc_unsigned_midpoint'else'refined_world_axis_crop',geometry_sha256=w['refinement']['native_geometry_sha256'],
            input_sha256=w['refinement']['input_sha256'],raw_PLY_sha256=w['refinement']['output_sha256'],
            **({'roi_sha256':w['refinement']['roi_sha256'],'frame_sha256':w['refinement']['frame']['sha256']}if witness_policy=='dc_unsigned_midpoint'else{'world_translation':w['refinement']['source_crop']['center_world']}),input_resolution=w['refinement']['effective_input_resolution'],
            final_resolution=w['refinement']['effective_final_resolution']) if 'refinement' in w else
            dict(loop_index=i,basis='initial_component_local_witness',geometry_sha256=initial) for i,w in enumerate(witnesses)]
    return out_v,out_f,proof
