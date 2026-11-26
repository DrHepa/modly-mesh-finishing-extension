# SPDX-FileCopyrightText: 2001-2002 NaN Holding BV. All rights reserved.
# SPDX-FileCopyrightText: 2023 Blender Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fail-closed invalid-frame repair, isolated Mikk limit and normal conversion.

Raster expressions below derive from Blender fbe6228777e7 bake.cc/zbuf.cc.
Only affected triangles are scanned; native masks remain coverage authority.
"""
import math
import os
import struct
import numpy as np

_EPS=2.**-23
_F=struct.Struct('<f')

def _require(ok,message):
    if not ok:raise ValueError('Tangent plan: '+message)


class TangentProjectionZero(ValueError):
    """Finite nonzero native T projected to exactly zero, not a general refusal."""
    def __init__(self, loops):
        self.loops=np.asarray(loops,np.int64).copy()
        super().__init__('Tangent plan: invalid or zero orthogonal tangent projection')

def _f(value):
    try: result = _F.unpack(_F.pack(value))[0]
    except (OverflowError, struct.error): raise ValueError('Tangent raster: unsupported float32 overflow') from None
    if not math.isfinite(result): raise ValueError('Tangent plan: unsupported nonfinite arithmetic')
    return result


def _add(a,b): return _f(a+b)
def _sub(a,b): return _f(a-b)
def _mul(a,b): return _f(a*b)
def _div(a,b): return _f(a/b)

# Same pinned binary32 paint-origin arithmetic, invariant for every UV corner.
_PAINT_BIAS=(_add(.5,_f(.001)),_add(.5,_f(.002)))


def _edge(vertices, v1, v2, resolution):
    lo,hi=(v1,v2) if vertices[v1][1]<vertices[v2][1] else (v2,v1)
    minv,maxv=vertices[lo],vertices[hi]
    my0=math.ceil(minv[1]);my2=math.floor(maxv[1])
    if my2<0 or my0>=resolution:return None
    my2=min(my2,resolution-1);my0=max(my0,0)
    if my0>my2:return None
    delta=_sub(maxv[1],minv[1])
    if delta>_EPS:
        dx=_div(_sub(minv[0],maxv[0]),delta)
        xs=_add(_mul(dx,_sub(minv[1],_f(my2))),minv[0])
    else:dx=0.;xs=min(minv[0],maxv[0])
    return lo,hi,my0,my2,dx,xs


def _scan(vertices, span1, span2, resolution, check):
    """Literal pointer-chain span selection; IDs, NOT equal vertex values."""
    try:
        minp1=maxp1=minp2=maxp2=None
        miny1=miny2=resolution+1;maxy1=maxy2=-1
        for v1,v2 in ((0,1),(1,2),(2,0)):
            check()
            edge=_edge(vertices,v1,v2,resolution)
            if edge is None:continue
            lo,hi,my0,my2,dx,xs=edge
            minv,maxv=vertices[lo],vertices[hi]
            first=maxp1 is None or hi==minp1 or lo==maxp1
            if first:
                if minp1 is None or vertices[minp1][1]>minv[1]:minp1=lo
                if maxp1 is None or vertices[maxp1][1]<maxv[1]:maxp1=hi
                miny1=min(my0,miny1);maxy1=max(my2,maxy1)
            else:
                if minp2 is None or vertices[minp2][1]>minv[1]:minp2=lo
                if maxp2 is None or vertices[maxp2][1]<maxv[1]:maxp2=hi
                miny2=min(my0,miny2);maxy2=max(my2,maxy2)
            for y in range(my2,my0-1,-1):
                check()
                if first:span1[y]=xs
                else:span2[y]=xs
                xs=_add(xs,dx)
        if minp2 is None or maxp2 is None:return
        my0=max(miny1,miny2);my2=min(maxy1,maxy2)
        if my2<my0:return
        a,b,c=vertices
        x1=_sub(a[0],b[0]);x2=_sub(b[0],c[0]);y1=_sub(a[1],b[1]);y2=_sub(b[1],c[1])
        x0=_sub(_mul(y1,0.),_mul(1.,y2));y0=_sub(_mul(1.,x2),_mul(x1,0.))
        z0=_sub(_mul(x1,y2),_mul(y1,x2))
        if z0==0.:return
        xx=_add(_div(_add(_mul(x0,a[0]),_mul(y0,a[1])),z0),1.)
        uxd=_f(-float(x0)/float(z0));uyd=_f(-float(y0)/float(z0))
        uy0=_f(float(my2)*uyd+float(xx))
        x0=_sub(_mul(y1,1.),_mul(-1.,y2));y0=_sub(_mul(-1.,x2),_mul(x1,1.))
        xx=_div(_add(_mul(x0,a[0]),_mul(y0,a[1])),z0)
        vxd=_f(-float(x0)/float(z0));vyd=_f(-float(y0)/float(z0))
        vy0=_f(float(my2)*vyd+float(xx))
        for i,y in enumerate(range(my2,my0-1,-1)):
            check()
            left=math.floor(min(span1[y],span2[y]));right=math.floor(max(span1[y],span2[y]))
            _require(-2147483648<=left<=2147483646 and -2147483648<=right<=2147483647,
                     'unsafe span C integer conversion')
            left=max(left+1,0);right=min(right,resolution-1)
            u=_f((float(left)*uxd+uy0)-_mul(_f(i),uyd))
            v=_f((float(left)*vxd+vy0)-_mul(_f(i),vyd))
            for j,x in enumerate(range(left,right+1)):
                check()
                yield y*resolution+x,_add(u,_mul(_f(j),uxd)),_add(v,_mul(_f(j),vxd))
    finally:
        # Clear array arguments even when a raw traceback remains alive.
        span1=span2=vertices=None



def analytic_tangent(positions, uv, normal):
    """Positive-angle singleton Mikk limit; handedness remains native, not derived."""
    p=np.asarray(positions,np.float64);uv=np.asarray(uv,np.float64);n=np.asarray(normal,np.float64)
    _require(np.isfinite(p).all() and np.isfinite(uv).all() and np.isfinite(n).all(),'nonfinite analytic inputs')
    nn=float(n@n);_require(nn>0,'zero normal')
    derivatives=np.linalg.solve(uv[1:]-uv[0],p[1:]-p[0])
    t=derivatives[0]-n*float(n@derivatives[0])/nn
    length=float(np.linalg.norm(t));_require(math.isfinite(length) and length>0,'zero projected dPdu')
    return (t/length).astype(np.float32)


def plan_tangents(raw):
    """Repair invalid N/T before bake; preserve every already-valid vector and sign."""
    planned={k:np.asarray(raw[k]) for k in raw.keys()};n=planned['n'];t=planned['t'];sign=planned['sign']
    _require(np.isfinite(n).all() and np.isfinite(t).all() and np.isfinite(sign).all(),'nonfinite native frame')
    nl=np.linalg.norm(n.astype(np.float64),axis=1)
    _require(np.all(nl>0) and np.isin(sign,(-1,1)).all(),'invalid native normal/sign')
    loops=planned['triangle_loops'];lv=planned['loop_vertices']
    _require(loops.shape==planned['faces'].shape and loops.size==len(t) and
             np.array_equal(np.sort(loops.ravel()),np.arange(len(t))) and
             np.array_equal(lv[loops],planned['faces']),'triangle/loop identity differs')
    planned['n']=n.copy();planned['t']=t.copy()
    bad_n=np.abs(nl-1)>1e-4;planned['n'][bad_n]=(n[bad_n].astype(np.float64)/nl[bad_n,None]).astype(np.float32)
    pn=planned['n'].astype(np.float64);pt=t.astype(np.float64);tl=np.linalg.norm(pt,axis=1)
    bad_t=(tl>0)&((np.abs(tl-1)>1e-4)|(np.abs(np.einsum('ij,ij->i',pn,pt))>1e-4))
    projected=pt[bad_t]-pn[bad_t]*(np.einsum('ij,ij->i',pn[bad_t],pt[bad_t])/np.einsum('ij,ij->i',pn[bad_t],pn[bad_t]))[:,None]
    length=np.linalg.norm(projected,axis=1)
    _require(np.isfinite(projected).all(),'invalid or zero orthogonal tangent projection')
    collapsed=np.all(projected==0,axis=1)
    if collapsed.any():raise TangentProjectionZero(np.flatnonzero(bad_t)[collapsed])
    _require(np.all(length>0),'invalid or zero orthogonal tangent projection')
    planned['t'][bad_t]=(projected/length[:,None]).astype(np.float32)
    zero=np.flatnonzero(tl==0)
    if len(zero):
        # Value-key equality alone is not a Mikk fan. Only compatible reversed
        # halfedges connect corners; a disconnected face must not contribute.
        keys=np.column_stack((planned['vertices'][lv],n,planned['uv'])).astype(np.float32)
        _,inverse,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
        order=np.argsort(inverse);starts=np.r_[0,np.cumsum(counts)]
        face_for_loop=np.empty(len(t),np.int32);face_for_loop[loops.ravel()]=np.repeat(np.arange(len(loops)),3)
        prev=np.empty(len(t),np.int32);following=np.empty_like(prev)
        prev[loops.ravel()]=np.roll(loops,1,axis=1).ravel();following[loops.ravel()]=np.roll(loops,-1,axis=1).ravel()
        for loop in zero:
            key=inverse[loop];members=order[starts[key]:starts[key+1]]
            compatible=(members!=loop)&(sign[members]==sign[loop])&(
                (inverse[following[members]]==inverse[prev[loop]])|(inverse[prev[members]]==inverse[following[loop]]))
            _require(not compatible.any(),'zero tangent has a connected multi-contribution Mikk fan')
            face=int(face_for_loop[loop]);corners=loops[face]
            planned['t'][loop]=analytic_tangent(planned['vertices'][planned['faces'][face]],planned['uv'][corners],planned['n'][loop])
    _require(np.all(np.linalg.norm(planned['t'].astype(np.float64),axis=1)>0),'planned tangent is zero')
    changed=np.flatnonzero(np.any(planned['n']!=n,axis=1)|np.any(planned['t']!=t,axis=1))
    return planned,changed


def singleton_cyclic_choice(raw, error, charts):
    """Choose ONE positive cyclic UV assignment by worst relative projected dPdu."""
    _require(type(error) is TangentProjectionZero,'cyclic choice requires the typed zero projection')
    loops=raw['triangle_loops'];charts=np.asarray(charts)
    _require(charts.shape==(len(loops),) and np.issubdtype(charts.dtype,np.integer),'invalid cyclic chart census')
    _require(len(error.loops)>0 and np.isin(error.loops,loops).all(),'invalid zero-projection loop identity')
    affected=np.flatnonzero(np.isin(loops,error.loops).any(axis=1))
    _require(len(affected)==1,'cyclic choice requires exactly one affected face')
    face=int(affected[0]);corners=loops[face]
    _require(np.count_nonzero(charts==charts[face])==1,'cyclic choice requires an exact singleton chart')
    positions=raw['vertices'][raw['faces'][face]].astype(np.float64)
    uv=raw['uv'][corners];normal=raw['n'][corners].copy()
    lengths=np.linalg.norm(normal.astype(np.float64),axis=1)
    _require(np.isfinite(lengths).all() and np.all(lengths>0),'invalid cyclic normals')
    bad=np.abs(lengths-1)>1e-4
    normal[bad]=(normal[bad].astype(np.float64)/lengths[bad,None]).astype(np.float32)
    n=normal.astype(np.float64);nn=np.einsum('ij,ij->i',n,n);best=None
    for permutation in ((1,2,0),(2,0,1)):
        candidate=uv[list(permutation)].astype(np.float64)
        try:du=np.linalg.solve(candidate[1:]-candidate[0],positions[1:]-positions[0])[0]
        except np.linalg.LinAlgError:continue
        magnitude=float(du@du)
        if not np.isfinite(du).all() or not np.isfinite(magnitude) or magnitude<=0:continue
        projected=du-n*(n@du/nn)[:,None]
        score=float(np.min(np.einsum('ij,ij->i',projected,projected)/magnitude))
        if np.isfinite(score) and score>0 and (best is None or score>best['minimum_relative_projection_squared']):
            best=dict(face=face,chart=int(charts[face]),permutation=list(permutation),minimum_relative_projection_squared=score)
    _require(best is not None,'no positive cyclic projected derivative')
    return best


def frame_repair_metrics(raw,planned,changed):
    n=raw['n'].astype(np.float64);t=raw['t'].astype(np.float64);pn=planned['n'].astype(np.float64)
    nl=np.linalg.norm(n,axis=1);tl=np.linalg.norm(t,axis=1);nonzero=tl>0
    nc=np.any(raw['n'].view(np.uint32)!=planned['n'].view(np.uint32),axis=1)
    tc=np.any(raw['t'].view(np.uint32)!=planned['t'].view(np.uint32),axis=1)
    return dict(raw_zero_tangents=int((~nonzero).sum()),raw_nonunit_normals=int((np.abs(nl-1)>1e-4).sum()),
        raw_nonzero_tangent_failures=int((nonzero&((np.abs(tl-1)>1e-4)|(np.abs(np.einsum('ij,ij->i',pn,t))>1e-4))).sum()),
        repaired_normal_loops=int(nc.sum()),repaired_tangent_loops=int(tc.sum()),changed_loops=len(changed),
        raw_normal_length_min=float(nl.min()),raw_normal_length_max=float(nl.max()),
        max_normal_repair_direction_degrees=float(_angles(n[nc],pn[nc]).max(initial=0)),
        max_nonzero_tangent_repair_degrees=float(_angles(t[tc&nonzero],planned['t'][tc&nonzero]).max(initial=0)),
        policy='Invalid-to-valid repair angles are observations, not the 0.01-degree valid-plan stability gate')


def triangle_samples(uv,resolution):
    """Pinned binary32 scan conversion, restricted to one affected UV triangle."""
    _require(type(resolution) is int and 1<=resolution<=8192,'unsupported resolution')
    uv=np.asarray(uv,np.float32);_require(uv.shape==(3,2) and np.isfinite(uv).all(),'invalid UV triangle')
    vertices=tuple(tuple(_sub(_mul(float(v),float(resolution)),_PAINT_BIAS[k])
                         for k,v in enumerate(point)) for point in uv)
    span1=np.zeros(resolution,np.float32);span2=np.zeros(resolution,np.float32)
    for index,u,v in _scan(vertices,span1,span2,resolution,lambda:None):
        yield index,np.array([u,v,_sub(_sub(1.,u),v)],np.float32)


def _frame(snapshot,corners,weights,raw_face=None):
    # bake.cc 852-878: interpolate UNNORMALIZED N/T, then B=s*cross(N,T).
    def interp(values):
        a,b,c=np.asarray(values[corners],np.float32)
        u,v,w=weights
        return np.add(np.add(np.multiply(a,u),np.multiply(b,v)),np.multiply(c,w))
    n=interp(snapshot['n'])
    if raw_face is not None and 'smooth' in snapshot and not snapshot['smooth'][raw_face]:n=snapshot['face_n'][raw_face]
    t=interp(snapshot['t']);sign=-1. if float(interp(snapshot['sign']))<0 else 1.
    return np.column_stack((t,sign*np.cross(n,t),n)).astype(np.float64)


def normal_conversion_samples(raw_snapshots,planned_snapshots,repair_loops,occupancy,resolution):
    """Preflight actual occupied affected pixels BEFORE PBR; no invented coverage."""
    occupied=np.asarray(occupancy,bool).reshape(-1)
    _require(len(occupied)==resolution*resolution,'occupancy dimensions differ')
    indices=[];parts=[];faces=[];weights=[];old=[];new=[];world=[];seen=set()
    for part,(raw,planned,changed) in enumerate(zip(raw_snapshots,planned_snapshots,repair_loops)):
        affected=np.flatnonzero(np.isin(raw['triangle_loops'],changed).any(axis=1)) if len(changed) else []
        world_from_local=np.linalg.inv(np.asarray(raw['matrix'],np.float64)[:3,:3].T)
        _require(np.isfinite(world_from_local).all(),'invalid world normal transform')
        for face in affected:
            corners=raw['triangle_loops'][face]
            for index,bary in triangle_samples(raw['uv'][corners],resolution):
                if not occupied[index]:continue
                _require(index not in seen,'ambiguous affected pixel owner')
                a=_frame(raw,corners,bary,raw_face=face);b=_frame(planned,corners,bary)
                _require(np.isfinite(a).all() and np.isfinite(b).all(),'nonfinite pixel frame')
                # Exact invertibility is required, never a pseudoinverse/fill.
                _require(np.isfinite(np.linalg.solve(a,np.eye(3))).all() and np.isfinite(np.linalg.solve(b,np.eye(3))).all(),'singular pixel frame')
                seen.add(index);indices.append(index);parts.append(part);faces.append(int(face));weights.append(bary);old.append(a);new.append(b);world.append(world_from_local)
    return dict(indices=np.asarray(indices,np.int64),parts=np.asarray(parts,np.int32),faces=np.asarray(faces,np.int32),
                weights=np.asarray(weights,np.float32).reshape(-1,3),old=np.asarray(old,np.float64).reshape(-1,3,3),new=np.asarray(new,np.float64).reshape(-1,3,3),
                world_from_local=np.asarray(world,np.float64).reshape(-1,3,3))


def convert_normal_samples(data,samples):
    """Change coordinates, not world normals; primary/cage selection is already done."""
    indices=samples['indices']
    if not len(indices):return dict(valid=True,converted_texels=0,max_condition_number=0.,minimum_singular_value=None,max_normalized_world_roundtrip_residual=0.,world_roundtrip_limit=1e-6)
    # Explicit POS_X/POS_Y/POS_Z normal swizzles are pinned by the caller.
    q=(data[indices,:3].astype(np.float64)-.5-1e-5)*2
    qlength=np.linalg.norm(q,axis=1)
    _require(np.isfinite(q).all() and np.all(np.abs(qlength-1)<=1e-4),'expected unit live float normal sample before coordinate conversion')
    local=np.einsum('nij,nj->ni',samples['old'],q)
    converted=np.linalg.solve(samples['new'],local[...,None])[...,0]
    lengths=np.linalg.norm(converted,axis=1);_require(np.isfinite(converted).all() and np.all(lengths>0),'invalid converted normal')
    converted/=lengths[:,None];data[indices,:3]=(converted*.5+.5+1e-5).astype(np.float32)
    decoded=(data[indices,:3].astype(np.float64)-.5-1e-5)*2
    restored_local=np.einsum('nij,nj->ni',samples['new'],decoded)
    expected_world=np.einsum('nij,nj->ni',samples['world_from_local'],local)
    restored_world=np.einsum('nij,nj->ni',samples['world_from_local'],restored_local)
    def unit(values):
        length=np.linalg.norm(values,axis=1)
        _require(np.isfinite(values).all() and np.all(length>0),'invalid roundtrip world normal')
        return values/length[:,None]
    residual=float(np.linalg.norm(unit(restored_world)-unit(expected_world),axis=1).max())
    singular=np.linalg.svd(samples['old'],compute_uv=False)
    return dict(valid=residual<=1e-6,converted_texels=len(indices),max_condition_number=float(np.max(singular[:,0]/singular[:,-1])),
                minimum_singular_value=float(singular[:,-1].min()),max_normalized_world_roundtrip_residual=residual,world_roundtrip_limit=1e-6)


def pad_normal(data,occupancy,resolution,margin=2):
    """Bounded Euclidean nearest-covered extension; never touch covered samples."""
    covered=np.asarray(occupancy,bool).reshape(resolution,resolution);rgba=data.reshape(resolution,resolution,4)
    assigned=covered.copy();count=0
    # Ordered squared distance + lexicographic offset gives deterministic ties.
    offsets=sorted((dy*dy+dx*dx,dy,dx) for dy in range(-margin,margin+1) for dx in range(-margin,margin+1)
                   if 0<dy*dy+dx*dx<=margin*margin)
    for _,dy,dx in offsets:
        y0=max(0,-dy);y1=min(resolution,resolution-dy);x0=max(0,-dx);x1=min(resolution,resolution-dx)
        destination=assigned[y0:y1,x0:x1];source=covered[y0+dy:y1+dy,x0+dx:x1+dx]
        take=~destination&source
        rgba[y0:y1,x0:x1][take]=rgba[y0+dy:y1+dy,x0+dx:x1+dx][take]
        destination[take]=True;count+=int(take.sum())
    return count


def _angles(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    a=a/np.linalg.norm(a,axis=1)[:,None];b=b/np.linalg.norm(b,axis=1)[:,None]
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.einsum('ij,ij->i',a,b)))


def _yup(values):
    return np.column_stack((values[:,0],values[:,2],-values[:,1])).astype(np.float32)


def _keys(values):
    values=np.ascontiguousarray(values,dtype=np.float32).copy();values[values==0]=0
    return values.view(np.dtype([('v'+str(i),'<f4') for i in range(values.shape[1])])).reshape(-1)


def _oriented_triangles(corners):
    """Canonical cyclic rotation only; reversal is a different triangle."""
    corners=np.ascontiguousarray(corners,dtype=np.float32)
    _require(corners.ndim==3 and corners.shape[1:]==(3,5) and np.isfinite(corners).all(),'invalid triangle provenance')
    keys=_keys(corners.reshape(-1,5)).reshape(-1,3)
    rotation=np.argsort(keys,axis=1)[:,0];order=(rotation[:,None]+np.arange(3))%3
    canonical=np.take_along_axis(corners,order[:,:,None],axis=1)
    return _keys(canonical.reshape(-1,15)),order


def _frame_gate(n,t,sign):
    _require(np.isfinite(n).all() and np.isfinite(t).all() and np.isin(sign,(-1,1)).all(),'nonfinite frame or invalid sign')
    nl=np.linalg.norm(n.astype(np.float64),axis=1);tl=np.linalg.norm(t.astype(np.float64),axis=1)
    _require(np.all(nl>0) and np.all(tl>0),'zero frozen frame')
    metrics=dict(max_norm_error=float(max(np.abs(nl-1).max(initial=0),np.abs(tl-1).max(initial=0))),
                 max_abs_dot_nt=float(np.abs(np.einsum('ij,ij->i',n.astype(np.float64),t.astype(np.float64))).max(initial=0)))
    _require(metrics['max_norm_error']<=1e-4 and metrics['max_abs_dot_nt']<=1e-4,'frozen frame exceeds unit/orthogonal bounds: '+str(metrics))
    return metrics


def predict_serialization(raw_frames,planned_frames):
    """Freeze proper-Yup float32 frames by exact oriented (POS, flipped-UV) triangles.

    Joining may change evaluated normals, so no rounded-exporter normal is used
    as identity or accepted as the bake frame. Common part matrices are required.
    """
    _require(all(np.array_equal(s['matrix'],raw_frames[0]['matrix']) for s in raw_frames),'mixed part transforms cannot predict exact joined rows')
    corners=[];normals=[];tangents=[];signs=[];stable=[];offset=0;raw_zeros=0
    for raw,planned in zip(raw_frames,planned_frames):
        loops=raw['triangle_loops'];uv=raw['uv'].copy();uv[:,1]*=-1;uv[:,1]+=1
        corners.append(np.concatenate((_yup(raw['vertices'][raw['faces'].ravel()]).reshape(-1,3,3),uv[loops]),axis=2))
        normals.append(_yup(planned['n'][loops.ravel()]).reshape(-1,3,3))
        tangents.append(_yup(planned['t'][loops.ravel()]).reshape(-1,3,3));signs.append(planned['sign'][loops])
        stable.append(loops.astype(np.int64)+offset);offset+=len(raw['t']);raw_zeros+=int(np.all(raw['t']==0,axis=1).sum())
    corners=np.concatenate(corners);key,order=_oriented_triangles(corners)
    _require(len(np.unique(key))==len(key),'duplicate frozen oriented triangle provenance')
    def canonical(values):
        values=np.concatenate(values)
        return np.take_along_axis(values,order[:,:,None] if values.ndim==3 else order,axis=1)
    plan=dict(corners=np.take_along_axis(corners,order[:,:,None],axis=1),n=canonical(normals),t=canonical(tangents),
              sign=canonical(signs),stable_loop=canonical(stable))
    metrics=_frame_gate(plan['n'].reshape(-1,3),plan['t'].reshape(-1,3),plan['sign'].ravel())
    return plan,dict(metrics,triangles=len(key),raw_zero_tangents=raw_zeros,planned_zero_tangents=0,
                     method='Exact oriented triangle provenance; serialize frozen float32 N/T, preserve native w; no exporter rounding')


def restore_serialized_frames(plan,positions,uv,indices,old_n,old_t):
    """Exact authored corner frames; split conflicting render rows, never average."""
    _require(indices.ndim==2 and indices.shape[1]==3 and indices.min(initial=0)>=0 and indices.max(initial=-1)<len(positions),'invalid triangle indices')
    key,order=_oriented_triangles(np.concatenate((positions[indices],uv[indices]),axis=2));expected,_=_oriented_triangles(plan['corners'])
    _require(len(key)==len(expected) and len(np.unique(key))==len(key) and len(np.unique(expected))==len(expected),'triangle provenance count/uniqueness differs')
    sort=np.argsort(expected);found=np.searchsorted(expected[sort],key);_require(np.all(found<len(sort)),'missing/extra or reversed serialized triangle');_require(np.array_equal(expected[sort[found]],key),'missing/extra or reversed serialized triangle')
    match=sort[found];rows=np.take_along_axis(indices,order,axis=1).ravel();n=plan['n'][match].reshape(-1,3);t=plan['t'][match].reshape(-1,3);sign=plan['sign'][match].ravel();stable=plan['stable_loop'][match].ravel()
    _frame_gate(n,t,sign);_require(np.isfinite(old_n).all() and np.isfinite(old_t).all(),'invalid old serialized frame');_require(len(np.unique(rows))==len(positions),'unused serialized rows lack frozen provenance');_require(len(positions)<2**32,'render row count unsupported')
    # Exact float32 bit keys preserve even tiny T distinctions and signed zero.
    keys=np.column_stack((rows.astype(np.uint32),n.view(np.uint32),t.view(np.uint32),sign.view(np.uint32)))
    _,inverse=np.unique(keys,axis=0,return_inverse=True);by_variant=np.lexsort((stable,inverse));first=np.r_[0,np.flatnonzero(np.diff(inverse[by_variant]))+1];representative=by_variant[first]
    variant_rows=rows[representative];by_row=np.lexsort((stable[representative],variant_rows));row_first=np.r_[0,np.flatnonzero(np.diff(variant_rows[by_row]))+1];base_variant=by_row[row_first]
    primary=np.zeros(len(representative),bool);primary[base_variant]=True;extra=by_row[~primary[by_row]];clones=variant_rows[extra].astype(np.int64)
    destination=np.empty(len(representative),np.int64);destination[base_variant]=variant_rows[base_variant];destination[extra]=np.arange(len(positions),len(positions)+len(extra));new_rows=destination[inverse].reshape(indices.shape)
    remapped=np.empty_like(indices);np.put_along_axis(remapped,order,new_rows,axis=1)
    nr=np.concatenate((old_n,old_n[clones])).astype(np.float32);tr=np.concatenate((old_t,old_t[clones])).astype(np.float32)
    nr[destination]=n[representative];tr[destination,:3]=t[representative];tr[destination,3]=sign[representative]
    _require(np.array_equal(nr[new_rows.ravel()].view(np.uint32),n.view(np.uint32)) and np.array_equal(tr[new_rows.ravel(),:3].copy().view(np.uint32),t.view(np.uint32)) and np.array_equal(tr[new_rows.ravel(),3].view(np.uint32),sign.view(np.uint32)),'exact authored frame restoration differs')
    metrics=_frame_gate(nr,tr[:,:3],tr[:,3]);conflicting=np.flatnonzero(np.bincount(variant_rows,minlength=len(positions))>1)
    return nr,tr,remapped,clones,dict(metrics,triangles=len(key),old_rows=len(positions),new_rows=len(nr),added_render_rows=len(clones),remapped_render_corners=int((remapped!=indices).sum()),conflicting_original_rows=conflicting.tolist(),oriented_triangle_bijection=True,authored_N_T_sign_per_corner_byteexact=True,source_sign_per_corner_unchanged=bool(np.array_equal(old_t[rows,3],sign)),max_direction_error_degrees=0.,old_zero_tangent_rows=int((np.linalg.norm(old_t[:,:3].astype(float),axis=1)==0).sum()))


def restore_serialized_glb_frames(path,groups):
    """Compact own static-exporter views, cloning only exact frame-conflict rows.

    One tightly packed accessor per view, one primitive per planned material.
    Unsupported external/strided/shared/sparse/animated layouts fail explicitly.
    Existing image payloads are copied once; old vertex buffers are REPLACED.
    """
    from conservative_repair import _split_glb,_write_glb
    doc,binary=_split_glb(path);_require(len(doc.get('buffers',[]))==1 and 'uri' not in doc['buffers'][0] and not doc.get('skins') and not doc.get('animations'),'unsupported static exporter buffers/animation')
    views=doc['bufferViews'];accessors=doc['accessors'];payloads={};used_views=set();used_accessors=set();reports=[];seen=set();types={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4};dtypes={5120:'i1',5121:'u1',5122:'<i2',5123:'<u2',5125:'<u4',5126:'<f4'}
    def read(index):
        _require(index not in used_accessors,'unsupported shared accessor');used_accessors.add(index);a=accessors[index];vi=a['bufferView'];v=views[vi]
        _require(vi not in used_views and v.get('buffer',0)==0 and not v.get('byteStride') and not a.get('byteOffset') and not a.get('sparse') and a['type'] in types and a['componentType'] in dtypes,'unsupported tightly packed standalone accessor layout')
        used_views.add(vi);dtype=np.dtype(dtypes[a['componentType']]);width=types[a['type']];size=a['count']*width*dtype.itemsize;_require(v['byteLength']==size and 0<=v.get('byteOffset',0)<=len(binary)-size,'unsupported accessor/view range')
        return np.frombuffer(binary,dtype=dtype,count=a['count']*width,offset=v.get('byteOffset',0)).reshape(-1,width).copy()
    def store(index,values):
        a=accessors[index];a['count']=len(values);payloads[a['bufferView']]=values.tobytes()
    for mesh in doc.get('meshes',[]):
        _require(not mesh.get('weights'),'unsupported mesh weights')
        for primitive in mesh['primitives']:
            _require(not primitive.get('targets') and not primitive.get('extensions') and primitive.get('mode',4)==4,'unsupported compressed/morph/nontriangle primitive')
            material=doc['materials'][primitive['material']]['name'];_require(material in groups and material not in seen,'unsupported missing/shared planned material primitive');seen.add(material)
            attrs=primitive['attributes'];_require({'POSITION','NORMAL','TANGENT','TEXCOORD_0'}<=set(attrs),'missing authored frame attributes');values={name:read(index) for name,index in attrs.items()};old_ids=read(primitive['indices'])
            _require(old_ids.shape[1]==1 and old_ids.dtype.kind=='u' and old_ids.size%3==0,'unsupported index layout');indices=old_ids.reshape(-1,3).astype(np.int64);count=len(values['POSITION']);_require(all(len(v)==count for v in values.values()),'vertex attribute counts differ')
            _require(values['POSITION'].shape[1]==3 and values['NORMAL'].shape[1]==3 and values['TANGENT'].shape[1]==4 and values['TEXCOORD_0'].shape[1]==2 and all(values[k].dtype==np.dtype('<f4') for k in ['POSITION','NORMAL','TANGENT','TEXCOORD_0']),'unsupported float32 frame layout')
            for name in ('NORMAL','TANGENT'):_require(not {'min','max'}&set(accessors[attrs[name]]),'unsupported bounded frame accessor')
            with np.load(groups[material]) as saved:plan={k:saved[k] for k in saved.files}
            n,t,remapped,clones,metrics=restore_serialized_frames(plan,values['POSITION'],values['TEXCOORD_0'],indices,values['NORMAL'],values['TANGENT'])
            _require(remapped.max(initial=0)<=np.iinfo(old_ids.dtype).max,'unsupported index width after exact frame split')
            for name,index in attrs.items():
                result=n if name=='NORMAL' else t if name=='TANGENT' else np.concatenate((values[name],values[name][clones]))
                if name not in ('NORMAL','TANGENT'):_require(np.array_equal(result[remapped].view(np.uint8),values[name][indices].view(np.uint8)),'render split changed nonframe corner attribute '+name)
                store(index,result)
            new_ids=remapped.reshape(-1,1).astype(old_ids.dtype);store(primitive['indices'],new_ids);ia=accessors[primitive['indices']]
            if 'min' in ia:ia['min']=[int(new_ids.min())]
            if 'max' in ia:ia['max']=[int(new_ids.max())]
            reports.append(dict(material=material,nonframe_corner_attributes_byteexact=True,**metrics))
    _require(seen==set(groups) and used_accessors==set(range(len(accessors))),'unsupported unused accessor or planned material omitted')
    image_views=set()
    for image in doc.get('images',[]):
        _require('bufferView' in image and 'uri' not in image,'unsupported external image');image_views.add(image['bufferView'])
    _require(not image_views&used_views and image_views|used_views==set(range(len(views))),'unsupported shared/unused buffer view')
    ranges=sorted((v.get('byteOffset',0),v.get('byteOffset',0)+v['byteLength']) for v in views)
    _require(all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:])),'unsupported overlapping buffer views')
    rebuilt=bytearray()
    for vi,v in enumerate(views):
        offset=v.get('byteOffset',0);size=v['byteLength'];_require(v.get('buffer',0)==0 and 0<=offset<=len(binary)-size,'unsupported buffer view range')
        while len(rebuilt)%4:rebuilt.append(0)
        content=payloads.get(vi,binary[offset:offset+size]);v['byteOffset']=len(rebuilt);v['byteLength']=len(content);rebuilt.extend(content)
    while len(rebuilt)%4:rebuilt.append(0)
    doc['buffers'][0]['byteLength']=len(rebuilt);stage=path.with_suffix('.tangent.glb');_require(not stage.exists(),'existing frame-restoration stage')
    _write_glb(stage,doc,bytes(rebuilt));stage.replace(path)
    return dict(valid=True,parts=reports,exact_render_attribute_row_split=True,geometric_corner_attributes_material_images_preserved=True,image_payload_bytes_unchanged=True,old_BIN_bytes=len(binary),new_BIN_bytes=len(rebuilt),BIN_byte_growth=len(rebuilt)-len(binary),old_views_replaced_not_appended=True,viewer_tbn_validation='NOT_RUN; exact authored N/T is distinct from Blender regenerated Mikk')
