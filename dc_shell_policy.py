# SPDX-License-Identifier: GPL-3.0-or-later
"""Opt-in, bounded exclusion of one generated inward unsigned offset sheet.

This proves a cross-surface nesting relation, not global self-intersection
freedom, a signed solid, or appearance correspondence. No source faces change.
"""
import math
import numpy as np

MAX_PROOF_FACES=100000
MAX_PAIR_TESTS=128_000_000
MAX_CANDIDATES=16384
MARGIN=4096*np.finfo(np.float64).eps


def proof_capacity(source_faces,raw_faces,source_vertices,raw_vertices,*,remaining_bytes):
    """Conservative extra host-RSS estimate, not a hard allocation/VRAM bound."""
    counts=(source_faces,raw_faces,source_vertices,raw_vertices)
    if any(type(n) is not int or n<=0 for n in counts) or type(remaining_bytes) is not int:
        raise ValueError('DC sheet proof requires positive integer face/vertex counts and integer RSS headroom')
    faces=source_faces+raw_faces;vertices=source_vertices+raw_vertices
    if faces>MAX_PROOF_FACES:raise ValueError(f'DC sheet proof face bound exceeded: source={source_faces}, raw={raw_faces}, total={faces}, limit={MAX_PROOF_FACES}')
    if vertices>3*faces:raise ValueError(f'DC sheet proof vertex-row bound exceeded: source={source_vertices}, raw={raw_vertices}, total={vertices}, limit={3*faces}')
    estimate=64*1024**2+4096*faces
    if estimate>remaining_bytes:raise MemoryError(f'DC sheet proof estimated extra RSS exceeds remaining declared budget: faces={faces}, vertex_rows={vertices}, estimate={estimate} bytes, remaining={remaining_bytes} bytes')
    return dict(source_faces=source_faces,raw_faces=raw_faces,total_faces=faces,source_vertex_rows=source_vertices,raw_vertex_rows=raw_vertices,
                total_vertex_rows=vertices,face_limit=MAX_PROOF_FACES,vertex_row_limit=3*faces,estimated_extra_RSS_bytes=estimate,
                remaining_declared_RSS_bytes=remaining_bytes,scope='Cooperative conservative extra host RSS estimate; not a hard allocation or GPU-memory guarantee')


def _volume(triangles):
    terms=np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2]))/6
    value=math.fsum(terms.tolist());error=64*np.finfo(float).eps*math.fsum(np.abs(terms).tolist())
    if not math.isfinite(value) or abs(value)<=max(error,MARGIN):raise ValueError('DC sheet volume sign is unresolved')
    return value,error


def _aabb_candidates(first,second,work,check):
    """Bounded median tree; same F64 leaf predicate, original (i,j) order."""
    check()
    if not len(first) or not len(second):return
    lo=second.min(1);hi=second.max(1);nodes=[]
    def charge(kind,count):
        # Shared cap counts construction bounds, coarse nodes and exact leaves.
        before=work.get('pair_tests',0)
        if before+count>MAX_PAIR_TESTS:raise ValueError('DC sheet pair-work bound exceeded')
        work['pair_tests']=before+count;work[kind]=work.get(kind,0)+count
        if before//256!=(before+count)//256:check()
    def build(ids):
        charge('aabb_build_bounds',len(ids));lower=lo[ids].min(0);upper=hi[ids].max(0)
        index=len(nodes);nodes.append(None)
        if len(ids)<=16:nodes[index]=(lower,upper,ids,None,None)
        else:
            axis=int(np.argmax(upper-lower));order=np.argsort(lo[ids,axis]+hi[ids,axis],kind='stable');middle=len(ids)//2
            left=build(ids[order[:middle]]);right=build(ids[order[middle:]])
            nodes[index]=(lower,upper,None,left,right)
        return index
    root=build(np.arange(len(second),dtype=np.int32));check()
    for i,triangle in enumerate(first):
        lower=triangle.min(0);upper=triangle.max(0);stack=[root];hits=[]
        while stack:
            index=stack.pop();a,b,ids,left,right=nodes[index];charge('aabb_coarse_tests',1)
            # Exact enclosing extrema: this cannot reject an admitted leaf.
            if not (np.all(lower-MARGIN<=b) and np.all(a<=upper+MARGIN)):continue
            if ids is None:stack.extend((right,left));continue
            charge('aabb_leaf_tests',len(ids))
            possible=np.all(lower-MARGIN<=hi[ids],axis=1)&np.all(lo[ids]<=upper+MARGIN,axis=1)
            found=ids[possible];count=len(found)
            if work.get('candidate_pairs',0)+count>MAX_CANDIDATES:raise ValueError('DC sheet candidate-pair bound exceeded')
            work['candidate_pairs']=work.get('candidate_pairs',0)+count;hits.extend(found.tolist())
        for j in sorted(hits):yield i,j


def _separate(first,second,work,check):
    """Every raw AABB candidate needs a positive SAT separation witness.

    Coplanar in-plane axes are included: BVHTree.overlap misses coincident
    triangles and point/edge contacts. Unresolved candidates always refuse.
    Coordinates are normalized to absolute value <=1; MARGIN bounds the
    small fixed float64 cross/dot arithmetic, not a geometry displacement.
    """
    minimum=None;count=0
    for i,j in _aabb_candidates(first,second,work,check):
        count+=1;check();a=first[i];b=second[j];ea=np.roll(a,-1,axis=0)-a;eb=np.roll(b,-1,axis=0)-b
        na=np.cross(ea[0],ea[1]);nb=np.cross(eb[0],eb[1])
        axes=np.vstack((na,nb,np.cross(ea[:,None,:],eb[None,:,:]).reshape(-1,3),np.cross(na,ea),np.cross(nb,eb)))
        axes=axes[np.linalg.norm(axes,axis=1)>1e-12]
        pa=a@axes.T;pb=b@axes.T
        gap=float(np.maximum(pa.min(0)-pb.max(0),pb.min(0)-pa.max(0)).max()) if len(axes) else 0.
        if not math.isfinite(gap) or gap<=MARGIN:raise ValueError('DC sheets touch, cross or lack a numerical separation proof')
        minimum=gap if minimum is None else min(minimum,gap)
    return dict(AABB_candidates=count,minimum_raw_axis_gap=minimum,normalized_arithmetic_margin=MARGIN)


def _winding(triangles,point,expected,check):
    check();r=triangles-point;n=np.linalg.norm(r,axis=2)
    if not np.isfinite(n).all() or n.min()<=MARGIN:raise ValueError('DC sheet containment witness touches a surface')
    numerator=np.einsum('ij,ij->i',r[:,0],np.cross(r[:,1],r[:,2]))
    denominator=n.prod(1)+np.einsum('ij,ij->i',r[:,0],r[:,1])*n[:,2]+np.einsum('ij,ij->i',r[:,1],r[:,2])*n[:,0]+np.einsum('ij,ij->i',r[:,2],r[:,0])*n[:,1]
    value=math.fsum((2*np.arctan2(numerator,denominator)).tolist())/(4*math.pi)
    if not math.isfinite(value) or abs(value-expected)>1e-7:raise ValueError('DC sheet containment/orientation is ambiguous or unsupported')
    return value


def verified_outer(source_vertices,source_faces,vertices,faces,*,work,check,remaining_bytes):
    """Return the exact generated outward subset, or fail closed; never largest."""
    from geometric_edge_components import analyze_geometric_edges
    from wtivo_reconstruct import validate_output
    check()
    capacity=proof_capacity(len(source_faces),len(faces),len(source_vertices),len(vertices),remaining_bytes=remaining_bytes)
    # Render seams are identified exactly for proof only, never welded in source.
    sv,inverse=np.unique(source_vertices,axis=0,return_inverse=True)
    sf=inverse[source_faces].astype(np.int32);check()
    source_topology=validate_output(sv.astype(np.float32),sf,backend='CuMesh DC closed-source policy')
    check()
    source_groups=analyze_geometric_edges(sv,sf.tolist(),check_cancelled=check)
    check()
    if source_groups.get('assessment')!='assessed' or len(source_groups['components'])!=1:raise ValueError('DC sheet policy requires one closed source component')
    validate_output(vertices,faces,backend='CuMesh DC raw sheets')
    check()
    groups=analyze_geometric_edges(vertices,faces.tolist(),check_cancelled=check)
    check()
    if groups.get('assessment')!='assessed' or len(groups['components'])!=2:raise ValueError('DC sheet policy requires exactly two generated closed sheets')
    center=(np.minimum(sv.min(0),vertices.min(0)).astype(float)+np.maximum(sv.max(0),vertices.max(0)).astype(float))*.5
    scale=float(np.max(np.abs(np.r_[sv.astype(float)-center,vertices.astype(float)-center])))
    if not math.isfinite(scale) or scale<=0:raise ValueError('Invalid DC sheet normalization')
    s=(sv.astype(float)-center)/scale;v=(vertices.astype(float)-center)/scale;st=s[sf]
    check();source_volume,source_error=_volume(st);check()
    if source_volume<=0:raise ValueError('DC sheet policy requires outward-oriented closed source, not an intentional inward boundary')
    sheets=[]
    for ids in groups['components']:
        check();ids=np.asarray(ids,np.int32);f=faces[ids];rows,inv=np.unique(f,return_inverse=True)
        proof=validate_output(vertices[rows],inv.reshape(-1,3).astype(np.int32),backend='CuMesh DC sheet')
        check()
        if proof['indexed']['Euler']!=source_topology['indexed']['Euler']:raise ValueError('DC sheet policy cannot change source genus/openings')
        volume,error=_volume(v[f]);check();sheets.append(dict(ids=ids,volume=volume,error=error,faces=len(f),vertices=len(rows)))
    inner=[x for x in sheets if x['volume']<0];outer=[x for x in sheets if x['volume']>0]
    if len(inner)!=1 or len(outer)!=1:raise ValueError('DC sheet policy requires opposite unambiguous generated orientations')
    inner=inner[0];outer=outer[0];it=v[faces[inner['ids']]];ot=v[faces[outer['ids']]]
    pairs={name:_separate(a,b,work,check) for name,a,b in [('source_inner',st,it),('source_outer',st,ot),('inner_outer',it,ot)]}
    winding={name:_winding(t,p,expected,check) for name,t,p,expected in [('inner_in_source',st,it[0,0],1),('outer_outside_source',st,ot[0,0],0),
        ('source_in_outer',ot,st[0,0],1),('source_outside_inner',it,st[0,0],0),('inner_in_outer',ot,it[0,0],1),('outer_outside_inner',it,ot[0,0],0)]}
    check()
    kept=faces[outer['ids']];rows,inverse=np.unique(kept,return_inverse=True)
    observations=[]
    for label,sheet in [('inner_removed',inner),('outer_kept',outer)]:
        physical_volume=sheet['volume']*scale**3
        if not math.isfinite(physical_volume):raise ValueError('Nonfinite DC sheet volume')
        observations.append(dict(role=label,faces=sheet['faces'],vertices=sheet['vertices'],signed_volume_local=physical_volume,
            normalized_volume=sheet['volume'],normalized_volume_roundoff_bound=sheet['error']))
    import hashlib
    record=dict(policy='verified_outer',removed_faces=inner['faces'],kept_faces=outer['faces'],raw_faces=len(faces),raw_vertices=len(vertices),
        removed_face_ids_sha256=hashlib.sha256(inner['ids'].tobytes()).hexdigest(),kept_face_ids_sha256=hashlib.sha256(outer['ids'].tobytes()).hexdigest(),
        source=dict(faces=len(sf),signed_volume_local=source_volume*scale**3,normalized_volume_roundoff_bound=source_error,topology=source_topology),
        sheets=observations,cross_pairs=pairs,winding=winding,work=dict(work),capacity=capacity,limits=dict(faces=MAX_PROOF_FACES,pair_tests=MAX_PAIR_TESTS,candidate_pairs=MAX_CANDIDATES),
        scope='Per original component only; exact generated outward subset, not signed-solid reconstruction. Cross-surface separation/nesting, not global self-intersection or PBR certification')
    check();return vertices[rows],inverse.reshape(-1,3).astype(np.int32),record
