# SPDX-License-Identifier: GPL-3.0-or-later
"""Own-HIGH paint-sample provenance, isolated proxy geometry and normal coordinates.

Only geometry/numeric operations live here. Textures and PBR are evaluated by
Blender's native shader bake, never by this module.
"""
import numpy as np
import math
import time
from production_tangents import triangle_samples, _frame, analytic_tangent

MAX_ATLAS_PIXELS=16_777_216
MAX_PROXY_SAMPLES=4_194_304
NORMAL_BIAS=1e-5


ORIENTED_RECORD_DTYPE=np.dtype([
    ('sample_row','<u4'),('owner_row','<u4'),('source_face0','<i4'),('component','<i4'),
    ('Q0','<f8',(3,)),('d0','<f8'),('query_count','u1'),('selected_radius','<f8'),
    ('SOURCE_Ns0','<f8',(3,)),('SOURCE_Ns1','<f8',(3,)),('LOW_Ns','<f8',(3,)),
    ('initial_strategy','u1'),('preferred_status','u1'),('preferred_reason','u1'),('initial_margin_ULPs','u1')])
ORIENTED_POSITION_DTYPE=np.dtype([
    ('oriented_row','<u4'),('final_strategy','u1'),('final_margin_ULPs','u1'),('encoded','u1'),('accepted','u1'),
    ('native_RGB','<f4',(3,)),('position','<f8',(3,)),('Q_error','<f8'),('LOW_to_Q1','<f8'),('LOW_to_hit','<f8')])
ORIENTED_STRATEGIES=('source_directed_delta_half','nearest_geometric_normal_precision_floor','own_HIGH_interior_auxiliary_normal_sampler')
ORIENTED_PREFERENCES=('not_available','admitted','rejected')
ORIENTED_REASONS=('none','bounded_source_ULP_candidates_exhausted','aim_outside_original_bound','inset_unrepresentable',
                  'auxiliary_geometry_guard','strict_own_face_witness_unavailable','original_LOW_relative_caps')
ORIENTED_METADATA_RESERVE=1024**2
ORIENTED_FILE_HEADER_RESERVE=4096
AUXILIARY_CHANGE_DTYPE=np.dtype([('sample_row','<u4'),('stage','u1'),('Aim','<f8',(3,)),('height','<f8'),('method','u1')])
AUXILIARY_FALLBACK_ROWS=256


def oriented_resource_bytes(records,parts,human_bytes):
    """Conservative EXTRA payload admission, not compressed size or texture data.

    Reserve both numeric stages before POSITION exists. 256 parts / 65536 rows
    / 32 full <=8KiB details fit the unchanged16MiB, including NPY/ZIP overhead.
    """
    if not (0<=records<=65536 and 0<=parts<=256 and 0<=human_bytes<=32*8192):raise ValueError('Compact oriented provenance bounds exceeded')
    return records*(ORIENTED_RECORD_DTYPE.itemsize+ORIENTED_POSITION_DTYPE.itemsize)+parts*2*ORIENTED_FILE_HEADER_RESERVE+human_bytes+ORIENTED_METADATA_RESERVE+AUXILIARY_FALLBACK_ROWS*AUXILIARY_CHANGE_DTYPE.itemsize


def oriented_provenance_schema():
    return dict(version=2,format='NPZ numeric arrays; allow_pickle=False',record_array='oriented',POSITION_array='oriented_POSITION',
        record_dtype=ORIENTED_RECORD_DTYPE.descr,POSITION_dtype=ORIENTED_POSITION_DTYPE.descr,
        strategies=dict(enumerate(ORIENTED_STRATEGIES)),preferred_status=dict(enumerate(ORIENTED_PREFERENCES)),preferred_reason=dict(enumerate(ORIENTED_REASONS)),
        margin_zero='Not selected/unavailable; never inferred',human_details='First32 globally across this grid, diagnostics only; full rows are numeric',
        orientation_basis=dict(array='orientation_basis',values={0:'SOURCE_Ns0/1 and LOW_Ns are unperturbed authored/planned-base normals',
            1:'SOURCE_Ns0/1 and LOW_Ns are geometric world orientation vectors, NOT authored/effective shader normals'}),
        dense_bindings=dict(Q1='nearest[sample_row]',g0='guards[sample_row]',P='LOW_points[sample_row]',source_face1='source_faces[sample_row]',
                            pixel='indices[sample_row]',LOW_face='faces[sample_row]',owner_row='native-owners-decoded.npz global row'),
        position_binding='oriented_POSITION.oriented_row indexes oriented; final strategy distinct from initial',
        sampler_changes=dict(version=1,array='oriented_sampler_changes',dtype=AUXILIARY_CHANGE_DTYPE.descr,
                             stages={0:'initial construction in nearest NPZ',1:'POSITION retry in POSITION NPZ'},
                             methods={1:'dyadic Aim/height',2:'one shared bounded cell refinement'},
                             scope='Changed auxiliary geometry only, sample_row binds existing dense Q1/P/guard/source face; not texture filling',
                             seed='Existing admitted auxiliary Aim, or selected Q1 when preferred construction exhausted; rejected preferred status and margin0 remain truthful'),
        resource_scope='Uncompressed extra numeric payload plus bounded JSON, reserved NPY/ZIP headers and<=1MiB schema/descriptors; existing dense fields/raw maps excluded')


def oriented_numeric_record(selection,sample_row,owner_row,fact):
    """One transient selection -> one numeric row; no per-texel dictionary cache."""
    preference=fact.get('oriented_preferred_auxiliary') or {};margins=fact.get('auxiliary_margin_candidates') or {}
    if not 0<=sample_row<MAX_PROXY_SAMPLES or not 0<=owner_row<MAX_PROXY_SAMPLES:raise ValueError('Invalid oriented owner-row binding')
    if not 1<=selection['query']['query_count']<=32:raise ValueError('Invalid oriented query count')
    result=np.empty((),ORIENTED_RECORD_DTYPE)
    geometric=selection.get('orientation_basis')=='geometric_world'
    vectors=(selection['SOURCE_geoN0'],selection['SOURCE_geoN1'],selection['LOW_geoN']) if geometric else (selection['SOURCE_authoredNs0'],selection['SOURCE_authoredNs1'],selection['LOW_plannedBaseNs'])
    result[()]=(sample_row,owner_row,selection['source_face0'],selection['original_component'],selection['Q0'],selection['d0'],
        selection['query']['query_count'],selection['query']['selected_radius'],*vectors,
        ORIENTED_STRATEGIES.index(fact['strategy']),ORIENTED_PREFERENCES.index(preference.get('status','not_available')),
        ORIENTED_REASONS.index(preference.get('reason','none')),margins.get('selected_margin_ULPs') or 0)
    return result


def oriented_position_records(sparse,data,indices,positions,errors,low_to_Q,low_to_hit,accepted,final_strategy,final_margin):
    """Retain actual final POSITION before FACE, including refused numeric values.

    Missing/invalid encoding is marked, never interpreted as a physical hit.
    This is retention only: it does not decide any native acceptance gate.
    """
    if sparse.dtype!=ORIENTED_RECORD_DTYPE or len(sparse)>65536:raise ValueError('Invalid oriented numeric records')
    rows=sparse['sample_row'];result=np.empty(len(rows),ORIENTED_POSITION_DTYPE);rgb=data[indices[rows],:3]
    result['oriented_row']=np.arange(len(rows),dtype=np.uint32);result['final_strategy']=final_strategy;result['final_margin_ULPs']=final_margin
    result['native_RGB']=rgb;rgb64=rgb.astype(float);result['encoded']=np.all((rgb64>=.25-1e-6)&(rgb64<=.75+1e-6),axis=1)
    result['accepted']=accepted[rows];result['position']=positions[rows];result['Q_error']=errors[rows]
    result['LOW_to_Q1']=low_to_Q[rows];result['LOW_to_hit']=low_to_hit[rows]
    return result


class OrientedSourceRejected(ValueError):
    def __init__(self, reason, facts):
        self.facts=dict(facts,reason=reason)
        super().__init__(reason)


def geometric_world_normals(world_vertices, faces, matrix):
    """Cache geometric normals in the renderer's inverse-transpose convention."""
    determinant=float(np.linalg.det(np.asarray(matrix,np.float64)[:3,:3]))
    if not np.isfinite(determinant) or determinant==0:raise ValueError('Singular/nonfinite geometric normal transform')
    result=np.empty((len(faces),3),np.float64)
    for start in range(0,len(faces),65536):
        triangle=world_vertices[faces[start:start+65536]]
        normal=np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0])
        length=np.linalg.norm(normal,axis=1)
        if not np.isfinite(normal).all() or np.any(length<=0):raise ValueError('Invalid geometric face normal')
        result[start:start+len(triangle)]=normal/length[:,None]*(1 if determinant>0 else -1)
    return result


def original_source_face_ids(ranges, ids, face_count):
    """Original census ranges, never a recensus after partitioning/filtering."""
    if not 0<face_count<=2_000_000 or len(ranges)!=len(ids):raise ValueError('Missing original SOURCE component ranges')
    labels=np.empty(face_count,np.int32);previous=0
    for (start,end,component),expected in zip(ranges,ids):
        if (type(start) is not int or type(end) is not int or type(component) is not int
                or start!=previous or not start<end<=face_count or component!=expected or component<0):
            raise ValueError('Invalid original SOURCE component ranges')
        labels[start:end]=component;previous=end
    if previous!=face_count or len(set(ids))!=len(ids):raise ValueError('Incomplete/duplicate original SOURCE component ranges')
    return labels


def exact_face_position_index(vertices, faces):
    """Lazy own-HIGH exact position multiset index, including reversed duplicates.

    At most2M 36-byte keys; sorting peaks below150MiB including bounded chunks.
    Native NumPy sorting is finite/source-capped, not preemptible by Python.
    No world-coordinate rounding, welding or attribute equivalence is inferred.
    """
    if (vertices.dtype!=np.float32 or vertices.ndim!=2 or vertices.shape[1]!=3
            or len(vertices)>6_000_000 or not np.isfinite(vertices).all()
            or faces.ndim!=2 or faces.shape[1]!=3 or not np.issubdtype(faces.dtype,np.integer)
            or not 0<len(faces)<=2_000_000 or faces.min()<0 or faces.max()>=len(vertices)):
        raise ValueError('Invalid bounded exact-face SOURCE geometry')
    keys=np.empty(len(faces),dtype='V36')
    for start in range(0,len(faces),65536):
        triangles=vertices[faces[start:start+65536]]
        order=np.lexsort((triangles[:,:,2],triangles[:,:,1],triangles[:,:,0]),axis=1)
        canonical=triangles[np.arange(len(triangles))[:,None],order]
        canonical[canonical==0]=np.float32(0)  # Lookup only: +0/-0 are coincident positions.
        keys[start:start+len(triangles)]=np.ascontiguousarray(canonical).reshape(-1,9).view('V36').ravel()
    keys.sort()
    return keys


def exact_surviving_source_face(index, vertices, faces, low_triangle, low_matrix, high_matrix,
                                face, component, material, components, materials):
    """Only an exact cyclic face in the already-proven original part namespace."""
    if (np.asarray(low_matrix).dtype!=np.asarray(high_matrix).dtype
            or np.asarray(low_matrix).tobytes()!=np.asarray(high_matrix).tobytes()
            or type(face)is not int or not 0<=face<len(faces)
            or components[face]!=component or materials[face]!=material):return False
    triangle=np.asarray(low_triangle)
    if triangle.dtype!=np.float32 or triangle.shape!=(3,3)or not np.isfinite(triangle).all():return False
    high=vertices[faces[face]]
    if not any(np.roll(high,k,axis=0).tobytes()==triangle.tobytes()for k in range(3)):return False
    order=np.lexsort((triangle[:,2],triangle[:,1],triangle[:,0]))
    canonical=triangle[order].copy();canonical[canonical==0]=np.float32(0)
    key=np.ascontiguousarray(canonical).reshape(1,9).view('V36').ravel()[0]
    return int(np.searchsorted(index,key,side='right')-np.searchsorted(index,key,side='left'))==1


def source_shading_cache(world, faces, loops, corner_n, matrix):
    """Bounded actual-RNA authored Ns cache for an unperturbed SOURCE only."""
    world=np.asarray(world);faces=np.asarray(faces);loops=np.asarray(loops);corner_n=np.asarray(corner_n);matrix=np.asarray(matrix,float)
    if (world.ndim!=2 or world.shape[1]!=3 or len(world)>6_000_000 or not np.isfinite(world).all()
            or faces.ndim!=2 or faces.shape[1]!=3 or not 0<len(faces)<=2_000_000
            or not np.issubdtype(faces.dtype,np.integer) or np.any(faces<0) or np.any(faces>=len(world))
            or loops.shape!=faces.shape or not np.issubdtype(loops.dtype,np.integer)
            or corner_n.dtype!=np.float32 or corner_n.ndim!=2 or corner_n.shape[1]!=3 or len(corner_n)>6_000_000
            or np.any(loops<0) or np.any(loops>=len(corner_n)) or matrix.shape!=(4,4) or not np.isfinite(matrix).all()):
        raise ValueError('Invalid/unbounded actual SOURCE corner-normal cache inputs')
    try:inverse=np.linalg.inv(matrix[:3,:3])
    except np.linalg.LinAlgError as error:raise ValueError('Singular SOURCE normal transform') from error
    if not np.isfinite(inverse).all():raise ValueError('Nonfinite SOURCE inverse transpose')
    for start in range(0,len(corner_n),65536):
        block=corner_n[start:start+65536].astype(float)
        if not np.isfinite(block).all() or np.any(np.einsum('ij,ij->i',block,block)<=0):raise ValueError('Nonfinite/zero actual SOURCE corner normal')
    duals=np.empty((len(faces),2,3),np.float64)
    for start in range(0,len(faces),65536):
        tri=world[faces[start:start+65536]].astype(float);ab=tri[:,1]-tri[:,0];ac=tri[:,2]-tri[:,0]
        cross=np.cross(ab,ac);squared=np.einsum('ij,ij->i',cross,cross)
        if not np.isfinite(squared).all() or np.any(squared<=0):raise ValueError('Degenerate/nonfinite SOURCE shading triangle')
        duals[start:start+len(tri),0]=np.cross(ac,cross)/squared[:,None]
        duals[start:start+len(tri),1]=np.cross(cross,ab)/squared[:,None]
    if not np.isfinite(duals).all():raise ValueError('Nonfinite SOURCE barycentric duals')
    return dict(world=world,faces=faces,loops=loops,corner_n=corner_n,inverse=inverse,duals=duals,
                additional_bytes=corner_n.nbytes+duals.nbytes+inverse.nbytes)


def source_shading_normal(cache, face, point):
    """Interpolate ORIGINAL object-space RNA corners, then own invT and unit.

    Signed planar barycentrics are not clamped/epsilon-corrected. No texture or
    BSDF normal perturbation is evaluated here. Geometric cage N is separate.
    """
    if type(face) is not int or not 0<=face<len(cache['faces']):raise ValueError('Invalid SOURCE shading face')
    point=np.asarray(point,float)
    if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('Nonfinite/invalid SOURCE shading point')
    uv=cache['duals'][face]@(point-cache['world'][cache['faces'][face,0]])
    bary=np.array([1-uv.sum(),uv[0],uv[1]])
    normal=(bary@cache['corner_n'][cache['loops'][face]].astype(float))@cache['inverse']
    squared=float(normal@normal)
    if not np.isfinite(np.r_[bary,normal,squared]).all() or squared<=0:raise ValueError('Nonfinite/zero interpolated SOURCE authored shading normal')
    return normal/np.sqrt(squared),bary


def planned_base_shading_normals(snapshot, faces, weights):
    """Exact _frame N F32 multiply/add order, then LOW's own invT and unit.

    Only a bounded chunk of three-corner values exists at once. This reads the
    frozen planned base; it never rewrites N/T or the later normal-map ABI.
    """
    faces=np.asarray(faces);weights=np.asarray(weights);n=np.asarray(snapshot['n']);loops=np.asarray(snapshot['triangle_loops'])
    if (faces.ndim!=1 or not np.issubdtype(faces.dtype,np.integer) or len(faces)>MAX_PROXY_SAMPLES
            or np.any(faces<0) or np.any(faces>=len(loops)) or weights.dtype!=np.float32
            or weights.shape!=(len(faces),3) or not np.isfinite(weights).all() or n.dtype!=np.float32
            or n.ndim!=2 or n.shape[1]!=3 or len(n)>6_000_000 or loops.ndim!=2 or loops.shape[1]!=3
            or not np.issubdtype(loops.dtype,np.integer) or np.any(loops<0) or np.any(loops>=len(n))):raise ValueError('Invalid frozen LOW planned-base shading inputs')
    matrix=np.asarray(snapshot['matrix'],float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all():raise ValueError('Invalid LOW normal transform')
    try:inverse=np.linalg.inv(matrix[:3,:3])
    except np.linalg.LinAlgError as error:raise ValueError('Singular LOW normal transform') from error
    result=np.empty((len(faces),3),np.float64)
    for start in range(0,len(faces),65536):
        end=min(start+65536,len(faces));corners=n[loops[faces[start:end]]];w=weights[start:end]
        local=np.add(np.add(np.multiply(corners[:,0],w[:,0,None]),np.multiply(corners[:,1],w[:,1,None])),np.multiply(corners[:,2],w[:,2,None]))
        world=local.astype(float)@inverse;length=np.linalg.norm(world,axis=1)
        if not np.isfinite(world).all() or not np.isfinite(length).all() or np.any(length<=0):raise ValueError('Nonfinite/zero LOW planned-base shading normal')
        result[start:end]=world/length[:,None]
    return result


def oriented_source(point, low_normal, q0, normal0, face0, d0, guard, ray_cap,
                    component, material, context, nearest_range, budget, *, low_geometric_normal=None, clock=time.monotonic):
    """Propose minimum (squared distance, face ID) in the first compatible API window.

    The BVH API uses its F32 strict-radius semantics; this is not an exhaustive
    F64 mathematical nearest/closed-ball certificate. Native witnesses decide.
    """
    facts=dict(Q0=np.asarray(q0,float).tolist(),source_face0=int(face0),d0=float(d0),g0=float(guard),
               original_component=int(component),material=int(material))
    budget['clock_mark']=clock()
    def charge():
        now=clock();budget['work_seconds']+=now-budget['clock_mark'];budget['clock_mark']=now
        return budget['work_seconds']
    def reject(reason):
        charge();raise OrientedSourceRejected(reason,facts)
    def check():
        if charge()>budget['max_seconds']:reject('oriented_query_deadline')
    if not 0<=face0<len(context['faces'])<=2_000_000:reject('invalid_Q0_source_face_or_domain')
    geometric_low=np.asarray(low_normal if low_geometric_normal is None else low_geometric_normal,float)
    appearance=context.get('shading')
    geometric=appearance is None and context.get('geometric_recovery_qualified') is True
    if appearance is None:
        if float(context['normals'][face0]@geometric_low)>0:return q0,normal0,face0,None
        facts.update(classifier_method='Perturbed SOURCE: compatible legacy geometry path only; appearance recovery NOT_QUALIFIED',
                     SOURCE_geoN0=context['normals'][face0].tolist(),LOW_geoN=geometric_low.tolist())
        if not geometric:reject('perturbed_SOURCE_orientation_recovery_not_qualified')
        if not np.isfinite(np.r_[context['normals'][face0],geometric_low]).all():reject('invalid_geometric_orientation')
        facts.update(classifier_method='HIGH_geometric_N_vs_LOW_geometric_N; perturbed normal exclusively from native shader',orientation_basis='geometric_world')
    else:
        shading0,bary0=source_shading_normal(appearance,face0,q0)
        if float(shading0@low_normal)>0:return q0,normal0,face0,None
        facts.update(classifier_method='HIGH_unperturbed_RNA_Ns_vs_LOW_plannedBase_Ns',orientation_basis='authored_planned_base',
                     SOURCE_authoredNs0=shading0.tolist(),SOURCE_bary0=bary0.tolist(),LOW_plannedBaseNs=np.asarray(low_normal).tolist())
    facts.update(source_component0=int(context['components'][face0]),source_material0=int(context['materials'][face0]),
                 SOURCE_geoN0=context['normals'][face0].tolist(),LOW_geoN=geometric_low.tolist())
    if context['components'][face0]!=component or context['materials'][face0]!=material:reject('foreign_Q0_provenance')
    if budget['samples']>=budget['max_samples']:reject('oriented_sample_budget')
    if not np.isfinite([d0,guard,ray_cap]).all() or d0<0 or min(guard,ray_cap)<=0:reject('invalid_oriented_search_caps')
    budget['samples']+=1;check();radius=min(float(ray_cap),max(float(d0+guard),float(guard)))
    query=dict(method='Full own-HIGH BVH.find_nearest_range bounded doubling windows',origin=np.asarray(point,float).tolist(),original_ray_cap=float(ray_cap),
               initial_radius=radius,query_count=0,max_queries=32,returned_hits=0,
               scope='Minimum returned compatible candidate in FIRST admissible F32 API window, not exact F64 nearest/closed-ball proof')
    facts['query']=query
    for window in range(32):
        check();query.update(query_count=window+1,radius=radius)
        hits=nearest_range(point,radius);budget['range_queries']+=1
        budget['returned_hits']+=len(hits);query['returned_hits']+=len(hits);query['last_window_returned_hits']=len(hits);check()
        # FULL native list exists before Python length/time admission; source2M
        # bounds the domain, not native allocation or query duration.
        if len(hits)>budget['max_hits_per_sample'] or budget['returned_hits']>budget['max_returned_hits']:reject('oriented_range_hit_budget')
        seen=set();best=None;filtered=dict(foreign_provenance=0,incompatible_orientation=0,outside_recomputed_cap=0)
        query['last_window_filtered']=filtered
        for ordinal,(where,normal,face,distance) in enumerate(hits):
            if ordinal%256==0:check()
            where=np.asarray(where,float);normal=np.asarray(normal,float)
            if (type(face) is not int or not 0<=face<len(context['faces']) or where.shape!=(3,) or normal.shape!=(3,)
                    or not isinstance(distance,(int,float,np.integer,np.floating)) or isinstance(distance,(bool,np.bool_))
                    or not np.isfinite(np.r_[where,normal,distance]).all() or distance<0):reject('invalid_oriented_range_result')
            if face in seen:reject('duplicate_oriented_range_face')
            seen.add(face)
            if context['components'][face]!=component or context['materials'][face]!=material:
                filtered['foreign_provenance']+=1;continue
            if geometric:
                shading=context['normals'][face];bary=None;dot=float(shading@geometric_low)
                if not np.isfinite(dot):reject('invalid_geometric_orientation')
            else:
                shading,bary=source_shading_normal(appearance,face,where);dot=float(shading@low_normal)
            if dot<=0:filtered['incompatible_orientation']+=1;continue
            delta=where-point;squared=float(delta@delta);d1=float(np.sqrt(squared))
            if d1>ray_cap:filtered['outside_recomputed_cap']+=1;continue
            if best is None or (squared,face)<best[0]:best=((squared,face),where,normal,d1,float(distance),dot,shading,bary)
        if best is not None:
            query['selected_radius']=radius;break
        if radius>=ray_cap:reject('no_oriented_range_candidate_at_cap')
        next_radius=min(float(ray_cap),radius*2)
        if not np.isfinite(next_radius) or next_radius<=radius:reject('oriented_radius_no_progress')
        radius=next_radius
    else:reject('oriented_query_window_budget')
    (squared,face),where,normal,d1,native_distance,dot,shading1,bary1=best;check();budget['proposed']+=1
    facts.update(Q1=where.tolist(),source_face1=face,source_component1=int(context['components'][face]),
                 source_material1=int(context['materials'][face]),SOURCE_geoN1=context['normals'][face].tolist(),
                 d1=d1,squared_distance=squared,native_reported_distance=native_distance,orientation_dot=dot,
                 geometric_dot=float(context['normals'][face]@geometric_low),
                 reason='incompatible_Q0_first_admissible_API_window_minimum_distance_faceID_proposal',
                 acceptance='PENDING native POSITION and SOURCEFACE witnesses; coincident-call stability NOT_QUALIFIED')
    if not geometric:facts.update(appearance_dot=dot,SOURCE_authoredNs1=shading1.tolist(),SOURCE_bary1=bary1.tolist())
    return where,normal,face,facts


def source_face_polygon_values(triangle_polygons, polygon_count):
    """Flat FACE values encode own-HIGH triangle IDs, even with polygon permutation."""
    ids=np.asarray(triangle_polygons)
    if (type(polygon_count) is not int or not 0<polygon_count<2**24 or ids.shape!=(polygon_count,)
            or not np.issubdtype(ids.dtype,np.integer) or not np.array_equal(np.sort(ids),np.arange(polygon_count))):
        raise ValueError('SOURCEFACE requires a bijection of triangles to triangulated polygons')
    values=np.empty(polygon_count,np.float32);values[ids]=np.arange(1,polygon_count+1,dtype=np.float32)
    return values


class SourceFaceMismatch(ValueError):
    """Only a valid full native FACE field with same-component ID disagreement."""
    def __init__(self,message,rows,observed,proposed):
        super().__init__(message)
        self.rows=rows.copy();self.observed=observed.copy();self.proposed=proposed.copy()


class Q1SelfGeometricIneligible(ValueError):
    """Well-formed same-domain Q1 cannot use the separate sparse SELF context."""


def decode_source_face_witness(rgba, indices, face_count, recovered_rows, proposed_faces, components):
    """Native full-proxy coverage and exact proposed SOURCE triangle identity, not PBR."""
    rgba=np.asarray(rgba);indices=np.asarray(indices);rows=np.asarray(recovered_rows)
    proposed=np.asarray(proposed_faces);components=np.asarray(components)
    if (rgba.dtype!=np.float32 or rgba.ndim!=2 or rgba.shape[1]!=4 or len(rgba)>MAX_ATLAS_PIXELS
            or type(face_count) is not int or not 0<face_count<=2_000_000
            or indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer)
            or len(indices)>MAX_PROXY_SAMPLES or np.any(indices<0) or np.any(indices>=len(rgba))
            or len(np.unique(indices))!=len(indices) or proposed.shape!=indices.shape
            or not np.issubdtype(proposed.dtype,np.integer) or np.any(proposed<0) or np.any(proposed>=face_count)
            or rows.ndim!=1 or not np.issubdtype(rows.dtype,np.integer) or not 0<len(rows)<=65536
            or np.any(rows<0) or np.any(rows>=len(indices)) or len(np.unique(rows))!=len(rows)
            or components.shape!=(face_count,) or not np.issubdtype(components.dtype,np.integer) or np.any(components<0)):
        raise ValueError('Invalid SOURCEFACE witness input/domain/resource bound')
    if not np.isfinite(rgba).all():raise ValueError('Nonfinite native SOURCEFACE field')
    covered=np.flatnonzero(np.any(rgba[:,:3]!=0,axis=1))
    if not np.array_equal(covered,np.sort(indices)):raise ValueError('Native SOURCEFACE sample set differs from full proxy')
    rgb=rgba[indices,:3];tags=rgb[:,0]
    if (np.any(rgb!=tags[:,None]) or np.any(tags!=np.floor(tags)) or np.any(tags<1) or np.any(tags>face_count)):
        raise ValueError('Invalid native SOURCEFACE integer/range encoding')
    observed=tags.astype(np.int32)-1;bad=rows[observed[rows]!=proposed[rows]]
    if len(bad):
        message=f'Native SOURCEFACE differs from Q1: count={len(bad)} first_pixels={indices[bad[:8]].tolist()} observed={observed[bad[:8]].tolist()} proposed={proposed[bad[:8]].tolist()}'
        if not np.array_equal(components[observed[rows]],components[proposed[rows]]):raise ValueError(message)
        raise SourceFaceMismatch(message,bad,observed[bad],proposed[bad])
    if not np.array_equal(components[observed[rows]],components[proposed[rows]]):raise ValueError('Native SOURCEFACE original component differs')
    return dict(valid=True,full_proxy_samples=len(indices),recovered_rows=len(rows),exact_source_triangle_ID=True,
                original_component_ID=True,coincident_independent_call_stability='NOT_QUALIFIED',
                meaning='FACE and POSITION witnesses are separate native calls, not a PBR coincident-hit certificate')


def q1_self_interiors(error,recovered_rows,nearest,guards,low_points,ray_cap,context,low_shading):
    """Qualify a different SELF evaluation context, never forgive a ray ID."""
    if not isinstance(error,SourceFaceMismatch):raise ValueError('Sparse Q1 SELF requires a typed same-domain mismatch')
    rows=error.rows;recovered_rows=np.asarray(recovered_rows);nearest=np.asarray(nearest);guards=np.asarray(guards);low_points=np.asarray(low_points);low_shading=np.asarray(low_shading)
    if (rows.ndim!=1 or not np.issubdtype(rows.dtype,np.integer) or not 0<len(rows)<=32 or len(np.unique(rows))!=len(rows)
        or error.observed.shape!=rows.shape or error.proposed.shape!=rows.shape
        or any(not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0) or np.any(ids>=len(context['faces'])) for ids in (error.observed,error.proposed))
        or nearest.ndim!=2 or nearest.shape[1]!=3 or guards.shape!=(len(nearest),)
        or low_points.shape!=nearest.shape or low_shading.shape!=nearest.shape or np.any(rows<0) or np.any(rows>=len(nearest))
        or recovered_rows.ndim!=1 or not np.issubdtype(recovered_rows.dtype,np.integer)
        or len(recovered_rows)>65536 or np.any(recovered_rows<0) or np.any(recovered_rows>=len(nearest))):
        raise ValueError('Sparse Q1 SELF invalid sample/face inputs')
    if (context.get('shading') is None
        or len(np.unique(error.proposed))!=len(rows) or not np.isin(rows,recovered_rows).all()):raise ValueError('Sparse Q1 SELF requires unique unperturbed recovered faces')
    barycentrics=[];details=[];geometric_ineligible=False
    for row,observed,proposed in zip(rows,error.observed,error.proposed):
        row=int(row);observed=int(observed);proposed=int(proposed);point=nearest[row];guard=float(guards[row])
        if context['components'][observed]!=context['components'][proposed] or context['materials'][observed]!=context['materials'][proposed]:raise ValueError('Sparse Q1 SELF foreign original domain')
        chosen,bary=source_shading_normal(context['shading'],proposed,point);other,other_bary=source_shading_normal(context['shading'],observed,point)
        selected_dot=float(chosen@low_shading[row]);observed_dot=float(other@low_shading[row]);opposition=float(context['normals'][proposed]@context['normals'][observed])
        a=bary@context['shading']['world'][context['faces'][proposed]];b=other_bary@context['shading']['world'][context['faces'][observed]]
        if (not np.isfinite(np.r_[point,guard,selected_dot,observed_dot,opposition,a,b,low_points[row],low_shading[row],context['normals'][[proposed,observed]].ravel(),ray_cap]).all()
            or np.linalg.norm(low_shading[row])<=0 or np.any(np.linalg.norm(context['normals'][[proposed,observed]],axis=1)<=0) or guard<=0 or ray_cap<=0
            or np.linalg.norm(point-low_points[row])>ray_cap):raise ValueError('Sparse Q1 SELF invalid position/original ray bound')
        if (min(bary.min(),other_bary.min())<=32*np.finfo(np.float32).eps or selected_dot<=0 or observed_dot>0 or opposition>-.99999
            or max(np.linalg.norm(a-point),np.linalg.norm(b-point),np.linalg.norm(a-b))>guard/8):geometric_ineligible=True
        barycentrics.append(bary);details.append(dict(row=row,observed=observed,proposed=proposed,selected_authored_dot=selected_dot,observed_authored_dot=observed_dot,geometric_opposition=opposition))
    # Validate every row first: a later invalid/foreign input is never retryable.
    if geometric_ineligible:raise Q1SelfGeometricIneligible('Sparse Q1 SELF overlap/interior/orientation qualification failed')
    return np.asarray(barycentrics),details


def q1_self_uv(triangles,barycentrics,indices,resolution,nearest,guards):
    """F32 biased texel origin; exact one native-raster candidate, no UV repair."""
    result=[];errors=[];shape=np.array([[-.3,-.3],[.3,-.3],[-.3,.3]])
    for triangle,bary,pixel,point,guard in zip(triangles,barycentrics,indices,nearest,guards):
        center=(np.array([int(pixel)%resolution,int(pixel)//resolution])+np.array([.501,.502],np.float32).astype(float))/resolution
        uv=(center+(shape-bary@shape)/resolution).astype(np.float32);samples=list(triangle_samples(uv,resolution))
        if len(samples)!=1 or samples[0][0]!=pixel:raise ValueError('Sparse Q1 SELF target UV lacks exact one-texel support')
        error=float(np.linalg.norm(samples[0][1].astype(float)@triangle-point))
        if not np.isfinite(error) or error>guard:raise ValueError('Sparse Q1 SELF F32 target cannot preserve original position guard')
        result.append(uv);errors.append(error)
    return np.asarray(result,np.float32),errors


LOW_TRIANGLE_CACHE_BYTES=16*1024**2


def low_triangle_cache(snapshot, *, max_bytes=LOW_TRIANGLE_CACHE_BYTES):
    """Bounded part-local cache; never batch/reassociate transforms.

    None selects the original uncached expressions. Source snapshots are frozen
    by the caller; this cache never survives the current part/packing grid.
    """
    count=len(snapshot['faces']);required=count*(2*9*8+6*8)+2*9*8+4096
    if required>min(max_bytes,LOW_TRIANGLE_CACHE_BYTES):return None
    local=np.empty((count,3,3),np.float64);world=np.empty_like(local)
    matrix=snapshot['matrix']
    for face in range(count):
        triangle=snapshot['vertices'][snapshot['faces'][face]].astype(np.float64)
        local[face]=triangle
        world[face]=triangle@matrix[:3,:3].T+matrix[:3,3]
    local.setflags(write=False);world.setflags(write=False)
    # Lazy basis admission preserves the old rejection order: an unused
    # degenerate face must not fail the pack before its first raster sample.
    basis=np.full((count,2,3),np.nan,np.float64)
    return dict(local=local,world=world,basis=basis,accounted_bytes=required)


def sample_world(snapshot, face, bary, *, local_triangle=None):
    triangle=(snapshot['vertices'][snapshot['faces'][face]].astype(np.float64)
              if local_triangle is None else local_triangle)
    matrix=np.asarray(snapshot['matrix'],np.float64)
    return (np.asarray(bary,np.float64)@triangle)@matrix[:3,:3].T+matrix[:3,3]


def world_frame(snapshot, face, bary):
    frame=_frame(snapshot,snapshot['triangle_loops'][face],bary)
    return np.linalg.inv(np.asarray(snapshot['matrix'],np.float64)[:3,:3].T)@frame


def decode_native_owners(rgba, face_counts, resolution, *, part):
    """Decode native SELF FACE/CORNER attributes, never select owners on CPU.

    R is a flat global triangle ID+1 and the positive geometry coverage tag;
    G=w0, B=-w1. Alpha is intentionally irrelevant. Signed native corner
    weights are not geometric point-in-triangle tests and are never clamped.
    """
    values=np.asarray(rgba);counts=np.asarray(face_counts)
    if (type(resolution) is not int or resolution<=0 or resolution*resolution>MAX_ATLAS_PIXELS
        or values.dtype!=np.float32 or values.shape!=(resolution*resolution,4)):
        raise ValueError('Native owner image shape/type or atlas resource bound exceeded')
    grouped=(counts.ndim==1 and type(part) is tuple and 2<=len(part)<=128 and all(type(p) is int for p in part)
             and len(set(part))==len(part) and all(0<=p<len(counts) for p in part))
    if (counts.ndim!=1 or not np.issubdtype(counts.dtype,np.integer) or len(counts)==0
        or np.any(counts<0) or counts.astype(object).sum()>=2**24
        or not (grouped or type(part) is int and 0<=part<len(counts))):
        raise ValueError('Native owner part/face counts or exact FLOAT ID bound exceeded')
    if not np.isfinite(values).all():raise ValueError('Nonfinite native owner encoding')
    tags=values[:,0]
    if np.any(tags<0):raise ValueError('Negative native geometry coverage tag')
    covered=tags>0;indices=np.flatnonzero(covered)
    if len(indices)>MAX_PROXY_SAMPLES:raise ValueError('Native owner occupied sample resource bound exceeded')
    if np.any(values[~covered,1:3]!=0):raise ValueError('Native weights without positive geometry coverage tag')
    ids=tags[indices]
    if np.any(ids!=np.floor(ids)):raise ValueError('Fractional or ambiguous native FACE ID')
    offsets=np.r_[0,np.cumsum(counts,dtype=np.int64)]
    if grouped:
        if np.any(ids<1) or np.any(ids>offsets[-1]):raise ValueError('Native FACE ID outside the baked group range')
        global_faces=ids.astype(np.int64)-1
        observed_parts=np.searchsorted(offsets[1:],global_faces,side='right').astype(np.int32)
        if not np.isin(observed_parts,part).all():raise ValueError('Native FACE ID outside the baked group range')
    else:
        if np.any(ids<offsets[part]+1) or np.any(ids>offsets[part+1]):
            raise ValueError('Native FACE ID outside the baked own-part range')
        global_faces=ids.astype(np.int64)-1
        observed_parts=np.full(len(indices),part,np.int32)
    weights=np.empty((len(indices),3),np.float32);weights[:,0]=values[indices,1];weights[:,1]=-values[indices,2]
    # Pinned ordered corners: w2=fl(fl(1-w0)-w1). No renormalization.
    with np.errstate(over='ignore',invalid='ignore'):
        intermediate=np.float32(1)-weights[:,0];weights[:,2]=intermediate-weights[:,1]
        # Two correctly rounded F32 subtractions: absolute error <= half an
        # ULP at each result. F64 exactly sums these three supported F32 terms.
        bound=(np.abs(np.spacing(intermediate)).astype(float)+np.abs(np.spacing(weights[:,2])).astype(float))*.5
        residual=np.abs(weights.astype(float).sum(axis=1)-1.)
    if (not np.isfinite(weights).all() or not np.isfinite(bound).all() or np.any(bound>=1)
        or np.any(residual>bound)):
        raise ValueError('Native corner weights fail finite/unit-sum representability')
    return dict(indices=indices,parts=observed_parts,
                faces=(global_faces-offsets[observed_parts]).astype(np.int32),weights=weights,
                negative_weight_samples=int(np.any(weights<0,axis=1).sum()),
                maximum_unit_sum_residual=float(residual.max(initial=0)),
                signed_transport_observed=bool(np.any(values[indices,2]<0)),
                HDR_FACE_ID_observed=bool(np.any(ids>1)))


# Optional ownership batching has independent small scratch/work bounds.
_OWNER_BATCH_PIXELS=4_194_304
_OWNER_BATCH_PARTS=128
_OWNER_BATCH_TRIANGLES=131072
_OWNER_BATCH_RECTANGLE_PIXELS=8_388_608
_OWNER_BATCH_PIXEL_WORK=256*1024**2


def _owner_batch_domain(rectangles, resolution, out):
    out.fill(False)
    for x0,y0,x1,y1 in rectangles:out[y0:y1+1,x0:x1+1]=True


def native_owner_batch_plan(raw_frames, resolution, *, cpu, margin):
    """One optional disjoint SELF group, never a CPU ownership selection.

    Two reusable pixel masks, bounded rectangle rows, no parts-times-atlas cache.
    Unproved parts retain the existing independent native bake.
    """
    if (cpu is not True or type(margin) is not int or margin!=0 or type(resolution) is not int
            or resolution<=0 or resolution*resolution>_OWNER_BATCH_PIXELS or not 8<=len(raw_frames)<=1024):return None
    buckets={}
    for pi,raw in enumerate(raw_frames):
        if any(name not in raw or not isinstance(raw[name],np.ndarray) for name in ('matrix','vertices','faces','triangle_loops','uv','loop_vertices','smooth','face_n')):continue
        matrix=raw['matrix']
        if matrix.dtype!=np.float64 or matrix.shape!=(4,4) or not np.isfinite(matrix).all():continue
        buckets.setdefault(matrix.tobytes(),[]).append(pi)
    if not buckets:return None
    candidates=max(buckets.values(),key=len)
    if len(candidates)<8:return None
    union=np.zeros((resolution,resolution),bool);scratch=np.zeros_like(union)
    selected=[];rectangles={};triangles=rectangle_pixels=pixel_work=0
    for pi in sorted(candidates,key=lambda i:(len(raw_frames[i]['faces']),i))[:_OWNER_BATCH_PARTS]:
        raw=raw_frames[pi];v=raw['vertices'];f=raw['faces'];loops=raw['triangle_loops'];uv=raw['uv'];lv=raw['loop_vertices']
        if (v.dtype!=np.float32 or v.ndim!=2 or v.shape[1:]!=(3,) or not np.isfinite(v).all()
                or f.dtype!=np.int32 or f.ndim!=2 or f.shape[1:]!=(3,) or not len(f) or len(v)>3*len(f)
                or loops.dtype!=np.int32 or loops.shape!=f.shape or uv.dtype!=np.float32 or uv.shape!=(len(f)*3,2)
                or raw['smooth'].dtype!=bool or raw['smooth'].shape!=(len(f),)
                or raw['face_n'].dtype!=np.float32 or raw['face_n'].shape!=(len(f),3)
                or lv.dtype!=np.int32 or lv.shape!=(len(uv),) or np.any(f<0) or np.any(f>=len(v))
                or not np.array_equal(np.sort(loops.ravel()),np.arange(len(uv))) or not np.array_equal(lv[loops],f)
                or not np.isfinite(uv).all() or np.any(uv<0) or np.any(uv>1)):continue
        if triangles+len(f)>_OWNER_BATCH_TRIANGLES:continue
        t=uv[loops];lower=np.maximum(0,np.floor(t.min(axis=1).astype(np.float64)*resolution).astype(np.int32)-1)
        upper=np.minimum(resolution-1,np.ceil(t.max(axis=1).astype(np.float64)*resolution).astype(np.int32)+1)
        rect=np.column_stack((lower,upper));work=int(np.prod((upper-lower+1).astype(np.int64),axis=1).sum())
        if rectangle_pixels+work>_OWNER_BATCH_RECTANGLE_PIXELS or pixel_work+2*resolution*resolution>_OWNER_BATCH_PIXEL_WORK:break
        rectangle_pixels+=work;pixel_work+=2*resolution*resolution
        _owner_batch_domain(rect,resolution,scratch)
        if np.any(scratch&union):continue
        union|=scratch;rect.setflags(write=False);rectangles[pi]=rect;selected.append(pi);triangles+=len(f)
    if len(selected)<8:return None
    union.setflags(write=False)
    return dict(parts=tuple(sorted(selected)),rectangles=rectangles,domain=union,
                facts=dict(parts=sorted(selected),triangles=triangles,conservative_domain_pixels=int(union.sum()),
                    rectangle_pixel_visits=rectangle_pixels,pixel_work=pixel_work,
                    maximum_pixels=_OWNER_BATCH_PIXELS,maximum_parts=_OWNER_BATCH_PARTS,
                    maximum_triangles=_OWNER_BATCH_TRIANGLES,maximum_rectangle_pixel_visits=_OWNER_BATCH_RECTANGLE_PIXELS,
                    maximum_pixel_work=_OWNER_BATCH_PIXEL_WORK,domain_rule='Closed floor(minUV*resolution)-1 through ceil(maxUV*resolution)+1, clipped; exact pairwise disjoint'))


def decode_native_owner_batch(rgba, face_counts, resolution, plan):
    """Validate the ENTIRE observation before attributing/zeroing any part."""
    decoded=decode_native_owners(rgba,face_counts,resolution,part=plan['parts'])
    if plan['domain'].dtype!=bool or plan['domain'].shape!=(resolution,resolution):raise ValueError('Native batch domain shape/type differs')
    if not plan['domain'].ravel()[decoded['indices']].all():raise ValueError('Native grouped ownership outside proved pixel domain')
    scratch=np.zeros_like(plan['domain']);rows={}
    for pi in plan['parts']:
        indices=decoded['indices'][decoded['parts']==pi]
        _owner_batch_domain(plan['rectangles'][pi],resolution,scratch)
        if not scratch.ravel()[indices].all():raise ValueError('Native grouped ownership outside its own part domain')
        rows[pi]=indices
    return rows


def position_domain(world):
    """Full own-HIGH WORLD bounds plus fixed 1% padding; no geometry guard input."""
    world=np.asarray(world,np.float64)
    if world.ndim!=2 or world.shape[1]!=3 or len(world)==0 or not np.isfinite(world).all():raise ValueError('Invalid full own-HIGH WORLD positions')
    lower=world.min(axis=0);upper=world.max(axis=0)
    scale=max(1.,float(np.abs(world).max()),float(np.ptp(world,axis=0).max()))
    # A constant coordinate still needs a representable encoding interval.
    width=np.maximum(upper-lower,16*float(np.finfo(np.float32).eps)*scale)
    lo=(lower+upper)*.5-width*.51;span=width*1.02
    f=np.float32;wl=np.nextafter(lower.astype(f),f(-np.inf));wh=np.nextafter(upper.astype(f),f(np.inf))
    dl=np.nextafter(wl-lo.astype(f),f(-np.inf));dh=np.nextafter(wh-lo.astype(f),f(np.inf));mult=(.5/span).astype(f)
    minimum=np.nextafter(np.nextafter(dl*mult,f(-np.inf))+f(.25),f(-np.inf));maximum=np.nextafter(np.nextafter(dh*mult,f(np.inf))+f(.25),f(np.inf))
    if not np.isfinite(span).all() or not np.all(minimum>.25) or not np.all(maximum<.75):
        raise ValueError('Full own-HIGH WORLD position encoding is not representable strictly inside .25-.75')
    return lo,span,dict(world_min=lower.tolist(),world_max=upper.tolist(),padding_fraction_each_side=.01,
                        encoded_min=minimum.tolist(),encoded_max=maximum.tolist(),world_scale=scale)


class _FloorCageRejected(ValueError):
    """Only a cage-dependent floor guard may request cell refinement."""


class _FloorCellUnsupported(ValueError):
    def __init__(self, original, reason, record):
        super().__init__(str(original)+'; bounded floor cell refinement unsupported: '+reason)
        self.refinement=dict(record,admitted=False,failure=reason)


def _same_f32(first, second):
    return np.array_equal(first.view(np.uint32),second.view(np.uint32))


def _h_bits(value):return int(np.asarray(value,np.float64).view(np.uint64))
def _h_value(bits):return float(np.asarray(bits,np.uint64).view(np.float64))


def _cell_lower_bits(quantize, upper, target):
    """First positive binary64 input in an actual quantizer cell.

    Each coordinate of F32(V+(delta+signed_normal*h)) is monotone in h.
    Equality to the current cell is therefore a suffix of [1,upper]. All
    comparisons execute that quantizer, including binary64/F32 rounding ties.
    """
    if not _same_f32(quantize(_h_value(upper)),target):raise ValueError('Cell upper quantizer mismatch')
    if _same_f32(quantize(_h_value(1)),target):return 1
    low=1;high=upper
    while high-low>1:
        middle=(low+high)//2
        if _same_f32(quantize(_h_value(middle)),target):high=middle
        else:low=middle
    return high


def _coordinate_transition_cost(first, second):
    def ordered(values):
        bits=values.view(np.uint32)
        return np.where(bits&np.uint32(0x80000000),~bits,bits^np.uint32(0x80000000)).astype(np.int64)
    return int(np.abs(ordered(first)-ordered(second)).sum())


def _refine_floor_cell(vertices, delta, unit_normal, sign, nominal_h, nominal_cage, validate, original, *, transition_limit=256):
    """Explicit bounded search of actual F32 cells, nearest nominal first.

    Transitions count visited coordinate ULP steps, at most256. Prospective
    over-budget edges are not visited; their cost and every quantizer call are
    reported separately. Exhaustion is unsupported, not proof of infeasibility.
    """
    if type(transition_limit) is not int or not 0<=transition_limit<=256:raise ValueError('Invalid floor transition budget')
    record=dict(invoked=True,admitted=False,transitions=0,cells_tested=0,quantizer_evaluations=0,transition_limit=transition_limit)
    def fail(reason):raise _FloorCellUnsupported(original,reason,record)
    def quantize(h):
        record['quantizer_evaluations']+=1
        return (vertices.astype(float)+(delta+unit_normal*(sign*h))).astype(np.float32)
    upper=_h_bits(nominal_h);current=quantize(nominal_h)
    if not np.isfinite(nominal_h) or nominal_h<=0 or not _same_f32(current,nominal_cage):fail('nominal_quantizer_mismatch')
    lower=_cell_lower_bits(quantize,upper,current)
    while lower>1:
        if record['transitions']>=transition_limit:fail('transition_budget_exhausted')
        next_upper=lower-1;candidate=quantize(_h_value(next_upper));cost=_coordinate_transition_cost(current,candidate)
        if cost<=0:fail('cell_boundary_not_distinct')
        if record['transitions']+cost>transition_limit:
            record['next_transition_cost']=cost;fail('transition_budget_exhausted')
        record['transitions']+=cost
        next_lower=_cell_lower_bits(quantize,next_upper,candidate);middle=(next_lower+next_upper)//2;h=_h_value(middle)
        # Verify both actual cell endpoints, the representative, and adjacent
        # outside inputs. No analytically computed F64 boundary is trusted.
        if not all(_same_f32(quantize(_h_value(bits)),candidate) for bits in (next_lower,middle,next_upper)):fail('cell_representative_mismatch')
        if next_lower>1 and _same_f32(quantize(_h_value(next_lower-1)),candidate):fail('cell_lower_neighbor_mismatch')
        if _same_f32(quantize(_h_value(next_upper+1)),candidate):fail('cell_upper_neighbor_mismatch')
        record['cells_tested']+=1
        try:metrics=validate(candidate,delta+unit_normal*(sign*h))
        except _FloorCageRejected:
            current=candidate;lower=next_lower;continue
        except ValueError as exc:fail('independent_guard_rejected: '+str(exc))
        record.update(admitted=True,chosen_h=h,nominal_h=float(nominal_h),cell_lower_h=_h_value(next_lower),cell_upper_h=_h_value(next_upper),cell_lower_bits=next_lower,cell_upper_bits=next_upper,representative_bits=middle,actual_cell_neighbors_verified=True)
        return candidate,metrics,record
    fail('no_positive_admissible_cell')


class _ProxyGeometryRejected(ValueError):
    """Valid proxy inputs whose stored geometry cannot satisfy existing guards."""


class _NominalRayTravelRejected(_ProxyGeometryRejected):
    """Only the nominal source-directed origin exceeded the original ray cap."""


class _AuxiliaryProposalRejected(ValueError):
    """Optional auxiliary admission only; never final native correspondence."""
    def __init__(self, reason, detail):
        super().__init__(detail)
        self.proposal=dict(status='proposal_rejected',reason=reason,detail=detail,nominal_DIRECT_retained=True,PBR_fallback_or_recovery=False)


def _cross3_f64(a,b):
    """Fixed F64 three-vector NumPy cross order; other domains stay generic."""
    if a.shape!=(3,) or b.shape!=(3,) or a.dtype!=np.float64 or b.dtype!=np.float64:
        return np.cross(a,b)
    return np.array((a[1]*b[2]-a[2]*b[1],
                     a[2]*b[0]-a[0]*b[2],
                     a[0]*b[1]-a[1]*b[0]),np.float64)


def _source_face_geometry(triangle):
    T=np.asarray(triangle,float)
    if T.shape!=(3,3) or not np.isfinite(T).all():raise ValueError('Invalid own HIGH triangle')
    n=_cross3_f64(T[1]-T[0],T[2]-T[0]);length=float(np.linalg.norm(n))
    if not length>0:raise ValueError('Degenerate own HIGH triangle')
    n/=length;inward=[];base=[];height=[]
    for i in range(3):
        a,b=T[(i+1)%3],T[(i+2)%3];v=_cross3_f64(n,b-a);v/=np.linalg.norm(v)
        if v@(T[i]-a)<0:v=-v
        inward.append(v);base.append(a);height.append(float(v@(T[i]-a)))
    return T,n,np.array(inward),np.array(base),np.array(height)


def _source_ray_witness(vertices,cage,bary,triangle,nearest,guard, *, _origins=None, details=True):
    T,n,inward,base,_=_source_face_geometry(triangle);minimum=np.inf;maximum=0.;hits=[]
    for i,(a,b,c) in enumerate(((0,1,2),(0,2,1),(1,2,0))):
        if _origins is None:
            def interp(V):return np.add(np.add(V[a]*bary[a],V[b]*bary[b]),V[c]*bary[c]).astype(float)
            q=interp(vertices);o=interp(cage)
        else:q,o=_origins[i]
        ray=q-o;den=float(n@ray)
        if den==0:return None
        t=float(n@(T[0]-o)/den);hit=o+t*ray;clear=np.einsum('ij,ij->i',hit-base,inward);error=float(np.linalg.norm(hit-nearest))
        if hit.dtype==np.float64 and clear.dtype==np.float64 and hit.shape==(3,) and clear.shape==(3,):
            if (not (math.isfinite(t) and math.isfinite(hit[0]) and math.isfinite(hit[1]) and math.isfinite(hit[2])
                     and math.isfinite(clear[0]) and math.isfinite(clear[1]) and math.isfinite(clear[2]) and math.isfinite(error))
                or t<=0 or not (clear[0]>0 and clear[1]>0 and clear[2]>0) or error>guard):return None
        elif not np.isfinite(np.r_[t,hit,clear,error]).all() or t<=0 or not np.all(clear>0) or error>guard:return None
        if details:
            minimum=min(minimum,float(clear.min()));maximum=max(maximum,error);hits.append(hit)
    if not details:return True
    return dict(minimum_analytic_edge_clearance=minimum,maximum_analytic_hit_to_original_Q=maximum,hits=[x.tolist() for x in hits])


def _auxiliary_source_sampler_candidate(point,nearest,triangle,low_triangle,index,resolution,extrusion,ray_cap,guard,margin_ulps):
    """One source-interior auxiliary sample, not a replacement final LOW frame."""
    if type(margin_ulps) is not int or not 1<=margin_ulps<=8:raise ValueError('Invalid auxiliary source-ULP candidate')
    T,n,inward,base,height=_source_face_geometry(triangle);Q=np.asarray(nearest,float);P=np.asarray(point,float)
    # Integer source-coordinate-ULP inward strips avoid choosing a pixel-tuned
    # coefficient. Exact three-order ray admissions remain mandatory below.
    ulp=np.abs(np.spacing(T.astype(np.float32))).astype(float).max(axis=0)
    coordinate_inset=(np.abs(inward)@ulp)*margin_ulps
    lower=coordinate_inset/height;mass=1-float(lower.sum())
    if not mass>0:raise _AuxiliaryProposalRejected('inset_unrepresentable',f'Own HIGH has no representable inset triangle pixel={index}')
    inner=lower@T+mass*T;projected=Q-n*float(n@(Q-T[0]));uv=np.linalg.lstsq((inner[1:]-inner[0]).T,projected-inner[0],rcond=None)[0];weights=np.r_[1-uv.sum(),uv]
    if np.all(weights>=0):aim=weights@inner
    else:
        choices=[]
        for i in range(3):
            a,b=inner[i],inner[(i+1)%3];e=b-a;choices.append(a+np.clip((projected-a)@e/(e@e),0,1)*e)
        aim=min(choices,key=lambda x:np.linalg.norm(x-Q))
    aim_distance=float(np.linalg.norm(aim-Q))
    if not np.isfinite(aim).all() or aim_distance>guard/8:raise _AuxiliaryProposalRejected('aim_outside_original_bound',f'Own HIGH interior aim exceeds original guard/8 pixel={index} distance={aim_distance}')
    # Reuse only auxiliary geometry construction. The original owner/P/TBN and
    # the outer POSITION comparison target Q are not passed as this center.
    try:V,C,U,nested=proxy_triangle(aim,aim,n,low_triangle,index,resolution,extrusion,ray_cap,guard,zero_distance=True)
    except (_ProxyGeometryRejected,_FloorCageRejected,_FloorCellUnsupported) as exc:
        raise _AuxiliaryProposalRejected('auxiliary_geometry_guard',str(exc)) from exc
    nested.pop('auxiliary_proposal_rejected',None);nested.pop('auxiliary_margin_candidates',None)
    bary=np.array(nested['raster_bary'],np.float32);witness=_source_ray_witness(V,C,bary,T,Q,guard)
    if witness is None:raise _AuxiliaryProposalRejected('strict_own_face_witness_unavailable',f'Auxiliary source sampler has no strict three-order own-face witness pixel={index}')
    low_to_center=float(np.linalg.norm(aim-P));low_to_origins=[];low_to_samples=[]
    for a,b,c in ((0,1,2),(0,2,1),(1,2,0)):
        center=np.add(np.add(V[a]*bary[a],V[b]*bary[b]),V[c]*bary[c]).astype(float)
        origin=np.add(np.add(C[a]*bary[a],C[b]*bary[b]),C[c]*bary[c]).astype(float)
        low_to_samples.append(float(np.linalg.norm(center-P)));low_to_origins.append(float(np.linalg.norm(origin-P)))
    low_to_vertices=float(np.linalg.norm(C.astype(float)-P,axis=1).max());low_to_hit=max(float(np.linalg.norm(np.array(hit)-P)) for hit in witness['hits'])
    if max(low_to_origins+[low_to_vertices])>extrusion or max(low_to_samples+[low_to_center,low_to_hit])>ray_cap:raise _AuxiliaryProposalRejected('original_LOW_relative_caps',f'Auxiliary sampler exceeds original LOW-relative user caps pixel={index}')
    auxiliary=dict(source_exact_geometric_normal=n.tolist(),source_coordinate_ULP_inset=coordinate_inset.tolist(),center=aim.tolist(),anchor=aim.tolist(),center_reference='auxiliary interior Aim, not final LOW',original_LOW=P.tolist(),original_Q=Q.tolist(),aim_displacement=aim_distance,aim_limit=float(guard/8),original_LOW_to_center=low_to_center,maximum_original_LOW_to_sample=max(low_to_samples),maximum_original_LOW_to_origin=max(low_to_origins),maximum_original_LOW_to_cage_vertex=low_to_vertices,maximum_original_LOW_to_analytic_hit=low_to_hit,geometry=nested,**witness)
    fact=dict(nested);fact.update(auxiliary_proposal_rejected=None,strategy='own_HIGH_interior_auxiliary_normal_sampler',precision_floor_reason=None,empty_ball_claim=False,nearest_distance=float(np.linalg.norm(Q-P)),precision_floor_anchor=None,precision_floor_orientation=None,floor_refinement=None,center_reference='auxiliary interior Aim',auxiliary_sampler=auxiliary)
    return V,C,U,fact


def _auxiliary_source_sampler(point,nearest,triangle,low_triangle,index,resolution,extrusion,ray_cap,guard,*,strongest=False):
    """Existing bounded inset family; nominal order is unchanged by default."""
    counts=dict(attempted=0,selected=0,rejected=0,maximum_candidates=8,selected_margin_ULPs=None,rejection_reasons=[])
    if strongest:counts['selection_order']='descending_existing_8_to_1'
    for margin in (range(8,0,-1) if strongest else range(1,9)):
        counts['attempted']+=1
        try:
            V,C,U,fact=_auxiliary_source_sampler_candidate(point,nearest,triangle,low_triangle,index,resolution,extrusion,ray_cap,guard,margin)
        except _AuxiliaryProposalRejected as exc:
            counts['rejected']+=1;counts['rejection_reasons'].append(dict(margin_ULPs=margin,reason=exc.proposal['reason'],detail=exc.proposal['detail']))
            continue
        counts.update(selected=1,selected_margin_ULPs=margin);fact['auxiliary_margin_candidates']=counts
        return V,C,U,fact
    message=(f'All8..1 source-ULP auxiliary proposals rejected pixel={index}; original nominal sampler retained for native acceptance' if strongest
             else f'All1..8 source-ULP auxiliary proposals rejected pixel={index}; original nominal DIRECT retained for native acceptance')
    exc=_AuxiliaryProposalRejected('bounded_source_ULP_candidates_exhausted',message)
    exc.proposal['margin_candidates']=counts
    raise exc


class _AuxiliaryWorkLimit(RuntimeError):
    pass


def admit_oriented_auxiliary(vertices,cages,uv,fact,point,nearest,triangle,low_triangle,index,resolution,extrusion,ray_cap,guard,first_hit,expected_face,budget,*,seed_from_Q=False):
    """Full-own-HIGH first-hit admission; native POSITION/FACE remain authority.

    Only Q1 callers use this. Existing admitted bytes are returned unchanged.
    Explicit Q1 seeding skips a nonexistent auxiliary's fast path, not its
    cumulative fallback work budget or any geometric/native acceptance gate.
    Fallback is a finite witnessed search, not a monotonic visibility oracle.
    """
    P=np.asarray(point,float);Q=np.asarray(nearest,float);T,n,_,_,_=_source_face_geometry(triangle)
    fact=dict(fact);original=(vertices,cages,uv,fact);auxiliary=fact['auxiliary_sampler']
    if seed_from_Q:
        preference=fact.get('oriented_preferred_auxiliary') or {}
        if auxiliary is not None or preference.get('status')!='rejected' or preference.get('reason')!='bounded_source_ULP_candidates_exhausted':
            raise ValueError('Q1 seed requires exhausted preferred auxiliary construction')
        Aim=Q.copy();bary=np.asarray(fact['raster_bary'],np.float32)
        budget['seeded_samples']=budget.get('seeded_samples',0)+1
        if budget['seeded_samples']>65536+32:raise ValueError('Oriented Q1-seed row bound exceeded')
    else:
        Aim=np.asarray(auxiliary['center'],float);bary=np.asarray(auxiliary['geometry']['raster_bary'],np.float32)
        budget['fast_samples']+=1
        if budget['fast_samples']>65536+32:raise ValueError('Oriented auxiliary fast-check row bound exceeded')
    if not isinstance(expected_face,(int,np.integer)) or expected_face<0:raise ValueError('Invalid expected own-HIGH source face')
    if np.linalg.norm(Aim-Q)>guard/8:raise ValueError('Initial auxiliary Aim exceeds original guard/8')
    active=False;started=None
    def work():
        if active and (budget['rays']>=budget['max_rays'] or budget['work_seconds']+time.monotonic()-started>budget['max_seconds']):
            raise _AuxiliaryWorkLimit('first_hit_work_budget')
    def validate(V,C,center):
        work();maximum=float(np.linalg.norm(C.astype(float)-V.astype(float),axis=1).max())
        if maximum>min(extrusion,guard/4):raise _FloorCageRejected('Auxiliary stored offset exceeds original cap')
        witness=_source_ray_witness(V,C,bary,T,Q,guard)
        if witness is None:raise _FloorCageRejected('Auxiliary strict three-order source witness unavailable')
        gamma=5*float(np.finfo(np.float32).eps)/(1-5*float(np.finfo(np.float32).eps))
        bound=float(np.linalg.norm(bary.astype(float)@(np.abs(np.spacing(V)).astype(float)*.5)+gamma*(np.abs(V.astype(float))*bary.astype(float)[:,None]).sum(axis=0)))
        own=True;faces=[];origins=[];samples=[]
        for a,b,c in ((0,1,2),(0,2,1),(1,2,0)):
            interp=lambda x:np.add(np.add(x[a]*bary[a],x[b]*bary[b]),x[c]*bary[c]).astype(float)
            sample=interp(V);origin=interp(C);d=sample-origin;separation=float(np.linalg.norm(d))
            if (np.linalg.norm(sample-center)>min(bound,guard/4) or not 0<separation<=min(extrusion,guard/4)
                or np.linalg.norm(center-origin)>ray_cap or np.linalg.norm(origin-P)>extrusion or np.linalg.norm(sample-P)>ray_cap):
                raise _FloorCageRejected('Auxiliary original center/origin/LOW caps failed')
            work()
            if active:budget['rays']+=1
            else:budget['fast_rays']+=1
            hit,_,face,distance=first_hit(origin,d/separation,ray_cap)
            if face is None:
                if hit is not None or distance is not None:raise ValueError('Inconsistent own-HIGH ray miss')
                own=False;faces.append(None)
            else:
                hit=np.asarray(hit,float)
                if hit.shape!=(3,) or not np.isfinite(hit).all() or not isinstance(face,(int,np.integer)) or face<0 or distance is None or not np.isfinite(distance) or distance<0:
                    raise ValueError('Invalid own-HIGH first-hit result')
                faces.append(int(face));own&=face==expected_face and np.linalg.norm(hit-Q)<=guard and np.linalg.norm(hit-P)<=ray_cap
            origins.append(origin);samples.append(sample)
        if np.linalg.norm(C.astype(float)-P,axis=1).max()>extrusion or max(np.linalg.norm(np.asarray(h)-P) for h in witness['hits'])>ray_cap:
            raise _FloorCageRejected('Auxiliary original LOW cage/hit caps failed')
        if not own:raise _FloorCageRejected('Full own-HIGH first hit differs from selected source face/guards')
        return dict(first_hit_faces=faces,maximum_center_error=float(np.linalg.norm(np.asarray(samples)-center,axis=1).max()),F32_center_roundoff_bound=bound,maximum_offset=maximum,maximum_origin_separation=float(np.linalg.norm(np.asarray(origins)-np.asarray(samples),axis=1).max()),
                    maximum_original_LOW_to_cage_vertex=float(np.linalg.norm(C.astype(float)-P,axis=1).max()),
                    maximum_original_LOW_to_analytic_hit=max(float(np.linalg.norm(np.asarray(h)-P)) for h in witness['hits']),
                    maximum_original_LOW_to_origin=float(np.linalg.norm(np.asarray(origins)-P,axis=1).max()),
                    maximum_original_LOW_to_sample=float(np.linalg.norm(np.asarray(samples)-P,axis=1).max()),**witness)
    if not seed_from_Q:
        try:validate(vertices,cages,Aim)
        except _FloorCageRejected:pass
        else:
            budget['unchanged']+=1
            fact['own_HIGH_first_hit']=dict(status='unchanged_admitted',texel_fill=False)
            return original
    def refuse(reason):
        budget['refused']+=1;budget['refusal_reasons'][reason]=budget['refusal_reasons'].get(reason,0)+1
        fact['own_HIGH_first_hit']=dict(status='not_admitted_original_retained_for_native_gates',reason=reason)
        return original
    if budget['rows']>=budget['max_rows']:return refuse('fallback_row_budget')
    budget['rows']+=1;started=time.monotonic();active=True
    nominal=3*guard/32;old_Aim=Aim.copy();lengths=np.linalg.norm(np.roll(T,-1,axis=0)-np.roll(T,-2,axis=0),axis=1)
    incenter=lengths@T/lengths.sum();direction=incenter-old_Aim;length=float(np.linalg.norm(direction));limit=guard/8
    if not length>0:
        budget['work_seconds']+=time.monotonic()-started
        return refuse('no_incenter_direction')
    direction/=length;offset=old_Aim-Q;dot=float(offset@direction)
    upper=min(length,-dot+np.sqrt(dot*dot+limit*limit-float(offset@offset)));lower=0.
    if not np.isfinite(upper) or upper<=0:
        budget['work_seconds']+=time.monotonic()-started
        return refuse('no_incenter_ball_step')
    if np.linalg.norm(old_Aim+direction*upper-Q)<=limit:lower=upper
    else:
        for _ in range(64):
            middle=lower+(upper-lower)*.5
            if middle==lower or middle==upper:break
            if np.linalg.norm(old_Aim+direction*middle-Q)<=limit:lower=middle
            else:upper=middle
    seen=set();last=None;chosen=None
    try:
        for exponent in range(8,-1,-1):
            work();center=old_Aim+direction*(lower/(2**exponent))
            if np.linalg.norm(center-Q)>limit:continue
            try:V,U,next_bary=_proxy_sample_geometry(center,low_triangle,index,resolution,guard)
            except _ProxyGeometryRejected:continue
            if next_bary.tobytes()!=bary.tobytes():raise ValueError('Auxiliary first-hit construction changed paint barycentrics')
            if U.tobytes()!=uv.tobytes():raise ValueError('Auxiliary first-hit construction changed UV')
            if V.tobytes() in seen:continue
            seen.add(V.tobytes());height_seen=set();last=(V,U,center)
            for k in range(32):
                work();h=nominal/(2**k);C=(V.astype(float)+n*h).astype(np.float32)
                if _same_f32(C,V):break
                if C.tobytes() in height_seen:continue
                height_seen.add(C.tobytes());budget['height_seeds']+=1
                try:metrics=validate(V,C,center)
                except _FloorCageRejected:continue
                chosen=(V,C,U,center,h,1,metrics);break
            if chosen is not None:break
        # ONE allowance per row, after the finite seeds, on the largest
        # distinct admitted Aim. No first-hit monotonicity is assumed.
        if chosen is None and last is not None:
            V,U,center=last;C=(V.astype(float)+n*nominal).astype(np.float32)
            try:
                C,metrics,cell=_refine_floor_cell(V,np.zeros(3),n,1,nominal,C,lambda candidate,offset:validate(V,candidate,center),_FloorCageRejected('Finite height seeds exhausted'))
                budget['cell_transition_charge']+=cell['transitions'];chosen=(V,C,U,center,cell['chosen_h'],2,metrics)
            except _FloorCellUnsupported as error:budget['cell_transition_charge']+=error.refinement['transitions']
            except _AuxiliaryWorkLimit:
                budget['cell_transition_charge']+=256
                budget['interrupted_cell_allowances']=budget.get('interrupted_cell_allowances',0)+1
                raise
    except _AuxiliaryWorkLimit as error:return refuse(str(error))
    finally:budget['work_seconds']+=time.monotonic()-started
    if chosen is None:return refuse('finite_Aim_height_family_exhausted')
    V,C,U,center,h,method,metrics=chosen;result=dict(fact)
    result.update(maximum_center_error=metrics['maximum_center_error'],F32_center_roundoff_bound=metrics['F32_center_roundoff_bound'],
                  maximum_offset=metrics['maximum_offset'],maximum_origin_separation=metrics['maximum_origin_separation'],
                  constant_offset_quantization_error=float(np.linalg.norm(C.astype(float)-V.astype(float)-n*h,axis=1).max()))
    if seed_from_Q:
        # No integer-ULP inset was admitted. Do not invent a selected margin or
        # reuse the rejected DIRECT cage as an auxiliary geometric witness.
        auxiliary=dict(source_coordinate_ULP_inset=None,center_reference='auxiliary interior Aim, not final LOW',
                       original_LOW=P.tolist(),original_Q=Q.tolist(),aim_limit=float(guard/8),geometry=dict(raster_bary=bary.tolist()))
        result.update(strategy='own_HIGH_interior_auxiliary_normal_sampler',precision_floor_reason=None,empty_ball_claim=False,
                      empty_ball_radius=0.,center_reference='auxiliary interior Aim',precision_floor_anchor=None,
                      precision_floor_orientation=None,floor_refinement=None)
    result['auxiliary_sampler']=dict(auxiliary,center=center.tolist(),anchor=center.tolist(),aim_displacement=float(np.linalg.norm(center-Q)),
                                     original_LOW_to_center=float(np.linalg.norm(center-P)),
                                     geometry=dict(auxiliary['geometry'],maximum_center_error=metrics['maximum_center_error'],F32_center_roundoff_bound=metrics['F32_center_roundoff_bound'],
                                                   maximum_offset=metrics['maximum_offset'],maximum_origin_separation=metrics['maximum_origin_separation'],
                                                   constant_offset_quantization_error=result['constant_offset_quantization_error'],
                                                   floor_refinement=dict(method='bounded own-HIGH first-hit construction',chosen_h=h,nominal_h=nominal)),
                                     source_exact_geometric_normal=n.tolist(),**metrics)
    result['own_HIGH_first_hit']=dict(status='changed',method=method,Aim=center.tolist(),height=h,texel_fill=False,
                                      family='bounded incenter Aim/dyadic heights; optional ONE256-cell allowance')
    if seed_from_Q:result['own_HIGH_first_hit']['seed_reference']='selected_Q1_no_auxiliary_admitted'
    budget['changed']+=1
    return V,C,U,result


def position_sampler_change_reason(current_vertices,current_cages,current_uv,vertices,cages,uv):
    """Refuse UV mutation or a byte-identical strongest candidate; no input writes."""
    for old,new in ((current_vertices,vertices),(current_cages,cages),(current_uv,uv)):
        if old.dtype!=new.dtype or old.shape!=new.shape or not np.isfinite(new).all():raise ValueError('Invalid sampler change storage')
    if current_uv.tobytes()!=uv.tobytes():return 'candidate_UV_changed'
    if current_vertices.tobytes()==vertices.tobytes() and current_cages.tobytes()==cages.tobytes():return 'strongest_admissible_candidate_makes_no_progress'
    return None


def position_refinement_rows(indices,nonclear_mask,encoded_mask,accepted,used):
    """Admit only a bounded missing-only native set with already-valid present hits."""
    indices=np.asarray(indices);nonclear=np.asarray(nonclear_mask);encoded=np.asarray(encoded_mask);accepted=np.asarray(accepted)
    if (indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer) or nonclear.ndim!=1 or encoded.shape!=nonclear.shape
        or nonclear.dtype!=np.bool_ or encoded.dtype!=np.bool_ or accepted.dtype!=np.bool_ or accepted.shape!=indices.shape
        or np.any(indices<0) or np.any(indices>=len(nonclear)) or np.any(np.diff(indices)<=0)
        or type(used) is not int or not 0<=used<=32):raise ValueError('Invalid native POSITION refinement inputs/budget')
    present=nonclear[indices];rows=np.flatnonzero(~present)
    if not np.array_equal(nonclear,encoded):return rows,'nonencoded_present_pixels'
    if int(nonclear.sum())!=int(present.sum()):return rows,'extra_native_pixels'
    if not np.all(accepted[present]):return rows,'present_original_Q_or_LOW_cap_failed'
    if not len(rows):return rows,'not_needed'
    if len(rows)>32 or used+len(rows)>32:return rows,'grid_refinement_budget_exceeded'
    return rows,'eligible'


def _low_proxy_basis(low_triangle,index):
    """Original scalar LOW basis expression/order, shared with lazy cache."""
    triangle=np.asarray(low_triangle,np.float64);edge=triangle[1]-triangle[0];normal=_cross3_f64(edge,triangle[2]-triangle[0])
    if not np.linalg.norm(edge)>0 or not np.linalg.norm(normal)>0:raise ValueError(f'Degenerate LOW face for proxy pixel={index}')
    edge/=np.linalg.norm(edge);normal/=np.linalg.norm(normal);across=_cross3_f64(normal,edge)
    return edge,across


# Invariant proxy expressions use the same NumPy F32/F64 operation sequence.
_PROXY_PAINT_ORIGIN=np.add(np.float32(.5),np.array([.001,.002],np.float32)).astype(float)
_PROXY_SQRT3=np.sqrt(3)
_PROXY_UV_SHAPE=np.array([[0.,.5],[-_PROXY_SQRT3*.25,-.25],[_PROXY_SQRT3*.25,-.25]])


def _proxy_paint_batch(indices,resolution):
    """Bounded internal paint raster; uncertain rows retain the scalar router.

    Each f-operation promotes binary32 operands to binary64 exactly as the
    scalar Python operation does, then immediately rounds back to binary32.
    No scan arithmetic is reassociated, and there is no cross-call cache.
    """
    indices=np.asarray(indices)
    if indices.ndim!=1 or len(indices)>2048:raise ValueError('Paint batch exceeds bounded row shape')
    fallback=[None]*len(indices)
    if (type(resolution) is not int or resolution not in (256,512,1024,2048,4096)
            or not np.issubdtype(indices.dtype,np.integer)
            or np.any(indices<0) or np.any(indices>=resolution*resolution)):
        return fallback
    from production_tangents import _PAINT_BIAS, _EPS
    def fadd(a,b):return np.add(np.asarray(a,np.float64),np.asarray(b,np.float64)).astype(np.float32)
    def fsub(a,b):return np.subtract(np.asarray(a,np.float64),np.asarray(b,np.float64)).astype(np.float32)
    def fmul(a,b):return np.multiply(np.asarray(a,np.float64),np.asarray(b,np.float64)).astype(np.float32)
    def fdiv(a,b):return np.divide(np.asarray(a,np.float64),np.asarray(b,np.float64)).astype(np.float32)
    x=indices%resolution;y=indices//resolution
    center=np.stack((x,y),axis=1)+_PROXY_PAINT_ORIGIN
    uv=((center[:,None,:]+_PROXY_UV_SHAPE)/resolution).astype(np.float32)
    v=fsub(fmul(uv,float(resolution)),np.asarray(_PAINT_BIAS))
    a,b,c=v[:,0],v[:,1],v[:,2]
    # _scan sees edges B->A and C->A, with no horizontal B->C span.
    delta=fsub(a[:,1],b[:,1]);delta2=fsub(a[:,1],c[:,1])
    good=(np.isfinite(v).all(axis=(1,2)) & (b[:,1].view(np.uint32)==c[:,1].view(np.uint32))
          & (delta>_EPS) & (delta2>_EPS) & (b[:,0]<a[:,0]) & (a[:,0]<c[:,0])
          & (np.ceil(b[:,1])==y) & (np.floor(a[:,1])==y)
          & (np.ceil(c[:,1])==y) & (np.ceil(b[:,1])>np.floor(b[:,1])))
    if not np.any(good):return fallback
    # Invalid/uncertain rows never enter division; their scalar route is intact.
    rows=np.flatnonzero(good);a=a[rows];b=b[rows];c=c[rows];xx=x[rows];yy=y[rows]
    delta=delta[rows];delta2=delta2[rows]
    left_span=fadd(fmul(fdiv(fsub(b[:,0],a[:,0]),delta),fsub(b[:,1],yy)),b[:,0])
    right_span=fadd(fmul(fdiv(fsub(c[:,0],a[:,0]),delta2),fsub(c[:,1],yy)),c[:,0])
    left=np.maximum(np.floor(np.minimum(left_span,right_span))+1,0)
    right=np.minimum(np.floor(np.maximum(left_span,right_span)),resolution-1)
    good=(left==xx)&(right==xx)
    x1=fsub(a[:,0],b[:,0]);x2=fsub(b[:,0],c[:,0]);y1=fsub(a[:,1],b[:,1]);y2=fsub(b[:,1],c[:,1])
    z0=fsub(fmul(x1,y2),fmul(y1,x2));good&=np.isfinite(z0)&(z0!=0)
    # A zero/uncertain determinant keeps the ordinary scanner and its errors.
    z0=np.where(good,z0,np.float32(1))
    x0=fsub(fmul(y1,0.),fmul(1.,y2));y0=fsub(fmul(1.,x2),fmul(x1,0.))
    intercept=fadd(fdiv(fadd(fmul(x0,a[:,0]),fmul(y0,a[:,1])),z0),1.)
    uxd=(-x0.astype(np.float64)/z0.astype(np.float64)).astype(np.float32)
    uyd=(-y0.astype(np.float64)/z0.astype(np.float64)).astype(np.float32)
    uy0=(yy.astype(np.float64)*uyd.astype(np.float64)+intercept.astype(np.float64)).astype(np.float32)
    x0=fsub(fmul(y1,1.),fmul(-1.,y2));y0=fsub(fmul(-1.,x2),fmul(x1,1.))
    intercept=fdiv(fadd(fmul(x0,a[:,0]),fmul(y0,a[:,1])),z0)
    vxd=(-x0.astype(np.float64)/z0.astype(np.float64)).astype(np.float32)
    vyd=(-y0.astype(np.float64)/z0.astype(np.float64)).astype(np.float32)
    vy0=(yy.astype(np.float64)*vyd.astype(np.float64)+intercept.astype(np.float64)).astype(np.float32)
    u=((left.astype(np.float64)*uxd.astype(np.float64)+uy0.astype(np.float64))
       -fmul(np.float32(0),uyd).astype(np.float64)).astype(np.float32)
    w=((left.astype(np.float64)*vxd.astype(np.float64)+vy0.astype(np.float64))
       -fmul(np.float32(0),vyd).astype(np.float64)).astype(np.float32)
    u=fadd(u,fmul(np.float32(0),uxd));w=fadd(w,fmul(np.float32(0),vxd))
    bary=np.stack((u,w,fsub(fsub(1.,u),w)),axis=1)
    good&=np.isfinite(bary).all(axis=1)&np.all(bary>0,axis=1)
    for local,row in enumerate(rows):
        if good[local]:fallback[int(row)]=(int(indices[row]),resolution,uv[row].copy(),bary[local].copy())
    return fallback


def _proxy_sample_geometry(point,low_triangle,index,resolution,guard,*,low_basis=None,paint_sample=None):
    """Shared paint V/UV/bary arithmetic only; no cage construction/refinement."""
    p=np.asarray(point,np.float64)
    if paint_sample is None:
        center=np.array([index%resolution,index//resolution])+_PROXY_PAINT_ORIGIN
        uv=((center+_PROXY_UV_SHAPE)/resolution).astype(np.float32)
        samples=list(triangle_samples(uv,resolution))
        if len(samples)!=1 or samples[0][0]!=index or not np.all(samples[0][1]>0):raise ValueError(f'Proxy has not exactly one strict sample at pixel={index}')
        bary=samples[0][1]
    else:
        cached_index,cached_resolution,uv,bary=paint_sample
        if cached_index!=index or cached_resolution!=resolution:raise ValueError('Paint batch row binding differs')
    if low_basis is None:edge,across=_low_proxy_basis(low_triangle,index)
    else:
        if np.isnan(low_basis[0,0]):low_basis[:]=_low_proxy_basis(low_triangle,index)
        edge,across=low_basis
    ulp=np.abs(np.spacing(p.astype(np.float32))).astype(float)
    radius=max(8*float(ulp.max()),guard/16)
    offsets=np.array([[0.,2*radius],[-_PROXY_SQRT3*radius,-radius],[_PROXY_SQRT3*radius,-radius]])@np.stack((edge,across))
    offsets-=bary.astype(float)@offsets;vertices=(p+offsets).astype(np.float32)
    if not np.linalg.norm(_cross3_f64(vertices[1].astype(float)-vertices[0],vertices[2].astype(float)-vertices[0]))>0:
        raise _ProxyGeometryRejected(f'Physical proxy collapsed at pixel={index}')
    return vertices,uv,bary


def proxy_triangle(point, nearest, source_geometric_normal, low_triangle, index, resolution, extrusion, ray_cap, guard, *, zero_distance=False, source_triangle=None, original_nearest_distance=None, first_hit=None, expected_source_face=None, auxiliary_budget=None, allow_q1_seed=False, low_basis=None, _paint_sample=None):
    """Nominal proxy first; explicit bounded floor-only F32-cell admission."""
    p=np.asarray(point,np.float64);nearest=np.asarray(nearest,np.float64);delta=nearest-p;distance=float(np.linalg.norm(delta))
    ball_distance=distance if original_nearest_distance is None else float(original_nearest_distance)
    if original_nearest_distance is not None and (not np.isfinite(ball_distance) or not 0<=ball_distance<=ray_cap):raise ValueError('Invalid original nearest source distance')
    if not np.isfinite(np.r_[p,nearest,extrusion,ray_cap,guard]).all() or min(extrusion,ray_cap,guard)<=0:
        raise ValueError(f'Invalid/zero proxy caps at pixel={index}')
    if distance>ray_cap:raise ValueError(f'Original LOW-to-nearest HIGH distance exceeds original ray cap pixel={index} distance={distance} cap={ray_cap}')
    preferred_auxiliary=None
    if original_nearest_distance is not None and source_triangle is not None:
        try:
            V,C,U,fact=_auxiliary_source_sampler(p,nearest,source_triangle,low_triangle,index,resolution,extrusion,ray_cap,guard,strongest=True)
        except _AuxiliaryProposalRejected as error:
            preferred_auxiliary=dict(status='rejected',reason=error.proposal['reason'],detail=error.proposal['detail'])
            preferred_rejection=error.proposal
        else:
            fact['oriented_preferred_auxiliary']=dict(status='admitted',original_d0=ball_distance,texel_fill=False)
            if first_hit is not None:
                return admit_oriented_auxiliary(V,C,U,fact,p,nearest,source_triangle,low_triangle,index,resolution,extrusion,ray_cap,guard,first_hit,expected_source_face,auxiliary_budget)
            return V,C,U,fact
    vertices,uv,bary=_proxy_sample_geometry(p,low_triangle,index,resolution,guard,low_basis=low_basis,paint_sample=_paint_sample)
    def origins(cage):
        results=[]
        orders=((0,1,2),(0,2,1),(1,2,0)) if original_nearest_distance is None else ((0,1,2),(0,2,1),(1,2,0),(2,0,1))
        for order in orders:
            def interp(v):
                a,b,c=order
                return np.add(np.add(v[a]*bary[a],v[b]*bary[b]),v[c]*bary[c]).astype(float)
            results.append((interp(vertices),interp(cage)))
        return results
    offset=-delta*(min(extrusion,ball_distance/2)/distance) if distance>0 else np.zeros(3)
    cage=(vertices.astype(float)+offset).astype(np.float32);strategy='source_directed_delta_half';floor_sign=None
    sampled_origins=origins(cage);initial_cage=cage
    initial_separations=[None]*len(sampled_origins)
    def initial_separation(i):
        value=initial_separations[i]
        if value is None:
            q,o=sampled_origins[i];value=np.linalg.norm(o-q);initial_separations[i]=value
        return value
    collapsed=any(not initial_separation(i)>0 for i in range(len(sampled_origins)))
    overshoot=ball_distance>0 and any(initial_separation(i)>=ball_distance for i in range(len(sampled_origins)))
    # With guard=max(.01*delta,16*eps*W), this positive branch can only
    # occur in the precision term: delta<=2*eps*W has unreliable direction.
    near_precision=0<ball_distance<=guard/8
    floor_reason='zero_nearest_distance' if ball_distance==0 or zero_distance else 'F32_collapsed_origin' if collapsed else ('F32_near_precision_overshoot' if overshoot else 'F32_near_precision_distance') if near_precision else None
    if ball_distance==0 or zero_distance or collapsed or near_precision:
        n=np.asarray(source_geometric_normal,np.float64);length=float(np.linalg.norm(n))
        if not np.isfinite(n).all() or not length>0:raise ValueError(f'No valid nearest HIGH geometric normal for precision floor pixel={index}')
        # Choose the nearest-surface side away from LOW. At exact dot zero,
        # retain +1: np.sign(0) would incorrectly erase the precision floor.
        unit_normal=n/length;floor_sign=-1 if float(unit_normal@delta)<0 else 1
        offset=delta+unit_normal*(floor_sign*3*guard/32);cage=(vertices.astype(float)+offset).astype(np.float32);strategy='nearest_geometric_normal_precision_floor'
    validated_origins=None
    def validate(cage, offset):
        nonlocal validated_origins
        actual_offsets=cage.astype(float)-vertices.astype(float);maximum=float(np.linalg.norm(actual_offsets,axis=1).max())
        if maximum>extrusion or (strategy.endswith('precision_floor') and maximum>guard/4):
            raise (_FloorCageRejected if strategy.endswith('precision_floor') else _ProxyGeometryRejected)(f'Stored proxy offset exceeds original cap pixel={index} strategy={strategy} offset={maximum} extrusion={extrusion} guard={guard}')
        max_error=max_roundoff=0.
        gamma5=5*float(np.finfo(np.float32).eps)/(1-5*float(np.finfo(np.float32).eps))
        half_ulp=np.abs(np.spacing(vertices)).astype(float)*.5
        bound=float(np.linalg.norm(bary.astype(float)@half_ulp+gamma5*(np.abs(vertices.astype(float))*bary.astype(float)[:,None]).sum(axis=0)))
        direct_initial=strategy=='source_directed_delta_half' and cage is initial_cage
        validated_origins=sampled_origins if direct_initial else origins(cage)
        for i,(q,o) in enumerate(validated_origins):
            error=float(np.linalg.norm(q-p));separation=float(initial_separations[i]) if direct_initial else float(np.linalg.norm(o-q));travel=float(np.linalg.norm(nearest-o))
            if error>bound or error>guard/4:
                raise _ProxyGeometryRejected(f'Proxy center/origin/ray is unsupported pixel={index} error={error} bound={bound} separation={separation} travel={travel}')
            if not separation>0 or separation>extrusion or travel>ray_cap:
                rejected=_FloorCageRejected if strategy.endswith('precision_floor') else _ProxyGeometryRejected
                if direct_initial and 0<separation<=extrusion and travel>ray_cap:rejected=_NominalRayTravelRejected
                raise rejected(f'Proxy center/origin/ray is unsupported pixel={index} error={error} bound={bound} separation={separation} travel={travel}')
            if strategy=='source_directed_delta_half' and separation>=ball_distance:raise _ProxyGeometryRejected(f'Proxy origin outside source-distance ball pixel={index}')
            if strategy.endswith('precision_floor') and separation>guard/4:raise _FloorCageRejected(f'Precision floor exceeds original guard/4 pixel={index}')
            max_error=max(max_error,error);max_roundoff=max(max_roundoff,separation)
        return dict(maximum_offset=maximum,maximum_center_error=max_error,F32_center_roundoff_bound=bound,maximum_origin_separation=max_roundoff,constant_offset_quantization_error=float(np.linalg.norm(actual_offsets-offset,axis=1).max()))
    refinement=None;ray_cap_refinement=None
    try:metrics=validate(cage,offset)
    except _NominalRayTravelRejected:
        # One Q0-only shorter step; no preferred auxiliary/floor route changes.
        if original_nearest_distance is not None or preferred_auxiliary is not None:raise
        remaining=ray_cap-distance;nominal_h=min(extrusion,ball_distance/2)
        h=min(nominal_h,remaining/2)
        if not 0<h<nominal_h:raise
        offset=-delta*(h/distance);cage=(vertices.astype(float)+offset).astype(np.float32)
        metrics=validate(cage,offset)
        ray_cap_refinement=dict(status='construction_validated',attempts=1,nominal_h=nominal_h,chosen_h=h,
                                remaining_ray_budget=remaining,scope='Original SOURCE/native correspondence checks remain mandatory')
    except _FloorCageRejected as original:
        cage,metrics,refinement=_refine_floor_cell(vertices,delta,unit_normal,floor_sign,3*guard/32,cage,validate,original)
    proposal_rejected=preferred_rejection if preferred_auxiliary is not None else None
    if strategy=='source_directed_delta_half' and source_triangle is not None and preferred_auxiliary is None:
        witness=_source_ray_witness(vertices,cage,bary,source_triangle,nearest,guard,_origins=validated_origins,details=False)
        if witness is None:
            try:return _auxiliary_source_sampler(p,nearest,source_triangle,np.asarray(low_triangle,np.float64),index,resolution,extrusion,ray_cap,guard)
            except _AuxiliaryProposalRejected as exc:proposal_rejected=exc.proposal
    fact=dict(oriented_preferred_auxiliary=preferred_auxiliary,auxiliary_margin_candidates=proposal_rejected.get('margin_candidates') if proposal_rejected else None,auxiliary_proposal_rejected=proposal_rejected,center_reference='original LOW',auxiliary_sampler=None,strategy=strategy,precision_floor_reason=floor_reason,empty_ball_claim=strategy=='source_directed_delta_half',empty_ball_radius=ball_distance,nearest_distance=distance,**metrics,raster_bary=bary.tolist(),precision_floor_anchor='nearest_own_HIGH_point' if strategy.endswith('precision_floor') else None,precision_floor_orientation=floor_sign,floor_refinement=refinement)
    if ray_cap_refinement is not None:fact['source_directed_ray_cap_refinement']=ray_cap_refinement
    if allow_q1_seed and original_nearest_distance is not None and preferred_auxiliary is not None and first_hit is not None:
        return admit_oriented_auxiliary(vertices,cage,uv,fact,p,nearest,source_triangle,low_triangle,index,resolution,extrusion,ray_cap,guard,first_hit,expected_source_face,auxiliary_budget,seed_from_Q=True)
    return vertices,cage,uv,fact


PNG8_VECTOR_ERROR=np.sqrt(3.)/255.
PNG8_WORLD_ERROR_BOUND=2*PNG8_VECTOR_ERROR/(1-PNG8_VECTOR_ERROR)


def _unit_target(v):
    length=np.linalg.norm(v,axis=-1,keepdims=True)
    if not np.isfinite(v).all() or not np.isfinite(length).all() or np.any(length<=0):raise ValueError('Nonfinite/zero Three target-frame vector')
    return v/length


def target_world_frames(snapshot,faces,weights):
    """Three0.171 USE_TANGENT ABI before rigid camera rotation, not bake.cc."""
    corners=snapshot['triangle_loops'][faces];w=np.asarray(weights,np.float32);m=np.asarray(snapshot['matrix'],float)[:3,:3]
    nv=_unit_target(snapshot['n'][corners].astype(float)@np.linalg.inv(m))
    tv=_unit_target(snapshot['t'][corners].astype(float)@m.T)
    bv=_unit_target(np.cross(nv,tv)*snapshot['sign'][corners][...,None])
    def interp(v):return (v[:,0]*w[:,0,None]+v[:,1]*w[:,1,None])+v[:,2]*w[:,2,None]
    return np.stack((_unit_target(interp(tv)),_unit_target(interp(bv)),_unit_target(interp(nv))),axis=-1)


def target_normal_errors(encoded,world,snapshot,faces,weights):
    """Unbiased glTF RGB decoder; PNG8 envelope is separate from RGB32 gate."""
    f32=np.empty(len(faces),float);png=np.empty_like(f32)
    for begin in range(0,len(faces),65536):
        sl=slice(begin,begin+65536);A=target_world_frames(snapshot,faces[sl],weights[sl]);rgb=encoded[sl].astype(float)
        for output,values in ((f32,rgb),(png,np.clip(np.rint(rgb*255),0,255)/255)):
            restored=_unit_target(np.einsum('nij,nj->ni',A,values*2-1));output[sl]=np.linalg.norm(restored-world[sl],axis=1)
    return f32,png


def _encode_target_normals(world,snapshot,faces,weights):
    rgb=np.empty((len(faces),3),np.float32)
    for begin in range(0,len(faces),65536):
        sl=slice(begin,begin+65536);A=target_world_frames(snapshot,faces[sl],weights[sl]);local=_unit_target(np.linalg.solve(A,world[sl,:,None])[...,0]);rgb[sl]=(local*.5+.5).astype(np.float32)
    errors,png=target_normal_errors(rgb,world,snapshot,faces,weights)
    return rgb,errors,png


def repair_target_faces(snapshot,faces):
    """Canonical N/T on explicitly proven offending faces only; no pseudoinverse."""
    result={k:np.asarray(v) for k,v in snapshot.items()};result['n']=snapshot['n'].copy();result['t']=snapshot['t'].copy()
    for face in faces:
        corners=snapshot['triangle_loops'][face];sign=snapshot['sign'][corners]
        if not np.isin(sign,(-1,1)).all() or not np.all(sign==sign[0]):raise ValueError('Target face has inconsistent handedness')
        P=snapshot['vertices'][snapshot['faces'][face]].astype(float);uv=snapshot['uv'][corners]
        n=_unit_target(np.cross(P[1]-P[0],P[2]-P[0])).astype(np.float32);t=analytic_tangent(P,uv,n)
        result['n'][corners]=n;result['t'][corners]=t
    return result


def source_object_world(rgba):
    q=(np.asarray(rgba)[:,:3].astype(float)-.5-NORMAL_BIAS)*2;length=np.linalg.norm(q,axis=1)
    if not np.isfinite(q).all() or np.any(np.abs(length-1)>1e-4):raise ValueError('Expected finite unit native OBJECT normal')
    return q/length[:,None]


def object_normals_to_final(data,indices,snapshot,faces,weights):
    """Re-evaluate cached native WORLD field in frozen Three target ABI."""
    faces=np.asarray(faces);weights=np.asarray(weights,np.float32)
    world=source_object_world(data[indices]);rgb,errors,png=_encode_target_normals(world,snapshot,faces,weights)
    bad32=errors>1e-6;bad8=png>PNG8_WORLD_ERROR_BOUND;affected=np.unique(faces[bad32|bad8]);target=snapshot
    if len(affected):
        if len(affected)==len(snapshot['faces']):raise ValueError('Conditioning repair would replace all target faces; refusing global flat target frame')
        target=repair_target_faces(snapshot,affected);rgb,errors,png=_encode_target_normals(world,target,faces,weights)
    if not np.isfinite(rgb).all() or np.any(errors>1e-6) or np.any(png>PNG8_WORLD_ERROR_BOUND):
        raise ValueError(f'Three target normal repair failed unchanged gates: RGB32={errors.max(initial=0)} PNG8_model={png.max(initial=0)}')
    nc=np.flatnonzero(np.any(snapshot['n'].view(np.uint32)!=target['n'].view(np.uint32),axis=1));tc=np.flatnonzero(np.any(snapshot['t'].view(np.uint32)!=target['t'].view(np.uint32),axis=1))
    data[indices,:3]=rgb
    metrics=dict(valid=True,converted_texels=len(indices),target_ABI='Three0.171 USE_TANGENT: normalize N/T/B per vertex, interpolate B, normalize columns in fragment, RGB*2-1 without bias',source_OBJECT_bias_read=float(NORMAL_BIAS),glTF_RGB_bias=0.,world_roundtrip_limit=1e-6,max_normalized_world_roundtrip_residual=float(errors.max(initial=0)),PNG8_vector_error=float(PNG8_VECTOR_ERROR),PNG8_normalized_world_error_bound=float(PNG8_WORLD_ERROR_BOUND),PNG8_model_max_residual=float(png.max(initial=0)),PNG8_actual_readback='PENDING save/readback, never1e-6claim',pre_repair_RGB32_failed_texels=int(bad32.sum()),pre_repair_PNG8_bound_failed_texels=int(bad8.sum()),conditioning_repair=dict(affected_faces=affected.tolist(),changed_normal_loops=nc.tolist(),changed_tangent_loops=tc.tolist(),positions_faces_UV_matrix_sign_and_other_faces_unchanged=True,method='Geometric N and UV-gradient tangent constant on proven offending faces only; original WORLD field re-encoded; no pseudoinverse'))
    return metrics,target
