# SPDX-License-Identifier: GPL-3.0-or-later
"""Informational bounded reconstruction diagnostics, never geometry admission."""
import time
import numpy as np

MAX_PAIR_FACES=100_000
MAX_PAIR_VERTICES=100_000
MAX_WORK_BYTES=64*1024**2
MAX_SECONDS=5.0  # Shared across all components of one reconstruction.
SEMANTICS='Deterministic bidirectional sampled surface distance, not exact Hausdorff or a visual pass'


def surface_points(world,faces):
    """The existing finishing sampler, including all six axial extrema."""
    sampled=faces[np.linspace(0,len(faces)-1,min(2048,len(faces)),dtype=int)]
    a,b,c=(world[sampled[:,i]] for i in range(3))
    return np.concatenate((a,b,c,(a+b+c)/3,(a+b)/2,(b+c)/2,(c+a)/2,
                           world[np.r_[world.argmin(axis=0),world.argmax(axis=0)]]))


def world_bounds(vertices,matrix,check):
    """Exact vertex AABB reductions with at most 8192 world rows materialized."""
    lower=np.full(3,np.inf);upper=-lower
    for start in range(0,len(vertices),8192):
        check();world=np.asarray(vertices[start:start+8192],np.float64)@matrix[:3,:3].T+matrix[:3,3]
        if not np.isfinite(world).all():raise ValueError('Nonfinite reconstruction diagnostic world positions')
        lower=np.minimum(lower,world.min(0));upper=np.maximum(upper,world.max(0))
    check()
    return dict(min=lower.tolist(),max=upper.tolist())


def bounds_pair(source,output):
    diagonal=float(np.linalg.norm(np.subtract(source['max'],source['min'])))
    shift=max(abs(a-b) for k in ('min','max') for a,b in zip(source[k],output[k]))
    # Reconstruction has no geometry-error UI option. These are explicitly a
    # comparison reference, not an added gate or a new parameter default.
    cap=min(.01,.003*diagonal)
    return dict(source=source,output=output,source_diagonal_world=diagonal,shift_world=shift,
                reference_cap_world=cap,exceeds_reference=shift>cap,
                reference='Finishing .003 relative/.01 world comparison only; not reconstruction admission')


class _DiagnosticSpent(Exception):pass


def component(vertices,faces,out_vertices,out_faces,matrix,check,work,*,local=None,remaining_bytes=MAX_WORK_BYTES,build_bvh=None,clock=time.monotonic):
    """Reuse existing arrays, bounded BVH accounting, no duplicate Blender mesh.

    Operation cancellation, MemoryError and the operation deadline propagate.
    Only this diagnostic's own budget produces NOT_MEASURED.
    """
    check();matrix=np.asarray(matrix,np.float64)
    bounds=bounds_pair(world_bounds(vertices,matrix,check),world_bounds(out_vertices,matrix,check))
    result=dict(bounds=bounds,acceptance='INFORMATIONAL_ONLY',semantics=SEMANTICS,
                distances=dict(status='NOT_MEASURED',reason='not_started'))
    if local is not None:
        if not np.array_equal(vertices,out_vertices[:len(vertices)]) or not np.array_equal(faces,out_faces[:len(faces)]):
            raise ValueError('Local fidelity diagnostic requires proven original geometry prefix')
        result['distances']=dict(status='EXACT_RETAINED_PREFIX',source_to_output_max_world=0.0,
            output_to_source_max_world=None,scope='Original surface is an exact subset; no nearest/authored identity for appended geometry')
        result['synthetic_faces']=dict(count=len(out_faces)-len(faces),first_component_face_id=len(faces),
            source_geometry_sha256=local['source_geometry_sha256'],identity='UNMATCHED_SYNTHETIC; no HIGH/UV/material/authorial correspondence')
        check();return result
    nf=len(faces)+len(out_faces);nv=len(vertices)+len(out_vertices)
    # Only one side's Python conversion, world array and BVH are live at once.
    # Both unchanged sample arrays and conservative stencil temporaries remain
    # included; this is guarded additional working space, not measured RSS.
    sides=[512*(len(f)+len(v))+24*len(v) for v,f in ((vertices,faces),(out_vertices,out_faces))]
    estimated=max(sides)+4*14342*24
    limit=min(MAX_WORK_BYTES,max(0,remaining_bytes))
    result['work']=dict(pair_faces=nf,pair_vertices=nv,estimated_additional_bytes=estimated,
                        working_limit_bytes=limit,shared_seconds=MAX_SECONDS,maximum_live_bvhs=1,
                        active_side_estimated_bytes=sides,sample_and_stencil_allowance_bytes=4*14342*24)
    if nf>MAX_PAIR_FACES or nv>MAX_PAIR_VERTICES or estimated>limit:
        result['distances']['reason']='bounded_triangle_vertex_or_working_byte_limit';check();return result
    started=clock();deadline=started+max(0.0,MAX_SECONDS-work.get('spent_seconds',0.0))
    def diagnostic_check():
        check()  # Never soften the actual operation's cancellation/resources.
        if clock()>=deadline:raise _DiagnosticSpent('shared_diagnostic_time_limit')
    try:
        diagnostic_check()
        vector=lambda point:point
        if build_bvh is None:
            from mathutils import Vector
            from mathutils.bvhtree import BVHTree
            vector=Vector
            build_bvh=lambda world,triangles:BVHTree.FromPolygons(world.tolist(),triangles.tolist(),all_triangles=True)
        samples=[]
        for v,f in ((vertices,faces),(out_vertices,out_faces)):
            diagnostic_check();world=np.asarray(v,np.float64)@matrix[:3,:3].T+matrix[:3,3]
            samples.append(surface_points(world,f));del world;diagnostic_check()
        def maximum(points,tree):
            maximum=0.0
            for index,point in enumerate(points):
                if index%128==0:diagnostic_check()
                hit=tree.find_nearest(vector(point))
                if hit is None or hit[0] is None or not np.isfinite(hit[3]):raise _DiagnosticSpent('nearest_query_unavailable')
                maximum=max(maximum,float(hit[3]))
            diagnostic_check();return maximum
        def directed(points,vertices,faces):
            diagnostic_check();world=np.asarray(vertices,np.float64)@matrix[:3,:3].T+matrix[:3,3];tree=None
            try:
                tree=build_bvh(world,faces);diagnostic_check();return maximum(points,tree)
            finally:
                # Release the native tree before constructing the reverse one,
                # including cancellation, own-budget and allocation failures.
                del tree,world
        forward=directed(samples[0],out_vertices,out_faces)
        backward=directed(samples[1],vertices,faces)
        result['distances']=dict(status='MEASURED',source_to_output_max_sample_world=forward,
            output_to_source_max_sample_world=backward,source_samples=len(samples[0]),output_samples=len(samples[1]),
            exceeds_reference=max(forward,backward)>bounds['reference_cap_world'])
    except _DiagnosticSpent as error:result['distances']=dict(status='NOT_MEASURED',reason=str(error))
    finally:work['spent_seconds']=work.get('spent_seconds',0.0)+max(0.0,clock()-started)
    check();return result


def parent_bounds(records):
    """Aggregate matched component bounds only; no extra BVH or geometry copy."""
    return bounds_pair(*[dict(min=np.min([r['bounds'][side]['min'] for r in records],axis=0).tolist(),
                              max=np.max([r['bounds'][side]['max'] for r in records],axis=0).tolist())
                         for side in ('source','output')])


def notice(records):
    departures=sum(r['bounds']['exceeds_reference'] or r['distances'].get('exceeds_reference',False) for r in records)
    missing=sum(r['distances']['status']=='NOT_MEASURED' for r in records)
    return (f'Reconstruction shape diagnostic: {departures}/{len(records)} components exceed finishing reference; '
            f'{missing} surface-distance checks NOT_MEASURED. Topology success is not detail/visual fidelity.') if departures or missing else ''
