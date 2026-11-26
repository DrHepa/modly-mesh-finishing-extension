# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded exact-interior overlap rejection for stored float32 target UVs."""
import math
import struct


def float32_uv_area(triangle):
    p=[tuple(int(float(v)*(1<<149)) for v in row) for row in triangle]
    a,b,c=p
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def reconcile_uv_vertices(values, groups):
    """Restore exact pre-operator UV-vertex equality within eight float32 ULPs."""
    import numpy as np
    values=np.asarray(values,dtype=np.float32);groups=np.asarray(groups,dtype=np.int32)
    counts=np.bincount(groups)
    means=np.column_stack([np.bincount(groups,weights=values[:,axis])/counts for axis in range(2)]).astype(np.float32)
    fixed=means[groups]
    def ordered_bits(data):
        bits=data.view(np.int32).astype(np.int64)
        return np.where(bits<0,-2147483648-bits,bits)
    ulps=int(np.max(np.abs(ordered_bits(fixed)-ordered_bits(values))))
    if not np.isfinite(fixed).all() or ulps>8:raise ValueError(f'UV operator separated an equivalent vertex by more than eight ULPs: max={ulps}')
    return fixed,dict(changed_uv_values=int(np.count_nonzero(fixed.view(np.uint32)!=values.view(np.uint32))),
                      max_ulps=ulps,max_uv_displacement=float(np.max(np.abs(fixed.astype(np.float64)-values))))


def separate_float32_collapse(triangle, reference, *, movable_corners=(0,1,2)):
    """Correct only operator quantization, preserving a nonzero input orientation."""
    import numpy as np
    original=np.asarray(triangle,dtype=np.float32);orientation=float32_uv_area(reference)
    if float32_uv_area(original)!=0 or not orientation:raise ValueError('UV correction requires a collapsed result and nonzero pre-operator reference')
    for steps in range(1,5):
        choices=[]
        for corner in movable_corners:
            for axis in range(2):
                for direction in (-np.inf,np.inf):
                    candidate=original.copy()
                    for _ in range(steps):candidate[corner,axis]=np.nextafter(candidate[corner,axis],np.float32(direction))
                    delta=abs(float(candidate[corner,axis])-float(original[corner,axis]))
                    if 0<=candidate[corner,axis]<=1 and delta<=4*np.finfo(np.float32).eps and float32_uv_area(candidate)*orientation>0:
                        choices.append((delta,corner,axis,candidate))
        if choices:
            best=min(choices,key=lambda value:value[:3])
            return best[3],best[0],steps
    raise ValueError('Collapsed UV cannot be represented within four float32 ULPs')


def _interiors_overlap(first, second):
    # Every finite float32 in [0,1] is an integer on the 2**-149 grid.
    # Integer separating-axis tests distinguish shared edges from true overlap
    # without epsilons that could silently accept narrow overlapping slivers.
    for triangle in (first,second):
        for i in range(3):
            a,b=triangle[i],triangle[(i+1)%3]
            axis=(b[1]-a[1],a[0]-b[0])
            left=[axis[0]*v[0]+axis[1]*v[1] for v in first]
            right=[axis[0]*v[0]+axis[1]*v[1] for v in second]
            if max(left)<=min(right) or max(right)<=min(left):return False
    return True


class UVTriangleOverlapError(ValueError):
    """A measured strict interior overlap, not a generic UV/resource refusal."""
    def __init__(self, first, second):
        self.faces=(int(first),int(second))
        super().__init__(f'Destination UV triangle interiors overlap: faces {first}, {second}')


def validate_uv_triangles(triangles, *, max_triangles=2_000_000,
                          max_cell_visits=16_000_000,max_pairs=8_000_000,
                          collect_overlaps=False, collect_overlap_pairs=False):
    """Spatial bins generate candidates, never a global Python all-pairs scan.

    Work/cell overflow is an explicit unsupported-layout refusal, not a pass.
    Triangles from every component sharing an atlas must be supplied together.
    Collection diagnoses a bounded repair set, never an atlas validation pass.
    """
    if collect_overlap_pairs and not collect_overlaps:
        raise ValueError('UV pair witnesses require overlap collection')
    witnesses=[];witness_faces=set()
    count=len(triangles)
    if not 1<=count<=max_triangles:raise ValueError('Destination UV triangle count exceeds validation limits')
    grid=min(256,max(1,int(math.sqrt(count/4))))
    overlap_mask=None
    if collect_overlaps:
        import numpy as np
        overlap_mask=np.zeros(count,dtype=np.bool_)
    bins={};stored=[];bounds=[];visits=pairs=0;scale=1<<149;zero_area_faces=0
    for ordinal,triangle in enumerate(triangles):
        points=[]
        for row in triangle:
            if len(row)!=2:raise ValueError('Destination UV requires two coordinates')
            point=tuple(float(v) for v in row)
            if any(not math.isfinite(v) or not 0<=v<=1 for v in point):raise ValueError('Destination UV must be finite inside the 0–1 tile')
            if any(struct.unpack('<f',struct.pack('<f',v))[0]!=v for v in point):raise ValueError('Destination UV validation requires the actual stored float32 values')
            points.append(tuple(int(v*scale) for v in point))
        if len(points)!=3:raise ValueError('Destination UV must be triangulated')
        a,b,c=points
        if (b[0]-a[0])*(c[1]-a[1])==(b[1]-a[1])*(c[0]-a[0]):
            if not collect_overlaps:raise ValueError(f'Destination UV has a zero-area triangle: face {ordinal}')
            overlap_mask[ordinal]=True;zero_area_faces+=1
            stored.append(None);bounds.append(None)
            continue
        x0=min(v[0] for v in points);x1=max(v[0] for v in points)
        y0=min(v[1] for v in points);y1=max(v[1] for v in points)
        box=(x0,x1,y0,y1)
        gx0=min(grid-1,x0*grid//scale);gx1=min(grid-1,x1*grid//scale)
        gy0=min(grid-1,y0*grid//scale);gy1=min(grid-1,y1*grid//scale)
        cells=(gx1-gx0+1)*(gy1-gy0+1)
        if cells>4096 or visits+cells>max_cell_visits:raise ValueError('Destination UV spatial validation work limit exceeded')
        keys=[(x,y) for x in range(gx0,gx1+1) for y in range(gy0,gy1+1)]
        candidates=set()
        for key in keys:
            candidates.update(bins.get(key,()))
            if len(candidates)>8192:raise ValueError('Destination UV layout is too dense or ambiguous to validate')
        for other in candidates:
            old=bounds[other]
            if x1<=old[0] or old[1]<=x0 or y1<=old[2] or old[3]<=y0:continue
            pairs+=1
            if pairs>max_pairs:raise ValueError('Destination UV overlap-pair work limit exceeded')
            if _interiors_overlap(points,stored[other]):
                if not collect_overlaps:raise UVTriangleOverlapError(other,ordinal)
                overlap_mask[other]=True;overlap_mask[ordinal]=True
                if collect_overlap_pairs:
                    witnesses.append((other,ordinal));witness_faces.update((other,ordinal))
                    if len(witnesses)>32 or len(witness_faces)>32:
                        raise ValueError('Destination UV overlap witness limit exceeded')
        stored.append(points);bounds.append(box);visits+=cells
        for key in keys:
            bucket=bins.setdefault(key,[])
            if len(bucket)>=8192:raise ValueError('Destination UV bucket capacity exceeded')
            bucket.append(ordinal)
    result=dict(triangles=count,overlap_pairs_tested=pairs,spatial_cell_visits=visits,
                method='Exact float32 integer SAT after bounded spatial-bin candidate generation',overlap=False)
    if collect_overlaps:
        indices=np.flatnonzero(overlap_mask)
        result.update(overlap=bool(len(indices)),overlap_faces=indices,
                      zero_area_seed_faces=zero_area_faces,collection_mask_bytes=overlap_mask.nbytes,collection_index_bytes=indices.nbytes)
    if collect_overlap_pairs:result['overlap_face_pairs']=witnesses
    return result


class UVRasterSeedError(ValueError):
    """A generated chart cannot receive a genuine own interior raster sample."""
    def __init__(self, report):
        self.report=report
        super().__init__(f"UV raster seed refusal: orphan={report['orphan_charts']}, aligned={report['aligned_charts']}, regularized={report.get('regularized_charts',0)}, unsupported={report['unsupported_charts']}")


def separate_packed_overlap(triangles, reference, chart_faces, pair, movable, resolution):
    """One generated post-repack scalar correction; no packing or seed recovery.

    The worker supplies actual same-part/chart membership and singleton logical
    vertices. Coordinate uniqueness is an additional condition, never topology.
    Scratch is linear in the existing two-million-face domain; local work is
    capped at96 proposals,256 bbox neighbors and4096 raster samples per face.
    Only the first local/raster-valid proposal gets ONE strict full-atlas SAT.
    """
    import numpy as np
    from production_tangents import triangle_samples
    uv=np.asarray(triangles);ref=np.asarray(reference);ids=np.asarray(chart_faces);move=np.asarray(movable)
    if (uv.dtype!=np.float32 or uv.ndim!=3 or uv.shape[1:]!=(3,2) or not 1<=len(uv)<=2_000_000
            or type(resolution) is not int or not 1<=resolution<=8192 or ids.ndim!=1 or not 1<=len(ids)<=32
            or not np.issubdtype(ids.dtype,np.integer) or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(uv))
            or ref.dtype!=np.float32 or ref.shape!=(len(ids),3,2) or move.dtype!=bool or move.shape!=(2,3)
            or len(pair)!=2 or pair[0]==pair[1] or any(p not in ids for p in pair)
            or not np.isfinite(uv).all() or np.any(uv<0) or np.any(uv>1)):
        return None
    try:validate_uv_triangles(ref)
    except ValueError:return None
    try:validate_uv_triangles(uv[list(pair)])
    except UVTriangleOverlapError:pass
    except ValueError:return None
    else:return None
    if any(float32_uv_area(before)*float32_uv_area(after)<=0 for before,after in zip(ref,uv[ids])):return None

    def raster(triangle):
        span=np.ceil(np.ptp(triangle.astype(float),axis=0)*resolution)+3
        if np.prod(span)>4096:return None
        result=[]
        for index,weights in triangle_samples(triangle,resolution):
            if len(result)==4096:return None
            result.append((index,weights.tobytes()))
        return tuple(result)

    flat=uv.reshape(-1,2);lower=uv.min(axis=1);upper=uv.max(axis=1);eligible=[]
    for which,face in enumerate(pair):
        samples=raster(uv[face])
        if samples is None:continue
        for corner in np.flatnonzero(move[which]):
            if np.count_nonzero(np.all(flat==uv[face,corner],axis=1))==1:
                eligible.append((int(face),int(corner),samples))
    for steps in range(1,5):
        for face,corner,samples in eligible:
            for axis in range(2):
                for direction in (-np.inf,np.inf):
                    triangle=uv[face].copy()
                    for _ in range(steps):triangle[corner,axis]=np.nextafter(triangle[corner,axis],np.float32(direction))
                    if (np.any(triangle<0) or np.any(triangle>1) or not np.isfinite(triangle).all()
                            or float32_uv_area(triangle)*float32_uv_area(uv[face])<=0
                            or np.any(np.all(flat==triangle[corner],axis=1))):continue
                    lo=triangle.min(axis=0);hi=triangle.max(axis=0)
                    neighbors=np.flatnonzero(np.all(upper>=lo,axis=1)&np.all(lower<=hi,axis=1))
                    nearby=np.unique(np.r_[neighbors,pair,face])
                    if len(nearby)>256:return None
                    local=uv[nearby].copy();local[np.flatnonzero(nearby==face)[0]]=triangle
                    try:validate_uv_triangles(local)
                    except UVTriangleOverlapError:continue
                    if raster(triangle)!=samples:continue
                    candidate=uv.copy();candidate[face]=triangle
                    validation=validate_uv_triangles(candidate)
                    report=dict(face=face,corner=corner,axis=axis,ulps=steps,
                        old_bits=int(uv[face,corner,axis].view(np.uint32)),new_bits=int(triangle[corner,axis].view(np.uint32)),
                        chart_faces=len(ids),local_faces=len(nearby),resolution=resolution,
                        native_model_raster_bitexact=True,raster_samples=len(samples),full_atlas_validations=1,
                        policy='One unique singleton logical UV vertex scalar; prepack exact nonoverlap and orientation; final exact SAT; no seed or coverage claim')
                    return face,triangle,report,validation
    return None


def align_unseeded_charts(values, triangle_loops, charts, resolution, *, repair=True, detail_limit=32):
    """Align unseeded charts and enforce a generated singleton representation floor.

    One regular candidate is a UV shape/density exception, not a translation.

    A seed requires strictly positive barycentrics, not area, AABB occupancy,
    an edge/corner sample or neighboring-chart padding. This is a necessary
    raster precondition, never proof of HIGH coverage or filtered appearance.
    """
    import numpy as np
    from production_tangents import triangle_samples
    uv=np.asarray(values);loops=np.asarray(triangle_loops);charts=np.asarray(charts)
    if (uv.dtype!=np.float32 or uv.ndim!=2 or uv.shape[1]!=2 or not np.isfinite(uv).all()
            or np.any(uv<0) or np.any(uv>1) or loops.ndim!=2 or loops.shape[1]!=3
            or not np.issubdtype(loops.dtype,np.integer) or loops.size==0 or loops.min()<0 or loops.max()>=len(uv)
            or charts.shape!=(len(loops),) or not np.issubdtype(charts.dtype,np.integer)
            or not 1<=len(loops)<=2_000_000 or type(resolution) is not int or not 1<=resolution<=8192
            or type(detail_limit) is not int or not 0<=detail_limit<=32):
        raise ValueError('Invalid bounded UV raster chart inputs')
    chart_ids=np.unique(charts)
    if not np.array_equal(chart_ids,np.arange(len(chart_ids))):raise ValueError('UV chart IDs must be dense and nonnegative')
    order=np.argsort(charts,kind='stable');groups=np.split(order,np.flatnonzero(np.diff(charts[order]))+1)
    owners_min=np.full(len(uv),len(groups),np.int64);owners_max=np.full(len(uv),-1,np.int64)
    np.minimum.at(owners_min,loops.ravel(),np.repeat(charts,3));np.maximum.at(owners_max,loops.ravel(),np.repeat(charts,3))
    used=owners_max>=0
    if np.any(owners_min[used]!=owners_max[used]):raise ValueError('UV loop crosses chart ownership')
    result=uv.copy();seeds=np.full(len(groups),-1,np.int64);examined=0;seeded_loops=np.zeros(len(uv),bool)
    failure_details=[];regular_details=[];success_details=[];regular_faces=[];unsupported_faces=[]
    report=dict(charts=len(groups),seeded_charts=0,orphan_charts=0,aligned_charts=0,regularized_charts=0,minimum_floor_charts=0,seeded_floor_exceptions=0,unsupported_charts=0,
                minimum_singleton_inradius_texels=.001,max_regular_displacement_texels=0.,max_displacement_texels=0.,max_rounding_error_texels=0.,details=[],detail_limit=detail_limit,
                criterion='Pinned bake binary32 raster sample with finite strictly interior barycentrics; w=fl(fl(1-u)-v)',
                policy='One <=0.5px rigid translation; generated singleton inradius below0.001px or failed interior alignment may receive one regular r0.25px UV density/shape exception; all other seeded charts unchanged; supplied UV unchanged; no repack')
    def seed(data, faces):
        nonlocal examined
        for face in faces:
            for index,bary in triangle_samples(data[loops[face]],resolution):
                examined+=1
                if examined>16_000_000:raise ValueError('UV raster sample work limit exceeded')
                expected=np.float32(np.float32(np.float32(1.)-bary[0])-bary[1])
                if (0<=index<resolution*resolution and np.isfinite(bary).all() and np.all(bary>0)
                        and np.all(bary<1) and bary[2]==expected):
                    return int(index),int(face),bary.tolist()
        return None
    for chart,faces in enumerate(groups):
        found=seed(uv,faces);initially_seeded=found is not None;minimum_floor=False
        if repair and len(faces)==1:
            singleton=uv[loops[faces[0]]].astype(np.float64)*resolution
            edges=np.linalg.norm(np.roll(singleton,-1,axis=0)-singleton,axis=1)
            a,b=singleton[1]-singleton[0],singleton[2]-singleton[0]
            radius=abs(a[0]*b[1]-a[1]*b[0])/edges.sum() if edges.sum()>0 else 0.
            minimum_floor=bool(0<radius<.001)
        if initially_seeded:
            seeds[chart]=found[0];report['seeded_charts']+=1
            if not minimum_floor:
                seeded_loops[loops[faces].ravel()]=True;continue
            report['seeded_floor_exceptions']+=1
        else:report['orphan_charts']+=1
        detail=dict(chart=chart,faces=len(faces),first_face=int(faces[0]),initially_seeded=initially_seeded);reason=None
        if not repair:reason='Supplied UV has no genuine interior chart seed; UV unchanged'
        else:
            triangles=uv[loops[faces]].astype(np.float64)*resolution
            sides=np.linalg.norm(np.roll(triangles,-1,axis=1)-triangles,axis=2)
            cross=(triangles[:,1,0]-triangles[:,0,0])*(triangles[:,2,1]-triangles[:,0,1])-(triangles[:,1,1]-triangles[:,0,1])*(triangles[:,2,0]-triangles[:,0,0])
            perimeter=sides.sum(axis=1);radii=np.divide(np.abs(cross),perimeter,out=np.zeros(len(faces)),where=perimeter>0)
            best=int(np.argmax(radii));face=int(faces[best]);triangle=triangles[best];weights=sides[best,[1,2,0]]
            if not np.isfinite(radii[best]) or radii[best]<=0:reason='No positive-inradius triangle in chart'
            else:
                center=(weights[:,None]*triangle).sum(axis=0)/weights.sum()
                # bake.cc subtracts binary32 (0.5 + axis bias), not exactly 0.5.
                origin=np.add(np.float32(.5),np.array([.001,.002],np.float32)).astype(np.float64)
                destination=np.floor(center-origin+.5)+origin;translation=destination-center
                ids=np.unique(loops[faces]);transformed=uv[ids].astype(np.float64)+translation/resolution;stored=transformed.astype(np.float32)
                displacement=float(np.max(np.abs(stored.astype(np.float64)-uv[ids].astype(np.float64)))*resolution)
                rounding=float(np.max(np.abs(stored.astype(np.float64)-transformed))*resolution)
                detail.update(reference_face=face,inradius_texels=float(radii[best]),translation_texels=translation.tolist(),stored_max_displacement_texels=displacement,
                              reference_uv_before=uv[loops[face]].tolist(),reference_uv_attempted=stored[np.searchsorted(ids,loops[face])].tolist(),
                              attempted_chart_uv_min=stored.min(axis=0).tolist(),attempted_chart_uv_max=stored.max(axis=0).tolist(),
                              maximum_float32_rounding_texels=rounding)
                if not np.isfinite(stored).all() or np.any(stored<0) or np.any(stored>1):reason='Translated chart leaves the 0-1 tile'
                elif displacement>.5:reason='Stored float32 translation exceeds 0.5 texel bound'
                else:
                    previous=result[ids].copy();result[ids]=stored;candidate=result
                    if any(float32_uv_area(candidate[loops[f]])*float32_uv_area(uv[loops[f]])<=0 for f in faces):
                        reason='Stored translation changes a triangle orientation or collapses it'
                    else:
                        found=seed(candidate,faces);regularized=False
                        if found is None or minimum_floor:
                            reason='Generated singleton below minimum UV inradius' if minimum_floor else 'One incenter alignment has no genuine stored binary32 interior seed'
                            found=None
                            if len(faces)==1:
                                # Explicit local UV shape/density exception, NOT a relaxed translation cap.
                                regular=(destination+np.array([[0.,.5],[-math.sqrt(3)*.25,-.25],[math.sqrt(3)*.25,-.25]]))/resolution
                                if float32_uv_area(uv[loops[face]])<0:regular=regular[[0,2,1]]
                                regular_stored=regular.astype(np.float32)
                                detail.update(UV_density_exception='singleton_regular_r0.25px_minimum_inradius' if minimum_floor else 'singleton_regular_r0.25px_after_failed_interior_translation',
                                              minimum_representation_reason='minimum_UV_inradius' if minimum_floor else 'failed_interior_translation',
                                              minimum_UV_inradius_threshold_texels=.001,minimum_UV_inradius_before_texels=float(radii[best]),
                                              regular_uv=regular_stored.tolist(),regular_target_inradius_texels=.25,
                                              regular_float32_rounding_texels=float(np.max(np.abs(regular_stored.astype(np.float64)-regular))*resolution))
                                if not np.isfinite(regular_stored).all() or np.any(regular_stored<0) or np.any(regular_stored>1):
                                    reason='One regular singleton candidate leaves the 0-1 tile'
                                elif float32_uv_area(regular_stored)*float32_uv_area(uv[loops[face]])<=0:
                                    reason='One regular singleton candidate changes orientation or collapses'
                                else:
                                    result[loops[face]]=regular_stored;found=seed(result,faces)
                                    if found is None:reason='One regular singleton candidate has no genuine stored binary32 interior seed'
                                    else:
                                        xy=regular_stored.astype(np.float64)*resolution
                                        lengths=np.linalg.norm(np.roll(xy,-1,axis=0)-xy,axis=1)
                                        area=abs(np.cross(xy[1]-xy[0],xy[2]-xy[0]))*.5
                                        regular_displacement=float(np.max(np.abs(regular_stored.astype(np.float64)-uv[loops[face]].astype(np.float64)))*resolution)
                                        regularized=True;reason=None;regular_faces.append(face)
                                        rounding=max(rounding,detail['regular_float32_rounding_texels'])
                                        report['regularized_charts']+=1
                                        report['minimum_floor_charts']+=int(minimum_floor)
                                        report['max_regular_displacement_texels']=max(report['max_regular_displacement_texels'],regular_displacement)
                                        detail.update(regular_inradius_texels=float(2*area/lengths.sum()),minimum_UV_inradius_after_texels=float(2*area/lengths.sum()),regular_area_texel2=float(area),
                                                      UV_area_density_multiplier=float(2*area/abs(cross[best])),
                                                      regular_max_displacement_texels=regular_displacement,
                                                      translation_cap_applies=False,regular_clearance='PENDING atlas geometry proof >3px')
                        if found is not None:
                            seeds[chart]=found[0]
                            if not regularized:
                                report['aligned_charts']+=1
                                report['max_displacement_texels']=max(report['max_displacement_texels'],displacement)
                            report['max_rounding_error_texels']=max(report['max_rounding_error_texels'],rounding)
                            detail.update(seed_index=found[0],seed_face=found[1],seed_barycentric=found[2])
                    if reason is not None:result[ids]=previous
        if reason is not None:
            seeds[chart]=-1
            report['unsupported_charts']+=1;detail['unsupported_reason']=reason;unsupported_faces.extend(int(f) for f in faces)
            if len(failure_details)<detail_limit:failure_details.append(detail)
        elif 'UV_density_exception' in detail:
            if len(regular_details)<detail_limit:regular_details.append(detail)
        elif len(success_details)<detail_limit:success_details.append(detail)
    if not np.array_equal(result[seeded_loops].view(np.uint32),uv[seeded_loops].view(np.uint32)):
        raise ValueError('A seeded UV chart outside the declared singleton floor changed')
    report['details']=(failure_details+regular_details+success_details)[:detail_limit]
    report.update(examined_raster_samples=examined,details_omitted=max(0,report['orphan_charts']+report['seeded_floor_exceptions']-len(report['details'])),
                  unsupported_details_omitted=report['unsupported_charts']-len(failure_details),
                  regular_details_omitted=report['regularized_charts']-len(regular_details),
                  detail_policy='Unsupported charts, then explicit UV density exceptions, then successful translations',_regular_faces=regular_faces,_unsupported_faces=unsupported_faces)
    return result,seeds,report


def native_chart_seed_report(seed_sets, occupancy, *, detail_limit=32):
    """Actual native occupancy must retain every distinct pinned chart seed."""
    import numpy as np
    occupied=np.asarray(occupancy,bool).reshape(-1)
    arrays=[np.asarray(s) for s in seed_sets]
    if (not arrays or any(not a.size or not np.issubdtype(a.dtype,np.integer) for a in arrays)
            or type(detail_limit) is not int or not 0<=detail_limit<=32):
        raise ValueError('Missing or invalid bounded UV chart seed provenance')
    seeds=np.concatenate([a.astype(np.int64,copy=False).reshape(-1) for a in arrays])
    if not len(seeds) or np.any(seeds<0) or np.any(seeds>=len(occupied)):
        raise ValueError('Missing or out-of-range UV chart seed provenance')
    if len(np.unique(seeds))!=len(seeds):raise ValueError('UV charts have ambiguous shared raster seed ownership')
    missing=np.flatnonzero(~occupied[seeds]);count=len(missing)
    return dict(valid=count==0,charts=len(seeds),native_occupied_chart_seeds=len(seeds)-count,
                unsupported_charts=count,missing_seed_indices=seeds[missing[:detail_limit]].tolist(),
                missing_seed_details_omitted=max(0,count-detail_limit),
                criterion='Every distinct pinned strictly interior chart seed must exist in actual zero-margin native occupancy; HIGH coverage remains a separate strict gate')


def regular_chart_clearance(triangles, selected_faces, resolution):
    """Bounded geometric clearance for singleton density exceptions, never occupancy.

    Predeclared floor >3px = native padding2 + paint footprint0.5 + headroom0.5.
    Does NOT establish disjoint 2+2 gutters, mip equivalence or visual quality.
    """
    import numpy as np
    stored=np.asarray(triangles);selected=np.asarray(selected_faces)
    if (stored.dtype!=np.float32 or stored.ndim!=3 or stored.shape[1:]!=(3,2)
            or not 1<=len(stored)<=2_000_000 or not np.isfinite(stored).all()
            or np.any(stored<0) or np.any(stored>1) or selected.ndim!=1
            or not np.issubdtype(selected.dtype,np.integer) or len(selected)>32
            or len(np.unique(selected))!=len(selected) or np.any(selected<0) or np.any(selected>=len(stored))
            or type(resolution) is not int or not 1<=resolution<=8192):
        raise ValueError('Unsupported bounded regular-chart clearance inputs (maximum32 exceptions)')
    uv=stored.astype(np.float64)*resolution;lower=uv.min(axis=1);upper=uv.max(axis=1)
    checked=0;details=[]
    def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
    def distances(a,b):
        nonlocal checked
        checked+=len(b)
        if checked>8192:raise ValueError('Regular-chart geometric clearance work limit exceeded')
        best=np.full(len(b),np.inf);intersects=np.zeros(len(b),bool)
        for point in a:
            for k in range(3):
                start=b[:,k];edge=b[:,(k+1)%3]-start;length=np.sum(edge*edge,axis=1)
                if np.any(length<=0):raise ValueError('Degenerate UV edge in clearance candidates')
                t=np.clip(np.sum((point-start)*edge,axis=1)/length,0,1)
                best=np.minimum(best,np.sum((point-start-t[:,None]*edge)**2,axis=1))
        for k in range(3):
            start=a[k];end=a[(k+1)%3];edge=end-start;length=float(edge@edge)
            if length<=0:raise ValueError('Degenerate regular UV edge')
            for point in np.moveaxis(b,1,0):
                t=np.clip(np.sum((point-start)*edge,axis=1)/length,0,1)
                best=np.minimum(best,np.sum((point-start-t[:,None]*edge)**2,axis=1))
            for j in range(3):
                b0=b[:,j];b1=b[:,(j+1)%3];bc=b1-b0
                o1=cross(edge,b0-start);o2=cross(edge,b1-start);o3=cross(bc,start-b0);o4=cross(bc,end-b0)
                boxes=np.all(np.maximum(np.minimum(b0,b1),np.minimum(start,end))<=np.minimum(np.maximum(b0,b1),np.maximum(start,end)),axis=1)
                intersects|=boxes&(((o1>=0)&(o2<=0))|((o1<=0)&(o2>=0)))&(((o3>=0)&(o4<=0))|((o3<=0)&(o4>=0)))
        signs=np.stack([cross(b[:,(k+1)%3]-b[:,k],a[0]-b[:,k]) for k in range(3)],axis=1)
        intersects|=np.all(signs>=0,axis=1)|np.all(signs<=0,axis=1)
        signs=np.stack([cross(a[(k+1)%3]-a[k],b[:,0]-a[k]) for k in range(3)],axis=1)
        intersects|=np.all(signs>=0,axis=1)|np.all(signs<=0,axis=1)
        return np.where(intersects,0.,np.sqrt(best))
    for face in selected:
        face=int(face);a=uv[face]
        # Coordinate-axis AABB gaps are lower bounds, not acceptance authority.
        bound=np.max(np.maximum(np.maximum(lower-a.max(axis=0),a.min(axis=0)-upper),0),axis=1)
        indices=np.flatnonzero(np.arange(len(uv))!=face)
        if not len(indices):
            details.append(dict(global_face=face,nearest_global_face=None,clearance_texels=None,valid=True));continue
        first=indices[np.argsort(bound[indices],kind='stable')[:64]]
        initial=distances(a,uv[first]);best=float(initial.min())
        candidates=np.unique(np.concatenate((first,indices[bound[indices]<=best])))
        actual=distances(a,uv[candidates]);nearest=int(np.argmin(actual));distance=float(actual[nearest])
        details.append(dict(global_face=face,nearest_global_face=int(candidates[nearest]),clearance_texels=distance,
                            valid=bool(math.isfinite(distance) and distance>3.)))
    return dict(valid=all(d['valid'] for d in details),minimum_clearance_required_texels=3.,comparison='strictly_greater_than',
                native_padding_texels=2,paint_footprint_texels=.5,float32_headroom_texels=.5,
                geometry_triangle_checks=checked,details=details,
                limitations='Vacant gutters may mix; 4px disjoint gutters, mip/filter equivalence and appearance are NOT proved')

def face_recovery_part_indices(frames, recovery_mask, resolution, part_count, *, native_margin=0):
    """Conservative atlas-local direct-writer superset for face PBR with native margin ZERO only.

    UV bounds retain a conservative three-pixel halo; native padding must be zero.
    This is only a work filter, never a coverage or source-correspondence test.
    Invalid context retains ALL parts; raw snapshots and mask are never edited.
    """
    import hashlib
    import numpy as np
    all_parts=list(range(part_count))
    report=dict(method='Boolean integral image and triangle UV bbox superset',atlas_part_count=part_count,
                halo_pixels=3,coordinate_origin='bottom-left mask[y,x]',native_margin_required_pixels=0,
                mask_sha256=None,prefix_bytes=0,fallback_all_reason=None,
                pbr_bit_equivalence='NOT_RUN against unfiltered margin0 plus identical post-composite padding')
    def result(indices,reason=None):
        report.update(part_indices=indices,candidate_part_count=len(indices),parts_omitted=part_count-len(indices),fallback_all_reason=reason)
        return indices,report
    try:
        if native_margin!=0:raise ValueError('Part filtering requires native margin zero')
        if type(resolution)is not int or not 1<=resolution<=4096:raise ValueError('Unsupported integral-image resolution')
        mask=np.asarray(recovery_mask)
        if mask.dtype!=np.bool_ or mask.shape not in ((resolution,resolution),(resolution*resolution,)):
            raise ValueError('Recovery mask must be a matching boolean atlas')
        report['mask_sha256']=hashlib.sha256(mask.tobytes()).hexdigest();report['mask_texels']=int(np.count_nonzero(mask))
        if len(frames)!=part_count:raise ValueError('Snapshot count differs from atlas part count')
        if not report['mask_texels']:return result([])
        prefix=np.zeros((resolution+1,resolution+1),np.int32);prefix[1:,1:]=mask.reshape(resolution,resolution)
        np.cumsum(prefix[1:,1:],axis=0,dtype=np.int32,out=prefix[1:,1:])
        np.cumsum(prefix[1:,1:],axis=1,dtype=np.int32,out=prefix[1:,1:])
        report['prefix_bytes']=int(prefix.nbytes);selected=[]
        for part,frame in enumerate(frames):
            uv=np.asarray(frame['uv']);loops=np.asarray(frame['triangle_loops'])
            if uv.dtype!=np.float32 or uv.ndim!=2 or uv.shape[1]!=2 or not np.isfinite(uv).all() or np.any(uv<0) or np.any(uv>1):
                raise ValueError('Invalid pinned float32 UV snapshot')
            if loops.ndim!=2 or loops.shape[1]!=3 or loops.dtype.kind not in 'iu' or not loops.size or np.any(loops<0) or np.any(loops>=len(uv)):
                raise ValueError('Invalid triangle-loop snapshot')
            triangles=uv[loops].astype(np.float64)*resolution
            lower=np.clip(np.floor(triangles.min(axis=1)).astype(np.int64)-3,0,resolution)
            upper=np.clip(np.ceil(triangles.max(axis=1)).astype(np.int64)+4,0,resolution)
            x0,y0=lower.T;x1,y1=upper.T
            hits=prefix[y1,x1]-prefix[y0,x1]-prefix[y1,x0]+prefix[y0,x0]
            if np.any(hits>0):selected.append(part)
        del prefix
        if not selected:return result(all_parts,'Nonempty recovery mask has no conservative candidate')
        return result(selected)
    except (ValueError,TypeError,IndexError,KeyError,OverflowError,MemoryError) as error:
        return result(all_parts,(type(error).__name__+': '+str(error))[:256])


class UVFilterSupportError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(report['error'])


def chart_bilinear_support(frames, face_charts, owners, occupancy, resolution, *,
                           max_bytes=384*1024**2, max_candidate_cells=64_000_000,
                           max_seconds=60.):
    """Closed base-level bilinear support of EVERY stored UV triangle, REPEAT.

    Exact integer SAT yields row spans for triangle versus a texel-centred
    square of half-width one pixel. Closed boundary contacts are deliberately
    included (conservative zero-weight ties), without a UV epsilon. This is
    not a mip/anisotropic footprint or an appearance/correspondence certificate.
    """
    import time
    import hashlib
    import numpy as np
    started = time.monotonic()
    report = dict(valid=False, resolution=resolution, sampler='REPEAT',
                  scope='Conservative complete geometric chart support, base-level bilinear only',
                  zero_weight_boundary_contacts='Included conservatively; may refuse a touching layout',
                  coordinate_origin='bottom-left native image buffer / Blender UV',
                  required_taps=0, gutter_texels=0, native_occupied_texels=0,
                  candidate_cells=0, candidate_cell_accounting='Conservative charged triangle bounding-box cells, not exact SAT invocation count',
                  conflicts=0, conflict_count_scope='First witnessed conflict; immediate refusal, not a global census',
                  coverage_filled_texels=0, recovered_texels=0,
                  assignment='Deterministic row-major multi-source eight-neighbour graph padding within SAME chart; not exact Euclidean nearest',
                  origin_policy='Every copied origin is an ORIGINAL source-valid native occupied sample, never padded data',
                  resource_limits=dict(max_bytes=max_bytes, max_candidate_cells=max_candidate_cells,
                                       max_seconds=max_seconds))

    def refuse(message, **details):
        report.update(error=message, **details, elapsed_seconds=time.monotonic()-started)
        raise UVFilterSupportError(report)

    def check_time():
        if time.monotonic()-started > max_seconds:
            refuse('UV bilinear support time admission limit exceeded; reduce layout complexity or use a less fragmented UV layout')

    if (type(resolution) is not int or not 1 <= resolution <= 4096 or
            type(max_bytes) is not int or max_bytes <= 0 or
            type(max_candidate_cells) is not int or max_candidate_cells <= 0 or
            not math.isfinite(max_seconds) or max_seconds <= 0):
        refuse('Invalid UV bilinear support resource limits or resolution')
    count = resolution*resolution
    mask = np.asarray(occupancy)
    if mask.dtype != np.bool_ or mask.shape not in ((count,), (resolution,resolution)) or not mask.flags.c_contiguous:
        refuse('UV support requires immutable native boolean occupancy of this grid')
    mask = mask.reshape(-1)
    if len(frames) != len(face_charts) or not 1 <= len(frames) <= 256:
        refuse('Missing complete bounded face-to-chart provenance')
    if not 1 <= sum(len(c) for c in face_charts) <= 2_000_000:
        refuse('UV support triangle count exceeds admission limit')
    indices = np.asarray(owners['indices']); parts = np.asarray(owners['parts']); faces = np.asarray(owners['faces'])
    if (any(a.ndim != 1 or not np.issubdtype(a.dtype, np.signedinteger) for a in (indices, parts, faces)) or
            not 1 <= len(indices) <= count or any(len(a) != len(indices) for a in (parts, faces))):
        refuse('Native owner dimensions/type exceed grid admission')
    # Before dense allocation or vector validation scratch: three dense int32
    # arrays, chart/native provenance temporaries and 16 MiB bounded scratch.
    estimated = 12*count + 32*len(indices) + 24*sum(len(c) for c in face_charts) + 16*1024**2
    report['estimated_additional_bytes'] = estimated
    if estimated > max_bytes:
        refuse('UV bilinear support memory admission limit exceeded before allocation', requested_bytes=estimated)
    if (np.any(indices < 0) or np.any(indices >= count) or np.any(np.diff(indices) <= 0) or
            np.any(parts < 0) or np.any(parts >= len(frames)) or
            int(mask.sum()) != len(indices) or not mask[indices].all()):
        refuse('Native owners do not exactly identify the immutable occupied set')
    report['native_occupied_texels'] = len(indices)
    sample_charts = np.empty(len(indices), np.int32)
    next_chart = 0
    for pi, (frame, charts) in enumerate(zip(frames, face_charts)):
        uv = np.asarray(frame['uv']); loops = np.asarray(frame['triangle_loops']); charts = np.asarray(charts)
        if (uv.dtype != np.float32 or uv.ndim != 2 or uv.shape[1] != 2 or
                not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv > 1) or
                loops.ndim != 2 or loops.shape[1] != 3 or not len(loops) or not np.issubdtype(loops.dtype, np.integer) or
                np.any(loops < 0) or np.any(loops >= len(uv)) or
                charts.shape != (len(loops),) or not np.issubdtype(charts.dtype, np.integer) or
                np.any(charts < 0) or np.any(charts >= 2_000_000)):
            refuse('Invalid stored float32 UV triangle/chart provenance', part=pi)
        selected = np.flatnonzero(parts == pi)
        if np.any(faces[selected] < 0) or np.any(faces[selected] >= len(loops)):
            refuse('Native owner face out of chart provenance range', part=pi)
        sample_charts[selected] = charts[faces[selected]]
        labels = np.unique(charts)
        if labels[0] != next_chart or int(labels[-1])+1-next_chart != len(labels):
            refuse('Chart IDs must be globally dense and disjoint across parts', part=pi)
        next_chart += len(labels)
    required = np.full(count, -1, np.int32)
    origins = np.full(count, -1, np.int32)
    queue = np.empty(count, np.int32)
    scale = 1 << 149
    ceil_div = lambda a, b: -((-a)//b)  # b is strictly positive
    # SAT axes are triangle edge normals plus the two rectangle axes. All
    # finite stored float32 UV values lie on this exact power-of-two grid.
    for pi, (frame, charts) in enumerate(zip(frames, face_charts)):
        for face, (loop, chart) in enumerate(zip(frame['triangle_loops'], charts)):
            check_time()
            tri = [tuple(int(float(v)*scale)*resolution for v in row) for row in frame['uv'][loop]]
            area = ((tri[1][0]-tri[0][0])*(tri[2][1]-tri[0][1]) -
                    (tri[1][1]-tri[0][1])*(tri[2][0]-tri[0][0]))
            if not area:refuse('Zero-area UV face has no bilinear domain', part=pi, face=face)
            lower = [ceil_div(min(p[a] for p in tri)-3*scale//2, scale) for a in (0, 1)]
            upper = [(max(p[a] for p in tri)+scale//2)//scale for a in (0, 1)]
            report['candidate_cells'] += (upper[0]-lower[0]+1)*(upper[1]-lower[1]+1)
            if report['candidate_cells'] > max_candidate_cells:
                refuse('UV bilinear support candidate-cell admission limit exceeded; no partial support accepted', part=pi, face=face)
            axes = []
            for a, b in zip(tri, tri[1:]+tri[:1]):
                nx, ny = b[1]-a[1], a[0]-b[0]
                projected = [nx*p[0]+ny*p[1] for p in tri]
                radius = (abs(nx)+abs(ny))*scale
                axes.append((nx*scale, ny*scale, (nx+ny)*scale//2,
                             min(projected)-radius, max(projected)+radius))
            for y in range(lower[1], upper[1]+1):
                x0, x1 = lower[0], upper[0]
                for ax, ay, bias, lo, hi in axes:
                    constant = ay*y+bias
                    if ax > 0:
                        x0 = max(x0, ceil_div(lo-constant, ax)); x1 = min(x1, (hi-constant)//ax)
                    elif ax < 0:
                        x0 = max(x0, ceil_div(constant-hi, -ax)); x1 = min(x1, (constant-lo)//(-ax))
                    elif not lo <= constant <= hi:
                        x1 = x0-1; break
                if x0 > x1:continue
                # At most resolution+3 candidates in a row; modulo is actual
                # output REPEAT behaviour, not UV clipping at an atlas edge.
                taps = (y % resolution)*resolution + np.arange(x0, x1+1, dtype=np.int32) % resolution
                previous = required[taps]
                collision = (previous != -1) & (previous != chart)
                if collision.any():
                    pixel = int(taps[np.flatnonzero(collision)[0]])
                    refuse('Bilinear tap conflicts with another chart; packing/UV layout unsupported at this grid',
                           conflicts=1, pixel=pixel, x=pixel % resolution, y=pixel//resolution,
                           required_chart=int(chart), previous_chart=int(required[pixel]),
                           native_occupied=bool(mask[pixel]),
                           native_owner_chart=int(sample_charts[np.searchsorted(indices,pixel)]) if mask[pixel] else None,
                           part=pi, face=face)
                required[taps] = chart
    if np.any(required[indices] != sample_charts):
        row = int(np.flatnonzero(required[indices] != sample_charts)[0])
        refuse('Native occupied tap conflicts with or lies outside its own chart support',
               pixel=int(indices[row]), owner_chart=int(sample_charts[row]), required_chart=int(required[indices[row]]))
    origins[indices] = indices
    queue[:len(indices)] = indices
    head = 0; tail = len(indices)
    while head < tail:
        if head % 4096 == 0:check_time()
        pixel = int(queue[head]); head += 1
        x, y = pixel % resolution, pixel//resolution; chart = required[pixel]
        for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            neighbour = ((y+dy) % resolution)*resolution + (x+dx) % resolution
            if required[neighbour] == chart and origins[neighbour] == -1:
                origins[neighbour] = origins[pixel]; queue[tail] = neighbour; tail += 1
    required_count = int(np.count_nonzero(required >= 0))
    for start in range(0, count, 65536):
        stop = min(start+65536, count); needed = required[start:stop] >= 0
        missing = needed & (origins[start:stop] < 0)
        if missing.any():
            pixel = start+int(np.flatnonzero(missing)[0])
            refuse('Required chart support component has no ORIGINAL native valid seed; no default fill',
                   pixel=pixel, chart=int(required[pixel]))
        source = origins[start:stop][needed]
        if not mask[source].all() or not np.array_equal(required[source], required[start:stop][needed]):
            refuse('Chart gutter origin is not an ORIGINAL valid occupied sample of the same chart')
    check_time()
    report.update(valid=True, charts=next_chart, native_occupied_texels=len(indices),
                  required_taps=required_count, gutter_texels=required_count-len(indices),
                  elapsed_seconds=time.monotonic()-started, actual_dense_bytes=required.nbytes+origins.nbytes+queue.nbytes,
                  immutable_occupancy_sha256=hashlib.sha256(mask).hexdigest(),
                  packing=dict(generated_default_target_gap_pixels=4, conservative_two_square_support_gap=2*math.sqrt(2),
                               actual_post_UV_support_conflicts=0, spacing_proof='Disjoint actual chart support masks, not a measured global minimum gap', no_repack=True))
    required.setflags(write=False); origins.setflags(write=False)
    return dict(required_charts=required, origins=origins, report=report)


def apply_chart_gutters(data, occupancy, resolution, plan):
    """Copy composed RGBA only from ORIGINAL same-chart native seeds, in chunks."""
    import hashlib
    import numpy as np
    mask = np.asarray(occupancy).reshape(-1); origins = plan['origins']; charts = plan['required_charts']
    if (not plan['report']['valid'] or plan['report']['resolution'] != resolution or
            data.dtype != np.float32 or data.shape != (resolution*resolution, 4) or
            mask.dtype != np.bool_ or mask.shape != origins.shape or charts.shape != origins.shape):
        raise ValueError('Chart gutter plan/pixels differ from validated native grid')
    if hashlib.sha256(mask).hexdigest() != plan['report']['immutable_occupancy_sha256']:
        raise ValueError('Native occupancy changed after chart gutter planning')
    before = hashlib.sha256(); after = hashlib.sha256(); copied = 0
    for start in range(0, len(mask), 65536):
        stop = min(start+65536, len(mask)); occupied = mask[start:stop]
        before.update(data[start:stop][occupied].tobytes())
        take = (charts[start:stop] >= 0) & ~occupied
        destination = start+np.flatnonzero(take); source = origins[destination]
        if (np.any(source < 0) or np.any(source >= len(mask)) or not mask[source].all() or
                not np.array_equal(charts[destination], charts[source]) or not np.isfinite(data[source]).all()):
            raise ValueError('Gutter attempted to copy a default, nonfinite value or another chart')
        data[destination] = data[source]
        after.update(data[start:stop][occupied].tobytes()); copied += len(destination)
    if before.digest() != after.digest():raise ValueError('Chart padding changed native occupied RGBA bytes')
    return dict(padded_unoccupied_texels=copied, native_occupied_RGBA_byteexact=True,
                occupied_RGBA_sha256=before.hexdigest(), coverage_filled_texels=0, recovered_texels=0,
                assignment=plan['report']['assignment'], sampler='REPEAT', guarantee='base-level bilinear support only')
