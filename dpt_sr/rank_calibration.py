"""Exact optimization over open ranking-stability intervals on one score axis.

Coordinate selection is validated on disjoint inner users. This is neither a
globally optimal multi-parameter solver nor a guarantee of test improvement.
"""
import numpy as np

def gains(rank,k=10):
    r=np.asarray(rank);return np.where((r>0)&(r<=k),1/np.log2(np.maximum(r,1)+1),0.)

def user_events(intercept,slope,target,seen,ties,lo,hi):
    keep=np.ones(len(intercept),bool);keep[np.asarray(seen,int)]=False;keep[target]=False
    da=intercept[keep]-intercept[target];db=slope[keep]-slope[target];tr=ties[keep]<ties[target]
    nz=db!=0;roots=-da[nz]/db[nz];sign=np.sign(db[nz]).astype(np.int32)
    use=(roots>lo)&(roots<hi)&np.isfinite(roots);roots=roots[use];sign=sign[use]
    first=roots.min() if len(roots) else hi;t0=(lo+first)/2
    z=da+t0*db;rank=1+np.count_nonzero(z>0)+np.count_nonzero((z==0)&tr)
    initial=float(gains(rank))
    if not len(roots):return initial,np.empty(0),np.empty(0)
    order=np.argsort(roots,kind='stable');roots=roots[order];sign=sign[order]
    unique,idx=np.unique(roots,return_index=True);change=np.add.reduceat(sign,idx)
    after=rank+np.cumsum(change);before=after-change;delta=gains(after)-gains(before)
    # Changes outside top-10 contribute nothing to the objective, so only its
    # relevant breakpoints are retained; the score still ranks the full catalog.
    active=delta!=0
    return initial,unique[active],delta[active]

def optimum(initial,events,lo,hi,anchor=0.):
    """Best stable interval; among ties choose the point nearest the anchor."""
    if not events:return float(np.clip(anchor,lo,hi)),float(initial)
    roots=np.concatenate([e[0] for e in events]);delta=np.concatenate([e[1] for e in events])
    if not len(roots):return float(np.clip(anchor,lo,hi)),float(initial)
    order=np.argsort(roots,kind='stable');roots=roots[order];delta=delta[order]
    roots,idx=np.unique(roots,return_index=True);delta=np.add.reduceat(delta,idx)
    value=initial+np.r_[0.,np.cumsum(delta)];edges=np.r_[lo,roots,hi]
    best=np.max(value);ids=np.flatnonzero(value>=best-1e-11)
    points=[]
    for j in ids:
        l,r=edges[j:j+2]
        points.append(anchor if l<anchor<r else (l+r)/2)
    point=points[int(np.argmin(np.abs(np.asarray(points)-anchor)))]
    return float(point),float(best)

def value_at(initial,events,point):
    return float(initial+sum(float(delta[roots<point].sum()) for roots,delta in events))
