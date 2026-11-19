"""Paired user-level uncertainty for deterministic ranking models.

Grouping identical deltas is an exact sufficient representation for resampling
the mean. It does not replace users by ranks or treat items as independent.
"""
import numpy as np


def paired_summary(delta,seed=2026,bootstrap_draws=5000,flip_draws=20000):
    delta=np.asarray(delta,dtype=float)
    if delta.ndim!=1 or not len(delta) or not np.isfinite(delta).all():raise ValueError('Finite nonempty user deltas required')
    n=len(delta);observed=float(delta.mean());rng=np.random.default_rng(seed)
    values,counts=np.unique(delta,return_counts=True);means=[]
    for start in range(0,bootstrap_draws,256):
        draws=rng.multinomial(n,counts/n,size=min(256,bootstrap_draws-start))
        means.extend((draws@values/n).tolist())
    lo,hi=np.quantile(means,[.025,.975])
    result={'users':n,'difference':observed,'ci95_low':float(lo),'ci95_high':float(hi),
            'users_improved':int((delta>0).sum()),'users_worse':int((delta<0).sum()),
            'users_equal':int((delta==0).sum()),'bootstrap_draws':bootstrap_draws}
    if flip_draws:
        av,ac=np.unique(np.abs(delta[delta!=0]),return_counts=True);extreme=0
        for start in range(0,flip_draws,256):
            size=min(256,flip_draws-start)
            draws=rng.binomial(ac,.5,size=(size,len(ac)))
            null=((2*draws-ac)@av)/n
            extreme+=int((np.abs(null)>=abs(observed)-1e-15).sum())
        result.update({'p_two_sided':float((extreme+1)/(flip_draws+1)),'sign_flip_draws':flip_draws})
    return result


def holm(pvalues):
    p=np.asarray(pvalues,dtype=float);order=np.argsort(p);m=len(p)
    adjusted=np.minimum(1,np.maximum.accumulate(p[order]*(m-np.arange(m))))
    out=np.empty(m);out[order]=adjusted
    return out
