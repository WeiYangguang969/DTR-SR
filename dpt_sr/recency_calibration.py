"""Small convex calibration of recency experts and conditional path evidence.

Learning uses next-item pairs from an inner training holdout. It neither reads
the test set nor assumes that the newest interaction is always most useful.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from capacity_probes.models.pctm import kernel_weights
from .calibration import calibrate_history
from .model import score_histories

def kernels_for(original):
    return list(dict.fromkeys([original,'last','exp:0.3','exp:0.6','exp:0.85','inv:0.5']))

def history_experts(model,ev,kernels):
    out=[]
    for kernel in kernels:
        h=ev.history_matrix(model.max_history,lambda n:kernel_weights(kernel,n))
        if model.calibration_config.exposure_power:
            h=calibrate_history(h,model.history_scale)
        h.eliminate_zeros();out.append(h)
    return out

def dense_features(model,histories,start,stop):
    """Bounded user blocks, no dense item-by-item matrix."""
    features=[score_histories(model.base.base,h[start:stop]).toarray() for h in histories]
    features.append((histories[0][start:stop]@model.completion_evidence).toarray())
    features.append(np.broadcast_to(model.log_popularity,(stop-start,len(model.catalog))))
    return features

def prior_vector(model,kernels):
    w=np.zeros(len(kernels)+2);w[0]=1.;w[-2]=1.;w[-1]=model.pctm_config.pop_boost
    return w

def fit_weights(differences,prior,regularization,temperature=5.):
    """Strictly convex pairwise logistic objective with analytic gradient."""
    x=np.asarray(differences,float);b=len(prior)-2
    def fg(w):
        z=temperature*(x@w)
        loss=np.logaddexp(0,-z).mean()+.5*regularization*np.sum((w-prior)**2)
        g=-(temperature/len(x))*(x.T@expit(-z))+regularization*(w-prior)
        return float(loss),g
    solved=minimize(fg,prior.copy(),jac=True,method='SLSQP',bounds=[(0,1)]*b+[(0,2),(-1,1)],
        constraints=[{'type':'eq','fun':lambda w:w[:b].sum()-1,
                      'jac':lambda w:np.r_[np.ones(b),0.,0.]}],
        options={'maxiter':200,'ftol':1e-10})
    if not solved.success:raise RuntimeError(str(solved.message))
    w=solved.x
    if abs(w[:b].sum()-1)>1e-7 or np.min(w[:b])<-1e-8:raise AssertionError('Simplex violated')
    return w,dict(objective=float(solved.fun),iterations=int(solved.nit),converged=bool(solved.success),
                  regularization=regularization,temperature=temperature,pairs=len(x),
                  unique_objective_minimum=True,termination=str(solved.message))

def fit_weights_softmax(differences,importance,prior,regularization,temperature=5.):
    """Stratified sampled partition: hard items exact, others importance weighted.

    Each user has one positive and equally many sampled negatives. The partition
    estimate is unbiased; its logarithm and its gradient need not be unbiased.
    For the fixed samples the ridge-regularized objective is strictly convex.
    """
    x=np.asarray(differences,float);iw=np.asarray(importance,float);b=len(prior)-2
    from scipy.special import logsumexp
    logiw=np.log(iw)
    def fg(w):
        z=-temperature*np.einsum('unf,f->un',x,w,optimize=True)+logiw
        logden=logsumexp(np.concatenate([np.zeros((len(x),1)),z],axis=1),axis=1)
        probability=np.exp(z-logden[:,None])
        loss=logden.mean()+.5*regularization*np.sum((w-prior)**2)
        g=-(temperature/len(x))*np.einsum('un,unf->f',probability,x,optimize=True)+regularization*(w-prior)
        return float(loss),g
    solved=minimize(fg,prior.copy(),jac=True,method='SLSQP',bounds=[(0,1)]*b+[(0,2),(-1,1)],
        constraints=[{'type':'eq','fun':lambda w:w[:b].sum()-1,'jac':lambda w:np.r_[np.ones(b),0.,0.]}],
        options={'maxiter':200,'ftol':1e-10})
    if not solved.success:raise RuntimeError(str(solved.message))
    return solved.x,dict(objective=float(solved.fun),iterations=int(solved.nit),converged=True,
        regularization=regularization,temperature=temperature,users=len(x),negatives_per_user=x.shape[1],
        objective_name='stratified sampled log-partition',partition_unbiased=True,log_partition_unbiased=False)

def weighted_history(histories,w):
    h=histories[0]*float(w[0])
    for hh,ww in zip(histories[1:],w[1:len(histories)]):
        if ww>0:h=h+hh*float(ww)
    return h.tocsr()

def score_calibrated(model,histories,w,start,stop):
    b=len(histories)
    if model.config.power==0:
        h=weighted_history([h[start:stop] for h in histories],w)
        x=score_histories(model.base.base,h).toarray()
    else:
        x=None
        for hh,ww in zip(histories,w[:b]):
            if ww<=0:continue
            term=score_histories(model.base.base,hh[start:stop]).toarray()*ww
            x=term if x is None else x+term
    if w[-2]>0:x+=w[-2]*(histories[0][start:stop]@model.completion_evidence).toarray()
    x+=w[-1]*model.log_popularity
    return x
