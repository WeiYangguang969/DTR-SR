"""Directional shrinkage of pairwise transitions, with sparse power pooling.

Empirical-Bayes regularization of weighted counts; no claim of a calibrated
Bayesian posterior for dependent events. Bradley-Terry is established theory.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit

from capacity_probes.counts import OffsetCache, _store_identity
from capacity_probes.models.pctm import (PCTMConfig, count_weights, kernel_weights, log_popularity)
from capacity_probes.ranking import RecommendationCollector, tie_ranks, topk


@dataclass(frozen=True)
class DPTConfig:
    kappa: float = 1.0
    power: float = 0.0
    ridge: float = 1.0
    maxiter: int = 200
    ftol: float = 1e-9
    prior: str = "bt"

    def __post_init__(self):
        values=(self.kappa,self.power,self.ridge,self.ftol)
        if not all(np.isfinite(x) for x in values):
            raise ValueError("All numerical parameters must be finite")
        if self.kappa<0 or not 0<=self.power<=1 or self.ridge<=0 or self.ftol<=0 or self.maxiter<1:
            raise ValueError("kappa>=0, power in [0,1], ridge>0, ftol>0, maxiter>=1")
        if self.prior not in ("bt","neutral"):
            raise ValueError("prior must be bt or neutral")


@dataclass
class DPTModel:
    config: DPTConfig
    pctm_config: PCTMConfig
    catalog: np.ndarray
    evidence: sparse.csr_matrix
    row_background: np.ndarray
    log_popularity: np.ndarray
    theta: np.ndarray
    max_history: int
    diagnostics: dict


def cache_for(store,root):
    return Path(root)/_store_identity(store)["sha256"][:24]


def pair_arrays(counts):
    total=(counts+counts.T).tocsr()
    total.sum_duplicates();total.sort_indices()
    row=np.repeat(np.arange(counts.shape[0]),np.diff(total.indptr))
    column=total.indices.copy()
    forward=np.asarray(counts[row,column]).ravel()
    return total,row,column,forward


def bt_objective(theta,i,j,total,forward,ridge):
    delta=theta[j]-theta[i]
    loss=float(np.sum(total*np.logaddexp(0,delta)-forward*delta)+0.5*ridge*np.dot(theta,theta))
    change=total*expit(delta)-forward
    gradient=(np.bincount(j,weights=change,minlength=len(theta))
              -np.bincount(i,weights=change,minlength=len(theta))+ridge*theta)
    return loss,gradient


def fit_direction(counts,config):
    total,row,col,forward=pair_arrays(counts)
    keep=row<col
    args=(row[keep],col[keep],total.data[keep],forward[keep],config.ridge)
    start=time.perf_counter()
    if config.prior=="neutral":
        theta=np.zeros(counts.shape[0]);info={"prior":"neutral","iterations":0,"converged":True}
    else:
        solved=minimize(bt_objective,np.zeros(counts.shape[0]),args=args,jac=True,
            method="L-BFGS-B",options={"maxiter":config.maxiter,"ftol":config.ftol,"maxcor":10})
        if not np.all(np.isfinite(solved.x)) or not np.isfinite(solved.fun):
            raise FloatingPointError("Direction optimization produced nonfinite values")
        theta=solved.x
        info={"prior":"bt","iterations":int(solved.nit),"converged":bool(solved.success),
              "termination":str(solved.message),"objective":float(solved.fun),
              "gradient_inf_norm":float(np.max(np.abs(solved.jac)))}
        if not solved.success:
            raise RuntimeError(f"Direction optimization did not converge: {info}. Increase maxiter before test evaluation.")
    probability=expit(theta[col]-theta[row])
    strength=total.data/(total.data+config.kappa)
    shrunk=total.copy()
    shrunk.data=strength*forward+(1-strength)*total.data*probability
    # i==j: total=2Cii and probability=1/2 gives shrunk Cii=Cii exactly.
    shrunk.eliminate_zeros()
    info.update({"direction_seconds":time.perf_counter()-start,"pair_support_nnz":int(total.nnz),
                 "new_direction_edges":int(shrunk.nnz-counts.nnz)})
    return shrunk,theta,info


def make_evidence(counts,tau,power):
    evidence=counts.copy().astype(np.float64)
    if power==0:
        evidence.data=np.log1p(evidence.data*counts.shape[0]/tau)
        background=np.ones(counts.shape[0])
    else:
        mass=np.asarray(counts.sum(axis=1)).ravel()
        background=(tau/(mass+tau))**power
        evidence.data=np.expm1(power*np.log1p(evidence.data*counts.shape[0]/tau))
        evidence=(sparse.diags(background)@evidence).tocsr()
    return evidence,background


def score_histories(model,histories):
    scores=(histories@model.evidence).tocsr()
    if model.config.power>0:
        z=np.asarray(histories@model.row_background).ravel()
        scores=(sparse.diags(1/z)@scores).tocsr()
        scores.data=np.log1p(scores.data)/model.config.power
    return scores


def array_bytes(model):
    e=model.evidence
    return int(e.data.nbytes+e.indices.nbytes+e.indptr.nbytes+model.catalog.nbytes
               +model.row_background.nbytes+model.log_popularity.nbytes+model.theta.nbytes)
