"""Conditional path evidence with a source uncertainty budget.

The residual is log(1 + n eta u_i B_ij/(tau+n A_ij)). Unlike an
independent path expert it discounts evidence already supplied by A.
No dataset-name switch is used by fitting or scoring.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from .full_model import (CompletionConfig, fit_completion, save_completion, load_completion)
from .model import make_evidence

@dataclass(frozen=True)
class BudgetConfig:
    conditional: bool = True
    uncertainty: bool = True

def direct_counts(model):
    e=model.evidence.copy();r=model.config.power;n=e.shape[0];tau=model.pctm_config.tau
    if r:
        row=np.repeat(np.arange(n),np.diff(e.indptr))
        e.data=np.expm1(np.log1p(e.data/model.row_background[row])/r)*tau/n
    else:e.data=np.expm1(e.data)*tau/n
    return e

def apply_budget(model,budget=BudgetConfig()):
    """In-place replace a fitted, independent path expert by a bounded residual."""
    if model.diagnostics.get('budget_config') is not None:raise ValueError('Budget already applied')
    e=model.completion_evidence.copy();n=e.shape[0];tau=model.pctm_config.tau
    row=np.repeat(np.arange(n),np.diff(e.indptr))
    # Existing path evidence contains eta B; invert its log transform exactly.
    x=np.expm1(e.data)
    a=direct_counts(model)
    if budget.conditional:
        observed=np.asarray(a[row,e.indices]).ravel()
        x/=1+n*observed/tau
    if budget.uncertainty:
        # The gated direct mass is observed support, not a fitted exposure model.
        mass=np.asarray(a.sum(axis=1)).ravel()
        u=tau/(tau+mass)
        x*=u[row]
    else:u=np.ones(n)
    e.data=np.log1p(x);e.eliminate_zeros()
    model.completion_evidence=e
    model.diagnostics=dict(model.diagnostics,budget_config=asdict(budget),
        source_budget_quantiles=np.quantile(u,[0,.25,.5,.75,1]).tolist(),
        path_residual_max=float(e.data.max()) if e.nnz else 0.,
        score_definition='S_power(A) + H log(1+n*eta*u*B/(tau+n*A)) + beta log(popularity)')
    return model

def fit_budget(store,pctm_values,dpt_config,calibration_config,completion_config,max_history,cache_root,budget=BudgetConfig()):
    model=fit_completion(store,pctm_values,dpt_config,calibration_config,completion_config,max_history,cache_root)
    return apply_budget(model,budget)

# The serialized completion matrix is the actual scoring residual. The existing
# portable checkpoint and exact top-k implementation therefore require no fork.
save_budget=save_completion
load_budget=load_completion
