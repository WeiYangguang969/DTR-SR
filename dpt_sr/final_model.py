"""Final DPT-SR: directional counts, conditional path budgets, interval calibration."""
import numpy as np
import pandas as pd
from .model import DPTConfig,score_histories
from .calibration import CalibrationConfig
from .full_model import CompletionConfig,load_completion,save_completion
from .budget_model import fit_budget
from .recency_calibration import history_experts,weighted_history
from capacity_probes.ranking import topk,tie_ranks

def fit_final(store,configuration,cache_root):
    s=configuration['spec']
    m=fit_budget(store,s['pctm'],DPTConfig(**s['dpt']),CalibrationConfig(**s['calibration']),
        CompletionConfig(**s['completion']),s['max_history'],cache_root)
    m.diagnostics['rank_calibration']={k:configuration[k] for k in ['kernels','weights','prior','selected_axis','refit_parameter']}
    m.diagnostics['calibration_fitted_scalar_count']=0 if configuration['selected_axis']=='fixed' else 1
    m.diagnostics['extra_learned_parameters']=m.diagnostics['calibration_fitted_scalar_count']
    m.diagnostics['direction_parameter_count']=len(m.catalog)
    return m

def recommend_final(model,evaluation,k=20,batch_size=128):
    if not np.array_equal(model.catalog,evaluation.store.catalog):raise ValueError('Catalogue mismatch')
    cal=model.diagnostics['rank_calibration'];allw=np.asarray(cal['weights'])
    active=[0]+[i for i in range(1,len(cal['kernels'])) if allw[i]>1e-12]
    kernels=[cal['kernels'][i] for i in active];w=np.r_[allw[active],allw[-2:]]
    h=history_experts(model,evaluation,kernels);ties=tie_ranks(model.catalog,'sha256')
    pop=model.log_popularity;beta=w[-1]
    background_order=np.lexsort((ties,pop)) if beta<0 else np.lexsort((ties,-pop)) if beta>0 else np.argsort(ties)
    blocked=np.zeros(len(model.catalog),bool)
    users=[];items=[];scores=[];ranks=[]
    for start in range(0,len(evaluation.users),batch_size):
        stop=min(start+batch_size,len(evaluation.users));hh=[x[start:stop] for x in h]
        if model.config.power==0:x=score_histories(model.base.base,weighted_history(hh,w))
        else:x=sum(float(ww)*score_histories(model.base.base,z) for ww,z in zip(w,hh) if ww>1e-12)
        x=(x+w[-2]*(hh[0]@model.completion_evidence)).tocsr()
        for local,row in enumerate(range(start,stop)):
            seen=np.unique(evaluation.store.group_items(int(evaluation.groups[row])));blocked[seen]=True
            lo,hi=x.indptr[local:local+2];ids=x.indices[lo:hi];value=x.data[lo:hi];keep=~blocked[ids];ids,value=ids[keep],value[keep]
            blocked[ids]=True;bg=[];eligible=min(k,len(model.catalog)-len(seen))
            if eligible:
                for j in background_order:
                    if not blocked[j]:
                        bg.append(int(j))
                        if len(bg)==eligible:break
                candidates=np.r_[ids,np.asarray(bg,np.int32)];s=np.r_[value,np.zeros(len(bg))]+beta*pop[candidates]
                pick=topk(candidates,s,eligible,ties)
                users.extend([evaluation.users[row]]*len(pick));items.extend(model.catalog[candidates[pick]]);scores.extend(s[pick]);ranks.extend(range(1,len(pick)+1))
            blocked[seen]=False;blocked[ids]=False
    return pd.DataFrame(dict(user_id=users,item_id=items,score=scores,rank=ranks))

load_final=load_completion

def save_final(model,path):
    """Complete each ZIP archive in memory before its atomic file replacement."""
    from pathlib import Path
    from dataclasses import asdict
    import io,json,zipfile
    from scipy import sparse
    path=Path(path);path.mkdir(parents=True,exist_ok=False);(path/'base/base').mkdir(parents=True)
    def put(rel,raw):
        p=path/rel;q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(raw);q.replace(p)
    def metadata(rel,value):put(rel,(json.dumps(value,indent=2)+'\n').encode())
    def npz(rel,values=None,matrix=None):
        b=io.BytesIO()
        if matrix is not None:sparse.save_npz(b,matrix)
        else:np.savez_compressed(b,**values)
        raw=b.getvalue()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if z.testzip() is not None:raise IOError('Incomplete checkpoint archive')
        put(rel,raw)
    m=model.base.base
    metadata('base/base/metadata.json',dict(format=1,config=asdict(m.config),pctm_config=asdict(m.pctm_config),max_history=m.max_history,diagnostics=m.diagnostics))
    npz('base/base/arrays.npz',dict(catalog=m.catalog,theta=m.theta,row_background=m.row_background,log_popularity=m.log_popularity))
    npz('base/base/evidence.npz',matrix=m.evidence)
    metadata('base/calibration.json',dict(format='DPT-SR-calibration-1',calibration_config=asdict(model.calibration_config)))
    npz('base/history_scale.npz',dict(history_scale=model.history_scale))
    npz('completion.npz',matrix=model.completion_evidence)
    metadata('completion.json',dict(format='DPT-completion-1',config=asdict(model.completion_config),diagnostics=model.diagnostics))
