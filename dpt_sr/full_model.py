"""Complete DPT-SR. Modified release: standalone modules and interfaces.

Classical Bradley--Terry and graph walks are explicitly acknowledged.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import json,time
import numpy as np
from scipy import sparse
from capacity_probes.counts import OffsetCache
from capacity_probes.models.pctm import PCTMConfig,count_weights,kernel_weights,log_popularity
from capacity_probes.ranking import RecommendationCollector,tie_ranks,topk
from .model import DPTModel,fit_direction,make_evidence,cache_for,score_histories
from .calibration import CalibratedModel,calibrate_history,array_bytes_calibrated,save_calibrated,load_calibrated

@dataclass(frozen=True)
class CompletionConfig:
    strength: float=1.
    fanout: int=32
    exclude_observed: bool=True
    renormalize_paths: bool=False
    row_restore: float=0.
    def __post_init__(self):
        if not np.isfinite(self.strength) or self.strength<0:raise ValueError('Nonnegative finite strength required')
        if isinstance(self.fanout,bool) or int(self.fanout)!=self.fanout or self.fanout<1:raise ValueError('Positive integer fanout required')
        if not np.isfinite(self.row_restore) or not 0<=self.row_restore<=1:raise ValueError('row_restore must be in [0,1]')

@dataclass
class CompletionModel:
    base: CalibratedModel
    completion_config: CompletionConfig
    completion_evidence: sparse.csr_matrix
    diagnostics: dict
    def __getattr__(self,name):return getattr(self.base,name)

def sparse_topk(matrix,k):
    x=matrix.tocsr(copy=True);x.sum_duplicates();x.eliminate_zeros();x.sort_indices()
    if np.any(x.data<0) or not np.isfinite(x.data).all():raise ValueError('Nonnegative finite matrix required')
    values=[];columns=[];ptr=[0]
    for i in range(x.shape[0]):
        lo,hi=x.indptr[i:i+2];v=x.data[lo:hi];j=x.indices[lo:hi]
        if len(v)>k:
            threshold=np.partition(v,len(v)-k)[len(v)-k]
            greater=np.flatnonzero(v>threshold);equal=np.flatnonzero(v==threshold)[:k-len(greater)]
            keep=np.sort(np.concatenate((greater,equal)));v,j=v[keep],j[keep]
        values.extend(v);columns.extend(j);ptr.append(len(values))
    return sparse.csr_matrix((np.asarray(values,float),np.asarray(columns,np.int32),np.asarray(ptr,np.int32)),shape=x.shape)

def completion_counts(directed,forbidden,config,batch_size=128):
    mass=np.asarray(directed.sum(axis=1)).ravel();p=sparse_topk(directed,config.fanout)
    kept=np.asarray(p.sum(axis=1)).ravel();denom=kept if config.renormalize_paths else mass
    p=(sparse.diags(np.divide(1.,denom,out=np.zeros_like(mass),where=denom>0))@p).tocsr()
    parts=[];max_product=0
    for start in range(0,p.shape[0],batch_size):
        z=(p[start:start+batch_size]@p).tocsr();max_product=max(max_product,z.nnz)
        if config.exclude_observed:
            support=forbidden[start:start+batch_size].copy();support.data[:]=1
            z=z-z.multiply(support);z.eliminate_zeros()
        co=z.tocoo();keep=co.col!=co.row+start
        z=sparse.csr_matrix((co.data[keep],(co.row[keep],co.col[keep])),shape=z.shape)
        parts.append(sparse_topk(z,config.fanout))
    q=sparse.vstack(parts,format='csr');qmass=np.asarray(q.sum(axis=1)).ravel()
    if np.any(qmass>1+1e-12):raise AssertionError('Substochastic bound violated')
    result=(sparse.diags(mass)@q).tocsr()
    return result,{'fanout':config.fanout,'completion_nnz':result.nnz,'nnz_bound':len(mass)*config.fanout,
        'max_block_product_nnz':max_product,'block_product_nnz_bound':batch_size*config.fanout**2,
        'retained_one_hop_mass_quantiles':np.quantile(np.divide(kept,mass,out=np.zeros_like(mass),where=mass>0),[0,.25,.5,.75,1]).tolist(),
        'completion_probability_mass_quantiles':np.quantile(qmass,[0,.25,.5,.75,1]).tolist(),
        'max_completion_probability_mass':float(qmass.max()),'exclude_observed':config.exclude_observed,'renormalize_paths':config.renormalize_paths}

def fit_completion(store,pctm_values,dpt_config,calibration_config,completion_config,max_history,cache_root):
    tick=time.perf_counter();pc=PCTMConfig(**pctm_values);cache=OffsetCache(cache_for(store,cache_root),store)
    weights=count_weights(pc.counts);cache.build(len(weights));c=cache.combine(pc.counts,weights).astype(float)
    if dpt_config.kappa>0:directed,theta,info=fit_direction(c,dpt_config)
    else:directed=c.copy();theta=np.zeros(store.num_items);info={'prior':'disabled','converged':True,'iterations':0}
    total=(c+c.T).tocsr();total.sort_indices();mass=np.asarray(directed.sum(axis=1)).ravel()
    scale=((mass+1)/(mass+pc.tau+1))**(-calibration_config.exposure_power);adjusted=directed.copy()
    if calibration_config.pair_penalty>0:
        rows=np.repeat(np.arange(store.num_items),np.diff(adjusted.indptr))
        support=np.asarray(total[rows,adjusted.indices]).ravel();adjusted.data*=support/(support+calibration_config.pair_penalty)
        del rows,support
    if completion_config.row_restore>0:
        gated=np.asarray(adjusted.sum(axis=1)).ravel()
        restore=np.divide(mass,gated,out=np.ones_like(mass),where=gated>0)**completion_config.row_restore
        adjusted=(sparse.diags(restore)@adjusted).tocsr()
    evidence,background=make_evidence(adjusted,pc.tau,dpt_config.power);del adjusted,c
    comp_tick=time.perf_counter()
    if completion_config.strength>0:
        added,compinfo=completion_counts(directed,total,completion_config);added.data*=completion_config.strength
        comp_evidence,_=make_evidence(added,pc.tau,0.);del added
    else:comp_evidence=sparse.csr_matrix(directed.shape,dtype=float);compinfo={'completion_nnz':0,'disabled':True}
    compinfo['build_seconds']=time.perf_counter()-comp_tick
    base=DPTModel(dpt_config,pc,store.catalog.copy(),evidence,background,log_popularity(store),theta,max_history,info)
    diag={'direction':info,'completion':compinfo,'fit_seconds':time.perf_counter()-tick,'base_evidence_nnz':evidence.nnz,
        'total_stored_evidence_nnz':evidence.nnz+comp_evidence.nnz,'dense_item_item_matrix':False,'gpu_used':False,'extra_learned_parameters':0,'completion_config':asdict(completion_config)}
    return CompletionModel(CalibratedModel(base,calibration_config,scale),completion_config,comp_evidence,diag)

def score_completion(model,calibrated_history):
    out=score_histories(model.base.base,calibrated_history)
    if model.completion_evidence.nnz:out=(out+calibrated_history@model.completion_evidence).tocsr()
    return out

def recommend_completion(model,evaluation,batch_size=256):
    if not np.array_equal(model.catalog,evaluation.store.catalog):raise ValueError('Catalog mismatch')
    h=evaluation.history_matrix(model.max_history,lambda n:kernel_weights(model.pctm_config.kernel,n))
    if model.calibration_config.exposure_power>0:h=calibrate_history(h,model.history_scale)
    ranks=tie_ranks(model.catalog,'sha256');boost=model.pctm_config.pop_boost;pop=model.log_popularity
    order=np.lexsort((ranks,pop)) if boost<0 else np.lexsort((ranks,-pop)) if boost>0 else np.argsort(ranks)
    blocked=np.zeros(len(pop),dtype=bool);collector=RecommendationCollector(evaluation,10)
    for start in range(0,len(evaluation.users),batch_size):
        batch=score_completion(model,h[start:start+batch_size])
        for local,row in enumerate(range(start,min(start+batch_size,len(evaluation.users)))):
            lo,hi=batch.indptr[local:local+2];ids,values=batch.indices[lo:hi],batch.data[lo:hi]
            seen=np.unique(evaluation.store.group_items(int(evaluation.groups[row])));blocked[seen]=True
            keep=~blocked[ids];ids,values=ids[keep],values[keep];reachable=bool(np.any(np.isin(evaluation.targets[row],ids)))
            blocked[ids]=True;bg=[]
            for item in order:
                if not blocked[item]:
                    bg.append(int(item))
                    if len(bg)==10:break
            ids_all=np.concatenate((ids,np.asarray(bg,np.int32)));v=np.concatenate((values,np.zeros(len(bg))))
            v+=np.float32(boost)*pop[ids_all];pick=topk(ids_all,v,10,ranks)
            collector.add(row,ids_all[pick],v[pick],reachable);blocked[seen]=False;blocked[ids]=False
    return collector.finish()

def array_bytes_completion(model):
    e=model.completion_evidence
    return array_bytes_calibrated(model.base)+e.data.nbytes+e.indices.nbytes+e.indptr.nbytes

def save_completion(model,path):
    path=Path(path);path.mkdir(parents=True,exist_ok=False);save_calibrated(model.base,path/'base')
    sparse.save_npz(path/'completion.npz',model.completion_evidence)
    (path/'completion.json').write_text(json.dumps({'format':'DPT-completion-1','config':asdict(model.completion_config),'diagnostics':model.diagnostics},indent=2),encoding='utf-8')

def load_completion(path):
    path=Path(path);meta=json.loads((path/'completion.json').read_text(encoding='utf-8'))
    if meta['format']!='DPT-completion-1':raise ValueError('Unsupported format')
    base=load_calibrated(path/'base');e=sparse.load_npz(path/'completion.npz').tocsr()
    if e.shape!=base.evidence.shape or np.any(e.data<0) or not np.isfinite(e.data).all():raise ValueError('Invalid completion matrix')
    return CompletionModel(base,CompletionConfig(**meta['config']),e,meta['diagnostics'])
