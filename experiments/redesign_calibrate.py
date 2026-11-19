"""Train/validate a convex recency calibrator on the inner holdout only."""
from pathlib import Path
import sys,json,time,argparse,copy,hashlib,datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from dpt_sr.data import ROOT,load_split,sha256
from dpt_sr.model import DPTConfig,DPTModel,fit_direction,make_evidence,cache_for
from dpt_sr.calibration import CalibrationConfig,CalibratedModel
from dpt_sr.full_model import CompletionConfig,CompletionModel,completion_counts,save_completion
from dpt_sr.budget_model import apply_budget,BudgetConfig
from dpt_sr.recency_calibration import kernels_for,history_experts,dense_features,prior_vector,fit_weights,score_calibrated
from capacity_probes.models.pctm import PCTMConfig,log_popularity,count_weights
from capacity_probes.sequences import SequenceStore,EvaluationSet
from capacity_probes.ranking import tie_ranks,topk
from capacity_probes.counts import OffsetCache
from experiments.redesign_probe import ranks_dense,metrics

def get_spec(d):
    if d.startswith('ml'):return copy.deepcopy(json.loads((ROOT/'configs/extension_protocol.json').read_text())['external_config'][d])
    return copy.deepcopy(json.loads((ROOT/'configs/final_model.json').read_text())[d])

def from_graph(store,spec,c,d,theta,info):
    pc=PCTMConfig(**spec['pctm']);dc=DPTConfig(**spec['dpt']);hc=CalibrationConfig(**spec['calibration']);cc=CompletionConfig(**spec['completion'])
    total=(c+c.T).tocsr();mass=np.asarray(d.sum(axis=1)).ravel();a=d.copy()
    if hc.pair_penalty:
        row=np.repeat(np.arange(store.num_items),np.diff(d.indptr));s=np.asarray(total[row,d.indices]).ravel()
        a.data*=s/(s+hc.pair_penalty);del row,s
    e,b=make_evidence(a,pc.tau,dc.power);del a
    paths,pinfo=completion_counts(d,total,cc);paths.data*=cc.strength;pe,_=make_evidence(paths,pc.tau,0.);del paths,total
    g=((mass+1)/(mass+pc.tau+1))**(-hc.exposure_power)
    base=DPTModel(dc,pc,store.catalog,e,b,log_popularity(store),theta,spec['max_history'],info)
    return CompletionModel(CalibratedModel(base,hc,g),cc,pe,dict(direction=info,completion=pinfo))

def load_inner(d):
    out=ROOT/'results/redesign'/d;out.mkdir(parents=True,exist_ok=True);p=out/'inner_store.npz'
    if p.exists():
        with np.load(p,allow_pickle=False) as a:
            users=a['users'];gu=a['group_users'];store=SequenceStore(a['catalog'],users,a['items'],gu,a['starts'],a['ends'],{int(u):i for i,u in enumerate(gu)})
            ev=EvaluationSet(store,a['eval_users'],a['eval_groups'],tuple(np.asarray([t],np.int32) for t in a['targets']))
        verified=json.loads((out/'inner_data.json').read_text());return store,ev,verified
    train,held,verified=load_split(d,ROOT/'data','valid');store=SequenceStore.from_frame(train);ev=EvaluationSet.from_holdout(store,held)
    np.savez(p,catalog=store.catalog,users=store.users,items=store.items,group_users=store.group_users,starts=store.starts,ends=store.ends,
             eval_users=ev.users,eval_groups=ev.groups,targets=np.asarray([t[0] for t in ev.targets]))
    (out/'inner_data.json').write_text(json.dumps(verified,indent=2));return store,ev,verified

def main(a):
    tick=time.perf_counter();out=ROOT/'results/redesign'/a.dataset;out.mkdir(parents=True,exist_ok=True)
    store,ev,verified=load_inner(a.dataset);all_users=len(ev.users)
    if len(ev.users)>8192:
        ix=np.sort(np.random.default_rng(20260909).choice(len(ev.users),8192,replace=False))
        ev=EvaluationSet(store,ev.users[ix],ev.groups[ix],tuple(ev.targets[i] for i in ix))
    spec=get_spec(a.dataset)
    # Pooling screening is recorded separately. Calibration uses the unchanged
    # starting scorer so that improvements cannot be credited to a wider search.
    kernels=kernels_for(spec['pctm']['kernel']);n=store.num_items;ties=tie_ranks(store.catalog,'sha256')
    protocol=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataset=a.dataset,
        selection_split='80/20 deterministic user partition of training-internal LOO',sample_users=len(ev.users),all_inner_users=all_users,
        data_sha256=verified,kernels=kernels,regularization=[.001,.01,.1],temperature=5.,hard_negatives=32,uniform_negatives=32,
        objective='pairwise logistic; simplex recency weights; path weight [0,2]; popularity [-1,1]',
        primary_selection_metric='full-catalogue NDCG@10 on inner validation users',
        independent_screening_not_used_for_configuration=True)
    (out/'calibration_protocol.json').write_text(json.dumps(protocol,indent=2))
    graph=out/'valid_graph';graph.mkdir(exist_ok=True)
    if (graph/'directed.npz').exists():
        c=sparse.load_npz(graph/'counts.npz');d=sparse.load_npz(graph/'directed.npz');theta=np.load(graph/'theta.npy');info=json.loads((graph/'fit.json').read_text())
    else:
        oc=OffsetCache(cache_for(store,ROOT/'redesign_cache'),store);w=count_weights(spec['pctm']['counts']);oc.build(len(w));c=oc.combine(spec['pctm']['counts'],w).astype(float)
        d,theta,info=fit_direction(c,DPTConfig(**spec['dpt']));sparse.save_npz(graph/'counts.npz',c);sparse.save_npz(graph/'directed.npz',d)
        np.save(graph/'theta.npy',theta);(graph/'fit.json').write_text(json.dumps(info,indent=2))
    model=apply_budget(from_graph(store,spec,c,d,theta,info));del c,d
    hh=history_experts(model,ev,kernels);prior=prior_vector(model,kernels)
    trainmask=np.asarray([int(hashlib.sha256(f'calibration:{int(u)}'.encode()).hexdigest()[:8],16)%5!=0 for u in ev.users])
    pairs=[];owner=[];rng=np.random.default_rng(20260909);start_features=time.perf_counter()
    for start in range(0,len(ev.users),64):
        stop=min(start+64,len(ev.users));ff=dense_features(model,hh,start,stop)
        for b in range(stop-start):
            row=start+b;t=int(ev.targets[row][0]);seen=store.group_items(int(ev.groups[row]))
            score=ff[0][b]+ff[-2][b]+spec['pctm']['pop_boost']*ff[-1][b]
            score=score.copy();score[seen]=-np.inf;score[t]=-np.inf
            hard=topk(np.arange(n),score,32,ties)
            blocked=set(seen.tolist()+[t]);random=[]
            while len(random)<32:
                j=int(rng.integers(n))
                if j not in blocked:random.append(j);blocked.add(j)
            neg=np.unique(np.r_[hard,random]);neg=neg[np.isfinite(score[neg])]
            f=np.stack([v[b,t]-v[b,neg] for v in ff],axis=1)
            pairs.append(f);owner.extend([row]*len(neg))
        if start%1024==0:print('FEATURES',a.dataset,stop,len(ev.users),flush=True)
    xx=np.concatenate(pairs);owner=np.asarray(owner);mask=trainmask[owner];del pairs
    np.savez_compressed(out/'calibration_pairs.npz',differences=xx.astype(np.float32),owner=owner,user_id=ev.users,fit_user_mask=trainmask)
    weights={};fits={}
    for lam in protocol['regularization']:
        w,fit=fit_weights(xx[mask],prior,lam);weights[str(lam)]=w;fits[str(lam)]=fit
    iv=np.flatnonzero(~trainmask);val=EvaluationSet(store,ev.users[iv],ev.groups[iv],tuple(ev.targets[i] for i in iv));vh=[h[iv] for h in hh]
    valr={k:[] for k in weights};valr['fixed']=[]
    for start in range(0,len(iv),64):
        stop=min(start+64,len(iv));ff=dense_features(model,vh,start,stop)
        for name,w in dict(weights,fixed=prior).items():
            score=sum(float(ww)*f for ww,f in zip(w,ff));valr[name].append(ranks_dense(score,val,start,ties))
    validation={k:metrics(np.concatenate(v)) for k,v in valr.items()}
    best=max(weights,key=lambda k:validation[k]['ndcg@10'])
    # Select ridge on held-out users, then refit on all sampled inner users.
    final,fit=fit_weights(xx,prior,float(best));del xx
    result=dict(dataset=a.dataset,spec=spec,kernels=kernels,weights=final.tolist(),prior=prior.tolist(),chosen_regularization=float(best),
        validation=validation,validation_weights={k:v.tolist() for k,v in weights.items()},fits=fits,refit=fit,
        validation_users=len(iv),inner_sample_users=len(ev.users),feature_and_calibration_seconds=time.perf_counter()-start_features,
        preparation_fit_calibration_seconds=time.perf_counter()-tick,
        scope='New model development uses training-internal holdout only; historical test inspection remains disclosed.',
        algorithm_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'dpt_sr/budget_model.py',ROOT/'dpt_sr/recency_calibration.py']})
    (out/'calibrated_config.json').write_text(json.dumps(result,indent=2))
    print('CALIBRATED',a.dataset,json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['beauty','sports','toys','ml1m','ml20m'],required=True)
    with threadpool_limits(limits=1):main(p.parse_args())
