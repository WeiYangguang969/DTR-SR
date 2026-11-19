"""Training-only screening of pooling and evidence budgets; never reads test labels."""
from pathlib import Path
import sys, json, time, argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from dpt_sr.data import load_split, ROOT, sha256
from dpt_sr.model import DPTConfig, fit_direction, cache_for, make_evidence
from dpt_sr.calibration import calibrate_history
from dpt_sr.full_model import CompletionConfig, completion_counts
from capacity_probes.counts import OffsetCache
from capacity_probes.sequences import SequenceStore, EvaluationSet
from capacity_probes.models.pctm import count_weights, kernel_weights, log_popularity
from capacity_probes.ranking import tie_ranks

def metrics(r):
    return {f'{m}@{k}':float(np.mean(np.where((r>0)&(r<=k),1/np.log2(r+1),0) if m=='ndcg' else ((r>0)&(r<=k))))
            for k in (5,10,20) for m in ('ndcg','recall')}

def ranks_dense(x,ev,start,ties):
    """Exact full-catalogue target rank; avoid top-k sorting during screening."""
    result=[]
    for b in range(len(x)):
        row=start+b;target=int(ev.targets[row][0]);s=x[b]
        s[np.unique(ev.store.group_items(int(ev.groups[row])))]=-np.inf
        v=s[target]
        result.append(1+np.count_nonzero(s>v)+np.count_nonzero((s==v)&(ties<ties[target])))
    return np.asarray(result,np.int32)

def main(a):
    out=ROOT/'results/redesign'/a.dataset;out.mkdir(parents=True,exist_ok=True)
    train,held,verified=load_split(a.dataset,ROOT/'data','valid')
    store=SequenceStore.from_frame(train);ev=EvaluationSet.from_holdout(store,held)
    if a.users and len(ev.users)>a.users:
        rng=np.random.default_rng(20260909);idx=np.sort(rng.choice(len(ev.users),a.users,replace=False))
        ev=EvaluationSet(store,ev.users[idx],ev.groups[idx],tuple(ev.targets[i] for i in idx))
    spec=json.loads((ROOT/'configs/extension_protocol.json').read_text())['external_config'][a.dataset]
    (out/'screen_protocol.json').write_text(json.dumps(dict(dataset=a.dataset,split='training-internal leave-one-out',users=len(ev.users),seed=20260909,
        powers=[0,.5,1],tau_multipliers=[1,10],kernels=[spec['pctm']['kernel'],'exp:0.6','exp:0.85','inv:0.5'],
        popularity=[-.7,-.5,-.4,-.3,-.2,-.1,0,.1],data_sha256=verified),indent=2))
    modeldir=out/'valid_graph';modeldir.mkdir(exist_ok=True)
    if (modeldir/'directed.npz').exists():d=sparse.load_npz(modeldir/'directed.npz');c=sparse.load_npz(modeldir/'counts.npz')
    else:
        w=count_weights(spec['pctm']['counts']);oc=OffsetCache(cache_for(store,ROOT/'redesign_cache'),store);oc.build(len(w))
        c=oc.combine(spec['pctm']['counts'],w).astype(float);tick=time.perf_counter()
        d,theta,info=fit_direction(c,DPTConfig(**spec['dpt']))
        sparse.save_npz(modeldir/'directed.npz',d);sparse.save_npz(modeldir/'counts.npz',c)
        np.save(modeldir/'theta.npy',theta);(modeldir/'fit.json').write_text(json.dumps(info,indent=2))
        print('DIRECTION',a.dataset,time.perf_counter()-tick,flush=True)
    total=(c+c.T).tocsr();rr=np.repeat(np.arange(store.num_items),np.diff(d.indptr));support=np.asarray(total[rr,d.indices]).ravel()
    adjusted=d.copy();adjusted.data*=support/(support+spec['calibration']['pair_penalty']);mass=np.asarray(d.sum(axis=1)).ravel()
    del rr,support,total
    hists={}
    for ker in [spec['pctm']['kernel'],'exp:0.6','exp:0.85','inv:0.5']:
        h=ev.history_matrix(spec['max_history'],lambda n:kernel_weights(ker,n))
        scale=((mass+1)/(mass+spec['pctm']['tau']+1))**(-spec['calibration']['exposure_power'])
        hists[ker]=calibrate_history(h,scale)
    pop=log_popularity(store);ties=tie_ranks(store.catalog,'sha256');records=[]
    for power in (0,.5,1):
      for tm in (1,10):
        e,bg=make_evidence(adjusted,spec['pctm']['tau']*tm,power)
        for kernel,h in hists.items():
            tick=time.perf_counter();buff={b:[] for b in (-.7,-.5,-.4,-.3,-.2,-.1,0,.1)}
            for start in range(0,len(ev.users),128):
                hh=h[start:start+128];x=(hh@e).toarray()
                if power:x=np.log1p(x/np.asarray(hh@bg).ravel()[:,None])/power
                for beta in buff:buff[beta].append(ranks_dense(x+np.float32(beta)*pop,ev,start,ties))
            for beta,parts in buff.items():
                r=np.concatenate(parts);rec=dict(power=power,tau_multiplier=tm,kernel=kernel,pop_boost=beta,**metrics(r));records.append(rec)
            print('SCREEN',a.dataset,power,tm,kernel,'best',max(records[-8:],key=lambda r:r['ndcg@10']),'seconds',time.perf_counter()-tick,flush=True)
            (out/'screen_results.json').write_text(json.dumps(records,indent=2))
    (out/'screen_complete.json').write_text(json.dumps(dict(best=max(records,key=lambda r:r['ndcg@10']),candidates=len(records),users=len(ev.users)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['ml1m','ml20m'],required=True);p.add_argument('--users',type=int,default=8192)
    with threadpool_limits(limits=1):main(p.parse_args())
