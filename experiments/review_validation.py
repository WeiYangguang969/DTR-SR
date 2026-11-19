"""Frozen supplementary evaluation. Production model files are never edited."""
from pathlib import Path
import argparse, copy, datetime, hashlib, json, os, sys, time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import replace, asdict
import numpy as np
from scipy import sparse
from scipy.special import expit
from threadpoolctl import threadpool_limits
from dpt_sr.data import load_split, sha256
from dpt_sr.model import DPTConfig, cache_for, fit_direction
from dpt_sr.calibration import CalibrationConfig, calibrate_history
from dpt_sr.full_model import CompletionConfig, fit_completion, recommend_completion, save_completion, load_completion, score_completion, array_bytes_completion
from dpt_sr.runtime import environment, peak_rss_mib, write_json
from capacity_probes.counts import OffsetCache
from capacity_probes.sequences import SequenceStore, EvaluationSet
from capacity_probes.models.pctm import build_pctm, recommend_pctm, kernel_weights, count_weights, _recommend_pctm_rows
from capacity_probes.models.seqrules import build_seqrules, recommend_seqrules, _recommend_seqrules_rows, history_matrix as rules_history
from capacity_probes.ranking import RecommendationCollector, tie_ranks, topk
from capacity_probes.metrics import official_metrics
from experiments.paired_stats import paired_summary, holm
ROOT=Path(__file__).resolve().parents[1]
DS=['beauty','sports','toys']

def freeze():
    path=ROOT/'configs/extension_protocol.json';p=json.loads(path.read_text())
    for name,value in p['core_sha256'].items():
        if sha256(ROOT/name)!=value:raise ValueError('Frozen source changed: '+name)
    return p,sha256(path)

def spec_for(d,var):
    p,_=freeze()
    conf=copy.deepcopy(p['external_config'][d] if d not in DS else json.loads((ROOT/'configs/final_model.json').read_text())[d])
    if var=='no_direction':conf['dpt']['kappa']=0.
    if var=='no_history':conf['calibration']['exposure_power']=0.
    if var=='no_path':conf['completion']['strength']=0.
    return conf

def fit(store,spec,var,cache):
    if var=='pctm':return build_pctm(store,cache_for(store,cache),spec['pctm'])
    if var=='seqrules':return build_seqrules(store,cache_for(store,cache),spec['seqrules'])
    return fit_completion(store,spec['pctm'],DPTConfig(**spec['dpt']),CalibrationConfig(**spec['calibration']),CompletionConfig(**spec['completion']),spec['max_history'],cache)

def recommend_k(model,ev,k=20):
    """The published DPT exact sparse/background ranking with only k exposed."""
    h=ev.history_matrix(model.max_history,lambda n:kernel_weights(model.pctm_config.kernel,n))
    if model.calibration_config.exposure_power>0:h=calibrate_history(h,model.history_scale)
    ranks=tie_ranks(model.catalog,'sha256');boost=model.pctm_config.pop_boost;pop=model.log_popularity
    order=np.lexsort((ranks,pop)) if boost<0 else np.lexsort((ranks,-pop)) if boost>0 else np.argsort(ranks)
    blocked=np.zeros(len(pop),bool);collector=RecommendationCollector(ev,k)
    for start in range(0,len(ev.users),256):
        batch=score_completion(model,h[start:start+256])
        for local,row in enumerate(range(start,min(start+256,len(ev.users)))):
            lo,hi=batch.indptr[local:local+2];ids,values=batch.indices[lo:hi],batch.data[lo:hi]
            seen=np.unique(ev.store.group_items(int(ev.groups[row])));blocked[seen]=True
            keep=~blocked[ids];ids,values=ids[keep],values[keep]
            reachable=bool(np.any(np.isin(ev.targets[row],ids)));blocked[ids]=True;bg=[]
            for item in order:
                if not blocked[item]:
                    bg.append(int(item))
                    if len(bg)==k:break
            all_ids=np.concatenate((ids,np.asarray(bg,np.int32)))
            v=np.concatenate((values,np.zeros(len(bg))));v+=np.float32(boost)*pop[all_ids]
            pick=topk(all_ids,v,k,ranks);collector.add(row,all_ids[pick],v[pick],reachable)
            blocked[seen]=False;blocked[ids]=False
    return collector.finish()

def save_ranking(out,reco,train,held,ev,meta,ks=(10,)):
    out.mkdir(parents=True,exist_ok=True)
    hits=reco.merge(held[['user_id','item_id']],on=['user_id','item_id'],how='inner')
    rank=np.zeros(len(ev.users),np.int16)
    rank[np.searchsorted(ev.users,hits.user_id.to_numpy())]=hits['rank'].to_numpy()
    arrays={'user_id':ev.users,'rank_in_top'+str(max(ks)):rank}
    official={}
    for k in ks:
        keep=(rank>0)&(rank<=k);n=np.zeros(len(rank));n[keep]=1/np.log2(rank[keep].astype(float)+1)
        m=official_metrics(reco[reco['rank']<=k],train,held,k)
        np.testing.assert_allclose([n.mean(),keep.mean()],[m[f'ndcg@{k}'],m[f'recall@{k}']],atol=1e-12,rtol=0)
        arrays['ndcg'+str(k)]=n;arrays['recall'+str(k)]=keep.astype(float);official.update(m)
    np.savez_compressed(out/'per_user.npz',**arrays)
    meta.update(official_metrics=official,per_user_sha256=sha256(out/'per_user.npz'))
    write_json(out/'result.json',meta);print(json.dumps({'case':str(out),'metrics':official}),flush=True)

def external(a):
    spec=spec_for(a.dataset,a.variant);_,protocol_sha=freeze();out=a.output/'external'/a.dataset/a.variant
    if (out/'result.json').exists():return
    start=time.perf_counter();train,held,verified=load_split(a.dataset,a.data_dir,'test')
    store=SequenceStore.from_frame(train);ev=EvaluationSet.from_holdout(store,held)
    cache_existed=cache_for(store,a.cache_dir).exists();tick=time.perf_counter();reused=False
    checkpoint=a.output/'external'/a.dataset/'full/model'
    if a.variant in ('no_history','no_path') and checkpoint.exists():
        model=load_completion(checkpoint);reused=True
        model.diagnostics=copy.deepcopy(model.diagnostics)
        model.diagnostics['fitted_components_reused_from']='full'
        if a.variant=='no_history':
            model.base.calibration_config=replace(model.calibration_config,exposure_power=0.)
            model.base.history_scale=np.ones_like(model.history_scale)
        else:
            model.completion_config=replace(model.completion_config,strength=0.)
            model.completion_evidence=sparse.csr_matrix(model.evidence.shape,dtype=float)
            model.diagnostics=copy.deepcopy(model.diagnostics)
            model.diagnostics['completion']={'disabled':True,'completion_nnz':0,'fitted_components_reused':True}
            model.diagnostics['completion_config']=asdict(model.completion_config)
            model.diagnostics['total_stored_evidence_nnz']=model.evidence.nnz
    else:model=fit(store,spec,a.variant,a.cache_dir)
    fitted=time.perf_counter()-tick;print('Fit completed',a.dataset,a.variant,fitted,flush=True);tick=time.perf_counter()
    if a.variant=='pctm':_,reco=recommend_pctm(model,ev,spec['max_history'])
    elif a.variant=='seqrules':_,reco=recommend_seqrules(model,ev,spec['max_history'])
    else:_,reco=recommend_completion(model,ev)
    ranked=time.perf_counter()-tick
    if a.variant=='full' and not checkpoint.exists():save_completion(model,checkpoint)
    e=model.evidence if a.variant=='pctm' else model.rules if a.variant=='seqrules' else None
    arrays=array_bytes_completion(model) if e is None else e.data.nbytes+e.indices.nbytes+e.indptr.nbytes
    meta=dict(dataset=a.dataset,variant=a.variant,spec=spec,split='test',data_sha256=verified,
        protocol_sha256=protocol_sha,fit_rows=len(train),train_users=len(store.group_users),catalog_items=store.num_items,
        holdout_rows=len(held),diagnostics=getattr(model,'diagnostics',{}),environment=environment(),
        resources=dict(fit_seconds=fitted,ranking_seconds=ranked,elapsed_seconds=time.perf_counter()-start,
            peak_process_rss_mib=peak_rss_mib(),model_array_mib=arrays/2**20,cache_existed_at_start=cache_existed,
            fitted_components_reused=reused,blas_threads=1,gpu_used=False),
        scope=freeze()[0]['scope'][a.dataset])
    save_ranking(out,reco,train,held,ev,meta)

def mechanisms(a):
    d=a.dataset;spec=spec_for(d,'full');out=a.output/'mechanism'/d
    if (out/'complete.json').exists():return
    train,held,verified=load_split(d,a.data_dir,'valid');store=SequenceStore.from_frame(train);ev=EvaluationSet.from_holdout(store,held)
    oc=OffsetCache(cache_for(store,a.cache_dir),store);weights=count_weights(spec['pctm']['counts']);oc.build(len(weights))
    c=oc.combine(spec['pctm']['counts'],weights).astype(float);directed,theta,info=fit_direction(c,DPTConfig(**spec['dpt']))
    mass=np.asarray(directed.sum(axis=1)).ravel();total=(c+c.T).tocsr()
    users=[];src=[];dst=[];w=[]
    for row,group in enumerate(ev.groups):
        seq=store.group_items(int(group))[-len(weights):][::-1];target=int(ev.targets[row][0])
        for distance,item in enumerate(seq):
            if item!=target:users.append(int(ev.users[row]));src.append(int(item));dst.append(target);w.append(float(weights[distance]))
    users=np.asarray(users);src=np.asarray(src);dst=np.asarray(dst);w=np.asarray(w)
    i=np.minimum(src,dst);j=np.maximum(src,dst);y=(dst>src).astype(float)
    support=np.asarray(total[i,j]).ravel();forward=np.asarray(c[i,j]).ravel();pi=expit(theta[j]-theta[i]);kappa=spec['dpt']['kappa']
    supported=support>0;unsupported={'events':int((~supported).sum()),'fraction':float((~supported).mean()),'weight_fraction':float(w[~supported].sum()/w.sum())}
    users,y,w,support,forward,pi=[v[supported] for v in (users,y,w,support,forward,pi)]
    probs={'empirical':forward/support,'laplace':(forward+1)/(support+2),'jeffreys':(forward+.5)/(support+1),
        'neutral':(forward+kappa*.5)/(support+kappa),'bt_prior':pi,'bt_shrinkage':(forward+kappa*pi)/(support+kappa)}
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'direction_events.npz',user_id=users,label=y,weight=w,support=support,**probs)
    rows=[]
    for label,mask in [('(0,1]',support<=1),('(1,3]',(support>1)&(support<=3)),('(3,10]',(support>3)&(support<=10)),('>10',support>10),('all',np.ones(len(support),bool))]:
        for name,p in probs.items():
            v=np.clip(p[mask],1e-12,1-1e-12);yy=y[mask];ww=w[mask]
            rows.append(dict(dataset=d,bin=label,estimator=name,events=int(mask.sum()),weight=float(ww.sum()),
                brier=float(np.average((p[mask]-yy)**2,weights=ww)),logloss=float(np.average(-yy*np.log(v)-(1-yy)*np.log1p(-v),weights=ww))))
    write_json(out/'direction.json',rows)
    write_json(out/'direction_design.json',{'dataset':d,'split':'training-only inner leave-one-out','unsupported':unsupported,
        'conditioning':'Supported non-self item pairs; binary temporal orientation, not next-item probability calibration',
        'weights':spec['pctm']['counts'],'diagnostics':info,'kappa':kappa,'events_sha256':sha256(out/'direction_events.npz')})
    model=fit(store,spec,'full',a.cache_dir);alpha=spec['calibration']['exposure_power']
    df=np.zeros(store.num_items)
    for group in range(len(store.group_users)):df[np.unique(store.group_items(group))]+=1
    scales={'proposed':model.history_scale.copy(),'uniform':np.ones(store.num_items),
        'inverse_mass':np.maximum(mass,1.)**(-alpha),'idf':(1+np.log((len(store.group_users)+1)/(df+1)))**alpha}
    for name,scale in scales.items():
        case=out/('history_'+name)
        if (case/'result.json').exists():continue
        model.base.history_scale=scale;_,reco=recommend_completion(model,ev)
        save_ranking(case,reco,train,held,ev,dict(dataset=d,variant=name,split='internal_cut1',
            data_sha256=verified,spec=spec,scope='Fixed history weighting controls; identical fitted direction and path components.'))
        if name=='proposed':
            reference=a.original_results/f'internal/{d}/cut1/full/per_user.npz'
            if reference.exists():
                with np.load(reference) as old,np.load(case/'per_user.npz') as new:
                    np.testing.assert_array_equal(old['ndcg10'],new['ndcg10'])
    write_json(out/'complete.json',{'all_six_direction_estimators':True,'all_four_history_controls':True})

def multik(a):
    d=a.dataset;var=a.variant;out=a.output/'multik'/d/var
    if (out/'result.json').exists():return
    spec=spec_for(d,var);train,held,verified=load_split(d,a.data_dir,'test');store=SequenceStore.from_frame(train);ev=EvaluationSet.from_holdout(store,held)
    checkpoint=a.original_results/f'test/{d}/full/model'
    model=load_completion(checkpoint) if var=='full' and checkpoint.exists() else fit(store,spec,var,a.cache_dir)
    if var=='pctm':
        h=ev.history_matrix(spec['max_history'],lambda n:kernel_weights(spec['pctm']['kernel'],n));_,reco=_recommend_pctm_rows(model,ev,h,k=20)
    elif var=='seqrules':
        h=rules_history(ev,model.config.history_length,model.config.history_weighting,spec['max_history']);_,reco=_recommend_seqrules_rows(model,ev,h,k=20)
    else:_,reco=recommend_k(model,ev,20)
    save_ranking(out,reco,train,held,ev,dict(dataset=d,variant=var,spec=spec,data_sha256=verified,
        scope='Descriptive cutoff robustness on the existing benchmark; no additional tuning.'),ks=(5,10,20))
    with np.load(out/'per_user.npz') as new,np.load(a.original_results/f'test/{d}/{var}/per_user.npz') as old:
        np.testing.assert_array_equal(new['ndcg10'],old['ndcg10'])

def baseline_statistics(a):
    rows=[]
    for d in DS:
        with np.load(a.original_results/f'test/{d}/full/per_user.npz') as f:
            for base in ['pctm','seqrules']:
                with np.load(a.original_results/f'test/{d}/{base}/per_user.npz') as b:
                    np.testing.assert_array_equal(f['user_id'],b['user_id'])
                    for key in ['ndcg10','recall10']:rows.append(dict(dataset=d,baseline=base,metric=key,**paired_summary(f[key]-b[key])))
    for r,p in zip(rows,holm([r['p_two_sided'] for r in rows])):r['p_holm']=float(p)
    write_json(a.output/'amazon_baseline_statistics.json',rows)

def external_locked(a):
    """Serialize duplicate requests for one case; distinct cases can run independently."""
    locks=a.output/'.run_locks';locks.mkdir(parents=True,exist_ok=True)
    lock=locks/f'external_{a.dataset}_{a.variant}.lock';start=time.monotonic()
    while True:
        try:
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd);break
        except FileExistsError:
            if time.monotonic()-start>3600:raise TimeoutError(f'Case still locked: {lock}')
            time.sleep(1)
    try:external(a)
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['external','mechanism','multik','baseline_statistics'])
    p.add_argument('--dataset',choices=DS+['ml1m','ml20m']);p.add_argument('--variant',default='full')
    p.add_argument('--output',type=Path,default=ROOT/'results/review');p.add_argument('--original-results',type=Path,default=ROOT/'results')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data');p.add_argument('--cache-dir',type=Path,default=ROOT/'review_count_cache')
    a=p.parse_args();freeze()
    with threadpool_limits(limits=1):
        {'external':external_locked,'mechanism':mechanisms,'multik':multik,'baseline_statistics':baseline_statistics}[a.mode](a)
