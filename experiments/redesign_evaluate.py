"""Evaluate frozen redesigned model and paired ablations on the official test split.

All variants share unaffected fitted components. Variant timings are therefore
not independent cold-start training measurements. Full-catalogue targets are
used only to evaluate the ranking, never to produce the top-20 recommendation.
"""
from pathlib import Path
import sys,json,time,argparse,copy,datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from dpt_sr.archive_io import save_npz_atomic
from scipy import sparse
from threadpoolctl import threadpool_limits
from dataclasses import replace
from dpt_sr.data import ROOT,load_split,sha256
from dpt_sr.model import DPTConfig,fit_direction,cache_for,score_histories
from dpt_sr.full_model import load_completion,save_completion,array_bytes_completion,fit_completion
from dpt_sr.calibration import CalibrationConfig
from dpt_sr.full_model import CompletionConfig
from dpt_sr.final_model import save_final
from dpt_sr.budget_model import apply_budget,BudgetConfig
from dpt_sr.recency_calibration import history_experts,weighted_history,score_calibrated
from dpt_sr.runtime import environment,peak_rss_mib
from capacity_probes.sequences import SequenceStore,EvaluationSet
from capacity_probes.counts import OffsetCache
from capacity_probes.models.pctm import count_weights
from capacity_probes.ranking import tie_ranks,topk
from experiments.redesign_calibrate import from_graph
from experiments.redesign_probe import ranks_dense,metrics

def main(a):
    out=a.output or ROOT/'results/final_redesign'/a.dataset;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():print('ALREADY COMPLETE',out);return
    confpath=ROOT/'results/redesign'/a.dataset/'rankfit_config.json';conf=json.loads(confpath.read_text());spec=conf['spec']
    for p,h in conf['algorithm_sha256'].items():
        if sha256(ROOT/p)!=h:raise ValueError('Frozen algorithm changed: '+p)
    freeze=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),config_sha256=sha256(confpath),
        algorithm_sha256=conf['algorithm_sha256'],test_metrics_seen_before_this_revision=True,
        new_revision_selection='Training-internal users only; frozen before this evaluation')
    (out/'freeze.json').write_text(json.dumps(freeze,indent=2));start=time.perf_counter()
    train,held,verified=load_split(a.dataset,a.data_dir,'test');store=SequenceStore.from_frame(train);ev=EvaluationSet.from_holdout(store,held)
    print('TEST DATA VERIFIED',a.dataset,len(train),len(ev.users),flush=True)
    # Reusing the original fitted graph is exact: the estimator, count kernel,
    # direction strength, pair gate and smoothing remain unchanged.
    legacy=ROOT/'results/review/external'/a.dataset/'full/model' if a.dataset.startswith('ml') else ROOT/'results/test'/a.dataset/'full/model'
    tick=time.perf_counter();reused=(not a.refit_graph and (legacy/'completion.json').exists())
    original=load_completion(legacy) if reused else fit_completion(store,spec['pctm'],DPTConfig(**spec['dpt']),CalibrationConfig(**spec['calibration']),CompletionConfig(**spec['completion']),spec['max_history'],a.cache_dir)
    assert np.array_equal(original.catalog,store.catalog)
    assert original.config.kappa==spec['dpt']['kappa'] and original.pctm_config.tau==spec['pctm']['tau']
    def clone_original():
        return type(original)(original.base,original.completion_config,
            original.completion_evidence.copy(),copy.deepcopy(original.diagnostics))
    full=clone_original();apply_budget(full)
    full.diagnostics['rank_calibration']={k:conf[k] for k in ['kernels','weights','prior','selected_axis','refit_parameter']}
    if not (out/'model').exists():save_final(full,out/'model')
    prepared=time.perf_counter()-tick
    hh=history_experts(full,ev,conf['kernels']);w=np.asarray(conf['weights']);prior=np.asarray(conf['prior']);ties=tie_ranks(store.catalog,'sha256')
    # Ablated residual matrices reuse the same learned direction and path graph.
    variants={'full':full.completion_evidence,'no_path':sparse.csr_matrix(full.evidence.shape),
              'no_uncertainty':apply_budget(clone_original(),BudgetConfig(True,False)).completion_evidence,
              'no_conditional':apply_budget(clone_original(),BudgetConfig(False,True)).completion_evidence,
              'independent_paths':original.completion_evidence}
    allr={k:[] for k in variants};allr['no_rank_calibration']=[]
    recids=[];recscores=[];tick=time.perf_counter()
    # Each prediction batch materializes at most O(batch_size * catalogue) scores.
    for begin in range(0,len(ev.users),128):
        stop=min(begin+128,len(ev.users));h0=hh[0][begin:stop]
        if full.config.power==0:
            pooled=weighted_history([h[begin:stop] for h in hh],w);direct=score_histories(full.base.base,pooled).toarray()
        else:
            direct=sum(float(ww)*score_histories(full.base.base,h[begin:stop]).toarray() for ww,h in zip(w,hh) if ww>1e-12)
        direct+=w[-1]*full.log_popularity
        for name,e in variants.items():
            x=direct+w[-2]*(h0@e).toarray()
            allr[name].append(ranks_dense(x.copy(),ev,begin,ties))
            if name=='full':
                for local,row in enumerate(range(begin,stop)):
                    s=x[local];s[np.unique(store.group_items(int(ev.groups[row])))]=-np.inf
                    pick=topk(np.arange(store.num_items),s,20,ties);recids.append(pick);recscores.append(s[pick])
        x=score_histories(full.base.base,h0).toarray()+(h0@full.completion_evidence).toarray()+prior[-1]*full.log_popularity
        allr['no_rank_calibration'].append(ranks_dense(x,ev,begin,ties))
        if begin%8192==0:print('TEST RANK',a.dataset,stop,len(ev.users),flush=True)
    ranking=time.perf_counter()-tick
    ids=np.stack(recids);scores=np.stack(recscores);targets=np.asarray([t[0] for t in ev.targets]);fullrank=np.concatenate(allr['full'])
    rr=np.where(ids==targets[:,None],np.arange(1,21)[None,:],0).max(axis=1)
    np.testing.assert_array_equal(rr,np.where(fullrank<=20,fullrank,0))
    save_npz_atomic(out/'recommendations_top20.npz',user_id=ev.users,item_id=store.catalog[ids],score=scores)
    records=[]
    for name,parts in allr.items():
        rank=np.concatenate(parts);case=out/name;case.mkdir(exist_ok=True)
        save_npz_atomic(case/'per_user.npz',user_id=ev.users,rank=rank,history_length=store.ends[ev.groups]-store.starts[ev.groups],target_item=store.catalog[targets],target_frequency=store.item_frequency()[targets])
        rec=dict(dataset=a.dataset,variant=name,metrics=metrics(rank),users=len(rank),data_sha256=verified,
            config_sha256=freeze['config_sha256'],per_user_sha256=sha256(case/'per_user.npz'))
        (case/'result.json').write_text(json.dumps(rec,indent=2));records.append(rec)
    # The final scorer is also checked against the official RecTools definition.
    import pandas as pd
    from capacity_probes.metrics import official_metrics
    reco=pd.DataFrame(dict(user_id=np.repeat(ev.users,20),item_id=store.catalog[ids.ravel()],score=scores.ravel(),rank=np.tile(np.arange(1,21),len(ev.users))))
    official=official_metrics(reco[reco['rank']<=10],train,held)
    np.testing.assert_allclose([official['ndcg@10'],official['recall@10']],[records[0]['metrics']['ndcg@10'],records[0]['metrics']['recall@10']],atol=1e-12,rtol=0)
    result=dict(dataset=a.dataset,records=records,official_metrics=official,
        resources=dict(fitted_graph_reused=reused,original_checkpoint=str(legacy.relative_to(ROOT)) if reused else None,residual_and_serialization_seconds=prepared,
            joint_six_variant_ranking_seconds=ranking,total_seconds=time.perf_counter()-start,peak_rss_mib=peak_rss_mib(),
            model_array_mib=array_bytes_completion(full)/2**20,blas_threads=1,gpu_used=False),environment=environment(),
        top20_target_ranks_verified=True,scope='Fixed configuration selected from internal validation.')
    (out/'complete.json').write_text(json.dumps(result,indent=2));print('TEST COMPLETE',a.dataset,official,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['beauty','sports','toys','ml1m','ml20m'],required=True);p.add_argument('--output',type=Path)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data');p.add_argument('--cache-dir',type=Path,default=ROOT/'temporary_cache/final')
    p.add_argument('--refit-graph',action='store_true',help='Rebuild count and directional graphs from verified data')
    with threadpool_limits(limits=1):main(p.parse_args())
