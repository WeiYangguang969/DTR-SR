"""Measure final all-user recommendation, separately from multi-control evaluation."""
from pathlib import Path
import sys,json,time,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from threadpoolctl import threadpool_limits
from dpt_sr.data import ROOT,load_split
from dpt_sr.final_model import load_final,recommend_final
from dpt_sr.full_model import array_bytes_completion
from dpt_sr.runtime import environment,peak_rss_mib
from capacity_probes.sequences import SequenceStore,EvaluationSet

def main(a):
    out=a.results_root/'final_redesign'/a.dataset;train,held,_=load_split(a.dataset,a.data_dir,'test')
    store=SequenceStore.from_frame(train);ev=EvaluationSet.from_holdout(store,held);tick=time.perf_counter();m=load_final(out/'model');loaded=time.perf_counter()-tick
    times=[]
    for _ in range(a.repeats):
        tick=time.perf_counter();reco=recommend_final(m,ev);times.append(time.perf_counter()-tick)
    with np.load(out/'recommendations_top20.npz',allow_pickle=False) as reference:
        np.testing.assert_array_equal(reco.item_id.to_numpy().reshape(-1,20),reference['item_id'])
        np.testing.assert_allclose(reco.score.to_numpy().reshape(-1,20),reference['score'],atol=1e-12,rtol=0)
    cal=m.diagnostics['rank_calibration'];result=dict(dataset=a.dataset,users=len(ev.users),catalog_items=store.num_items,
        load_seconds=loaded,recommendation_seconds=times,mean_seconds=float(np.mean(times)),
        std_seconds=float(np.std(times,ddof=1)) if len(times)>1 else None,peak_rss_mib=peak_rss_mib(),
        array_mib=(array_bytes_completion(m)+8*len(cal['weights']))/2**20,
        active_history_experts=sum(w>1e-12 for w in cal['weights'][:-2]),
        ranking_identity_verified=True,environment=environment(),
        scope='Loaded-model all-user top-20 recommendation; one CPU thread; repeats within one process; excludes data preparation, fitting and inner calibration.')
    (out/'profile.json').write_text(json.dumps(result,indent=2));print('PROFILE',a.dataset,result['mean_seconds'],result['peak_rss_mib'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['beauty','sports','toys','ml1m','ml20m'],required=True);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--results-root',type=Path,default=ROOT/'results');p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    with threadpool_limits(limits=1):main(p.parse_args())
