"""Direction and history-scale deletions for the frozen redesigned scorer."""
from pathlib import Path
import sys,json,time,argparse,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from dpt_sr.archive_io import save_npz_atomic
from scipy import sparse
from threadpoolctl import threadpool_limits
from dataclasses import replace
from dpt_sr.data import ROOT,load_split,sha256
from dpt_sr.full_model import load_completion
from dpt_sr.budget_model import apply_budget
from dpt_sr.recency_calibration import history_experts,score_calibrated
from dpt_sr.model import cache_for
from capacity_probes.counts import OffsetCache
from capacity_probes.models.pctm import count_weights
from capacity_probes.sequences import SequenceStore,EvaluationSet
from capacity_probes.ranking import tie_ranks
from experiments.redesign_calibrate import from_graph
from experiments.redesign_probe import ranks_dense,metrics

def main(a):
    root=a.results_root/'final_redesign'/a.dataset;conf=json.loads((ROOT/'results/redesign'/a.dataset/'rankfit_config.json').read_text());spec=conf['spec'];w=np.asarray(conf['weights'])
    train,held,verified=load_split(a.dataset,a.data_dir,'test');store=SequenceStore.from_frame(train);ev=EvaluationSet.from_holdout(store,held);ties=tie_ranks(store.catalog,'sha256')
    full=load_completion(root/'model')
    for variant in ('no_history_scale','no_direction'):
        out=root/variant
        if (out/'result.json').exists():continue
        out.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
        if variant=='no_history_scale':
            original=full.base;newcal=type(original)(original.base,replace(original.calibration_config,exposure_power=0.),np.ones_like(original.history_scale))
            m=type(full)(newcal,full.completion_config,full.completion_evidence,full.diagnostics)
        else:
            oc=OffsetCache(cache_for(store,ROOT/'redesign_cache'),store);kw=count_weights(spec['pctm']['counts']);oc.build(len(kw));c=oc.combine(spec['pctm']['counts'],kw).astype(float)
            ss=copy.deepcopy(spec);ss['dpt']['kappa']=0
            m=apply_budget(from_graph(store,ss,c,c,np.zeros(store.num_items),{'prior':'disabled','converged':True,'iterations':0}));del c
        hh=history_experts(m,ev,conf['kernels']);ranks=[]
        for begin in range(0,len(ev.users),128):
            stop=min(begin+128,len(ev.users));x=score_calibrated(m,hh,w,begin,stop);ranks.append(ranks_dense(x,ev,begin,ties))
            if begin%16384==0:print('EXTRA ABLATION',a.dataset,variant,stop,len(ev.users),flush=True)
        rank=np.concatenate(ranks);save_npz_atomic(out/'per_user.npz',user_id=ev.users,rank=rank)
        rec=dict(dataset=a.dataset,variant=variant,metrics=metrics(rank),users=len(rank),data_sha256=verified,
            config_sha256=sha256(ROOT/'results/redesign'/a.dataset/'rankfit_config.json'),per_user_sha256=sha256(out/'per_user.npz'),
            seconds=time.perf_counter()-tick,scope='Fixed configuration deletion; fitted components reused where unaffected.')
        (out/'result.json').write_text(json.dumps(rec,indent=2));print('EXTRA COMPLETE',a.dataset,variant,rec['metrics'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['beauty','sports','toys','ml1m','ml20m'],required=True)
    p.add_argument('--results-root',type=Path,default=ROOT/'results');p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    with threadpool_limits(limits=1):main(p.parse_args())
