"""Fit the complete final DPT-SR from verified training data and save its top-20 output."""
from pathlib import Path
import argparse,json,time,tempfile
from threadpoolctl import threadpool_limits
from dpt_sr.data import ROOT,prepare,load_split,sha256
from dpt_sr.final_model import fit_final,save_final,recommend_final
from dpt_sr.runtime import environment,peak_rss_mib
from capacity_probes.sequences import SequenceStore,EvaluationSet
from capacity_probes.metrics import official_metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['beauty','sports','toys','ml1m','ml20m'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:prepare(a.dataset,a.data_dir)
    except RuntimeError:
        if not a.dataset.startswith('ml'):raise
        from experiments.prepare_external import run
        run(a.dataset,a.data_dir,'mirror',a.data_dir/'raw_mirror')
    confpath=ROOT/'configs/final_redesign.json';conf=json.loads(confpath.read_text())[a.dataset]
    train,held,verified=load_split(a.dataset,a.data_dir,'test');store=SequenceStore.from_frame(train);ev=EvaluationSet.from_holdout(store,held)
    with tempfile.TemporaryDirectory(prefix='dptsr_final_counts_') as cache,threadpool_limits(limits=1):
        tick=time.perf_counter();model=fit_final(store,conf,Path(cache));fitted=time.perf_counter()-tick
        tick=time.perf_counter();reco=recommend_final(model,ev);ranked=time.perf_counter()-tick
        official = {}
        for k in (10, 20):
            official.update(official_metrics(reco[reco['rank'] <= k], train, held, k=k))
        save_final(model,a.output/'model');reco.to_csv(a.output/'recommendations_top20.csv',index=False)
    result=dict(dataset=a.dataset,official_metrics=official,data_sha256=verified,config_sha256=sha256(confpath),
        fit_seconds=fitted,recommendation_seconds=ranked,peak_rss_mib=peak_rss_mib(),environment=environment(),
        scope='Fresh graph fit with released inner-calibrated coefficients; reproducing coefficient learning is a separate selection stage.')
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
