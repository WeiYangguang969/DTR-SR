"""Reproduce the final five-dataset rankings, ablations and evidence tables.

Use the pinned scoring environment. Paper rendering uses a separate authoring
 environment. Released results are never silently overwritten.
"""
from pathlib import Path
import argparse,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
DATASETS=['beauty','sports','toys','ml1m','ml20m']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'reproduced_final')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    p.add_argument('--from-results',action='store_true',help='Verify saved ranks and regenerate analysis from --output')
    p.add_argument('--reuse-graph',action='store_true',help='Reuse released original graph checkpoints; default rebuilds them')
    a=p.parse_args();a.output=a.output.resolve();a.data_dir=a.data_dir.resolve()
    env=os.environ.copy()
    for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS','PCTM_EVAL_JOBS']:env[key]='1'
    def call(script,args):
        print(script,*map(str,args),flush=True)
        subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],env=env,check=True)
    if not a.from_results:
        from dpt_sr.data import prepare
        from experiments.prepare_external import run
        for d in a.datasets:
            try:prepare(d,a.data_dir)
            except RuntimeError:
                if not d.startswith('ml'):raise
                run(d,a.data_dir,'mirror',a.output/'download_cache')
            case=a.output/'final_redesign'/d
            call('experiments/redesign_evaluate.py',['--dataset',d,'--output',case,'--data-dir',a.data_dir,'--cache-dir',a.output/'temporary_cache',* ([] if a.reuse_graph else ['--refit-graph'])])
            call('experiments/redesign_extra_ablations.py',['--dataset',d,'--results-root',a.output,'--data-dir',a.data_dir])
            if not (case/'profile.json').exists():
                call('experiments/redesign_profile.py',['--dataset',d,'--results-root',a.output,'--data-dir',a.data_dir,'--repeats',1 if d=='ml20m' else 3])
    if all((a.output/'final_redesign'/d/'profile.json').exists() for d in DATASETS):
        call('experiments/analyze_final.py',['--results-root',a.output])
    print('Final outputs:',a.output,flush=True)

if __name__=='__main__':main()
