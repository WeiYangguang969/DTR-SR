"""Build final-paper numbers from saved ranks and coherent author baseline records."""
from pathlib import Path
import sys,json,csv,hashlib,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from threadpoolctl import threadpool_limits
from dpt_sr.data import ROOT,sha256
from experiments.paired_stats import paired_summary,holm
from experiments.redesign_probe import metrics
DS=['beauty','sports','toys','ml1m','ml20m']
RESULTS=ROOT/'results'
VAR=['full','no_direction','no_history_scale','no_path','no_rank_calibration','no_uncertainty','no_conditional','independent_paths']

def read(p):return json.loads(Path(p).read_text())
def gains(rank,k=10,kind='ndcg'):
    r=np.asarray(rank);hit=(r>0)&(r<=k)
    return np.where(hit,1/np.log2(np.maximum(r,1)+1),0) if kind=='ndcg' else hit.astype(float)
def dump(name,rows):
    p=RESULTS/'final_analysis'/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.with_suffix('.json').write_text(json.dumps(rows,indent=2))
    if isinstance(rows,list) and rows:
        keys=list(dict.fromkeys(k for row in rows for k in row))
        with p.with_suffix('.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rows)

def main():
    pub=read(ROOT/'configs/published_baselines.json')['cleanroom'];acc=[];abl=[];stats=[];bstats=[];groups=[];profiles=[];budgets=[];configs=[];sources={}
    for d in DS:
        folder=RESULTS/'final_redesign'/d;complete=read(folder/'complete.json');profiles.append(read(folder/'profile.json'))
        config=read(ROOT/'results/redesign'/d/'rankfit_config.json');configs.append(config)
        with np.load(folder/'full/per_user.npz',allow_pickle=False) as f:
            users=f['user_id'];rank=f['rank'];length=f['history_length'];freq=f['target_frequency']
        actual=metrics(rank)
        for m,r in pub[d].items():acc.append(dict(dataset=d,model=m,ndcg10=r['ndcg10'],recall10=r['recall10'],source='author coherent same-run metadata'))
        acc.append(dict(dataset=d,model='full',ndcg10=actual['ndcg@10'],recall10=actual['recall@10'],source='final frozen model'))
        for v in VAR:
            record=read(folder/v/'result.json');p=folder/v/'per_user.npz';assert sha256(p)==record['per_user_sha256'];sources[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p.resolve())]=sha256(p)
            with np.load(p,allow_pickle=False) as a:
                np.testing.assert_array_equal(users,a['user_id']);vr=a['rank']
            values=metrics(vr)
            for k,value in values.items():np.testing.assert_allclose(value,record['metrics'][k],atol=1e-12,rtol=0)
            abl.append(dict(dataset=d,variant=v,**values))
            if v!='full':
                for m in ['ndcg','recall']:
                    delta=gains(rank,kind=m)-gains(vr,kind=m)
                    stats.append(dict(dataset=d,control=v,metric=m+'10',**paired_summary(delta,seed=20260909+len(stats))))
        for baseline in ['pctm','seqrules']:
            p=ROOT/('results/review/external' if d.startswith('ml') else 'results/test')/d/baseline/'per_user.npz'
            with np.load(p,allow_pickle=False) as a:
                np.testing.assert_array_equal(users,a['user_id'])
                for m in ['ndcg','recall']:
                    delta=gains(rank,kind=m)-a[m+'10'];bstats.append(dict(dataset=d,baseline=baseline,metric=m+'10',baseline_mean=float(a[m+'10'].mean()),**paired_summary(delta,seed=20261009+len(bstats))))
                baseline_ndcg=a['ndcg10'].copy()
            if baseline=='pctm':
                for name,mask in [('history_1_9',length<10),('history_10_49',(length>=10)&(length<50)),('history_50plus',length>=50),('frequency_1_9',freq<10),('frequency_10_99',(freq>=10)&(freq<100)),('frequency_100plus',freq>=100)]:
                    if mask.any():groups.append(dict(dataset=d,group=name,**paired_summary((gains(rank)-baseline_ndcg)[mask],seed=20261109+len(groups),flip_draws=0)))
        meta=read(folder/'model/completion.json')['diagnostics'];budgets.append(dict(dataset=d,**{f'q{int(q*100)}':v for q,v in zip([0,.25,.5,.75,1],meta['source_budget_quantiles'])}))
    for rows in [stats,bstats]:
        for row,p in zip(rows,holm([r['p_two_sided'] for r in rows])):row['p_holm']=float(p)
    dump('main_comparison',acc);dump('ablations',abl);dump('module_statistics',stats);dump('baseline_statistics',bstats);dump('groups',groups);dump('profiles',profiles);dump('source_budgets',budgets);dump('calibration',configs)
    multi=read(ROOT/'results/analysis/multiple_cutoffs.json');multi=[r for r in multi if r['variant']!='full']
    for d in DS:
        r=next(r for r in abl if r['dataset']==d and r['variant']=='full')
        for k in [5,10,20]:multi.append(dict(dataset=d,variant='full',k=k,ndcg=r[f'ndcg@{k}'],recall=r[f'recall@{k}']))
    dump('multiple_cutoffs',multi)
    assert len(acc)==40 and len(abl)==40
    dump('audit',dict(own_rank_records=len(sources),datasets=5,main_methods=8,main_metrics=2,main_cells=80,
        own_cutoffs=[5,10,20],module_holm_family=len(stats),baseline_holm_family=len(bstats),sources=sources,
        published_config_sha256=sha256(ROOT/'configs/published_baselines.json'),final_config_sha256=sha256(ROOT/'configs/final_redesign.json')))
    print('FINAL ANALYSIS: 80 main cells; 40 verified user-rank records; all tables from saved evidence',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results-root',type=Path,default=RESULTS);RESULTS=parser.parse_args().results_root
    with threadpool_limits(limits=1):main()
