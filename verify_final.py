"""Verify final release provenance, metrics, configurations and model archives."""
from pathlib import Path
import argparse,json,hashlib,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
DS=['beauty','sports','toys','ml1m','ml20m']
VAR=['full','no_direction','no_history_scale','no_path','no_rank_calibration','no_uncertainty','no_conditional','independent_paths']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--skip-model-crc',action='store_true');a=p.parse_args()
    config=json.loads((ROOT/'configs/final_redesign.json').read_text());checked=0;archives=0
    for d in DS:
        folder=ROOT/'results/final_redesign'/d;stored=json.loads((ROOT/'results/redesign'/d/'rankfit_config.json').read_text())
        assert config[d]==stored,(d,'configuration mismatch')
        for name,h in stored['algorithm_sha256'].items():assert sha(ROOT/name)==h,(d,name)
        complete=json.loads((folder/'complete.json').read_text());assert complete['top20_target_ranks_verified']
        for v in VAR:
            case=folder/v;r=json.loads((case/'result.json').read_text());assert sha(case/'per_user.npz')==r['per_user_sha256']
            with np.load(case/'per_user.npz',allow_pickle=False) as z:
                rank=z['rank'];users=z['user_id'];assert len(rank)==r['users']
                if v=='full':refusers=users.copy()
                else:np.testing.assert_array_equal(users,refusers)
                for k in [5,10,20]:
                    hit=(rank>0)&(rank<=k);ndcg=np.where(hit,1/np.log2(np.maximum(rank,1)+1),0).mean()
                    np.testing.assert_allclose([ndcg,hit.mean()],[r['metrics'][f'ndcg@{k}'],r['metrics'][f'recall@{k}']],atol=1e-12,rtol=0)
                if v=='full':
                    for k,value in complete['official_metrics'].items():np.testing.assert_allclose(r['metrics'][k],value,atol=1e-12,rtol=0)
            checked+=1
        profile=json.loads((folder/'profile.json').read_text());assert profile['ranking_identity_verified']
        if not a.skip_model_crc:
            for f in (folder/'model').rglob('*.npz'):
                with zipfile.ZipFile(f) as z:assert z.testzip() is None,f
                archives+=1
    audit=json.loads((ROOT/'results/final_analysis/audit.json').read_text());assert audit['own_rank_records']==checked==40 and audit['main_cells']==80
    print('PASS: 40 raw-rank records, six cutoff metrics, stored RecTools checks and frozen algorithm hashes.')
if __name__=='__main__':main()
