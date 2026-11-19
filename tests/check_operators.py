"""Synthetic algebra checks only; benchmark scores always use verified real data."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import sparse
from dpt_sr.full_model import CompletionConfig,completion_counts,sparse_topk

def dense_topk(x,k):
    y=np.zeros_like(x)
    for i in range(len(x)):
        ids=np.flatnonzero(x[i]>0);ids=ids[np.lexsort((ids,-x[i,ids]))[:k]];y[i,ids]=x[i,ids]
    return y

def run():
    rng=np.random.default_rng(2026);a=rng.integers(0,8,(73,73)).astype(float);a[rng.random(a.shape)>.18]=0;a[-1]=0;sp=sparse.csr_matrix(a)
    for k in [1,8,32,80]:np.testing.assert_array_equal(sparse_topk(sp,k).toarray(),dense_topk(a,k))
    mass=a.sum(axis=1);p=np.divide(dense_topk(a,8),mass[:,None],out=np.zeros_like(a),where=mass[:,None]>0)
    dense=p@p;dense[a>0]=0;np.fill_diagonal(dense,0);dense=dense_topk(dense,8)*mass[:,None]
    out,diag=completion_counts(sp,sp,CompletionConfig(fanout=8),batch_size=11)
    np.testing.assert_allclose(out.toarray(),dense,atol=1e-14,rtol=1e-14)
    assert not out.multiply(sp).nnz
    assert np.all(np.asarray(out.sum(axis=1)).ravel()<=mass+1e-12)
    assert out.nnz<=len(a)*8 and diag['max_block_product_nnz']<=11*8**2
    full=np.divide(a,mass[:,None],out=np.zeros_like(a),where=mass[:,None]>0)
    assert np.all(p@p<=full@full+1e-14)
    # Deterministic row error bound, including masking and the second top-k.
    raw=p@p;masked=raw.copy();masked[a>0]=0;np.fill_diagonal(masked,0)
    q=dense_topk(masked,8)
    rho=(full-p).sum(axis=1)
    mu=(raw-masked).sum(axis=1)
    zeta=(masked-q).sum(axis=1)
    epsilon=rho+p@rho+mu+zeta
    np.testing.assert_array_less((full@full-q).sum(axis=1),epsilon+1e-12)
    h=rng.random(len(a));h/=h.sum();scale=73/15.
    exact=h@np.log1p(scale*mass[:,None]*(full@full))
    approx=h@np.log1p(scale*mass[:,None]*q)
    assert np.all(exact>=approx-1e-12)
    assert np.max(exact-approx)<=scale*np.dot(h*mass,epsilon)+1e-12
    r={'deterministic_topk_matches_dense':True,'completion_matches_dense':True,'mass_bound':True,'sparse_support_bound':True,
       'unrenormalized_paths_do_not_exceed_original_paths':True,'row_l1_error_bound':True,'log_score_error_bound':True,
       'scope':'Synthetic identities, not recommendation performance'}
    (Path(__file__).parent/'operator_checks.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))

if __name__=='__main__':run()
