"""Explain users whose PCTM anchor top-20 differs from the upstream PCTM run.

Hypothesis: single-precision (upstream) vs double-precision (DPT-SR code path)
near-ties. For each such user, score the catalogue both ways and record the
target's rank under each and the score gap to every item whose order relative
to the target differs between the two precisions. Run after
review_matched_pctm.py; on the released data only ML-20M has such users (3).
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from dpt_sr.data import ROOT, load_split
from dpt_sr.model import cache_for
from capacity_probes.sequences import SequenceStore, EvaluationSet
from capacity_probes.counts import OffsetCache
from capacity_probes.models.pctm import count_weights, kernel_weights, smoothed_log_evidence, log_popularity
from capacity_probes.ranking import tie_ranks


def run(ds, out_dir, cache_root):
    base = out_dir / ds
    spec = json.loads((base / 'inner_fit.json').read_text())['spec']
    m = np.load(base / 'per_user.npz')
    up = np.load(ROOT / 'upstream_runs' / ds / 'pctm' / 'per_user.npz')
    mine = np.where(m['rank_anchor'] <= 20, m['rank_anchor'], 0)
    bad = np.flatnonzero(mine != up['rank_in_topk'])
    rows = []
    if len(bad):
        train, held, _ = load_split(ds, ROOT / 'data', 'test')
        store = SequenceStore.from_frame(train); ev = EvaluationSet.from_holdout(store, held)
        oc = OffsetCache(cache_for(store, cache_root), store); kw = count_weights(spec['pctm']['counts']); oc.build(len(kw))
        c = oc.combine(spec['pctm']['counts'], kw)
        tau = spec['pctm']['tau']; beta = spec['pctm']['pop_boost']; n = store.num_items
        e32 = smoothed_log_evidence(c, tau)                                    # upstream, single precision
        e64 = c.astype(np.float64); e64.data = np.log1p(e64.data * n / tau)    # DPT-SR path, double
        pop32 = log_popularity(store); pop64 = pop32.astype(np.float64)
        ties = tie_ranks(store.catalog, 'sha256')
        sub = EvaluationSet(store, ev.users[bad], ev.groups[bad], tuple(ev.targets[i] for i in bad))
        h = sub.history_matrix(spec['max_history'], lambda L: kernel_weights(spec['pctm']['kernel'], L))
        for k, row in enumerate(bad):
            s32 = (h[k] @ e32).toarray().ravel().astype(np.float32) + np.float32(beta) * pop32
            s64 = (h[k].astype(np.float64) @ e64).toarray().ravel() + beta * pop64
            seen = np.unique(store.group_items(int(ev.groups[row]))); t = int(ev.targets[row][0])
            s32[seen] = -np.inf; s64[seen] = -np.inf

            def rank(s):
                v = s[t]; return int(1 + np.count_nonzero(s > v) + np.count_nonzero((s == v) & (ties < ties[t])))
            flip = np.flatnonzero(((s32 > s32[t]) != (s64 > s64[t])) & np.isfinite(s64))
            gap = np.abs(s64[flip] - s64[t]); ulp = float(np.spacing(np.float32(s64[t])))
            rows.append(dict(row=int(row), user_id=int(ev.users[row]), rank_single=rank(s32), rank_double=rank(s64),
                             upstream_rank_in_top20=int(up['rank_in_topk'][row]), items_flipping=int(len(flip)),
                             max_gap=float(gap.max()) if len(gap) else 0., single_precision_ulp=ulp,
                             max_gap_in_ulp=float(gap.max() / ulp) if len(gap) else 0.))
            print(ds, rows[-1], flush=True)
    res = dict(dataset=ds, users=int(len(mine)), mismatched_users=int(len(bad)), cases=rows)
    (base / 'anchor_ties.json').write_text(json.dumps(res, indent=2))
    print('WROTE', base / 'anchor_ties.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', nargs='?', default='ml20m', choices=['beauty', 'sports', 'toys', 'ml1m', 'ml20m'])
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/review_r2/matched_pctm')
    p.add_argument('--cache-dir', type=Path, default=ROOT / 'review_cache/matched_pctm')
    a = p.parse_args()
    run(a.dataset, a.output_dir, a.cache_dir)
