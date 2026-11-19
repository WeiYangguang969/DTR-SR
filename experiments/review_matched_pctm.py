r"""Matched one-coordinate calibration for PCTM (supplement Sec. 1).

PCTM is DPT-SR at kappa=0, lambda=0, p=0, varsigma=0 and eta=0. Built through
the DPT-SR code path at those values, it receives exactly the calibration
budget DPT-SR received (experiments/redesign_rankfit.py): the same
training-internal split, the same 8192-user sample and 80/20 SHA-256 user
partition, the same exact breakpoint sweep over stable open intervals, the same
axes -- except the path axis, which is void at eta=0 -- and the same option of
keeping the anchor. The selected coordinate is frozen, then evaluated once on
the test split.

Checks: (1) the test anchor reproduces the upstream PCTM run user by user
within its top-20; (2) the held-out validation value of the selected coordinate
is recomputed from full-catalogue ranks, independently of the event sweep.
"""
from pathlib import Path
import sys, json, time, argparse, copy, hashlib, datetime
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.redesign_calibrate import load_inner, from_graph
from experiments.redesign_probe import ranks_dense, metrics
from experiments.paired_stats import paired_summary
from dpt_sr.data import ROOT, load_split, sha256
from dpt_sr.budget_model import apply_budget
from dpt_sr.model import cache_for, score_histories
from dpt_sr.recency_calibration import kernels_for, history_experts, dense_features, prior_vector
from dpt_sr.rank_calibration import user_events, optimum, value_at
from capacity_probes.sequences import SequenceStore, EvaluationSet
from capacity_probes.ranking import tie_ranks
from capacity_probes.counts import OffsetCache
from capacity_probes.models.pctm import count_weights


def counts_for(store, spec, cache_root):
    oc = OffsetCache(cache_for(store, cache_root), store)
    kw = count_weights(spec['pctm']['counts']); oc.build(len(kw))
    return oc.combine(spec['pctm']['counts'], kw).astype(float)


def neutral(spec):
    """DPT-SR at the values where it is PCTM."""
    s = copy.deepcopy(spec)
    s['dpt'].update(kappa=0., power=0.)
    s['calibration'] = dict(pair_penalty=0., exposure_power=0.)
    s['completion'].update(strength=0.)
    return s


def pctm_model(store, spec, c):
    # kappa=0 leaves D=C: the directed matrix is the count matrix itself.
    info = {'prior': 'disabled', 'converged': True, 'iterations': 0}
    model = apply_budget(from_graph(store, spec, c, c.copy(), np.zeros(store.num_items), info))
    assert model.completion_evidence.nnz == 0
    return model


def with_axis(prior, name, p):
    w = prior.copy()
    if name == 'popularity':
        w[-1] += p
    elif name.startswith('history_'):
        i = int(name.split('_')[1]); w[0] = 1 - p; w[i] = p
    return w


def calibrate(ds, out, cache_root):
    tick = time.perf_counter(); red = ROOT / 'results/redesign' / ds
    store, ev, verified = load_inner(ds)
    if len(ev.users) > 8192:
        ix = np.sort(np.random.default_rng(20260909).choice(len(ev.users), 8192, replace=False))
        ev = EvaluationSet(store, ev.users[ix], ev.groups[ix], tuple(ev.targets[i] for i in ix))
    dpt = json.loads((red / 'rankfit_config.json').read_text()); spec = neutral(dpt['spec'])
    kernels = dpt['kernels']; assert kernels == kernels_for(spec['pctm']['kernel'])
    # The inner graph DPT-SR was calibrated on is rebuilt from the saved inner
    # split and checked against its recorded pair support.
    c = counts_for(store, spec, cache_root); total = (c + c.T).tocsr(); total.sum_duplicates()
    recorded = json.loads((red / 'valid_graph' / 'fit.json').read_text())['pair_support_nnz']
    assert total.nnz == recorded, (total.nnz, recorded); del total
    model = pctm_model(store, spec, c); del c
    hh = history_experts(model, ev, kernels); prior = prior_vector(model, kernels); b = len(kernels)
    axes = [('popularity', -.2, .2)] + [(f'history_{i}', 0., 1.) for i in range(1, b)]
    mask = np.asarray([int(hashlib.sha256(f'calibration:{int(u)}'.encode()).hexdigest()[:8], 16) % 5 != 0
                       for u in ev.users])
    ties = tie_ranks(store.catalog, 'sha256')
    initial = {n: [] for n, _, _ in axes}; events = {n: [] for n, _, _ in axes}; fixed = []
    for start in range(0, len(ev.users), 64):
        stop = min(start + 64, len(ev.users)); ff = dense_features(model, hh, start, stop)
        assert not np.any(ff[-2])  # no path evidence at eta=0
        anchor = ff[0] + ff[-2] + spec['pctm']['pop_boost'] * ff[-1]
        fixed.append(ranks_dense(anchor.copy(), ev, start, ties))
        for name, lo, hi in axes:
            slope = ff[-1] if name == 'popularity' else ff[int(name.split('_')[1])] - ff[0]
            for local, row in enumerate(range(start, stop)):
                s0, x, dy = user_events(anchor[local], slope[local], int(ev.targets[row][0]),
                                        store.group_items(int(ev.groups[row])), ties, lo, hi)
                initial[name].append(s0); events[name].append((x, dy))
    ranks = np.concatenate(fixed); nv = int((~mask).sum()); nt = int(mask.sum())
    validation = {}; fits = {}
    for name, lo, hi in axes:
        ii = np.asarray(initial[name])
        te = [e for e, m in zip(events[name], mask) if m]; ve = [e for e, m in zip(events[name], mask) if not m]
        point, value = optimum(float(ii[mask].sum()), te, lo, hi)
        val = value_at(float(ii[~mask].sum()), ve, point) / nv
        validation[name] = val
        fits[name] = dict(parameter=point, train_ndcg=value / nt, validation_ndcg=val,
                          retained_events=sum(len(e[0]) for e in events[name]))
    validation['fixed'] = metrics(ranks[~mask])['ndcg@10']
    selected = max(validation, key=validation.get)
    final = prior.copy(); parameter = 0.
    if selected != 'fixed':
        _, lo, hi = next(x for x in axes if x[0] == selected)
        parameter, _ = optimum(sum(initial[selected]), events[selected], lo, hi)
        final = with_axis(prior, selected, parameter)
    # Independent check of the held-out value at the selected coordinate.
    iv = np.flatnonzero(~mask)
    valev = EvaluationSet(store, ev.users[iv], ev.groups[iv], tuple(ev.targets[i] for i in iv))
    vh = [h[iv] for h in hh]; check = []
    w = prior if selected == 'fixed' else with_axis(prior, selected, fits[selected]['parameter'])
    for start in range(0, nv, 64):
        stop = min(start + 64, nv); ff = dense_features(model, vh, start, stop)
        check.append(ranks_dense(sum(float(z) * f for z, f in zip(w, ff)), valev, start, ties))
    actual = metrics(np.concatenate(check))
    np.testing.assert_allclose(actual['ndcg@10'], validation[selected], atol=1e-10, rtol=0)
    fit = dict(dataset=ds, spec=spec, kernels=kernels, weights=final.tolist(), prior=prior.tolist(),
               selected_axis=selected, refit_parameter=parameter, validation_ndcg=validation, axis_fits=fits,
               validation_selected_metrics=actual, validation_users=nv, fit_users=nt, inner_users=len(ev.users),
               dpt_sr_reference=dict(selected_axis=dpt['selected_axis'], validation_ndcg=dpt['validation_ndcg']),
               data_sha256=verified, seconds=time.perf_counter() - tick)
    (out / 'inner_fit.json').write_text(json.dumps(fit, indent=2))
    print('FIT', ds, selected, round(parameter, 4), {k: round(v, 5) for k, v in validation.items()}, flush=True)
    return fit


def gain10(r):
    r = np.asarray(r); return np.where((r > 0) & (r <= 10), 1 / np.log2(np.maximum(r, 1) + 1), 0.)


def evaluate(ds, fit, out, cache_root):
    tick = time.perf_counter()
    train, held, verified = load_split(ds, ROOT / 'data', 'test')
    store = SequenceStore.from_frame(train); ev = EvaluationSet.from_holdout(store, held); spec = fit['spec']
    model = pctm_model(store, spec, counts_for(store, spec, cache_root))
    kernels = fit['kernels']; w = np.asarray(fit['weights']); prior = np.asarray(fit['prior'])
    use = [i for i in range(len(kernels)) if w[i] > 0 or prior[i] > 0]; assert use[0] == 0
    hh = history_experts(model, ev, [kernels[i] for i in use]); ties = tie_ranks(store.catalog, 'sha256')
    pop = model.log_popularity; anchor_r = []; cal_r = []
    for begin in range(0, len(ev.users), 128):
        stop = min(begin + 128, len(ev.users))
        f = [score_histories(model.base.base, h[begin:stop]).toarray() for h in hh]
        anchor_r.append(ranks_dense(f[0] + prior[-1] * pop, ev, begin, ties))
        cal = sum(float(w[i]) * f[k] for k, i in enumerate(use)) + w[-1] * pop
        cal_r.append(ranks_dense(cal, ev, begin, ties))
    anchor_r = np.concatenate(anchor_r); cal_r = np.concatenate(cal_r)
    # (1) the anchor is the upstream PCTM run, user by user within its top-20
    up = np.load(ROOT / 'upstream_runs' / ds / 'pctm' / 'per_user.npz')
    assert np.array_equal(up['user_id'], ev.users)
    upm = json.loads((ROOT / 'upstream_runs' / ds / 'pctm' / 'result.json').read_text())['metrics']
    mismatch = int((np.where(anchor_r <= 20, anchor_r, 0) != up['rank_in_topk']).sum())
    ref = ROOT / 'results/final_redesign' / ds
    dfull = np.load(ref / 'full' / 'per_user.npz'); dnc = np.load(ref / 'no_rank_calibration' / 'per_user.npz')
    assert np.array_equal(dfull['user_id'], ev.users) and np.array_equal(dnc['user_id'], ev.users)
    g = dict(pctm_anchor=gain10(anchor_r), pctm_calibrated=gain10(cal_r),
             dpt_sr=gain10(dfull['rank']), dpt_sr_uncalibrated=gain10(dnc['rank']))
    pairs = [('dpt_sr', 'pctm_anchor'), ('dpt_sr', 'pctm_calibrated'),
             ('dpt_sr_uncalibrated', 'pctm_calibrated'), ('pctm_calibrated', 'pctm_anchor')]
    comparisons = {f'{a} - {b}': paired_summary(g[a] - g[b]) for a, b in pairs}
    res = dict(dataset=ds, users=len(ev.users), data_sha256=verified,
               metrics=dict(pctm_anchor=metrics(anchor_r), pctm_calibrated=metrics(cal_r),
                            dpt_sr=metrics(dfull['rank']), dpt_sr_uncalibrated=metrics(dnc['rank'])),
               upstream_pctm_metrics=upm, anchor_top20_mismatches_vs_upstream=mismatch,
               comparisons_ndcg10=comparisons, seconds=time.perf_counter() - tick)
    np.savez_compressed(out / 'per_user.npz', user_id=ev.users, rank_anchor=anchor_r, rank_calibrated=cal_r)
    (out / 'test.json').write_text(json.dumps(res, indent=2))
    m = res['metrics']
    print('TEST', ds, 'mismatch', mismatch,
          {k: round(v['ndcg@10'], 5) for k, v in m.items()},
          {k: (round(1e3 * v['difference'], 2), v['p_two_sided']) for k, v in comparisons.items()}, flush=True)


def main(a):
    for ds in a.datasets:
        out = a.output_dir / ds; out.mkdir(parents=True, exist_ok=True)
        if (out / 'test.json').exists():
            print('DONE', ds); continue
        fit = calibrate(ds, out, a.cache_dir); evaluate(ds, fit, out, a.cache_dir)
    (a.output_dir / 'protocol.json').write_text(json.dumps(dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='Give PCTM the calibration budget DPT-SR received.',
        pctm_as_special_case='kappa=0, pair_penalty=0, exposure_power=0, power=0, completion strength=0',
        split='training-internal LOO; 8192-user sample (seed 20260909); 80/20 SHA-256 user partition',
        axes='popularity offset in (-0.2,0.2); history_i weight in (0,1) with remainder on kernel 0; '
             'path axis void at eta=0; anchor kept if best',
        optimizer='event sweep over open stable-ranking intervals (dpt_sr/rank_calibration.py)',
        test='selected coordinate frozen, evaluated once; paired bootstrap and sign-flip on user NDCG@10',
        script='experiments/review_matched_pctm.py', script_sha256=sha256(Path(__file__))), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('datasets', nargs='+', choices=['beauty', 'sports', 'toys', 'ml1m', 'ml20m'])
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/review_r2/matched_pctm')
    p.add_argument('--cache-dir', type=Path, default=ROOT / 'review_cache/matched_pctm')
    with threadpool_limits(limits=1):
        main(p.parse_args())
