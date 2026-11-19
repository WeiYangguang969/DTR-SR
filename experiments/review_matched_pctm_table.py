r"""LaTeX rows of the supplement's matched-calibration table (tab:matched).

Values are x1e-3 NDCG@10. "Both anchored" pairs DPT-SR with its calibrated
coordinate returned to the PCTM anchor (no_rank_calibration) against PCTM;
"both calibrated" pairs DPT-SR as reported against PCTM at its selected
coordinate. Each row is followed by a line of 95% bootstrap intervals.
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from dpt_sr.data import ROOT
from experiments.paired_stats import paired_summary

NAMES = dict(beauty='Beauty', sports='Sports', toys='Toys', ml1m='ML-1M', ml20m='ML-20M')


def gain10(r):
    r = np.asarray(r); return np.where((r > 0) & (r <= 10), 1 / np.log2(np.maximum(r, 1) + 1), 0.)


def coord(fit):
    s = fit['selected_axis']; w = fit['weights']; p = fit['prior']
    if s == 'fixed':
        return 'anchor kept'
    if s == 'popularity':
        return (r'$\beta$: $%.3f{\to}%.3f$' % (p[-1], w[-1])).replace('-', '{-}')
    i = int(s.split('_')[1])
    return r'$w_%d$: $0{\to}%.3f$' % (i, w[i])


def cell(s):
    return r'$%+.2f$' % (1e3 * s['difference'])


def ci(s):
    return r'\scriptsize$[%.2f,%.2f]$' % (1e3 * s['ci95_low'], 1e3 * s['ci95_high'])


def rows(results):
    out = []; raw = {}
    for ds, name in NAMES.items():
        base = results / ds
        if not (base / 'test.json').exists():
            continue
        fit = json.loads((base / 'inner_fit.json').read_text()); t = json.loads((base / 'test.json').read_text())
        m = np.load(base / 'per_user.npz')
        nc = np.load(ROOT / 'results/final_redesign' / ds / 'no_rank_calibration' / 'per_user.npz')
        assert np.array_equal(m['user_id'], nc['user_id'])
        both_anchor = paired_summary(gain10(nc['rank']) - gain10(m['rank_anchor']))
        c = t['comparisons_ndcg10']; shift = c['pctm_calibrated - pctm_anchor']; bc = c['dpt_sr - pctm_calibrated']
        v = fit['validation_ndcg']; sel = fit['selected_axis']; kept = sel == 'fixed'
        inner = (r'$%.1f$' % (1e3 * v['fixed']) if kept
                 else r'$%.1f{\to}%.1f$' % (1e3 * v['fixed'], 1e3 * v[sel]))
        out.append(r'%s & %s & %s & %s & %s & %s \\' % (
            name, coord(fit), inner, '$0$' if kept else cell(shift), cell(both_anchor), cell(bc)))
        out.append(r' & & & %s & %s & %s \\[2pt]' % ('' if kept else ci(shift), ci(both_anchor), ci(bc)))
        raw[ds] = dict(mismatch=t['anchor_top20_mismatches_vs_upstream'],
                       p=[round(x['p_two_sided'], 5) for x in (shift, both_anchor, bc)])
    return out, raw


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT / 'results/review_r2/matched_pctm')
    a = p.parse_args()
    lines, raw = rows(a.results)
    print('\n'.join(lines))
    print('%', json.dumps(raw))
