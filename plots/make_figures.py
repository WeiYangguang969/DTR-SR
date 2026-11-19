"""Figures for the reliability-weighting account of DPT-SR.

Self-contained: reads only published result tables under ../results, writes
only into ./figures. LNCS 122 mm text width, editable vector text, 900 dpi.
"""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = HERE.parent / 'results'
OUT = HERE / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

STYLE = {
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': .6, 'lines.linewidth': 1.0, 'lines.markersize': 3.2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.axisbelow': True, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'savefig.dpi': 900,
}
plt.rcParams.update(STYLE)
WIDTH = 122 / 25.4

DS = ['beauty', 'sports', 'toys', 'ml1m', 'ml20m']
NAMES = ['Beauty', 'Sports', 'Toys', 'ML-1M', 'ML-20M']
COL = ['#28618A', '#A97425', '#48745D', '#A83D49', '#72648E']
MARK = ['o', 's', '^', 'D', 'v']
ACCENT = '#28618A'
MUTED = '#C9D1D7'


def save(fig, stem):
    for ext in ('pdf', 'svg', 'png'):
        buf = io.BytesIO()
        fig.savefig(buf, format=ext, bbox_inches='tight', pad_inches=.02)
        p = OUT / f'{stem}.{ext}'
        tmp = p.with_suffix(p.suffix + '.tmp')
        tmp.write_bytes(buf.getvalue())
        tmp.replace(p)
    plt.close(fig)
    print('wrote', stem, flush=True)


def fig_direction():
    """Fig. 3: direction reliability by pair-support bin (Amazon datasets)."""
    dc = pd.read_csv(R / 'analysis/direction_calibration.csv')
    bins = ['(0,1]', '(1,3]', '(3,10]', '>10']
    spec = [('empirical', 'Empirical', '#8A929A', 'o', '-'),
            ('laplace', 'Laplace', '#A97425', 's', '-'),
            ('jeffreys', 'Jeffreys', '#916CAE', '^', '-'),
            ('neutral', 'Neutral', '#45AFA4', 'v', (0, (4, 2))),
            ('bt_prior', 'BT prior', '#28618A', 'D', (0, (4, 2))),
            ('bt_shrinkage', 'BT shrinkage', '#A83D49', '*', '-')]
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 2.40), layout='constrained')
    for k, (ax, d, name) in enumerate(zip(axs, DS[:3], NAMES[:3])):
        sub = dc[dc.dataset == d]
        get = lambda e, b: float(sub[(sub.estimator == e) & (sub.bin == b)]['brier'].iloc[0])
        lo = [get('bt_shrinkage', b) for b in bins]
        hi = [max(get(e, b) for e, *_ in spec) for b in bins]
        ax.fill_between(range(4), lo, hi, color='#A83D49', alpha=.05, lw=0, zorder=1)
        for est, lab, c, mk, ls in spec:
            best = est == 'bt_shrinkage'
            ax.plot(range(4), [get(est, b) for b in bins], color=c, marker=mk, ls=ls,
                    label=lab, lw=1.4 if best else .95,
                    markersize=4.4 if best else 3.0, zorder=5 if best else 2)
        ax.set_xticks(range(4), bins, rotation=28, ha='right')
        ax.set_title(f'({chr(97 + k)}) {name}', fontsize=8.5)
        ax.set_xlabel('Pair support', fontsize=7.5)
        ax.grid(axis='y', alpha=.18, lw=.4)
        if k == 0:
            ax.set_ylabel('Weighted Brier error', fontsize=7.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside upper center', ncol=6, frameon=False,
               fontsize=6.6, columnspacing=.9, handlelength=1.7)
    save(fig, 'fig03_direction')


def fig_path():
    """Fig. 4: independent-path failure and conditional recovery on MovieLens."""
    a = pd.read_csv(R / 'final_analysis/ablations.csv')
    cells = [('independent_paths', 'Independent\npaths'),
             ('no_conditional', '+ Source\ngate'),
             ('no_uncertainty', '+ Candidate\ngate'),
             ('full', 'Full\n(both)')]
    bar_c = ['#A83D49', '#C08A55', '#5C87A8', ACCENT]
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.50), layout='constrained')
    for k, (ax, d, name) in enumerate(zip(axs, DS[3:], NAMES[3:])):
        vals = [float(a[(a.dataset == d) & (a.variant == v)]['ndcg@10'].iloc[0]) for v, _ in cells]
        ref = float(a[(a.dataset == d) & (a.variant == 'no_path')]['ndcg@10'].iloc[0])
        ax.bar(range(4), vals, color=bar_c, width=.62, zorder=3)
        ax.axhline(ref, color='#2F3A42', lw=.9, ls=(0, (4, 2)), zorder=4)
        ax.text(3.45, ref * 1.012, 'no path evidence', ha='right', va='bottom',
                fontsize=6.4, color='#2F3A42')
        ax.annotate('', xy=(0, vals[0]), xytext=(0, ref), zorder=5,
                    arrowprops=dict(arrowstyle='<->', lw=.75, color='#A83D49',
                                    shrinkA=0, shrinkB=0))
        ax.text(.17, (vals[0] + ref) / 2,
                f'$-${(ref - vals[0]) * 1000:.1f}' + r'$\times10^{-3}$',
                fontsize=6.6, color='#A83D49', va='center')
        ax.set_xticks(range(4), [t for _, t in cells], fontsize=6.4)
        ax.set_ylim(0, max(max(vals), ref) * 1.22)
        ax.set_title(f'({chr(97 + k)}) {name}', fontsize=8.5)
        ax.grid(axis='y', alpha=.18, lw=.4)
        ax.tick_params(axis='x', length=0)
        if k == 0:
            ax.set_ylabel('NDCG@10', fontsize=7.5)
    save(fig, 'fig04_path')


def fig_ablation():
    """Fig. 4: deletions grouped by mechanism, one linear panel per regime."""
    a = pd.read_csv(R / 'final_analysis/ablations.csv')
    s = pd.read_csv(R / 'final_analysis/module_statistics.csv')

    # two-level structure: mechanism group -> control
    BLOCKS = [('Core operators',
               [('no_direction', 'w/o Direction'), ('no_path', 'w/o Path')]),
              ('Path safeguards',
               [('no_uncertainty', 'w/o Source gate'),
                ('no_conditional', 'w/o Candidate gate'),
                ('independent_paths', 'Independent')])]
    # y positions, with a gap between the two blocks
    ypos, ylab, heads = [], [], []
    y = 0.0
    for gi, (head, items) in enumerate(BLOCKS):
        heads.append((y - 0.62, head))
        for _, lab in items:
            ypos.append(y)
            ylab.append(lab)
            y += 1.0
        y += 0.85
    CTRL = [c for _, items in BLOCKS for c in items]

    # restrained palette: one family per regime, markers separate datasets
    REG = [(DS[:3], NAMES[:3],
            ['#1F4E6B', '#5B7A8C', '#9FB0BC'], ['o', 's', '^'], '(a) Amazon'),
           (DS[3:], NAMES[3:],
            ['#8C2F39', '#5D4A6B'], ['D', 'v'], '(b) MovieLens')]

    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.05), layout='constrained')
    for k, (ax, (ds, nm, cols, mks, title)) in enumerate(zip(axs, REG)):
        off = np.linspace(-.20, .20, len(ds))
        lo_all, hi_all = [0.0], [0.0]
        for r, (v, _) in enumerate(CTRL):
            for i, (d, c, mk) in enumerate(zip(ds, cols, mks)):
                f = float(a[(a.dataset == d) & (a.variant == 'full')]['ndcg@10'].iloc[0])
                x = float(a[(a.dataset == d) & (a.variant == v)]['ndcg@10'].iloc[0])
                st = s[(s.dataset == d) & (s.control == v) & (s.metric == 'ndcg10')].iloc[0]
                val, lo, hi = (f - x) * 1000, st.ci95_low * 1000, st.ci95_high * 1000
                lo_all.append(lo); hi_all.append(hi)
                yy = ypos[r] + off[i]
                ax.plot([lo, hi], [yy, yy], color=c, lw=.9,
                        solid_capstyle='butt', zorder=2)
                ax.plot([val], [yy], marker=mk, color=c, markersize=3.6, zorder=3,
                        markerfacecolor=c if st.p_holm < .05 else 'white',
                        markeredgewidth=.75)
        pad = (max(hi_all) - min(lo_all)) * .10
        ax.set_xlim(min(lo_all) - pad, max(hi_all) + pad)
        ax.axvline(0, color='#2F3A42', lw=.85, zorder=1)
        ax.set_yticks(ypos, ylab if ax is axs[0] else [''] * len(ypos), fontsize=7.2)
        ax.set_ylim(max(ypos) + .75, min(ypos) - 1.05)
        ax.tick_params(axis='y', length=0)
        ax.set_title(title, fontsize=8.5, pad=2)
        ax.grid(axis='x', alpha=.16, lw=.4)
        for hy, head in heads:                       # mechanism-group headers
            ax.text(0.015, hy, head, transform=ax.get_yaxis_transform(),
                    fontsize=6.8, style='italic', color='#4A5560',
                    ha='left', va='center')
        ax.axhline((ypos[1] + ypos[2]) / 2, color='#D7DDE2', lw=.6, zorder=0)
        handles = [plt.Line2D([], [], color=c, marker=mk, lw=0, markersize=3.5, label=n)
                   for n, c, mk in zip(nm, cols, mks)]
        ax.set_xlabel(r'$\Delta$NDCG@10 $=$ Full $-$ Control $(\times 10^{-3})$',
                      fontsize=7.0)
        ax.legend(handles=handles, loc='lower right' if k == 0 else 'upper right',
                  frameon=False, fontsize=6.6,
                  handlelength=1.0, borderpad=.15, labelspacing=.22)
    save(fig, 'fig05_ablation')


def fig_etasweep():
    """Robustness of path strength across sparsity regimes: ungated vs. gated
    path term, NDCG@10 relative to the no-path model, across a validation-only
    eta sweep on all five datasets. Answers whether local conditioning does
    something a single globally-tuned strength cannot."""
    import json
    rows = []
    for d in DS:
        f = R / 'final_analysis/eta_sweep' / f'{d}.json'
        if not f.exists():
            print('SKIP (no data yet):', d)
            continue
        rows += json.loads(f.read_text())
    if not rows:
        print('fig_etasweep: no data, skipping')
        return
    df = pd.DataFrame(rows)

    fig = plt.figure(figsize=(WIDTH, 4.75), layout='constrained')
    gs = fig.add_gridspec(2, 6)
    slots = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6], gs[1, 1:3], gs[1, 3:5]]
    handles = None
    for k, (d, name, c) in enumerate(zip(DS, NAMES, COL)):
        sub = df[df.dataset == d]
        if sub.empty:
            continue
        ax = fig.add_subplot(slots[k])
        ref = float(sub[(sub.eta == 0)]['ndcg10'].iloc[0])
        for variant, style, lab in (('ungated', dict(color='#A83D49', marker='o'), 'Ungated'),
                                     ('full_gated', dict(color=c, marker='s'), 'Gated')):
            v = sub[sub.variant == variant].sort_values('eta')
            v = v[v.eta > 0]
            xs = v.eta.to_numpy()
            ys = (v.ndcg10.to_numpy() - ref) / ref * 100
            ax.plot(xs, ys, lw=1.1, markersize=3.4, zorder=3, label=lab, **style)
        ax.axhline(0, color='#2F3A42', lw=.8, ls=(0, (4, 2)), zorder=1)
        ax.set_xscale('log')
        ax.set_title(f'({chr(97 + k)}) {name}', fontsize=8.2)
        ax.grid(alpha=.16, lw=.4)
        ax.tick_params(labelsize=6.6)
        if k in (0, 3):
            ax.set_ylabel(r'$\Delta$NDCG@10 vs. no path (%)', fontsize=6.8)
        if k >= 3:
            ax.set_xlabel(r'path strength $\eta$', fontsize=7)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    # One shared legend for the whole figure, centred above all panels,
    # rather than nested inside a single panel.
    fig.legend(handles, labels, loc='outside upper center', ncol=2,
               frameon=False, fontsize=7.2, handlelength=1.4,
               borderpad=.15, labelspacing=.3, columnspacing=1.4)
    save(fig, 'fig06_etasweep')


def fig_overlap():
    """Why conditioning matters: candidate-gate value and path-mass share,
    binned by the candidate's own direct support (deciles, sparsest to
    densest). Two single-purpose panels sharing an x-axis, rather than one
    dual-axis overlay, so each curve set can be read on its own scale."""
    import json
    records = {}
    for d in DS:
        f = R / 'final_analysis/overlap' / f'{d}.json'
        if f.exists():
            records[d] = json.loads(f.read_text())
    if not records:
        print('fig_overlap: no data, skipping')
        return

    fig, (axg, axm) = plt.subplots(1, 2, figsize=(WIDTH, 2.55),
                                    layout='constrained', sharex=True)
    handles, labels = [], []
    for d, name, c, mk in zip(DS, NAMES, COL, MARK):
        if d not in records:
            continue
        dec = records[d]['deciles']
        x = [r['decile'] for r in dec]
        gate = [r['mean_gate'] for r in dec]
        mass = [r['mass_share'] * 100 for r in dec]
        h, = axg.plot(x, gate, color=c, marker=mk, lw=1.1, markersize=3.2,
                      label=name, zorder=3)
        axm.plot(x, mass, color=c, marker=mk, lw=1.1, markersize=3.2,
                 zorder=3)
        handles.append(h); labels.append(name)

    axg.set_title('(a) Mean candidate gate', fontsize=8.2)
    axg.set_ylabel('mean candidate gate $g^c$', fontsize=7.5)
    axg.set_ylim(-0.03, 1.03)
    axm.set_title('(b) Share of path mass', fontsize=8.2)
    axm.set_ylabel('share of path mass (%)', fontsize=7.5)
    for ax in (axg, axm):
        ax.set_xlabel("direct-support decile\n(sparsest $\\to$ densest)",
                      fontsize=7)
        ax.set_xticks(range(10))
        ax.tick_params(labelsize=6.8)
        ax.grid(alpha=.16, lw=.4)
    fig.legend(handles, labels, loc='outside upper center', ncol=5,
               frameon=False, fontsize=7, handlelength=1.2, borderpad=.15,
               labelspacing=.25, columnspacing=1.1)
    save(fig, 'fig07_overlap')


if __name__ == '__main__':
    fig_direction()
    fig_path()
    fig_ablation()
    fig_etasweep()
    fig_overlap()
    print('figures:', sorted(p.name for p in OUT.glob('*.pdf')))
