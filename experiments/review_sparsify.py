"""Controlled within-dataset sparsification of MovieLens (supplement Sec. 2).

Sparsity is confounded with dataset family (Amazon vs MovieLens). This
truncates each user's history to the last N interactions inside a single
dataset, holding domain / preprocessing / semantics fixed, and tracks:
  (a) ungated-vs-gated path gap, relative to the no-path model
  (b) direct/path overlap
  (c) gate activation

Training-internal ('valid') split only; test is never touched.

NOTE: the evaluation set shrinks as truncation removes items from the
catalogue, so absolute NDCG is NOT comparable across truncation levels.
Within one level all three variants share an evaluation set, so the
relative gaps -- which is what the claim rests on -- are comparable.

--report prints the table the supplement is built from.
"""
import sys, time, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from dpt_sr.data import ROOT, load_split
from dpt_sr.model import DPTConfig
from dpt_sr.calibration import CalibrationConfig
from dpt_sr.full_model import CompletionConfig, fit_completion, recommend_completion
from dpt_sr.budget_model import BudgetConfig, apply_budget, direct_counts
from capacity_probes.sequences import SequenceStore, EvaluationSet

LEVELS = [None, 64, 32, 16, 8]          # None = untruncated


def truncate(frame, n):
    """Keep each user's LAST n interactions (chronological)."""
    if n is None:
        return frame
    ordered = frame.sort_values(["user_id", "datetime", "_source_order"], kind="stable")
    return ordered.groupby("user_id", sort=False).tail(n)


def variant_scores(model, ev, comp_evidence):
    v = type(model)(model.base, model.completion_config, comp_evidence, {})
    local, _ = recommend_completion(v, ev)
    return float(local.ndcg10), float(local.recall10)


def run(dataset, out_dir, cache_root):
    spec = json.loads((ROOT / "configs/final_redesign.json").read_text())[dataset]["spec"]
    train_full, held, _ = load_split(dataset, ROOT / "data", "valid")
    print(f"[{dataset}] loaded train rows={len(train_full)}", flush=True)

    records = []
    for level in LEVELS:
        t0 = time.perf_counter()
        train = truncate(train_full, level)
        store = SequenceStore.from_frame(train)
        ev = EvaluationSet.from_holdout(store, held)
        hist_len = (store.ends - store.starts)
        tag = "full" if level is None else str(level)

        cache = cache_root / f"{dataset}_{tag}"
        cache.mkdir(parents=True, exist_ok=True)
        with threadpool_limits(limits=1):
            model = fit_completion(store, spec["pctm"], DPTConfig(**spec["dpt"]),
                                   CalibrationConfig(**spec["calibration"]),
                                   CompletionConfig(**spec["completion"]),
                                   spec["max_history"], cache)

            n = model.completion_evidence.shape[0]
            tau = model.pctm_config.tau
            eta = model.completion_config.strength
            e = model.completion_evidence

            # --- (b) overlap and (c) gate activation, measured pre-gate ---
            if e.nnz:
                row = np.repeat(np.arange(n), np.diff(e.indptr))
                B = np.expm1(e.data) * tau / (n * eta)
                a = direct_counts(model)
                Ac = np.asarray(a[row, e.indices]).ravel()
                mass_tot = float(B.sum())
                redundant = Ac > 0
                frac_edges = float(redundant.mean())
                frac_mass = float(B[redundant].sum() / mass_tot) if mass_tot > 0 else None
                gate_c = tau / (tau + n * Ac)
                mean_gate_c = float(np.average(gate_c, weights=B)) if mass_tot > 0 else None
                src_mass = np.asarray(a.sum(axis=1)).ravel()
                u = tau / (tau + src_mass)
                median_u = float(np.median(u))
            else:
                frac_edges = frac_mass = mean_gate_c = median_u = None

            # --- (a) the gap: no-path / ungated / fully gated, same eval set ---
            empty = sparse.csr_matrix(e.shape, dtype=float)
            nd_nopath, rc_nopath = variant_scores(model, ev, empty)
            nd_ungated, rc_ungated = variant_scores(model, ev, e.copy())
            gated = type(model)(model.base, model.completion_config, e.copy(), {})
            apply_budget(gated, BudgetConfig(conditional=True, uncertainty=True))
            nd_gated, rc_gated = variant_scores(model, ev, gated.completion_evidence)

        rec = dict(dataset=dataset, level=tag,
                   mean_history=float(hist_len.mean()),
                   median_history=float(np.median(hist_len)),
                   items=int(store.num_items), eval_users=int(len(ev.users)),
                   train_rows=int(len(train)),
                   ndcg_nopath=nd_nopath, ndcg_ungated=nd_ungated, ndcg_gated=nd_gated,
                   rel_ungated=(nd_ungated - nd_nopath) / nd_nopath * 100,
                   rel_gated=(nd_gated - nd_nopath) / nd_nopath * 100,
                   frac_edges_redundant=frac_edges, frac_mass_redundant=frac_mass,
                   mass_weighted_gate=mean_gate_c, median_source_gate=median_u,
                   seconds=time.perf_counter() - t0)
        records.append(rec)
        print(f"[{dataset}] level={tag:>4s} meanhist={rec['mean_history']:6.1f} "
              f"ungated={rec['rel_ungated']:+7.2f}%  gated={rec['rel_gated']:+6.2f}%  "
              f"mass_overlap={frac_mass if frac_mass is None else round(frac_mass,3)}  "
              f"gate={mean_gate_c if mean_gate_c is None else round(mean_gate_c,4)}  "
              f"({rec['seconds']:.0f}s)", flush=True)
        (out_dir / f"{dataset}.json").write_text(json.dumps(records, indent=2))

    print(f"[{dataset}] WROTE {out_dir / (dataset + '.json')}", flush=True)


def report(out_dir):
    for ds in ("ml1m", "ml20m"):
        f = out_dir / f"{ds}.json"
        if not f.exists():
            print(f"(no data yet: {ds})")
            continue
        rows = json.loads(f.read_text())
        print(f"\n=== {ds} ===")
        print(f"{'level':>6s} {'mean_hist':>9s} {'items':>7s} {'eval_u':>7s} "
              f"{'ungated%':>9s} {'gated%':>8s} {'gap(pp)':>8s} "
              f"{'mass_ovlp':>10s} {'gate':>8s} {'med_u':>7s}")
        for r in rows:
            gap = r["rel_gated"] - r["rel_ungated"]
            mo = r["frac_mass_redundant"]
            g = r["mass_weighted_gate"]
            mu = r["median_source_gate"]
            print(f"{r['level']:>6s} {r['mean_history']:>9.1f} {r['items']:>7d} "
                  f"{r['eval_users']:>7d} {r['rel_ungated']:>+9.2f} {r['rel_gated']:>+8.2f} "
                  f"{gap:>+8.2f} "
                  f"{('-' if mo is None else f'{mo:.3f}'):>10s} "
                  f"{('-' if g is None else f'{g:.4f}'):>8s} "
                  f"{('-' if mu is None else f'{mu:.4f}'):>7s}")

    print("\nColumns: ungated%/gated% = NDCG@10 vs that level's own no-path baseline.")
    print("gap(pp) = gated - ungated, in percentage points: how much the gates buy.")
    print("mass_ovlp = share of pre-gate path mass on candidates with any direct support.")
    print("gate = mass-weighted mean candidate gate; med_u = median source gate.")
    print("Absolute NDCG is NOT comparable across levels (eval set shrinks); the")
    print("within-level relative columns are.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["ml1m", "ml20m", "beauty", "sports", "toys"])
    p.add_argument("--report", action="store_true", help="print the stored sweep instead of running")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/sparsification")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "review_cache/sparsify")
    a = p.parse_args()
    if a.report:
        report(a.output_dir)
    elif a.dataset:
        a.output_dir.mkdir(parents=True, exist_ok=True)
        run(a.dataset, a.output_dir, a.cache_dir)
    else:
        p.error("--dataset or --report is required")
