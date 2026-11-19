"""Direct-evidence overlap of two-hop path mass (main paper Sec. 4.7, Fig. 6).

Does two-hop path mass concentrate on candidates that already have strong
direct support? This is the mechanism claim behind the candidate gate. Fits
each dataset once on the training-internal split, then measures overlap
between A^c (direct evidence, count scale) and B (two-hop evidence, count
scale) at retained path edges.
"""
import sys, time, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from threadpoolctl import threadpool_limits
from dpt_sr.data import ROOT, load_split
from dpt_sr.model import DPTConfig
from dpt_sr.calibration import CalibrationConfig
from dpt_sr.full_model import CompletionConfig, fit_completion
from dpt_sr.budget_model import direct_counts
from capacity_probes.sequences import SequenceStore, EvaluationSet


def run(dataset, out_dir, cache_root):
    t0 = time.perf_counter()
    spec = json.loads((ROOT / "configs/final_redesign.json").read_text())[dataset]["spec"]
    train, held, verified = load_split(dataset, ROOT / "data", "valid")
    store = SequenceStore.from_frame(train)
    ev = EvaluationSet.from_holdout(store, held)
    print(f"[{dataset}] loaded items={store.num_items} ({time.perf_counter()-t0:.1f}s)", flush=True)

    cache_dir = cache_root / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        model = fit_completion(store, spec["pctm"], DPTConfig(**spec["dpt"]),
                                CalibrationConfig(**spec["calibration"]),
                                CompletionConfig(**spec["completion"]),
                                spec["max_history"], cache_dir)
        print(f"[{dataset}] fit done ({time.perf_counter()-t0:.1f}s)", flush=True)

        n = model.completion_evidence.shape[0]
        tau = model.pctm_config.tau
        eta = model.completion_config.strength
        e = model.completion_evidence
        row = np.repeat(np.arange(n), np.diff(e.indptr))
        B = np.expm1(e.data) * tau / (n * eta)          # count-scale two-hop evidence

        a = direct_counts(model)                         # A^c, count scale
        Ac = np.asarray(a[row, e.indices]).ravel()

        total_mass = float(B.sum())
        redundant = Ac > 0
        frac_edges_redundant = float(redundant.mean())
        frac_mass_redundant = float(B[redundant].sum() / total_mass) if total_mass > 0 else None

        g = tau / (tau + n * Ac)                          # candidate gate value at each edge
        mass_wtd_gate = float(np.average(g, weights=B)) if total_mass > 0 else None
        median_gate = float(np.median(g))
        edge_wtd_gate_on_redundant = float(np.mean(g[redundant])) if redundant.any() else None

        # decile summary: how does the gate value vary with direct support depth?
        order = np.argsort(Ac)
        deciles = []
        k = len(Ac) // 10 or 1
        for i in range(10):
            sl = order[i * k: (i + 1) * k] if i < 9 else order[i * k:]
            if len(sl) == 0:
                continue
            deciles.append(dict(decile=i, mean_Ac=float(Ac[sl].mean()),
                                 mean_gate=float(g[sl].mean()),
                                 mass_share=float(B[sl].sum() / total_mass) if total_mass > 0 else None))

    result = dict(dataset=dataset, n_items=n, tau=tau, eta=eta, nnz=int(e.nnz),
                  frac_edges_redundant=frac_edges_redundant,
                  frac_mass_redundant=frac_mass_redundant,
                  mass_weighted_gate=mass_wtd_gate, median_gate=median_gate,
                  edge_weighted_gate_on_redundant=edge_wtd_gate_on_redundant,
                  deciles=deciles, seconds=time.perf_counter() - t0)
    (out_dir / f"{dataset}.json").write_text(json.dumps(result, indent=2))
    print(f"[{dataset}] DONE: redundant_edges={frac_edges_redundant:.1%} "
          f"redundant_mass={frac_mass_redundant:.1%} "
          f"mass_wtd_gate={mass_wtd_gate:.4f}  ({time.perf_counter()-t0:.1f}s total)", flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                    choices=["beauty", "sports", "toys", "ml1m", "ml20m"])
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/final_analysis/overlap")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "review_cache/overlap")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    run(a.dataset, a.output_dir, a.cache_dir)
