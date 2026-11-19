"""Validation-only path-strength sweep (main paper Sec. 4.6, Fig. 5).

For each dataset, fit the graph ONCE (the eta-independent, expensive part),
then cheaply rescale + re-gate + rescore for every (eta, gate-variant) grid
point, reusing the trusted fit_completion / apply_budget /
recommend_completion / make_evidence functions unmodified.

Never touches the test split -- this is a training-internal holdout sweep,
exactly like the existing structural-parameter table.
"""
import sys, time, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from threadpoolctl import threadpool_limits
from dpt_sr.data import ROOT, load_split
from dpt_sr.model import DPTConfig
from dpt_sr.calibration import CalibrationConfig
from dpt_sr.full_model import CompletionConfig, fit_completion, recommend_completion, make_evidence
from dpt_sr.budget_model import BudgetConfig, apply_budget
from capacity_probes.sequences import SequenceStore, EvaluationSet

ETA_GRID = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
VARIANTS = [("ungated", False, False), ("full_gated", True, True)]


def run_dataset(dataset, out_dir, cache_root):
    t0 = time.perf_counter()
    spec = json.loads((ROOT / "configs/final_redesign.json").read_text())[dataset]["spec"]
    train, held, verified = load_split(dataset, ROOT / "data", "valid")
    store = SequenceStore.from_frame(train)
    ev = EvaluationSet.from_holdout(store, held)
    print(f"[{dataset}] loaded: items={store.num_items} users={len(ev.users)} "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    cache_dir = cache_root / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        t0 = time.perf_counter()
        model = fit_completion(store, spec["pctm"], DPTConfig(**spec["dpt"]),
                                CalibrationConfig(**spec["calibration"]),
                                CompletionConfig(**spec["completion"]),
                                spec["max_history"], cache_dir)
        t_fit = time.perf_counter() - t0
        print(f"[{dataset}] fit_completion done in {t_fit:.1f}s", flush=True)

        n = model.completion_evidence.shape[0]
        tau = model.pctm_config.tau
        eta0 = model.completion_config.strength
        # invert the log transform exactly to recover raw (pre-strength) two-hop counts
        e0 = model.completion_evidence
        raw = e0.copy()
        raw.data = np.expm1(e0.data) * tau / (n * eta0)

        records = []
        for eta in ETA_GRID:
            if eta == 0.0:
                comp = raw.copy()
                comp.data[:] = 0.0
                comp.eliminate_zeros()
            else:
                scaled = raw.copy()
                scaled.data = raw.data * eta
                comp, _ = make_evidence(scaled, tau, 0.0)
            for vname, cond, unc in VARIANTS:
                t1 = time.perf_counter()
                variant = type(model)(model.base, model.completion_config, comp.copy(), {})
                if eta > 0.0 and (cond or unc):
                    apply_budget(variant, BudgetConfig(conditional=cond, uncertainty=unc))
                local, _ = recommend_completion(variant, ev)
                dt = time.perf_counter() - t1
                rec = dict(dataset=dataset, eta=eta, variant=vname,
                           ndcg10=local.ndcg10, recall10=local.recall10, seconds=dt)
                records.append(rec)
                print(f"[{dataset}] eta={eta:<6g} {vname:<10s} "
                      f"ndcg10={local.ndcg10:.5f} ({dt:.1f}s)", flush=True)
                if eta == 0.0:
                    break  # no_path: gate choice is irrelevant, only run once

    out = out_dir / f"{dataset}.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"[{dataset}] WROTE {out}  total {time.perf_counter()-t0:.1f}s", flush=True)
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                    choices=["beauty", "sports", "toys", "ml1m", "ml20m"])
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/final_analysis/eta_sweep")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "review_cache/eta_sweep")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    run_dataset(a.dataset, a.output_dir, a.cache_dir)
