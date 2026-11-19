"""Refit a baseline using the bundled upstream implementation and the paper's cutoffs."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dpt_sr.data import DATASETS, verify_processed

ROOT = Path(__file__).resolve().parent
SCP = ROOT / "baselines/sequential-capacity-probes"
TRANSFORMERS = ROOT / "baselines/transformer_benchmark"
NEURAL = ("esasrec", "sasrec_plus")


def run(args):
    verify_processed(args.dataset, ROOT / "data")
    if args.model in NEURAL:
        config = ROOT / "upstream_runs" / args.dataset / args.model / "config.yaml"
        if not config.is_file():
            raise SystemExit("This run has no local transformer result. See the pinned @10 records in paper_tables.py.")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(TRANSFORMERS) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run([sys.executable, "src/evaluation/holdout_from_params.py", "--config_file", str(config)],
                       cwd=TRANSFORMERS, env=env, check=True)
        return

    sys.path.insert(0, str(SCP / "src"))
    os.environ["PCTM_EVAL_JOBS"] = "1"
    from capacity_probes.config import DatasetCase
    from capacity_probes.core_runner import _build_and_recommend, _load_case
    from capacity_probes.metrics import official_metrics

    config = json.loads((SCP / "configs/table1.json").read_bytes())
    ds = config["datasets"][args.dataset]
    case = DatasetCase(args.dataset, ds["directory"], ds["max_history"])
    train, held, store, evaluation, _ = _load_case(ROOT / "data", case)
    output = args.output / args.dataset / args.model
    if (output / "result.json").exists():
        raise FileExistsError(output / "result.json")
    output.mkdir(parents=True, exist_ok=True)
    _, _, reco, _ = _build_and_recommend(
        args.model, store, evaluation, ds.get(args.model, {}), output / "cache",
        case.max_history, args.device, config["protocol"]["refit_seed"],
    )
    metrics = {}
    for k in (10, 20):
        metrics.update(official_metrics(reco[reco["rank"] <= k], train, held, k=k))
    result = {"dataset": args.dataset, "model": args.model, "metrics": metrics}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True, choices=("mc", "fmc", "fmc_plus", "pctm", "seqrules", *NEURAL))
    parser.add_argument("--device", default="cpu", help="FMC/FMC+ training device. Transformers use their GPU trainer.")
    parser.add_argument("--output", type=Path, default=ROOT / "fresh_baselines",
                        help="Output root for the five non-transformer models.")
    run(parser.parse_args())
