"""Rebuild Tables 3 and 4 from the distributed evaluation records (CSV on stdout)."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATASETS = ("beauty", "sports", "toys", "ml1m", "ml20m")
MODELS = ("mc", "fmc", "fmc_plus", "sasrec_plus", "esasrec", "seqrules", "pctm", "DPT-SR")
METRICS = ("ndcg@10", "ndcg@20", "recall@10", "recall@20")


def read_json(path):
    return json.loads(path.read_bytes())


def check_ranks(path, field, users, metrics):
    with np.load(path, allow_pickle=False) as record:
        np.testing.assert_array_equal(record["user_id"], users)
        ranks = record[field]
        for k in (10, 20):
            hits = (ranks > 0) & (ranks <= k)
            gain = 1.0 / np.log2(1.0 + np.maximum(ranks, 1))
            np.testing.assert_allclose(
                [metrics[f"ndcg@{k}"], metrics[f"recall@{k}"]],
                [(gain * hits).mean(), hits.mean()], atol=1e-8, rtol=0,
            )


def table_rows():
    pinned = read_json(ROOT / "configs/published_baselines.json")["cleanroom"]
    for ds in DATASETS:
        full = ROOT / "results/final_redesign" / ds / "full"
        with np.load(full / "per_user.npz", allow_pickle=False) as archive:
            users = archive["user_id"]
        for model in MODELS:
            folder = ROOT / "upstream_runs" / ds / model
            if model == "DPT-SR":
                metrics = read_json(full / "result.json")["metrics"]
                check_ranks(full / "per_user.npz", "rank", users, metrics)
                source = "DPT-SR test ranks"
            elif ds == "ml20m" and model in ("sasrec_plus", "esasrec"):
                published = pinned[ds][model]
                metrics = {"ndcg@10": published["ndcg10"], "recall@10": published["recall10"]}
                source = published["source"]
            else:
                metrics = read_json(folder / "result.json")["metrics"]
                if model in ("sasrec_plus", "esasrec"):
                    with (folder / "metrics_report.csv").open(encoding="utf-8") as fp:
                        report = next(csv.DictReader(fp))
                    for name in METRICS:
                        np.testing.assert_allclose(metrics[name], float(report[name]), atol=1e-10, rtol=0)
                    source = "local transformer evaluation report"
                else:
                    check_ranks(folder / "per_user.npz", "rank_in_topk", users, metrics)
                    source = "local baseline top-20 ranks"
            yield {"dataset": ds, "model": model, **{k: metrics.get(k, "") for k in METRICS}, "source": source}


if __name__ == "__main__":
    writer = csv.DictWriter(sys.stdout, fieldnames=["dataset", "model", *METRICS, "source"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(table_rows())
