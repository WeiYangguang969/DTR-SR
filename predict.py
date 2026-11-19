"""Load the saved model and recommend from a chronological external-ID history."""
import argparse
import json
from pathlib import Path

import numpy as np

from capacity_probes.sequences import EvaluationSet, SequenceStore
from dpt_sr.final_model import recommend_final, load_final as load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--history", nargs="+", type=int, required=True, help="Oldest to newest; original dataset item IDs")
    args = p.parse_args()
    model = load(args.model_dir)
    external = np.asarray(args.history, dtype=np.int64)
    items = np.searchsorted(model.catalog, external)
    valid = items < len(model.catalog)
    pos = np.flatnonzero(valid)
    valid[pos] = model.catalog[items[pos]] == external[pos]
    if not np.all(valid):
        p.error(f"Unknown IDs {external[~valid].tolist()}; IDs must belong to this model's training catalog")
    items = items.astype(np.int32)
    store = SequenceStore(model.catalog, np.ones(len(items), dtype=np.int64), items,
                          np.asarray([1]), np.asarray([0]), np.asarray([len(items)]), {1: 0})
    # The placeholder target is used only by the shared collector, never for scoring.
    evaluation = EvaluationSet(store, np.asarray([1]), np.asarray([0]), (np.asarray([0]),))
    rec = recommend_final(model, evaluation, k=10)
    print(json.dumps(rec[["item_id", "score", "rank"]].to_dict("records"), indent=2))


if __name__ == "__main__":
    main()
