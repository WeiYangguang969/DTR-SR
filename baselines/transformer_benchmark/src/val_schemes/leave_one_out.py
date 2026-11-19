from rectools.metrics import (
    NDCG,
    AvgRecPopularity,
    CatalogCoverage,
    CoveredUsers,
    Recall,
    SufficientReco,
)

from src.evaluation.splitter import LastNSplitter

# ####  --------------  Holdout splitter  --------------  #### #

HOLDOUT_SPLITTER = LastNSplitter(
    n=1,
    n_splits=1,
    filter_cold_users=True,
    filter_cold_items=True,
    filter_already_seen=True,
)


# ####  --------------  Cross-validation splitter  --------------  #### #

CV_N_SPLITS = 1
CV_SPLITTER = LastNSplitter(
    n=1,
    n_splits=CV_N_SPLITS,
    filter_cold_users=True,
    filter_cold_items=True,
    filter_already_seen=True,
)

# ####  --------------  Metrics  --------------  #### #

# LOCAL MODIFICATION (evaluation depth only).
# `k` is used solely for `model.recommend(..., k=k)` after `model.fit()`, so
# raising it does not alter training or early stopping. NDCG@10 and Recall@10
# computed from a depth-20 list are identical to the same metrics computed from
# a depth-10 list, so the published @10 values remain a direct check that this
# change is inert.
K = 20
REPORT_KS = (10, 20)
METRICS = {}
for _k in REPORT_KS:
    METRICS[f"recall@{_k}"] = Recall(k=_k)
    METRICS[f"ndcg@{_k}"] = NDCG(k=_k, divide_by_achievable=True)
METRICS.update(
    {
        f"arp@{K}": AvgRecPopularity(k=K, normalize=True),
        f"coverage@{K}": CatalogCoverage(k=K, normalize=True),
        f"covered_users@{K}": CoveredUsers(k=K),
        f"sufficient_reco@{K}": SufficientReco(k=K),
    }
)
