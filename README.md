# DPT-SR reproduction artifact

This snapshot accompanies *DPT-SR: Reliability-Weighted Transition Evidence for Sequential Recommendation*. The [supplementary material](supplementary/supplementary.pdf) contains the matched PCTM calibration, the within-MovieLens history-truncation study and the implementation details. Run the commands below from the directory containing this README.

## Reproduce the reported tables offline

These two commands require Python 3.10 or later and NumPy. All inputs for them are in the archive.

```sh
python verify_final.py --skip-model-crc
python paper_tables.py
```

The first command checks 40 paired user-rank archives against their recorded hashes and recomputes six metrics for DPT-SR and its seven controls. The second prints Tables 3 and 4 in CSV format, with a source column. The same output is in [results/paper_tables.csv](results/paper_tables.csv). Print the values to four decimal places to match the manuscript.

Each table row contains NDCG@10, NDCG@20, Recall@10 and Recall@20. For DPT-SR and the five non-transformer baselines, the table script checks these metrics against user ranks. For the eight completed transformer runs it checks the four metrics against the corresponding evaluation report. For SASRec+ and eSASRec on ML-20M, it retrieves the two published @10 values from the bundled snapshot of upstream metadata and leaves @20 blank. This gives 156 reported values across 40 rows. A zero in a baseline's `rank_in_topk` array denotes a target outside its top 20.

## Fit DPT-SR from interactions

The reference scoring environment used Python 3.12, NumPy 1.26.4, SciPy 1.12.0, pandas 2.3.3 and RecTools 0.13.0. Install dependencies in a fresh environment. The full reference lock is in `requirements-lock.txt`. The three Amazon training and holdout splits are bundled and checked against SHA-256 before scoring.

```sh
python -m pip install -r requirements.txt
python main.py --dataset beauty --output fresh_beauty
python reproduce_all.py --datasets beauty --output fresh_ablations
```

The first fit writes a model checkpoint, full-catalogue top-20 recommendations and RecTools metrics at both cutoffs. It uses the released scoring coordinates from `configs/final_redesign.json`. The second command fits DPT-SR and its seven controls and profiles the loaded model. Omit `--datasets beauty` to run all five datasets. Use a fresh output path for a new fit. The archived study's timings are in `results/final_redesign/<dataset>/profile.json`.

The dataset identifiers are `beauty`, `sports`, `toys`, `ml1m` and `ml20m`. To prepare MovieLens for any fresh evaluation, run the following once per dataset. The original ratings stay with their provider. Both train and holdout must match the distributed hashes. A verified mirror is also supported via `--source mirror`.

```sh
python experiments/prepare_external.py --dataset ml1m --source public
python experiments/prepare_external.py --dataset ml20m --source public
```

## Reproduce mechanism and supplementary results

| Result | Archived evidence | Fresh evaluation |
|---|---|---|
| Fig. 2, held-out orientation error | `results/review/mechanism/<dataset>/direction_events.npz`, `results/analysis/direction_calibration.csv` | `python experiments/review_validation.py mechanism --dataset beauty --output fresh_direction` |
| Figs. 3–4, path interventions and deletions | `results/final_redesign/<dataset>/<variant>/per_user.npz`, `results/final_analysis/module_statistics.csv` | `python reproduce_all.py --output fresh_ablations` |
| Fig. 5, path-strength response | `results/final_analysis/eta_sweep/` | `python experiments/review_eta_sweep.py --dataset beauty --output-dir fresh_eta` |
| Table 5, Amazon structural parameters | `results/analysis/sensitivity.csv` | `python experiments/run_case.py --dataset beauty --split valid --strength 0.1 --output fresh_eta01 --cache-dir count_cache` (use `--fanout` for the second block) |
| Fig. 6, direct/path overlap | `results/final_analysis/overlap/` | `python experiments/review_overlap.py --dataset beauty --output-dir fresh_overlap` |
| Supplement, matched PCTM calibration | `results/review_r2/matched_pctm/` and the upstream PCTM user ranks in `upstream_runs/` | `python experiments/review_matched_pctm.py beauty --output-dir fresh_matched` |
| Supplement, history truncation | `results/sparsification/` | `python experiments/review_sparsify.py --dataset ml1m --output-dir fresh_truncation` |

To regenerate experimental figures from the archived data, run `python plots/make_figures.py`. It writes PDF, SVG and PNG files under `plots/figures/`. For a quick check of the supplementary results, run `python experiments/review_matched_pctm_table.py` and `python experiments/review_sparsify.py --report`.

To repeat the original DPT-SR scoring-coordinate selection, use a separate working copy with Python 3.12 and `requirements-calibration.txt`, prepare the relevant split and run `python experiments/redesign_rankfit.py --dataset beauty`. The script recomputes the training-internal graph and writes to `results/redesign/beauty/` in that copy. The fitting configurations, validation protocols and original algorithm hashes are included for comparison.

## Baseline implementations

The five MC/FMC/FMC+/SeqRules/PCTM implementations and their published configurations are under `baselines/sequential-capacity-probes/`. A separate environment matching the upstream Python 3.10 lock in that directory can be used for baseline work. FMC and FMC+ require PyTorch. The scoring adapter asks for 20 candidates and evaluates both cutoffs on external item IDs. To rerun an example on a prepared split:

```sh
python run_baseline.py --dataset beauty --model mc
```

For the two transformer baselines, install the bundled benchmark in a GPU environment with RecTools 0.13.0, CUDA-enabled PyTorch and PyTorch Lightning. The dataset-specific YAML files in `upstream_runs/` are the configurations used for the eight completed runs. The validation and early-stopping configuration stays in the upstream trainer. Holdout evaluation requests top 20. The launcher reads the verified splits in the artifact's `data/` directory and writes new reports under `baselines/transformer_benchmark/reports/`.

```sh
python -m pip install -r baselines/transformer_benchmark/requirements.txt
python -m pip install -e baselines/transformer_benchmark
python run_baseline.py --dataset beauty --model sasrec_plus
```

The primary sources are [Sequential Capacity Probes, commit 705710f](https://github.com/spotify-research/sequential-capacity-probes/tree/705710f78abd613cd8ee139f7435d813eef2f8ab) and [transformer_benchmark, commit 2b03992](https://github.com/blondered/transformer_benchmark/tree/2b039927ceb4c7654131d7ac7c43ea124b49d240). See the bundled licenses and NOTICE. The portable copies use local paths for cached counts, baseline results and data. Transformer report metadata identifies the pinned source revision without requiring Git history. Historical hashes in result records refer to the recorded runs. The file hashes for this distribution are listed in `SHA256SUMS`.
