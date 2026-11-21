# Paired MGCN / DAMPS Hyperparameter Search

English · [简体中文](zh/search.md) · [Documentation](README.md)

`python -m mmrecsys.experiment.search` evaluates the baseline and full DAMPS at the same backbone settings, data split, seeds, and budget. The candidates are ranked on validation data only; each run selects its own best epoch.

## Defaults and ranking

`configs/search/damps_mgcn.yaml` fixes two UI layers, one item layer, and k=10, and tries `model.cl_weight` values 0.001, 0.01, and 0.1. One seed yields six training runs; three seeds yield eighteen. The optional `configs/search/damps_mgcn_extended.yaml` is a wider grid, not a paper requirement. Other settings are inherited from the base configuration. The paper does not disclose a best contrastive weight for each dataset.

With the default relative objective, compute the percentage gain from baseline to DAMPS validation Recall@20 for each seed, then average those gains. A candidate with a zero baseline on any seed is ineligible for the relative ranking; `--objective absolute` uses the mean raw difference. Equal objective scores favor the higher mean DAMPS validation score, then grid order. `--min-baseline` can exclude settings whose mean baseline falls below a chosen floor. The example Baby threshold is not universal.

```bash
python -m mmrecsys.experiment.search --dry-run
python -m mmrecsys.experiment.search --device cuda:0 --seeds 999 2024 2025 --min-baseline 0.09
python -m mmrecsys.experiment.search --config configs/experiments/damps_mgcn_sports.yaml --device cuda:0
python -m mmrecsys.experiment.search --grid configs/search/damps_mgcn_extended.yaml --dry-run
```

Use `--limit 2` to run only the first two settings for a workflow check. A custom `--grid` YAML maps supported dotted keys to nonempty candidate lists: `model.n_ui_layers`, `model.n_item_layers`, `model.knn_k`, `model.cl_weight`, `model.reg_weight`, `model.embedding_dim`, `model.temperature`, and `optimizer.lr`. Grid values take precedence over matching `--set` values.

## Outputs, resume, and final testing

`plan.json` freezes the base configuration, grid, seeds, objective, and source/data fingerprints. `leaderboard.json` records completed paired candidates and `best.json` identifies the current winner. `best_baseline.yaml` and `best_full.yaml` are reusable training configurations. Candidate runs write `validation_result.json` instead of test metrics.

Replace `YOUR_SEARCH_DIRECTORY` with the directory printed by search:

```bash
python -m mmrecsys.experiment.search --resume YOUR_SEARCH_DIRECTORY
python -m mmrecsys.experiment.search --resume YOUR_SEARCH_DIRECTORY --evaluate-best
```

Resume uses the frozen plan, skipping completed runs and resuming unfinished runs from `last.pt` if present. It rejects source or data changes; do not run two resumes against the same directory. `--evaluate-best` writes `best_test.json` for the validation winner after the search completes without changing the ranking.

Maximizing validation gain is an exploratory objective and can favor a weak baseline. If the goal is to tune the baseline first, select its parameters using its own validation scores and hold them fixed for DAMPS. Report absolute scores and seed variability as well as gains. The selected grid result does not guarantee test improvement or statistical significance.
