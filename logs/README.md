# Experiment logs

Full console output of the runs behind the paper. Lines with Python `warnings.warn` were removed
from `precinct6_cluster_output.txt`; nothing else was edited.

| File | Script and command | Used in the paper for |
|---|---|---|
| `guide_inspect_output.txt` | `11_guide_inspect.py --dir guide` | Table 2, Sec. 3.4 |
| `guide_audit_output.txt` | `12_guide_audit.py --dir guide` | Table 3, Sec. 4.4; one-seed random-row reference (Sec. 7.7); first, row-cut temporal protocol (superseded, quoted in Sec. 7.7) |
| `guide_robust_output.txt` | `13_guide_robust.py --dir guide` | Fig. 2 and organization purity (section D); its temporal protocol is superseded |
| `guide_id_check_output.txt` | `14_guide_id_check.py --dir guide` | Sec. 4.4, identifiers shared across organizations |
| `guide_v2_output.txt` | `15_guide_v2.py --dir guide` | Table 9, Fig. 5, Sec. 5.3 and 7.7 |
| `guide_nrows_sensitivity_output.txt` | `17_guide_nrows_sensitivity.py --dir guide` (seed 0, content features) | Sec. 9 (Threats to validity, GUIDE): `n_rows_alert` counted within each side of the split vs on both files |
| `guide_final_output.txt` | `16_guide_final.py --dir guide` | Sec. 4.4 (alerts in several incidents), Table 9 strict-temporal rows, Sec. 7.7 (IncidentId ablation) |
| `precinct6_cluster_output.txt` | `07_repeat_seeds.py --data 100m --tag cluster --models logreg_tab,logreg_tab+ctx,lgbm_tab,lgbm_tab+ctx,logreg_text,iforest_ctx,iforest_ctx_all` | Table 6, Table 5 (all rows except GBDT on all features), Sec. 7.2–7.4 |
| `noncausal_cluster_output.txt` | `10_noncausal_context.py --data 100m --tag cluster` | Table 8, Sec. 7.6 |
| `noncausal_output.txt` | `10_noncausal_context.py --data 100m` (first run) | superseded: logistic regression scored with `predict_proba` |
| `logit_rerun_output.txt` | `10_...` and `07_...` with `--tag logit` | check that the `decision_function` change affects only one cell (Sec. 6.2) |

Re-runs (24 November 2026). The Colab session behind the outputs below lost its `results/` folder,
so they were regenerated from the same code and the same full-capture extract; the CSV files are in
`results/` (see `results/README.md`).

| File | Script and command | Used in the paper for | Published vs re-run |
|---|---|---|---|
| `diagnose_temporal_rerun_output.txt` | `06_diagnose_temporal.py --data 100m` | Sec. 5.1, Fig. 3 | identical (cut 27 Jul 06:19 UTC; 3,998 / 3,573 escalations; daily counts) |
| `gbdt_all_rerun_output.txt` | `07_repeat_seeds.py --data 100m --models lgbm_all --tag gbdt_all_rerun --n-cluster-boot 0` | Table 5, row "GBDT, all features" | 0.062 ± 0.021, 0.0495 ± 0.017 (published 0.049), 0.028 ± 0.004, 0.025 ± 0.015 |
| `baselines_naive_grouped_rerun_output.txt` | `04_baselines.py --data 100m --only naive_global_temporal,incident_grouped` | Sec. 5.1, corrected features | LightGBM ROC-AUC 0.688 and 0.906 (the manuscript reports these re-run values; the original Colab run gave 0.687 and 0.909) |
| `first_featurization_rerun_output.txt` | `extras/first_featurization/first_featurization_check.py` | Sec. 5.1, first featurization | dst_ip frequency ROC-AUC 0.105 (train) and 0.731 (test): identical; LightGBM ROC-AUC 0.243 / 0.942 / 0.921 for naive cut / random / incident-grouped (the manuscript reports these re-run values; an earlier draft quoted 0.295 / 0.945 / 0.930 from an intermediate code version that was not archived) |

CtxGraph (Table 7, Sec. 7.5) and the GUIDE results were not re-run: their original CSV files are in
`results/ctxgraph/` and `results/guide_*`.
