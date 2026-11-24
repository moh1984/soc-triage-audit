# Result files

CSV outputs of the runs behind the paper. Per-event score arrays (`*.npz`) and incident score tables
(`*.parquet`) are omitted to keep the repository small; every summary below was computed from them by
the scripts and is reproduced by re-running the listed step.

| Folder | Written by | Used in the paper for |
|---|---|---|
| `eda/` | `02_eda.py` (released small version) | Sec. 3.1, 4.2, Fig. 1 (subset bars) |
| `leakage/leakage_audit.csv` | `03_leakage_audit.py` | Sec. 4.1–4.2 (label-derived columns, log-source shortcut) |
| `extract/escalation_by_group.csv` | `05b_summarize_100m.py` | Table 1, Fig. 1 (full-capture bars) |
| `diagnostics/` | `06_diagnose_temporal.py --data 100m` | Sec. 5.1, Fig. 3 (where the naive cut falls, daily escalations, per-stream checks) |
| `first_featurization/` | `extras/first_featurization/first_featurization_check.py` (original phase-1 featurizer) | Sec. 5.1, first featurization (reversed ranking under the naive cut; dst_ip frequency flip) |
| `baselines/baselines_escalation_100m.csv` | `04_baselines.py --data 100m --only naive_global_temporal,incident_grouped` | Sec. 5.1 (corrected features: naive cut vs incident-grouped split) |
| `seeds/*_gbdt_all_rerun/` | `07_repeat_seeds.py --data 100m --models lgbm_all --tag gbdt_all_rerun --n-cluster-boot 0` | Table 5 and Fig. 4, row "GBDT, all features" |
| `seeds/*_logit/` | `07_repeat_seeds.py ... --tag logit` | check of the `decision_function` change (Sec. 6.2) |
| `ctxgraph/full_v2/`, `ctxgraph/cross_uda_v2/` | `08_ctxgraph.py` (grouped validation) | Table 7, Sec. 7.5 (reported version) |
| `ctxgraph/abl_nomem/`, `abl_noanom/`, `abl_notext/` | `08_ctxgraph.py --no-memory / --no-anomaly / --text none` | Table 7 ablations |
| `ctxgraph/asa_temporal_vpc_wave/`, `asa_to_vpc_vpc_to_asa_uda/`, `cross_no_uda/`, `asa_temporal/` | `08_ctxgraph.py`, first version (time-slice validation) | Sec. 7.5, first version (superseded; quoted for transparency) |
| `noncausal/` | `10_noncausal_context.py --data 100m` (first run, `predict_proba`) | superseded; Table 8 comes from `logs/noncausal_cluster_output.txt` |
| `noncausal/logit/` | `10_noncausal_context.py ... --tag logit` | check of the `decision_function` change (Sec. 6.2) |
| `guide_inspect/` | `11_guide_inspect.py` | Table 2, Sec. 3.4 |
| `guide_audit/` | `12_guide_audit.py` | Table 3, Sec. 4.4 |
| `guide_robust/` | `13_guide_robust.py` | Fig. 2 (`D_org_purity.csv`); its temporal protocol is superseded |
| `guide_id_check/` | `14_guide_id_check.py` | Sec. 4.4 (identifiers shared across organizations) |
| `guide_v2/` | `15_guide_v2.py` | Table 9, Fig. 5, Sec. 5.3 and 7.7 |
| `guide_nrows_sensitivity/` | `17_guide_nrows_sensitivity.py` | Sec. 9 (sensitivity of the content model to how `n_rows_alert` is counted) |

## Re-runs

The original Colab session that produced `diagnostics/`, `baselines/` and the first five-seed baseline
run lost its `results/` folder. These outputs (and `first_featurization/`) were regenerated on 24 November 2026 from the same
code and the same full-capture extract (`data/100m/rulehits.parquet`), with LightGBM 4.7.0 and
scikit-learn 1.9.1; console output is in `logs/*_rerun_output.txt`. Differences from the published
numbers are listed in `logs/README.md`.

The main seven-baseline run behind Tables 5–6 (`07_repeat_seeds.py --tag cluster`) is archived as its
full console log, `logs/precinct6_cluster_output.txt`; its SUMMARY and PAIRED COMPARISONS blocks hold
every number used in the paper.
