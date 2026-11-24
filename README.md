# What Do SOC Triage Models Learn? — code and logs

Code for the paper *"What Do SOC Triage Models Learn? Leakage, Sampling Bias, Organization
Identity and Protocol Effects on Two Real Multi-Organization Datasets"*
(M. Baklizi, M. Alkhazaleh, M. B. Alzghoul, L. Al Sardy).

The repository audits and benchmarks two public SOC datasets:

| Dataset | Version used | Task |
|---|---|---|
| WitFoo Precinct6 (small), Hugging Face `witfoo/precinct6-cybersecurity` | tag `v2.1.0` | audit only |
| WitFoo Precinct6 full capture, `witfoo/precinct6-cybersecurity-100m` | commit `ef20229d046b03c02061c396cd66b711745021ae` (2026-09-22) | **escalation**: malicious vs suspicious among 622,862 rule-matched events |
| Microsoft GUIDE, Kaggle `Microsoft/microsoft-security-incident-prediction` | `GUIDE_Train.csv`, `GUIDE_Test.csv` (downloaded 23–24 Oct 2026) | **triage**: TruePositive vs FalsePositive/BenignPositive |

Neither dataset is redistributed here; the scripts download them.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch                                       # only for 08_ctxgraph.py
```

Every script is run from the repository root. All experiments in the paper were run on
free Google Colab (2 CPU cores, about 12 GB RAM; a T4 GPU for `08_ctxgraph.py`).

Unit tests of the splits, causal context, label-column exclusion, incident aggregation and metrics
(no data needed):

```bash
python -m pytest tests/        # or: python tests/test_protocols.py
```

Smoke test on synthetic Precinct6-like data (numbers are meaningless; delete `data/` afterwards):

```bash
python tests/make_mock.py && python 03_leakage_audit.py && python 07_repeat_seeds.py --data small --seeds 0,1 --n-boot 50 --n-cluster-boot 50
```

## Reproducing the paper

Run the steps in order. Times are for free Colab.

### Precinct6

| Step | Command | Paper | Time |
|---|---|---|---|
| 1 | `python 01_download.py` | small version (Sec. 4.2, 4.3) | minutes |
| 2 | `python 02_eda.py` | label tables of the small version, Fig. 1 (subset bars) | minutes |
| 3 | `python 05_extract_100m.py` (resumable, streams 58 shards) | Sec. 3.3 → `data/100m/rulehits.parquet`, `minute_counts.parquet` | 1–2 h |
| 3b | `python 05b_summarize_100m.py` | Table 1, Fig. 1 (full-capture bars) → `results/extract/` | 1 min |
| 4 | `python 03_leakage_audit.py` and `python 03_leakage_audit.py --data 100m --sample 0` | Sec. 4.1–4.3 (label-derived columns, log-source shortcut, density proxy) | minutes |
| 5 | `python 06_diagnose_temporal.py --data 100m` | Sec. 5.1, Fig. 3 (why one global time cut fails) | minutes |
| 5b | `python extras/first_featurization/first_featurization_check.py` and `python 04_baselines.py --data 100m --only naive_global_temporal,incident_grouped` | Sec. 5.1 (first featurization vs corrected features) | 15 min |
| 6 | `python 07_repeat_seeds.py --data 100m --tag cluster` | Tables 4–6, Fig. 4, Sec. 7.1–7.4 (seven baselines and the unfiltered Isolation Forest, 5 seeds; event, incident and hour-block bootstrap) | 2–3 h |
| 7 | `python 08_ctxgraph.py --only asa_temporal,vpc_wave --seeds 0,1,2,3,4 --tag full_v2` and `python 08_ctxgraph.py --only asa_to_vpc,vpc_to_asa --seeds 0,1,2,3,4 --uda --tag cross_uda_v2`; ablations with `--no-memory`, `--no-anomaly`, `--text none` (give each a `--tag`) | Table 7, Sec. 7.5 (GPU) | 1–2 h |
| 8 | `python 09_compare_vs_baselines.py --ctx-tag full_v2 --only asa_temporal,vpc_wave` (and `--ctx-tag cross_uda_v2 --only asa_to_vpc,vpc_to_asa`) | Sec. 7.5 paired tests | < 1 h |
| 9 | `python 10_noncausal_context.py --data 100m --tag cluster` | Table 8, Sec. 7.6 (causal vs two-sided context) | 1–2 h |

### GUIDE

Download once (a Kaggle API key in `~/.kaggle/kaggle.json` may be needed):

```bash
kaggle datasets download -d Microsoft/microsoft-security-incident-prediction -p guide --unzip
```

| Step | Command | Paper | Time |
|---|---|---|---|
| 10 | `python 11_guide_inspect.py --dir guide` | Table 2, Sec. 3.4 (schema, grades, what the released split shares) | 10 min |
| 11 | `python 12_guide_audit.py --dir guide` | Table 3, Sec. 4.4 (single-column audit); one-seed random-row reference in Sec. 7.7 | 2 h |
| 12 | `python 13_guide_robust.py --dir guide` | Fig. 2 (organization label purity, section D) | 4–5 h (section D alone takes minutes) |
| 13 | `python 14_guide_id_check.py --dir guide` | Sec. 4.4 (identifiers shared across organizations) | 10 min |
| 14 | `python 15_guide_v2.py --dir guide` | Table 9, Fig. 5, Sec. 5.3 and 7.7 (released, temporal-by-incident-start and organization-held-out protocols; incident- and organization-level bootstrap) | 5 h |
| 15 | `python 16_guide_final.py --dir guide` | Sec. 4.4 (alerts shared between incidents), Table 9 strict-temporal rows, Sec. 7.7 (IncidentId ablation) | 2.5 h |
| 16 | `python 17_guide_nrows_sensitivity.py --dir guide` | Sensitivity of Table 9 to `n_rows_alert`, which `12_guide_audit.add_derived` counts on the training and test files together (pooled vs within-split vs dropped) | 1–2 h |

### Figures

```bash
cd paper && python make_figures.py      # writes paper/fig/fig1..fig5
```

The figure script reads every number it plots from `results/` (and, for Fig. 4, from the SUMMARY
block of `logs/precinct6_cluster_output.txt`); the source of each figure is listed at its top.

## Superseded runs (kept for transparency)

* The temporal protocol of `12_guide_audit.py` and `13_guide_robust.py` cuts evidence **rows** at
  10 June 2024, which places 13.5% of test rows in incidents that also have training rows. The paper
  uses the incident-based splits of `15_guide_v2.py` and `16_guide_final.py` instead.
* Logistic regression is scored with `decision_function`, not `predict_proba`, because the logistic
  function saturates and creates ties (Sec. 6.2). Results produced before this change are not used.
* `07_repeat_seeds.py --models ...` runs a subset of models; the paper's Table 5 needs the default (all models).

## Results

`results/` holds the CSV outputs of the runs behind the paper (score arrays and large parquet files are
omitted); `results/README.md` maps each folder to the tables and sections it supports.

## Logs

`logs/` holds the full console output of the runs behind the paper's GUIDE results and the
Precinct6 robustness checks (see `logs/README.md`). Every number in the paper's GUIDE tables and in
Tables 6 and 8 can be read from them.

## Known limitation

`12_guide_audit.add_derived` counts the evidence rows of each alert (`n_rows_alert`) on the training and
test files together, so alerts attached to incidents in both files get a count that crosses the split.
`17_guide_nrows_sensitivity.py` re-fits the content model with the count computed within each side:
incident-level ROC-AUC under the released split falls from 0.819 to 0.806 (PR-AUC 0.650 to 0.627), the
temporal protocol is unchanged (0.760), and organization hold-out is unaffected by construction. The
published numbers keep the original feature so that the code reproduces them; see Sec. 9 of the paper.

## Rules built into the code

* Label-derived columns (`config.LEAKY_COLS`) are never features.
* Precinct6 context features count past events only (causal); `10_noncausal_context.py` is the only
  place where two-sided context is computed, to measure its effect.
* Featurizers, vocabularies and thresholds are fitted on training data only; thresholds use a
  validation slice that contains positives.
* Metrics: PR-AUC with its base rate, ROC-AUC, recall at 1% false-positive rate. No accuracy.
* Paired comparisons use seed-averaged ranks and bootstrap resampling of events, incidents, one-hour
  blocks (Precinct6) or organizations (GUIDE). p-values are Monte Carlo estimates k/B; `0/B` means no
  resample had the opposite sign.

## Citation

See `CITATION.cff`. Please cite the paper if you use this code.

## License

MIT (see `LICENSE`). The datasets keep their own licences.
