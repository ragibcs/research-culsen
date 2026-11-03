# CulSen-BN: Results Tracking and Placeholders

This document tracks completed vs pending experimental outcomes to ensure strict empirical fidelity and avoid fabricating claims.

## Experiment Status Matrix

| Experiment ID | Description | Primary Metric | Current Status | Output Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **E1** | Baseline Comparison (SVM vs BanglaBERT vs XLM-R) | Macro-F1 / Accuracy | SVM Completed (Macro-F1: 0.8092, Acc: 0.8736) / Transformers Pending GPU | `reports/tables/baseline_svm_metrics.json` |
| **E2** | Label Reliability (Pseudo-labels vs Human Verification) | Disagreement Rate / Cohen's Kappa | Ready for Researcher Labeling | `data/annotations/human_verified_test.csv` |
| **E3** | Uncertainty-Aware Training (Confidence-weighted loss) | Macro-F1 / Calibration (ECE) | Implementation Ready / Pending Training | `artifacts/culsen_B1/metrics.json` |
| **E4** | Supervised Contrastive Learning (SupCon) | Macro-F1 / Class Separation | Implementation Ready / Pending Training | `artifacts/culsen_B2/metrics.json` |
| **E5** | Cross-Season Domain Generalization (Protocol B) | Transfer Macro-F1 Gap | SVM Completed (Mean Macro-F1: 0.8169, Acc: 0.8657 across S1-S4) / CulSen-BN Pending GPU | `reports/tables/cross_season_svm_metrics.json` |
| **E6** | Linguistic Robustness (Bengali, Banglish, Emojis, Negation) | Subgroup Macro-F1 | SVM Subgroups Evaluated | `reports/tables/baseline_svm_subgroups.json` |
| **E7** | Controlled Synthetic Noise (5%, 10%, 20%, 30%) | Noise Decay Slope | Pipeline Configured | `reports/tables/synthetic_noise.json` |
| **E8** | Full Framework Ablation (B0 to B5) | Component Gains | Implementation Ready | `reports/tables/ablation_study.json` |
| **E9** | External Benchmark Generalization | Macro-F1 | Optional / Pending Evaluation | `reports/tables/external_eval.json` |
