# CulSen-BN: An Uncertainty-Aware, Culture-Sensitive Contrastive Learning Framework for Robust Bengali Sentiment Analysis Across Music Seasons

## Project Overview
CulSen-BN is a research-grade NLP framework designed for Bengali sentiment analysis in low-resource, noisy-label, and culturally rich domain environments (specifically Coke Studio Bangla YouTube comments across Seasons 1–4).

## Core Principles
1. **Never fabricate experimental results, citations, or dataset numbers.** All statistics and results must derive from verifiable executions.
2. **Never treat machine-generated pseudo-labels as human ground truth.** Keep pseudo-labels, human annotations, and model predictions strictly separate.
3. **Prevent data leakage.** Comments from the same video or duplicate text must never appear across train/validation/test partitions.
4. **Reproducibility.** All random seeds, dataset revisions, split manifests, and configurations must be pinned and documented.
5. **Human verification is researcher-driven.** Human annotations must come exclusively from manual researcher review, never simulated.

## Environment & Commands
- **Python Environment**: Managed via `uv` in `.venv/` (Python 3.11).
- **Run audit**: `python scripts/audit_dataset.py --config configs/data.yaml`
- **Create splits**: `python scripts/create_splits.py --config configs/data.yaml`
- **Run SVM baseline**: `python scripts/train_svm.py --config configs/baseline_svm.yaml`
- **Run Transformer training (Colab / GPU)**: `python scripts/train_transformer.py --config configs/baseline_transformer.yaml`
- **Run tests**: `pytest tests/`

## Project Structure
- `configs/`: Versioned experiment parameters in YAML.
- `src/culsen_bn/`: Core modular library (data, models, training, evaluation, utils).
- `scripts/`: Executable CLI tools for audit, splitting, training, and evaluation.
- `notebooks/`: Structured Jupyter notebooks for exploration and Colab execution.
- `data/`: Raw, interim, processed splits, and annotation templates.
- `reports/`: Generated figures, tables, and error analyses.
- `paper/`: Research manuscript, references.bib, and results placeholders.
