# ESM-Cambrian Binding Affinity Analysis (ESMCBA)

This repository bundles code, data, notebooks, and trained models for exploring peptide–MHC (pMHC) binding with **ESM Cambrian** protein language models and for evaluating structure‑guided designs produced with **RFdiffusion**.

---

## Quick facts

| Item | Details |
|------|---------|
| Main package | `ESMCBA/` (Python 3.10 modules and utilities) |
| Core tasks | • Generate ESM embeddings<br>• Fine‑tune / evaluate binding‑affinity (BA) regressors and classifiers<br>• Compare to external predictors (MHCFlurry, HLAthena, MixMHCpred, MHCnuggets)<br>• Visualise embeddings (UMAP)<br>• Analyse RFdiffusion pMHC designs & contact maps |
| Key data sources | IEDB IC₅₀ tables, HLA sequences, Apollo test sets, RFdiffusion outputs |
| Figures | Publication‑ready PDFs under `figures/` and `figures_manuscript/` |
| Environment | Conda env **ESM_cambrian** (Python 3.10, PyTorch 2.6, transformers 4.46, esm 3.1.3) |

---

## Directory outline
```
ESMCBA/                   # importable package: modelling & utilities
│
├─ models/
│   ├─ ESM_Supervised/    # model definitions + checkpoints
│   └─ ESM_Unsupervised/
│
data/                     # CSV/TSV inputs and intermediate results
│   ├─ Amino_Acid_Properties.csv
│   ├─ IEDB_full_subset_filtered_out_MHCFlurry.csv
│   └─ ... (predictions_*.tsv, evaluation_*.csv, etc.)
│
figures/                  # exploratory plots (logos, ROC curves, etc.)
figures_manuscript/       # final manuscript figures
performances/             # aggregated model‑metric CSVs
jupyter_notebooks/        # reproducible analysis notebooks
└─ (GIFs, RFdiffusion outputs, misc.)
```

---

## Installation

```bash
git clone <repo>
cd <repo>

# reproduce environment (shown in `conda list`)
conda create -n ESM_cambrian python=3.10 -y
conda activate ESM_cambrian

pip install torch==2.6.0 transformers==4.46.3 esm==3.1.3 \
            biopython==1.85 umap-learn==0.5.7 scikit-learn==1.6.1 \
            seaborn==0.13.2 pandas==2.2.3 matplotlib==3.10.1
```

*(Install predictors like `mhcflurry` separately if you intend to rerun benchmarking notebooks.)*

---

## Typical workflow

| Step | Script / notebook | Output |
|------|-------------------|--------|
| 1 | `embeddings_generation.py` | Embedding files in `data/` |
| 2 | `make_ESMCBA_models.py` (supervised) or `forward_pass_unsupervised.py` | Checkpoints in `models/` |
| 3 | `evaluation_IEDB_qual.py` | Metric CSVs + ROC/AUC PDFs |
| 4 | `HLA_full_sequences_UMAP.py` | UMAP plots in `figures/` |
| 5 | Notebooks under `jupyter_notebooks/rdfiffusion/` | Contact maps, hit‑rate tables |

Run any script with `-h` to see its arguments.

---

## Citing

> S. Mares (2025). Continued domain-specific pre-training of protein language models for pMHC-I binding prediction.  
> [DOI / preprint.](https://arxiv.org/abs/2507.13077v1)

---

## Maintenance checklist

* Remove `__pycache__/` and large binaries from Git; ignore via `.gitignore` or track via Git‑LFS.  
* Consolidate duplicate CSVs in `performances/`.  
* Standardise file names with stray colon or non‑ASCII characters (e.g. `input_B_15:01_output.csv`).  
