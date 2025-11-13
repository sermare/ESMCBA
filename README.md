# ESM-Cambrian Binding Affinity (ESMCBA)

This repository bundles code, data, notebooks, and trained models for exploring peptide–MHC (pMHC) binding with **ESM Cambrian** protein language models and for evaluating structure‑guided designs produced with **RFdiffusion**.

**Code**: https://github.com/sermare/ESMCBA  
**Models**: https://huggingface.co/smares/ESMCBA

---

## Quick facts

| Item | Details |
|------|---------|
| Main package | `ESMCBA/` (Python 3.10 modules and utilities) |
| Core tasks | • Generate ESM embeddings<br>• Fine‑tune / evaluate binding‑affinity (BA) regressors and classifiers<br>• Compare to external predictors (MHCFlurry, HLAthena, MixMHCpred, MHCnuggets)<br>• Visualise embeddings (UMAP)<br>• Analyse RFdiffusion pMHC designs & contact maps |
| Key data sources | IEDB IC₅₀ tables, HLA sequences, Apollo test sets, RFdiffusion outputs |
| Model checkpoints | Available on Hugging Face: `smares/ESMCBA` |
| Figures | Publication‑ready PDFs under `figures/` and `figures_manuscript/` |
| Environment | Conda env **ESM_cambrian** (Python 3.10, PyTorch 2.6, transformers 4.46, esm 3.1.3) |

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
## Quick Start

You can access this notebook to run with google collab:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1R_9aUDhxm9YDdno_Ykg8KJECd3ygQbHl?usp=sharing)



## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/sermare/ESMCBA
cd ESMCBA
```

### Step 2: Create and activate the conda environment

```bash
# Create environment
conda create -n ESM_cambrian python=3.10 -y
conda activate ESM_cambrian
```

### Step 3: Install required packages

```bash
# Install core PyTorch and Transformers ecosystem
pip install torch
pip install transformers
pip install esm

# Install Hugging Face Hub utilities
pip install "huggingface-hub<1.0"

# Optional: Install hf_transfer for faster large file downloads
pip install hf_transfer

pip install biopython umap-learn scikit-learn seaborn pandas matplotlib

```

**Note**: The `esm` and `umap-learn` packages are essential for running the embeddings generation and visualization scripts.

*(Install predictors like `mhcflurry` separately if you intend to rerun benchmarking notebooks.)*

---

## Download Model Checkpoints

All trained model checkpoints are hosted on Hugging Face: **https://huggingface.co/smares/ESMCBA**

### Available checkpoints (examples):

- `ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_25_0.001_5e-05_AUG_3_HLAB1402_2_1e-05_1e-06__1_B1402_0404_Hubber_B1402_final.pth`
- `ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_25_0.001_5e-05_AUG_6_HLAB1503_2_0.0001_1e-05__2_B1503_0404_Hubber_B1503_final.pth`
- `ESMCBA_epitope_0.5_20_ESMMASK_epitope_FT_15_0.0001_1e-05_AUG_6_HLAB5101_5_0.001_1e-06__3_B5101_Hubber_B5101_final.pth`

Browse all files: https://huggingface.co/smares/ESMCBA

### Download options:

**Option A: Download all checkpoints to a local folder**

```bash
# Download everything to ./models
hf download smares/ESMCBA --repo-type model --local-dir ./models
```

**Option B: Download a specific checkpoint**

#or just get one model
huggingface-cli download smares/ESMCBA \
  "ESMCBA_epitope_0.8_30_ESMMASK_epitope_FT_5_0.001_1e-06_AUG_6_HLAA0201_2_0.001_1e-06__2_A0201_Hubber_A0201_final.pth" \
  --repo-type model \
  --local-dir ./models


**Option C: Use Hugging Face cache (automatic)**

If you omit `--local-dir`, files will be downloaded to your HF cache (e.g., `~/.cache/huggingface/hub/`).

To change the cache location:
```bash
export HF_HOME=/path/to/cache
```

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

## To Run Predictions (run embeddings.py)

The `embeddings_generation.py` script generates ESM predictions and the embeddings for peptide sequences.

### Example 1: Using a downloaded checkpoint

```bash
cd ESMCBA/ESMCBA

python3 embeddings_generation.py \
  --model_path ./models/ESMCBA_epitope_0.5_20_ESMMASK_epitope_FT_15_0.0001_1e-05_AUG_6_HLAB5101_5_0.001_1e-06__3_B5101_Hubber_B5101_final.pth \
  --name B5101-ESMCBA \
  --hla B5101 \
  --encoding epitope \
  --output_dir ./outputs \
  --peptides ASCQQQRAGHS ASCQQQRAGH ASCQQQRAG DVRLSAHHHR DVRLSAHHHRM GHSDVRLSAHH
```

### Example 2: Auto-download from Hugging Face

If the script supports Hugging Face paths, you can specify just the filename or an `hf://` path:

```bash
python3 embeddings_generation.py \
  --model_path "ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_25_0.001_5e-05_AUG_3_HLAB1402_2_1e-05_1e-06__1_B1402_0404_Hubber_B1402_final.pth" \
  --name B1402-ESMCBA \
  --hla B1402 \
  --encoding epitope \
  --output_dir ./outputs \
  --peptides ASCQQQRAGHS ASCQQQRAGH ASCQQQRAG DVRLSAHHHR DVRLSAHHHRM GHSDVRLSAHH
```

or with explicit `hf://` prefix:

```bash
python3 embeddings_generation.py \
  --model_path "hf://smares/ESMCBA/ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_25_0.001_5e-05_AUG_3_HLAB1402_2_1e-05_1e-06__1_B1402_0404_Hubber_B1402_final.pth" \
  --name B1402-ESMCBA \
  --hla B1402 \
  --encoding epitope \
  --output_dir ./outputs \
  --peptides ASCQQQRAGHS ASCQQQRAGH ASCQQQRAG DVRLSAHHHR DVRLSAHHHRM GHSDVRLSAHH
```

### GPU vs CPU

- By default, PyTorch will use GPU if available
- To force CPU: `export CUDA_VISIBLE_DEVICES=""`

---

## Troubleshooting

### Model downloads

- **"huggingface-cli download is deprecated"**: Use `hf download` instead
- **Permission errors**: Public models don't require login. For private models: `hf login`
- **Slow transfers**: Install `hf_transfer` and export `HF_HUB_ENABLE_HF_TRANSFER=1`
- **File not found**: Double-check the exact filename on the Hub (filenames are long—copy and paste)

### Import errors

- **"No module named 'esm'"**: Make sure you ran `pip install esm==3.1.3`
- **"No module named 'umap'"**: Install via `pip install umap-learn==0.5.7`

---

## Reproducibility tips

Record the exact commit of the code and the model snapshot for papers and reviews:

```
Code commit: <git SHA from ESMCBA repo>
Model snapshot: <commit SHA from HF snapshots path>
HLA: B5101
Encoding: epitope
```

---

## Citing

> S. Mares (2025). Continued domain-specific pre-training of protein language models for pMHC-I binding prediction.  
> [DOI / preprint.](https://arxiv.org/abs/2507.13077v1)

---

## Maintenance checklist

* Remove `__pycache__/` and large binaries from Git; ignore via `.gitignore` or track via Git‑LFS
* Consolidate duplicate CSVs in `performances/`
* Standardise file names with stray colon or non‑ASCII characters (e.g. `input_B_15:01_output.csv`)

---

## License

Follow the license in the GitHub repo for code and the model card in the Hugging Face repo for model weights.
