# ESM-Cambrian Binding Affinity (ESMCBA)

This repository bundles code, data, notebooks, and trained models for exploring peptide–MHC (pMHC) binding with **ESM Cambrian** protein language models.

![Graph](https://github.com/sermare/ESMCBA/blob/main/figures_manuscript/graph_abstract2.png)

**Code**: https://github.com/sermare/ESMCBA  
**Models**: https://huggingface.co/smares/ESMCBA

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1R_9aUDhxm9YDdno_Ykg8KJECd3ygQbHl?usp=sharing)


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

![SOTA](https://github.com/sermare/ESMCBA/blob/main/figures_manuscript/4a_SOTA_models-4.png)

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

## Model paths

| HLA   | Model checkpoint |
|-------|------------------|
| B5101 | `ESMCBA_epitope_0.5_20_ESMMASK_epitope_FT_15_0.0001_1e-05_AUG_6_HLAB5101_5_0.001_1e-06__3_B5101_Hubber_B5101_final.pth` |
| A0206 | `ESMCBA_epitope_0.5_20_ESMMASK_epitope_FT_25_0.0001_1e-06_AUG_1_HLAA0206_2_0.001_1e-06__1_A0206_Hubber_A0206_final.pth` |
| B3701 | `ESMCBA_epitope_0.5_20_ESMMASK_epitope_FT_25_0.001_1e-06_AUG_3_HLAB3701_1_0.0001_1e-05__1_B3701_0404_Hubber_B3701_final.pth` |
| B5301 | `ESMCBA_epitope_0.5_30_ESMMASK_epitope_FT_15_0.001_1e-06_AUG_6_HLAB5301_1_0.0001_1e-05__1_B5301_0404_Hubber_B5301_final.pth` |
| A2402 | `ESMCBA_epitope_0.5_30_ESMMASK_epitope_FT_20_0.001_1e-06_AUG_1_HLAA2402_1_0.0001_1e-06__2_A2402_0404_Hubber_A2402_final.pth` |
| C0802 | `ESMCBA_epitope_0.5_30_ESMMASK_epitope_FT_20_0.001_5e-05_AUG_1_HLAC0802_2_0.0001_1e-05__2_C0802_0404_Hubber_C0802_final.pth` |
| A0301 | `ESMCBA_epitope_0.5_30_ESMMASK_epitope_FT_25_0.001_0.001_AUG_1_HLAA0301_1_0.001_1e-06__1_A0301_Hubber_A0301_final.pth` |
| B3501 | `ESMCBA_epitope_0.5_30_ESMMASK_epitope_FT_25_0.001_1e-06_AUG_6_HLAB3501_2_0.001_0.001__4_B3501_Hubber_B3501_final.pth` |
| C1502 | `ESMCBA_epitope_0.5_30_ESMMASK_epitope_FT_25_0.001_5e-05_AUG_3_HLAC1502_2_0.0001_1e-06__1_C1502_0404_Hubber_C1502_final.pth` |
| B4601 | `ESMCBA_epitope_0.8_20_ESMMASK_epitope_FT_15_0.001_1e-06_AUG_6_HLAB4601_1_0.0001_1e-05__2_B4601_0404_Hubber_B4601_final.pth` |
| C0501 | `ESMCBA_epitope_0.8_20_ESMMASK_epitope_FT_15_0.001_1e-06_AUG_6_HLAC0501_2_0.0001_1e-06__2_C0501_0404_Hubber_C0501_final.pth` |
| A3201 | `ESMCBA_epitope_0.8_20_ESMMASK_epitope_FT_15_0.001_5e-05_AUG_1_HLAA3201_2_0.0001_1e-06__1_A3201_0404_Hubber_A3201_final.pth` |
| A0205 | `ESMCBA_epitope_0.8_20_ESMMASK_epitope_FT_15_0.001_5e-05_AUG_3_HLAA0205_2_0.0001_1e-06__2_A0205_0404_Hubber_A0205_final.pth` |
| A3001 | `ESMCBA_epitope_0.8_20_ESMMASK_epitope_FT_25_0.0001_1e-06_AUG_3_HLAA3001_4_0.0001_0.001__3_A3001_Hubber_A3001_final.pth` |
| A0101 | `ESMCBA_epitope_0.8_20_ESMMASK_epitope_FT_25_0.001_1e-05_AUG_6_HLAA0101_2_0.001_0.001__3_A0101_Hubber_A0101_final.pth` |
| C1203 | `ESMCBA_epitope_0.8_20_ESMMASK_epitope_FT_25_0.001_1e-06_AUG_1_HLAC1203_1_0.0001_1e-05__2_C1203_0404_Hubber_C1203_final.pth` |
| A0207 | `ESMCBA_epitope_0.8_20_ESMMASK_epitope_FT_25_0.001_1e-06_AUG_3_HLAA0207_1_0.0001_1e-06__2_A0207_0404_Hubber_A0207_final.pth` |
| A0211 | `ESMCBA_epitope_0.8_20_ESMMASK_epitope_FT_25_0.001_1e-06_AUG_6_HLAA0211_2_0.0001_1e-06__1_A0211_0404_Hubber_A0211_final.pth` |
| B5801 | `ESMCBA_epitope_0.8_20_ESMMASK_epitope_FT_25_0.001_1e-06_AUG_6_HLAB5801_2_0.0001_1e-06__2_B5801_0404_Hubber_B5801_final.pth` |
| B0702 | `ESMCBA_epitope_0.8_30_ESMMASK_epitope_FT_15_0.0001_0.001_AUG_6_HLAB0702_3_0.001_1e-06__4_B0702_Hubber_B0702_final.pth` |
| C0701 | `ESMCBA_epitope_0.8_30_ESMMASK_epitope_FT_15_0.001_5e-05_AUG_1_HLAC0701_2_0.0001_1e-05__1_C0701_0404_Hubber_C0701_final.pth` |
| B3801 | `ESMCBA_epitope_0.8_30_ESMMASK_epitope_FT_20_0.001_1e-06_AUG_3_HLAB3801_2_0.0001_1e-06__1_B3801_0404_Hubber_B3801_final.pth` |
| C0303 | `ESMCBA_epitope_0.8_30_ESMMASK_epitope_FT_20_0.001_1e-06_AUG_3_HLAC0303_1_0.0001_1e-05__2_C0303_0404_Hubber_C0303_final.pth` |
| B4501 | `ESMCBA_epitope_0.8_30_ESMMASK_epitope_FT_25_0.001_1e-06_AUG_1_HLAB4501_2_0.0001_1e-05__2_B4501_0404_Hubber_B4501_final.pth` |
| B4001 | `ESMCBA_epitope_0.8_30_ESMMASK_epitope_FT_25_0.001_5e-05_AUG_6_HLAB4001_1_0.0001_1e-06__2_B4001_0404_Hubber_B4001_final.pth` |
| A0201 | `ESMCBA_epitope_0.8_30_ESMMASK_epitope_FT_5_0.001_1e-06_AUG_6_HLAA0201_2_0.001_1e-06__2_A0201_Hubber_A0201_final.pth` |
| C0602 | `ESMCBA_epitope_0.95_20_ESMMASK_epitope_FT_15_0.001_5e-05_AUG_1_HLAC0602_2_0.0001_1e-06__1_C0602_0404_Hubber_C0602_final.pth` |
| A2501 | `ESMCBA_epitope_0.95_20_ESMMASK_epitope_FT_20_0.001_1e-06_AUG_1_HLAA2501_1_0.0001_1e-06__1_A2501_0404_Hubber_A2501_final.pth` |
| B5401 | `ESMCBA_epitope_0.95_20_ESMMASK_epitope_FT_20_0.001_5e-05_AUG_1_HLAB5401_2_0.0001_1e-06__2_B5401_0404_Hubber_B5401_final.pth` |
| A1101 | `ESMCBA_epitope_0.95_20_ESMMASK_epitope_FT_25_0.0001_1e-05_AUG_3_HLAA1101_5_0.001_1e-06__2_A1101_Hubber_A1101_final.pth` |
| B1801 | `ESMCBA_epitope_0.95_20_ESMMASK_epitope_FT_25_0.0001_1e-05_AUG_6_HLAB1801_1_0.001_1e-06__4_B1801_Hubber_B1801_final.pth` |
| B1501 | `ESMCBA_epitope_0.95_20_ESMMASK_epitope_FT_25_0.001_0.001_AUG_3_HLAB1501_2_0.001_0.001__2_B1501_Hubber_B1501_final.pth` |
| A6801 | `ESMCBA_epitope_0.95_20_ESMMASK_epitope_FT_25_0.001_1e-05_AUG_1_HLAA6801_2_0.0001_1e-06__4_A6801_Hubber_A6801_final.pth` |
| B2705 | `ESMCBA_epitope_0.95_20_ESMMASK_epitope_FT_25_0.001_1e-06_AUG_3_HLAB2705_2_0.0001_1e-06__2_B2705_0404_Hubber_B2705_final.pth` |
| C0401 | `ESMCBA_epitope_0.95_20_ESMMASK_epitope_FT_25_0.001_1e-06_AUG_3_HLAC0401_2_0.0001_1e-06__1_C0401_0404_Hubber_C0401_final.pth` |
| B1502 | `ESMCBA_epitope_0.95_20_ESMMASK_epitope_FT_25_0.001_5e-05_AUG_3_HLAB1502_1_1e-05_1e-05__1_B1502_0404_Hubber_B1502_final.pth` |
| A0202 | `ESMCBA_epitope_0.95_20_ESMMASK_epitope_FT_25_0.001_5e-05_AUG_6_HLAA0202_1_0.0001_1e-05__2_A0202_0404_Hubber_A0202_final.pth` |
| A2601 | `ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_15_0.0001_1e-05_AUG_1_HLAA2601_5_0.001_0.001__4_A2601_Hubber_A2601_final.pth` |
| C0702 | `ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_15_0.001_5e-05_AUG_1_HLAC0702_1_0.0001_1e-05__1_C0702_0404_Hubber_C0702_final.pth` |
| A3301 | `ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_20_0.001_0.001_AUG_1_HLAA3301_5_0.001_1e-06__4_A3301_Hubber_A3301_final.pth` |
| B0801 | `ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_20_0.001_1e-06_AUG_1_HLAB0801_1_0.0001_1e-06__1_B0801_0404_Hubber_B0801_final.pth` |
| B1517 | `ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_20_0.001_5e-05_AUG_3_HLAB1517_1_0.0001_1e-05__2_B1517_0404_Hubber_B1517_final.pth` |
| A0203 | `ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_25_0.001_0.001_AUG_6_HLAA0203_2_0.001_0.001__2_A0203_Hubber_A0203_final.pth` |
| B5701 | `ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_25_0.001_1e-05_AUG_1_HLAB5701_2_0.0001_1e-05__1_B5701_Hubber_B5701_final.pth` |
| B4402 | `ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_25_0.001_1e-05_AUG_3_HLAB4402_1_0.001_0.001__2_B4402_Hubber_B4402_final.pth` |
| A6802 | `ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_25_0.001_1e-06_AUG_6_HLAA6802_2_0.001_1e-06__4_A6802_Hubber_A6802_final.pth` |
| B4403 | `ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_25_0.001_1e-06_AUG_3_HLAB4403_1_0.0001_1e-06__1_B4403_0404_Hubber_B4403_final.pth` |
| C1402 | `ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_25_0.001_1e-06_AUG_3_HLAC1402_1_0.0001_1e-06__1_C1402_0404_Hubber_C1402_final.pth` |
| B4002 | `ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_25_0.001_1e-06_AUG_6_HLAB4002_2_0.0001_1e-05__1_B4002_0404_Hubber_B4002_final.pth` |
| A3101 | `ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_25_0.001_5e-05_AUG_3_HLAA3101_2_0.0001_1e-06__2_A3101_0404_Hubber_A3101_final.pth` |
| B1402 | `ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_25_0.001_5e-05_AUG_3_HLAB1402_2_1e-05_1e-06__1_B1402_0404_Hubber_B1402_final.pth` |
| B1503 | `ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_25_0.001_5e-05_AUG_6_HLAB1503_2_0.0001_1e-05__2_B1503_0404_Hubber_B1503_final.pth` |


---

## Maintenance checklist

* Remove `__pycache__/` and large binaries from Git; ignore via `.gitignore` or track via Git‑LFS
* Consolidate duplicate CSVs in `performances/`
* Standardise file names with stray colon or non‑ASCII characters (e.g. `input_B_15:01_output.csv`)

---

## License

Follow the license in the GitHub repo for code and the model card in the Hugging Face repo for model weights.
