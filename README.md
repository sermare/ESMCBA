# 🚀 ESM Cambrian Binding Affinity Analysis

Welcome to **ESM Cambrian Binding Affinity Analysis**! This repo is my playground for blending data science 🧪, molecular biology 🧬, and ML 🤖 to study peptide–MHC binding.

## 📚 Table of Contents
- [About the Project](#about-the-project)
- [🚩 Goals & Features](#-goals--features)
- [📁 Repo Structure](#-repo-structure)
- [🔧 Installation & Requirements](#-installation--requirements)
- [💻 How to Use](#-how-to-use)
- [🛠️ Key Scripts & Modules](#️-key-scripts--modules)
- [📄 License & Citation](#-license--citation)
- [✅ TODO](#-todo)

---

## About the Project
I’m fine-tuning ESM models on IEDB IC₅₀ data and concatenated HLA+epitope sequences to predict binding affinities. I compare against MHCFlurry & NetMHCpan, visualize embeddings with UMAP, and map hotspots for RFdiffusion designs.

---

## 🚩 Goals & Features
- **Data Prep**: clean & merge IEDB + HLA sequences
- **Model Training**: supervised fine-tuning of ESM on IC₅₀ regression
- **Benchmarking**: side-by-side with MHCFlurry & NetMHCpan
- **Visuals**: scatter, violin, PPV, UMAP plots
- **Hotspot Mapping**: identify interface residues by contact count

---

## 📁 Repo Structure
```
.
├── data/             
├── jupyter notebooks/        
├── ESMCBA/          
│   ├── preprocessing.py    
│   ├── train_esmc.py     
│   ├── evaluate.py       
│   └── hotspot_mapping.py
│   ├── Models   
│       ├── Pre-training
│       ├── Supervised
├── performances/           
├── requirements.txt  
├── README.md         
└── LICENSE          
```

---

## 🔧 Installation & Requirements
```bash
git clone https://github.com/<you>/ESM-Cambrian-Analysis.git
cd ESM-Cambrian-Analysis

conda create -n esmcambrian python=3.9 -y
conda activate esmcambrian
pip install -r requirements.txt

# Optional: benchmarking
pip install mhcflurry
mhcflurry-downloads fetch
```

---

## 💻 How to Use
1. **Preprocess data**  
   ```bash
   python scripts/preprocessing.py \
     --input data/raw/IEDB.csv \
     --hla-sequences data/raw/HLA_sequences.fasta \
     --output data/processed/
   ```
2. **Train model**  
   ```bash
   python scripts/train_esmc.py \
     --config config/train.yaml \
     --output-dir models/
   ```
3. **Evaluate**  
   ```bash
   python scripts/evaluate.py \
     --predictions results/predictions.csv \
     --measured data/processed/IC50.csv \
     --out-dir results/figures/
   ```
4. **Explore embeddings**  
   ```bash
   jupyter notebook notebooks/umap_visualization.ipynb
   ```

---

## 🛠️ Key Scripts & Modules
| Script                 | What it does                                       |
|------------------------|----------------------------------------------------|
| preprocessing.py       | merges & formats IEDB + HLA seqs                   |
| train_esmc.py          | fine-tunes ESM on IC₅₀ regression                  |
| evaluate.py            | computes Spearman/Pearson, PPV & plots             |
| hotspot_mapping.py     | maps contact-based hotspots for RFdiffusion        |
| visualization.py       | reusable plotting funcs (scatter, violin, UMAP)    |

---

## 📄 License & Citation
MIT License.  
If you use this work, please cite:  
> Mares *et al.*, “ESM-Cambrian Binding Affinity Analysis”, 2025.

---

## ✅ TODO
- [Running] Generate the HLA and epitope models for each of the alleles
- [ ] Upload the models to hugging face
- [ ] Generate notebooks for each of the benchmarks to do
- [ ] Generate more streamlined for other structures for RFdiffusion
- [ ] Perhaps finetuned the model on allele wide epitopes, and then allele specific ones
- [ ] Investigate if the RFdiffusion forces a Methionine on the beginning of the sequence
