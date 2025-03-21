

Welcome to the **ESM Cambrian Binding Affinity Analysis** repository! This project combines data science, molecular biology, and machine learning approaches to analyze peptide-MHC binding behavior using **ESM-based** language models. Below you will find an overview of the repository’s goals, structure, and usage instructions.

---

## **Table of Contents**

1. [Project Overview](#project-overview)  
2. [Main Features and Goals](#main-features-and-goals)  
3. [Repository Structure](#repository-structure)  
4. [Installation and Requirements](#installation-and-requirements)  
5. [Usage Guide](#usage-guide)  
6. [Key Functions and Scripts](#key-functions-and-scripts)  
7. [License and Citation](#license-and-citation)

---

## **Project Overview**

This codebase focuses on evaluating and visualizing binding affinity predictions for peptide-HLA complexes (or protein–peptide interactions) using a **fine-tuned ESM (Evolutionary Scale Modeling) Cambrian** model. The workflow integrates:

- **Data Preprocessing**: Merging predictions from [MHCFlurry](https://github.com/openvax/mhcflurry) and custom ESM-based predictions on various HLAs.  
- **Visualization**: Generating high-quality scatter plots, boxplots, KDE plots, and heatmaps.  
- **Statistical Analysis**: Performing significance testing (Mann-Whitney U tests, F-tests, Spearman/Pearson correlations, MSE, MAE, R²), including Bonferroni correction for multiple comparisons.  
- **Model Evaluation**: Comparing fine-tuned ESM Cambrian model performance with standard MHC prediction tools, measuring correlation, average precision, AUC, and more.

---

## **Main Features and Goals**

1. **Fine-Tuning ESM**  
   Extending the ESM Cambrian model with a custom MLM head for masked amino acid predictions.

2. **Peptide-MHC Binding Predictions**  
   - Comparing **ESM** predictions to **MHCFlurry** (traditional tool).  
   - Bootstrapping and generating confidence intervals for Spearman/Pearson correlation, MSE, MAE.

3. **Statistical Significance Testing**  
   - Mann–Whitney U, F-tests, Bonferroni correction.  
   - Generating significance brackets on boxplots for quick interpretability.

4. **Graphical Output**  
   - Multiple styles of charts: scatter + density, hexbin with marginal histograms, violin plots, heatmaps for correlations.  
   - Automatic labeling of metrics, significance text, color-coded density.

5. **HPC Job Handling**  
   - Example SLURM job scripts for launching training/evaluation jobs on HPC clusters.  
   - Organized logging, model checkpointing, data splitting.

6. **Utilities and Helper Functions**  
   - `bonferroni_correction`, `plot_predictions`, data splitting by train/validation/test, random sampling, etc.

---

## **Repository Structure**

Below is a suggested breakdown for how to organize the large Jupyter notebook code into smaller, maintainable Python modules. You can adapt these to your preference:

. ├── imports.py # Central place for all library imports ├── graph_utils.py # Plotting-related functions (e.g., plot_predictions) ├── other_utils.py # Data manipulation, correlation analysis, sampling, etc. ├── main.py # Main entry point tying everything together ├── README.md # This README file └── ...

### **1. `imports.py`**
Contains all import statements used throughout the codebase in a single location. 

### **1. `imports.py`**  
Contains all import statements used throughout the codebase in a single location.

---

### **2. `graph_utils.py`**  
Contains reusable plotting functions—most notably the `plot_predictions` function, which:  
- Performs correlation and regression metric calculations (Spearman, Pearson, MSE, MAE, R²).  
- Creates a 2×2 subplot layout with scatter, hexbin, residual, and KDE plots.  
- Annotates results with significance lines, identity line, and color-coded density.

---

### **3. `other_utils.py`**  
Houses miscellaneous helper functions, such as:  
- `split_dataset()`: Splitting a DataFrame into train/test subsets.  
- `compute_correlations()`: Calculating Spearman and Pearson correlations.  
- Statistical utilities for significance tests.

---

### **4. `main.py`**  
Example driver script that ties everything together. Responsible for:  
- Loading data (CSV, JSON, etc.).  
- Invoking `split_dataset()` or other utilities.  
- Calling `plot_predictions()` for train/test results.  
- Summarizing final model performance.  

You can create additional scripts for HPC job submission, advanced statistical workflows, or extensive hyperparameter sweeps.

---

## **Installation and Requirements**

**Clone the Repository**:
```bash
git clone https://github.com/<your_username>/ESM-Cambrian-Analysis.git
cd ESM-Cambrian-Analysis

### **Install Python Dependencies**  
*(Inside a conda environment or virtualenv)*:
```bash
conda create -n esmcambrian python=3.9 -y
conda activate esmcambrian
pip install -r requirements.txt

*(Optional) MHCFlurry Installation*
If you want to run MHCFlurry predictions side-by-side with ESM results:
```bash pip install mhcflurry
mhcflurry-downloads fetch