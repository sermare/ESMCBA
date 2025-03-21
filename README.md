

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

---
## **Other**


