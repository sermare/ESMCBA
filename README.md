

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

Name | What it captures | Threshold (after ≤ 5 Å cutoff) | hotspot = "…", ready to copy
core_hotspot | Only the strongest binding hot‑spots—absolute core you almost never want to mutate. |  ≥ 4 distinct protein contacts | "A66,A70,A159"
high_contact | Core + secondary hubs. Good default when you want to preserve the bulk of the binding energy. |  ≥ 3 contacts | "A66,A70,A73,A77,A97,A147,A159"
medium_contact | Moderately important residues; nice to keep fixed if you want a very native‑like interface. |  = 2 contacts | "A7,A63,A99,A116,A143,A146,A155,A156"
low_contact | Peripheral/edge residues; usually safe to let RFdiffusion mutate for better packing/solubility. |  = 1 contact | "A5,A9,A33,A45,A59,A67,A69,A76,A80,A81,A84,A95,A114,A123,A124,A142,A152,A160,A163,A167,A171"
combo_high_low | Lock the high‑energy core while leaving medium residues flexible (high + low combined). | see above | "A5,A9,A33,A45,A59,A66,A67,A69,A70,A73,A76,A77,A80,A81,A84,A95,A97,A114,A123,A124,A142,A147,A152,A159"
all_contacts | Every chain‑A residue that has ≥ 1 heavy‑atom within 5 Å of chain C (36 total). Use to freeze the entire native interface. |  ≥ 1 contact | "A5,A7,A9,A33,A45,A59,A63,A66,A67,A69,A70,A73,A76,A77,A80,A81,A84,A95,A97,A99,A114,A116,A123,A124,A142,A143,A146,A147,A152,A155,A156,A159,A160,A163,A167,A171"
user_hotspot | Your original 9‑residue list. | (user‑defined) | "A7,A9,A59,A63,A66,A70,A99,A159,A167"

