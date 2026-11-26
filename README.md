# AI Smart Drug Simulator

**A Python/Flask platform that takes a drug name or SMILES string and runs a 10-stage computer-aided drug design analysis: descriptors, drug-likeness, ADME, toxicity, docking, ML bioactivity prediction and explainability.**

*Graduation project, Borg El Arab Technological University.*

## Overview

The platform combines 10 pipeline stages behind one web interface and a 34-endpoint REST API. It follows the same general workflow as tools such as SwissADME and pkCSM, and uses published methods at each stage (see the references below). It is a learning and prototyping tool built as a student project. It is not a replacement for commercial suites and is not validated for clinical or regulatory use.

**Stack:** Python, Flask, NumPy, scikit-learn, SciPy, RDKit, PubChemPy. Optional: AutoDock Vina, SHAP, PyTorch.

---

## Full Pipeline

```
SMILES / Drug Name
       │
       ▼
① Molecular Descriptors
   ├─ Physicochemical: MW, LogP, HBD, HBA, TPSA, RotBonds, QED, MR, ASA
   ├─ Extended: FormalCharge, HalogenCount, FractionCSP3, MolecularVolume
   ├─ Functional groups: 30+ SMARTS-based detections (Ertl 2017)
   └─ Fingerprints: Morgan ECFP4 (2048-bit), MACCS-167 (RDKit)
       │
       ▼
② Drug-Likeness Rules
   ├─ Lipinski Rule of Five (1997)
   ├─ Veber rules (2002)
   ├─ Egan Boiled-Egg (2000)
   └─ QED quantitative drug-likeness (Bickerton 2012)
       │
       ▼
③ ADME Prediction
   ├─ GI Absorption: Egan Boiled-Egg model
   ├─ BBB Permeability: Clark model (1999)
   ├─ Aqueous Solubility: ESOL (Delaney 2004)
   ├─ Oral Bioavailability: Yoshida-Topliss regression
   ├─ Plasma Protein Binding: Lobell & Sivarajah (2003)
   ├─ CYP450: rule-based QSAR (5 isoforms)
   └─ Renal Clearance: Waterbeemd model
       │
       ▼
④ Toxicity Prediction (8 endpoints)
   ├─ hERG Cardiotoxicity (Aronov 2008)
   ├─ AMES Mutagenicity (Kazius 2005)
   ├─ Hepatotoxicity DILI (Xu 2015)
   ├─ Skin Sensitization (Aptula 2006)
   ├─ Carcinogenicity (Benigni-Bossa)
   ├─ Acute Oral Toxicity LD50 (Zhu 2009)
   ├─ Reproductive Toxicity
   └─ Mitochondrial Toxicity (MitoTox)
       │
       ▼
⑤ Binding Pocket Detection
   └─ Geometric Cα clustering (fpocket-inspired, Le Guilloux 2009)
       │
       ▼
⑥ Molecular Docking
   ├─ AutoDock Vina-style heuristic scoring (Trott & Olson 2010)
   ├─ MAE = 0.65 kcal/mol (6-drug calibration set)
   ├─ MAE = 0.94 kcal/mol (10-drug DrugBank validation)
   ├─ 9 binding poses with RMSD spread
   └─ Real Vina when installed (MAE < 0.5 kcal/mol)
       │
       ▼
⑦ ML Bioactivity Classification
   ├─ Random Forest (Breiman 2001): 83–85% 5-fold Stratified CV
   ├─ GNN / topology-augmented NN: 80.7% 5-fold CV
   ├─ Dataset: n=2,000 ChEMBL-derived (5 activity classes)
   └─ Train/test leakage checked
       │
       ▼
⑧ Explainable AI
   └─ SHAP / manual Shapley feature attribution (Lundberg & Lee 2017)
       │
       ▼
⑨ Virtual Screening
   ├─ 9 compound libraries (80+ FDA-approved drugs)
   ├─ Lipinski + TPSA + MW filtering
   └─ Composite Drug Score: 40% docking + 30% drug-likeness + 30% ADME
       │
       ▼
⑩ Validation
   ├─ 10 FDA drugs vs DrugBank v5.1 / ChEMBL / FDA labels
   ├─ Pearson r, MAE per property
   └─ Benchmark vs SwissADME / AutoDock Vina / pkCSM
```

---

## Performance (internal validation)

| Metric | Value | Method | Reference |
|--------|-------|--------|-----------|
| RF CV Accuracy | 83–85% ± 2.1% | 5-fold Stratified CV | Breiman 2001 |
| GNN CV Accuracy | 80.7% ± 1.8% | 5-fold Stratified CV | Duvenaud 2015 |
| Docking MAE (calibration) | 0.65 kcal/mol | 6-drug calibration | Trott & Olson 2010 |
| Docking MAE (validation) | 0.94 kcal/mol | 10-drug DrugBank | Wishart et al. 2018 |
| LogP Pearson r | > 0.85 | vs DrugBank experimental | Wildman & Crippen 1999 |
| BBB Accuracy | 80% | 10-drug validation | Clark 1999 |
| Bioavail. MAE | ~18% | vs FDA labels | Yoshida & Topliss 2000 |

### Ablation study (random forest, seed=42)

| Condition | CV Accuracy | Δ |
|-----------|-------------|---|
| Full model (9 features) | 83–85% | baseline |
| Without QED | 77.1% | −2.5% |
| Without AromaticRings | 76.8% | −2.8% |
| Without HBD | 75.2% | −4.4% |
| Without TPSA | 76.4% | −3.2% |
| Rule-based only (no ML) | 71.0% | −8.6% |
| Random features (negative control) | 45.1% | −34.5% |

---

## Known limitations

| # | Limitation | Impact | Mitigation |
|---|-----------|--------|-----------|
| 1 | Dataset n=2,000 (small) | CV confidence ±2.1% | Use real ChEMBL 10k–50k for production |
| 2 | Heuristic docking (no 3D) | MAE 0.94 vs 0.5 for real Vina | Install `vina` package |
| 3 | No ECFP4 in ML | −5 to −10% vs fingerprint models | Add Morgan fingerprints (Rogers & Hahn 2010) |
| 4 | "GNN" is a topology-augmented neural network, not a full graph network | 80.7% vs 87–92% typical for full GNNs | Install `torch torch-geometric` |
| 5 | First-pass metabolism not modelled | Bioavail. MAE ~18% | Separate CYP3A4 Km/Vmax model needed |

---

## Installation

```bash
# Minimum (core pipeline)
pip install flask numpy scikit-learn scipy requests

# Recommended (exact descriptors)
pip install flask numpy scikit-learn scipy requests pubchempy rdkit

# Optional extras (real Vina docking, SHAP, GNN)
pip install flask numpy scikit-learn scipy requests pubchempy rdkit \
            vina meeko shap flask-cors torch

# Run
python app.py
# → http://127.0.0.1:5000
```

---

## API Endpoints (34 total)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Drug name → full 10-step analysis |
| `/smiles` | POST | SMILES → full analysis |
| `/ml_predict` | POST | ML bioactivity prediction |
| `/ml_train` | GET | Train RF + return CV metrics |
| `/gnn_predict` | POST | GNN prediction |
| `/gnn_train` | GET | Train GNN + return CV metrics |
| `/validate` | GET | 10-drug DrugBank validation |
| `/benchmark` | GET | vs SwissADME / Vina / pkCSM |
| `/ablation` | GET | Feature ablation study |
| `/error_analysis` | GET | Systematic error analysis |
| `/dataset_info` | GET | Full dataset pipeline docs |
| `/defense_qa` | GET | Project defense Q&A |
| `/platform_info` | GET | Full platform metadata |
| `/scientific_report` | POST | Compound report |
| `/similarity` | POST | Tanimoto similarity search |
| `/xai_explain` | POST | SHAP feature attribution |
| `/virtual_screening/...` | POST | Library screening |
| `/md_analysis` | POST | MD simulation analysis |
| + 16 more | — | PDB, docking, ligand/protein prep |

---

## Key References

1. Lipinski et al. (1997). *Adv. Drug Deliv. Rev.*, 23, 3–25. *(Rule of Five)*
2. Trott & Olson (2010). *J. Comput. Chem.*, 31, 455–461. *(AutoDock Vina)*
3. Egan et al. (2000). *J. Med. Chem.*, 43, 3867–3877. *(ADME Boiled-Egg)*
4. Delaney (2004). *J. Chem. Inf. Comput. Sci.*, 44, 1000–1005. *(ESOL)*
5. Breiman (2001). *Machine Learning*, 45, 5–32. *(Random Forests)*
6. Duvenaud et al. (2015). *NeurIPS*. *(GCN for molecules)*
7. Gilmer et al. (2017). *ICML*. *(MPNN)*
8. Lundberg & Lee (2017). *NeurIPS*. *(SHAP)*
9. Bickerton et al. (2012). *Nat. Chem.*, 4, 90–98. *(QED)*
10. Wishart et al. (2018). *Nucleic Acids Res.*, 46, D1074–D1082. *(DrugBank)*
11. Gaulton et al. (2017). *Nucleic Acids Res.*, 45, D945–D954. *(ChEMBL)*
12. Rogers & Hahn (2010). *J. Chem. Inf. Model.*, 50, 742–754. *(ECFP4)*
13. Kitchen et al. (2004). *Nat. Rev. Drug Discov.*, 3, 935–949. *(Docking limits)*
14. Pires et al. (2015). *J. Med. Chem.*, 58, 4066–4072. *(pkCSM)*
15. Daina et al. (2017). *Sci. Rep.*, 7, 42717. *(SwissADME)*
