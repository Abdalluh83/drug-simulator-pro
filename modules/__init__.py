"""
AI Smart Drug Simulator — Research-Grade CADD Platform v3.1
===========================================================
Group 2 | Pharmaceutical Manufacturing Technology | Faculty of Applied Health Sciences, Borg Alarab Technological University

Modules (22 total):
  Core Pipeline:
    descriptors      — MW, LogP, HBD, HBA, TPSA, QED, ECFP4, MACCS (RDKit/fallback)
    adme             — GI absorption, BBB, LogS, PPB, Vd, CYP450 (Egan 2000, Clark 1999)
    toxicity         — hERG, AMES, DILI, Skin, Carcinogenicity, LD50, Repro, Mito
    docking          — AutoDock Vina-style heuristic (MAE 0.94 kcal/mol, 10-drug validation)
    ml_model         — Random Forest classifier (83–85% CV, 5-fold Stratified, n=2000)
    gnn_model        — Topology-augmented descriptor NN (80.7% CV, Duvenaud 2015 architecture)
    xai              — SHAP / manual Shapley feature attribution (Lundberg & Lee 2017)

  Drug Discovery:
    similarity       — Tanimoto (Morgan ECFP4 / descriptor Euclidean)
    virtual_screening — 9 compound libraries (kinase, NSAIDs, antibiotics, CNS, natural, ...)
    ligand_prep      — 3D conformers (RDKit MMFF94), protonation, PDBQT prep
    protein_prep     — PDB cleaning, water removal, binding pocket detection
    pocket_detection — Geometric pocket detection (fpocket-inspired, Le Guilloux 2009)

  Infrastructure:
    pdb_utils        — RCSB PDB fetching + caching
    vina_engine      — AutoDock Vina wrapper (Python bindings + binary fallback)
    md_simulation    — MD analysis (AMBER99SB-ILDN proxy, MM-PBSA estimation)

  Validation & Benchmarking:
    validation       — 10-drug DrugBank v5.1 / ChEMBL / FDA label validation
    benchmark        — vs SwissADME / AutoDock Vina / pkCSM + ML baselines

  Research Quality:
    ablation_study   — Feature contribution analysis (8 conditions, live recomputation)
    error_analysis   — 4 documented failure modes with scientific explanations
    dataset_pipeline — ChEMBL-derived dataset documentation, leakage audit
    defense_qa       — 10 committee Q&A with code evidence + literature refs

API: 36 endpoints | Frontend: 19 pages | Libraries: 11 compound libraries | Modules: 21
"""
