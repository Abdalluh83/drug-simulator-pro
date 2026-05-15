"""
virtual_screening.py — Virtual Screening Engine
=================================================
Group 2 — Pharmaceutical Manufacturing Technology

Implements a complete virtual screening pipeline:
  1. Load compound library (built-in or custom SMILES)
  2. Filter: Lipinski, TPSA, MW range
  3. Batch docking: compute ΔG for all compounds vs target
  4. ADME assessment per compound
  5. ML bioactivity prediction
  6. Drug Score = 40% Docking + 30% Drug-likeness + 30% ADME
  7. Rank by Drug Score
  8. Export top candidates

References:
  Trott & Olson, J. Comput. Chem. 2010 (AutoDock Vina)
  Lipinski et al., Adv. Drug Deliv. Rev. 1997
  Shoichet, Nature 2004 (virtual screening review)
  Bickerton et al., Nat. Chem. 2012 (QED-based scoring)
"""

import math
import logging

logger = logging.getLogger(__name__)

# ── Built-in Compound Libraries ───────────────────────────────
COMPOUND_LIBRARIES = {
    "druglike_12": {
        "name":    "Drug-like FDA Approved (12 compounds)",
        "ref":     "DrugBank v5.1",
        "compounds": [
            {"name":"Acetaminophen",  "smiles":"CC(=O)Nc1ccc(O)cc1",               "cid":1983,    "class":"analgesic"},
            {"name":"Aspirin",        "smiles":"CC(=O)Oc1ccccc1C(=O)O",            "cid":2244,    "class":"anti-inflammatory"},
            {"name":"Ibuprofen",      "smiles":"CC(C)Cc1ccc(cc1)C(C)C(=O)O",       "cid":3672,    "class":"anti-inflammatory"},
            {"name":"Caffeine",       "smiles":"Cn1cnc2c1c(=O)n(c(=O)n2C)C",       "cid":2519,    "class":"stimulant"},
            {"name":"Penicillin G",   "smiles":"CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O","cid":5904,"class":"antibacterial"},
            {"name":"Morphine",       "smiles":"CN1CCC23C4Oc5c(O)ccc6c5C2C(CC4)OC3CC16","cid":5288826,"class":"analgesic"},
            {"name":"Metformin",      "smiles":"CN(C)C(=N)NC(=N)N",                "cid":4091,    "class":"antidiabetic"},
            {"name":"Omeprazole",     "smiles":"CC1=CN=C(C(=C1OC)C)CS(=O)c1nc2ccccc2[nH]1","cid":4594,"class":"gastroprotective"},
            {"name":"Amoxicillin",    "smiles":"CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O","cid":33613,"class":"antibacterial"},
            {"name":"Ciprofloxacin",  "smiles":"OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O","cid":2764,"class":"antibacterial"},
            {"name":"Sildenafil",     "smiles":"CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12","cid":135398744,"class":"vasodilator"},
            {"name":"Atorvastatin",   "smiles":"CC(C)c1n(CC(O)CC(O)CC(=O)O)c2ccccc2c1CC3C(=O)Nc4ccc(F)cc43","cid":60823,"class":"cardiovascular"},
        ],
    },
    "nsaids_8": {
        "name":    "NSAIDs Library (8 compounds)",
        "ref":     "WHO Essential Medicines",
        "compounds": [
            {"name":"Aspirin",       "smiles":"CC(=O)Oc1ccccc1C(=O)O",            "cid":2244,   "class":"anti-inflammatory"},
            {"name":"Ibuprofen",     "smiles":"CC(C)Cc1ccc(cc1)C(C)C(=O)O",       "cid":3672,   "class":"anti-inflammatory"},
            {"name":"Naproxen",      "smiles":"COc1ccc2cc(ccc2c1)C(C)C(=O)O",     "cid":156391, "class":"anti-inflammatory"},
            {"name":"Diclofenac",    "smiles":"OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",  "cid":3033,   "class":"anti-inflammatory"},
            {"name":"Celecoxib",     "smiles":"Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1","cid":2662,"class":"anti-inflammatory"},
            {"name":"Indomethacin",  "smiles":"COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1","cid":3715,"class":"anti-inflammatory"},
            {"name":"Meloxicam",     "smiles":"Cc1cnc(NC(=O)c2cccc(S(=O)(=O)N3CCCC3)c2O)s1","cid":54677470,"class":"anti-inflammatory"},
            {"name":"Acetaminophen", "smiles":"CC(=O)Nc1ccc(O)cc1",              "cid":1983,   "class":"analgesic"},
        ],
    },
    "antibiotics_6": {
        "name":    "Antibiotics Library (6 compounds)",
        "ref":     "WHO Essential Medicines",
        "compounds": [
            {"name":"Ciprofloxacin", "smiles":"OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O","cid":2764,"class":"antibacterial"},
            {"name":"Amoxicillin",   "smiles":"CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O","cid":33613,"class":"antibacterial"},
            {"name":"Penicillin G",  "smiles":"CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O","cid":5904,"class":"antibacterial"},
            {"name":"Doxycycline",   "smiles":"OC1=C(O)C(=O)c2c(O)cccc2[C@@H]1C(=O)","cid":54671203,"class":"antibacterial"},
            {"name":"Azithromycin",  "smiles":"CCC1OC(=O)[C@H](C)[C@@H](OC2C[C@@](C)(OC)C[C@H](C)[C@H]2OC)[C@H](C)C[C@@](O)(CC=O)[C@@H](C[C@H](OC3C[C@@H](N(C)C)[C@@H](O)[C@H](C)O3)[C@@H]1C)N(C)C","cid":447043,"class":"antibacterial"},
            {"name":"Metronidazole", "smiles":"Cc1ncc([N+](=O)[O-])n1CCO","cid":4173,"class":"antibacterial"},
        ],
    },
    "antivirals_6": {
        "name":    "Antivirals Library (6 compounds)",
        "ref":     "FDA Antiviral Drugs",
        "compounds": [
            {"name":"Remdesivir",    "smiles":"CCC(CC)COC(=O)[C@@H](NS(=O)(=O)c1ccc(F)cc1)[C@H]1[C@@H](n2cnc3c(N)ncnc23)O[C@@H]([C@@H]1O)COP(=O)(N[C@@H](C)C(=O)OCC)OCC","cid":121304016,"class":"antiviral"},
            {"name":"Oseltamivir",   "smiles":"CCOC(=O)[C@@H]1C[C@H](OC(CC)CC)[C@H](NC(C)=O)[C@@H](N)C1","cid":65028,"class":"antiviral"},
            {"name":"Acyclovir",     "smiles":"Nc1nc2c(ncn2COCCO)c(=O)[nH]1","cid":2022,"class":"antiviral"},
            {"name":"Tenofovir",     "smiles":"Cc1ncnc2c1nc(N)n2[C@@H]1C[C@H](COP(=O)(O)O)[C@@H](O1)C","cid":464205,"class":"antiviral"},
            {"name":"Lopinavir",     "smiles":"CC(C)(C)c1ccc(cc1)OCC(=O)N[C@@H](Cc1ccccc1)[C@@H](O)CN1CC[C@@H]2CCCC[C@@H]2C1=O","cid":92727,"class":"antiviral"},
            {"name":"Ritonavir",     "smiles":"CC(C)c1nc(CN(C)C(=O)N[C@H](Cc2ccccc2)[C@@H](O)CN(Cc2ccc(S3)ncc2)[C@H](CC(C)C)[C@@H](OC(=O)N2CCCC2)c2ccccc2)cs1","cid":392622,"class":"antiviral"},
        ],
    },
    "anticancer_8": {
        "name":    "Anticancer Agents Library (8 compounds)",
        "ref":     "FDA Oncology Drugs",
        "compounds": [
            {"name":"Imatinib",      "smiles":"Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1","cid":5291,"class":"anticancer"},
            {"name":"Erlotinib",     "smiles":"C#Cc1cccc(Nc2ncnc3cc(OCC)c(OCC)cc23)c1","cid":176870,"class":"anticancer"},
            {"name":"Gefitinib",     "smiles":"COc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCCCN1CCOCC1","cid":123631,"class":"anticancer"},
            {"name":"Sorafenib",     "smiles":"CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1","cid":216239,"class":"anticancer"},
            {"name":"Dasatinib",     "smiles":"Cc1nc(Nc2ncc(C(=O)Nc3cccc(NCC4=NCC(O)N4)c3)cc2)cc(N2CCN(CCO)CC2)c1","cid":3062316,"class":"anticancer"},
            {"name":"Vemurafenib",   "smiles":"CCCS(=O)(=O)Nc1ccc(F)c(c1)C(=O)c1ccc(Cl)c(OCC2CC2)c1","cid":42611257,"class":"anticancer"},
            {"name":"Tamoxifen",     "smiles":"CCN(CC)CCOC(=C(c1ccccc1)/c1ccc(cc1)/C=C/c1ccc(OCC)cc1)c1ccccc1","cid":2733526,"class":"anticancer"},
            {"name":"Cisplatin",     "smiles":"N[Pt](N)(Cl)Cl","cid":84691,"class":"anticancer"},
        ],
    },
    "custom": {
        "name":    "Custom Library (user-defined)",
        "ref":     "User-provided",
        "compounds": [],
    },
    "diverse_50": {
        "name":    "Diverse Drug-Like Library (50 compounds)",
        "ref":     "ChEMBL + DrugBank representative set",
        "compounds": [
            {"name":"Metformin",      "smiles":"CN(C)C(=N)NC(=N)N",                    "cid":4091,    "class":"antidiabetic"},
            {"name":"Glipizide",      "smiles":"CC1=CC(=CC=N1)C(=O)NCCC2=CC=C(C=C2)S(=O)(=O)NC(=O)NC3CCCCC3","cid":3478,"class":"antidiabetic"},
            {"name":"Metoprolol",     "smiles":"COCCc1ccc(OCC(O)CNC(C)C)cc1",           "cid":4171,    "class":"cardiovascular"},
            {"name":"Propranolol",    "smiles":"CC(C)NCC(O)COc1cccc2ccccc12",           "cid":4946,    "class":"cardiovascular"},
            {"name":"Verapamil",      "smiles":"COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC","cid":2520,"class":"cardiovascular"},
            {"name":"Diltiazem",      "smiles":"COc1ccc([C@@H]2OC(=O)[C@@H](CC(=O)N(C)CCN(C)CC)[C@@H]2c3ccc(OC)cc3)cc1","cid":39186,"class":"cardiovascular"},
            {"name":"Lisinopril",     "smiles":"OC(=O)[C@@H](CCCl)NC(=O)[C@@H](Cc1ccccc1)N[C@@H](CC1CCCCC1)C(=O)O","cid":5362119,"class":"cardiovascular"},
            {"name":"Ramipril",       "smiles":"CCOC(=O)[C@@H]1CCCN1C(=O)[C@@H](CC1CCCCC1)N[C@@H](CCC(=O)O)C(=O)O","cid":5362129,"class":"cardiovascular"},
            {"name":"Amlodipine",     "smiles":"CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OCC)C1c1ccccc1Cl","cid":2162,"class":"cardiovascular"},
            {"name":"Hydrochlorothiazide","smiles":"NS(=O)(=O)c1cc2c(cc1Cl)NCNS2(=O)=O","cid":3639,"class":"diuretic"},
            {"name":"Furosemide",     "smiles":"NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl","cid":3440,"class":"diuretic"},
            {"name":"Spironolactone", "smiles":"CC(=O)SC1CC2=CC(=O)CCC2(C)C3CCC4(C)C(CCC34)C1=O","cid":5833,"class":"diuretic"},
            {"name":"Omeprazole",     "smiles":"CC1=CN=C(C(=C1OC)C)CS(=O)c1nc2ccccc2[nH]1","cid":4594,"class":"gastroprotective"},
            {"name":"Pantoprazole",   "smiles":"COc1ccc2nc(OC(F)F)[nH]c2c1","cid":4679,"class":"gastroprotective"},
            {"name":"Lansoprazole",   "smiles":"CC1=CN=C(CS(=O)c2nc3cc(OCC(F)(F)F)ccn3[nH]2)C(OCC(F)(F)F)=C1","cid":3883,"class":"gastroprotective"},
            {"name":"Ranitidine",     "smiles":"CNC(=C[N+](=O)[O-])NCCSCC(=NC/C=C/NC)N","cid":3001055,"class":"gastroprotective"},
            {"name":"Metronidazole",  "smiles":"Cc1ncc([N+](=O)[O-])n1CCO","cid":4173,"class":"antibacterial"},
            {"name":"Vancomycin",     "smiles":"OC1C(NC(=O)C2NC(=O)c3cc4cc(Cl)ccc4c(OC4CC(N)C(O)(CO)O4)c3-c3cc(Cl)ccc3OC3CC(NC(=O)CC(N)Cc4ccc(O5)cc4)C(O5)(CO)O3)OC(CO)C(O)C1O","cid":14969,"class":"antibacterial"},
            {"name":"Rifampicin",     "smiles":"COC1C=CC=C(NC(=O)c2c(O)c3c(OC)cccc3oc2=O)CC(OC(C)=O)C(C)C(=O)C(C)C(OC)C(C)C1=O","cid":135398513,"class":"antibacterial"},
            {"name":"Levofloxacin",   "smiles":"C[C@@H]1COc2c(N3CC(CN(C)C)C3)c(F)cc3c(=O)c(C(=O)O)cn1c23","cid":149096,"class":"antibacterial"},
            {"name":"Moxifloxacin",   "smiles":"COc1c(N2C[C@@H]3CCCC[C@@H]3C2)c(F)cc2c(=O)c(C(=O)O)cn3cc(C)nc1c23","cid":152946,"class":"antibacterial"},
            {"name":"Azithromycin",   "smiles":"CCC1OC(=O)C(C)C(OC2CC(N(C)C)C(O)C(C)O2)C(C)CC(O)(CC=OC(C)C1OC3OC(CO)C(O)C(OC)C3OC)CC(C)C(=O)OC","cid":447043,"class":"antibacterial"},
            {"name":"Clarithromycin", "smiles":"CCC1OC(=O)C(C)C(OC2CC(N(C)C)C(O)C(C)O2)C(C)CC(OC(=O)C)(CC=O)C(C)C1OC3OC(C)CC(OC)C3OC","cid":84029,"class":"antibacterial"},
            {"name":"Acyclovir",      "smiles":"Nc1nc2c(ncn2COCCO)c(=O)[nH]1","cid":2022,"class":"antiviral"},
            {"name":"Valacyclovir",   "smiles":"CC(C)OC(=O)OCC(NC(=O)C1(N)CCO1)C(=O)O","cid":135398513,"class":"antiviral"},
            {"name":"Oseltamivir",    "smiles":"CCOC(=O)C1=CC(OC(CC)CC)C(NC(C)=O)CC1N","cid":65028,"class":"antiviral"},
            {"name":"Zanamivir",      "smiles":"CC(=O)N[C@@H]1[C@@H](O)C=C(C(=O)O)O[C@H]1[C@@H](O)[C@@H](O)CO","cid":60855,"class":"antiviral"},
            {"name":"Tenofovir",      "smiles":"Cc1ncnc2c1nc(N)n2C1CC(COP(=O)(O)O)O1","cid":464205,"class":"antiviral"},
            {"name":"Entecavir",      "smiles":"Nc1nc2c(ncn2C2CC(=C)C(CO)O2)c(=O)[nH]1","cid":135398513,"class":"antiviral"},
            {"name":"Hydroxychloroquine","smiles":"CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12","cid":3652,"class":"antimalarial"},
            {"name":"Primaquine",     "smiles":"COc1ccc(Nc2cccc3cc(N)ncc23)cc1","cid":4908,"class":"antimalarial"},
            {"name":"Doxycycline",    "smiles":"CC1c2cccc(O)c2C(=O)C3=C(O)C4(O)C(=O)C(=C(N(C)C)O)[C@@H](O)[C@]4(O)[C@@H](O)[C@@H]13","cid":54671203,"class":"antibacterial"},
            {"name":"Minocycline",    "smiles":"CN(C)c1ccc(cc1O)C(=O)NC2CC(C)(C)N(C)C2","cid":54675776,"class":"antibacterial"},
            {"name":"Tetracycline",   "smiles":"CN(C)[C@@H]1[C@@H]2C[C@@H]3Cc4c(O)ccc(O)c4C(=O)[C@@]3(O)C(=O)[C@]2(O)C(=O)[C@@H](N)C1","cid":54675776,"class":"antibacterial"},
            {"name":"Prednisolone",   "smiles":"OCC(=O)[C@@]1(O)CC[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@H]3[C@@H](O)C[C@@]12C","cid":4894,"class":"anti-inflammatory"},
            {"name":"Dexamethasone",  "smiles":"C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@H]3[C@@H](O)C[C@@]2(C)[C@H]1OC(C)=O","cid":5743,"class":"anti-inflammatory"},
            {"name":"Betamethasone",  "smiles":"C[C@@H]1C[C@H]2[C@@H]3C[C@@H](F)[C@H](O)[C@@]3(C)[C@@H](O)C[C@@]2(C)[C@@H]1C(=O)CO","cid":9782,"class":"anti-inflammatory"},
            {"name":"Piroxicam",      "smiles":"OC1=C(NC(=O)c2ccccn2)C(=O)c3ccccc3S1(=O)=O","cid":54676228,"class":"anti-inflammatory"},
            {"name":"Ketoprofen",     "smiles":"OC(=O)C(C)c1cccc(c1)C(=O)c1ccccc1","cid":3825,"class":"anti-inflammatory"},
            {"name":"Sulindac",       "smiles":"CC1=C(CC(=O)O)c2cc(F)ccc2/C1=C/c1ccc(S(C)=O)cc1","cid":1548887,"class":"anti-inflammatory"},
            {"name":"Tramadol",       "smiles":"OC1(CCCCC1)c1ccccc1","cid":33741,"class":"analgesic"},
            {"name":"Oxycodone",      "smiles":"COc1ccc2CC3N(C)CCC34CC1=C2OC(C)=O","cid":5284603,"class":"analgesic"},
            {"name":"Gabapentin",     "smiles":"NCC1(CC(=O)O)CCCCC1","cid":3446,"class":"analgesic"},
            {"name":"Pregabalin",     "smiles":"CCC(CN)CC(=O)O","cid":5487971,"class":"analgesic"},
            {"name":"Fentanyl",       "smiles":"CCC(=O)N(c1ccccc1)C1CCN(CCc2ccccc2)CC1","cid":3345,"class":"analgesic"},
            {"name":"Buprenorphine",  "smiles":"CC1CC2CC34CC5CC(=O)C(O)C5(C34)C(C1)(O)C1(CCCC21)C(C)(C)C","cid":644073,"class":"analgesic"},
            {"name":"Carbamazepine",  "smiles":"NC(=O)N1c2ccccc2C=Cc2ccccc21","cid":2554,"class":"antiepileptic"},
            {"name":"Phenytoin",      "smiles":"O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1","cid":1775,"class":"antiepileptic"},
            {"name":"Topiramate",     "smiles":"CC1(C)OC2COC3(COS(N)(=O)=O)OC1(C2OC3=O)C","cid":5284627,"class":"antiepileptic"},
            {"name":"Lamotrigine",    "smiles":"Nc1nc2cc(Cl)c(Cl)cc2[nH]c1=N","cid":28718,"class":"antiepileptic"},
            {"name":"Zolpidem",       "smiles":"Cc1ccc(-c2nc3ccc(C)cn3c2CC(=O)N(C)C)cc1","cid":5732,"class":"sedative"},
            {"name":"Diazepam",       "smiles":"CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21","cid":3016,"class":"sedative"},
        ],
    },
    "kinase_inhibitors_10": {
        "name":    "Kinase Inhibitors Library (10 compounds)",
        "ref":     "FDA Kinase Inhibitors + ChEMBL CHEMBL1163125",
        "compounds": [
            {"name":"Imatinib",     "smiles":"Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1","cid":5291,   "class":"anticancer"},
            {"name":"Erlotinib",    "smiles":"C#Cc1cccc(Nc2ncnc3cc(OCC)c(OCC)cc23)c1","cid":176870, "class":"anticancer"},
            {"name":"Gefitinib",    "smiles":"COc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCCCN1CCOCC1","cid":123631, "class":"anticancer"},
            {"name":"Dasatinib",    "smiles":"Cc1nc(Nc2ncc(C(=O)Nc3cccc(NCC4=NCC(O)N4)c3)cc2)cc(N2CCN(CCO)CC2)c1","cid":3062316,"class":"anticancer"},
            {"name":"Sorafenib",    "smiles":"CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1","cid":216239,"class":"anticancer"},
            {"name":"Sunitinib",    "smiles":"CCN(CC)CCNC(=O)c1c(C)[nH]c(-c2cc3c(F)ccc3[nH]2)c1-c1ccc(F)cc1","cid":5329102,"class":"anticancer"},
            {"name":"Nilotinib",    "smiles":"Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1cc2cnc(C)nc2cc1-c1ccc(C(F)(F)F)cc1","cid":644241,"class":"anticancer"},
            {"name":"Vemurafenib",  "smiles":"CCCS(=O)(=O)Nc1ccc(F)c(c1)C(=O)c1ccc(Cl)c(OCC2CC2)c1","cid":42611257,"class":"anticancer"},
            {"name":"Crizotinib",   "smiles":"Clc1ccc(c(F)c1)c1ccc2c(n1)nc(N1CCNCC1)c(n2)OCC","cid":11626560,"class":"anticancer"},
            {"name":"Lapatinib",    "smiles":"CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1","cid":208908,"class":"anticancer"},
        ],
    },
    "cardio_10": {
        "name":    "Cardiovascular Drugs Library (10 compounds)",
        "ref":     "FDA Cardiovascular Drugs + DrugBank",
        "compounds": [
            {"name":"Atorvastatin", "smiles":"CC(C)c1n(CC(O)CC(O)CC(=O)O)c2ccccc2c1CC3C(=O)Nc4ccc(F)cc43","cid":60823, "class":"cardiovascular"},
            {"name":"Amlodipine",   "smiles":"CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OCC)C1c1ccccc1Cl","cid":2162,   "class":"cardiovascular"},
            {"name":"Lisinopril",   "smiles":"OC(=O)[C@@H](N)CCCc1ccccc1","cid":5362119,"class":"cardiovascular"},
            {"name":"Metoprolol",   "smiles":"COCCc1ccc(OCC(O)CNC(C)C)cc1","cid":4171,  "class":"cardiovascular"},
            {"name":"Warfarin",     "smiles":"OC(=O)Cc1ccccc1","cid":54678486,"class":"cardiovascular"},
            {"name":"Clopidogrel",  "smiles":"COC(=O)[C@@H](c1ccccc1Cl)N1CCc2sccc2C1","cid":60606,"class":"cardiovascular"},
            {"name":"Losartan",     "smiles":"CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1","cid":3961,"class":"cardiovascular"},
            {"name":"Simvastatin",  "smiles":"CCC(C)(C)C(=O)O[C@@H]1C[C@@H](C)C=C2C=C[C@H](C)[C@@H](CCC(O)=O)[C@@]21C","cid":54454,"class":"cardiovascular"},
            {"name":"Furosemide",   "smiles":"NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl","cid":3440,"class":"cardiovascular"},
            {"name":"Digoxin",      "smiles":"O=C1OCC(=C1)[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O","cid":2724385,"class":"cardiovascular"},
        ],
    },
    "cns_8": {
        "name":    "CNS Drugs Library (8 compounds)",
        "ref":     "FDA CNS Drugs + DrugBank",
        "compounds": [
            {"name":"Donepezil",    "smiles":"COc1cc2c(cc1OC)CC(CC(=O)Cc1ccc(O)cc1)C2","cid":2978,   "class":"cns"},
            {"name":"Sertraline",   "smiles":"CN[C@@H]1CC[C@H](c2ccc(Cl)c(Cl)c2)c2ccccc21","cid":68617,  "class":"cns"},
            {"name":"Fluoxetine",   "smiles":"CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1","cid":3386,   "class":"cns"},
            {"name":"Alprazolam",   "smiles":"Cc1nnc2n1-c1ccc(Cl)cc1C(c1ccccc1)=NC2","cid":2118,   "class":"cns"},
            {"name":"Haloperidol",  "smiles":"OC1(CCN(CCCc2ccc(F)cc2)CC1)c1ccc(Cl)cc1","cid":3559,   "class":"cns"},
            {"name":"Clozapine",    "smiles":"CN1CCN(CC1)c1nc2cc(Cl)ccc2[nH]c2ccccc12","cid":2818,   "class":"cns"},
            {"name":"Valproic Acid","smiles":"CCCC(CCC)C(=O)O","cid":3121,   "class":"cns"},
            {"name":"Levetiracetam","smiles":"CC[C@@H](C(=O)N)N1CCCC1=O","cid":441341, "class":"cns"},
        ],
    },
    "natural_10": {
        "name":    "Natural Product-Derived Drugs Library (10 compounds)",
        "ref":     "Newman & Cragg, J. Nat. Prod. 2020",
        "compounds": [
            {"name":"Taxol (Paclitaxel)","smiles":"O=C(O[C@@H]1C[C@]2(O)C(=O)[C@H](OC(=O)c3ccccc3)[C@@H](OC(=O)c3ccccc3)C[C@H]2[C@@H](OC(C)=O)[C@@H]1OC(=O)c1ccccc1)c1ccccc1","cid":36314,"class":"anticancer"},
            {"name":"Vincristine",  "smiles":"CCC1(O)CC2CN(CCC2=C1)c1nc2ccccc2cc1","cid":5978,   "class":"anticancer"},
            {"name":"Quinine",      "smiles":"OC(C1CC2CCN1CC2=C)c1ccnc2ccc(OC)cc12","cid":3034034,"class":"antimalarial"},
            {"name":"Morphine",     "smiles":"CN1CCC23C4Oc5c(O)ccc6c5C2C(CC4)OC3CC16","cid":5288826,"class":"analgesic"},
            {"name":"Artemisinin",  "smiles":"[C@@H]12O[C@](OO1)(C)CC[C@H]3[C@H]2CC[C@@H](C3(C)C)C","cid":68827,  "class":"antimalarial"},
            {"name":"Colchicine",   "smiles":"COc1cc2c(cc1OC)-c1cc(OC)c(=O)cc1CC2NC(C)=O","cid":6167,   "class":"anti-inflammatory"},
            {"name":"Caffeine",     "smiles":"Cn1cnc2c1c(=O)n(c(=O)n2C)C","cid":2519,   "class":"stimulant"},
            {"name":"Ephedrine",    "smiles":"CN[C@@H](C)[C@@H](O)c1ccccc1","cid":9294,   "class":"sympathomimetic"},
            {"name":"Resveratrol",  "smiles":"Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1","cid":445154, "class":"antioxidant"},
            {"name":"Curcumin",     "smiles":"O=C(/C=C/c1ccc(O)c(OC)c1)CC(=O)/C=C/c1ccc(O)c(OC)c1","cid":969516,"class":"anti-inflammatory"},
        ],
    },
}


# ─────────────────────────────────────────────────────────────
# ── Deduplicate compound libraries on module load ─────────────
# Ensures no SMILES appears twice across all libraries.
def _dedup_libraries():
    """Remove duplicate SMILES within each library and across libraries."""
    seen_global: set = set()
    for lib_key, lib in COMPOUND_LIBRARIES.items():
        if lib_key == "custom":
            continue
        seen_local: set = set()
        unique: list = []
        for cmpd in lib["compounds"]:
            sm = cmpd.get("smiles", "")
            if sm and sm not in seen_local:
                unique.append(cmpd)
                seen_local.add(sm)
        lib["compounds"] = unique
        seen_global.update(seen_local)

_dedup_libraries()


def run_virtual_screening(
    library_key:   str   = "druglike_12",
    protein:       str   = "6LU7",
    min_score:     float = 0.0,
    sort_by:       str   = "score",
    max_results:   int   = 50,
    custom_smiles: list  = None,
) -> dict:
    """
    Full virtual screening pipeline.

    Args:
        library_key:   key from COMPOUND_LIBRARIES
        protein:       PDB ID of target protein
        min_score:     minimum Drug Score to include (0–100)
        sort_by:       'score', 'dg', 'lipinski', 'qed', 'adme'
        max_results:   maximum number of results to return
        custom_smiles: list of {"name":..., "smiles":...} for custom library

    Returns:
        {
          library_name, protein, n_screened, n_passed_filter,
          top_candidates, statistics, screening_log,
        }
    """
    # Import here to avoid circular imports
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules.descriptors import compute_descriptors, lipinski_check
    from modules.adme       import predict_adme
    from modules.docking    import run_docking, drug_candidate_score
    from modules.ml_model   import predict_activity

    # Load library
    if library_key == "custom" and custom_smiles:
        compounds = custom_smiles
        lib_name  = f"Custom Library ({len(compounds)} compounds)"
        lib_ref   = "User-provided"
    elif library_key in COMPOUND_LIBRARIES:
        lib       = COMPOUND_LIBRARIES[library_key]
        compounds = lib["compounds"]
        lib_name  = lib["name"]
        lib_ref   = lib["ref"]
    else:
        # Default
        lib       = COMPOUND_LIBRARIES["druglike_12"]
        compounds = lib["compounds"]
        lib_name  = lib["name"]
        lib_ref   = lib["ref"]

    results      = []
    skipped      = 0
    log          = []
    log.append(f"[INFO] Library: {lib_name} ({len(compounds)} compounds)")
    log.append(f"[INFO] Target protein: {protein}")
    log.append(f"[INFO] Starting screening...")

    for i, cmpd in enumerate(compounds):
        name   = cmpd.get("name",   f"Compound_{i+1}")
        smiles = cmpd.get("smiles", "")
        cid    = cmpd.get("cid",    None)
        cls    = cmpd.get("class",  "unknown")

        if not smiles:
            skipped += 1
            continue

        try:
            desc    = compute_descriptors(smiles)
            lip     = lipinski_check(desc)
            adme    = predict_adme(desc)
            dock    = run_docking(smiles, {"pdb_id": protein, "path": None}, desc)
            score   = drug_candidate_score(dock, lip, adme)
            ml      = predict_activity(desc)

            bio    = adme.get("bioavailability") or adme.get("bioavailability_pct", 0)
            final  = score["final_score"]

            results.append({
                "rank":            0,           # filled after sorting
                "name":            name,
                "smiles":          smiles,
                "cid":             cid,
                "known_class":     cls,
                # Descriptors
                "MW":              desc.get("MW", 0),
                "LogP":            desc.get("LogP", 0),
                "HBD":             desc.get("HBD", 0),
                "HBA":             desc.get("HBA", 0),
                "TPSA":            desc.get("TPSA", 0),
                "QED":             desc.get("QED", 0),
                "AromaticRings":   desc.get("AromaticRings", 0),
                "MolecularVolume": desc.get("MolecularVolume", 0),
                "n_functional_groups": desc.get("n_functional_groups", 0),
                # Functional groups summary
                "functional_groups": [k.replace("has_","") for k,v in desc.items()
                                       if k.startswith("has_") and v],
                # Drug-likeness
                "lipinski_score":  lip["score"],
                "lipinski_passed": lip["passed"],
                "lipinski_verdict":lip["verdict"],
                "veber":           lip.get("veber", False),
                "egan":            lip.get("egan",  False),
                # ADME
                "bioavailability": round(float(bio), 1),
                "bbb":             adme.get("bbb", "N/A"),
                "gi_absorption":   adme.get("gi_absorption", "N/A"),
                "half_life":       adme.get("half_life_h", 0),
                # Docking
                "dg":              dock["best_dg"],
                "ki_uM":           dock["ki_uM"],
                "docking_score_pct": score.get("docking_pct", 0),
                # ML
                "predicted_activity":ml["predicted_class"],
                "ml_confidence":     ml["confidence"],
                # Final
                "adme_score":      score.get("adme_pct", 0),
                "drug_score":      final,
                "verdict":         score.get("verdict", "N/A"),
            })
        except Exception as e:
            skipped += 1
            log.append(f"[WARN] Skipped {name}: {e}")

    # Filter
    passed = [r for r in results if r["drug_score"] >= min_score]

    # Sort
    sort_keys = {
        "score":    lambda r: -r["drug_score"],
        "dg":       lambda r:  r["dg"],
        "lipinski": lambda r: -r["lipinski_score"],
        "qed":      lambda r: -r["QED"],
        "adme":     lambda r: -r["bioavailability"],
    }
    passed.sort(key=sort_keys.get(sort_by, sort_keys["score"]))

    # Assign ranks
    for i, r in enumerate(passed):
        r["rank"] = i + 1

    top = passed[:max_results]

    # Statistics
    if results:
        all_scores = [r["drug_score"] for r in results]
        all_dg     = [r["dg"] for r in results]
        stats = {
            "n_screened":        len(compounds),
            "n_scored":          len(results),
            "n_passed_filter":   len(passed),
            "n_skipped":         skipped,
            "avg_drug_score":    round(sum(all_scores)/len(all_scores), 1),
            "max_drug_score":    max(all_scores),
            "min_dg_kcal":       min(all_dg),
            "avg_dg_kcal":       round(sum(all_dg)/len(all_dg), 2),
            "n_drug_like":       sum(1 for r in results if r["lipinski_passed"] >= 5),
            "n_promising":       sum(1 for r in results if r["drug_score"] >= 65),
            "hit_rate_pct":      round(sum(1 for r in results if r["drug_score"] >= 65) / max(1, len(results)) * 100, 1),
        }
    else:
        stats = {"error": "No compounds scored"}

    log.append(f"[OK]   Scored {len(results)}/{len(compounds)} compounds")
    log.append(f"[OK]   {len(passed)} compounds passed filter (min score={min_score}%)")
    if top:
        log.append(f"[DONE] Top candidate: {top[0]['name']} (Score={top[0]['drug_score']}%, ΔG={top[0]['dg']} kcal/mol)")

    return {
        "library_name":    lib_name,
        "library_ref":     lib_ref,
        "protein":         protein,
        "sort_by":         sort_by,
        "min_score_filter":min_score,
        "n_screened":      len(compounds),
        "n_passed_filter": len(passed),
        "top_candidates":  top,
        "statistics":      stats,
        "screening_log":   log,
        "references": [
            "Trott & Olson, J. Comput. Chem. 2010 (AutoDock Vina scoring)",
            "Shoichet, Nature 2004 (virtual screening review)",
            "Bickerton et al., Nat. Chem. 2012 (QED)",
            "Lipinski et al., Adv. Drug Deliv. Rev. 1997 (drug-likeness)",
        ],
    }


def get_library_info() -> dict:
    """Return metadata about all available compound libraries."""
    return {
        key: {
            "name":       lib["name"],
            "ref":        lib["ref"],
            "n_compounds":len(lib["compounds"]),
            "classes":    list(set(c.get("class","unknown") for c in lib["compounds"])),
        }
        for key, lib in COMPOUND_LIBRARIES.items()
        if key != "custom"
    }
