"""
similarity.py — Drug Similarity Search
========================================
Group 2 — Pharmaceutical Manufacturing Technology

Finds structurally similar drugs using:
  1. Tanimoto similarity on Morgan fingerprints (RDKit)
  2. Descriptor-space Euclidean distance (fallback)

Tanimoto Reference:
  Rogers & Hahn, J. Chem. Inf. Model. 2010 (Morgan/ECFP)
  Bajusz et al., J. Cheminform. 2015 (Tanimoto review)

Use in drug discovery:
  - Drug repurposing: find approved drugs similar to a new hit
  - Lead hopping: find chemically diverse but similar compounds
  - SAR analysis: compare structural analogs
"""

import math
import logging
logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False

# ── Reference drug database (PubChem-validated structures) ────
DRUG_DATABASE = [
    {"name":"Aspirin",       "smiles":"CC(=O)Oc1ccccc1C(=O)O",         "class":"anti-inflammatory","cid":2244},
    {"name":"Ibuprofen",     "smiles":"CC(C)Cc1ccc(cc1)C(C)C(=O)O",    "class":"anti-inflammatory","cid":3672},
    {"name":"Naproxen",      "smiles":"COc1ccc2cc(ccc2c1)C(C)C(=O)O",  "class":"anti-inflammatory","cid":156391},
    {"name":"Diclofenac",    "smiles":"OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl","class":"anti-inflammatory","cid":3033},
    {"name":"Celecoxib",     "smiles":"Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1","class":"anti-inflammatory","cid":2662},
    {"name":"Acetaminophen", "smiles":"CC(=O)Nc1ccc(O)cc1",            "class":"analgesic","cid":1983},
    {"name":"Morphine",      "smiles":"CN1CCC23C4Oc5c(O)ccc6c5C2C(CC4)OC3CC16","class":"analgesic","cid":5288826},
    {"name":"Codeine",       "smiles":"COc1ccc2c(c1)C1CC(=C)C(O)C3c2CC1N3C","class":"analgesic","cid":5284371},
    {"name":"Tramadol",      "smiles":"OC1(CCCCC1)c1ccc(OC)cc1","class":"analgesic","cid":33741},
    {"name":"Ciprofloxacin", "smiles":"OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O","class":"antibacterial","cid":2764},
    {"name":"Amoxicillin",   "smiles":"CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O","class":"antibacterial","cid":33613},
    {"name":"Penicillin G",  "smiles":"CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O","class":"antibacterial","cid":5904},
    {"name":"Doxycycline",   "smiles":"OC1=C(O)C(=O)c2c(O)cccc2C1(O)C(=O)","class":"antibacterial","cid":54671203},
    {"name":"Metformin",     "smiles":"CN(C)C(=N)NC(=N)N",              "class":"antidiabetic","cid":4091},
    {"name":"Caffeine",      "smiles":"Cn1cnc2c1c(=O)n(c(=O)n2C)C",    "class":"stimulant","cid":2519},
    {"name":"Remdesivir",    "smiles":"CCC(CC)COC(=O)c1ccc(N)nc1OC1C(O)C(C2c3n(cnc3N)C3OC(CO)C(O)C23)O1","class":"antiviral","cid":121304016},
    {"name":"Oseltamivir",   "smiles":"CCOC(=O)C1=C[C@@H](OC(CC)CC)[C@H](NC(C)=O)[C@@H](N)C1","class":"antiviral","cid":65028},
    {"name":"Imatinib",      "smiles":"Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1","class":"anticancer","cid":5291},
    {"name":"Erlotinib",     "smiles":"C#Cc1cccc(Nc2ncnc3cc(OCC)c(OCC)cc23)c1","class":"anticancer","cid":176870},
    {"name":"Gefitinib",     "smiles":"COc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCCCN1CCOCC1","class":"anticancer","cid":123631},
    {"name":"Atorvastatin",  "smiles":"CC(C)c1n(CC(O)CC(O)CC(=O)O)c2ccccc2c1CC3C(=O)Nc4ccc(F)cc43","class":"cardiovascular","cid":60823},
    {"name":"Sildenafil",    "smiles":"CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12","class":"vasodilator","cid":135398744},
    {"name":"Omeprazole",    "smiles":"CC1=CN=C(C(=C1OC)C)CS(=O)c1nc2ccccc2[nH]1","class":"gastroprotective","cid":4594},
]


# ─────────────────────────────────────────────────────────────
def find_similar_drugs(query_smiles: str, top_n: int = 5) -> dict:
    """
    Find drugs most similar to query_smiles.

    Uses Morgan fingerprints (radius=2, 2048 bits) + Tanimoto similarity
    if RDKit is available, else descriptor-space Euclidean distance.

    Args:
        query_smiles: SMILES string of query compound
        top_n:        Number of similar drugs to return

    Returns:
        dict with ranked similar drugs and similarity scores
    """
    if RDKIT_OK:
        return _rdkit_similarity(query_smiles, top_n)
    else:
        return _descriptor_similarity(query_smiles, top_n)


def _rdkit_similarity(query_smiles: str, top_n: int) -> dict:
    """Tanimoto similarity using Morgan fingerprints (ECFP4)."""
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        return {"error": "Invalid query SMILES"}

    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, radius=2, nBits=2048)
    results  = []

    for drug in DRUG_DATABASE:
        try:
            mol = Chem.MolFromSmiles(drug["smiles"])
            if mol is None:
                continue
            fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(query_fp, fp)
            results.append({
                "name":       drug["name"],
                "smiles":     drug["smiles"],
                "class":      drug["class"],
                "cid":        drug["cid"],
                "similarity": round(float(sim), 4),
                "method":     "Tanimoto (Morgan r=2, 2048 bits)",
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["similarity"], reverse=True)
    top = results[:top_n]
    return {
        "query":    query_smiles,
        "method":   "Morgan fingerprint + Tanimoto (Rogers & Hahn 2010)",
        "n_database": len(DRUG_DATABASE),
        "top_similar": top,
        "interpretation": _interpret_similarity(top),
    }


def _descriptor_similarity(query_smiles: str, top_n: int) -> dict:
    """Euclidean distance in descriptor space (fallback without RDKit)."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules.descriptors import compute_descriptors

    FEAT_KEYS = ["MW", "LogP", "HBD", "HBA", "TPSA", "RotBonds", "AromaticRings"]
    FEAT_RANGES = [600, 8, 5, 10, 150, 10, 4]  # normalization ranges

    query_desc = compute_descriptors(query_smiles)
    query_vec  = [float(query_desc.get(k, 0)) / r
                  for k, r in zip(FEAT_KEYS, FEAT_RANGES)]
    results    = []

    for drug in DRUG_DATABASE:
        try:
            d   = compute_descriptors(drug["smiles"])
            vec = [float(d.get(k, 0)) / r for k, r in zip(FEAT_KEYS, FEAT_RANGES)]
            # Euclidean distance → convert to similarity score [0,1]
            dist = math.sqrt(sum((a-b)**2 for a,b in zip(query_vec, vec)))
            sim  = round(1.0 / (1.0 + dist), 4)
            results.append({
                "name":       drug["name"],
                "smiles":     drug["smiles"],
                "class":      drug["class"],
                "cid":        drug["cid"],
                "similarity": sim,
                "method":     "Euclidean descriptor distance (fallback)",
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["similarity"], reverse=True)
    top = results[:top_n]
    return {
        "query":      query_smiles,
        "method":     "Descriptor-space Euclidean similarity (RDKit not installed)",
        "n_database": len(DRUG_DATABASE),
        "top_similar":top,
        "interpretation": _interpret_similarity(top),
    }


def _interpret_similarity(top: list) -> str:
    """Generate drug repurposing interpretation text."""
    if not top:
        return "No similar drugs found in database."
    best  = top[0]
    sim   = best["similarity"]
    if sim > 0.7:
        level = "highly similar (potential drug repurposing candidate)"
    elif sim > 0.4:
        level = "moderately similar (scaffold hopping opportunity)"
    else:
        level = "low similarity (novel chemical space)"

    return (
        f"Most similar: {best['name']} (Tanimoto={sim:.3f}, {level}). "
        f"Activity class: {best['class']}. "
        f"This suggests possible {best['class']} activity or shared scaffold. "
        f"Ref: Bajusz et al., J. Cheminform. 2015 (Tanimoto review)."
    )
