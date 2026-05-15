"""
validation.py — Scientific Validation Module
=============================================
Group 2 — Graduation Project

Validates platform predictions against experimentally published data.

Validation Dataset:
  10 FDA-approved drugs with known experimental values for:
  - Binding affinity (Ki from IC50/experimental docking)
  - LogP (experimental, not computed)
  - Bioavailability F%
  - Half-life
  - BBB permeation

Sources:
  DrugBank v5.1 (Wishart et al., Nucleic Acids Res. 2018)
  ChEMBL (Gaulton et al., 2017)
  PubChem BioAssay Database
  FDA drug label data

Purpose:
  Answer the committee question: "How do you know your predictions are accurate?"
  By showing Pearson correlation between predicted and experimental values.
"""

import math
import logging
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  EXPERIMENTAL VALIDATION DATASET
#  10 FDA-approved drugs with known values
#  Sources: DrugBank 5.1, ChEMBL, FDA labels
# ─────────────────────────────────────────────────────────────
VALIDATION_DRUGS = [
    {
        "name":       "Aspirin",
        "smiles":     "CC(=O)Oc1ccccc1C(=O)O",
        "cid":        2244,
        "target":     "COX-1/COX-2",
        "experimental": {
            "MW":            180.16,
            "LogP":          1.19,    # experimental (DrugBank)
            "bioavail_pct":  80.0,    # oral F% (FDA label)
            "half_life_h":   0.33,    # 20 min aspirin → 2h salicylate
            "bbb":           "Permeable",
            "Ki_uM":         30.0,    # COX-1 inhibition ~30 μM (ChEMBL)
            "tpsa":          63.6,
            "hbd":           1,
            "hba":           4,
        },
        "reference": "DrugBank DB00945; Vane, Nature 1971",
    },
    {
        "name":       "Ibuprofen",
        "smiles":     "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "cid":        3672,
        "target":     "COX-1/COX-2",
        "experimental": {
            "MW":            206.28,
            "LogP":          3.97,
            "bioavail_pct":  87.0,
            "half_life_h":   2.0,
            "bbb":           "Non-permeable",
            "Ki_uM":         1.0,     # COX-1 ~1 μM
            "tpsa":          37.3,
            "hbd":           1,
            "hba":           2,
        },
        "reference": "DrugBank DB01050; ChEMBL CHEMBL521",
    },
    {
        "name":       "Acetaminophen",
        "smiles":     "CC(=O)Nc1ccc(O)cc1",
        "cid":        1983,
        "target":     "COX-2 / CB1",
        "experimental": {
            "MW":            151.16,
            "LogP":         -0.09,
            "bioavail_pct":  89.0,
            "half_life_h":   2.5,
            "bbb":           "Permeable",
            "Ki_uM":         25.0,
            "tpsa":          49.3,
            "hbd":           2,
            "hba":           2,
        },
        "reference": "DrugBank DB00316; Hinz et al., FASEB J. 2008",
    },
    {
        "name":       "Ciprofloxacin",
        "smiles":     "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
        "cid":        2764,
        "target":     "DNA Gyrase / Topoisomerase IV",
        "experimental": {
            "MW":            331.34,
            "LogP":          0.28,
            "bioavail_pct":  70.0,
            "half_life_h":   4.0,
            "bbb":           "Non-permeable",
            "Ki_uM":         0.008,   # MIC ~8 ng/mL
            "tpsa":          87.3,
            "hbd":           2,
            "hba":           6,
        },
        "reference": "DrugBank DB00537; EUCAST MIC data",
    },
    {
        "name":       "Metformin",
        "smiles":     "CN(C)C(=N)NC(=N)N",
        "cid":        4091,
        "target":     "AMPK / Complex I",
        "experimental": {
            "MW":            129.16,
            "LogP":         -1.43,
            "bioavail_pct":  55.0,
            "half_life_h":   6.2,
            "bbb":           "Non-permeable",
            "Ki_uM":         500.0,
            "tpsa":          88.4,
            "hbd":           4,
            "hba":           4,
        },
        "reference": "DrugBank DB00331; Graham et al., Clin. Pharm. 2011",
    },
    {
        "name":       "Caffeine",
        "smiles":     "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "cid":        2519,
        "target":     "Adenosine Receptors",
        "experimental": {
            "MW":            194.19,
            "LogP":         -0.07,
            "bioavail_pct":  99.0,
            "half_life_h":   5.0,
            "bbb":           "Permeable",
            "Ki_uM":         44.0,    # A1 receptor
            "tpsa":          58.4,
            "hbd":           0,
            "hba":           3,
        },
        "reference": "DrugBank DB00201; Fredholm et al., Pharmacol. Rev. 1999",
    },
    {
        "name":       "Amoxicillin",
        "smiles":     "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
        "cid":        33613,
        "target":     "PBP (Penicillin-binding protein)",
        "experimental": {
            "MW":            365.40,
            "LogP":          0.87,
            "bioavail_pct":  93.0,
            "half_life_h":   1.3,
            "bbb":           "Non-permeable",
            "Ki_uM":         0.05,
            "tpsa":          158.2,
            "hbd":           4,
            "hba":           8,
        },
        "reference": "DrugBank DB01060; Neu, N. Engl. J. Med. 1982",
    },
    {
        "name":       "Atorvastatin",
        "smiles":     "CC(C)c1n(CC(O)CC(O)CC(=O)O)c2ccccc2c1CC3C(=O)Nc4ccc(F)cc43",
        "cid":        60823,
        "target":     "HMG-CoA Reductase",
        "experimental": {
            "MW":            558.64,
            "LogP":          4.06,
            "bioavail_pct":  14.0,    # significant first-pass metabolism
            "half_life_h":   14.0,
            "bbb":           "Non-permeable",
            "Ki_uM":         0.008,
            "tpsa":          111.9,
            "hbd":           4,
            "hba":           8,
        },
        "reference": "DrugBank DB01076; Istvan & Deisenhofer, Science 2001",
    },
    {
        "name":       "Omeprazole",
        "smiles":     "CC1=CN=C(C(=C1OC)C)CS(=O)c1nc2ccccc2[nH]1",
        "cid":        4594,
        "target":     "H+/K+-ATPase (Proton Pump)",
        "experimental": {
            "MW":            345.42,
            "LogP":          2.23,
            "bioavail_pct":  65.0,
            "half_life_h":   1.0,
            "bbb":           "Non-permeable",
            "Ki_uM":         3.2,
            "tpsa":          86.3,
            "hbd":           1,
            "hba":           6,
        },
        "reference": "DrugBank DB00338; Lindberg et al., Gastroenterol. 1990",
    },
    {
        "name":       "Sildenafil",
        "smiles":     "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12",
        "cid":        135398744,
        "target":     "PDE5 (cGMP-specific phosphodiesterase)",
        "experimental": {
            "MW":            474.58,
            "LogP":          1.90,
            "bioavail_pct":  41.0,
            "half_life_h":   4.0,
            "bbb":           "Non-permeable",
            "Ki_uM":         0.0035,  # IC50 ~3.5 nM
            "tpsa":          113.0,
            "hbd":           1,
            "hba":           8,
        },
        "reference": "DrugBank DB00203; Corbin & Francis, J. Biol. Chem. 1999",
    },
]


# ─────────────────────────────────────────────────────────────
def run_validation(analysis_func) -> dict:
    """
    Run validation across all 10 drugs.
    Compares platform predictions vs experimental values.

    Args:
        analysis_func: function(smiles, protein, name) → full analysis dict

    Returns:
        Validation report with per-drug results and correlation stats
    """
    results = []
    for drug in VALIDATION_DRUGS:
        try:
            pred = analysis_func(drug["smiles"], "6LU7", drug["name"])
            exp  = drug["experimental"]
            p    = pred.get("properties", {})
            adme = pred.get("adme", {})
            dock = pred.get("docking", {})

            # Predicted values
            pred_mw    = _gv(p,    "MW",    "mw")
            pred_logp  = _gv(p,    "LogP",  "logp")
            pred_bio   = _gv(adme, "bioavailability", "bioavailability_pct")
            pred_hl    = _gv(adme, "half_life_h")
            pred_bbb   = adme.get("bbb", "")
            pred_dg    = dock.get("best_dg", 0)
            pred_ki    = dock.get("ki_uM", 0)

            # Errors
            mw_err  = abs(float(pred_mw  or 0) - exp["MW"])   if pred_mw  else None
            logp_err= abs(float(pred_logp or 0) - exp["LogP"]) if pred_logp else None
            bio_err = abs(float(pred_bio  or 0) - exp["bioavail_pct"]) if pred_bio else None
            hl_err  = abs(float(pred_hl   or 0) - exp["half_life_h"])  if pred_hl  else None
            bbb_ok  = (pred_bbb.lower().startswith("perm") == (exp["bbb"] == "Permeable"))

            results.append({
                "name":           drug["name"],
                "smiles":         drug["smiles"],
                "target":         drug["target"],
                "reference":      drug["reference"],
                "experimental":   exp,
                "predicted": {
                    "MW":           pred_mw,
                    "LogP":         pred_logp,
                    "bioavail_pct": pred_bio,
                    "half_life_h":  pred_hl,
                    "bbb":          pred_bbb,
                    "docking_dg":   pred_dg,
                    "ki_uM":        pred_ki,
                },
                "errors": {
                    "MW_Da":         round(mw_err,  2) if mw_err  is not None else None,
                    "LogP":          round(logp_err, 2) if logp_err is not None else None,
                    "bioavail_pct":  round(bio_err,  1) if bio_err  is not None else None,
                    "half_life_h":   round(hl_err,   2) if hl_err   is not None else None,
                    "bbb_correct":   bbb_ok,
                },
                "status": "ok",
            })
        except Exception as e:
            logger.warning(f"Validation failed for {drug['name']}: {e}")
            results.append({"name": drug["name"], "status": "error", "error": str(e)})

    # Summary statistics
    ok_results = [r for r in results if r.get("status") == "ok"]
    stats = _compute_stats(ok_results)
    return {
        "n_drugs":        len(VALIDATION_DRUGS),
        "n_successful":   len(ok_results),
        "results":        results,
        "statistics":     stats,
        "dataset_source": "DrugBank v5.1 + ChEMBL + FDA labels",
        "references": [
            "Wishart et al., Nucleic Acids Res. 2018 (DrugBank)",
            "Gaulton et al., Nucleic Acids Res. 2017 (ChEMBL)",
        ],
    }


def _gv(d: dict, *keys):
    """Get value from dict with multiple key fallbacks."""
    for k in keys:
        if d and d.get(k) is not None:
            return d[k]
    return None


def _compute_stats(results: list) -> dict:
    """Compute mean absolute errors and correlation for validation."""
    if not results:
        return {}

    mw_errs, logp_errs, bio_errs, hl_errs = [], [], [], []
    bbb_correct = 0

    for r in results:
        e = r.get("errors", {})
        if e.get("MW_Da")        is not None: mw_errs.append(e["MW_Da"])
        if e.get("LogP")         is not None: logp_errs.append(e["LogP"])
        if e.get("bioavail_pct") is not None: bio_errs.append(e["bioavail_pct"])
        if e.get("half_life_h")  is not None: hl_errs.append(e["half_life_h"])
        if e.get("bbb_correct"):              bbb_correct += 1

    def mean(lst): return round(sum(lst)/len(lst), 2) if lst else None

    # Pearson correlation for LogP (most reliable descriptor comparison)
    logp_corr = None
    try:
        exp_logp  = [r["experimental"]["LogP"] for r in results if r.get("errors",{}).get("LogP") is not None]
        pred_logp = [float(r["predicted"].get("LogP") or 0) for r in results if r.get("errors",{}).get("LogP") is not None]
        if len(exp_logp) >= 3:
            logp_corr = round(_pearson(exp_logp, pred_logp), 3)
    except Exception as e:
        logger.warning(f"Pearson correlation calculation failed: {e}")

    return {
        "MAE_MW_Da":        mean(mw_errs),
        "MAE_LogP":         mean(logp_errs),
        "MAE_bioavail_pct": mean(bio_errs),
        "MAE_half_life_h":  mean(hl_errs),
        "BBB_accuracy_pct": round(bbb_correct / len(results) * 100, 1) if results else None,
        "LogP_Pearson_r":   logp_corr,
        "interpretation": {
            "MAE_MW_Da":        "< 5 Da is excellent (RDKit exact, fallback ±15 Da)",
            "MAE_LogP":         "< 0.8 is good (Crippen LogP model ±0.5–1.0)",
            "MAE_bioavail_pct": "< 20% is acceptable (empirical model ±15–25%)",
            "MAE_half_life_h":  "< 3 h is acceptable (highly variable in vivo)",
            "BBB_accuracy_pct": "≥ 70% is good (Clark model 74% literature accuracy)",
        },
    }


def _pearson(x: list, y: list) -> float:
    """Compute Pearson correlation coefficient."""
    n  = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi-mx)*(yi-my) for xi, yi in zip(x, y))
    dx  = math.sqrt(sum((xi-mx)**2 for xi in x))
    dy  = math.sqrt(sum((yi-my)**2 for yi in y))
    if dx * dy == 0:
        return 0.0
    return num / (dx * dy)


# ─────────────────────────────────────────────────────────────
def validate_single_drug(drug_name: str, predicted: dict) -> dict:
    """Validate predictions for a single known drug."""
    drug = next((d for d in VALIDATION_DRUGS
                 if d["name"].lower() == drug_name.lower()), None)
    if not drug:
        return {"error": f"'{drug_name}' not in validation dataset"}

    exp  = drug["experimental"]
    p    = predicted.get("properties", {})
    adme = predicted.get("adme", {})

    pred_mw   = _gv(p,    "MW",    "mw")
    pred_logp = _gv(p,    "LogP",  "logp")
    pred_bio  = _gv(adme, "bioavailability", "bioavailability_pct")
    pred_bbb  = adme.get("bbb", "")
    bbb_ok    = pred_bbb.lower().startswith("perm") == (exp["bbb"] == "Permeable")

    return {
        "drug":         drug_name,
        "target":       drug["target"],
        "experimental": exp,
        "predicted": {
            "MW":    pred_mw,
            "LogP":  pred_logp,
            "bioavail_pct": pred_bio,
            "bbb":   pred_bbb,
        },
        "comparison": {
            "MW_error_Da":    round(abs(float(pred_mw  or 0) - exp["MW"]),   2),
            "LogP_error":     round(abs(float(pred_logp or 0) - exp["LogP"]), 2),
            "bio_error_pct":  round(abs(float(pred_bio  or 0) - exp["bioavail_pct"]), 1),
            "bbb_correct":    bbb_ok,
        },
        "reference": drug["reference"],
    }
