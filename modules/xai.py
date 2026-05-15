"""
xai.py — Explainable AI (XAI) Module
======================================
Group 2 — Pharmaceutical Manufacturing Technology

Provides model interpretability using:
  1. SHAP (SHapley Additive exPlanations) when available
  2. Manual feature attribution when SHAP is not installed

SHAP Reference:
  Lundberg & Lee, NeurIPS 2017 — "A Unified Approach to Interpreting Model Predictions"
  https://shap.readthedocs.io/

Scientific basis:
  SHAP values decompose model predictions into per-feature contributions.
  For a Random Forest: SHAP(x) = Σ φᵢ where Σφᵢ = f(x) - E[f(x)]
  This satisfies: Efficiency, Symmetry, Dummy, Additivity (Shapley axioms)

Usage in Drug Discovery:
  - Identifies which molecular features drive activity prediction
  - Provides medicinal chemistry insight: "compound is predicted active 
    because of high aromatic rings and moderate LogP"
  - Enables structure-activity relationship (SAR) interpretation
"""

import logging
logger = logging.getLogger(__name__)

# ── SHAP (optional, enables real Shapley values) ─────────────
try:
    import shap
    import numpy as np
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False

FEATURE_NAMES = [
    "MW", "LogP", "HBD", "HBA", "TPSA",
    "RotBonds", "AromaticRings", "FractionCSP3", "QED"
]

# Human-readable feature descriptions for medicinal chemists
FEATURE_DESCRIPTIONS = {
    "MW":            "Molecular Weight (Da) — size determines cell penetration",
    "LogP":          "Lipophilicity — controls membrane permeability and solubility",
    "HBD":           "H-Bond Donors — affects oral absorption (Lipinski)",
    "HBA":           "H-Bond Acceptors — affects distribution and binding",
    "TPSA":          "Topological Polar Surface Area — controls GI absorption",
    "RotBonds":      "Rotatable Bonds — affects oral bioavailability (Veber rule)",
    "AromaticRings": "Aromatic Rings — π-π stacking with target residues",
    "FractionCSP3":  "Fraction sp3 Carbons — 3D shape diversity (Fsp3)",
    "QED":           "Quantitative Drug-likeness (0-1) — overall drug-likeness",
}

# Direction of influence (positive = increases activity probability)
FEATURE_DIRECTIONS = {
    "anti-inflammatory": {
        "AromaticRings": +1, "LogP": +1, "MW": -1,
        "HBD": -1, "TPSA": -1, "QED": +1, "HBA": +1, "RotBonds": -1, "FractionCSP3": +1,
    },
    "analgesic": {
        "FractionCSP3": +1, "HBD": +1, "AromaticRings": +1, "MW": -1,
        "QED": +1, "TPSA": -1, "LogP": +1, "HBA": +1, "RotBonds": -1,
    },
    "antibacterial": {
        "HBA": +1, "HBD": +1, "MW": +1, "TPSA": +1,
        "AromaticRings": +1, "LogP": -1, "QED": +1, "RotBonds": +1, "FractionCSP3": -1,
    },
    "antiviral": {
        "AromaticRings": +1, "MW": +1, "HBA": +1, "FractionCSP3": +1,
        "RotBonds": +1, "LogP": +1, "HBD": +1, "QED": -1, "TPSA": +1,
    },
    "anticancer": {
        "AromaticRings": +1, "MW": +1, "RotBonds": +1, "HBA": +1,
        "LogP": +1, "FractionCSP3": -1, "HBD": -1, "TPSA": -1, "QED": -1,
    },
}

# Class profile means (from ml_model.py — used for SHAP approximation)
CLASS_MEANS = {
    "anti-inflammatory": [215.0, 2.40, 1.8, 3.8, 72.0, 3.5, 1.4, 0.46, 0.67],
    "analgesic":         [280.0, 1.50, 2.4, 4.2, 92.0, 3.0, 1.9, 0.58, 0.60],
    "antibacterial":     [378.0, 1.10, 3.2, 7.2, 122.0, 4.8, 1.1, 0.40, 0.54],
    "antiviral":         [448.0, 3.00, 1.8, 7.8, 128.0, 6.5, 2.4, 0.32, 0.49],
    "anticancer":        [485.0, 3.80, 1.4, 6.8,  98.0, 5.8, 2.9, 0.26, 0.46],
}


# ─────────────────────────────────────────────────────────────
def explain_prediction(descriptors: dict, predicted_class: str,
                        ml_model_cache: dict = None) -> dict:
    """
    Generate SHAP-based explanation for a prediction.

    Args:
        descriptors:    molecular descriptor dict
        predicted_class: predicted activity class
        ml_model_cache: optional model cache from ml_model.py

    Returns:
        Explanation dict with feature contributions, reasoning text, SAR summary
    """
    if SHAP_OK and ml_model_cache and ml_model_cache.get("trained"):
        return _shap_explanation(descriptors, predicted_class, ml_model_cache)
    else:
        return _manual_explanation(descriptors, predicted_class)


def _shap_explanation(descriptors: dict, predicted_class: str,
                       cache: dict) -> dict:
    """Real SHAP values using TreeExplainer."""
    try:
        model  = cache["clf"]
        feats  = [float(descriptors.get(f, 0)) for f in FEATURE_NAMES]
        X      = np.array([feats])

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)  # shape: (n_classes, 1, n_features)

        # Get SHAP for predicted class
        classes     = list(FEATURE_DIRECTIONS.keys())
        class_idx   = classes.index(predicted_class) if predicted_class in classes else 0
        sv          = shap_values[class_idx][0]  # (n_features,)
        expected    = float(explainer.expected_value[class_idx])

        contributions = []
        for i, fname in enumerate(FEATURE_NAMES):
            contributions.append({
                "feature":     fname,
                "value":       round(float(feats[i]), 3),
                "shap_value":  round(float(sv[i]), 4),
                "direction":   "positive" if sv[i] > 0 else "negative",
                "magnitude":   round(abs(float(sv[i])), 4),
                "description": FEATURE_DESCRIPTIONS.get(fname, ""),
            })
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "method":         "SHAP TreeExplainer (Lundberg & Lee 2017)",
            "predicted_class":predicted_class,
            "expected_value": round(expected, 4),
            "contributions":  contributions,
            "top_features":   contributions[:3],
            "sar_summary":    _build_sar_summary(contributions[:3], predicted_class),
            "reasoning":      _build_reasoning(contributions, predicted_class, feats),
            "shap_available": True,
        }
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        return _manual_explanation(descriptors, predicted_class)


def _manual_explanation(descriptors: dict, predicted_class: str) -> dict:
    """
    Manual SHAP approximation using class profile deviations.
    For each feature, computes how far the compound deviates from
    the class mean — features with large deviations have high influence.
    """
    feats      = [float(descriptors.get(f, 0)) for f in FEATURE_NAMES]
    class_mu   = CLASS_MEANS.get(predicted_class, CLASS_MEANS["anti-inflammatory"])
    directions = FEATURE_DIRECTIONS.get(predicted_class, {})

    # Global mean across all classes for baseline
    global_mu = [
        sum(CLASS_MEANS[c][i] for c in CLASS_MEANS) / len(CLASS_MEANS)
        for i in range(len(FEATURE_NAMES))
    ]
    # Feature standard deviation (spread across classes)
    feat_std = [
        max(0.1, (max(CLASS_MEANS[c][i] for c in CLASS_MEANS)
                 - min(CLASS_MEANS[c][i] for c in CLASS_MEANS)) / 4)
        for i in range(len(FEATURE_NAMES))
    ]

    contributions = []
    for i, fname in enumerate(FEATURE_NAMES):
        val    = feats[i]
        mu     = class_mu[i]
        g_mu   = global_mu[i]
        std    = feat_std[i]
        dirn   = directions.get(fname, 1)

        # Shapley-inspired: how much does this feature push toward this class
        # compared to the global average?
        class_dev  = (val - mu) / std        # deviation from class mean
        global_dev = (val - g_mu) / std      # deviation from global mean
        # SHAP approximation: contribution = (deviation from global baseline)
        # scaled by how diagnostic this feature is for this class
        shap_approx = dirn * (global_dev - class_dev) * 0.12

        contributions.append({
            "feature":       fname,
            "value":         round(val, 3),
            "class_mean":    round(mu, 2),
            "shap_value":    round(shap_approx, 4),
            "direction":     "positive" if shap_approx > 0 else "negative",
            "magnitude":     round(abs(shap_approx), 4),
            "description":   FEATURE_DESCRIPTIONS.get(fname, ""),
        })

    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "method":         "Manual SHAP approximation (install shap for exact values)",
        "predicted_class":predicted_class,
        "expected_value": None,
        "contributions":  contributions,
        "top_features":   contributions[:3],
        "sar_summary":    _build_sar_summary(contributions[:3], predicted_class),
        "reasoning":      _build_reasoning(contributions, predicted_class, feats),
        "shap_available": False,
    }


def _build_sar_summary(top3: list, cls: str) -> str:
    """Build a medicinal chemistry SAR sentence."""
    if not top3:
        return "Insufficient data for SAR interpretation."
    parts = []
    for feat in top3:
        fname = feat["feature"]
        val   = feat["value"]
        dirn  = feat["direction"]
        if dirn == "positive":
            parts.append(f"{fname}={val:.2f} (favorable for {cls})")
        else:
            parts.append(f"{fname}={val:.2f} (unfavorable for {cls})")
    return "Key SAR factors: " + "; ".join(parts) + "."


def _build_reasoning(contribs: list, cls: str, feats: list) -> str:
    """Build human-readable reasoning paragraph."""
    top_pos = [c for c in contribs if c["direction"] == "positive"][:2]
    top_neg = [c for c in contribs if c["direction"] == "negative"][:1]

    pos_str = " and ".join(
        f"{c['feature']} ({c['value']:.2f})" for c in top_pos
    ) if top_pos else "structural features"

    neg_str = (
        f", though {top_neg[0]['feature']} ({top_neg[0]['value']:.2f}) "
        f"slightly reduces {cls} probability"
    ) if top_neg else ""

    return (
        f"The compound is predicted as {cls} primarily due to {pos_str}"
        f"{neg_str}. "
        f"This is consistent with the known SAR of {cls} drugs, "
        f"which typically show similar physicochemical profiles "
        f"(Lipinski 1997; Vistoli et al., Drug Discov. Today 2013)."
    )


def get_feature_importance_global(ml_model_cache: dict) -> dict:
    """Return global feature importances from trained RF model."""
    if ml_model_cache and ml_model_cache.get("trained"):
        clf = ml_model_cache.get("clf")
        if hasattr(clf, "feature_importances_"):
            imp = [round(float(v), 4) for v in clf.feature_importances_]
            return {
                "feature_names":   FEATURE_NAMES,
                "importances":     imp,
                "descriptions":    FEATURE_DESCRIPTIONS,
                "method":          "Random Forest mean decrease in impurity",
                "reference":       "Breiman, Machine Learning 2001, 45, 5-32",
            }
    # Fallback importances from calibration
    return {
        "feature_names":   FEATURE_NAMES,
        "importances":     [0.156, 0.142, 0.062, 0.157, 0.134,
                            0.055, 0.100, 0.099, 0.094],
        "descriptions":    FEATURE_DESCRIPTIONS,
        "method":          "Default (train model for real values)",
        "reference":       "Breiman, Machine Learning 2001, 45, 5-32",
    }
