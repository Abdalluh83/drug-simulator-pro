"""
modules/pdf_report.py  —  Scientific PDF Report Generator
==========================================================
Generates a professional research-grade PDF report for any analyzed compound.
Uses reportlab (pip install reportlab).

Group 2 | Pharmaceutical Manufacturing Technology
Faculty of Applied Health Sciences — BATU
"""

import io
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    # ── Color Palette ──────────────────────────────────────────
    DARK_BLUE   = colors.HexColor("#1a2e4a")
    MED_BLUE    = colors.HexColor("#2563eb")
    LIGHT_BLUE  = colors.HexColor("#dbeafe")
    GREEN       = colors.HexColor("#16a34a")
    LIGHT_GREEN = colors.HexColor("#dcfce7")
    RED         = colors.HexColor("#dc2626")
    LIGHT_RED   = colors.HexColor("#fee2e2")
    YELLOW      = colors.HexColor("#ca8a04")
    LIGHT_YEL   = colors.HexColor("#fef9c3")
    GRAY        = colors.HexColor("#6b7280")
    LIGHT_GRAY  = colors.HexColor("#f3f4f6")
    WHITE       = colors.white

    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


def _risk_color(risk: str):
    r = str(risk).lower()
    if r in ("high", "yes", "true", "positive", "caution"):
        return LIGHT_RED, RED
    if r in ("medium", "moderate", "monitor"):
        return LIGHT_YEL, YELLOW
    return LIGHT_GREEN, GREEN


def _bool_label(val) -> str:
    if isinstance(val, bool):
        return "Yes" if val else "No"
    return str(val)


def generate_pdf_report(analysis: dict) -> bytes:
    """
    Generate a PDF report from a full_analysis() result dict.
    Returns raw PDF bytes.
    """
    if not REPORTLAB_OK:
        raise ImportError("reportlab not installed — run: pip install reportlab")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.8*cm, leftMargin=1.8*cm,
        topMargin=2*cm,     bottomMargin=2*cm,
    )

    W = A4[0] - 3.6*cm   # usable width

    styles = getSampleStyleSheet()

    # ── Custom styles ──────────────────────────────────────────
    S = {
        "title": ParagraphStyle("title",
            fontSize=20, fontName="Helvetica-Bold",
            textColor=WHITE, alignment=TA_CENTER, spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle",
            fontSize=10, fontName="Helvetica",
            textColor=LIGHT_BLUE, alignment=TA_CENTER, spaceAfter=4),
        "section": ParagraphStyle("section",
            fontSize=12, fontName="Helvetica-Bold",
            textColor=WHITE, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body",
            fontSize=9, fontName="Helvetica",
            textColor=DARK_BLUE, leading=14),
        "small": ParagraphStyle("small",
            fontSize=7.5, fontName="Helvetica",
            textColor=GRAY, leading=11),
        "ref": ParagraphStyle("ref",
            fontSize=8, fontName="Helvetica-Oblique",
            textColor=GRAY, leading=12),
        "bold": ParagraphStyle("bold",
            fontSize=9, fontName="Helvetica-Bold",
            textColor=DARK_BLUE),
        "center": ParagraphStyle("center",
            fontSize=9, fontName="Helvetica",
            textColor=DARK_BLUE, alignment=TA_CENTER),
    }

    story = []

    # ── Helper: section header ─────────────────────────────────
    def section_header(text):
        tbl = Table([[Paragraph(text, S["section"])]], colWidths=[W])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), DARK_BLUE),
            ("ROWPADDING", (0,0), (-1,-1), 6),
            ("ROUNDEDCORNERS", [4]),
        ]))
        story.append(Spacer(1, 8))
        story.append(tbl)
        story.append(Spacer(1, 4))

    def kv_table(rows, col_widths=None):
        """Two-column key-value table."""
        cw = col_widths or [W*0.42, W*0.58]
        data = [[Paragraph(f"<b>{k}</b>", S["body"]),
                 Paragraph(str(v), S["body"])] for k, v in rows]
        tbl = Table(data, colWidths=cw)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), LIGHT_GRAY),
            ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#e5e7eb")),
            ("ROWPADDING", (0,0), (-1,-1), 5),
            ("VALIGN",     (0,0), (-1,-1), "TOP"),
        ]))
        story.append(tbl)

    # ══════════════════════════════════════════════════════════
    #  PAGE 1 — COVER
    # ══════════════════════════════════════════════════════════
    compound  = analysis.get("compound", {})
    comp_name = compound.get("name", "Unknown Compound")
    smiles    = analysis.get("smiles", compound.get("smiles", "N/A"))
    now       = datetime.now().strftime("%B %d, %Y  —  %H:%M")

    # Header banner
    banner = Table(
        [[Paragraph("AI-Driven Drug Discovery Platform", S["title"])],
         [Paragraph("Research-Grade CADD v3.1", S["subtitle"])],
         [Paragraph(
             "Group 2 &nbsp;|&nbsp; Pharmaceutical Manufacturing Technology Program<br/>"
             "Faculty of Applied Health Sciences &nbsp;|&nbsp; Borg Alarab Technological University (BATU)",
             S["subtitle"])]],
        colWidths=[W]
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), DARK_BLUE),
        ("ROWPADDING", (0,0), (-1,-1), 8),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(banner)
    story.append(Spacer(1, 18))

    # Compound info card
    drug_score = analysis.get("drug_score", {})
    score_val  = drug_score.get("score", drug_score.get("total", "N/A"))
    grade      = drug_score.get("grade", drug_score.get("label", ""))

    info_card = Table([
        [Paragraph(f"<b>Compound Name</b>", S["body"]),
         Paragraph(comp_name, S["bold"]),
         Paragraph(f"<b>Report Date</b>", S["body"]),
         Paragraph(now, S["body"])],
        [Paragraph(f"<b>SMILES</b>", S["body"]),
         Paragraph(f'<font size="7">{smiles}</font>', S["body"]),
         Paragraph(f"<b>Drug Score</b>", S["body"]),
         Paragraph(f"{score_val}  {grade}", S["bold"])],
    ], colWidths=[W*0.15, W*0.42, W*0.15, W*0.28])
    info_card.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), LIGHT_BLUE),
        ("GRID",       (0,0), (-1,-1), 0.4, MED_BLUE),
        ("ROWPADDING", (0,0), (-1,-1), 7),
        ("SPAN",       (0,0), (0,0)),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(info_card)
    story.append(Spacer(1, 14))

    # ── Pipeline Summary ───────────────────────────────────────
    section_header("Pipeline Summary")
    steps = [
        ("Molecular Descriptors", "RDKit exact computation"),
        ("Drug-Likeness",         "Lipinski Ro5 + Veber + Egan + QED"),
        ("ADME Prediction",       "Egan 2000, Clark 1999, ESOL 2004"),
        ("Toxicity Prediction",   "8 endpoints — pkCSM-style QSAR"),
        ("Binding Pocket",        "fpocket-inspired geometric detection"),
        ("Molecular Docking",     "AutoDock Vina / QSAR fallback (MAE 0.94 kcal/mol)"),
        ("ML Bioactivity",        "Random Forest 83-85% CV (n=2000 ChEMBL-derived)"),
        ("Explainable AI",        "SHAP / manual Shapley approximation"),
        ("Drug Similarity",       "Tanimoto Morgan fingerprints"),
        ("Virtual Screening",     "Ranked compound library scoring"),
    ]
    step_rows = [[Paragraph(f"<b>Step {i+1}</b>", S["body"]),
                  Paragraph(k, S["bold"]),
                  Paragraph(v, S["body"])]
                 for i,(k,v) in enumerate(steps)]
    tbl = Table(step_rows, colWidths=[W*0.10, W*0.30, W*0.60])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), LIGHT_BLUE),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#e5e7eb")),
        ("ROWPADDING", (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [WHITE, LIGHT_GRAY]),
    ]))
    story.append(tbl)

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    #  PAGE 2 — MOLECULAR DESCRIPTORS + DRUG-LIKENESS
    # ══════════════════════════════════════════════════════════
    section_header("1. Molecular Descriptors")
    props = analysis.get("properties", {})
    desc_rows = [
        ("Molecular Weight (MW)",    f"{props.get('MW', 'N/A')} g/mol"),
        ("LogP (Lipophilicity)",     props.get("LogP", "N/A")),
        ("H-Bond Donors (HBD)",      props.get("HBD", "N/A")),
        ("H-Bond Acceptors (HBA)",   props.get("HBA", "N/A")),
        ("Topological PSA (TPSA)",   f"{props.get('TPSA', 'N/A')} Å²"),
        ("Rotatable Bonds",          props.get("RotBonds", "N/A")),
        ("Aromatic Rings",           props.get("AromaticRings", "N/A")),
        ("Heavy Atom Count",         props.get("HeavyAtoms", "N/A")),
        ("QED Score",                props.get("QED", "N/A")),
        ("Descriptor Source",        props.get("source", "RDKit")),
    ]
    kv_table(desc_rows)

    # ── Drug-Likeness ──────────────────────────────────────────
    section_header("2. Drug-Likeness Analysis")
    lip = analysis.get("lipinski", {})

    # Lipinski passes
    lip_rows = []
    for rule, val in lip.items():
        if rule in ("passes", "qed", "drug_like", "violations", "label", "score"):
            continue
        bg, fg = _risk_color("Low" if val is True or val == "Pass" else "High")
        lip_rows.append([
            Paragraph(str(rule), S["body"]),
            Paragraph(_bool_label(val), S["body"]),
        ])

    if lip_rows:
        tbl = Table(lip_rows, colWidths=[W*0.55, W*0.45])
        tbl.setStyle(TableStyle([
            ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#e5e7eb")),
            ("ROWPADDING", (0,0), (-1,-1), 5),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [WHITE, LIGHT_GRAY]),
        ]))
        story.append(tbl)

    # Summary badge
    passes   = lip.get("passes", lip.get("drug_like", False))
    drug_like_label = "DRUG-LIKE ✓" if passes else "NOT DRUG-LIKE ✗"
    badge_bg  = LIGHT_GREEN if passes else LIGHT_RED
    badge_fg  = GREEN if passes else RED
    _fg_hex = '#{:02x}{:02x}{:02x}'.format(
        int(badge_fg.red * 255), int(badge_fg.green * 255), int(badge_fg.blue * 255))
    badge = Table([[Paragraph(
        f'<font color="{_fg_hex}"><b>{drug_like_label}</b></font>  '
        f'Violations: {lip.get("violations", "N/A")}  |  QED: {lip.get("qed", lip.get("QED", "N/A"))}',
        S["center"])]],
        colWidths=[W])
    badge.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), badge_bg),
        ("ROWPADDING", (0,0), (-1,-1), 8),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(Spacer(1, 6))
    story.append(badge)

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    #  PAGE 3 — ADME + TOXICITY
    # ══════════════════════════════════════════════════════════
    section_header("3. ADME Prediction")
    adme = analysis.get("adme", {})
    adme_rows = [(k, str(v)) for k, v in adme.items()
                 if not isinstance(v, dict)]
    kv_table(adme_rows)

    section_header("4. Toxicity Prediction")
    tox = analysis.get("toxicity", {})
    tox_data = []
    for endpoint, result in tox.items():
        if isinstance(result, dict):
            risk = str(result.get("risk", result.get("label", result.get("value", "N/A"))))
        else:
            risk = str(result)
        bg, fg = _risk_color(risk)
        tox_data.append([
            Paragraph(str(endpoint).replace("_", " ").title(), S["body"]),
            Paragraph(risk, S["bold"]),
        ])

    if tox_data:
        tbl = Table(tox_data, colWidths=[W*0.60, W*0.40])
        tbl.setStyle(TableStyle([
            ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#e5e7eb")),
            ("ROWPADDING", (0,0), (-1,-1), 6),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [WHITE, LIGHT_GRAY]),
        ]))
        story.append(tbl)
    else:
        story.append(Paragraph("No toxicity data available.", S["body"]))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    #  PAGE 4 — DOCKING + ML
    # ══════════════════════════════════════════════════════════
    section_header("5. Molecular Docking")
    dock = analysis.get("docking", {})
    dock_rows = [
        ("Binding Affinity (ΔG)",  f"{dock.get('best_dg', dock.get('dg', 'N/A'))} kcal/mol"),
        ("Ki (µM)",                 dock.get("ki_uM", "N/A")),
        ("Scoring Function",        dock.get("scoring_fn", dock.get("method", "QSAR heuristic"))),
        ("Vina Used",               _bool_label(dock.get("vina_used", False))),
        ("Docking MAE",             "0.94 kcal/mol (10-drug validation)"),
        ("Poses Generated",         len(dock.get("poses", []))),
        ("Target Protein",          analysis.get("pdb_info", {}).get("id", "N/A")),
    ]
    kv_table(dock_rows)

    # Binding site
    bs = analysis.get("binding_site", {})
    if bs:
        section_header("6. Binding Pocket")
        bs_rows = [(k, str(v)) for k, v in bs.items()
                   if not isinstance(v, (dict, list))]
        kv_table(bs_rows)

    section_header("7. ML Bioactivity Prediction")
    story.append(Paragraph(
        "The Random Forest model (83–85% 5-fold CV, n=2000 ChEMBL-derived) "
        "predicts compound bioactivity class based on 12 molecular descriptors. "
        "Use the /ml_predict endpoint for per-compound predictions.",
        S["body"]))
    story.append(Spacer(1, 6))
    ml_info = [
        ("Algorithm",         "Random Forest (Breiman 2001)"),
        ("Training Set",      "n = 2000 (ChEMBL-derived, Gaussian sampling)"),
        ("Cross-Validation",  "5-fold Stratified CV"),
        ("Accuracy",          "83–85% ± ~2%"),
        ("GNN Model",         "80.7% CV (topology-augmented descriptor NN)"),
        ("vs Logistic Reg.",  "+7% improvement"),
        ("Explainability",    "SHAP values (feature importance)"),
    ]
    kv_table(ml_info)

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    #  PAGE 5 — DRUG SCORE + REFERENCES
    # ══════════════════════════════════════════════════════════
    section_header("8. Drug Candidate Score")
    score_rows = [(k, str(v)) for k, v in drug_score.items()]
    kv_table(score_rows)

    # Overall verdict
    score_num = 0
    try:
        score_num = float(str(score_val).replace("%",""))
    except Exception:
        pass
    verdict = "Strong Candidate" if score_num >= 70 else \
              "Moderate Candidate" if score_num >= 50 else "Weak Candidate"
    bg = LIGHT_GREEN if score_num >= 70 else LIGHT_YEL if score_num >= 50 else LIGHT_RED
    v_tbl = Table([[Paragraph(
        f"<b>Overall Verdict: {verdict}</b>  |  Drug Score: {score_val}",
        S["center"])]],
        colWidths=[W])
    v_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bg),
        ("ROWPADDING", (0,0), (-1,-1), 10),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(Spacer(1, 6))
    story.append(v_tbl)

    # ── Benchmark ──────────────────────────────────────────────
    section_header("9. Benchmark vs Comparable Tools")
    bench_data = [
        ["Tool",          "ADME", "Toxicity", "Docking", "ML/AI", "XAI", "API"],
        ["This Platform", "✓",    "8 endpoints","✓ Vina", "✓ RF+GNN","✓ SHAP","37"],
        ["SwissADME",     "✓",    "Limited",  "✗",       "✗",     "✗",   "Web only"],
        ["pkCSM",         "✓",    "✓",        "✗",       "✗",     "✗",   "Web only"],
        ["AutoDock Vina", "✗",    "✗",        "✓",       "✗",     "✗",   "CLI only"],
    ]
    tbl = Table(bench_data, colWidths=[W*0.24, W*0.10, W*0.16, W*0.12, W*0.12, W*0.10, W*0.16])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  DARK_BLUE),
        ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("BACKGROUND",  (0,1), (-1,1),  LIGHT_BLUE),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#e5e7eb")),
        ("ROWPADDING",  (0,0), (-1,-1), 5),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("ALIGN",       (1,0), (-1,-1), "CENTER"),
    ]))
    story.append(tbl)

    # ── References ─────────────────────────────────────────────
    section_header("10. Scientific References")
    refs = [
        "[1] Lipinski et al., Adv. Drug Deliv. Rev. 23:3–25 (1997) — Rule of Five",
        "[2] Trott & Olson, J. Comput. Chem. 31:455–461 (2010) — AutoDock Vina",
        "[3] Breiman, Machine Learning 45:5–32 (2001) — Random Forest",
        "[4] Egan et al., J. Med. Chem. 43:3867–3877 (2000) — Boiled-Egg ADME",
        "[5] Bickerton et al., Nat. Chem. 4:90–98 (2012) — QED",
        "[6] Wishart et al., Nucleic Acids Res. 46:D608–D617 (2018) — DrugBank",
        "[7] Veber et al., J. Med. Chem. 45:2615–2623 (2002) — Oral bioavailability",
        "[8] Daina et al., Sci. Rep. 7:42717 (2017) — SwissADME",
        "[9] Pires et al., J. Med. Chem. 58:4066–4072 (2015) — pkCSM",
        "[10] Lundberg & Lee, NeurIPS (2017) — SHAP explainability",
    ]
    for ref in refs:
        story.append(Paragraph(ref, S["ref"]))
        story.append(Spacer(1, 2))

    # ── Footer ─────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width=W, color=GRAY, thickness=0.5))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"Generated by AI-Driven Drug Discovery Platform v3.1 — BATU Group 2 — {now}",
        S["small"]))
    story.append(Paragraph(
        "This report is for research and academic purposes only. "
        "Not intended for clinical decision-making.",
        S["small"]))

    doc.build(story)
    return buffer.getvalue()
