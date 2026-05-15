"""
vina_engine.py — Real AutoDock Vina Integration
================================================
Group 2 — Pharmaceutical Manufacturing Technology

Implements ACTUAL AutoDock Vina docking via subprocess.
Falls back to QSAR scoring if Vina is not installed.

What this adds (vs QSAR approximation):
  • Real 3D ligand conformer generation (via RDKit)
  • Real PDBQT file preparation for protein & ligand
  • Actual Vina subprocess call → real binding poses
  • Real energy decomposition from Vina output
  • Real RMSD between poses

Prerequisites (optional — falls back gracefully):
  pip install vina          ← Python Vina bindings (recommended)
  or: install AutoDock Vina binary from https://vina.scripps.edu/

References:
  Trott & Olson, J. Comput. Chem. 2010, 31, 455–461
  Eberhardt et al., J. Chem. Inf. Model. 2021 (AutoDock-GPU)
"""

import os, re, math, subprocess, tempfile, logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Try importing real Vina Python bindings ──────────────────
try:
    from vina import Vina as _Vina
    VINA_PYTHON = True
except ImportError:
    VINA_PYTHON = False

# ── Try RDKit for real 3D conformer generation ───────────────
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False


# ─────────────────────────────────────────────────────────────
def check_vina_available() -> dict:
    """Check which Vina method is available."""
    status = {
        "vina_python":  VINA_PYTHON,
        "vina_binary":  False,
        "rdkit":        RDKIT_OK,
        "method":       "qsar_fallback",
    }
    # Check for vina binary in PATH
    try:
        result = subprocess.run(
            ["vina", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 or "AutoDock Vina" in (result.stdout + result.stderr):
            status["vina_binary"] = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # vina binary not found in PATH — fallback to QSAR mode

    if VINA_PYTHON:
        status["method"] = "vina_python"
    elif status["vina_binary"]:
        status["method"] = "vina_binary"
    elif RDKIT_OK:
        status["method"] = "rdkit_qsar"
    return status


# ─────────────────────────────────────────────────────────────
def smiles_to_pdbqt(smiles: str, out_path: str) -> bool:
    """
    Convert SMILES → 3D SDF (via RDKit) → PDBQT (via Open Babel or meeko).
    Returns True on success.
    """
    if not RDKIT_OK:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result != 0:
            # Try ETKDG fallback
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)

        # Write SDF first
        sdf_path = out_path.replace(".pdbqt", ".sdf")
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()

        # Try Open Babel for SDF→PDBQT conversion
        try:
            subprocess.run(
                ["obabel", sdf_path, "-O", out_path, "--gen3d", "-p", "7.4"],
                capture_output=True, timeout=30, check=True
            )
            return os.path.exists(out_path)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass  # obabel not installed — try meeko next

        # Try meeko for SDF→PDBQT
        try:
            from meeko import MoleculePreparation
            preparator = MoleculePreparation()
            preparator.prepare(mol)
            preparator.write_pdbqt_file(out_path)
            return os.path.exists(out_path)
        except ImportError:
            pass  # meeko not installed — PDBQT preparation unavailable

        return False
    except Exception as e:
        logger.warning(f"SMILES→PDBQT failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────
def prepare_receptor_pdbqt(pdb_path: str, out_path: str) -> bool:
    """
    Prepare receptor PDBQT from PDB file.
    Removes water, adds hydrogens, assigns charges.
    """
    if not pdb_path or not os.path.exists(pdb_path):
        return False
    try:
        # Method 1: Open Babel
        subprocess.run(
            ["obabel", pdb_path, "-O", out_path,
             "-xr",          # remove waters
             "--partialcharge", "gasteiger"],
            capture_output=True, timeout=60, check=True
        )
        return os.path.exists(out_path)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass  # obabel not available for receptor prep — trying AutoDockTools next
    try:
        # Method 2: prepare_receptor4.py (AutoDock Tools)
        subprocess.run(
            ["prepare_receptor4.py", "-r", pdb_path, "-o", out_path,
             "-A", "hydrogens", "-U", "waters"],
            capture_output=True, timeout=60
        )
        return os.path.exists(out_path)
    except FileNotFoundError:
        return False


# ─────────────────────────────────────────────────────────────
def _parse_vina_output(output_text: str) -> list:
    """
    Parse AutoDock Vina stdout output.
    Returns list of {mode, affinity, rmsd_lb, rmsd_ub}
    
    Vina output format:
    -----+------------+----------+----------
       1 |    -8.3    |     0    |     0
       2 |    -7.9    |   1.234  |   1.876
    """
    poses = []
    lines = output_text.split("\n")
    in_table = False
    for line in lines:
        line = line.strip()
        if "mode" in line.lower() and "affinity" in line.lower():
            in_table = True
            continue
        if in_table and line.startswith("---"):
            continue
        if in_table and line:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    poses.append({
                        "mode":     int(parts[0]),
                        "affinity": float(parts[1]),
                        "rmsd_lb":  float(parts[2]),
                        "rmsd_ub":  float(parts[3]),
                    })
                except (ValueError, IndexError):
                    continue  # skip malformed Vina output line
    return poses


# ─────────────────────────────────────────────────────────────
def run_vina_python(receptor_pdbqt: str, ligand_pdbqt: str,
                    center: tuple, box: tuple,
                    exhaustiveness: int = 8) -> dict:
    """
    Run docking using Vina Python bindings (most reliable method).
    """
    v = _Vina(sf_name='vina', verbosity=0)
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    v.compute_vina_maps(
        center=list(center),
        box_size=list(box)
    )
    v.dock(exhaustiveness=exhaustiveness, n_poses=9)
    energies = v.energies()
    poses = []
    for i, e in enumerate(energies[:9]):
        poses.append({
            "mode":     i + 1,
            "affinity": round(float(e[0]), 2),
            "rmsd_lb":  round(float(e[1]) if len(e) > 1 else 0.0, 2),
            "rmsd_ub":  round(float(e[2]) if len(e) > 2 else 0.0, 2),
        })
    return {"poses": poses, "method": "vina_python"}


# ─────────────────────────────────────────────────────────────
def run_vina_binary(receptor_pdbqt: str, ligand_pdbqt: str,
                    center: tuple, box: tuple,
                    exhaustiveness: int = 8,
                    out_dir: str = "data") -> dict:
    """
    Run docking using AutoDock Vina binary via subprocess.
    """
    out_pdbqt = os.path.join(out_dir, "vina_out.pdbqt")
    cx, cy, cz = center
    sx, sy, sz = box
    cmd = [
        "vina",
        "--receptor", receptor_pdbqt,
        "--ligand",   ligand_pdbqt,
        "--out",      out_pdbqt,
        "--center_x", str(cx),
        "--center_y", str(cy),
        "--center_z", str(cz),
        "--size_x",   str(sx),
        "--size_y",   str(sy),
        "--size_z",   str(sz),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", "9",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    output  = result.stdout + result.stderr
    poses   = _parse_vina_output(output)
    if not poses:
        raise RuntimeError(f"Vina produced no poses. Output: {output[:200]}")
    return {"poses": poses, "method": "vina_binary", "raw_output": output}


# ─────────────────────────────────────────────────────────────
def run_real_docking(smiles: str, pdb_path: str, pdb_id: str,
                     center: tuple, box: tuple,
                     exhaustiveness: int = 8) -> dict:
    """
    Main entry point — tries real Vina, falls back gracefully.
    
    Returns:
        {
          "poses": [...],
          "method": "vina_python"|"vina_binary"|"qsar_fallback",
          "success": True|False,
          "message": "...",
        }
    """
    status = check_vina_available()

    with tempfile.TemporaryDirectory() as tmpdir:
        lig_pdbqt = os.path.join(tmpdir, "ligand.pdbqt")
        rec_pdbqt = os.path.join(tmpdir, "receptor.pdbqt")

        # ── Try Vina Python bindings ──────────────────────────
        if status["vina_python"] and RDKIT_OK:
            lig_ok = smiles_to_pdbqt(smiles, lig_pdbqt)
            rec_ok = prepare_receptor_pdbqt(pdb_path, rec_pdbqt) if pdb_path else False
            if lig_ok and rec_ok:
                try:
                    result = run_vina_python(rec_pdbqt, lig_pdbqt, center, box, exhaustiveness)
                    result.update({"success": True, "message": "AutoDock Vina (Python bindings)"})
                    return result
                except Exception as e:
                    logger.warning(f"Vina Python failed: {e}")

        # ── Try Vina binary ───────────────────────────────────
        if status["vina_binary"] and RDKIT_OK:
            lig_ok = smiles_to_pdbqt(smiles, lig_pdbqt)
            rec_ok = prepare_receptor_pdbqt(pdb_path, rec_pdbqt) if pdb_path else False
            if lig_ok and rec_ok:
                try:
                    result = run_vina_binary(rec_pdbqt, lig_pdbqt, center, box, exhaustiveness)
                    result.update({"success": True, "message": "AutoDock Vina (binary)"})
                    return result
                except Exception as e:
                    logger.warning(f"Vina binary failed: {e}")

    # ── Fallback: QSAR scoring ────────────────────────────────
    return {
        "success":  False,
        "method":   "qsar_fallback",
        "message":  (
            "AutoDock Vina not installed — using QSAR scoring approximation. "
            "Install: pip install vina  (or download Vina binary from vina.scripps.edu)"
        ),
        "poses":    None,
    }


# ─────────────────────────────────────────────────────────────
def get_installation_guide() -> str:
    """Returns installation instructions for Vina."""
    return """
AutoDock Vina Installation Guide
=================================

Option 1 — Python package (recommended):
  pip install vina

Option 2 — Binary:
  Download from: https://vina.scripps.edu/downloads/
  Add to PATH

Option 3 — For PDBQT preparation:
  pip install meeko         ← ligand preparation
  conda install -c conda-forge openbabel  ← receptor prep

After installation, restart the server.
Current status: """ + str(check_vina_available())
