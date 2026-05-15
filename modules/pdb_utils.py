"""
pdb_utils.py — Protein Data Bank Utilities
============================================
Fetches PDB structures from RCSB and delegates
pocket detection to pocket_detection.py.
"""

import os, re, requests
DATA_DIR = "data"


def fetch_pdb(pdb_id: str) -> dict:
    pdb_id = pdb_id.strip().upper()
    os.makedirs(DATA_DIR, exist_ok=True)
    path   = os.path.join(DATA_DIR, f"{pdb_id}.pdb")
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            return {"pdb_id": pdb_id, "path": None,
                    "error": str(e), "status": "download_failed"}
    meta = _parse_pdb_metadata(path, pdb_id)
    meta["path"]   = path
    meta["status"] = "ok"
    return meta


def _parse_pdb_metadata(path: str, pdb_id: str) -> dict:
    title = ""; organism = ""; resolution = None
    chains = set(); residues = set()
    atom_count = 0; hetatm_count = 0; ligands = []
    try:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                tag = line[:6].strip()
                if tag == "TITLE" and not title:
                    title = line[10:].strip()
                elif tag == "SOURCE" and "ORGANISM_SCIENTIFIC" in line:
                    organism = line.split(":")[-1].strip().rstrip(";")
                elif tag == "REMARK" and "RESOLUTION." in line:
                    m = re.search(r"(\d+\.\d+)\s+ANGSTROMS", line)
                    if m: resolution = float(m.group(1))
                elif tag == "ATOM":
                    atom_count += 1
                    chains.add(line[21])
                    residues.add((line[17:20].strip(), line[22:26].strip()))
                elif tag == "HETATM":
                    hetatm_count += 1
                    lig = line[17:20].strip()
                    if lig not in ("HOH","WAT","MG","CA","ZN","NA","CL") and lig not in ligands:
                        ligands.append(lig)
    except Exception as e:
        logger.debug(f"PDB parse warning for {pdb_id}: {e}")
    return {
        "pdb_id":        pdb_id,
        "title":         (title[:80] or f"Protein {pdb_id}"),
        "organism":      organism or "Unknown",
        "resolution_A":  resolution,
        "chains":        sorted(chains),
        "residue_count": len(residues),
        "atom_count":    atom_count,
        "hetatm_count":  hetatm_count,
        "ligands_found": ligands[:5],
    }


def get_binding_site(pdb_id: str) -> dict:
    """Delegate to pocket_detection module."""
    from modules.pocket_detection import get_binding_pocket
    return get_binding_pocket(pdb_id.upper())
