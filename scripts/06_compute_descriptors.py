#!/usr/bin/env python3
"""Phase 6: recompute molecular descriptors from verified SMILES with RDKit."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from scripts._workflow_utils import (
    PROJECT_ROOT,
    load_extraction_dataset,
    normalize_nullable_text,
    resolve_extraction_columns,
)

LOG_PATH = PROJECT_ROOT / "logs" / "06_compute_descriptors.log"
VERIFY_PATH = PROJECT_ROOT / "data" / "processed" / "compound_verification_report.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "molecular_descriptors_rdkit.xlsx"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "ID",
    "Phytochemical Name",
    "SMILES",
    "MW (g/mol)",
    "LogP (ALogP)",
    "TPSA (A2)",
    "HBD",
    "HBA",
    "Rotatable Bonds",
    "Aromatic Rings",
    "Ring Count",
    "Heavy Atom Count",
    "Fraction Csp3",
    "MR (Molar Refractivity)",
    "Num Heteroatoms",
    "Num H Acceptors",
    "Num H Donors",
    "Num Valence Electrons",
    "Exact MW",
    "BertzCT (Complexity)",
    "Chi0n",
    "Chi1n",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "PEOE_VSA1",
    "SlogP_VSA1",
    "Ro5 Violations",
    "QED Score",
    "Notes",
]


def compute_descriptors(smiles: str) -> Dict[str, object]:
    from rdkit import Chem  # pylint: disable=import-error
    from rdkit.Chem import Crippen, Descriptors, GraphDescriptors, Lipinski, QED, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    ro5_violations = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)

    return {
        "MW (g/mol)": round(mw, 2),
        "LogP (ALogP)": round(logp, 2),
        "TPSA (A2)": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "HBD": int(hbd),
        "HBA": int(hba),
        "Rotatable Bonds": int(Lipinski.NumRotatableBonds(mol)),
        "Aromatic Rings": int(Lipinski.NumAromaticRings(mol)),
        "Ring Count": int(Descriptors.RingCount(mol)),
        "Heavy Atom Count": int(mol.GetNumHeavyAtoms()),
        "Fraction Csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "MR (Molar Refractivity)": round(Crippen.MolMR(mol), 2),
        "Num Heteroatoms": int(Descriptors.NumHeteroatoms(mol)),
        "Num H Acceptors": int(hba),
        "Num H Donors": int(hbd),
        "Num Valence Electrons": int(Descriptors.NumValenceElectrons(mol)),
        "Exact MW": round(Descriptors.ExactMolWt(mol), 4),
        "BertzCT (Complexity)": round(GraphDescriptors.BertzCT(mol), 2),
        "Chi0n": round(GraphDescriptors.Chi0n(mol), 4),
        "Chi1n": round(GraphDescriptors.Chi1n(mol), 4),
        "Kappa1": round(GraphDescriptors.Kappa1(mol), 4),
        "Kappa2": round(GraphDescriptors.Kappa2(mol), 4),
        "Kappa3": round(GraphDescriptors.Kappa3(mol), 4),
        "LabuteASA": round(rdMolDescriptors.CalcLabuteASA(mol), 4),
        "PEOE_VSA1": round(Descriptors.PEOE_VSA1(mol), 4),
        "SlogP_VSA1": round(Descriptors.SlogP_VSA1(mol), 4),
        "Ro5 Violations": int(ro5_violations),
        "QED Score": round(QED.qed(mol), 4),
    }


def load_base_compounds() -> pd.DataFrame:
    replaced_path = PROJECT_ROOT / "data" / "processed" / "dataset_with_replacements.xlsx"

    if replaced_path.exists():
        df = pd.read_excel(replaced_path)
    else:
        df = load_extraction_dataset(PROJECT_ROOT / "data" / "raw" / "Phytochemical_Extraction_10K_Dataset.xlsx")

    cols = resolve_extraction_columns(df.columns)
    name_col = cols.get("name")
    smiles_col = cols.get("smiles")

    if not name_col or not smiles_col:
        raise RuntimeError("Required columns missing: Phytochemical Name and/or SMILES")

    compounds = (
        df[[name_col, smiles_col]]
        .dropna(subset=[name_col])
        .drop_duplicates(subset=[name_col])
        .reset_index(drop=True)
        .rename(columns={name_col: "Phytochemical Name", smiles_col: "SMILES"})
    )

    compounds["Phytochemical Name"] = compounds["Phytochemical Name"].map(normalize_nullable_text)
    compounds["SMILES"] = compounds["SMILES"].map(normalize_nullable_text)
    return compounds


def apply_verified_smiles(compounds: pd.DataFrame) -> pd.DataFrame:
    if not VERIFY_PATH.exists():
        return compounds

    verify_df = pd.read_excel(VERIFY_PATH)

    required_cols = {"original_name", "verified_smiles"}
    if not required_cols.issubset(set(verify_df.columns)):
        return compounds

    verified_map = (
        verify_df.dropna(subset=["original_name", "verified_smiles"])
        .set_index("original_name")["verified_smiles"]
        .to_dict()
    )

    compounds = compounds.copy()
    for idx, row in compounds.iterrows():
        name = row["Phytochemical Name"]
        if name in verified_map and str(verified_map[name]).strip():
            compounds.at[idx, "SMILES"] = str(verified_map[name]).strip()

    return compounds


def main() -> None:
    try:
        import rdkit  # pylint: disable=unused-import,import-error
    except ImportError as exc:
        raise RuntimeError("RDKit is required. Install with: pip install rdkit-pypi") from exc

    compounds = load_base_compounds()
    compounds = apply_verified_smiles(compounds)

    logger.info("Computing descriptors for %s compounds", len(compounds))

    rows: List[Dict[str, object]] = []
    for idx, row in compounds.iterrows():
        name = row["Phytochemical Name"]
        smiles = row["SMILES"]

        if not smiles:
            logger.warning("[%s] %s: missing SMILES", idx + 1, name)
            continue

        descriptor_values = compute_descriptors(smiles)
        if not descriptor_values:
            logger.warning("[%s] %s: RDKit parse failed", idx + 1, name)
            continue

        output_row: Dict[str, object] = {
            "ID": name,
            "Phytochemical Name": name,
            "SMILES": smiles,
            "Notes": "Computed via RDKit",
        }
        output_row.update(descriptor_values)

        rows.append(output_row)
        logger.info(
            "[%s/%s] %s: MW=%s LogP=%s",
            idx + 1,
            len(compounds),
            name,
            output_row["MW (g/mol)"],
            output_row["LogP (ALogP)"],
        )

    desc_df = pd.DataFrame(rows)
    if not desc_df.empty:
        for col in OUTPUT_COLUMNS:
            if col not in desc_df.columns:
                desc_df[col] = pd.NA
        desc_df = desc_df[OUTPUT_COLUMNS]
    else:
        desc_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    desc_df.to_excel(OUTPUT_PATH, index=False)

    logger.info("=" * 60)
    logger.info("DESCRIPTOR COMPUTATION COMPLETE")
    logger.info("Profiles written: %s", len(desc_df))
    logger.info("Output: %s", OUTPUT_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
