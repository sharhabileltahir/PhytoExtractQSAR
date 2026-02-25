#!/usr/bin/env python3
"""Phase 1: verify compound identities against PubChem."""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import pubchempy as pcp

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from scripts._workflow_utils import (
    PROJECT_ROOT,
    ensure_parent_dir,
    load_extraction_dataset,
    normalize_nullable_text,
    resolve_extraction_columns,
)

LOG_PATH = PROJECT_ROOT / "logs" / "01_verify_compounds.log"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "compound_verification_report.xlsx"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")


def _extract_cas_from_synonyms(compound: pcp.Compound) -> str:
    synonyms = list(compound.synonyms or [])

    if not synonyms:
        try:
            synonym_records = pcp.get_synonyms(compound.cid, namespace="cid")
            if synonym_records and isinstance(synonym_records, list):
                first = synonym_records[0]
                synonyms = list(first.get("Synonym", []))
        except Exception:
            pass

    for synonym in synonyms:
        if CAS_PATTERN.match(str(synonym).strip()):
            return str(synonym).strip()

    return ""


def verify_compound(name: str, cas: str, smiles: str) -> Dict[str, object]:
    result: Dict[str, object] = {
        "original_name": name,
        "original_cas": cas,
        "original_smiles": smiles,
        "verified": False,
        "pubchem_cid": None,
        "verified_smiles": None,
        "verified_cas": None,
        "verified_mw": None,
        "verified_iupac": None,
        "discrepancy_notes": "",
    }

    query_plan = []
    if name:
        query_plan.append(("name", name))
    if cas:
        query_plan.append(("name", cas))
    if smiles:
        query_plan.append(("smiles", smiles))

    try:
        compounds = []
        for namespace, query in query_plan:
            compounds = pcp.get_compounds(query, namespace)
            if compounds:
                break

        if not compounds:
            result["discrepancy_notes"] = "NOT FOUND in PubChem"
            return result

        compound = compounds[0]
        verified_smiles = compound.isomeric_smiles or compound.canonical_smiles
        verified_cas = _extract_cas_from_synonyms(compound)

        result.update(
            {
                "verified": True,
                "pubchem_cid": compound.cid,
                "verified_smiles": verified_smiles,
                "verified_cas": verified_cas or None,
                "verified_mw": compound.molecular_weight,
                "verified_iupac": compound.iupac_name,
            }
        )

        notes = []
        if verified_smiles and smiles and verified_smiles.strip() != smiles.strip():
            notes.append("SMILES differs from PubChem")
        if verified_cas and cas and verified_cas.strip() != cas.strip():
            notes.append("CAS differs from PubChem")
        result["discrepancy_notes"] = "; ".join(notes)

    except Exception as exc:
        result["discrepancy_notes"] = f"Error: {exc}"

    return result


def main() -> None:
    logger.info("Loading extraction dataset")
    df = load_extraction_dataset()
    cols = resolve_extraction_columns(df.columns)

    if not cols["name"] or not cols["smiles"]:
        raise RuntimeError("Required columns not found: Phytochemical Name and/or SMILES")

    cas_col = cols["cas"]

    compounds = (
        df[[c for c in [cols["name"], cas_col, cols["smiles"]] if c is not None]]
        .drop_duplicates(subset=[cols["name"]])
        .reset_index(drop=True)
    )

    name_col = cols["name"]
    smiles_col = cols["smiles"]

    logger.info("Verifying %s unique compounds", len(compounds))

    results = []
    for idx, row in compounds.iterrows():
        name = normalize_nullable_text(row.get(name_col, ""))
        cas = normalize_nullable_text(row.get(cas_col, "")) if cas_col else ""
        smiles = normalize_nullable_text(row.get(smiles_col, ""))

        logger.info("[%s/%s] %s", idx + 1, len(compounds), name or "<missing name>")
        results.append(verify_compound(name=name, cas=cas, smiles=smiles))
        time.sleep(0.4)

    df_results = pd.DataFrame(results)
    ensure_parent_dir(OUTPUT_PATH)
    df_results.to_excel(OUTPUT_PATH, index=False)

    verified_count = int(df_results["verified"].sum()) if not df_results.empty else 0
    discrepancy_count = int(
        (df_results["discrepancy_notes"].fillna("").str.len() > 0).sum()
    )

    logger.info("=" * 60)
    logger.info("VERIFICATION COMPLETE")
    logger.info("Verified compounds: %s/%s", verified_count, len(compounds))
    logger.info("Records with notes: %s", discrepancy_count)
    logger.info("Output: %s", OUTPUT_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
