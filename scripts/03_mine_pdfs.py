#!/usr/bin/env python3
"""Phase 3: mine downloaded PDFs for extraction parameters."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pdfplumber

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from scripts._workflow_utils import PROJECT_ROOT, ensure_parent_dir

from scripts.extraction_data_mining_pipeline import ExtractionParameterMiner

LOG_PATH = PROJECT_ROOT / "logs" / "03_mine_pdfs.log"
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "mined_pdf_data.xlsx"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "compound",
    "compound_hint",
    "doi",
    "year",
    "source_file",
    "method",
    "solvent",
    "temperature_c",
    "time_min",
    "pressure_mpa",
    "power_w",
    "frequency_khz",
    "yield_pct",
    "tpc",
    "tfc",
    "ic50",
    "tables_found",
    "text_length",
    "source",
    "confidence",
]

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")


def infer_compound_hint(stem: str) -> str:
    cleaned = re.sub(r"[_\-]+", " ", stem).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def extract_doi(text: str) -> str:
    match = DOI_PATTERN.search(text)
    return match.group(0).rstrip(".);,") if match else ""


def extract_year(text: str) -> int:
    years = [int(match.group(1)) for match in YEAR_PATTERN.finditer(text)]
    if not years:
        return 0

    # Use the first plausible publication year in a modern range.
    for year in years:
        if 1990 <= year <= 2030:
            return year
    return years[0]


def extract_from_pdf(pdf_path: Path, miner: ExtractionParameterMiner) -> Dict[str, object]:
    result: Dict[str, object] = {
        "source_file": pdf_path.name,
        "compound_hint": infer_compound_hint(pdf_path.stem),
        "full_text": "",
        "tables": [],
        "parameters": {},
    }

    with pdfplumber.open(str(pdf_path)) as pdf:
        text_parts: List[str] = []
        tables: List[pd.DataFrame] = []

        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text:
                text_parts.append(page_text)

            for table in page.extract_tables() or []:
                if table and len(table) > 1:
                    try:
                        tables.append(pd.DataFrame(table[1:], columns=table[0]))
                    except Exception:
                        continue

    full_text = "\n".join(text_parts)
    params = miner.extract_parameters(full_text)

    # Parse table text as a secondary signal.
    for table in tables:
        table_text = table.to_string(index=False)
        table_params = miner.extract_parameters(table_text)
        for key, value in table_params.items():
            if key not in params and value not in (None, "", []):
                params[key] = value

    result["full_text"] = full_text
    result["tables"] = tables
    result["parameters"] = params
    return result


def main() -> None:
    if not PDF_DIR.exists():
        logger.error("PDF directory does not exist: %s", PDF_DIR)
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in %s", PDF_DIR)
        return

    miner = ExtractionParameterMiner()
    records: List[Dict[str, object]] = []

    logger.info("Processing %s PDFs", len(pdf_files))

    for idx, pdf_path in enumerate(pdf_files, start=1):
        logger.info("[%s/%s] %s", idx, len(pdf_files), pdf_path.name)
        try:
            result = extract_from_pdf(pdf_path, miner)
            params = result["parameters"]

            if not params:
                continue

            full_text = str(result["full_text"])
            record = {
                "compound": "",
                "compound_hint": result["compound_hint"],
                "doi": extract_doi(full_text),
                "year": extract_year(full_text),
                "source_file": result["source_file"],
                "method": params.get("method", "Unknown"),
                "solvent": params.get("solvent", "Unknown"),
                "temperature_c": params.get("temperature"),
                "time_min": params.get("time"),
                "pressure_mpa": params.get("pressure"),
                "power_w": params.get("power"),
                "frequency_khz": params.get("frequency"),
                "yield_pct": params.get("yield"),
                "tpc": params.get("tpc"),
                "tfc": params.get("tfc"),
                "ic50": params.get("ic50"),
                "tables_found": len(result["tables"]),
                "text_length": len(full_text),
                "source": "PDF_fulltext",
                "confidence": "high",
            }
            records.append(record)
        except Exception as exc:
            logger.error("Failed processing %s: %s", pdf_path.name, exc)

    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    ensure_parent_dir(OUTPUT_PATH)
    df.to_excel(OUTPUT_PATH, index=False)

    logger.info("=" * 60)
    logger.info("PDF MINING COMPLETE")
    logger.info("Records saved: %s", len(df))
    logger.info("Output: %s", OUTPUT_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
