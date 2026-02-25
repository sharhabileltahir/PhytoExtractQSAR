#!/usr/bin/env python3
"""Phase 2: mine PubMed abstracts for extraction parameters."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from scripts._workflow_utils import PROJECT_ROOT, ensure_parent_dir, load_env_file

from scripts.extraction_data_mining_pipeline import ExtractionParameterMiner, PubMedSearcher

LOG_PATH = PROJECT_ROOT / "logs" / "02_mine_pubmed.log"
OUTPUT_XLSX = PROJECT_ROOT / "data" / "processed" / "mined_pubmed_data.xlsx"
OUTPUT_STATS = PROJECT_ROOT / "data" / "processed" / "mining_statistics.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

PRIORITY_COMPOUNDS = [
    "quercetin",
    "curcumin",
    "gallic acid",
    "caffeic acid",
    "resveratrol",
    "kaempferol",
    "rutin",
    "catechin",
    "epicatechin",
    "egcg",
    "luteolin",
    "apigenin",
    "naringenin",
    "myricetin",
    "chlorogenic acid",
    "ferulic acid",
    "rosmarinic acid",
    "berberine",
    "capsaicin",
    "piperine",
    "thymol",
    "carvacrol",
    "eugenol",
    "ursolic acid",
    "oleanolic acid",
    "betulinic acid",
    "silymarin",
    "withaferin a",
    "ellagic acid",
    "mangiferin",
]

METHODS = [
    "ultrasound-assisted extraction",
    "microwave-assisted extraction",
    "supercritical co2 extraction",
    "soxhlet extraction",
    "maceration",
    "pressurized liquid extraction",
    "hydrodistillation",
    "enzyme-assisted extraction",
    "deep eutectic solvent extraction",
]

OUTPUT_COLUMNS = [
    "compound",
    "doi",
    "pmid",
    "year",
    "title",
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
    "solid_liquid_ratio",
    "source",
    "confidence",
]


def build_queries(compound: str) -> List[str]:
    queries = [
        (
            f'("{compound}"[Title/Abstract]) AND '
            f'(extraction[Title/Abstract]) AND '
            f'(yield OR efficiency OR "total phenolic" OR TPC OR purity) AND '
            f'(temperature OR time OR solvent)'
        )
    ]

    for method in METHODS[:5]:
        queries.append(
            f'("{compound}"[Title/Abstract]) AND ("{method}"[Title/Abstract]) AND (yield OR efficiency)'
        )

    queries.append(
        (
            f'("{compound}"[Title/Abstract]) AND '
            f'(nanoparticle OR nanoformulation OR nanocarrier OR encapsulation) AND '
            f'(extraction OR preparation)'
        )
    )
    return queries


def _record_has_data(params: Dict[str, object]) -> bool:
    return any(
        [
            params.get("yield"),
            params.get("tpc"),
            params.get("tfc"),
            params.get("ic50"),
            params.get("temperature") and params.get("time"),
        ]
    )


def mine_single_compound(
    compound: str,
    searcher: PubMedSearcher,
    miner: ExtractionParameterMiner,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    seen_pmids = set()

    for query in build_queries(compound):
        try:
            pmids = searcher.search(query, max_results=50)
            new_pmids = [pmid for pmid in pmids if pmid not in seen_pmids]
            seen_pmids.update(new_pmids)

            if not new_pmids:
                continue

            articles = searcher.fetch_abstracts(new_pmids)
            for article in articles:
                text = f"{article.get('title', '')} {article.get('abstract', '')}"
                params = miner.extract_parameters(text)
                if not _record_has_data(params):
                    continue

                records.append(
                    {
                        "compound": compound,
                        "doi": article.get("doi", ""),
                        "pmid": article.get("pmid", ""),
                        "year": article.get("year", 0),
                        "title": article.get("title", ""),
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
                        "solid_liquid_ratio": params.get("solid_liquid_ratio", ""),
                        "source": "PubMed_abstract",
                        "confidence": "medium",
                    }
                )

            time.sleep(0.35)
        except Exception as exc:
            logger.error("Query failed for %s: %s", compound, exc)

    return records


def main() -> None:
    env_file_vars = load_env_file(PROJECT_ROOT / ".env")
    email = os.getenv("NCBI_EMAIL", env_file_vars.get("NCBI_EMAIL", ""))
    api_key = os.getenv("NCBI_API_KEY", env_file_vars.get("NCBI_API_KEY", ""))

    if not email:
        logger.warning("NCBI_EMAIL not set. Using placeholder email for requests.")
        email = "your.email@example.com"

    searcher = PubMedSearcher(email=email, api_key=api_key)
    miner = ExtractionParameterMiner()

    all_records: List[Dict[str, object]] = []
    per_compound: Dict[str, int] = {}

    logger.info("Starting PubMed mining for %s compounds", len(PRIORITY_COMPOUNDS))
    for idx, compound in enumerate(PRIORITY_COMPOUNDS, start=1):
        logger.info("[%s/%s] Mining %s", idx, len(PRIORITY_COMPOUNDS), compound)
        records = mine_single_compound(compound, searcher, miner)
        all_records.extend(records)
        per_compound[compound] = len(records)
        logger.info("Found %s extraction records", len(records))
        time.sleep(0.8)

    df = pd.DataFrame(all_records)
    if df.empty:
        df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Remove exact duplicates that can occur across overlapping queries.
    dedup_keys = ["compound", "pmid", "doi", "method", "temperature_c", "time_min", "yield_pct"]
    present_keys = [key for key in dedup_keys if key in df.columns]
    if present_keys:
        df = df.drop_duplicates(subset=present_keys, keep="first").reset_index(drop=True)

    ensure_parent_dir(OUTPUT_XLSX)
    df.to_excel(OUTPUT_XLSX, index=False)

    stats = {
        "total_records": int(len(df)),
        "compounds_searched": len(PRIORITY_COMPOUNDS),
        "records_with_yield": int(df["yield_pct"].notna().sum()) if "yield_pct" in df.columns else 0,
        "records_with_tpc": int(df["tpc"].notna().sum()) if "tpc" in df.columns else 0,
        "unique_dois": int(df["doi"].astype(str).replace("", pd.NA).dropna().nunique())
        if "doi" in df.columns
        else 0,
        "per_compound": per_compound,
    }
    ensure_parent_dir(OUTPUT_STATS)
    OUTPUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("PUBMED MINING COMPLETE")
    logger.info("Total records: %s", len(df))
    logger.info("Output: %s", OUTPUT_XLSX)
    logger.info("Stats: %s", OUTPUT_STATS)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
