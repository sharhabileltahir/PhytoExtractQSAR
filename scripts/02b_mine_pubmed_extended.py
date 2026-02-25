#!/usr/bin/env python3
"""
Phase 2B: Enhanced PubMed mining with optimization-study queries.
Run: python scripts/02b_mine_pubmed_extended.py

Targets:
- Response Surface Methodology (RSM) extraction optimization papers
- Box-Behnken / Central Composite Design studies
- Method-specific optimization studies
- Nanoformulation extraction studies (for QSAR relevance)
- Green extraction / deep eutectic solvent studies

These study types almost always report COMPLETE parameter sets
(temperature, time, solvent ratio, yield) in structured tables.
"""

import os
import sys
import time
import json
import logging
from typing import List, Dict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _workflow_utils import load_env_file
from extraction_data_mining_pipeline import PubMedSearcher, ExtractionParameterMiner

os.environ.update(load_env_file(Path(__file__).resolve().parents[1] / ".env"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('logs/02b_mine_pubmed_extended.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

EMAIL = os.getenv('NCBI_EMAIL', '')
API_KEY = os.getenv('NCBI_API_KEY', '')

# =============================================================================
# QUERY CATEGORIES
# =============================================================================

def get_optimization_queries() -> List[Dict[str, str]]:
    """RSM/DoE extraction optimization studies — richest parameter sources."""
    return [
        {"label": "RSM_extraction_general",
         "query": '("response surface methodology" OR "RSM") AND (extraction) AND (yield OR "total phenolic" OR TPC) AND (temperature AND time AND solvent)'},
        {"label": "BoxBehnken_extraction",
         "query": '("Box-Behnken" OR "Box Behnken") AND (extraction) AND (phytochemical OR phenolic OR flavonoid OR alkaloid)'},
        {"label": "CCD_extraction",
         "query": '("central composite design" OR "CCD") AND (extraction) AND (yield OR efficiency) AND (optimization)'},
        {"label": "DoE_extraction",
         "query": '("design of experiment" OR "factorial design") AND (extraction) AND (phytochemical OR polyphenol) AND (yield)'},
        {"label": "optimization_UAE",
         "query": '("ultrasound-assisted extraction" OR "UAE") AND (optimization OR "response surface") AND (yield) AND (temperature)'},
        {"label": "optimization_MAE",
         "query": '("microwave-assisted extraction" OR "MAE") AND (optimization OR "response surface") AND (yield) AND (temperature)'},
        {"label": "optimization_SFE",
         "query": '("supercritical fluid extraction" OR "supercritical CO2" OR "SFE") AND (optimization) AND (yield OR recovery) AND (pressure AND temperature)'},
        {"label": "optimization_PLE",
         "query": '("pressurized liquid extraction" OR "accelerated solvent extraction" OR "PLE" OR "ASE") AND (optimization) AND (yield) AND (temperature)'},
        {"label": "optimization_EAE",
         "query": '("enzyme-assisted extraction" OR "EAE") AND (optimization) AND (yield OR phenolic) AND (temperature AND time)'},
    ]


def get_method_specific_queries() -> List[Dict[str, str]]:
    """Method-focused queries for extraction parameters."""
    return [
        {"label": "Soxhlet_parameters",
         "query": '("Soxhlet extraction") AND (phytochemical OR phenolic OR flavonoid) AND (yield OR efficiency) AND (temperature OR time OR solvent)'},
        {"label": "maceration_parameters",
         "query": '(maceration) AND (extraction) AND (phenolic OR flavonoid OR alkaloid) AND (yield) AND (ethanol OR methanol OR water)'},
        {"label": "hydrodistillation_parameters",
         "query": '(hydrodistillation OR "hydro-distillation") AND (essential oil OR terpene OR monoterpene) AND (yield) AND (time)'},
        {"label": "steam_distillation_parameters",
         "query": '("steam distillation") AND (essential oil) AND (yield) AND (time OR temperature)'},
        {"label": "DES_extraction",
         "query": '("deep eutectic solvent" OR "DES" OR "NADES") AND (extraction) AND (phenolic OR flavonoid) AND (yield OR efficiency)'},
        {"label": "PEF_extraction",
         "query": '("pulsed electric field" OR "PEF") AND (extraction) AND (phytochemical OR polyphenol) AND (yield)'},
        {"label": "cold_pressing",
         "query": '("cold pressing" OR "cold press" OR "mechanical extraction") AND (oil OR phytochemical) AND (yield)'},
        {"label": "percolation_extraction",
         "query": '(percolation) AND (extraction) AND (alkaloid OR phenolic) AND (yield OR efficiency)'},
        {"label": "reflux_extraction",
         "query": '("reflux extraction" OR "heat reflux") AND (phytochemical OR phenolic) AND (yield) AND (temperature)'},
        {"label": "decoction_extraction",
         "query": '(decoction) AND (traditional OR herbal) AND (phenolic OR flavonoid) AND ("total phenolic" OR TPC OR yield)'},
    ]


def get_compound_class_queries() -> List[Dict[str, str]]:
    """Queries targeting specific phytochemical classes."""
    return [
        {"label": "flavonoid_extraction",
         "query": '(flavonoid OR flavone OR flavonol OR flavanone) AND (extraction) AND (yield OR recovery) AND (optimization OR parameters)'},
        {"label": "phenolic_acid_extraction",
         "query": '("phenolic acid" OR "hydroxycinnamic" OR "hydroxybenzoic") AND (extraction) AND (yield) AND (solvent OR temperature)'},
        {"label": "alkaloid_extraction",
         "query": '(alkaloid) AND (extraction) AND (yield OR recovery) AND (optimization OR parameters) AND (solvent)'},
        {"label": "terpene_extraction",
         "query": '(terpene OR terpenoid OR "essential oil") AND (extraction) AND (yield) AND (method OR optimization)'},
        {"label": "anthocyanin_extraction",
         "query": '(anthocyanin OR anthocyanidin) AND (extraction) AND (yield OR recovery) AND (temperature OR pH OR solvent)'},
        {"label": "carotenoid_extraction",
         "query": '(carotenoid OR "beta-carotene" OR lycopene OR lutein) AND (extraction) AND (yield) AND (solvent OR supercritical)'},
        {"label": "tannin_extraction",
         "query": '(tannin OR "tannic acid" OR "ellagic acid") AND (extraction) AND (yield OR TPC) AND (solvent OR temperature)'},
        {"label": "saponin_extraction",
         "query": '(saponin OR ginsenoside) AND (extraction) AND (yield) AND (optimization OR parameters)'},
        {"label": "coumarin_extraction",
         "query": '(coumarin OR scopoletin OR umbelliferone) AND (extraction) AND (yield OR recovery) AND (solvent)'},
        {"label": "lignan_extraction",
         "query": '(lignan OR sesamin OR podophyllotoxin) AND (extraction) AND (yield) AND (method OR optimization)'},
    ]


def get_nanoformulation_queries() -> List[Dict[str, str]]:
    """Queries for nanoformulation-relevant extraction data (QSAR relevance)."""
    return [
        {"label": "nano_phytochemical_extraction",
         "query": '(nanoparticle OR nanoformulation OR nanoencapsulation) AND (phytochemical OR polyphenol) AND (extraction) AND (yield OR encapsulation)'},
        {"label": "nano_curcumin",
         "query": '(curcumin) AND (nanoparticle OR nanoformulation) AND (extraction OR preparation) AND (yield OR encapsulation efficiency)'},
        {"label": "nano_quercetin",
         "query": '(quercetin) AND (nanoparticle OR nanocarrier) AND (extraction OR loading) AND (yield OR efficiency)'},
        {"label": "nano_resveratrol",
         "query": '(resveratrol) AND (nanoparticle OR nanoencapsulation) AND (extraction OR formulation) AND (yield)'},
        {"label": "gum_arabic_phytochemical",
         "query": '("gum arabic" OR "Acacia senegal" OR "acacia gum") AND (phytochemical OR polyphenol OR extract) AND (nanoparticle OR encapsulation OR formulation)'},
        {"label": "biopolymer_nanocarrier_extract",
         "query": '(biopolymer OR polysaccharide) AND (nanocarrier OR nanoparticle) AND (phytochemical OR phenolic) AND (extraction OR encapsulation)'},
    ]


def get_plant_specific_queries() -> List[Dict[str, str]]:
    """Queries targeting the 50 plant sources in the dataset."""
    # Focus on plants most represented and relevant to the research
    plants = [
        ("Curcuma longa", "turmeric"),
        ("Camellia sinensis", "green tea"),
        ("Nigella sativa", "black seed"),
        ("Moringa oleifera", "moringa"),
        ("Withania somnifera", "ashwagandha"),
        ("Punica granatum", "pomegranate"),
        ("Silybum marianum", "milk thistle"),
        ("Rosmarinus officinalis", "rosemary"),
        ("Thymus vulgaris", "thyme"),
        ("Zingiber officinale", "ginger"),
        ("Olea europaea", "olive"),
        ("Centella asiatica", "gotu kola"),
        ("Hippophae rhamnoides", "sea buckthorn"),
        ("Vaccinium myrtillus", "bilberry"),
        ("Acacia senegal", "gum arabic"),
    ]
    
    queries = []
    for latin, common in plants:
        queries.append({
            "label": f"plant_{common.replace(' ', '_')}",
            "query": f'("{latin}" OR "{common}") AND (extraction) AND (yield OR TPC OR "total phenolic") AND (temperature OR solvent OR optimization)'
        })
    
    return queries


def get_all_queries() -> List[Dict[str, str]]:
    """Combine all query categories."""
    all_q = []
    all_q.extend(get_optimization_queries())
    all_q.extend(get_method_specific_queries())
    all_q.extend(get_compound_class_queries())
    all_q.extend(get_nanoformulation_queries())
    all_q.extend(get_plant_specific_queries())
    return all_q


# =============================================================================
# MINING LOGIC
# =============================================================================

def mine_query(query_info: Dict, searcher: PubMedSearcher,
               miner: ExtractionParameterMiner,
               seen_pmids: set, max_results: int = 80) -> List[Dict]:
    """Mine a single query and return records."""
    records = []
    label = query_info['label']
    query = query_info['query']

    try:
        pmids = searcher.search(query, max_results=max_results)
        new_pmids = [p for p in pmids if p not in seen_pmids]
        seen_pmids.update(new_pmids)

        if not new_pmids:
            return records

        articles = searcher.fetch_abstracts(new_pmids)

        for article in articles:
            text = f"{article.get('title', '')} {article.get('abstract', '')}"
            params = miner.extract_parameters(text)

            has_data = any([
                params.get('yield'),
                params.get('tpc'),
                params.get('tfc'),
                params.get('ic50'),
                params.get('temperature') and params.get('time'),
            ])

            if has_data:
                record = {
                    'query_label': label,
                    'doi': article.get('doi', ''),
                    'pmid': article.get('pmid', ''),
                    'year': article.get('year', 0),
                    'title': article.get('title', ''),
                    'method': params.get('method', 'Unknown'),
                    'solvent': params.get('solvent', 'Unknown'),
                    'temperature_c': params.get('temperature'),
                    'time_min': params.get('time'),
                    'pressure_mpa': params.get('pressure'),
                    'power_w': params.get('power'),
                    'frequency_khz': params.get('frequency'),
                    'yield_pct': params.get('yield'),
                    'tpc': params.get('tpc'),
                    'tfc': params.get('tfc'),
                    'ic50': params.get('ic50'),
                    'solid_liquid_ratio': params.get('solid_liquid_ratio', ''),
                    'source': 'PubMed_abstract',
                    'confidence': 'medium',
                }
                records.append(record)

        time.sleep(0.4)

    except Exception as e:
        logger.error(f"Error in query '{label}': {e}")

    return records


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not EMAIL:
        logger.error("NCBI_EMAIL not set. Configure .env file first.")
        return

    searcher = PubMedSearcher(email=EMAIL, api_key=API_KEY)
    miner = ExtractionParameterMiner()

    all_queries = get_all_queries()
    logger.info(f"Running {len(all_queries)} enhanced queries...")

    # Load existing PMIDs to avoid duplicates with Phase 2
    seen_pmids = set()
    existing_path = Path('data/processed/mined_pubmed_data.xlsx')
    if existing_path.exists():
        df_existing = pd.read_excel(existing_path)
        if 'pmid' in df_existing.columns:
            seen_pmids = set(df_existing['pmid'].dropna().astype(str))
        logger.info(f"Loaded {len(seen_pmids)} existing PMIDs to skip")

    all_records = []
    query_stats = {}

    for i, q in enumerate(all_queries):
        logger.info(f"[{i+1}/{len(all_queries)}] {q['label']}")
        records = mine_query(q, searcher, miner, seen_pmids, max_results=80)
        all_records.extend(records)
        query_stats[q['label']] = len(records)

        if len(records) > 0:
            logger.info(f"  → {len(records)} records with data")
        time.sleep(0.5)

    # Save extended results
    df_extended = pd.DataFrame(all_records)
    output_path = 'data/processed/mined_pubmed_extended.xlsx'
    df_extended.to_excel(output_path, index=False)

    # Also merge with original Phase 2 data
    if existing_path.exists():
        df_existing = pd.read_excel(existing_path)
        df_combined = pd.concat([df_existing, df_extended], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset='pmid', keep='first')
        combined_path = 'data/processed/mined_pubmed_data.xlsx'
        df_combined.to_excel(combined_path, index=False)
        logger.info(f"Combined total: {len(df_combined)} records → {combined_path}")

    # Stats
    stats = {
        'total_new_records': len(all_records),
        'queries_run': len(all_queries),
        'new_unique_pmids': len(seen_pmids),
        'per_query': query_stats,
        'by_category': {
            'optimization': sum(v for k, v in query_stats.items() if k.startswith(('RSM', 'Box', 'CCD', 'DoE', 'optimization'))),
            'method_specific': sum(v for k, v in query_stats.items() if k.startswith(('Soxhlet', 'maceration', 'hydro', 'steam', 'DES', 'PEF', 'cold', 'percolation', 'reflux', 'decoction'))),
            'compound_class': sum(v for k, v in query_stats.items() if k.startswith(('flavonoid', 'phenolic_acid', 'alkaloid', 'terpene', 'anthocyanin', 'carotenoid', 'tannin', 'saponin', 'coumarin', 'lignan'))),
            'nanoformulation': sum(v for k, v in query_stats.items() if k.startswith(('nano', 'gum_arabic', 'biopolymer'))),
            'plant_specific': sum(v for k, v in query_stats.items() if k.startswith('plant_')),
        },
    }

    if len(df_extended) > 0:
        stats['records_with_yield'] = int(df_extended['yield_pct'].notna().sum())
        stats['records_with_tpc'] = int(df_extended['tpc'].notna().sum())
        stats['records_with_temperature'] = int(df_extended['temperature_c'].notna().sum())
        stats['records_with_time'] = int(df_extended['time_min'].notna().sum())
        stats['unique_dois'] = int(df_extended['doi'].nunique())

    stats_path = 'data/processed/mining_statistics_extended.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"EXTENDED MINING COMPLETE")
    logger.info(f"  New records mined: {len(all_records)}")
    logger.info(f"  Extended data: {output_path}")
    logger.info(f"  Combined data: mined_pubmed_data.xlsx")
    logger.info(f"  Stats: {stats_path}")
    logger.info(f"")
    logger.info(f"  By category:")
    for cat, count in stats.get('by_category', {}).items():
        logger.info(f"    {cat}: {count}")
    logger.info(f"")
    logger.info(f"  NEXT: Re-run Phase 4 to integrate new data")
    logger.info(f"    python scripts/04_replace_synthetic.py")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
