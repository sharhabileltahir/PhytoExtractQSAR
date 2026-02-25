#!/usr/bin/env python3
"""
Phase 2E: Targeted PubMed Mining for TFC & Yield Data
=======================================================

Run: python scripts/02e_mine_tfc_yield.py

Runs highly specific PubMed queries designed to find papers
reporting TFC (mg QE/g) and extraction yield (%) values.

Strategy:
  - TFC queries target flavonoid quantification studies
  - Yield queries target optimization studies with RSM/CCD
  - Compound-specific queries for top compounds in dataset
  - Method-specific queries for UAE, MAE, SFE with yield reporting
"""

import os
import sys
import time
import json
import re
import logging
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
from Bio import Entrez

sys.path.insert(0, str(Path(__file__).parent))
from _workflow_utils import load_env_file

# Load .env — adjust path if needed
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_env_file(str(env_path))
else:
    load_env_file()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('logs/02e_mine_tfc_yield.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

Entrez.email = os.getenv('NCBI_EMAIL', '')
Entrez.api_key = os.getenv('NCBI_API_KEY', '')

if not Entrez.email:
    logger.error("NCBI_EMAIL not set.")
    sys.exit(1)


# =============================================================================
# TARGETED QUERIES
# =============================================================================

TARGETED_QUERIES = {
    # --- TFC-SPECIFIC QUERIES ---
    'tfc_flavonoid_content': '"total flavonoid content" AND ("mg QE" OR "mg quercetin equivalent" OR "mg CE" OR "mg catechin equivalent")',
    'tfc_quantification': '"flavonoid" AND "quantification" AND ("extraction" OR "solvent") AND ("mg/g" OR "mg QE/g")',
    'tfc_uae': '"ultrasound" AND "flavonoid" AND ("total flavonoid" OR "TFC") AND "extraction"',
    'tfc_mae': '"microwave" AND "flavonoid" AND ("total flavonoid" OR "TFC") AND "extraction"',
    'tfc_optimization_rsm': '"response surface" AND "flavonoid" AND ("TFC" OR "total flavonoid content")',
    'tfc_hplc_flavonoid': '"HPLC" AND "flavonoid" AND "extraction" AND ("yield" OR "content" OR "mg/g")',
    'tfc_phenolic_flavonoid': '"total phenolic" AND "total flavonoid" AND "extraction" AND "antioxidant"',
    'tfc_plant_extract': '"plant extract" AND "total flavonoid" AND ("optimization" OR "response surface")',

    # --- YIELD-SPECIFIC QUERIES ---
    'yield_rsm_optimization': '"response surface methodology" AND "extraction yield" AND ("Box-Behnken" OR "central composite")',
    'yield_uae_optimization': '"ultrasound-assisted extraction" AND "yield" AND "optimization" AND ("RSM" OR "response surface")',
    'yield_mae_optimization': '"microwave-assisted extraction" AND "yield" AND "optimization"',
    'yield_sfe_optimization': '"supercritical" AND "extraction yield" AND "optimization" AND "CO2"',
    'yield_ple_optimization': '"pressurized liquid" AND "extraction" AND "yield" AND "optimization"',
    'yield_eae_optimization': '"enzyme-assisted" AND "extraction" AND "yield" AND "optimization"',
    'yield_soxhlet_comparison': '"Soxhlet" AND "extraction yield" AND ("comparison" OR "ultrasound" OR "microwave")',
    'yield_phenolic_extraction': '"phenolic" AND "extraction yield" AND ("optimization" OR "RSM")',
    'yield_essential_oil': '"essential oil" AND "extraction yield" AND ("hydrodistillation" OR "steam distillation")',
    'yield_alkaloid': '"alkaloid" AND "extraction" AND "yield" AND ("optimization" OR "solvent")',

    # --- COMPOUND-SPECIFIC (top compounds needing more data) ---
    'compound_quercetin': '"quercetin" AND "extraction" AND ("yield" OR "content" OR "mg/g") AND "optimization"',
    'compound_curcumin': '"curcumin" AND "extraction" AND ("yield" OR "content") AND ("optimization" OR "RSM")',
    'compound_catechin': '"catechin" AND "extraction" AND ("yield" OR "total flavonoid") AND "optimization"',
    'compound_gallic_acid': '"gallic acid" AND "extraction" AND ("yield" OR "content") AND ("optimization" OR "RSM")',
    'compound_rutin': '"rutin" AND "extraction" AND ("yield" OR "content" OR "mg/g")',
    'compound_kaempferol': '"kaempferol" AND "extraction" AND ("yield" OR "flavonoid")',
    'compound_resveratrol': '"resveratrol" AND "extraction" AND ("yield" OR "content") AND "optimization"',
    'compound_thymol': '"thymol" AND "extraction" AND ("yield" OR "essential oil")',
    'compound_eugenol': '"eugenol" AND "extraction" AND ("yield" OR "essential oil" OR "content")',
    'compound_berberine': '"berberine" AND "extraction" AND ("yield" OR "content" OR "alkaloid")',

    # --- PLANT-SPECIFIC HIGH-VALUE ---
    'plant_green_tea': '"green tea" AND "extraction" AND ("catechin" OR "flavonoid") AND ("yield" OR "content")',
    'plant_turmeric': '"turmeric" AND "curcuminoid" AND "extraction" AND ("yield" OR "optimization")',
    'plant_rosemary': '"rosemary" AND "extraction" AND ("yield" OR "phenolic" OR "carnosic")',
    'plant_olive_leaf': '"olive leaf" AND "extraction" AND ("oleuropein" OR "phenolic") AND "yield"',
    'plant_grape': '"grape" AND ("pomace" OR "seed" OR "skin") AND "extraction" AND ("phenolic" OR "anthocyanin") AND "yield"',

    # --- METHOD COMPARISON STUDIES (rich in multi-parameter data) ---
    'comparison_uae_mae': '("UAE" OR "ultrasound") AND ("MAE" OR "microwave") AND "extraction" AND "comparison" AND "yield"',
    'comparison_green_extraction': '"green extraction" AND ("yield" OR "phenolic" OR "flavonoid") AND "optimization"',
    'comparison_deep_eutectic': '"deep eutectic solvent" AND "extraction" AND ("yield" OR "phenolic" OR "flavonoid")',
}


# =============================================================================
# MINING FUNCTIONS (reused from 02_mine_pubmed.py)
# =============================================================================

YIELD_PATTERN = re.compile(
    r'(?:extraction\s+)?yield[s]?\s*(?:of|was|were|=|:)?\s*'
    r'(\d+\.?\d*)\s*(?:%|percent|g/\s*100\s*g)',
    re.IGNORECASE
)

TPC_PATTERN = re.compile(
    r'(?:total\s+phenolic\s+content|TPC)\s*(?:of|was|were|=|:)?\s*'
    r'(\d+\.?\d*)\s*(?:mg\s*GAE|mg\s*gallic)',
    re.IGNORECASE
)

TFC_PATTERN = re.compile(
    r'(?:total\s+flavonoid\s+content|TFC)\s*(?:of|was|were|=|:)?\s*'
    r'(\d+\.?\d*)\s*(?:mg\s*(?:QE|quercetin|CE|catechin|RE|rutin))',
    re.IGNORECASE
)

IC50_PATTERN = re.compile(
    r'IC50\s*(?:value|of|was|were|=|:)?\s*(\d+\.?\d*)\s*(?:\u00b5g/mL|ug/mL|mcg/mL|mg/mL)',
    re.IGNORECASE
)

TEMP_PATTERN = re.compile(
    r'(?:temperature|temp\.?)\s*(?:of|was|at|=|:)?\s*(\d+\.?\d*)\s*(?:\u00b0C|degrees?\s*C|C\b)',
    re.IGNORECASE
)

TIME_PATTERN = re.compile(
    r'(?:time|duration|period)\s*(?:of|was|for|=|:)?\s*(\d+\.?\d*)\s*(?:min|minutes?|h|hours?)',
    re.IGNORECASE
)

SOLVENT_PATTERN = re.compile(
    r'(?:using|with|in|solvent)\s+((?:methanol|ethanol|water|acetone|hexane|'
    r'ethyl acetate|chloroform|DMSO|isopropanol|n-butanol|petroleum ether|'
    r'dichloromethane|acetonitrile|CO2|supercritical)'
    r'(?:\s*(?::|/)\s*(?:water|methanol|ethanol))?'
    r'(?:\s*\(\d+[:%]\d*\))?)',
    re.IGNORECASE
)

DOI_PATTERN = re.compile(r'(10\.\d{4,}/[^\s]+)')


def extract_params(text: str) -> dict:
    """Extract extraction parameters from abstract text."""
    params = {}

    m = YIELD_PATTERN.search(text)
    if m:
        val = float(m.group(1))
        if 0.01 <= val <= 100:
            params['yield'] = val

    m = TPC_PATTERN.search(text)
    if m:
        val = float(m.group(1))
        if 0.01 <= val <= 1000:
            params['tpc'] = val

    m = TFC_PATTERN.search(text)
    if m:
        val = float(m.group(1))
        if 0.01 <= val <= 1000:
            params['tfc'] = val

    m = IC50_PATTERN.search(text)
    if m:
        val = float(m.group(1))
        if 0.001 <= val <= 10000:
            params['ic50'] = val

    m = TEMP_PATTERN.search(text)
    if m:
        val = float(m.group(1))
        if -20 <= val <= 300:
            params['temperature'] = val

    m = TIME_PATTERN.search(text)
    if m:
        val = float(m.group(1))
        time_text = m.group(0).lower()
        if 'hour' in time_text or time_text.endswith('h'):
            val *= 60  # Convert to minutes
        if 0.1 <= val <= 10080:
            params['time_min'] = val

    m = SOLVENT_PATTERN.search(text)
    if m:
        params['solvent'] = m.group(1).strip()

    return params


def search_pubmed(query: str, max_results: int = 200) -> list:
    """Search PubMed and extract parameters from abstracts."""
    records = []

    try:
        handle = Entrez.esearch(db='pubmed', term=query,
                                retmax=max_results, sort='relevance')
        result = Entrez.read(handle)
        handle.close()

        pmids = result.get('IdList', [])
        if not pmids:
            return records

        # Fetch abstracts in batches
        for i in range(0, len(pmids), 50):
            batch = pmids[i:i+50]

            try:
                handle = Entrez.efetch(db='pubmed', id=','.join(batch),
                                       rettype='xml', retmode='xml')
                articles = Entrez.read(handle)
                handle.close()

                for article in articles.get('PubmedArticle', []):
                    try:
                        medline = article['MedlineCitation']
                        pmid = str(medline['PMID'])
                        art = medline['Article']

                        title = str(art.get('ArticleTitle', ''))
                        abstract_parts = art.get('Abstract', {}).get('AbstractText', [])
                        abstract = ' '.join(str(p) for p in abstract_parts)

                        # Extract DOI
                        doi = ''
                        for aid in art.get('ELocationID', []):
                            if str(aid.attributes.get('EIdType', '')) == 'doi':
                                doi = str(aid)

                        if not doi:
                            for ref in article.get('PubmedData', {}).get('ArticleIdList', []):
                                if str(ref.attributes.get('IdType', '')) == 'doi':
                                    doi = str(ref)

                        # Extract parameters
                        full_text = f"{title} {abstract}"
                        params = extract_params(full_text)

                        # Journal info
                        journal = art.get('Journal', {}).get('Title', '')
                        pub_date = art.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
                        year = pub_date.get('Year', '')

                        record = {
                            'pmid': pmid,
                            'doi': doi,
                            'title': title,
                            'journal': journal,
                            'year': year,
                            'abstract_length': len(abstract),
                            'has_yield': 'yield' in params,
                            'has_tpc': 'tpc' in params,
                            'has_tfc': 'tfc' in params,
                            'has_ic50': 'ic50' in params,
                            **{f'extracted_{k}': v for k, v in params.items()},
                        }
                        records.append(record)

                    except Exception:
                        continue

                time.sleep(0.35)

            except Exception as e:
                logger.warning(f"Fetch error: {e}")
                time.sleep(1)

    except Exception as e:
        logger.error(f"Search error for query: {e}")

    return records


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 60)
    logger.info("TARGETED MINING: TFC & Yield Data")
    logger.info("=" * 60)

    all_records = []
    query_stats = {}

    for label, query in TARGETED_QUERIES.items():
        logger.info(f"\n[{label}] Searching...")
        records = search_pubmed(query, max_results=200)

        # Count useful records
        has_tfc = sum(1 for r in records if r.get('has_tfc'))
        has_yield = sum(1 for r in records if r.get('has_yield'))
        has_tpc = sum(1 for r in records if r.get('has_tpc'))

        query_stats[label] = {
            'total': len(records),
            'has_tfc': has_tfc,
            'has_yield': has_yield,
            'has_tpc': has_tpc,
        }

        for r in records:
            r['query_label'] = label

        all_records.extend(records)
        logger.info(f"  Found: {len(records)} papers "
                    f"(TFC: {has_tfc}, Yield: {has_yield}, TPC: {has_tpc})")

        time.sleep(0.5)

    # Deduplicate by PMID
    df_new = pd.DataFrame(all_records)
    if len(df_new) > 0:
        df_new = df_new.drop_duplicates(subset='pmid', keep='first')

    logger.info(f"\nTotal unique records mined: {len(df_new)}")

    # Save targeted results
    output_path = Path('data/processed/mined_tfc_yield_targeted.xlsx')
    df_new.to_excel(str(output_path), index=False)

    # Merge with existing mined data
    existing_path = Path('data/processed/mined_pubmed_data.xlsx')
    if existing_path.exists():
        df_existing = pd.read_excel(existing_path)
        logger.info(f"Existing mined records: {len(df_existing)}")

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset='pmid', keep='first')

        df_combined.to_excel(str(existing_path), index=False)
        logger.info(f"Combined total: {len(df_combined)} unique records")
        new_added = len(df_combined) - len(df_existing)
        logger.info(f"New records added: {new_added}")
    else:
        df_new.to_excel(str(existing_path), index=False)
        new_added = len(df_new)

    # Stats summary
    total_tfc = df_new['has_tfc'].sum() if 'has_tfc' in df_new.columns else 0
    total_yield = df_new['has_yield'].sum() if 'has_yield' in df_new.columns else 0
    total_tpc = df_new['has_tpc'].sum() if 'has_tpc' in df_new.columns else 0

    report = f"""
{'='*60}
TARGETED MINING REPORT: TFC & Yield
{'='*60}
Queries executed:     {len(TARGETED_QUERIES)}
Total records mined:  {len(df_new)}
New records added:    {new_added}

Records with TFC:    {total_tfc}
Records with Yield:  {total_yield}
Records with TPC:    {total_tpc}

Output: {output_path}
Updated: {existing_path}

NEXT STEPS:
  1. Download new PDFs:  python scripts/02d_retry_pdf_downloads.py --reset-failed
  2. Mine PDFs:          python scripts/03_mine_pdfs.py
  3. Replace data:       python scripts/04_replace_synthetic.py
  4. Run QSAR v2:        python scripts/09b_qsar_verified_only.py
{'='*60}
"""
    logger.info(report)

    with open('logs/02e_targeted_mining_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # Save query stats
    pd.DataFrame(query_stats).T.to_excel(
        'data/processed/targeted_query_statistics.xlsx'
    )


if __name__ == '__main__':
    main()
