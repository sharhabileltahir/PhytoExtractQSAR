#!/usr/bin/env python3
"""
Phase 2C: Automated PDF Downloader for Mined Papers
Run: python scripts/02c_download_pdfs.py

Downloads full-text PDFs from legal open-access sources:
  1. PubMed Central (PMC) â€” free full text for ~6M articles
  2. Unpaywall API â€” finds legal open-access versions
  3. DOI direct resolution â€” follows publisher redirects for OA papers
  4. Europe PMC â€” additional OA repository
  5. Semantic Scholar â€” OA PDF links

Requirements:
  pip install requests tqdm pandas

NOTE: This script ONLY downloads from legal open-access sources.
      For paywalled papers, use your university library proxy.
"""

import os
import sys
import time
import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from urllib.parse import quote

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _workflow_utils import load_env_file

os.environ.update(load_env_file(Path(__file__).resolve().parents[1] / ".env"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('logs/02c_download_pdfs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

EMAIL = os.getenv('NCBI_EMAIL', '')
API_KEY = os.getenv('NCBI_API_KEY', '')

# Output directory for downloaded PDFs
PDF_DIR = Path('data/pdfs')
PDF_DIR.mkdir(parents=True, exist_ok=True)

# Track download attempts to avoid retrying failures
DOWNLOAD_LOG = Path('logs/pdf_download_tracker.json')

# Request session with retry
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': f'PhytochemPipeline/1.0 (mailto:{EMAIL})',
    'Accept': 'application/pdf,text/html,application/xhtml+xml',
})


# =============================================================================
# 1. PMID â†’ PMCID CONVERTER
# =============================================================================

def pmids_to_pmcids(pmids: List[str]) -> Dict[str, str]:
    """Convert PMIDs to PMCIDs using NCBI ID Converter API."""
    pmcid_map = {}

    # Process in batches of 200
    for i in range(0, len(pmids), 200):
        batch = pmids[i:i+200]
        ids_str = ','.join(batch)

        try:
            url = 'https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/'
            params = {
                'ids': ids_str,
                'format': 'json',
                'tool': 'phytochem_pipeline',
                'email': EMAIL,
            }
            resp = SESSION.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for record in data.get('records', []):
                pmid = record.get('pmid', '')
                pmcid = record.get('pmcid', '')
                if pmid and pmcid:
                    pmcid_map[pmid] = pmcid

            time.sleep(0.4)

        except Exception as e:
            logger.warning(f"PMID->PMCID batch conversion error: {e}")

    logger.info(f"Converted {len(pmcid_map)}/{len(pmids)} PMIDs to PMCIDs")
    return pmcid_map


# =============================================================================
# 2. DOWNLOAD STRATEGIES
# =============================================================================

def download_from_pmc(pmcid: str, output_path: Path) -> bool:
    """Download PDF from PubMed Central."""
    try:
        # PMC OA Service
        url = f'https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/'
        resp = SESSION.get(url, timeout=60, allow_redirects=True)

        if resp.status_code == 200 and 'application/pdf' in resp.headers.get('Content-Type', ''):
            output_path.write_bytes(resp.content)
            return True

        # Alternative: PMC FTP-style URL
        url2 = f'https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf'
        resp2 = SESSION.get(url2, timeout=60)

        if resp2.status_code == 200 and len(resp2.content) > 10000:
            if resp2.content[:5] == b'%PDF-':
                output_path.write_bytes(resp2.content)
                return True

    except Exception as e:
        logger.debug(f"PMC download failed for {pmcid}: {e}")

    return False


def download_from_unpaywall(doi: str, output_path: Path) -> bool:
    """Find and download OA PDF via Unpaywall API."""
    if not doi or not EMAIL:
        return False

    try:
        url = f'https://api.unpaywall.org/v2/{quote(doi, safe="")}?email={EMAIL}'
        resp = SESSION.get(url, timeout=30)

        if resp.status_code != 200:
            return False

        data = resp.json()

        # Try best_oa_location first, then all oa_locations
        locations = []
        if data.get('best_oa_location'):
            locations.append(data['best_oa_location'])
        locations.extend(data.get('oa_locations', []))

        for loc in locations:
            pdf_url = loc.get('url_for_pdf') or loc.get('url')
            if not pdf_url:
                continue

            try:
                pdf_resp = SESSION.get(pdf_url, timeout=60, allow_redirects=True)
                if (pdf_resp.status_code == 200 and
                    len(pdf_resp.content) > 10000 and
                    pdf_resp.content[:5] == b'%PDF-'):
                    output_path.write_bytes(pdf_resp.content)
                    return True
            except Exception:
                continue

    except Exception as e:
        logger.debug(f"Unpaywall failed for {doi}: {e}")

    return False


def download_from_doi_direct(doi: str, output_path: Path) -> bool:
    """Try direct DOI resolution â€” some publishers serve OA PDFs directly."""
    if not doi:
        return False

    try:
        # Some known OA publisher PDF patterns
        pdf_patterns = [
            # MDPI (molecules, foods, antioxidants â€” very common in phytochem)
            (r'10\.3390/', lambda d: f'https://www.mdpi.com/{d.split("/")[-1]}/pdf'),
            # PLoS ONE
            (r'10\.1371/', lambda d: f'https://journals.plos.org/plosone/article/file?id={d}&type=printable'),
            # Hindawi
            (r'10\.1155/', lambda d: f'https://downloads.hindawi.com/journals/{d.split("/")[-1]}.pdf'),
            # Frontiers
            (r'10\.3389/', lambda d: f'https://www.frontiersin.org/articles/{d}/pdf'),
        ]

        for pattern, url_builder in pdf_patterns:
            if re.match(pattern, doi):
                pdf_url = url_builder(doi)
                try:
                    resp = SESSION.get(pdf_url, timeout=60, allow_redirects=True)
                    if (resp.status_code == 200 and
                        len(resp.content) > 10000 and
                        resp.content[:5] == b'%PDF-'):
                        output_path.write_bytes(resp.content)
                        return True
                except Exception:
                    pass

        # Generic: follow DOI redirect and look for PDF link
        doi_url = f'https://doi.org/{doi}'
        resp = SESSION.get(doi_url, timeout=30, allow_redirects=True,
                          headers={'Accept': 'application/pdf'})

        if (resp.status_code == 200 and
            len(resp.content) > 10000 and
            resp.content[:5] == b'%PDF-'):
            output_path.write_bytes(resp.content)
            return True

    except Exception as e:
        logger.debug(f"DOI direct failed for {doi}: {e}")

    return False


def download_from_europe_pmc(pmid: str, output_path: Path) -> bool:
    """Try Europe PMC for OA full text."""
    try:
        # Check if full text is available
        url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmid}+AND+SRC:MED&format=json'
        resp = SESSION.get(url, timeout=30)

        if resp.status_code != 200:
            return False

        data = resp.json()
        results = data.get('resultList', {}).get('result', [])

        if not results:
            return False

        result = results[0]
        pmcid = result.get('pmcid', '')

        if pmcid and result.get('isOpenAccess') == 'Y':
            pdf_url = f'https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf'
            pdf_resp = SESSION.get(pdf_url, timeout=60)

            if (pdf_resp.status_code == 200 and
                len(pdf_resp.content) > 10000 and
                pdf_resp.content[:5] == b'%PDF-'):
                output_path.write_bytes(pdf_resp.content)
                return True

    except Exception as e:
        logger.debug(f"Europe PMC failed for PMID {pmid}: {e}")

    return False


def download_from_semantic_scholar(doi: str, output_path: Path) -> bool:
    """Try Semantic Scholar for OA PDF link."""
    if not doi:
        return False

    try:
        url = f'https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf'
        resp = SESSION.get(url, timeout=30)

        if resp.status_code != 200:
            return False

        data = resp.json()
        oa_pdf = data.get('openAccessPdf', {})
        pdf_url = oa_pdf.get('url', '')

        if pdf_url:
            pdf_resp = SESSION.get(pdf_url, timeout=60, allow_redirects=True)
            if (pdf_resp.status_code == 200 and
                len(pdf_resp.content) > 10000 and
                pdf_resp.content[:5] == b'%PDF-'):
                output_path.write_bytes(pdf_resp.content)
                return True

    except Exception as e:
        logger.debug(f"Semantic Scholar failed for {doi}: {e}")

    return False


# =============================================================================
# 3. ORCHESTRATOR
# =============================================================================

def sanitize_filename(text: str, max_len: int = 80) -> str:
    """Create a safe filename from text."""
    safe = re.sub(r'[^\w\s-]', '', text)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:max_len]


def load_download_tracker() -> Dict:
    """Load tracking log of previous download attempts."""
    if DOWNLOAD_LOG.exists():
        with open(DOWNLOAD_LOG) as f:
            return json.load(f)
    return {'downloaded': {}, 'failed': {}, 'skipped': {}}


def save_download_tracker(tracker: Dict):
    """Save tracking log."""
    with open(DOWNLOAD_LOG, 'w') as f:
        json.dump(tracker, f, indent=2)


def attempt_download(pmid: str, doi: str, pmcid: str,
                     title: str, tracker: Dict) -> Tuple[bool, str]:
    """Try all download strategies for a single paper."""

    # Skip if already downloaded or recently failed
    key = pmid or doi
    if key in tracker['downloaded']:
        return True, 'already_downloaded'
    if key in tracker['failed']:
        return False, 'previously_failed'

    # Create filename
    safe_title = sanitize_filename(title) if title else key
    filename = f"PMID{pmid}_{safe_title}.pdf" if pmid else f"DOI_{sanitize_filename(doi)}.pdf"
    output_path = PDF_DIR / filename

    if output_path.exists():
        tracker['downloaded'][key] = str(output_path)
        return True, 'file_exists'

    # Strategy 1: PMC (highest success rate for OA papers)
    if pmcid:
        if download_from_pmc(pmcid, output_path):
            tracker['downloaded'][key] = str(output_path)
            return True, 'pmc'
        time.sleep(0.3)

    # Strategy 2: Unpaywall (finds OA versions across repositories)
    if doi:
        if download_from_unpaywall(doi, output_path):
            tracker['downloaded'][key] = str(output_path)
            return True, 'unpaywall'
        time.sleep(0.3)

    # Strategy 3: Direct DOI (works well for MDPI, PLoS, Hindawi, Frontiers)
    if doi:
        if download_from_doi_direct(doi, output_path):
            tracker['downloaded'][key] = str(output_path)
            return True, 'doi_direct'
        time.sleep(0.3)

    # Strategy 4: Europe PMC
    if pmid:
        if download_from_europe_pmc(pmid, output_path):
            tracker['downloaded'][key] = str(output_path)
            return True, 'europe_pmc'
        time.sleep(0.3)

    # Strategy 5: Semantic Scholar
    if doi:
        if download_from_semantic_scholar(doi, output_path):
            tracker['downloaded'][key] = str(output_path)
            return True, 'semantic_scholar'
        time.sleep(0.5)

    # All strategies failed â€” paper is likely paywalled
    tracker['failed'][key] = {'doi': doi, 'pmid': pmid, 'title': title}
    return False, 'all_failed'


# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    if not EMAIL:
        logger.error("NCBI_EMAIL not set. Configure .env file first.")
        return

    # Load mined data
    mined_path = Path('data/processed/mined_pubmed_data.xlsx')
    if not mined_path.exists():
        logger.error("No mined data found. Run Phase 2 first.")
        return

    df = pd.read_excel(mined_path)
    logger.info(f"Loaded {len(df)} mined records")

    # Get unique papers (by PMID)
    papers = df.drop_duplicates(subset='pmid', keep='first')
    papers = papers[papers['pmid'].notna()].copy()
    logger.info(f"Unique papers to process: {len(papers)}")

    # Convert PMIDs to PMCIDs (batch)
    pmids = papers['pmid'].astype(str).tolist()
    pmcid_map = pmids_to_pmcids(pmids)
    logger.info(f"Papers with PMC full text available: {len(pmcid_map)}")

    # Load tracker
    tracker = load_download_tracker()
    already = len(tracker['downloaded'])
    logger.info(f"Previously downloaded: {already}")

    # Download loop
    results = {'pmc': 0, 'unpaywall': 0, 'doi_direct': 0,
               'europe_pmc': 0, 'semantic_scholar': 0,
               'already_downloaded': 0, 'file_exists': 0,
               'all_failed': 0, 'previously_failed': 0}

    for _, row in tqdm(papers.iterrows(), total=len(papers), desc="Downloading PDFs"):
        pmid = str(int(row['pmid'])) if pd.notna(row['pmid']) else ''
        doi = str(row['doi']) if pd.notna(row['doi']) else ''
        title = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
        pmcid = pmcid_map.get(pmid, '')

        success, source = attempt_download(pmid, doi, pmcid, title, tracker)
        results[source] = results.get(source, 0) + 1

        # Save tracker periodically
        if sum(results.values()) % 20 == 0:
            save_download_tracker(tracker)

    # Final save
    save_download_tracker(tracker)

    # Count actual PDF files
    pdf_count = len(list(PDF_DIR.glob('*.pdf')))

    # Report
    report = f"""
{'='*60}
PDF DOWNLOAD REPORT
{'='*60}
Papers processed:        {len(papers)}
PMCIDs available:        {len(pmcid_map)}

Download results:
  PMC:                   {results.get('pmc', 0)}
  Unpaywall:             {results.get('unpaywall', 0)}
  DOI direct (MDPI etc): {results.get('doi_direct', 0)}
  Europe PMC:            {results.get('europe_pmc', 0)}
  Semantic Scholar:      {results.get('semantic_scholar', 0)}
  Already downloaded:    {results.get('already_downloaded', 0)}
  File exists:           {results.get('file_exists', 0)}
  Failed (paywalled):    {results.get('all_failed', 0)}
  Previously failed:     {results.get('previously_failed', 0)}

Total PDFs in data/pdfs/: {pdf_count}
Download tracker:         {DOWNLOAD_LOG}

NEXT STEPS:
  1. Run Phase 3 to mine downloaded PDFs:
     python scripts/03_mine_pdfs.py

  2. For paywalled papers, download manually via:
     - Alexandria University library proxy
     - Sci-Hub (sci-hub.se) â€” paste DOIs
     - Author ResearchGate profiles

  3. Paywalled DOIs saved in: {DOWNLOAD_LOG}
     (check the 'failed' section)
{'='*60}
"""
    logger.info(report)

    with open('logs/02c_pdf_download_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # Save list of failed DOIs for manual download
    failed_dois = Path('data/processed/paywalled_dois_for_manual_download.txt')
    with open(failed_dois, 'w', encoding='utf-8') as f:
        f.write("# Paywalled papers â€” download manually via university library or Sci-Hub\n")
        f.write("# Paste DOIs at https://sci-hub.se or use your library proxy\n\n")
        for key, info in tracker['failed'].items():
            doi = info.get('doi', '')
            title = info.get('title', '')[:80]
            if doi:
                f.write(f"https://doi.org/{doi}  # {title}\n")

    logger.info(f"Paywalled DOIs list: {failed_dois}")


if __name__ == '__main__':
    main()
