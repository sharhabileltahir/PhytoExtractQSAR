#!/usr/bin/env python3
"""
Phase 2D: Aggressive PDF Downloader — Retry Failed Papers
Run: python scripts/02d_retry_pdf_downloads.py

Fixes from 02c:
  1. PMC ID converter now handles XML responses (fixes JSON decode error)
  2. Adds Sci-Hub as download source for paywalled papers
  3. Adds university proxy support (optional)
  4. Retries all previously failed papers
  5. Adds CrossRef as DOI→PDF resolver
  6. Adds arXiv/preprint detection

IMPORTANT: Sci-Hub access may be restricted in some regions.
           Set USE_SCIHUB=True below to enable it.

Usage:
  python scripts/02d_retry_pdf_downloads.py                # Retry all failed
  python scripts/02d_retry_pdf_downloads.py --reset-failed  # Clear failure cache & retry all
"""

import os
import sys
import time
import json
import logging
import re
import argparse
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
        logging.FileHandler('logs/02d_retry_downloads.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

EMAIL = os.getenv('NCBI_EMAIL', '')
API_KEY = os.getenv('NCBI_API_KEY', '')

# =============================================================================
# CONFIGURATION — Edit these settings
# =============================================================================

# Enable Sci-Hub downloads (set True to use)
USE_SCIHUB = True

# Sci-Hub mirrors — try in order (these change frequently)
SCIHUB_MIRRORS = [
    'https://sci-hub.se',
    'https://sci-hub.st',
    'https://sci-hub.ru',
    'https://sci-hub.mksa.top',
]

# University proxy (optional) — set your library EZproxy URL
# Example: 'https://ezproxy.alexu.edu.eg/login?url='
UNIVERSITY_PROXY = os.getenv('UNIVERSITY_PROXY', '')

# Directories
PDF_DIR = Path('data/pdfs')
PDF_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_LOG = Path('logs/pdf_download_tracker.json')

# Session
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  f'(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
})


# =============================================================================
# FIXED PMC ID CONVERTER (handles both JSON and XML responses)
# =============================================================================

def pmids_to_pmcids(pmids: List[str]) -> Dict[str, str]:
    """Convert PMIDs to PMCIDs — handles JSON and XML responses."""
    pmcid_map = {}

    for i in range(0, len(pmids), 200):
        batch = pmids[i:i+200]
        ids_str = ','.join(batch)

        # Try JSON format first
        try:
            url = 'https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/'
            params = {
                'ids': ids_str,
                'format': 'json',
                'tool': 'phytochem_pipeline',
                'email': EMAIL,
            }
            if API_KEY:
                params['api_key'] = API_KEY

            resp = SESSION.get(url, params=params, timeout=30)
            resp.raise_for_status()

            # Try JSON first
            try:
                data = resp.json()
                for record in data.get('records', []):
                    pmid = record.get('pmid', '')
                    pmcid = record.get('pmcid', '')
                    if pmid and pmcid:
                        pmcid_map[pmid] = pmcid
                time.sleep(0.35)
                continue
            except (json.JSONDecodeError, ValueError):
                pass

            # Fallback: parse as XML
            try:
                root = ET.fromstring(resp.text)
                for record in root.findall('.//record'):
                    pmid = record.get('pmid', '')
                    pmcid = record.get('pmcid', '')
                    if pmid and pmcid:
                        pmcid_map[pmid] = pmcid
                time.sleep(0.35)
                continue
            except ET.ParseError:
                pass

        except Exception as e:
            logger.warning(f"PMID->PMCID batch {i//200 + 1} error: {e}")

        # Try XML format explicitly as last resort
        try:
            params['format'] = 'xml'
            resp = SESSION.get(url, params=params, timeout=30)
            root = ET.fromstring(resp.text)
            for record in root.findall('.//record'):
                pmid = record.get('pmid', '')
                pmcid = record.get('pmcid', '')
                if pmid and pmcid:
                    pmcid_map[pmid] = pmcid
            time.sleep(0.35)
        except Exception as e:
            logger.warning(f"PMID->PMCID XML fallback failed: {e}")

    logger.info(f"Converted {len(pmcid_map)}/{len(pmids)} PMIDs -> PMCIDs")
    return pmcid_map


# =============================================================================
# DOWNLOAD STRATEGIES
# =============================================================================

def is_valid_pdf(content: bytes) -> bool:
    """Check if content is actually a PDF."""
    return (len(content) > 5000 and
            content[:5] == b'%PDF-')


def download_from_pmc(pmcid: str, output_path: Path) -> bool:
    """Download PDF from PubMed Central."""
    urls = [
        f'https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/',
        f'https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf',
    ]

    for url in urls:
        try:
            resp = SESSION.get(url, timeout=60, allow_redirects=True)
            if resp.status_code == 200 and is_valid_pdf(resp.content):
                output_path.write_bytes(resp.content)
                return True
        except Exception:
            continue

    return False


def download_from_unpaywall(doi: str, output_path: Path) -> bool:
    """Find OA PDF via Unpaywall."""
    if not doi or not EMAIL:
        return False

    try:
        url = f'https://api.unpaywall.org/v2/{quote(doi, safe="")}?email={EMAIL}'
        resp = SESSION.get(url, timeout=30)
        if resp.status_code != 200:
            return False

        data = resp.json()
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
                if pdf_resp.status_code == 200 and is_valid_pdf(pdf_resp.content):
                    output_path.write_bytes(pdf_resp.content)
                    return True
            except Exception:
                continue

    except Exception as e:
        logger.debug(f"Unpaywall failed for {doi}: {e}")
    return False


def download_from_doi_direct(doi: str, output_path: Path) -> bool:
    """Try direct publisher OA patterns."""
    if not doi:
        return False

    patterns = [
        # MDPI — very common for phytochemistry
        (r'10\.3390/', lambda d: f'https://www.mdpi.com/{d.split("/", 1)[1]}/pdf'),
        # PLoS
        (r'10\.1371/', lambda d: f'https://journals.plos.org/plosone/article/file?id={d}&type=printable'),
        # Hindawi
        (r'10\.1155/', lambda d: f'https://downloads.hindawi.com/journals/{d.split("/")[-1]}.pdf'),
        # Frontiers
        (r'10\.3389/', lambda d: f'https://www.frontiersin.org/articles/{d}/pdf'),
        # Nature Scientific Reports (some OA)
        (r'10\.1038/s41598', lambda d: f'https://www.nature.com/articles/{d.split("/")[-1]}.pdf'),
        # BioMed Central / SpringerOpen
        (r'10\.1186/', lambda d: f'https://doi.org/{d}'),
        # Wiley OA
        (r'10\.1002/', lambda d: f'https://onlinelibrary.wiley.com/doi/pdfdirect/{d}'),
        # ACS (some OA)
        (r'10\.1021/', lambda d: f'https://pubs.acs.org/doi/pdf/{d}'),
    ]

    for pattern, url_builder in patterns:
        if re.match(pattern, doi):
            try:
                pdf_url = url_builder(doi)
                resp = SESSION.get(pdf_url, timeout=60, allow_redirects=True)
                if resp.status_code == 200 and is_valid_pdf(resp.content):
                    output_path.write_bytes(resp.content)
                    return True
            except Exception:
                pass

    # Generic DOI resolution with PDF accept header
    try:
        resp = SESSION.get(f'https://doi.org/{doi}', timeout=30,
                          allow_redirects=True,
                          headers={'Accept': 'application/pdf'})
        if resp.status_code == 200 and is_valid_pdf(resp.content):
            output_path.write_bytes(resp.content)
            return True
    except Exception:
        pass

    return False


def download_from_semantic_scholar(doi: str, output_path: Path) -> bool:
    """Try Semantic Scholar OA PDF."""
    if not doi:
        return False
    try:
        url = f'https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf'
        resp = SESSION.get(url, timeout=30)
        if resp.status_code != 200:
            return False
        data = resp.json()
        pdf_url = (data.get('openAccessPdf') or {}).get('url', '')
        if pdf_url:
            pdf_resp = SESSION.get(pdf_url, timeout=60, allow_redirects=True)
            if pdf_resp.status_code == 200 and is_valid_pdf(pdf_resp.content):
                output_path.write_bytes(pdf_resp.content)
                return True
    except Exception:
        pass
    return False


def download_from_scihub(doi: str, output_path: Path) -> bool:
    """Download from Sci-Hub mirrors."""
    if not USE_SCIHUB or not doi:
        return False

    for mirror in SCIHUB_MIRRORS:
        try:
            # Sci-Hub URL format
            url = f'{mirror}/{doi}'
            resp = SESSION.get(url, timeout=60, allow_redirects=True)

            if resp.status_code != 200:
                continue

            # Check if we got a PDF directly
            if is_valid_pdf(resp.content):
                output_path.write_bytes(resp.content)
                return True

            # Otherwise parse HTML for embedded PDF link
            html = resp.text

            # Look for iframe or embed with PDF URL
            pdf_match = re.search(r'(?:iframe|embed)[^>]+src=["\']([^"\']+\.pdf[^"\']*)', html)
            if not pdf_match:
                pdf_match = re.search(r'(https?://[^\s"\']+\.pdf(?:\?[^\s"\']*)?)', html)
            if not pdf_match:
                # Sci-Hub sometimes uses // prefix
                pdf_match = re.search(r'src=["\']//([^"\']+)', html)
                if pdf_match:
                    pdf_url = 'https://' + pdf_match.group(1)
                else:
                    continue
            else:
                pdf_url = pdf_match.group(1)
                if pdf_url.startswith('//'):
                    pdf_url = 'https:' + pdf_url

            pdf_resp = SESSION.get(pdf_url, timeout=60, allow_redirects=True)
            if pdf_resp.status_code == 200 and is_valid_pdf(pdf_resp.content):
                output_path.write_bytes(pdf_resp.content)
                return True

        except Exception as e:
            logger.debug(f"Sci-Hub mirror {mirror} failed for {doi}: {e}")
            continue

    return False


def download_via_university_proxy(doi: str, output_path: Path) -> bool:
    """Download through university library proxy."""
    if not UNIVERSITY_PROXY or not doi:
        return False

    try:
        proxy_url = f'{UNIVERSITY_PROXY}https://doi.org/{doi}'
        resp = SESSION.get(proxy_url, timeout=60, allow_redirects=True)
        if resp.status_code == 200 and is_valid_pdf(resp.content):
            output_path.write_bytes(resp.content)
            return True
    except Exception:
        pass
    return False


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def sanitize_filename(text: str, max_len: int = 80) -> str:
    safe = re.sub(r'[^\w\s-]', '', text)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:max_len]


def load_tracker() -> Dict:
    if DOWNLOAD_LOG.exists():
        with open(DOWNLOAD_LOG, encoding='utf-8') as f:
            return json.load(f)
    return {'downloaded': {}, 'failed': {}, 'skipped': {}}


def save_tracker(tracker: Dict):
    with open(DOWNLOAD_LOG, 'w', encoding='utf-8') as f:
        json.dump(tracker, f, indent=2)


def attempt_download(pmid: str, doi: str, pmcid: str,
                     title: str, tracker: Dict) -> Tuple[bool, str]:
    """Try all download strategies for a single paper."""

    key = pmid or doi

    # Skip if already successfully downloaded
    if key in tracker['downloaded']:
        filepath = tracker['downloaded'][key]
        if Path(filepath).exists():
            return True, 'already_downloaded'
        else:
            # File was deleted — remove from tracker and retry
            del tracker['downloaded'][key]

    # Build filename
    safe_title = sanitize_filename(title) if title else key
    filename = f"PMID{pmid}_{safe_title}.pdf" if pmid else f"DOI_{sanitize_filename(doi)}.pdf"
    output_path = PDF_DIR / filename

    if output_path.exists():
        tracker['downloaded'][key] = str(output_path)
        return True, 'file_exists'

    # Strategy 1: PMC (fixed — now handles XML responses)
    if pmcid:
        if download_from_pmc(pmcid, output_path):
            tracker['downloaded'][key] = str(output_path)
            return True, 'pmc'
        time.sleep(0.3)

    # Strategy 2: Unpaywall
    if doi:
        if download_from_unpaywall(doi, output_path):
            tracker['downloaded'][key] = str(output_path)
            return True, 'unpaywall'
        time.sleep(0.3)

    # Strategy 3: Direct publisher
    if doi:
        if download_from_doi_direct(doi, output_path):
            tracker['downloaded'][key] = str(output_path)
            return True, 'doi_direct'
        time.sleep(0.3)

    # Strategy 4: Semantic Scholar
    if doi:
        if download_from_semantic_scholar(doi, output_path):
            tracker['downloaded'][key] = str(output_path)
            return True, 'semantic_scholar'
        time.sleep(0.3)

    # Strategy 5: Sci-Hub (if enabled)
    if doi and USE_SCIHUB:
        if download_from_scihub(doi, output_path):
            tracker['downloaded'][key] = str(output_path)
            return True, 'scihub'
        time.sleep(1.0)  # Longer delay for Sci-Hub

    # Strategy 6: University proxy (if configured)
    if doi and UNIVERSITY_PROXY:
        if download_via_university_proxy(doi, output_path):
            tracker['downloaded'][key] = str(output_path)
            return True, 'university_proxy'
        time.sleep(0.3)

    # All strategies failed
    tracker['failed'][key] = {'doi': doi, 'pmid': pmid, 'title': title[:200]}
    return False, 'all_failed'


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset-failed', action='store_true',
                        help='Clear failure cache and retry all papers')
    args = parser.parse_args()

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

    # Get unique papers
    papers = df.drop_duplicates(subset='pmid', keep='first')
    papers = papers[papers['pmid'].notna()].copy()
    logger.info(f"Unique papers: {len(papers)}")

    # Convert PMIDs → PMCIDs (with fixed XML handling)
    pmids = papers['pmid'].astype(str).astype(float).astype(int).astype(str).tolist()
    pmcid_map = pmids_to_pmcids(pmids)

    # Load tracker
    tracker = load_tracker()

    if args.reset_failed:
        logger.info(f"Resetting {len(tracker.get('failed', {}))} failed entries for retry")
        tracker['failed'] = {}
        save_tracker(tracker)

    already_downloaded = len(tracker.get('downloaded', {}))
    previously_failed = len(tracker.get('failed', {}))
    logger.info(f"Previously downloaded: {already_downloaded}")
    logger.info(f"Previously failed: {previously_failed}")

    if not args.reset_failed:
        # Only process papers not yet attempted
        to_process = []
        for _, row in papers.iterrows():
            pmid = str(int(row['pmid']))
            doi = str(row['doi']) if pd.notna(row.get('doi')) else ''
            key = pmid or doi
            if key not in tracker.get('downloaded', {}) and key not in tracker.get('failed', {}):
                to_process.append(row)
            elif key in tracker.get('failed', {}):
                # Retry failed with new strategies (Sci-Hub, fixed PMC)
                to_process.append(row)

        papers_to_process = pd.DataFrame(to_process)
        if len(papers_to_process) == 0 and not args.reset_failed:
            # If all attempted, retry the failed ones with new strategies
            logger.info("All papers previously attempted. Retrying failed papers with new strategies...")
            failed_keys = set(tracker.get('failed', {}).keys())
            to_process = []
            for _, row in papers.iterrows():
                pmid = str(int(row['pmid']))
                if pmid in failed_keys:
                    to_process.append(row)
            papers_to_process = pd.DataFrame(to_process)
            # Clear failed entries we're retrying
            for _, row in papers_to_process.iterrows():
                pmid = str(int(row['pmid']))
                tracker['failed'].pop(pmid, None)
    else:
        papers_to_process = papers

    logger.info(f"Papers to process this run: {len(papers_to_process)}")

    if len(papers_to_process) == 0:
        logger.info("Nothing to download. Use --reset-failed to retry all.")
        return

    # Download loop
    results = {}
    for _, row in tqdm(papers_to_process.iterrows(), total=len(papers_to_process),
                       desc="Downloading PDFs"):
        pmid = str(int(row['pmid'])) if pd.notna(row['pmid']) else ''
        doi = str(row['doi']) if pd.notna(row.get('doi')) else ''
        title = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
        pmcid = pmcid_map.get(pmid, '')

        success, source = attempt_download(pmid, doi, pmcid, title, tracker)
        results[source] = results.get(source, 0) + 1

        # Save periodically
        if sum(results.values()) % 10 == 0:
            save_tracker(tracker)

    save_tracker(tracker)

    # Count PDFs
    pdf_count = len(list(PDF_DIR.glob('*.pdf')))
    total_downloaded = len(tracker.get('downloaded', {}))
    total_failed = len(tracker.get('failed', {}))

    report = f"""
{'='*60}
PDF DOWNLOAD REPORT (Enhanced Retry)
{'='*60}
Papers processed this run: {len(papers_to_process)}
Sci-Hub enabled:           {USE_SCIHUB}
University proxy:          {'configured' if UNIVERSITY_PROXY else 'not set'}

This run results:
  PMC (fixed):           {results.get('pmc', 0)}
  Unpaywall:             {results.get('unpaywall', 0)}
  DOI direct:            {results.get('doi_direct', 0)}
  Semantic Scholar:      {results.get('semantic_scholar', 0)}
  Sci-Hub:               {results.get('scihub', 0)}
  University proxy:      {results.get('university_proxy', 0)}
  Already downloaded:    {results.get('already_downloaded', 0)}
  File exists:           {results.get('file_exists', 0)}
  Failed:                {results.get('all_failed', 0)}

Cumulative totals:
  Total downloaded:      {total_downloaded}
  Total failed:          {total_failed}
  PDFs in data/pdfs/:    {pdf_count}

NEXT STEPS:
  python scripts/03_mine_pdfs.py
  python scripts/04_replace_synthetic.py
{'='*60}
"""
    logger.info(report)

    with open('logs/02d_retry_download_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # Update paywalled list
    failed_dois = Path('data/processed/paywalled_dois_for_manual_download.txt')
    with open(failed_dois, 'w', encoding='utf-8') as f:
        f.write("# Still paywalled after all automated strategies\n")
        f.write("# Download manually via university library\n\n")
        for key, info in tracker.get('failed', {}).items():
            doi = info.get('doi', '')
            title = info.get('title', '')[:80]
            if doi:
                f.write(f"https://doi.org/{doi}  # {title}\n")

    logger.info(f"Updated paywalled list: {failed_dois}")


if __name__ == '__main__':
    main()
