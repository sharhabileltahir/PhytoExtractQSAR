#!/usr/bin/env python3
"""
=============================================================================
Phytochemical Extraction Data Mining Pipeline
=============================================================================
Purpose: Semi-automated extraction of phytochemical extraction parameters 
         from peer-reviewed literature to populate/enrich the dataset.

Author: For Sharhabil's MSc Biotechnology Research
Usage:  python extraction_data_mining_pipeline.py

Requirements:
    pip install requests pandas openpyxl rdkit-pypi pubchempy biopython pdfplumber
=============================================================================
"""

import re
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 1. DATA MODELS
# =============================================================================

@dataclass
class ExtractionRecord:
    """Single phytochemical extraction data point."""
    doi: str = ""
    year: int = 0
    phytochemical_name: str = ""
    phytochemical_class: str = ""
    smiles: str = ""
    cas_number: str = ""
    plant_source_latin: str = ""
    plant_part: str = ""
    plant_pretreatment: str = ""
    extraction_method: str = ""
    solvent_system: str = ""
    solvent_ratio: str = ""
    solvent_volume_ml_per_g: Optional[float] = None
    temperature_c: Optional[float] = None
    time_min: Optional[float] = None
    pressure_mpa: Optional[float] = None
    power_w: Optional[float] = None
    frequency_khz: Optional[float] = None
    solid_liquid_ratio: str = ""
    ph: Optional[float] = None
    number_of_cycles: Optional[int] = None
    yield_pct: Optional[float] = None
    purity_pct: Optional[float] = None
    tpc_mg_gae_per_g: Optional[float] = None
    tfc_mg_qe_per_g: Optional[float] = None
    antioxidant_ic50: Optional[float] = None
    scale: str = "Lab"
    notes: str = ""


# =============================================================================
# 2. PUBMED SEARCH MODULE
# =============================================================================

class PubMedSearcher:
    """Search PubMed for phytochemical extraction studies."""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, email: str, api_key: str = ""):
        self.email = email
        self.api_key = api_key
    
    def search(self, query: str, max_results: int = 100) -> List[str]:
        """Search PubMed and return PMIDs."""
        import requests
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        
        resp = requests.get(f"{self.BASE_URL}/esearch.fcgi", params=params)
        data = resp.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        logger.info(f"Found {len(pmids)} articles for query: {query[:60]}...")
        return pmids
    
    def fetch_abstracts(self, pmids: List[str]) -> List[Dict]:
        """Fetch article metadata and abstracts."""
        import requests
        
        articles = []
        # Process in batches of 50
        for i in range(0, len(pmids), 50):
            batch = pmids[i:i+50]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
                "email": self.email,
            }
            resp = requests.get(f"{self.BASE_URL}/efetch.fcgi", params=params)
            # Parse XML (simplified - use xml.etree for full implementation)
            articles.extend(self._parse_pubmed_xml(resp.text))
            time.sleep(0.35)  # Rate limiting
        
        return articles
    
    def _parse_pubmed_xml(self, xml_text: str) -> List[Dict]:
        """Parse PubMed XML response."""
        import xml.etree.ElementTree as ET
        
        articles = []
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall('.//PubmedArticle'):
                pmid = article.findtext('.//PMID', '')
                title = article.findtext('.//ArticleTitle', '')
                abstract = article.findtext('.//AbstractText', '')
                year = article.findtext('.//PubDate/Year', '')
                doi_elem = article.find('.//ArticleId[@IdType="doi"]')
                doi = doi_elem.text if doi_elem is not None else ''
                
                articles.append({
                    'pmid': pmid,
                    'title': title,
                    'abstract': abstract or '',
                    'year': int(year) if year.isdigit() else 0,
                    'doi': doi,
                })
        except ET.ParseError:
            logger.warning("XML parsing error")
        
        return articles
    
    @staticmethod
    def build_extraction_queries(compounds: List[str] = None) -> List[str]:
        """Generate targeted PubMed queries for extraction data."""
        
        methods = [
            "ultrasound-assisted extraction",
            "microwave-assisted extraction",
            "supercritical fluid extraction",
            "pressurized liquid extraction",
            "Soxhlet extraction",
            "maceration",
            "hydrodistillation",
            "enzyme-assisted extraction",
        ]
        
        base_queries = []
        for method in methods:
            q = (f'("{method}"[Title/Abstract]) AND '
                 f'(yield OR "extraction efficiency" OR "total phenolic") AND '
                 f'(temperature OR time OR solvent)')
            base_queries.append(q)
        
        if compounds:
            for comp in compounds[:20]:  # Limit to avoid too many queries
                q = f'("{comp}"[Title/Abstract]) AND (extraction AND (yield OR efficiency))'
                base_queries.append(q)
        
        return base_queries


# =============================================================================
# 3. TEXT MINING MODULE - Extract parameters from abstracts/full text
# =============================================================================

class ExtractionParameterMiner:
    """Extract numerical extraction parameters from text using regex patterns."""
    
    # Regex patterns for extraction parameters
    PATTERNS = {
        'temperature': [
            r'(?:temperature|temp\.?)\s*(?:of|at|was|=|:)?\s*(\d+(?:\.\d+)?)\s*°?\s*C',
            r'(\d+(?:\.\d+)?)\s*°\s*C',
            r'(\d+(?:\.\d+)?)\s*degrees?\s*(?:Celsius|C)',
        ],
        'time': [
            r'(?:time|duration)\s*(?:of|at|was|=|:)?\s*(\d+(?:\.\d+)?)\s*min',
            r'(\d+(?:\.\d+)?)\s*min(?:utes?)?',
            r'(\d+(?:\.\d+)?)\s*h(?:ours?)?\b',  # Will multiply by 60
        ],
        'pressure': [
            r'(?:pressure)\s*(?:of|at|was|=|:)?\s*(\d+(?:\.\d+)?)\s*MPa',
            r'(\d+(?:\.\d+)?)\s*MPa',
            r'(\d+(?:\.\d+)?)\s*bar',  # Convert: 1 bar = 0.1 MPa
        ],
        'power': [
            r'(?:power|wattage)\s*(?:of|at|was|=|:)?\s*(\d+(?:\.\d+)?)\s*W\b',
            r'(\d+(?:\.\d+)?)\s*W(?:atts?)?\b',
        ],
        'frequency': [
            r'(?:frequency)\s*(?:of|at|was|=|:)?\s*(\d+(?:\.\d+)?)\s*kHz',
            r'(\d+(?:\.\d+)?)\s*kHz',
        ],
        'yield': [
            r'(?:yield|recovery)\s*(?:of|was|=|:)?\s*(\d+(?:\.\d+)?)\s*%',
            r'extraction\s+(?:yield|efficiency)\s*(?:of|was|=|:)?\s*(\d+(?:\.\d+)?)\s*%',
        ],
        'tpc': [
            r'(?:TPC|total\s+phenolic\s+content)\s*(?:of|was|=|:)?\s*(\d+(?:\.\d+)?)\s*mg\s*GAE',
            r'(\d+(?:\.\d+)?)\s*mg\s*GAE\s*/\s*g',
        ],
        'tfc': [
            r'(?:TFC|total\s+flavonoid\s+content)\s*(?:of|was|=|:)?\s*(\d+(?:\.\d+)?)\s*mg\s*(?:QE|RE)',
            r'(\d+(?:\.\d+)?)\s*mg\s*(?:QE|RE)\s*/\s*g',
        ],
        'ic50': [
            r'IC50\s*(?:of|was|=|:)?\s*(\d+(?:\.\d+)?)\s*[µu]g\s*/\s*mL',
            r'(\d+(?:\.\d+)?)\s*[µu]g\s*/\s*mL\s*(?:IC50)?',
        ],
        'solid_liquid_ratio': [
            r'(?:solid.liquid|S/L|S:L)\s*(?:ratio)?\s*(?:of|was|=|:)?\s*(1\s*:\s*\d+)',
            r'(\d+)\s*(?:g|mg)\s*(?:in|per|/)\s*(\d+)\s*mL',
        ],
        'solvent_ratio': [
            r'(?:ethanol|methanol|acetone)\s*(?::|\s*/\s*)\s*water\s*\(?(\d+\s*:\s*\d+)\)?',
            r'(\d+)%\s*(?:v/v)?\s*(?:ethanol|methanol|acetone)',
        ],
    }
    
    # Extraction method keywords
    METHOD_KEYWORDS = {
        'Ultrasound-assisted extraction': ['ultrasound', 'UAE', 'sonication', 'ultrasonic'],
        'Microwave-assisted extraction': ['microwave', 'MAE', 'MW-assisted'],
        'Supercritical CO2 extraction': ['supercritical', 'SFE', 'SC-CO2', 'scCO2'],
        'Pressurized liquid extraction': ['pressurized liquid', 'PLE', 'ASE', 'accelerated solvent'],
        'Soxhlet extraction': ['Soxhlet', 'soxhlet'],
        'Maceration': ['maceration', 'macerate'],
        'Hydrodistillation': ['hydrodistillation', 'hydro-distillation'],
        'Steam distillation': ['steam distillation'],
        'Enzyme-assisted extraction': ['enzyme-assisted', 'EAE', 'enzymatic extraction'],
        'Cold pressing': ['cold press', 'cold-press', 'mechanical press'],
        'Deep eutectic solvent extraction': ['deep eutectic', 'DES', 'NADES'],
        'Pulsed electric field extraction': ['pulsed electric', 'PEF'],
    }
    
    # Solvent keywords
    SOLVENT_KEYWORDS = {
        'Water': ['water', 'aqueous', 'H2O'],
        'Ethanol': ['ethanol', 'EtOH'],
        'Methanol': ['methanol', 'MeOH'],
        'Acetone': ['acetone'],
        'Ethyl acetate': ['ethyl acetate', 'EtOAc'],
        'Hexane': ['hexane', 'n-hexane'],
        'CO2 (supercritical)': ['CO2', 'carbon dioxide'],
        'Dichloromethane': ['dichloromethane', 'DCM', 'methylene chloride'],
    }
    
    def extract_parameters(self, text: str) -> Dict:
        """Extract all extraction parameters from text."""
        params = {}
        
        for param_name, patterns in self.PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    try:
                        val = matches[0] if isinstance(matches[0], str) else matches[0][0]
                        val = val.replace(' ', '')
                        if param_name in ['solid_liquid_ratio', 'solvent_ratio']:
                            params[param_name] = val
                        else:
                            params[param_name] = float(val)
                    except (ValueError, IndexError):
                        pass
                    break
        
        # Time conversion: hours to minutes
        if 'time' in params and any(re.search(r'\d+\s*h(?:ours?)?', text, re.I) 
                                     for _ in [1]):
            if params['time'] < 24:  # Likely hours
                params['time'] *= 60
        
        # Pressure conversion: bar to MPa
        if 'pressure' in params and 'bar' in text.lower() and 'MPa' not in text:
            params['pressure'] *= 0.1
        
        # Identify extraction method
        params['method'] = self._identify_method(text)
        params['solvent'] = self._identify_solvent(text)
        
        return params
    
    def _identify_method(self, text: str) -> str:
        for method, keywords in self.METHOD_KEYWORDS.items():
            if any(kw.lower() in text.lower() for kw in keywords):
                return method
        return "Unknown"
    
    def _identify_solvent(self, text: str) -> str:
        for solvent, keywords in self.SOLVENT_KEYWORDS.items():
            if any(kw.lower() in text.lower() for kw in keywords):
                return solvent
        return "Unknown"


# =============================================================================
# 4. COMPOUND IDENTIFICATION MODULE
# =============================================================================

class CompoundIdentifier:
    """Resolve compound names to SMILES, CAS, and molecular descriptors."""
    
    @staticmethod
    def from_pubchem(name: str) -> Dict:
        """Look up compound in PubChem."""
        try:
            import pubchempy as pcp
            results = pcp.get_compounds(name, 'name')
            if results:
                comp = results[0]
                return {
                    'smiles': comp.isomeric_smiles or comp.canonical_smiles,
                    'mw': comp.molecular_weight,
                    'iupac': comp.iupac_name,
                    'cid': comp.cid,
                }
        except Exception as e:
            logger.warning(f"PubChem lookup failed for {name}: {e}")
        return {}
    
    @staticmethod
    def compute_descriptors(smiles: str) -> Dict:
        """Compute molecular descriptors from SMILES using RDKit."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            return {
                'MW': round(Descriptors.MolWt(mol), 2),
                'LogP': round(Descriptors.MolLogP(mol), 2),
                'TPSA': round(Descriptors.TPSA(mol), 2),
                'HBD': Lipinski.NumHDonors(mol),
                'HBA': Lipinski.NumHAcceptors(mol),
                'RotatableBonds': Lipinski.NumRotatableBonds(mol),
                'AromaticRings': Lipinski.NumAromaticRings(mol),
                'HeavyAtoms': mol.GetNumHeavyAtoms(),
                'FractionCSP3': round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
                'MolarRefractivity': round(Descriptors.MolMR(mol), 2),
            }
        except ImportError:
            logger.warning("RDKit not installed. Install with: pip install rdkit-pypi")
            return {}


# =============================================================================
# 5. PDF PARSER MODULE
# =============================================================================

class PDFExtractor:
    """Extract text and tables from PDF papers."""
    
    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """Extract full text from PDF."""
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except ImportError:
            logger.warning("pdfplumber not installed. Install: pip install pdfplumber")
            return ""
    
    @staticmethod
    def extract_tables(pdf_path: str) -> List[pd.DataFrame]:
        """Extract tables from PDF."""
        try:
            import pdfplumber
            tables = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    for table in page.extract_tables():
                        if table and len(table) > 1:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables.append(df)
            return tables
        except ImportError:
            return []


# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================

class ExtractionDataPipeline:
    """Main pipeline orchestrating all modules."""
    
    def __init__(self, email: str, api_key: str = "", output_dir: str = "./output"):
        self.searcher = PubMedSearcher(email, api_key)
        self.miner = ExtractionParameterMiner()
        self.identifier = CompoundIdentifier()
        self.pdf_extractor = PDFExtractor()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[ExtractionRecord] = []
    
    def run_pubmed_mining(self, compounds: List[str] = None, max_per_query: int = 50):
        """Run full PubMed mining pipeline."""
        queries = self.searcher.build_extraction_queries(compounds)
        
        all_pmids = set()
        for query in queries:
            pmids = self.searcher.search(query, max_results=max_per_query)
            all_pmids.update(pmids)
            time.sleep(0.5)
        
        logger.info(f"Total unique PMIDs: {len(all_pmids)}")
        
        # Fetch abstracts
        articles = self.searcher.fetch_abstracts(list(all_pmids))
        logger.info(f"Fetched {len(articles)} article abstracts")
        
        # Mine parameters from abstracts
        for article in articles:
            text = f"{article['title']} {article['abstract']}"
            params = self.miner.extract_parameters(text)
            
            if params.get('yield') or params.get('tpc'):
                record = ExtractionRecord(
                    doi=article['doi'],
                    year=article['year'],
                    extraction_method=params.get('method', ''),
                    solvent_system=params.get('solvent', ''),
                    temperature_c=params.get('temperature'),
                    time_min=params.get('time'),
                    pressure_mpa=params.get('pressure'),
                    power_w=params.get('power'),
                    frequency_khz=params.get('frequency'),
                    yield_pct=params.get('yield'),
                    tpc_mg_gae_per_g=params.get('tpc'),
                    tfc_mg_qe_per_g=params.get('tfc'),
                    antioxidant_ic50=params.get('ic50'),
                    solid_liquid_ratio=params.get('solid_liquid_ratio', ''),
                    notes=f"Mined from abstract. PMID: {article['pmid']}"
                )
                self.records.append(record)
        
        logger.info(f"Extracted {len(self.records)} records with data")
    
    def run_pdf_mining(self, pdf_dir: str):
        """Mine extraction data from downloaded PDFs."""
        pdf_path = Path(pdf_dir)
        if not pdf_path.exists():
            logger.warning(f"PDF directory not found: {pdf_dir}")
            return
        
        for pdf_file in pdf_path.glob("*.pdf"):
            logger.info(f"Processing: {pdf_file.name}")
            
            # Extract text
            text = self.pdf_extractor.extract_text(str(pdf_file))
            if not text:
                continue
            
            # Extract tables
            tables = self.pdf_extractor.extract_tables(str(pdf_file))
            
            # Mine from text
            params = self.miner.extract_parameters(text)
            
            if params.get('yield') or params.get('tpc') or params.get('temperature'):
                record = ExtractionRecord(
                    extraction_method=params.get('method', ''),
                    solvent_system=params.get('solvent', ''),
                    temperature_c=params.get('temperature'),
                    time_min=params.get('time'),
                    pressure_mpa=params.get('pressure'),
                    power_w=params.get('power'),
                    frequency_khz=params.get('frequency'),
                    yield_pct=params.get('yield'),
                    tpc_mg_gae_per_g=params.get('tpc'),
                    tfc_mg_qe_per_g=params.get('tfc'),
                    antioxidant_ic50=params.get('ic50'),
                    solid_liquid_ratio=params.get('solid_liquid_ratio', ''),
                    notes=f"Mined from PDF: {pdf_file.name}. {len(tables)} tables found."
                )
                self.records.append(record)
    
    def enrich_compounds(self):
        """Enrich records with SMILES and descriptors from PubChem."""
        unique_names = set(r.phytochemical_name for r in self.records if r.phytochemical_name)
        
        compound_cache = {}
        for name in unique_names:
            info = self.identifier.from_pubchem(name)
            if info:
                compound_cache[name] = info
                time.sleep(0.3)
        
        for record in self.records:
            if record.phytochemical_name in compound_cache:
                info = compound_cache[record.phytochemical_name]
                record.smiles = info.get('smiles', '')
        
        logger.info(f"Enriched {len(compound_cache)} unique compounds")
    
    def export_to_excel(self, template_path: str = None, output_path: str = None):
        """Export records to Excel matching the template format."""
        if not output_path:
            output_path = str(self.output_dir / "mined_extraction_data.xlsx")
        
        rows = [asdict(r) for r in self.records]
        df = pd.DataFrame(rows)
        
        # Rename columns to match template
        column_map = {
            'doi': 'DOI / Reference',
            'year': 'Year',
            'phytochemical_name': 'Phytochemical Name',
            'phytochemical_class': 'Phytochemical Class',
            'smiles': 'SMILES',
            'cas_number': 'CAS Number',
            'plant_source_latin': 'Plant Source (Latin)',
            'plant_part': 'Plant Part',
            'plant_pretreatment': 'Plant Pretreatment',
            'extraction_method': 'Extraction Method',
            'solvent_system': 'Solvent System',
            'solvent_ratio': 'Solvent Ratio (if mixed)',
            'solvent_volume_ml_per_g': 'Solvent Volume (mL/g plant)',
            'temperature_c': 'Temperature (°C)',
            'time_min': 'Time (min)',
            'pressure_mpa': 'Pressure (MPa)',
            'power_w': 'Power (W)',
            'frequency_khz': 'Frequency (kHz)',
            'solid_liquid_ratio': 'Solid:Liquid Ratio',
            'ph': 'pH',
            'number_of_cycles': 'Number of Cycles',
            'yield_pct': 'Yield (%)',
            'purity_pct': 'Purity (%)',
            'tpc_mg_gae_per_g': 'TPC (mg GAE/g)',
            'tfc_mg_qe_per_g': 'TFC (mg QE/g)',
            'antioxidant_ic50': 'Antioxidant Activity (IC50, µg/mL)',
            'scale': 'Scale (Lab/Pilot/Industrial)',
            'notes': 'Notes',
        }
        df = df.rename(columns=column_map)
        
        # Add ID column
        df.insert(0, 'ID', [f'EXT-{i+1:05d}' for i in range(len(df))])
        
        df.to_excel(output_path, index=False, sheet_name='Extraction_Data')
        logger.info(f"Exported {len(df)} records to {output_path}")
        return output_path
    
    def generate_report(self) -> str:
        """Generate a summary report of mined data."""
        if not self.records:
            return "No records mined yet."
        
        df = pd.DataFrame([asdict(r) for r in self.records])
        
        report = [
            "=" * 60,
            "EXTRACTION DATA MINING REPORT",
            "=" * 60,
            f"Total records: {len(self.records)}",
            f"Records with yield data: {df['yield_pct'].notna().sum()}",
            f"Records with TPC data: {df['tpc_mg_gae_per_g'].notna().sum()}",
            f"Records with temperature: {df['temperature_c'].notna().sum()}",
            f"Unique extraction methods: {df['extraction_method'].nunique()}",
            f"Unique solvents: {df['solvent_system'].nunique()}",
            "",
            "Method distribution:",
        ]
        for method, count in df['extraction_method'].value_counts().items():
            report.append(f"  {method}: {count}")
        
        return "\n".join(report)


# =============================================================================
# 7. USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # ---- CONFIGURATION ----
    EMAIL = "your.email@example.com"  # Required for NCBI API
    API_KEY = ""  # Optional: get from https://www.ncbi.nlm.nih.gov/account/settings/
    
    # Target compounds (add your compounds of interest)
    target_compounds = [
        "quercetin", "curcumin", "resveratrol", "gallic acid",
        "caffeic acid", "kaempferol", "rutin", "catechin",
        "berberine", "ursolic acid", "thymol", "eugenol",
        "capsaicin", "piperine", "ginsenoside",
    ]
    
    # ---- RUN PIPELINE ----
    pipeline = ExtractionDataPipeline(
        email=EMAIL,
        api_key=API_KEY,
        output_dir="./extraction_data_output"
    )
    
    # Step 1: Mine from PubMed abstracts
    print("\n[STEP 1] Mining PubMed abstracts...")
    pipeline.run_pubmed_mining(compounds=target_compounds, max_per_query=100)
    
    # Step 2: Mine from downloaded PDFs (if available)
    print("\n[STEP 2] Mining PDFs...")
    pipeline.run_pdf_mining("./pdfs")  # Put PDFs in this directory
    
    # Step 3: Enrich with compound identifiers
    print("\n[STEP 3] Enriching compound data...")
    pipeline.enrich_compounds()
    
    # Step 4: Export
    print("\n[STEP 4] Exporting...")
    output_file = pipeline.export_to_excel()
    
    # Step 5: Report
    print("\n" + pipeline.generate_report())
    print(f"\nOutput saved to: {output_file}")
    
    # ---- NEXT STEPS ----
    print("""
    ================================================
    NEXT STEPS TO REACH 10,000 VERIFIED ENTRIES:
    ================================================
    
    1. Run this pipeline with your NCBI API key for faster access
    2. Download full-text PDFs for high-value papers and re-run PDF mining
    3. Manually verify a random 10% sample of mined data against papers
    4. Use the existing 10K dataset as a scaffold:
       - Replace synthetic DOIs with real ones as you verify
       - Update extraction parameters with literature values
    5. Cross-reference with databases:
       - COCONUT (coconut.naturalproducts.net) for natural products
       - PubChem for SMILES/CAS verification
       - ChEMBL for bioactivity data
    6. Use RDKit to recompute all molecular descriptors from verified SMILES
    
    For your QSAR nanocarrier research, prioritize:
    - Compounds with nanoformulation studies
    - Entries with complete extraction parameter sets
    - Gum arabic-related phytochemical extraction data
    """)
