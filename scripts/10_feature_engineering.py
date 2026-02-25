#!/usr/bin/env python3
"""
Phase 10: Feature Engineering for QSAR Improvement
====================================================

Run: python scripts/10_feature_engineering.py

Creates engineered features and saves an enriched dataset:
  1. Interaction terms (Temperature × Time, LogP × Solvent polarity, etc.)
  2. Polynomial features (squared terms for key parameters)
  3. Solvent polarity descriptors (polarity index, dielectric constant, etc.)
  4. Extraction intensity indices (combined method parameters)
  5. Compound-method compatibility scores
  6. Ratio features (solid-to-solvent normalization)
  7. Log-transformed skewed targets

Output: data/processed/dataset_engineered.xlsx
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('logs/10_feature_engineering.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# SOLVENT PROPERTIES DATABASE
# =============================================================================

# Polarity Index (Snyder), Dielectric Constant, Hansen Solubility (δD, δP, δH)
# Sources: Snyder (1974), Barton (1983), Hansen (2007)
SOLVENT_PROPERTIES = {
    'water':        {'polarity_index': 10.2, 'dielectric': 80.1, 'boiling_point': 100, 'viscosity': 1.00,  'hansen_d': 15.5, 'hansen_p': 16.0, 'hansen_h': 42.3},
    'methanol':     {'polarity_index': 5.1,  'dielectric': 32.7, 'boiling_point': 65,  'viscosity': 0.54,  'hansen_d': 15.1, 'hansen_p': 12.3, 'hansen_h': 22.3},
    'ethanol':      {'polarity_index': 4.3,  'dielectric': 24.6, 'boiling_point': 78,  'viscosity': 1.07,  'hansen_d': 15.8, 'hansen_p': 8.8,  'hansen_h': 19.4},
    'acetone':      {'polarity_index': 5.1,  'dielectric': 20.7, 'boiling_point': 56,  'viscosity': 0.31,  'hansen_d': 15.5, 'hansen_p': 10.4, 'hansen_h': 7.0},
    'ethyl acetate':{'polarity_index': 4.4,  'dielectric': 6.0,  'boiling_point': 77,  'viscosity': 0.43,  'hansen_d': 15.8, 'hansen_p': 5.3,  'hansen_h': 7.2},
    'hexane':       {'polarity_index': 0.1,  'dielectric': 1.9,  'boiling_point': 69,  'viscosity': 0.30,  'hansen_d': 14.9, 'hansen_p': 0.0,  'hansen_h': 0.0},
    'dichloromethane':{'polarity_index': 3.1,'dielectric': 8.9,  'boiling_point': 40,  'viscosity': 0.41,  'hansen_d': 18.2, 'hansen_p': 6.3,  'hansen_h': 6.1},
    'chloroform':   {'polarity_index': 4.1,  'dielectric': 4.8,  'boiling_point': 61,  'viscosity': 0.54,  'hansen_d': 17.8, 'hansen_p': 3.1,  'hansen_h': 5.7},
    'dmso':         {'polarity_index': 7.2,  'dielectric': 46.7, 'boiling_point': 189, 'viscosity': 1.99,  'hansen_d': 18.4, 'hansen_p': 16.4, 'hansen_h': 10.2},
    'isopropanol':  {'polarity_index': 3.9,  'dielectric': 17.9, 'boiling_point': 82,  'viscosity': 2.04,  'hansen_d': 15.8, 'hansen_p': 6.1,  'hansen_h': 16.4},
    'n-butanol':    {'polarity_index': 3.9,  'dielectric': 17.8, 'boiling_point': 118, 'viscosity': 2.54,  'hansen_d': 16.0, 'hansen_p': 5.7,  'hansen_h': 15.8},
    'diethyl ether':{'polarity_index': 2.8,  'dielectric': 4.3,  'boiling_point': 35,  'viscosity': 0.22,  'hansen_d': 14.5, 'hansen_p': 2.9,  'hansen_h': 5.1},
    'petroleum ether':{'polarity_index':0.1, 'dielectric': 1.9,  'boiling_point': 60,  'viscosity': 0.30,  'hansen_d': 14.9, 'hansen_p': 0.0,  'hansen_h': 0.0},
    'toluene':      {'polarity_index': 2.4,  'dielectric': 2.4,  'boiling_point': 111, 'viscosity': 0.56,  'hansen_d': 18.0, 'hansen_p': 1.4,  'hansen_h': 2.0},
    'acetonitrile': {'polarity_index': 5.8,  'dielectric': 37.5, 'boiling_point': 82,  'viscosity': 0.34,  'hansen_d': 15.3, 'hansen_p': 18.0, 'hansen_h': 6.1},
    'co2':          {'polarity_index': 1.0,  'dielectric': 1.6,  'boiling_point': -78, 'viscosity': 0.07,  'hansen_d': 15.7, 'hansen_p': 5.7,  'hansen_h': 5.7},  # supercritical
    'acetic acid':  {'polarity_index': 6.0,  'dielectric': 6.2,  'boiling_point': 118, 'viscosity': 1.06,  'hansen_d': 14.5, 'hansen_p': 8.0,  'hansen_h': 13.5},
}


# =============================================================================
# EXTRACTION METHOD PROPERTIES
# =============================================================================

METHOD_PROPERTIES = {
    'Maceration':               {'energy_input': 1, 'mass_transfer': 1, 'selectivity': 3, 'is_conventional': 1, 'is_green': 0},
    'Soxhlet extraction':       {'energy_input': 3, 'mass_transfer': 3, 'selectivity': 2, 'is_conventional': 1, 'is_green': 0},
    'Ultrasound-assisted extraction': {'energy_input': 4, 'mass_transfer': 5, 'selectivity': 3, 'is_conventional': 0, 'is_green': 1},
    'Microwave-assisted extraction':  {'energy_input': 5, 'mass_transfer': 4, 'selectivity': 4, 'is_conventional': 0, 'is_green': 1},
    'Supercritical fluid extraction':  {'energy_input': 4, 'mass_transfer': 4, 'selectivity': 5, 'is_conventional': 0, 'is_green': 1},
    'Pressurized liquid extraction':   {'energy_input': 4, 'mass_transfer': 4, 'selectivity': 4, 'is_conventional': 0, 'is_green': 1},
    'Enzyme-assisted extraction':      {'energy_input': 2, 'mass_transfer': 3, 'selectivity': 5, 'is_conventional': 0, 'is_green': 1},
    'Steam distillation':       {'energy_input': 3, 'mass_transfer': 3, 'selectivity': 4, 'is_conventional': 1, 'is_green': 0},
    'Hydrodistillation':        {'energy_input': 3, 'mass_transfer': 3, 'selectivity': 3, 'is_conventional': 1, 'is_green': 0},
    'Cold pressing':            {'energy_input': 2, 'mass_transfer': 2, 'selectivity': 2, 'is_conventional': 1, 'is_green': 1},
    'Reflux extraction':        {'energy_input': 3, 'mass_transfer': 3, 'selectivity': 2, 'is_conventional': 1, 'is_green': 0},
    'Decoction':                {'energy_input': 2, 'mass_transfer': 2, 'selectivity': 2, 'is_conventional': 1, 'is_green': 0},
    'Infusion':                 {'energy_input': 1, 'mass_transfer': 1, 'selectivity': 2, 'is_conventional': 1, 'is_green': 0},
    'Percolation':              {'energy_input': 1, 'mass_transfer': 2, 'selectivity': 3, 'is_conventional': 1, 'is_green': 0},
    'Pulsed electric field extraction': {'energy_input': 5, 'mass_transfer': 5, 'selectivity': 4, 'is_conventional': 0, 'is_green': 1},
}


# =============================================================================
# PHYTOCHEMICAL CLASS PROPERTIES
# =============================================================================

CLASS_PROPERTIES = {
    'Phenolic acid':    {'typical_polarity': 'high',   'mw_range': 'low',    'thermal_stability': 3, 'solubility_water': 4},
    'Flavonoid':        {'typical_polarity': 'medium', 'mw_range': 'medium', 'thermal_stability': 3, 'solubility_water': 3},
    'Terpene':          {'typical_polarity': 'low',    'mw_range': 'medium', 'thermal_stability': 2, 'solubility_water': 1},
    'Alkaloid':         {'typical_polarity': 'medium', 'mw_range': 'medium', 'thermal_stability': 4, 'solubility_water': 2},
    'Carotenoid':       {'typical_polarity': 'low',    'mw_range': 'high',   'thermal_stability': 1, 'solubility_water': 1},
    'Anthocyanin':      {'typical_polarity': 'high',   'mw_range': 'medium', 'thermal_stability': 1, 'solubility_water': 4},
    'Stilbenoid':       {'typical_polarity': 'medium', 'mw_range': 'low',    'thermal_stability': 2, 'solubility_water': 2},
    'Lignan':           {'typical_polarity': 'medium', 'mw_range': 'medium', 'thermal_stability': 3, 'solubility_water': 2},
    'Coumarin':         {'typical_polarity': 'medium', 'mw_range': 'low',    'thermal_stability': 3, 'solubility_water': 2},
    'Xanthone':         {'typical_polarity': 'medium', 'mw_range': 'low',    'thermal_stability': 4, 'solubility_water': 2},
    'Tannin':           {'typical_polarity': 'high',   'mw_range': 'high',   'thermal_stability': 2, 'solubility_water': 3},
    'Saponin':          {'typical_polarity': 'medium', 'mw_range': 'high',   'thermal_stability': 3, 'solubility_water': 3},
    'Essential oil':    {'typical_polarity': 'low',    'mw_range': 'low',    'thermal_stability': 1, 'solubility_water': 1},
    'Quinone':          {'typical_polarity': 'medium', 'mw_range': 'low',    'thermal_stability': 3, 'solubility_water': 2},
    'Glycoside':        {'typical_polarity': 'high',   'mw_range': 'medium', 'thermal_stability': 2, 'solubility_water': 4},
}

POLARITY_MAP = {'low': 1, 'medium': 2, 'high': 3}
MW_RANGE_MAP = {'low': 1, 'medium': 2, 'high': 3}


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def add_solvent_features(df: pd.DataFrame) -> int:
    """Add numerical solvent property features from the solvent system column."""
    added = 0
    solvent_col = 'Solvent System'

    if solvent_col not in df.columns:
        return 0

    # Initialize new columns
    for prop in ['polarity_index', 'dielectric', 'boiling_point', 'viscosity',
                 'hansen_d', 'hansen_p', 'hansen_h']:
        df[f'solvent_{prop}'] = np.nan

    for idx, row in df.iterrows():
        solvent_str = str(row[solvent_col]).lower().strip()

        # Try exact match first
        matched = None
        for solvent_name, props in SOLVENT_PROPERTIES.items():
            if solvent_name in solvent_str:
                matched = props
                break

        # For mixtures (e.g., "Ethanol:Water 70:30"), average properties
        if matched is None and ':' in solvent_str:
            parts = solvent_str.split(':')
            if len(parts) >= 2:
                props_list = []
                for part in parts:
                    part_clean = part.strip().split()[0]  # Take first word
                    for sname, sprops in SOLVENT_PROPERTIES.items():
                        if sname in part_clean or part_clean in sname:
                            props_list.append(sprops)
                            break
                if props_list:
                    matched = {}
                    for key in props_list[0]:
                        matched[key] = np.mean([p[key] for p in props_list])

        if matched:
            for prop, val in matched.items():
                df.at[idx, f'solvent_{prop}'] = val
            added += 1

    # Fill remaining NaN with median
    for col in df.columns:
        if col.startswith('solvent_') and df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())

    logger.info(f"  Solvent features: matched {added}/{len(df)} rows")
    return 7  # Number of features added


def add_method_features(df: pd.DataFrame) -> int:
    """Add numerical extraction method properties."""
    method_col = 'Extraction Method'
    if method_col not in df.columns:
        return 0

    for prop in ['energy_input', 'mass_transfer', 'selectivity', 'is_conventional', 'is_green']:
        df[f'method_{prop}'] = np.nan

    matched = 0
    for idx, row in df.iterrows():
        method = str(row[method_col]).strip()
        if method in METHOD_PROPERTIES:
            for prop, val in METHOD_PROPERTIES[method].items():
                df.at[idx, f'method_{prop}'] = val
            matched += 1

    for col in df.columns:
        if col.startswith('method_') and df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())

    logger.info(f"  Method features: matched {matched}/{len(df)} rows")
    return 5


def add_class_features(df: pd.DataFrame) -> int:
    """Add phytochemical class properties."""
    class_col = 'Phytochemical Class'
    if class_col not in df.columns:
        return 0

    df['class_polarity'] = np.nan
    df['class_mw_range'] = np.nan
    df['class_thermal_stability'] = np.nan
    df['class_solubility_water'] = np.nan

    matched = 0
    for idx, row in df.iterrows():
        cls = str(row[class_col]).strip()
        if cls in CLASS_PROPERTIES:
            props = CLASS_PROPERTIES[cls]
            df.at[idx, 'class_polarity'] = POLARITY_MAP.get(props['typical_polarity'], 2)
            df.at[idx, 'class_mw_range'] = MW_RANGE_MAP.get(props['mw_range'], 2)
            df.at[idx, 'class_thermal_stability'] = props['thermal_stability']
            df.at[idx, 'class_solubility_water'] = props['solubility_water']
            matched += 1

    for col in ['class_polarity', 'class_mw_range', 'class_thermal_stability', 'class_solubility_water']:
        df[col] = df[col].fillna(df[col].median())

    logger.info(f"  Class features: matched {matched}/{len(df)} rows")
    return 4


def add_interaction_terms(df: pd.DataFrame) -> int:
    """Add physically meaningful interaction features."""
    added = 0

    # Temperature × Time (total thermal exposure)
    if 'Temperature (\u00b0C)' in df.columns and 'Time (min)' in df.columns:
        temp = df['Temperature (\u00b0C)'].fillna(0)
        time = df['Time (min)'].fillna(0)
        df['thermal_exposure'] = temp * time
        df['temp_time_ratio'] = temp / (time + 1)  # Intensity vs duration
        added += 2

    # LogP × Solvent polarity (like-dissolves-like compatibility)
    if 'LogP (ALogP)' in df.columns and 'solvent_polarity_index' in df.columns:
        logp = df['LogP (ALogP)'].fillna(0)
        pol = df['solvent_polarity_index'].fillna(5)
        df['logp_solvent_compatibility'] = -np.abs(logp - (10 - pol))  # Higher = better match
        added += 1

    # MW × Temperature (thermal degradation risk)
    if 'MW (g/mol)' in df.columns and 'Temperature (\u00b0C)' in df.columns:
        mw = df['MW (g/mol)'].fillna(300)
        temp = df['Temperature (\u00b0C)'].fillna(50)
        df['mw_temp_interaction'] = mw * temp / 1000
        added += 1

    # TPSA × Solvent polarity (polar matching)
    if 'TPSA (\u00c5\u00b2)' in df.columns and 'solvent_polarity_index' in df.columns:
        tpsa = df['TPSA (\u00c5\u00b2)'].fillna(50)
        pol = df['solvent_polarity_index'].fillna(5)
        df['tpsa_polarity_match'] = tpsa * pol / 100
        added += 1

    # HBD + HBA × Solvent H-bonding (hydrogen bond matching)
    if 'HBD' in df.columns and 'HBA' in df.columns and 'solvent_hansen_h' in df.columns:
        hb_total = df['HBD'].fillna(0) + df['HBA'].fillna(0)
        hansen_h = df['solvent_hansen_h'].fillna(10)
        df['hbond_solvent_match'] = hb_total * hansen_h / 10
        added += 1

    # Power × Time (total energy delivered, for UAE/MAE)
    if 'Power (W)' in df.columns and 'Time (min)' in df.columns:
        power = df['Power (W)'].fillna(0)
        time = df['Time (min)'].fillna(0)
        df['total_energy_kJ'] = power * time * 60 / 1000  # W * s = J, /1000 = kJ
        added += 1

    # Pressure × Temperature (for SFE)
    if 'Pressure (MPa)' in df.columns and 'Temperature (\u00b0C)' in df.columns:
        pres = df['Pressure (MPa)'].fillna(0)
        temp = df['Temperature (\u00b0C)'].fillna(50)
        df['pres_temp_interaction'] = pres * temp / 100
        added += 1

    # Aromatic rings × Solvent dielectric (pi-stacking affinity)
    if 'Aromatic Rings' in df.columns and 'solvent_dielectric' in df.columns:
        arom = df['Aromatic Rings'].fillna(0)
        diel = df['solvent_dielectric'].fillna(20)
        df['aromatic_dielectric'] = arom * np.log1p(diel)
        added += 1

    logger.info(f"  Interaction terms added: {added}")
    return added


def add_polynomial_features(df: pd.DataFrame) -> int:
    """Add squared terms for key non-linear relationships."""
    added = 0
    poly_cols = ['Temperature (\u00b0C)', 'Time (min)', 'LogP (ALogP)', 'MW (g/mol)']

    for col in poly_cols:
        if col in df.columns:
            vals = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
            new_col = f'{col.split("(")[0].strip()}_squared'
            df[new_col] = vals ** 2
            added += 1

    logger.info(f"  Polynomial features added: {added}")
    return added


def add_extraction_intensity_index(df: pd.DataFrame) -> int:
    """Create composite extraction intensity score."""
    added = 0

    # Normalized extraction intensity (0-1 scale)
    components = []
    weights = []

    if 'Temperature (\u00b0C)' in df.columns:
        temp_norm = df['Temperature (\u00b0C)'].fillna(50) / 300  # Max ~300°C
        components.append(temp_norm)
        weights.append(0.3)

    if 'Time (min)' in df.columns:
        time_norm = np.log1p(df['Time (min)'].fillna(60)) / np.log1p(10080)  # Max 1 week
        components.append(time_norm)
        weights.append(0.2)

    if 'Power (W)' in df.columns:
        power_norm = df['Power (W)'].fillna(0) / 10000
        components.append(power_norm)
        weights.append(0.2)

    if 'Pressure (MPa)' in df.columns:
        pres_norm = df['Pressure (MPa)'].fillna(0) / 100
        components.append(pres_norm)
        weights.append(0.15)

    if 'method_energy_input' in df.columns:
        method_norm = df['method_energy_input'].fillna(3) / 5
        components.append(method_norm)
        weights.append(0.15)

    if components:
        total_weight = sum(weights[:len(components)])
        df['extraction_intensity'] = sum(
            c * w for c, w in zip(components, weights[:len(components)])
        ) / total_weight
        added += 1

    logger.info(f"  Extraction intensity index added: {added}")
    return added


def add_log_targets(df: pd.DataFrame) -> int:
    """Add log-transformed versions of skewed targets."""
    added = 0
    targets = ['Yield (%)', 'TPC (mg GAE/g)', 'TFC (mg QE/g)',
               'Antioxidant Activity (IC50, \u00b5g/mL)']

    for col in targets:
        if col in df.columns:
            vals = df[col]
            if vals.notna().any() and (vals > 0).any():
                skew = vals.dropna().skew()
                if abs(skew) > 1.0:
                    df[f'log_{col.split("(")[0].strip()}'] = np.log1p(vals.clip(lower=0))
                    added += 1
                    logger.info(f"    {col}: skew={skew:.2f} -> log-transformed")

    return added


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Load dataset
    dataset_path = Path('data/processed/dataset_with_replacements.xlsx')
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    logger.info(f"Loading dataset from {dataset_path}...")
    df = pd.read_excel(dataset_path)
    original_cols = len(df.columns)
    logger.info(f"Original: {len(df)} rows, {original_cols} columns")

    # Load and merge molecular descriptors
    desc_path = Path('data/processed/molecular_descriptors_rdkit.xlsx')
    if desc_path.exists():
        df_desc = pd.read_excel(desc_path)
        merge_cols = ['Phytochemical Name'] + [c for c in df_desc.columns if c != 'Phytochemical Name']
        available = [c for c in merge_cols if c in df_desc.columns]
        df = df.merge(
            df_desc[available].drop_duplicates('Phytochemical Name'),
            on='Phytochemical Name', how='left', suffixes=('', '_desc')
        )
        logger.info(f"Merged molecular descriptors: {len(df_desc)} compounds")

    # Apply feature engineering
    logger.info(f"\n{'='*60}")
    logger.info("FEATURE ENGINEERING")
    logger.info(f"{'='*60}")

    total_new = 0

    logger.info("\n[1/7] Solvent properties...")
    total_new += add_solvent_features(df)

    logger.info("\n[2/7] Extraction method properties...")
    total_new += add_method_features(df)

    logger.info("\n[3/7] Phytochemical class properties...")
    total_new += add_class_features(df)

    logger.info("\n[4/7] Interaction terms...")
    total_new += add_interaction_terms(df)

    logger.info("\n[5/7] Polynomial features...")
    total_new += add_polynomial_features(df)

    logger.info("\n[6/7] Extraction intensity index...")
    total_new += add_extraction_intensity_index(df)

    logger.info("\n[7/7] Log-transformed targets...")
    total_new += add_log_targets(df)

    # Save enriched dataset
    output_path = Path('data/processed/dataset_engineered.xlsx')
    df.to_excel(str(output_path), index=False)

    final_cols = len(df.columns)
    logger.info(f"\n{'='*60}")
    logger.info(f"FEATURE ENGINEERING COMPLETE")
    logger.info(f"  Original columns: {original_cols}")
    logger.info(f"  New features: {final_cols - original_cols}")
    logger.info(f"  Total columns: {final_cols}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"{'='*60}")

    # List all new features
    logger.info(f"\nNew feature columns:")
    new_cols = [c for c in df.columns if c not in pd.read_excel(dataset_path).columns]
    for col in new_cols:
        non_null = df[col].notna().sum()
        logger.info(f"  {col}: {non_null}/{len(df)} non-null")


if __name__ == '__main__':
    main()
