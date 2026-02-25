#!/usr/bin/env python3
"""
QSAR Modeling Pipeline v3 — With Engineered Features
======================================================

Run: python scripts/09c_qsar_engineered.py

Uses the enriched dataset from Phase 10 (feature engineering) with:
  - Solvent polarity descriptors (7 features)
  - Method properties (5 features)
  - Compound class properties (4 features)
  - Interaction terms (9 features)
  - Polynomial features (4 features)
  - Extraction intensity index (1 feature)

Trains on literature-verified data only with 5-fold CV.
"""

import os
import sys
import json
import pickle
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import (
    KFold, cross_val_score, cross_val_predict,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('logs/09c_qsar_engineered.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

TARGETS = {
    'Yield (%)':  {'name': 'yield', 'min': 0.01, 'max': 100},
    'TPC (mg GAE/g)': {'name': 'tpc', 'min': 0.01, 'max': 1000},
    'TFC (mg QE/g)': {'name': 'tfc', 'min': 0.01, 'max': 1000},
    'Antioxidant Activity (IC50, \u00b5g/mL)': {'name': 'ic50', 'min': 0.001, 'max': 10000},
}

# ALL numeric features including engineered ones
NUMERIC_FEATURES = [
    # Original molecular descriptors
    'MW (g/mol)', 'LogP (ALogP)', 'TPSA (\u00c5\u00b2)', 'HBD', 'HBA',
    'Rotatable Bonds', 'Aromatic Rings', 'Heavy Atoms',
    'Fraction CSP3', 'Molar Refractivity',
    # Original extraction parameters
    'Temperature (\u00b0C)', 'Time (min)', 'Pressure (MPa)',
    'Power (W)', 'Frequency (kHz)', 'pH',
    # Solvent properties
    'solvent_polarity_index', 'solvent_dielectric', 'solvent_boiling_point',
    'solvent_viscosity', 'solvent_hansen_d', 'solvent_hansen_p', 'solvent_hansen_h',
    # Method properties
    'method_energy_input', 'method_mass_transfer', 'method_selectivity',
    'method_is_conventional', 'method_is_green',
    # Class properties
    'class_polarity', 'class_mw_range', 'class_thermal_stability', 'class_solubility_water',
    # Interaction terms
    'thermal_exposure', 'temp_time_ratio', 'logp_solvent_compatibility',
    'mw_temp_interaction', 'tpsa_polarity_match', 'hbond_solvent_match',
    'total_energy_kJ', 'pres_temp_interaction', 'aromatic_dielectric',
    # Polynomial
    'Temperature_squared', 'Time_squared', 'LogP_squared', 'MW_squared',
    # Composite
    'extraction_intensity',
]


def get_models():
    models = {
        'Ridge': (Ridge(), {'alpha': [0.01, 0.1, 1, 10, 100, 1000]}),
        'Lasso': (Lasso(max_iter=5000), {'alpha': [0.001, 0.01, 0.1, 1, 10]}),
        'ElasticNet': (ElasticNet(max_iter=5000), {
            'alpha': [0.01, 0.1, 1, 10],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        }),
        'KNN': (KNeighborsRegressor(), {
            'n_neighbors': [3, 5, 7, 11],
            'weights': ['uniform', 'distance'],
        }),
        'SVR': (SVR(), {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.5],
            'kernel': ['rbf', 'linear'],
        }),
        'RandomForest': (RandomForestRegressor(random_state=42, n_jobs=-1), {
            'n_estimators': [100, 200, 500],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }),
        'GradientBoosting': (GradientBoostingRegressor(random_state=42), {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
        }),
        'ExtraTrees': (ExtraTreesRegressor(random_state=42, n_jobs=-1), {
            'n_estimators': [100, 200, 500],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5],
        }),
        'MLP': (MLPRegressor(random_state=42, max_iter=2000,
                              early_stopping=True, validation_fraction=0.15), {
            'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
            'alpha': [0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.01],
        }),
    }

    if HAS_XGBOOST:
        models['XGBoost'] = (XGBRegressor(random_state=42, n_jobs=-1,
                                           verbosity=0, tree_method='hist'), {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.7, 0.9],
            'reg_lambda': [1, 5, 10],
        })

    return models


def main():
    output_dir = Path('output/qsar_results_v3')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'best_models').mkdir(exist_ok=True)
    (output_dir / 'predictions').mkdir(exist_ok=True)
    (output_dir / 'shap_analysis').mkdir(exist_ok=True)

    # Load engineered dataset
    eng_path = Path('data/processed/dataset_engineered.xlsx')
    if not eng_path.exists():
        logger.error("Engineered dataset not found. Run Phase 10 first.")
        return

    df = pd.read_excel(eng_path)
    logger.info(f"Loaded engineered dataset: {df.shape}")

    # Filter to verified only
    if '_data_source' in df.columns:
        df = df[df['_data_source'] != 'synthetic'].copy()
        logger.info(f"Verified entries: {len(df)}")

    all_results = []
    summary = {}

    for target_col, tinfo in TARGETS.items():
        tname = tinfo['name']
        logger.info(f"\n{'='*70}")
        logger.info(f"TARGET: {target_col}")
        logger.info(f"{'='*70}")

        # Filter valid target
        mask = (
            df[target_col].notna() &
            (df[target_col] >= tinfo['min']) &
            (df[target_col] <= tinfo['max'])
        )
        df_valid = df[mask].copy()
        logger.info(f"  Valid samples: {len(df_valid)}")

        if len(df_valid) < 30:
            logger.warning(f"  Skipping — too few samples")
            continue

        # Select available numeric features
        available_features = [f for f in NUMERIC_FEATURES
                              if f in df_valid.columns
                              and f != target_col
                              and df_valid[f].notna().mean() > 0.15]

        logger.info(f"  Available features: {len(available_features)}")

        X = df_valid[available_features].copy()
        y = df_valid[target_col].values

        # Preprocessor
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])

        X_processed = preprocessor.fit_transform(X)
        logger.info(f"  Processed shape: {X_processed.shape}")
        logger.info(f"  Target stats: mean={y.mean():.2f}, std={y.std():.2f}")

        # CV setup
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        models = get_models()
        best_models = {}

        for mname, (estimator, params) in models.items():
            logger.info(f"\n  Training {mname}...")

            try:
                n_combos = 1
                for v in params.values():
                    n_combos *= len(v)

                if n_combos > 80:
                    search = RandomizedSearchCV(
                        estimator, params, n_iter=40,
                        cv=inner_cv, scoring='r2',
                        random_state=42, n_jobs=-1, verbose=0
                    )
                else:
                    search = GridSearchCV(
                        estimator, params,
                        cv=inner_cv, scoring='r2',
                        n_jobs=-1, verbose=0
                    )

                search.fit(X_processed, y)
                best = search.best_estimator_

                # CV metrics
                cv_r2 = cross_val_score(best, X_processed, y, cv=outer_cv,
                                         scoring='r2', n_jobs=-1)
                cv_rmse = cross_val_score(best, X_processed, y, cv=outer_cv,
                                           scoring='neg_root_mean_squared_error', n_jobs=-1)
                y_pred_cv = cross_val_predict(best, X_processed, y,
                                               cv=outer_cv, n_jobs=-1)

                # Q2
                ss_res = np.sum((y - y_pred_cv) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                q2 = 1 - (ss_res / ss_tot)

                # Train R2
                y_pred_train = best.predict(X_processed)
                train_r2 = r2_score(y, y_pred_train)

                r_val, p_val = stats.pearsonr(y, y_pred_cv)

                metrics = {
                    'Model': mname, 'Target': tname,
                    'N_samples': len(y), 'N_features': X_processed.shape[1],
                    'Train R2': round(train_r2, 4),
                    'CV R2 (mean)': round(cv_r2.mean(), 4),
                    'CV R2 (std)': round(cv_r2.std(), 4),
                    'Q2': round(q2, 4),
                    'CV RMSE': round(-cv_rmse.mean(), 4),
                    'Pearson r': round(r_val, 4),
                    'p-value': f"{p_val:.2e}",
                    'Best Params': str(search.best_params_),
                }
                all_results.append(metrics)

                best_models[mname] = {
                    'model': best, 'q2': q2, 'cv_r2': cv_r2.mean(),
                    'train_r2': train_r2, 'params': search.best_params_,
                    'y_pred_cv': y_pred_cv, 'y_true': y,
                }

                logger.info(f"    Train R2: {train_r2:.4f} | CV R2: {cv_r2.mean():.4f} | Q2: {q2:.4f}")

            except Exception as e:
                logger.error(f"    {mname} failed: {e}")

        if not best_models:
            continue

        # Best model
        best_name = max(best_models, key=lambda k: best_models[k]['q2'])
        best_info = best_models[best_name]
        summary[tname] = {
            'best_model': best_name,
            'train_r2': best_info['train_r2'],
            'cv_r2': best_info['cv_r2'],
            'q2': best_info['q2'],
            'n_samples': len(y),
            'n_features': X_processed.shape[1],
        }

        logger.info(f"\n  BEST: {best_name} (Q2={best_info['q2']:.4f})")

        # Save models
        for mname, minfo in best_models.items():
            with open(output_dir / 'best_models' / f'{tname}_{mname}.pkl', 'wb') as f:
                pickle.dump({
                    'model': minfo['model'],
                    'preprocessor': preprocessor,
                    'features': available_features,
                    'q2': minfo['q2'],
                }, f)

        # Save predictions + scatter plot
        pred_dir = output_dir / 'predictions'
        for mname, minfo in best_models.items():
            pd.DataFrame({
                'Actual': minfo['y_true'],
                'CV_Predicted': minfo['y_pred_cv'],
                'Residual': minfo['y_true'] - minfo['y_pred_cv'],
            }).to_excel(pred_dir / f'{tname}_{mname}.xlsx', index=False)

        if HAS_MPL:
            top = sorted(best_models, key=lambda k: best_models[k]['q2'], reverse=True)[:6]
            ncols = min(3, len(top))
            nrows = (len(top) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
            axes = np.array(axes).flatten() if len(top) > 1 else [axes]

            for i, mn in enumerate(top):
                ax = axes[i]
                yt = best_models[mn]['y_true']
                yp = best_models[mn]['y_pred_cv']
                ax.scatter(yt, yp, alpha=0.4, s=15, c='#2196F3')
                lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
                ax.plot(lims, lims, 'r--', alpha=0.8)
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted (CV)')
                ax.set_title(f'{mn}\nQ2={best_models[mn]["q2"]:.4f}')

            for i in range(len(top), len(axes)):
                axes[i].set_visible(False)

            plt.suptitle(f'{tname} — Engineered Features', fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(pred_dir / f'scatter_{tname}.png', dpi=200, bbox_inches='tight')
            plt.close()

        # SHAP
        if HAS_SHAP:
            tree_order = ['XGBoost', 'GradientBoosting', 'RandomForest', 'ExtraTrees']
            for tn in tree_order:
                if tn in best_models:
                    try:
                        explainer = shap.TreeExplainer(best_models[tn]['model'])
                        sample = X_processed[:min(300, len(X_processed))]
                        sv = explainer.shap_values(sample)

                        imp = pd.DataFrame({
                            'Feature': available_features,
                            'Mean |SHAP|': np.abs(sv).mean(axis=0)
                        }).sort_values('Mean |SHAP|', ascending=False)
                        imp.to_excel(output_dir / 'shap_analysis' / f'shap_{tname}.xlsx', index=False)

                        logger.info(f"\n  SHAP top 10 ({tn}):")
                        for _, row in imp.head(10).iterrows():
                            logger.info(f"    {row['Feature']}: {row['Mean |SHAP|']:.4f}")

                        if HAS_MPL:
                            plt.figure(figsize=(10, 8))
                            shap.summary_plot(sv, sample, feature_names=available_features,
                                            show=False, max_display=20)
                            plt.tight_layout()
                            plt.savefig(output_dir / 'shap_analysis' / f'shap_beeswarm_{tname}.png',
                                       dpi=200, bbox_inches='tight')
                            plt.close()
                    except Exception as e:
                        logger.error(f"  SHAP failed: {e}")
                    break

    # Save all results
    if all_results:
        df_res = pd.DataFrame(all_results).sort_values(['Target', 'Q2'], ascending=[True, False])
        with pd.ExcelWriter(output_dir / 'model_performance_summary.xlsx') as w:
            df_res.to_excel(w, sheet_name='All Models', index=False)
            if summary:
                pd.DataFrame(summary).T.to_excel(w, sheet_name='Best Models')

        # Comparison with v2
        logger.info(f"\n{'='*70}")
        logger.info("RESULTS SUMMARY (v3 — Engineered Features)")
        logger.info(f"{'='*70}")
        for tname, info in summary.items():
            logger.info(f"  {tname}: {info['best_model']} "
                        f"(Q2={info['q2']:.4f}, CV R2={info['cv_r2']:.4f}, "
                        f"N={info['n_samples']}, Features={info['n_features']})")
        logger.info(f"\nOutput: {output_dir}")
        logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
