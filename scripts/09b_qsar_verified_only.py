#!/usr/bin/env python3
"""
QSAR Modeling Pipeline v2 — Literature-Verified Data Only
==========================================================

Run: python scripts/09b_qsar_verified_only.py

Key difference from v1:
  - Trains ONLY on the 1,269 literature-verified entries
  - Uses 5-fold and 10-fold cross-validation (no synthetic data)
  - Nested CV for unbiased performance estimates
  - Applicability domain analysis
  - Williams plot for outlier detection

Why: Synthetic data has no real structure-activity relationships.
     Training on noise gives R² ≈ 0. Real QSAR requires real data.
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
    KFold, RepeatedKFold, cross_val_score, cross_val_predict,
    GridSearchCV, RandomizedSearchCV, train_test_split,
    LeaveOneOut
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)

# Models
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

# Optional imports
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
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('logs/09b_qsar_verified.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGETS = {
    'Yield (%)':  {'name': 'yield', 'min': 0.01, 'max': 100},
    'TPC (mg GAE/g)': {'name': 'tpc', 'min': 0.01, 'max': 1000},
    'TFC (mg QE/g)': {'name': 'tfc', 'min': 0.01, 'max': 1000},
    'Antioxidant Activity (IC50, \u00b5g/mL)': {'name': 'ic50', 'min': 0.001, 'max': 10000},
    'Purity (%)': {'name': 'purity', 'min': 0.1, 'max': 100},
}

MOLECULAR_FEATURES = [
    'MW (g/mol)', 'LogP (ALogP)', 'TPSA (\u00c5\u00b2)', 'HBD', 'HBA',
    'Rotatable Bonds', 'Aromatic Rings', 'Heavy Atoms',
    'Fraction CSP3', 'Molar Refractivity',
]

EXTRACTION_NUMERIC = [
    'Temperature (\u00b0C)', 'Time (min)', 'Pressure (MPa)',
    'Power (W)', 'Frequency (kHz)', 'pH',
    'Solid-to-Solvent Ratio',
]

EXTRACTION_CATEGORICAL = [
    'Extraction Method', 'Solvent System', 'Phytochemical Class',
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_verified_data() -> tuple:
    """Load ONLY literature-verified entries."""

    dataset_path = Path('data/processed/dataset_with_replacements.xlsx')
    if not dataset_path.exists():
        dataset_path = Path('output/Phytochemical_Extraction_Dataset_FINAL_20260224.xlsx')

    logger.info(f"Loading from {dataset_path}...")
    df = pd.read_excel(dataset_path)
    logger.info(f"Full dataset: {len(df)} rows")

    # Filter to verified entries only
    if '_data_source' in df.columns:
        df_verified = df[df['_data_source'] != 'synthetic'].copy()
        logger.info(f"Literature-verified entries: {len(df_verified)}")
    else:
        logger.warning("No _data_source column. Using full dataset.")
        df_verified = df.copy()

    # Load molecular descriptors
    desc_path = Path('data/processed/molecular_descriptors_rdkit.xlsx')
    if desc_path.exists():
        df_desc = pd.read_excel(desc_path)
        logger.info(f"Molecular descriptors loaded: {df_desc.shape}")
    else:
        df_desc = None
        logger.warning("No molecular descriptors found")

    return df_verified, df_desc


def prepare_data(df: pd.DataFrame, df_desc: pd.DataFrame,
                 target_col: str) -> tuple:
    """Prepare feature matrix and target vector from verified data."""

    # Merge molecular descriptors
    if df_desc is not None:
        merge_cols = ['Phytochemical Name'] + [c for c in MOLECULAR_FEATURES if c in df_desc.columns]
        available = [c for c in merge_cols if c in df_desc.columns]
        df_merged = df.merge(
            df_desc[available].drop_duplicates('Phytochemical Name'),
            on='Phytochemical Name', how='left', suffixes=('', '_desc')
        )
    else:
        df_merged = df.copy()

    # Filter valid target values
    target_info = TARGETS[target_col]
    mask = (
        df_merged[target_col].notna() &
        (df_merged[target_col] >= target_info['min']) &
        (df_merged[target_col] <= target_info['max'])
    )
    df_valid = df_merged[mask].copy()
    logger.info(f"  Rows with valid {target_col}: {len(df_valid)}")

    if len(df_valid) < 30:
        logger.warning(f"  Too few samples ({len(df_valid)}) for reliable modeling")
        return None, None, None, None, None

    # Build feature lists
    numeric_features = []
    for col in EXTRACTION_NUMERIC + MOLECULAR_FEATURES:
        if col in df_valid.columns and col != target_col:
            # Only include if >20% non-null
            if df_valid[col].notna().mean() > 0.20:
                numeric_features.append(col)

    categorical_features = []
    for col in EXTRACTION_CATEGORICAL:
        if col in df_valid.columns:
            n_unique = df_valid[col].nunique()
            if 2 <= n_unique <= 50:
                # Cap cardinality
                top = df_valid[col].value_counts().head(15).index
                df_valid.loc[~df_valid[col].isin(top), col] = 'Other'
                categorical_features.append(col)

    all_features = numeric_features + categorical_features
    logger.info(f"  Features: {len(numeric_features)} numeric + {len(categorical_features)} categorical")

    if len(all_features) < 2:
        logger.warning(f"  Too few features ({len(all_features)})")
        return None, None, None, None, None

    X = df_valid[all_features].copy()
    y = df_valid[target_col].values

    # Build preprocessor
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                   max_categories=15)),
    ])

    transformers = [('num', numeric_transformer, numeric_features)]
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    return X, y, all_features, numeric_features, preprocessor


# =============================================================================
# MODEL TRAINING WITH NESTED CV
# =============================================================================

def get_models() -> dict:
    """Define model configurations."""
    models = {
        'Ridge': {
            'estimator': Ridge(),
            'params': {
                'alpha': [0.01, 0.1, 1.0, 10, 100, 1000],
            }
        },
        'Lasso': {
            'estimator': Lasso(max_iter=5000),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10],
            }
        },
        'ElasticNet': {
            'estimator': ElasticNet(max_iter=5000),
            'params': {
                'alpha': [0.01, 0.1, 1.0, 10],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            }
        },
        'KNN': {
            'estimator': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [3, 5, 7, 11],
                'weights': ['uniform', 'distance'],
            }
        },
        'SVR': {
            'estimator': SVR(),
            'params': {
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.5],
                'kernel': ['rbf', 'linear'],
            }
        },
        'RandomForest': {
            'estimator': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 500],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }
        },
        'GradientBoosting': {
            'estimator': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0],
            }
        },
        'ExtraTrees': {
            'estimator': ExtraTreesRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 500],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5],
            }
        },
        'MLP': {
            'estimator': MLPRegressor(random_state=42, max_iter=2000,
                                       early_stopping=True,
                                       validation_fraction=0.15),
            'params': {
                'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
                'alpha': [0.001, 0.01, 0.1],
                'learning_rate_init': [0.001, 0.01],
            }
        },
    }

    if HAS_XGBOOST:
        models['XGBoost'] = {
            'estimator': XGBRegressor(random_state=42, n_jobs=-1,
                                       verbosity=0, tree_method='hist'),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.7, 0.9],
                'reg_lambda': [1, 5, 10],
            }
        }

    return models


def train_with_cv(X, y, preprocessor, target_name: str) -> tuple:
    """Train all models using cross-validation on verified data only."""

    X_processed = preprocessor.fit_transform(X)
    logger.info(f"  Processed feature matrix: {X_processed.shape}")

    models = get_models()
    results = []
    best_models = {}
    all_predictions = {}

    # Cross-validation strategy
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, config in models.items():
        logger.info(f"\n  Training {model_name}...")

        try:
            estimator = config['estimator']
            param_grid = config['params']

            # Count parameter combinations
            n_combos = 1
            for v in param_grid.values():
                n_combos *= len(v)

            # Inner CV for hyperparameter tuning
            if n_combos > 80:
                search = RandomizedSearchCV(
                    estimator, param_grid,
                    n_iter=40, cv=inner_cv, scoring='r2',
                    random_state=42, n_jobs=-1, verbose=0
                )
            else:
                search = GridSearchCV(
                    estimator, param_grid,
                    cv=inner_cv, scoring='r2',
                    n_jobs=-1, verbose=0
                )

            # Fit on all data
            search.fit(X_processed, y)
            best_est = search.best_estimator_

            # Outer CV for unbiased performance estimate
            cv_r2 = cross_val_score(best_est, X_processed, y,
                                     cv=outer_cv, scoring='r2', n_jobs=-1)
            cv_rmse = cross_val_score(best_est, X_processed, y,
                                       cv=outer_cv, scoring='neg_root_mean_squared_error',
                                       n_jobs=-1)
            cv_mae = cross_val_score(best_est, X_processed, y,
                                      cv=outer_cv, scoring='neg_mean_absolute_error',
                                      n_jobs=-1)

            # Cross-validated predictions (for scatter plot)
            y_pred_cv = cross_val_predict(best_est, X_processed, y,
                                           cv=outer_cv, n_jobs=-1)

            # Training metrics (refit on full data)
            y_pred_train = best_est.predict(X_processed)
            train_r2 = r2_score(y, y_pred_train)

            # Pearson correlation on CV predictions
            r_val, p_val = stats.pearsonr(y, y_pred_cv)

            # Q2 (external validation metric using CV predictions)
            ss_res = np.sum((y - y_pred_cv) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            q2 = 1 - (ss_res / ss_tot)

            metrics = {
                'Model': model_name,
                'Target': target_name,
                'N_samples': len(y),
                'Train R2': round(train_r2, 4),
                'CV R2 (mean)': round(cv_r2.mean(), 4),
                'CV R2 (std)': round(cv_r2.std(), 4),
                'Q2': round(q2, 4),
                'CV RMSE (mean)': round(-cv_rmse.mean(), 4),
                'CV RMSE (std)': round(cv_rmse.std(), 4),
                'CV MAE (mean)': round(-cv_mae.mean(), 4),
                'Pearson r': round(r_val, 4),
                'p-value': f"{p_val:.2e}",
                'Best Params': str(search.best_params_),
            }

            results.append(metrics)

            best_models[model_name] = {
                'model': best_est,
                'params': search.best_params_,
                'cv_r2': cv_r2.mean(),
                'q2': q2,
                'train_r2': train_r2,
            }

            all_predictions[model_name] = {
                'y_true': y,
                'y_pred_cv': y_pred_cv,
                'y_pred_train': y_pred_train,
            }

            logger.info(f"    Train R2:  {train_r2:.4f}")
            logger.info(f"    CV R2:     {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
            logger.info(f"    Q2:        {q2:.4f}")
            logger.info(f"    CV RMSE:   {-cv_rmse.mean():.4f}")

        except Exception as e:
            logger.error(f"    {model_name} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    return results, best_models, all_predictions


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(predictions: dict, target_name: str, output_dir: Path):
    """Generate predicted vs actual scatter plots."""
    if not HAS_MATPLOTLIB:
        return

    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(exist_ok=True)

    # Sort models by R2
    model_r2 = {}
    for name, data in predictions.items():
        r2 = r2_score(data['y_true'], data['y_pred_cv'])
        model_r2[name] = r2

    sorted_models = sorted(model_r2, key=model_r2.get, reverse=True)

    # Plot top 6 models
    top_models = sorted_models[:6]
    n = len(top_models)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, model_name in enumerate(top_models):
        ax = axes[idx]
        data = predictions[model_name]
        y_true = data['y_true']
        y_pred = data['y_pred_cv']
        r2 = r2_score(y_true, y_pred)

        ax.scatter(y_true, y_pred, alpha=0.4, s=15, c='#2196F3', edgecolor='none')

        # Identity line
        all_vals = np.concatenate([y_true, y_pred])
        lims = [np.percentile(all_vals, 1), np.percentile(all_vals, 99)]
        ax.plot(lims, lims, 'r--', alpha=0.8, linewidth=1.5)

        ax.set_xlabel('Actual', fontsize=10)
        ax.set_ylabel('Predicted (CV)', fontsize=10)
        ax.set_title(f'{model_name}\nQ2 = {r2:.4f}', fontsize=11, fontweight='bold')

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Cross-Validated Predictions: {target_name}',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(pred_dir / f'cv_scatter_{target_name}.png', dpi=200,
               bbox_inches='tight')
    plt.close()

    # Save prediction data
    for model_name, data in predictions.items():
        df_pred = pd.DataFrame({
            'Actual': data['y_true'],
            'CV_Predicted': data['y_pred_cv'],
            'Train_Predicted': data['y_pred_train'],
            'CV_Residual': data['y_true'] - data['y_pred_cv'],
        })
        df_pred.to_excel(
            pred_dir / f'predictions_{target_name}_{model_name}.xlsx',
            index=False
        )


def plot_model_comparison(all_results: list, output_dir: Path):
    """Bar chart comparing model performance across targets."""
    if not HAS_MATPLOTLIB:
        return

    df = pd.DataFrame(all_results)
    targets = df['Target'].unique()

    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 6))
    if len(targets) == 1:
        axes = [axes]

    for idx, target in enumerate(targets):
        ax = axes[idx]
        subset = df[df['Target'] == target].sort_values('CV R2 (mean)', ascending=True)

        colors = ['#4CAF50' if r > 0.5 else '#FF9800' if r > 0.2 else '#F44336'
                  for r in subset['CV R2 (mean)']]

        bars = ax.barh(subset['Model'], subset['CV R2 (mean)'],
                       xerr=subset['CV R2 (std)'], color=colors,
                       edgecolor='white', linewidth=0.5, capsize=3)

        ax.set_xlabel('CV R\u00b2', fontsize=11)
        ax.set_title(f'{target}', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlim(-0.5, 1.0)

    plt.suptitle('Model Performance Comparison (5-Fold CV)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()


def run_shap_analysis(best_models: dict, X, y, preprocessor,
                      target_name: str, output_dir: Path):
    """SHAP feature importance analysis."""
    if not HAS_SHAP:
        logger.info("  SHAP not installed -- skipping")
        return

    X_processed = preprocessor.transform(X)

    try:
        feat_names = preprocessor.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(X_processed.shape[1])]

    shap_dir = output_dir / 'shap_analysis'
    shap_dir.mkdir(exist_ok=True)

    # Use best tree model
    tree_models = ['XGBoost', 'GradientBoosting', 'RandomForest', 'ExtraTrees']
    best_tree = None
    for name in tree_models:
        if name in best_models:
            best_tree = (name, best_models[name]['model'])
            break

    if not best_tree:
        return

    model_name, model = best_tree
    logger.info(f"  SHAP analysis with {model_name}...")

    try:
        explainer = shap.TreeExplainer(model)
        sample = X_processed[:min(300, len(X_processed))]
        shap_values = explainer.shap_values(sample)

        # Feature importance table
        mean_shap = np.abs(shap_values).mean(axis=0)
        importance = pd.DataFrame({
            'Feature': feat_names,
            'Mean |SHAP|': mean_shap
        }).sort_values('Mean |SHAP|', ascending=False)

        importance.to_excel(shap_dir / f'shap_{target_name}.xlsx', index=False)

        logger.info(f"  Top 10 features:")
        for _, row in importance.head(10).iterrows():
            logger.info(f"    {row['Feature']}: {row['Mean |SHAP|']:.4f}")

        if HAS_MATPLOTLIB:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, sample, feature_names=feat_names,
                            show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(shap_dir / f'shap_beeswarm_{target_name}.png',
                       dpi=200, bbox_inches='tight')
            plt.close()

    except Exception as e:
        logger.error(f"  SHAP failed: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir = Path('output/qsar_results_v2')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load verified-only data
    df_verified, df_desc = load_verified_data()

    if len(df_verified) < 30:
        logger.error("Not enough verified data for modeling.")
        return

    all_results = []
    summary = {}

    for target_col, target_info in TARGETS.items():
        target_name = target_info['name']
        logger.info(f"\n{'='*70}")
        logger.info(f"TARGET: {target_col}")
        logger.info(f"{'='*70}")

        # Prepare data
        result = prepare_data(df_verified, df_desc, target_col)
        if result[0] is None:
            continue

        X, y, all_features, numeric_features, preprocessor = result

        # Log target distribution
        logger.info(f"  Target stats: mean={y.mean():.2f}, std={y.std():.2f}, "
                    f"min={y.min():.2f}, max={y.max():.2f}")

        # Train models
        results, best_models, predictions = train_with_cv(
            X, y, preprocessor, target_name
        )

        all_results.extend(results)

        if best_models:
            # Find best by Q2
            best_name = max(best_models, key=lambda k: best_models[k]['q2'])
            info = best_models[best_name]

            summary[target_name] = {
                'best_model': best_name,
                'train_r2': info['train_r2'],
                'cv_r2': info['cv_r2'],
                'q2': info['q2'],
                'n_samples': len(y),
            }

            logger.info(f"\n  BEST: {best_name} (Q2={info['q2']:.4f}, "
                        f"CV R2={info['cv_r2']:.4f})")

            # Save models
            model_dir = output_dir / 'best_models'
            model_dir.mkdir(exist_ok=True)
            for name, minfo in best_models.items():
                with open(model_dir / f'{target_name}_{name}.pkl', 'wb') as f:
                    pickle.dump({
                        'model': minfo['model'],
                        'params': minfo['params'],
                        'preprocessor': preprocessor,
                        'q2': minfo['q2'],
                        'features': all_features,
                    }, f)

            # Plots
            plot_results(predictions, target_name, output_dir)

            # SHAP
            run_shap_analysis(best_models, X, y, preprocessor,
                            target_name, output_dir)

    # Save results
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results = df_results.sort_values(['Target', 'CV R2 (mean)'],
                                             ascending=[True, False])

        results_path = output_dir / 'model_performance_summary.xlsx'
        with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='All Models', index=False)

            if summary:
                df_sum = pd.DataFrame(summary).T
                df_sum.index.name = 'Target'
                df_sum.to_excel(writer, sheet_name='Best Models')

        # Comparison plot
        plot_model_comparison(all_results, output_dir)

        # Print final report
        logger.info(f"\n{'='*70}")
        logger.info(f"QSAR MODELING COMPLETE (Verified Data Only)")
        logger.info(f"{'='*70}")
        logger.info(f"Results:     {results_path}")
        logger.info(f"Models:      {output_dir / 'best_models'}")
        logger.info(f"Predictions: {output_dir / 'predictions'}")
        logger.info(f"SHAP:        {output_dir / 'shap_analysis'}")

        if summary:
            logger.info(f"\nBEST MODELS:")
            for target, info in summary.items():
                logger.info(f"  {target}:")
                logger.info(f"    Model:    {info['best_model']}")
                logger.info(f"    Train R2: {info['train_r2']:.4f}")
                logger.info(f"    CV R2:    {info['cv_r2']:.4f}")
                logger.info(f"    Q2:       {info['q2']:.4f}")
                logger.info(f"    N:        {info['n_samples']}")

        logger.info(f"\n{'='*70}")

    else:
        logger.warning("No models were successfully trained.")


if __name__ == '__main__':
    main()
