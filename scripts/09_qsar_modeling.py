#!/usr/bin/env python3
"""
QSAR Modeling Pipeline for Phytochemical Extraction Dataset
=============================================================

Run: python scripts/09_qsar_modeling.py

This pipeline builds QSAR models to predict extraction outcomes
(yield, TPC, TFC, IC50) from molecular descriptors + extraction conditions.

Models: Random Forest, Gradient Boosting, XGBoost, Ridge, SVR, MLP
Validation: Literature-verified subset used as external test set
Interpretation: SHAP feature importance analysis

Outputs:
  - output/qsar_results/model_performance_summary.xlsx
  - output/qsar_results/best_models/ (saved .pkl files)
  - output/qsar_results/shap_analysis/ (feature importance plots)
  - output/qsar_results/predictions/ (predicted vs actual)
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
    KFold, cross_val_score, GridSearchCV, RandomizedSearchCV,
    train_test_split
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    explained_variance_score
)

# Models
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('logs/09_qsar_modeling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Targets to model
TARGETS = {
    'Yield (%)': {'name': 'yield', 'min': 0.01, 'max': 100},
    'TPC (mg GAE/g)': {'name': 'tpc', 'min': 0.01, 'max': 1000},
    'TFC (mg QE/g)': {'name': 'tfc', 'min': 0.01, 'max': 1000},
    'Antioxidant Activity (IC50, \u00b5g/mL)': {'name': 'ic50', 'min': 0.001, 'max': 10000},
}

# Molecular descriptor features (from Molecular_Descriptors sheet)
MOLECULAR_FEATURES = [
    'MW (g/mol)', 'LogP (ALogP)', 'TPSA (\u00c5\u00b2)', 'HBD', 'HBA',
    'Rotatable Bonds', 'Aromatic Rings', 'Heavy Atoms',
    'Fraction CSP3', 'Molar Refractivity',
]

# Extraction condition features (numerical)
EXTRACTION_NUMERIC = [
    'Temperature (\u00b0C)', 'Time (min)', 'Pressure (MPa)',
    'Power (W)', 'Frequency (kHz)', 'pH',
]

# Categorical features
EXTRACTION_CATEGORICAL = [
    'Extraction Method', 'Solvent System', 'Phytochemical Class',
    'Scale', 'Pretreatment',
]

# Model configurations
MODELS = {
    'RandomForest': {
        'estimator': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.5],
        }
    },
    'GradientBoosting': {
        'estimator': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
        }
    },
    'Ridge': {
        'estimator': Ridge(),
        'params': {
            'alpha': [0.01, 0.1, 1.0, 10, 100, 1000],
        }
    },
    'SVR': {
        'estimator': SVR(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.5],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto'],
        }
    },
    'MLP': {
        'estimator': MLPRegressor(random_state=42, max_iter=1000,
                                   early_stopping=True, validation_fraction=0.15),
        'params': {
            'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01],
            'activation': ['relu', 'tanh'],
        }
    },
    'KNN': {
        'estimator': KNeighborsRegressor(n_jobs=-1),
        'params': {
            'n_neighbors': [3, 5, 7, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
        }
    },
    'ExtraTrees': {
        'estimator': ExtraTreesRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
        }
    },
}

# Try importing XGBoost (optional)
try:
    from xgboost import XGBRegressor
    MODELS['XGBoost'] = {
        'estimator': XGBRegressor(random_state=42, n_jobs=-1,
                                   verbosity=0, tree_method='hist'),
        'params': {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 5, 10],
        }
    }
    logger.info("XGBoost available")
except ImportError:
    logger.info("XGBoost not installed (pip install xgboost) — skipping")


# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================

def load_data() -> tuple:
    """Load extraction dataset and molecular descriptors."""

    # Load main dataset
    dataset_path = Path('data/processed/dataset_with_replacements.xlsx')
    if not dataset_path.exists():
        dataset_path = Path('output/Phytochemical_Extraction_Dataset_FINAL_20260224.xlsx')

    logger.info(f"Loading dataset from {dataset_path}...")
    df = pd.read_excel(dataset_path)
    logger.info(f"Dataset shape: {df.shape}")

    # Load molecular descriptors
    desc_path = Path('data/processed/molecular_descriptors_rdkit.xlsx')
    if desc_path.exists():
        df_desc = pd.read_excel(desc_path)
        logger.info(f"Molecular descriptors: {df_desc.shape}")
    else:
        df_desc = None
        logger.warning("Molecular descriptors not found")

    return df, df_desc


def prepare_features(df: pd.DataFrame, df_desc: pd.DataFrame,
                     target_col: str) -> tuple:
    """
    Prepare feature matrix X and target vector y.

    Returns: X_train, X_test, y_train, y_test, feature_names, preprocessor
    """

    # Merge molecular descriptors
    if df_desc is not None:
        # Match on compound name
        merge_cols = ['Phytochemical Name'] + [c for c in MOLECULAR_FEATURES if c in df_desc.columns]
        df_merged = df.merge(
            df_desc[merge_cols].drop_duplicates('Phytochemical Name'),
            on='Phytochemical Name', how='left', suffixes=('', '_desc')
        )
    else:
        df_merged = df.copy()

    # Filter rows with valid target
    mask = df_merged[target_col].notna()
    target_info = TARGETS[target_col]
    mask &= df_merged[target_col] >= target_info['min']
    mask &= df_merged[target_col] <= target_info['max']
    df_valid = df_merged[mask].copy()

    logger.info(f"Valid rows for {target_col}: {len(df_valid)}")

    if len(df_valid) < 100:
        logger.warning(f"Too few valid rows ({len(df_valid)}) for {target_col}")
        return None, None, None, None, None, None

    # Build feature lists
    numeric_features = []
    for col in EXTRACTION_NUMERIC + MOLECULAR_FEATURES:
        if col in df_valid.columns and col != target_col:
            numeric_features.append(col)

    categorical_features = []
    for col in EXTRACTION_CATEGORICAL:
        if col in df_valid.columns:
            # Limit cardinality
            top_values = df_valid[col].value_counts().head(20).index
            df_valid.loc[~df_valid[col].isin(top_values), col] = 'Other'
            categorical_features.append(col)

    all_features = numeric_features + categorical_features
    logger.info(f"Features: {len(numeric_features)} numeric + {len(categorical_features)} categorical")

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
                                   max_categories=20)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='drop'
    )

    # Split: use literature-verified as test set if available
    if '_data_source' in df_valid.columns:
        verified_mask = df_valid['_data_source'] != 'synthetic'
        n_verified = verified_mask.sum()

        if n_verified >= 50:
            # Use verified data as external test set
            X_train = X[~verified_mask]
            y_train = y[~verified_mask]
            X_test = X[verified_mask]
            y_test = y[verified_mask]
            logger.info(f"Split: {len(X_train)} train (synthetic) | {len(X_test)} test (literature-verified)")
        else:
            # Not enough verified data — random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            logger.info(f"Split: {len(X_train)} train | {len(X_test)} test (random 80/20)")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info(f"Split: {len(X_train)} train | {len(X_test)} test (random 80/20)")

    return X_train, X_test, y_train, y_test, all_features, preprocessor


# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    """Calculate comprehensive regression metrics."""
    metrics = {
        'Model': model_name,
        'R2': round(r2_score(y_true, y_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        'MAE': round(mean_absolute_error(y_true, y_pred), 4),
        'Explained Variance': round(explained_variance_score(y_true, y_pred), 4),
    }

    # Percentage metrics
    if np.mean(np.abs(y_true)) > 0:
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        metrics['MAPE (%)'] = round(mape, 2)

    # Pearson correlation
    r, p = stats.pearsonr(y_true, y_pred)
    metrics['Pearson r'] = round(r, 4)
    metrics['p-value'] = f"{p:.2e}"

    return metrics


def train_and_evaluate(X_train, X_test, y_train, y_test,
                       preprocessor, target_name: str) -> tuple:
    """Train all models and return results."""

    results = []
    best_models = {}
    predictions = {}

    # Fit preprocessor
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logger.info(f"\nProcessed features shape: {X_train_processed.shape}")

    for model_name, config in MODELS.items():
        logger.info(f"\n  Training {model_name}...")

        try:
            estimator = config['estimator']
            param_grid = config['params']

            # Use RandomizedSearchCV for large param spaces
            n_combinations = 1
            for v in param_grid.values():
                n_combinations *= len(v)

            if n_combinations > 100:
                search = RandomizedSearchCV(
                    estimator, param_grid,
                    n_iter=50, cv=5, scoring='r2',
                    random_state=42, n_jobs=-1, verbose=0
                )
            else:
                search = GridSearchCV(
                    estimator, param_grid,
                    cv=5, scoring='r2',
                    n_jobs=-1, verbose=0
                )

            search.fit(X_train_processed, y_train)

            # Best model predictions
            y_pred_train = search.best_estimator_.predict(X_train_processed)
            y_pred_test = search.best_estimator_.predict(X_test_processed)

            # Evaluate
            train_metrics = evaluate_model(y_train, y_pred_train, f"{model_name} (train)")
            test_metrics = evaluate_model(y_test, y_pred_test, f"{model_name} (test)")

            # Cross-validation on training set
            cv_scores = cross_val_score(search.best_estimator_, X_train_processed,
                                        y_train, cv=5, scoring='r2', n_jobs=-1)

            test_metrics['CV R2 (mean)'] = round(cv_scores.mean(), 4)
            test_metrics['CV R2 (std)'] = round(cv_scores.std(), 4)
            test_metrics['Best Params'] = str(search.best_params_)

            results.append(train_metrics)
            results.append(test_metrics)

            best_models[model_name] = {
                'model': search.best_estimator_,
                'params': search.best_params_,
                'test_r2': test_metrics['R2'],
                'cv_r2_mean': cv_scores.mean(),
            }

            predictions[model_name] = {
                'y_true': y_test,
                'y_pred': y_pred_test,
            }

            logger.info(f"    Train R\u00b2: {train_metrics['R2']:.4f}")
            logger.info(f"    Test  R\u00b2: {test_metrics['R2']:.4f}")
            logger.info(f"    CV    R\u00b2: {cv_scores.mean():.4f} \u00b1 {cv_scores.std():.4f}")

        except Exception as e:
            logger.error(f"    {model_name} failed: {e}")
            continue

    return results, best_models, predictions


# =============================================================================
# SHAP ANALYSIS
# =============================================================================

def run_shap_analysis(best_models: dict, X_train, X_test, preprocessor,
                      feature_names: list, target_name: str, output_dir: Path):
    """Generate SHAP feature importance analysis."""
    try:
        import shap
    except ImportError:
        logger.info("SHAP not installed (pip install shap) — skipping analysis")
        return

    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after preprocessing
    try:
        proc_feature_names = preprocessor.get_feature_names_out()
    except Exception:
        proc_feature_names = [f"feature_{i}" for i in range(X_test_processed.shape[1])]

    shap_dir = output_dir / 'shap_analysis'
    shap_dir.mkdir(exist_ok=True)

    # Use the best tree-based model for SHAP
    tree_models = ['XGBoost', 'RandomForest', 'GradientBoosting', 'ExtraTrees']
    best_tree = None
    for name in tree_models:
        if name in best_models:
            best_tree = (name, best_models[name]['model'])
            break

    if best_tree is None:
        logger.info("No tree model available for SHAP")
        return

    model_name, model = best_tree
    logger.info(f"\n  Running SHAP analysis with {model_name}...")

    try:
        # Use TreeExplainer for tree models
        explainer = shap.TreeExplainer(model)
        # Limit to 500 samples for speed
        sample_size = min(500, X_test_processed.shape[0])
        X_sample = X_test_processed[:sample_size]
        shap_values = explainer.shap_values(X_sample)

        # Save SHAP summary data
        shap_df = pd.DataFrame(
            shap_values, columns=proc_feature_names
        )
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

        importance_df = pd.DataFrame({
            'Feature': mean_abs_shap.index,
            'Mean |SHAP|': mean_abs_shap.values
        })
        importance_df.to_excel(
            shap_dir / f'shap_importance_{target_name}.xlsx', index=False
        )

        # Save top 20 features
        logger.info(f"\n  Top 10 SHAP features for {target_name}:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"    {row['Feature']}: {row['Mean |SHAP|']:.4f}")

        # Generate SHAP plots
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Summary plot
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample,
                            feature_names=proc_feature_names,
                            show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(shap_dir / f'shap_summary_{target_name}.png', dpi=150,
                       bbox_inches='tight')
            plt.close()

            # Bar plot
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample,
                            feature_names=proc_feature_names,
                            plot_type='bar', show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(shap_dir / f'shap_bar_{target_name}.png', dpi=150,
                       bbox_inches='tight')
            plt.close()

            logger.info(f"  SHAP plots saved to {shap_dir}")

        except ImportError:
            logger.info("  matplotlib not available — SHAP plots skipped")

    except Exception as e:
        logger.error(f"  SHAP analysis failed: {e}")


# =============================================================================
# RESULTS EXPORT
# =============================================================================

def save_predictions(predictions: dict, target_name: str, output_dir: Path):
    """Save predicted vs actual values for all models."""
    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(exist_ok=True)

    for model_name, pred_data in predictions.items():
        df_pred = pd.DataFrame({
            'Actual': pred_data['y_true'],
            'Predicted': pred_data['y_pred'],
            'Residual': pred_data['y_true'] - pred_data['y_pred'],
        })
        df_pred.to_excel(
            pred_dir / f'predictions_{target_name}_{model_name}.xlsx',
            index=False
        )

    # Generate scatter plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_models = len(predictions)
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(5 * ((n_models + 1) // 2), 10))
        axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (model_name, pred_data) in enumerate(predictions.items()):
            if idx >= len(axes):
                break
            ax = axes[idx]
            y_true = pred_data['y_true']
            y_pred = pred_data['y_pred']
            r2 = r2_score(y_true, y_pred)

            ax.scatter(y_true, y_pred, alpha=0.3, s=10, c='steelblue')
            lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            ax.plot(lims, lims, 'r--', alpha=0.8, linewidth=1)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{model_name}\nR\u00b2 = {r2:.4f}')
            ax.set_aspect('equal', adjustable='box')

        # Hide empty subplots
        for idx in range(len(predictions), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'Predicted vs Actual: {target_name}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(pred_dir / f'scatter_{target_name}.png', dpi=150,
                   bbox_inches='tight')
        plt.close()

    except ImportError:
        pass


def save_models(best_models: dict, preprocessor, target_name: str, output_dir: Path):
    """Save trained models as pickle files."""
    model_dir = output_dir / 'best_models'
    model_dir.mkdir(exist_ok=True)

    for model_name, model_info in best_models.items():
        filepath = model_dir / f'{target_name}_{model_name}.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': model_info['model'],
                'params': model_info['params'],
                'preprocessor': preprocessor,
                'test_r2': model_info['test_r2'],
                'cv_r2_mean': model_info['cv_r2_mean'],
            }, f)

    logger.info(f"  Models saved to {model_dir}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # Create output directory
    output_dir = Path('output/qsar_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'best_models').mkdir(exist_ok=True)
    (output_dir / 'predictions').mkdir(exist_ok=True)
    (output_dir / 'shap_analysis').mkdir(exist_ok=True)

    # Load data
    df, df_desc = load_data()

    all_results = []
    summary = {}

    for target_col, target_info in TARGETS.items():
        target_name = target_info['name']
        logger.info(f"\n{'='*70}")
        logger.info(f"MODELING TARGET: {target_col}")
        logger.info(f"{'='*70}")

        # Prepare features
        X_train, X_test, y_train, y_test, features, preprocessor = \
            prepare_features(df, df_desc, target_col)

        if X_train is None:
            logger.warning(f"Skipping {target_col} — insufficient data")
            continue

        # Train and evaluate all models
        results, best_models, predictions = \
            train_and_evaluate(X_train, X_test, y_train, y_test,
                             preprocessor, target_name)

        # Add target info to results
        for r in results:
            r['Target'] = target_col

        all_results.extend(results)

        # Find best model
        if best_models:
            best_name = max(best_models, key=lambda k: best_models[k]['test_r2'])
            best_r2 = best_models[best_name]['test_r2']
            logger.info(f"\n  BEST MODEL: {best_name} (Test R\u00b2 = {best_r2:.4f})")

            summary[target_name] = {
                'best_model': best_name,
                'test_r2': best_r2,
                'cv_r2': best_models[best_name]['cv_r2_mean'],
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_features': X_train.shape[1],
            }

            # Save models
            save_models(best_models, preprocessor, target_name, output_dir)

            # Save predictions
            save_predictions(predictions, target_name, output_dir)

            # SHAP analysis
            run_shap_analysis(best_models, X_train, X_test, preprocessor,
                            features, target_name, output_dir)

    # Save comprehensive results
    if all_results:
        df_results = pd.DataFrame(all_results)

        # Sort by target and R2
        df_results = df_results.sort_values(['Target', 'R2'], ascending=[True, False])

        results_path = output_dir / 'model_performance_summary.xlsx'
        with pd.ExcelWriter(results_path) as writer:
            df_results.to_excel(writer, sheet_name='All Models', index=False)

            # Summary sheet
            if summary:
                df_summary = pd.DataFrame(summary).T
                df_summary.index.name = 'Target'
                df_summary.to_excel(writer, sheet_name='Best Models')

        logger.info(f"\n{'='*70}")
        logger.info(f"QSAR MODELING COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Results: {results_path}")
        logger.info(f"Models:  {output_dir / 'best_models'}")
        logger.info(f"Plots:   {output_dir / 'predictions'}")
        logger.info(f"SHAP:    {output_dir / 'shap_analysis'}")

        if summary:
            logger.info(f"\nBEST MODELS SUMMARY:")
            for target, info in summary.items():
                logger.info(f"  {target}: {info['best_model']} "
                           f"(R\u00b2={info['test_r2']:.4f}, "
                           f"CV R\u00b2={info['cv_r2']:.4f})")

        logger.info(f"\n{'='*70}")


if __name__ == '__main__':
    main()
