"""
Script to train ML models for MEV detection and price prediction
"""

import os
import logging
import time
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_lstm_models():
    """Train LSTM models for price prediction"""
    logger.info("Training LSTM models")
    
    # Import required modules
    from ml_model.lstm_model import get_model_trainer, train_all_models
    
    # Train LSTM models
    lstm_results = train_all_models(
        data_dir='data/processed',
        model_dir='models',
        plots_dir='plots',
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=50,
        patience=10
    )
    
    return lstm_results

def train_arbitrage_models():
    """Train models for arbitrage opportunity prediction"""
    logger.info("Training arbitrage models")
    
    results = {
        'direct': None,
        'triangular': None
    }
    
    # Check if we have arbitrage data
    direct_train_path = 'data/processed/arbitrage_direct_train.csv'
    triangular_train_path = 'data/processed/arbitrage_triangular_train.csv'
    
    # Train model for direct arbitrage
    if os.path.exists(direct_train_path):
        logger.info("Training direct arbitrage model")
        results['direct'] = train_arbitrage_model('direct')
    
    # Train model for triangular arbitrage
    if os.path.exists(triangular_train_path):
        logger.info("Training triangular arbitrage model")
        results['triangular'] = train_arbitrage_model('triangular')
    
    return results

def train_arbitrage_model(arbitrage_type: str):
    """Train model for arbitrage opportunity prediction"""
    train_path = f'data/processed/arbitrage_{arbitrage_type}_train.csv'
    test_path = f'data/processed/arbitrage_{arbitrage_type}_test.csv'
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.warning(f"Training data not found for {arbitrage_type} arbitrage")
        return None
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    if 'target' not in train_df.columns:
        logger.warning(f"Target column not found in {train_path}")
        return None
    
    # Prepare features
    excluded_cols = ['timestamp', 'datetime', 'target', 'path', 'exchange', 
                    'symbol', 'exchange1', 'exchange2']
    feature_cols = [col for col in train_df.columns if col not in excluded_cols]
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    y_train = train_df['target'].values
    
    X_test = scaler.transform(test_df[feature_cols])
    y_test = test_df['target'].values
    
    # Save scaler
    import joblib
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    scaler_path = os.path.join(model_dir, f'scaler_{arbitrage_type}_arbitrage.joblib')
    joblib.dump(scaler, scaler_path)
    
    # Train logistic regression model
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    logger.info(f"{arbitrage_type.capitalize()} Arbitrage Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save model
    model_path = os.path.join(model_dir, f'logreg_{arbitrage_type}_arbitrage.joblib')
    joblib.dump(model, model_path)
    
    # Save feature importances
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.coef_[0]
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    importance_path = os.path.join(model_dir, f'feature_importance_{arbitrage_type}_arbitrage.csv')
    feature_importance.to_csv(importance_path, index=False)
    
    # Create ROC curve
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {arbitrage_type.capitalize()} Arbitrage')
    plt.legend()
    
    roc_path = os.path.join(plots_dir, f'roc_{arbitrage_type}_arbitrage.png')
    plt.savefig(roc_path)
    plt.close()
    
    # Train Random Forest model
    from sklearn.ensemble import RandomForestClassifier
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate Random Forest model
    rf_y_pred = rf_model.predict(X_test)
    rf_y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    rf_metrics = {
        'accuracy': accuracy_score(y_test, rf_y_pred),
        'precision': precision_score(y_test, rf_y_pred, zero_division=0),
        'recall': recall_score(y_test, rf_y_pred, zero_division=0),
        'f1': f1_score(y_test, rf_y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, rf_y_pred_proba)
    }
    
    logger.info(f"{arbitrage_type.capitalize()} Arbitrage RF Metrics:")
    for metric, value in rf_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save Random Forest model
    rf_model_path = os.path.join(model_dir, f'rf_{arbitrage_type}_arbitrage.joblib')
    joblib.dump(rf_model, rf_model_path)
    
    # Save Random Forest feature importances
    rf_feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    })
    rf_feature_importance = rf_feature_importance.sort_values('importance', ascending=False)
    
    rf_importance_path = os.path.join(model_dir, f'rf_feature_importance_{arbitrage_type}_arbitrage.csv')
    rf_feature_importance.to_csv(rf_importance_path, index=False)
    
    # Training XGBoost model if XGBoost is available
    try:
        import xgboost as xgb
        logger.info(f"Training XGBoost model for {arbitrage_type} arbitrage")
        
        xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        
        # Evaluate XGBoost model
        xgb_y_pred = xgb_model.predict(X_test)
        xgb_y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        xgb_metrics = {
            'accuracy': accuracy_score(y_test, xgb_y_pred),
            'precision': precision_score(y_test, xgb_y_pred, zero_division=0),
            'recall': recall_score(y_test, xgb_y_pred, zero_division=0),
            'f1': f1_score(y_test, xgb_y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, xgb_y_pred_proba)
        }
        
        logger.info(f"{arbitrage_type.capitalize()} Arbitrage XGBoost Metrics:")
        for metric, value in xgb_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save XGBoost model
        xgb_model_path = os.path.join(model_dir, f'xgb_{arbitrage_type}_arbitrage.json')
        xgb_model.save_model(xgb_model_path)
        
        # Save XGBoost feature importances
        xgb_feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': xgb_model.feature_importances_
        })
        xgb_feature_importance = xgb_feature_importance.sort_values('importance', ascending=False)
        
        xgb_importance_path = os.path.join(model_dir, f'xgb_feature_importance_{arbitrage_type}_arbitrage.csv')
        xgb_feature_importance.to_csv(xgb_importance_path, index=False)
        
        # Save all metrics
        all_metrics = {
            'logistic_regression': metrics,
            'random_forest': rf_metrics,
            'xgboost': xgb_metrics
        }
    except:
        logger.warning("XGBoost not available, skipping XGBoost model")
        # Save metrics without XGBoost
        all_metrics = {
            'logistic_regression': metrics,
            'random_forest': rf_metrics
        }
    
    # Save all metrics
    metrics_path = os.path.join(model_dir, f'metrics_{arbitrage_type}_arbitrage.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    return all_metrics

def main():
    """Main function to train all models"""
    start_time = time.time()
    logger.info("Starting model training")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Train LSTM models
    lstm_results = train_lstm_models()
    
    # Train arbitrage models
    arbitrage_results = train_arbitrage_models()
    
    # Save overall results
    results = {
        'lstm': lstm_results,
        'arbitrage': arbitrage_results,
        'training_time': time.time() - start_time,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('models/training_results.json', 'w') as f:
        # Convert non-serializable types (e.g., numpy.float32)
        json_results = json.dumps(results, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        f.write(json_results)
    
    logger.info(f"Model training completed in {time.time() - start_time:.2f} seconds")
    
    return results

if __name__ == "__main__":
    main()