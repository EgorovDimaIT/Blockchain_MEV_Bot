"""
Script to collect data from exchanges, process it, and train ML models 
for both price prediction and MEV opportunity detection
"""

import os
import logging
import time
from datetime import datetime
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

# Import our custom modules
from data_collection.exchange_data_downloader import ExchangeDataDownloader
from ml_model.data_processor import DataProcessor
from ml_model.lstm_model import LSTMTrainer, train_all_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MEVPredictionModel:
    """Class for training models to predict MEV opportunities"""
    
    def __init__(self, data_dir: str = 'data/processed', model_dir: str = 'models'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
    def train_arbitrage_models(self, arbitrage_type: str = 'direct') -> Dict:
        """
        Train models to predict arbitrage opportunities
        
        Args:
            arbitrage_type: Type of arbitrage ('direct' or 'triangular')
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {arbitrage_type} arbitrage prediction models")
        
        # Define file paths
        train_path = os.path.join(self.data_dir, f'arbitrage_{arbitrage_type}_train.csv')
        test_path = os.path.join(self.data_dir, f'arbitrage_{arbitrage_type}_test.csv')
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logger.error(f"Arbitrage data files not found: {train_path} or {test_path}")
            return {}
            
        # Load data
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info(f"Loaded arbitrage data: {len(train_df)} training samples, {len(test_df)} testing samples")
            
            # Check if target column exists
            if 'target' not in train_df.columns or 'target' not in test_df.columns:
                logger.error(f"Target column not found in arbitrage data")
                return {}
                
            # Prepare features and target
            if arbitrage_type == 'direct':
                feature_cols = [col for col in train_df.columns if col not in ['timestamp', 'datetime', 'target', 'symbol', 'exchange1', 'exchange2']]
            else:  # triangular
                feature_cols = [col for col in train_df.columns if col not in ['timestamp', 'datetime', 'target', 'path', 'exchange']]
                
            # Now train different models
            results = {}
            
            # Train logistic regression model
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                # Scale features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(train_df[feature_cols])
                X_test = scaler.transform(test_df[feature_cols])
                
                y_train = train_df['target']
                y_test = test_df['target']
                
                # Train logistic regression
                model = LogisticRegression(random_state=42, class_weight='balanced')
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba)
                }
                
                logger.info(f"Logistic Regression metrics for {arbitrage_type} arbitrage: {metrics}")
                
                # Save model
                import joblib
                model_path = os.path.join(self.model_dir, f'logistic_regression_{arbitrage_type}_arbitrage.joblib')
                joblib.dump(model, model_path)
                
                # Save scaler
                scaler_path = os.path.join(self.model_dir, f'scaler_{arbitrage_type}_arbitrage.joblib')
                joblib.dump(scaler, scaler_path)
                
                # Store results
                results['logistic_regression'] = {
                    'metrics': metrics,
                    'model_path': model_path,
                    'scaler_path': scaler_path,
                    'feature_cols': feature_cols
                }
                
            except Exception as e:
                logger.error(f"Error training logistic regression for {arbitrage_type} arbitrage: {e}")
            
            # Train random forest model
            try:
                from sklearn.ensemble import RandomForestClassifier
                
                # Train random forest
                model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba)
                }
                
                logger.info(f"Random Forest metrics for {arbitrage_type} arbitrage: {metrics}")
                
                # Get feature importances
                feature_importances = list(zip(feature_cols, model.feature_importances_))
                feature_importances.sort(key=lambda x: x[1], reverse=True)
                
                logger.info(f"Top 5 features for {arbitrage_type} arbitrage: {feature_importances[:5]}")
                
                # Save model
                model_path = os.path.join(self.model_dir, f'random_forest_{arbitrage_type}_arbitrage.joblib')
                joblib.dump(model, model_path)
                
                # Store results
                results['random_forest'] = {
                    'metrics': metrics,
                    'model_path': model_path,
                    'scaler_path': scaler_path,  # Use same scaler as logistic regression
                    'feature_cols': feature_cols,
                    'feature_importances': feature_importances
                }
                
            except Exception as e:
                logger.error(f"Error training random forest for {arbitrage_type} arbitrage: {e}")
            
            # Train XGBoost model if available
            try:
                import xgboost as xgb
                
                # Create DMatrix
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)
                
                # Set parameters
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'max_depth': 4,
                    'eta': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'seed': 42
                }
                
                # Train model
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=100,
                    evals=[(dtrain, 'train'), (dtest, 'test')],
                    early_stopping_rounds=10,
                    verbose_eval=False
                )
                
                # Evaluate
                y_pred_proba = model.predict(dtest)
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba)
                }
                
                logger.info(f"XGBoost metrics for {arbitrage_type} arbitrage: {metrics}")
                
                # Get feature importances
                feature_importances = model.get_score(importance_type='gain')
                feature_importances = [(feature_cols[int(k.replace('f', ''))], v) for k, v in feature_importances.items()]
                feature_importances.sort(key=lambda x: x[1], reverse=True)
                
                logger.info(f"Top 5 XGBoost features for {arbitrage_type} arbitrage: {feature_importances[:5]}")
                
                # Save model
                model_path = os.path.join(self.model_dir, f'xgboost_{arbitrage_type}_arbitrage.json')
                model.save_model(model_path)
                
                # Store results
                results['xgboost'] = {
                    'metrics': metrics,
                    'model_path': model_path,
                    'scaler_path': scaler_path,  # Use same scaler as logistic regression
                    'feature_cols': feature_cols,
                    'feature_importances': feature_importances
                }
                
            except Exception as e:
                logger.error(f"Error training XGBoost for {arbitrage_type} arbitrage: {e}")
            
            # Save results summary
            summary_path = os.path.join(self.model_dir, f'{arbitrage_type}_arbitrage_models_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
            logger.info(f"Arbitrage models training completed, summary saved to {summary_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training arbitrage models: {e}")
            return {}
    
    def train_sandwich_models(self) -> Dict:
        """
        Train models to predict sandwich attack opportunities
        
        Returns:
            Dictionary with training results
        """
        logger.warning("Sandwich attack model training not implemented yet")
        # This would be similar to the arbitrage models but with sandwich-specific features
        return {}
    
    def train_all_mev_models(self) -> Dict:
        """
        Train all MEV opportunity prediction models
        
        Returns:
            Dictionary with training results
        """
        logger.info("Training all MEV opportunity prediction models")
        
        results = {
            'direct_arbitrage': self.train_arbitrage_models('direct'),
            'triangular_arbitrage': self.train_arbitrage_models('triangular'),
            'sandwich': self.train_sandwich_models()
        }
        
        # Save results summary
        summary_path = os.path.join(self.model_dir, 'mev_models_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"All MEV models training completed, summary saved to {summary_path}")
        
        return results


def run_data_collection_and_training():
    """
    Run the full pipeline: collect data, process it, and train models
    
    Steps:
    1. Download data from exchanges
    2. Process the data and add technical indicators
    3. Train LSTM models for price prediction
    4. Train models for MEV opportunity prediction
    """
    logger.info("Starting full data collection and model training pipeline")
    start_time = time.time()
    
    # Step 1: Download data from exchanges
    logger.info("Step 1: Downloading data from exchanges")
    downloader = ExchangeDataDownloader(days_to_fetch=7)
    data = downloader.run(include_orderbooks=False, create_arbitrage_datasets=True)
    
    # Log completion of step 1
    logger.info(f"Step 1 completed in {time.time() - start_time:.2f} seconds")
    step_time = time.time()
    
    # Step 2: Process data
    logger.info("Step 2: Processing data")
    processor = DataProcessor(input_dir='data', output_dir='data/processed')
    processed_data = processor.run()
    
    # Log completion of step 2
    logger.info(f"Step 2 completed in {time.time() - step_time:.2f} seconds")
    step_time = time.time()
    
    # Step 3: Train LSTM models for price prediction
    logger.info("Step 3: Training LSTM models for price prediction")
    lstm_results = train_all_models(data_dir='data/processed', model_dir='models', plots_dir='plots')
    
    # Log completion of step 3
    logger.info(f"Step 3 completed in {time.time() - step_time:.2f} seconds")
    step_time = time.time()
    
    # Step 4: Train models for MEV opportunity prediction
    logger.info("Step 4: Training models for MEV opportunity prediction")
    mev_model = MEVPredictionModel(data_dir='data/processed', model_dir='models')
    mev_results = mev_model.train_all_mev_models()
    
    # Log completion of step 4
    logger.info(f"Step 4 completed in {time.time() - step_time:.2f} seconds")
    
    # Log overall completion
    total_time = time.time() - start_time
    logger.info(f"Full pipeline completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Return all results
    return {
        'exchange_data': data,
        'processed_data': processed_data,
        'lstm_results': lstm_results,
        'mev_results': mev_results
    }


if __name__ == "__main__":
    # Run the full pipeline
    results = run_data_collection_and_training()
    
    # Print summary
    print("\n=== Data Collection and Model Training Summary ===")
    
    if results['exchange_data']:
        ohlcv_count = sum(len(exchange_data) for exchange_data in results['exchange_data']['ohlcv_data'].values()) if results['exchange_data']['ohlcv_data'] else 0
        print(f"Downloaded OHLCV datasets: {ohlcv_count}")
        
        tri_arb_count = len(results['exchange_data']['triangular_arbitrage']) if results['exchange_data']['triangular_arbitrage'] is not None else 0
        print(f"Triangular arbitrage opportunities: {tri_arb_count}")
        
        dir_arb_count = len(results['exchange_data']['direct_arbitrage']) if results['exchange_data']['direct_arbitrage'] is not None else 0
        print(f"Direct arbitrage opportunities: {dir_arb_count}")
        
    if results['processed_data']:
        ohlcv_processed = len(results['processed_data']['ohlcv']['processed_files']) if 'ohlcv' in results['processed_data'] and results['processed_data']['ohlcv'] else 0
        print(f"Processed OHLCV files: {ohlcv_processed}")
        
        lstm_datasets = len(results['processed_data']['ohlcv']['lstm_data']) if 'ohlcv' in results['processed_data'] and results['processed_data']['ohlcv'] else 0
        print(f"LSTM datasets created: {lstm_datasets}")
        
    if results['lstm_results']:
        print(f"LSTM models trained: {len(results['lstm_results'])}")
        
        # Print metrics for each model
        for dataset, result in results['lstm_results'].items():
            if 'metrics' in result:
                print(f"\nLSTM model for {dataset}:")
                print(f"  RMSE: {result['metrics'].get('rmse', 'N/A')}")
                print(f"  MAE: {result['metrics'].get('mae', 'N/A')}")
                print(f"  Directional Accuracy: {result['metrics'].get('directional_accuracy', 'N/A')}")
                
    if results['mev_results']:
        # Print direct arbitrage results
        if 'direct_arbitrage' in results['mev_results'] and results['mev_results']['direct_arbitrage']:
            print("\nDirect Arbitrage Models:")
            
            for model_type, model_result in results['mev_results']['direct_arbitrage'].items():
                if 'metrics' in model_result:
                    print(f"  {model_type.upper()}:")
                    print(f"    Accuracy: {model_result['metrics'].get('accuracy', 'N/A'):.4f}")
                    print(f"    Precision: {model_result['metrics'].get('precision', 'N/A'):.4f}")
                    print(f"    Recall: {model_result['metrics'].get('recall', 'N/A'):.4f}")
                    print(f"    F1: {model_result['metrics'].get('f1', 'N/A'):.4f}")
                    print(f"    ROC AUC: {model_result['metrics'].get('roc_auc', 'N/A'):.4f}")
                    
        # Print triangular arbitrage results
        if 'triangular_arbitrage' in results['mev_results'] and results['mev_results']['triangular_arbitrage']:
            print("\nTriangular Arbitrage Models:")
            
            for model_type, model_result in results['mev_results']['triangular_arbitrage'].items():
                if 'metrics' in model_result:
                    print(f"  {model_type.upper()}:")
                    print(f"    Accuracy: {model_result['metrics'].get('accuracy', 'N/A'):.4f}")
                    print(f"    Precision: {model_result['metrics'].get('precision', 'N/A'):.4f}")
                    print(f"    Recall: {model_result['metrics'].get('recall', 'N/A'):.4f}")
                    print(f"    F1: {model_result['metrics'].get('f1', 'N/A'):.4f}")
                    print(f"    ROC AUC: {model_result['metrics'].get('roc_auc', 'N/A'):.4f}")