"""
Script to train all ML models using processed data
"""

import os
import logging
import time
import json
import concurrent.futures
from typing import Dict, List, Any
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

def train_lstm_model(symbol: str, data_dir: str, model_dir: str, plots_dir: str, **kwargs):
    """
    Train LSTM model for a specific symbol
    
    Args:
        symbol: Symbol name (e.g., 'binance_ATHUSD')
        data_dir: Directory with processed data
        model_dir: Directory to save models
        plots_dir: Directory to save plots
        **kwargs: Additional parameters for model training
    """
    logger.info(f"Training LSTM model for {symbol}")
    
    # Import LSTM training function
    from train_lstm_ethweur import train_ethweur_model
    
    # Adjust symbol name to match file pattern
    if not symbol.startswith('uniswap_') and not symbol.startswith('binance_') and not symbol.startswith('kraken_'):
        if 'ETH' in symbol:
            symbol = f'uniswap_{symbol}'
        elif symbol in ['ATHUSD']:
            symbol = f'binance_{symbol}'
        else:
            symbol = f'kraken_{symbol}'
    
    # Prepare parameters
    params = {
        'data_dir': data_dir,
        'model_dir': model_dir,
        'plots_dir': plots_dir,
        'hidden_size': kwargs.get('hidden_size', 128),
        'num_layers': kwargs.get('num_layers', 2),
        'dropout': kwargs.get('dropout', 0.3),
        'learning_rate': kwargs.get('learning_rate', 0.001),
        'batch_size': kwargs.get('batch_size', 64),
        'num_epochs': kwargs.get('num_epochs', 50),
        'patience': kwargs.get('patience', 10)
    }
    
    try:
        # Check if data exists
        if not os.path.exists(os.path.join(data_dir, f'lstm_{symbol}_X_train.npy')):
            logger.warning(f"Data for {symbol} does not exist. Skipping.")
            return None
        
        # Replace symbol in train function
        temp_train_ethweur = train_ethweur_model
        
        # Call the function with updated parameters
        def train_fn():
            return temp_train_ethweur(**params)
        
        return train_fn()
    except Exception as e:
        logger.error(f"Error training model for {symbol}: {e}")
        return None

def find_available_symbols(data_dir: str) -> List[str]:
    """
    Find all available symbols for training
    
    Args:
        data_dir: Directory with processed data
        
    Returns:
        List of available symbols
    """
    symbols = []
    
    # Find all files matching the pattern lstm_*_X_train.npy
    for file in os.listdir(data_dir):
        if file.startswith('lstm_') and file.endswith('_X_train.npy'):
            # Extract symbol from file name
            symbol = file.replace('lstm_', '').replace('_X_train.npy', '')
            symbols.append(symbol)
    
    return symbols

def train_all_lstm_models(data_dir: str, model_dir: str, plots_dir: str, max_workers: int = 1, **kwargs) -> Dict[str, Any]:
    """
    Train LSTM models for all available symbols
    
    Args:
        data_dir: Directory with processed data
        model_dir: Directory to save models
        plots_dir: Directory to save plots
        max_workers: Maximum number of parallel workers
        **kwargs: Additional parameters for model training
        
    Returns:
        Dictionary with training results
    """
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Find available symbols
    symbols = find_available_symbols(data_dir)
    logger.info(f"Found {len(symbols)} symbols for training: {symbols}")
    
    results = {}
    
    # Use ThreadPoolExecutor for parallel training
    if max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(train_lstm_model, symbol, data_dir, model_dir, plots_dir, **kwargs): symbol
                for symbol in symbols
            }
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results[symbol] = result
                except Exception as e:
                    logger.error(f"Exception training model for {symbol}: {e}")
                    results[symbol] = {'success': False, 'error': str(e)}
    else:
        # Sequential training
        for symbol in symbols:
            try:
                result = train_lstm_model(symbol, data_dir, model_dir, plots_dir, **kwargs)
                results[symbol] = result
            except Exception as e:
                logger.error(f"Exception training model for {symbol}: {e}")
                results[symbol] = {'success': False, 'error': str(e)}
    
    return results

def train_arbitrage_models(data_dir: str, model_dir: str, plots_dir: str) -> Dict[str, Any]:
    """
    Train models for arbitrage opportunity prediction
    
    Args:
        data_dir: Directory with processed data
        model_dir: Directory to save models
        plots_dir: Directory to save plots
        
    Returns:
        Dictionary with training results
    """
    logger.info("Training arbitrage prediction models")
    
    # Check for arbitrage training data
    if not os.path.exists(os.path.join(data_dir, 'arbitrage_direct_train.csv')):
        logger.warning("No arbitrage training data found. Skipping.")
        return {}
    
    # Prepare directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load arbitrage data
    try:
        # Try both direct and triangular arbitrage datasets
        arbitrage_results = {}
        
        for arb_type in ['direct', 'triangular']:
            train_path = os.path.join(data_dir, f'arbitrage_{arb_type}_train.csv')
            test_path = os.path.join(data_dir, f'arbitrage_{arb_type}_test.csv')
            
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                logger.info(f"No training data for {arb_type} arbitrage. Skipping.")
                continue
                
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            if 'target' not in train_df.columns:
                logger.warning(f"Target column not found in {train_path}. Skipping.")
                continue
                
            logger.info(f"Loaded {len(train_df)} training samples and {len(test_df)} testing samples for {arb_type} arbitrage")
            
            # Prepare features and target
            feature_cols = [col for col in train_df.columns if col not in ['timestamp', 'datetime', 'target']]
            
            # Train model - starting with logistic regression
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_df[feature_cols])
            y_train = train_df['target'].values
            
            X_test = scaler.transform(test_df[feature_cols])
            y_test = test_df['target'].values
            
            # Train model
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1': float(f1_score(y_test, y_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_test, y_proba))
            }
            
            logger.info(f"{arb_type.capitalize()} arbitrage metrics: {metrics}")
            
            # Save model and results
            import joblib
            model_path = os.path.join(model_dir, f'logreg_{arb_type}_arbitrage.joblib')
            joblib.dump(model, model_path)
            
            scaler_path = os.path.join(model_dir, f'scaler_{arb_type}_arbitrage.joblib')
            joblib.dump(scaler, scaler_path)
            
            # Save feature importances
            feature_importances = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.coef_[0]
            }).sort_values('importance', ascending=False)
            
            feature_importances.to_csv(os.path.join(model_dir, f'feature_importance_{arb_type}_arbitrage.csv'), index=False)
            
            # Save metrics
            with open(os.path.join(model_dir, f'metrics_{arb_type}_arbitrage.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
                
            # Create ROC curve
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {arb_type.capitalize()} Arbitrage')
            plt.legend()
            plt.grid(True)
            
            roc_path = os.path.join(plots_dir, f'roc_{arb_type}_arbitrage.png')
            plt.savefig(roc_path)
            plt.close()
            
            # Add results to dict
            arbitrage_results[arb_type] = {
                'metrics': metrics,
                'model_path': model_path,
                'scaler_path': scaler_path,
                'roc_path': roc_path,
                'feature_importances': feature_importances.to_dict('records')[:10]  # Top 10 features
            }
        
        return arbitrage_results
            
    except Exception as e:
        logger.error(f"Error training arbitrage models: {e}")
        return {'error': str(e)}

def main():
    """Main function to train all models"""
    start_time = time.time()
    logger.info("Starting training of all models")
    
    # Define directories
    data_dir = 'data/processed'
    model_dir = 'models'
    plots_dir = 'plots'
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Train LSTM models for price prediction
    lstm_results = train_all_lstm_models(
        data_dir=data_dir,
        model_dir=model_dir,
        plots_dir=plots_dir,
        max_workers=1,  # Sequential training to avoid memory issues
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        learning_rate=0.001,
        batch_size=64,
        num_epochs=50,
        patience=10
    )
    
    # Train arbitrage models
    arbitrage_results = train_arbitrage_models(
        data_dir=data_dir,
        model_dir=model_dir,
        plots_dir=plots_dir
    )
    
    # Save overall results
    results = {
        'lstm_models': lstm_results,
        'arbitrage_models': arbitrage_results,
        'training_time': time.time() - start_time,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(model_dir, 'training_results.json'), 'w') as f:
        # Convert non-serializable types
        def json_serial(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, np.int64):
                return int(obj)
            raise TypeError(f"Type {type(obj)} not serializable")
            
        json.dump(results, f, indent=2, default=json_serial)
    
    logger.info(f"All model training completed in {time.time() - start_time:.2f} seconds")
    
    return results

if __name__ == "__main__":
    main()