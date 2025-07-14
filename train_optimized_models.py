"""
Script to train optimized ML models with 10 features and 10 epochs
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

def select_top_features(X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str], n_features: int = 10) -> List[int]:
    """
    Select top n features based on correlation with target
    
    Args:
        X_train: Training data features
        y_train: Training data targets
        feature_names: Names of features
        n_features: Number of features to select
        
    Returns:
        List of indices of top features
    """
    # For multi-target, we'll use the first target for feature selection
    target = y_train[:, 0]
    
    # Calculate correlation with target for each feature
    correlations = []
    for i in range(X_train.shape[1]):
        corr = np.corrcoef(X_train[:, i], target)[0, 1]
        correlations.append((i, abs(corr)))
    
    # Sort by absolute correlation and get top n
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_indices = [x[0] for x in correlations[:n_features]]
    
    # Print selected features
    selected_features = [feature_names[i] for i in top_indices]
    logger.info(f"Selected top {n_features} features: {selected_features}")
    
    return top_indices

def train_lstm_models_optimized():
    """Train LSTM models with optimized parameters"""
    logger.info("Training optimized LSTM models")
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Parameters
    num_features = 10
    num_epochs = 10
    hidden_size = 64
    num_layers = 2
    dropout = 0.3
    learning_rate = 0.001
    batch_size = 32
    patience = 5
    
    # Get list of processed datasets
    data_dir = 'data/processed'
    results = {}
    
    for file in os.listdir(data_dir):
        if file.startswith('lstm_') and file.endswith('_X_train.npy'):
            symbol = file.replace('lstm_', '').replace('_X_train.npy', '')
            logger.info(f"Found dataset for {symbol}")
            
            try:
                # Load data
                X_train = np.load(os.path.join(data_dir, f'lstm_{symbol}_X_train.npy'))
                y_train = np.load(os.path.join(data_dir, f'lstm_{symbol}_y_train.npy'))
                X_test = np.load(os.path.join(data_dir, f'lstm_{symbol}_X_test.npy'))
                y_test = np.load(os.path.join(data_dir, f'lstm_{symbol}_y_test.npy'))
                
                # Load feature and target names
                with open(os.path.join(data_dir, f'lstm_{symbol}_features.txt'), 'r') as f:
                    feature_names = f.read().splitlines()
                    
                with open(os.path.join(data_dir, f'lstm_{symbol}_targets.txt'), 'r') as f:
                    target_names = f.read().splitlines()
                
                # Select top features
                if X_train.shape[2] > num_features:
                    # We need to handle the temporal dimension of LSTM data
                    # Reshape to 2D for feature selection
                    X_train_flat = X_train.reshape(-1, X_train.shape[2])
                    y_train_flat = np.repeat(y_train, X_train.shape[1], axis=0)
                    
                    top_indices = select_top_features(X_train_flat, y_train_flat, feature_names, num_features)
                    
                    # Select only top features for training and testing
                    X_train = X_train[:, :, top_indices]
                    X_test = X_test[:, :, top_indices]
                    selected_feature_names = [feature_names[i] for i in top_indices]
                else:
                    selected_feature_names = feature_names
                    
                logger.info(f"Training model for {symbol} with shape: {X_train.shape}")
                
                # Import necessary components for training
                from ml_model.lstm_model import PricePredictionLSTM
                import torch
                import torch.nn as nn
                import torch.optim as optim
                
                # Convert to PyTorch tensors
                X_train_tensor = torch.FloatTensor(X_train)
                y_train_tensor = torch.FloatTensor(y_train)
                X_test_tensor = torch.FloatTensor(X_test)
                y_test_tensor = torch.FloatTensor(y_test)
                
                # Create datasets and data loaders
                train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
                
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                # Initialize model
                input_size = X_train.shape[2]  # Number of features
                output_size = y_train.shape[1]  # Number of targets
                
                model = PricePredictionLSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_size=output_size,
                    dropout=dropout
                )
                
                # Loss function and optimizer
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                
                # Training loop
                best_val_loss = float('inf')
                best_model_state = None
                patience_counter = 0
                training_losses = []
                validation_losses = []
                
                # Start training
                logger.info(f"Starting training for {symbol} with {num_epochs} epochs")
                start_time = time.time()
                
                for epoch in range(num_epochs):
                    # Training
                    model.train()
                    train_loss = 0.0
                    
                    for inputs, targets in train_loader:
                        # Forward pass
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        # Backward pass and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                    
                    train_loss /= len(train_loader)
                    training_losses.append(train_loss)
                    
                    # Validation
                    model.eval()
                    val_loss = 0.0
                    
                    with torch.no_grad():
                        for inputs, targets in test_loader:
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            val_loss += loss.item()
                    
                    val_loss /= len(test_loader)
                    validation_losses.append(val_loss)
                    
                    # Log progress
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                    
                    # Check for improvement
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = model.state_dict()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    # Early stopping
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                
                training_time = time.time() - start_time
                logger.info(f"Training completed in {training_time:.2f} seconds")
                
                # Load best model
                if best_model_state:
                    model.load_state_dict(best_model_state)
                
                # Evaluate on test set
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_test_tensor).numpy()
                    y_true = y_test
                
                # Calculate metrics for each target
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                import math
                
                metrics = {}
                for i, target_name in enumerate(target_names):
                    rmse = math.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
                    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
                    r2 = r2_score(y_true[:, i], y_pred[:, i])
                    
                    metrics[target_name] = {
                        'rmse': float(rmse),
                        'mae': float(mae),
                        'r2': float(r2)
                    }
                    
                    logger.info(f"Metrics for {target_name}: RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")
                
                # Save model
                model_path = os.path.join('models', f'lstm_{symbol}_optimized.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'output_size': output_size,
                    'dropout': dropout,
                    'feature_names': selected_feature_names,
                    'target_names': target_names,
                    'metrics': metrics,
                    'training_time': training_time,
                    'num_epochs': epoch + 1,
                    'best_val_loss': best_val_loss
                }, model_path)
                
                logger.info(f"Model saved to {model_path}")
                
                # Create and save training plot
                plt.figure(figsize=(10, 6))
                plt.plot(training_losses, label='Training Loss')
                plt.plot(validation_losses, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'LSTM Training for {symbol} (Optimized)')
                plt.legend()
                plt.grid(True)
                
                plot_path = os.path.join('plots', f'lstm_{symbol}_optimized_training.png')
                plt.savefig(plot_path)
                plt.close()
                
                logger.info(f"Training plot saved to {plot_path}")
                
                # Create and save prediction plot for the first target
                plt.figure(figsize=(10, 6))
                plt.plot(y_true[:100, 0], label='Actual')
                plt.plot(y_pred[:100, 0], label='Predicted')
                plt.xlabel('Time')
                plt.ylabel(target_names[0])
                plt.title(f'LSTM Prediction for {symbol} - {target_names[0]} (Optimized)')
                plt.legend()
                plt.grid(True)
                
                plot_path = os.path.join('plots', f'lstm_{symbol}_optimized_prediction_{target_names[0]}.png')
                plt.savefig(plot_path)
                plt.close()
                
                logger.info(f"Prediction plot saved to {plot_path}")
                
                # Save results
                results[symbol] = {
                    'success': True,
                    'model_path': model_path,
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'output_size': output_size,
                    'dropout': dropout,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'epochs': epoch + 1,
                    'training_time': training_time,
                    'best_val_loss': best_val_loss,
                    'metrics': metrics,
                    'feature_names': selected_feature_names,
                    'target_names': target_names
                }
                
            except Exception as e:
                logger.error(f"Error training model for {symbol}: {e}")
                results[symbol] = {
                    'success': False,
                    'error': str(e)
                }
    
    # Save overall results
    results_path = os.path.join('models', 'lstm_optimized_results.json')
    with open(results_path, 'w') as f:
        json_results = json.dumps(results, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        f.write(json_results)
    
    logger.info(f"Overall results saved to {results_path}")
    return results

def train_opportunity_detection_models():
    """Train models for MEV opportunity detection"""
    logger.info("Training MEV opportunity detection models")
    
    # Create directories if they don't exist
    os.makedirs('models/mev', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Train models for direct arbitrage
    results = {}
    
    try:
        # Create synthetic opportunity data if real data is not available
        train_data_path = 'data/processed/mev_opportunities.csv'
        if not os.path.exists(train_data_path):
            logger.warning("No real opportunity data found, creating example dataset from price data")
            create_opportunity_dataset()
            
        if os.path.exists(train_data_path):
            df = pd.read_csv(train_data_path)
            
            # Select top 10 features
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'datetime', 'opportunity_id', 'type', 'profit', 'executed']]
            
            if len(feature_cols) > 10:
                # Calculate correlation with profit
                correlations = []
                for col in feature_cols:
                    corr = df[col].corr(df['profit'])
                    correlations.append((col, abs(corr)))
                
                # Sort by absolute correlation and get top 10
                correlations.sort(key=lambda x: x[1], reverse=True)
                selected_features = [x[0] for x in correlations[:10]]
                logger.info(f"Selected top 10 features: {selected_features}")
            else:
                selected_features = feature_cols
            
            # Add target to selected features
            features_target = selected_features + ['profit']
            
            # Split train/test
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(df[features_target], test_size=0.2, random_state=42)
            
            X_train = train_df[selected_features].values
            y_train = train_df['profit'].values
            X_test = test_df[selected_features].values
            y_test = test_df['profit'].values
            
            # Scale features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Save scaler
            import joblib
            scaler_path = os.path.join('models/mev', 'opportunity_scaler.joblib')
            joblib.dump(scaler, scaler_path)
            
            # Train different models
            # 1. Linear Regression
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train)
            
            lr_preds = lr_model.predict(X_test_scaled)
            lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
            lr_mae = mean_absolute_error(y_test, lr_preds)
            lr_r2 = r2_score(y_test, lr_preds)
            
            logger.info(f"Linear Regression: RMSE={lr_rmse:.6f}, MAE={lr_mae:.6f}, R²={lr_r2:.6f}")
            
            # Save model
            lr_model_path = os.path.join('models/mev', 'linear_regression.joblib')
            joblib.dump(lr_model, lr_model_path)
            
            # 2. Random Forest
            from sklearn.ensemble import RandomForestRegressor
            
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
            rf_model.fit(X_train_scaled, y_train)
            
            rf_preds = rf_model.predict(X_test_scaled)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
            rf_mae = mean_absolute_error(y_test, rf_preds)
            rf_r2 = r2_score(y_test, rf_preds)
            
            logger.info(f"Random Forest: RMSE={rf_rmse:.6f}, MAE={rf_mae:.6f}, R²={rf_r2:.6f}")
            
            # Save model
            rf_model_path = os.path.join('models/mev', 'random_forest.joblib')
            joblib.dump(rf_model, rf_model_path)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': rf_model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('importance', reverse=True)
            
            # Plot feature importance
            plt.figure(figsize=(12, 6))
            plt.barh(feature_importance['feature'], feature_importance['importance'])
            plt.xlabel('Importance')
            plt.title('Feature Importance (Random Forest)')
            plt.tight_layout()
            plt.savefig('plots/feature_importance_mev.png')
            plt.close()
            
            # Save feature importance
            feature_importance.to_csv('models/mev/feature_importance.csv', index=False)
            
            # Try XGBoost if available
            try:
                import xgboost as xgb
                
                xgb_model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
                xgb_model.fit(X_train_scaled, y_train)
                
                xgb_preds = xgb_model.predict(X_test_scaled)
                xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
                xgb_mae = mean_absolute_error(y_test, xgb_preds)
                xgb_r2 = r2_score(y_test, xgb_preds)
                
                logger.info(f"XGBoost: RMSE={xgb_rmse:.6f}, MAE={xgb_mae:.6f}, R²={xgb_r2:.6f}")
                
                # Save model
                xgb_model_path = os.path.join('models/mev', 'xgboost.json')
                xgb_model.save_model(xgb_model_path)
                
                # Save results
                results = {
                    'linear_regression': {
                        'rmse': float(lr_rmse),
                        'mae': float(lr_mae),
                        'r2': float(lr_r2)
                    },
                    'random_forest': {
                        'rmse': float(rf_rmse),
                        'mae': float(rf_mae),
                        'r2': float(rf_r2)
                    },
                    'xgboost': {
                        'rmse': float(xgb_rmse),
                        'mae': float(xgb_mae),
                        'r2': float(xgb_r2)
                    },
                    'selected_features': selected_features
                }
            except:
                logger.warning("XGBoost not available, skipping")
                # Save results without XGBoost
                results = {
                    'linear_regression': {
                        'rmse': float(lr_rmse),
                        'mae': float(lr_mae),
                        'r2': float(lr_r2)
                    },
                    'random_forest': {
                        'rmse': float(rf_rmse),
                        'mae': float(rf_mae),
                        'r2': float(rf_r2)
                    },
                    'selected_features': selected_features
                }
            
            # Save results
            with open('models/mev/results.json', 'w') as f:
                json.dump(results, f, indent=2)
                
            # Create prediction plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, rf_preds, alpha=0.5)
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
            plt.xlabel('Actual Profit')
            plt.ylabel('Predicted Profit')
            plt.title('Random Forest: Actual vs Predicted Profit')
            plt.savefig('plots/mev_prediction.png')
            plt.close()
            
            logger.info("MEV opportunity models trained successfully")
        else:
            logger.error("No MEV opportunity data available for training")
            results = {'error': 'No MEV opportunity data available for training'}
    except Exception as e:
        logger.error(f"Error training MEV opportunity models: {e}")
        results = {'error': str(e)}
    
    return results

def create_opportunity_dataset():
    """Create synthetic opportunity dataset from price data"""
    try:
        # Find all processed price data
        data_dir = 'data/processed'
        dfs = []
        
        for file in os.listdir(data_dir):
            if file.endswith('_processed.csv'):
                df = pd.read_csv(os.path.join(data_dir, file))
                
                if df.shape[0] > 0:
                    # Add source file info
                    df['source'] = file.replace('_processed.csv', '')
                    dfs.append(df)
        
        if not dfs:
            logger.error("No processed price data found")
            return
            
        # Combine all data
        all_data = pd.concat(dfs, ignore_index=True)
        
        # Create opportunities dataset
        opportunities = []
        
        # Use some price features to create opportunity features
        for i in range(len(all_data) - 1):
            if i % 100 == 0:  # Sample every 100 rows to reduce size
                row = all_data.iloc[i]
                
                # Generate synthetic opportunity
                opportunity = {
                    'timestamp': row.get('timestamp', i),
                    'datetime': row.get('datetime', pd.Timestamp.now() - pd.Timedelta(days=i)),
                    'opportunity_id': f"opp_{i}",
                    'type': 'arbitrage' if i % 2 == 0 else 'sandwich',
                    'price_diff': row.get('close', 0) - row.get('open', 0),
                    'volatility': row.get('high', 0) - row.get('low', 0),
                    'volume': row.get('volume', 0),
                    'rsi_14': row.get('rsi_14', 50),
                    'macd': row.get('macd', 0),
                    'bollinger_band_width': row.get('bb_upper', 0) - row.get('bb_lower', 0) if 'bb_upper' in row and 'bb_lower' in row else 0,
                    'exchange_spread': np.random.uniform(0.01, 0.5),
                    'liquidity_score': np.random.uniform(0, 100),
                    'gas_estimate': np.random.uniform(50000, 200000),
                    'price_momentum': row.get('return_1h', 0) if 'return_1h' in row else 0,
                    'market_depth': np.random.uniform(1000, 10000),
                    'profit': abs(row.get('return_1h', 0.01)) * np.random.uniform(0.01, 0.1) if 'return_1h' in row else np.random.uniform(0.001, 0.01),
                    'executed': np.random.choice([True, False], p=[0.3, 0.7])
                }
                
                opportunities.append(opportunity)
        
        # Create DataFrame and save
        if opportunities:
            opps_df = pd.DataFrame(opportunities)
            os.makedirs('data/processed', exist_ok=True)
            opps_df.to_csv('data/processed/mev_opportunities.csv', index=False)
            logger.info(f"Created opportunity dataset with {len(opps_df)} examples")
        else:
            logger.error("Failed to create opportunity dataset")
            
    except Exception as e:
        logger.error(f"Error creating opportunity dataset: {e}")

def main():
    """Main function to train optimized models"""
    start_time = time.time()
    logger.info("Starting optimized model training")
    
    # Train LSTM models
    lstm_results = train_lstm_models_optimized()
    
    # Train MEV detection models
    mev_results = train_opportunity_detection_models()
    
    # Save overall results
    results = {
        'lstm': lstm_results,
        'mev': mev_results,
        'training_time': time.time() - start_time,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('models/training_results_optimized.json', 'w') as f:
        # Convert non-serializable types (e.g., numpy.float32)
        json_results = json.dumps(results, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        f.write(json_results)
    
    logger.info(f"Optimized model training completed in {time.time() - start_time:.2f} seconds")
    
    return results

if __name__ == "__main__":
    main()