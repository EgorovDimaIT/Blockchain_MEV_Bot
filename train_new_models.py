"""
Script to train LSTM models for all newly processed cryptocurrency pairs
"""

import os
import logging
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import LSTM model
class PricePredictionLSTM(nn.Module):
    """LSTM model for price prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(PricePredictionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_len, hidden_size)
        
        # Get the output from the last time step
        out = out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Linear layer
        out = self.fc(out)  # (batch_size, output_size)
        
        return out

def select_top_features(X_train_flat, y_train_flat, feature_names, n_features=10):
    """Select top n features based on correlation with target"""
    # For multi-target, we'll use the first target for feature selection
    target = y_train_flat[:, 0]
    
    # Calculate correlation with target for each feature
    correlations = []
    for i in range(X_train_flat.shape[1]):
        corr = np.corrcoef(X_train_flat[:, i], target)[0, 1]
        correlations.append((i, abs(corr)))
    
    # Sort by absolute correlation and get top n
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_indices = [x[0] for x in correlations[:n_features]]
    
    # Print selected features
    selected_features = [feature_names[i] for i in top_indices]
    logger.info(f"Selected top {n_features} features: {selected_features}")
    
    return top_indices

def train_model(symbol: str, exchange: str = 'binance', num_features: int = 10, num_epochs: int = 10):
    """Train LSTM model for a single symbol"""
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Parameters
    symbol_name = f"{exchange}_{symbol}"
    hidden_size = 64
    num_layers = 2
    dropout = 0.3
    learning_rate = 0.001
    batch_size = 32
    patience = 5
    
    try:
        # Load data
        data_dir = 'data/processed'
        X_train = np.load(os.path.join(data_dir, f'lstm_{symbol_name}_X_train.npy'))
        y_train = np.load(os.path.join(data_dir, f'lstm_{symbol_name}_y_train.npy'))
        X_test = np.load(os.path.join(data_dir, f'lstm_{symbol_name}_X_test.npy'))
        y_test = np.load(os.path.join(data_dir, f'lstm_{symbol_name}_y_test.npy'))
        
        # Load feature and target names
        with open(os.path.join(data_dir, f'lstm_{symbol_name}_features.txt'), 'r') as f:
            feature_names = f.read().splitlines()
            
        with open(os.path.join(data_dir, f'lstm_{symbol_name}_targets.txt'), 'r') as f:
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
        
        logger.info(f"Training model for {symbol_name} with shape: {X_train.shape}")
        
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
        logger.info(f"Starting training for {symbol_name} with {num_epochs} epochs")
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
        model_path = os.path.join('models', f'lstm_{symbol_name}_optimized_10f_10e.pth')
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
        plt.title(f'LSTM Training for {symbol_name} (10 features, 10 epochs)')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join('plots', f'lstm_{symbol_name}_optimized_10f_10e_training.png')
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Training plot saved to {plot_path}")
        
        # Create and save prediction plot for the first target
        plt.figure(figsize=(10, 6))
        plt.plot(y_true[:100, 0], label='Actual')
        plt.plot(y_pred[:100, 0], label='Predicted')
        plt.xlabel('Time')
        plt.ylabel(target_names[0])
        plt.title(f'LSTM Prediction for {symbol_name} - {target_names[0]} (10f, 10e)')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join('plots', f'lstm_{symbol_name}_optimized_10f_10e_prediction_{target_names[0]}.png')
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Prediction plot saved to {plot_path}")
        
        # Save results
        results = {
            'symbol': symbol_name,
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
        
        results_path = os.path.join('models', f'lstm_{symbol_name}_optimized_10f_10e_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error training model for {symbol_name}: {str(e)}")
        return {'error': str(e)}

def main():
    """Train all models for new cryptocurrency pairs"""
    start_time = time.time()
    logger.info("Starting model training for all pairs")
    
    # Define pairs to train
    pairs_to_train = [
        {'symbol': 'ETHUSDT', 'exchange': 'binance'},
        {'symbol': 'USDTGBP', 'exchange': 'binance'},
        {'symbol': 'ETHUSDC', 'exchange': 'binance'},
        {'symbol': 'ETHWETH', 'exchange': 'uniswap'},
        {'symbol': 'UNIETH', 'exchange': 'uniswap'},
        {'symbol': 'AAVEETH', 'exchange': 'uniswap'},
        {'symbol': 'LINKETH', 'exchange': 'uniswap'}
    ]
    
    # Results for all models
    all_results = {}
    
    # Train each model
    for pair_info in pairs_to_train:
        symbol = pair_info['symbol']
        exchange = pair_info['exchange']
        
        logger.info(f"Training model for {exchange}_{symbol}")
        try:
            # Check if data files exist
            data_dir = 'data/processed'
            if not os.path.exists(os.path.join(data_dir, f'lstm_{exchange}_{symbol}_X_train.npy')):
                logger.warning(f"Data files for {exchange}_{symbol} not found. Skipping.")
                continue
            
            # Train the model
            results = train_model(symbol, exchange)
            all_results[f"{exchange}_{symbol}"] = results
            
        except Exception as e:
            logger.error(f"Failed to train model for {exchange}_{symbol}: {str(e)}")
            all_results[f"{exchange}_{symbol}"] = {'error': str(e)}
    
    # Save all results
    with open('models/all_models_results.json', 'w') as f:
        json.dump(all_results, f, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x, indent=2)
    
    logger.info(f"All model training completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()