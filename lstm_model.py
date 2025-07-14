"""
LSTM model for price and MEV opportunity prediction
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import time
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PricePredictionLSTM(nn.Module):
    """LSTM model for price prediction"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layer
            num_layers: Number of LSTM layers
            output_size: Number of output values
            dropout: Dropout rate
        """
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
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

class LSTMModelTrainer:
    """Class for training LSTM models"""
    
    def __init__(self, data_dir: str = 'data/processed', model_dir: str = 'models', plots_dir: str = 'plots'):
        """
        Initialize LSTM model trainer
        
        Args:
            data_dir: Directory with processed data
            model_dir: Directory to save models
            plots_dir: Directory to save plots
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.plots_dir = plots_dir
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def train_model(self, symbol: str, hidden_size: int = 64, num_layers: int = 2, 
                    dropout: float = 0.2, learning_rate: float = 0.001, batch_size: int = 32, 
                    num_epochs: int = 50, patience: int = 10) -> Dict[str, Any]:
        """
        Train LSTM model for price prediction
        
        Args:
            symbol: Symbol to train model for
            hidden_size: Size of hidden layer
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            num_epochs: Number of epochs
            patience: Patience for early stopping
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training LSTM model for {symbol}")
        
        # Load data
        try:
            X_train = np.load(os.path.join(self.data_dir, f'lstm_{symbol}_X_train.npy'))
            y_train = np.load(os.path.join(self.data_dir, f'lstm_{symbol}_y_train.npy'))
            X_test = np.load(os.path.join(self.data_dir, f'lstm_{symbol}_X_test.npy'))
            y_test = np.load(os.path.join(self.data_dir, f'lstm_{symbol}_y_test.npy'))
            
            # Load feature and target names
            with open(os.path.join(self.data_dir, f'lstm_{symbol}_features.txt'), 'r') as f:
                feature_names = f.read().splitlines()
                
            with open(os.path.join(self.data_dir, f'lstm_{symbol}_targets.txt'), 'r') as f:
                target_names = f.read().splitlines()
                
            logger.info(f"Loaded data for {symbol}: {X_train.shape[0]} training samples, " +
                        f"{X_test.shape[0]} testing samples, {X_train.shape[2]} features, {y_train.shape[1]} targets")
                
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return {
                'symbol': symbol,
                'success': False,
                'error': str(e)
            }
        
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
        model_path = os.path.join(self.model_dir, f'lstm_{symbol}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size': output_size,
            'dropout': dropout,
            'feature_names': feature_names,
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
        plt.title(f'LSTM Training for {symbol}')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(self.plots_dir, f'lstm_{symbol}_training.png')
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Training plot saved to {plot_path}")
        
        # Create and save prediction plot for the first target
        plt.figure(figsize=(10, 6))
        plt.plot(y_true[:100, 0], label='Actual')
        plt.plot(y_pred[:100, 0], label='Predicted')
        plt.xlabel('Time')
        plt.ylabel(target_names[0])
        plt.title(f'LSTM Prediction for {symbol} - {target_names[0]}')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(self.plots_dir, f'lstm_{symbol}_prediction_{target_names[0]}.png')
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Prediction plot saved to {plot_path}")
        
        # Save results to JSON
        results = {
            'symbol': symbol,
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
            'feature_names': feature_names,
            'target_names': target_names
        }
        
        results_path = os.path.join(self.model_dir, f'lstm_{symbol}_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        return results
    
    def import_model(self, model_path: str) -> Tuple[PricePredictionLSTM, Dict[str, Any]]:
        """
        Import saved LSTM model
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Tuple of (model, model_info)
        """
        logger.info(f"Importing model from {model_path}")
        
        try:
            # Load model
            checkpoint = torch.load(model_path)
            
            # Create model
            model = PricePredictionLSTM(
                input_size=checkpoint['input_size'],
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers'],
                output_size=checkpoint['output_size'],
                dropout=checkpoint['dropout']
            )
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Set to eval mode
            model.eval()
            
            logger.info(f"Model imported successfully from {model_path}")
            
            return model, checkpoint
            
        except Exception as e:
            logger.error(f"Error importing model from {model_path}: {e}")
            return None, None
    
    def predict(self, model: PricePredictionLSTM, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with LSTM model
        
        Args:
            model: Trained LSTM model
            X: Input data of shape (num_samples, seq_len, num_features)
            
        Returns:
            Predictions of shape (num_samples, num_targets)
        """
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X)
        
        # Set model to evaluation mode
        model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = model(X_tensor).numpy()
        
        return predictions

def load_lstm_data(data_dir: str, symbol: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load LSTM data for a symbol
    
    Args:
        data_dir: Directory with processed data
        symbol: Symbol to load data for
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, feature_names, target_names)
    """
    logger.info(f"Loading LSTM data for {symbol}")
    
    try:
        X_train = np.load(os.path.join(data_dir, f'lstm_{symbol}_X_train.npy'))
        y_train = np.load(os.path.join(data_dir, f'lstm_{symbol}_y_train.npy'))
        X_test = np.load(os.path.join(data_dir, f'lstm_{symbol}_X_test.npy'))
        y_test = np.load(os.path.join(data_dir, f'lstm_{symbol}_y_test.npy'))
        
        # Load feature and target names
        with open(os.path.join(data_dir, f'lstm_{symbol}_features.txt'), 'r') as f:
            feature_names = f.read().splitlines()
            
        with open(os.path.join(data_dir, f'lstm_{symbol}_targets.txt'), 'r') as f:
            target_names = f.read().splitlines()
            
        logger.info(f"Loaded data for {symbol}: {X_train.shape[0]} training samples, " +
                    f"{X_test.shape[0]} testing samples, {X_train.shape[2]} features, {y_train.shape[1]} targets")
            
        return X_train, y_train, X_test, y_test, feature_names, target_names
        
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return None, None, None, None, None, None

def get_model_trainer(data_dir: str = 'data/processed', model_dir: str = 'models', plots_dir: str = 'plots') -> LSTMModelTrainer:
    """
    Get LSTM model trainer
    
    Args:
        data_dir: Directory with processed data
        model_dir: Directory to save models
        plots_dir: Directory to save plots
        
    Returns:
        LSTM model trainer
    """
    return LSTMModelTrainer(data_dir, model_dir, plots_dir)

def train_model_for_symbol(symbol: str, trainer: Optional[LSTMModelTrainer] = None, **kwargs) -> Dict[str, Any]:
    """
    Train LSTM model for a symbol
    
    Args:
        symbol: Symbol to train model for
        trainer: LSTM model trainer, if None, one will be created
        **kwargs: Additional arguments for trainer.train_model()
        
    Returns:
        Dictionary with training results
    """
    if trainer is None:
        trainer = get_model_trainer()
        
    return trainer.train_model(symbol, **kwargs)

def train_all_models(data_dir: str = 'data/processed', model_dir: str = 'models', plots_dir: str = 'plots', **kwargs) -> Dict[str, Any]:
    """
    Train LSTM models for all available symbols
    
    Args:
        data_dir: Directory with processed data
        model_dir: Directory to save models
        plots_dir: Directory to save plots
        **kwargs: Additional arguments for trainer.train_model()
        
    Returns:
        Dictionary with training results for all symbols
    """
    logger.info("Training LSTM models for all symbols")
    
    # Create trainer
    trainer = get_model_trainer(data_dir, model_dir, plots_dir)
    
    # Find all available symbols
    symbols = []
    for file in os.listdir(data_dir):
        if file.startswith('lstm_') and file.endswith('_X_train.npy'):
            symbol = file.replace('lstm_', '').replace('_X_train.npy', '')
            symbols.append(symbol)
    
    logger.info(f"Found {len(symbols)} symbols: {symbols}")
    
    # Train models for all symbols
    results = {}
    
    for symbol in symbols:
        symbol_results = trainer.train_model(symbol, **kwargs)
        results[symbol] = symbol_results
    
    # Save overall results
    overall_results = {
        'total_symbols': len(symbols),
        'successful_models': sum(1 for res in results.values() if res.get('success', False)),
        'failed_models': sum(1 for res in results.values() if not res.get('success', False)),
        'symbol_results': {symbol: {'success': res.get('success', False)} for symbol, res in results.items()}
    }
    
    results_path = os.path.join(model_dir, 'lstm_all_models_results.json')
    with open(results_path, 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    logger.info(f"Overall results saved to {results_path}")
    
    return results

if __name__ == "__main__":
    # Test the LSTM model
    trainer = get_model_trainer()
    results = train_all_models()