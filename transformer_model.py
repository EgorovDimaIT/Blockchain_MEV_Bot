"""
Transformer model for price prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import os
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

class TransformerModel(nn.Module):
    """
    Transformer model for financial time series prediction
    """
    
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_encoder_layers: int, 
                 dim_feedforward: int, output_dim: int, dropout: float = 0.1):
        """
        Initialize Transformer model
        
        Args:
            input_dim: Number of input features
            d_model: Size of the Transformer model
            nhead: Number of attention heads
            num_encoder_layers: Number of Transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            output_dim: Number of output features
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.output_dim = output_dim
        
        # Feature projection to d_model dimensions
        self.feature_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_encoder_layers
        )
        
        # Final prediction head
        self.output_layer = nn.Linear(d_model, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Project features to d_model dimensions
        x = self.feature_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply Transformer encoder
        x = self.transformer_encoder(x)
        
        # Get the last time step and predict
        x = x[:, -1, :]  # Last time step
        
        # Output layer
        output = self.output_layer(x)
        
        return output

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize positional encoding
        
        Args:
            d_model: Model dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        # Add positional encoding
        x = x + self.pe[:x.size(1), :]
        
        return self.dropout(x)

class TransformerModelTrainer:
    """
    Trainer for Transformer model
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize trainer
        
        Args:
            model_path: Optional path to load model from
        """
        self.model = None
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create saved models directory
        os.makedirs('ml_model/saved_models', exist_ok=True)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def create_model(self, input_dim: int, d_model: int = 64, nhead: int = 4, 
                     num_encoder_layers: int = 2, dim_feedforward: int = 256, 
                     output_dim: int = 1, dropout: float = 0.1) -> TransformerModel:
        """
        Create Transformer model
        
        Args:
            input_dim: Number of input features
            d_model: Size of the Transformer model
            nhead: Number of attention heads
            num_encoder_layers: Number of Transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            output_dim: Number of output features
            dropout: Dropout probability
            
        Returns:
            Transformer model
        """
        logger.info(f"Creating Transformer model with {input_dim} inputs, {d_model} model dimension, {nhead} heads")
        
        # Create model
        self.model = TransformerModel(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        return self.model
    
    def train(self, train_data: Dict[str, np.ndarray], epochs: int = 100, batch_size: int = 64, 
              learning_rate: float = 0.001, early_stopping: int = 10) -> Dict[str, Any]:
        """
        Train Transformer model
        
        Args:
            train_data: Dictionary with training data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping: Early stopping patience
            
        Returns:
            Dictionary with training results
        """
        if self.model is None:
            logger.error("Model not created or loaded. Call create_model() first.")
            return {}
            
        logger.info(f"Training Transformer model on {len(train_data['X_train'])} samples for {epochs} epochs")
        
        # Get training data
        X_train = torch.from_numpy(train_data['X_train']).float().to(self.device)
        y_train = torch.from_numpy(train_data['y_train']).float().to(self.device)
        X_val = torch.from_numpy(train_data['X_test']).float().to(self.device)
        y_val = torch.from_numpy(train_data['y_test']).float().to(self.device)
        
        # Optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training history
        train_losses = []
        val_losses = []
        
        # Train loop
        for epoch in range(epochs):
            # Training mode
            self.model.train()
            
            # Shuffle data
            indices = torch.randperm(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Batch training
            total_train_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                # Get batch
                batch_X = X_train_shuffled[i:i+batch_size]
                batch_y = y_train_shuffled[i:i+batch_size]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                
                # Calculate loss
                loss = criterion(outputs.squeeze(), batch_y)
                total_train_loss += loss.item()
                num_batches += 1
                
                # Backward pass and update
                loss.backward()
                optimizer.step()
            
            # Calculate average training loss
            avg_train_loss = total_train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs.squeeze(), y_val).item()
                val_losses.append(val_loss)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                # Calculate metrics on validation set
                val_pred = val_outputs.squeeze().cpu().numpy()
                val_true = y_val.cpu().numpy()
                
                rmse = np.sqrt(mean_squared_error(val_true, val_pred))
                mae = mean_absolute_error(val_true, val_pred)
                
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        # Save model
        self.save_model()
        
        # Final evaluation
        metrics = self.evaluate(train_data)
        
        # Create results dict
        results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses),
            'metrics': metrics
        }
        
        return results
    
    def evaluate(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            data: Dictionary with test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            logger.error("Model not created or loaded. Call create_model() or load_model() first.")
            return {}
            
        # Get test data
        X_test = torch.from_numpy(data['X_test']).float().to(self.device)
        y_test = data['y_test']
        
        # Evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_test).squeeze().cpu().numpy()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_test - predictions) / np.clip(np.abs(y_test), 1e-8, None))) * 100
        r2 = 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        
        # Log results
        logger.info(f"Transformer Evaluation - RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        if self.model is None:
            logger.error("Model not created or loaded. Call create_model() or load_model() first.")
            return np.array([])
            
        # Convert to tensor
        X_tensor = torch.from_numpy(X).float().to(self.device)
        
        # Evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_tensor).squeeze().cpu().numpy()
            
        return predictions
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save model to file
        
        Args:
            path: Path to save model to, defaults to a generated path
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            logger.error("No model to save")
            return ""
            
        if path is None:
            timestamp = int(time.time())
            path = f"ml_model/saved_models/transformer_model_{timestamp}.pt"
            
        # Create parent directory
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'd_model': self.model.d_model,
            'nhead': self.model.nhead,
            'num_encoder_layers': self.model.num_encoder_layers,
            'dim_feedforward': self.model.dim_feedforward,
            'output_dim': self.model.output_dim
        }
        
        torch.save(model_state, path)
        logger.info(f"Model saved to {path}")
        
        self.model_path = path
        return path
    
    def load_model(self, path: str) -> bool:
        """
        Load model from file
        
        Args:
            path: Path to load model from
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not os.path.exists(path):
            logger.error(f"Model file {path} not found")
            return False
            
        try:
            # Load model state
            model_state = torch.load(path, map_location=self.device)
            
            # Create model with same architecture
            self.create_model(
                input_dim=model_state['input_dim'],
                d_model=model_state['d_model'],
                nhead=model_state['nhead'],
                num_encoder_layers=model_state['num_encoder_layers'],
                dim_feedforward=model_state['dim_feedforward'],
                output_dim=model_state['output_dim']
            )
            
            # Load weights
            self.model.load_state_dict(model_state['model_state_dict'])
            
            logger.info(f"Model loaded from {path}")
            self.model_path = path
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def plot_training_curves(self, train_losses: List[float], val_losses: List[float], save_path: Optional[str] = None):
        """
        Plot training curves
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            save_path: Optional path to save plot to
        """
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Transformer Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training curves saved to {save_path}")
            
        plt.close()

# Singleton instance
_trainer = None

def get_transformer_trainer(model_path: Optional[str] = None) -> TransformerModelTrainer:
    """
    Get singleton instance of Transformer model trainer
    
    Args:
        model_path: Optional path to load model from
        
    Returns:
        TransformerModelTrainer instance
    """
    global _trainer
    if _trainer is None:
        _trainer = TransformerModelTrainer(model_path)
    return _trainer