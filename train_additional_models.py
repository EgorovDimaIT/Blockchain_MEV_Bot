"""
Train LSTM models on additional cryptocurrency pairs
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import json
import time
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ModelTrainer')

class PricePredictionLSTM(nn.Module):
    """LSTM model for price prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layer
            num_layers: Number of LSTM layers
            output_size: Number of output values
            dropout: Dropout probability
        """
        super(PricePredictionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor with shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor with shape (batch_size, output_size)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from last time step
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Linear layer
        out = self.fc(out)
        
        return out

def train_model(symbol, data_dir, output_dir, epochs=10, batch_size=32, learning_rate=0.001, exchange='uniswap'):
    """
    Train LSTM model for price prediction
    
    Args:
        symbol: Symbol to train model for
        data_dir: Directory with processed data
        output_dir: Directory to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        exchange: Exchange name
        
    Returns:
        True if training was successful, False otherwise
    """
    try:
        # Check if LSTM data exists
        X_train_path = os.path.join(data_dir, f"lstm_{exchange}_{symbol}_X_train.npy")
        y_train_path = os.path.join(data_dir, f"lstm_{exchange}_{symbol}_y_train.npy")
        X_test_path = os.path.join(data_dir, f"lstm_{exchange}_{symbol}_X_test.npy")
        y_test_path = os.path.join(data_dir, f"lstm_{exchange}_{symbol}_y_test.npy")
        features_path = os.path.join(data_dir, f"lstm_{exchange}_{symbol}_features.json")
        targets_path = os.path.join(data_dir, f"lstm_{exchange}_{symbol}_targets.json")
        
        # Check if all required files exist
        required_files = [X_train_path, y_train_path, X_test_path, y_test_path]
        if not all(os.path.exists(f) for f in required_files):
            logger.error(f"Missing LSTM data files for {symbol}")
            return False
        
        # Try to load feature and target names
        feature_names = []
        target_names = []
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                feature_names = json.load(f)
        if os.path.exists(targets_path):
            with open(targets_path, 'r') as f:
                target_names = json.load(f)
        
        # Load data
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        
        logger.info(f"Loaded data for {symbol}:")
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"y_test shape: {y_test.shape}")
        
        # Check if data is valid
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            logger.error(f"Empty training data for {symbol}")
            return False
        
        # Set device (CPU or GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create model
        input_size = X_train.shape[2]  # Number of features
        hidden_size = 64
        num_layers = 2
        output_size = y_train.shape[1]  # Number of targets
        dropout = 0.2
        
        model = PricePredictionLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        ).to(device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
        
        # Training loop
        logger.info(f"Starting training for {symbol} with {epochs} epochs")
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            
            # Generate random indices for batches
            indices = np.random.permutation(len(X_train))
            n_batches = len(X_train) // batch_size
            
            for i in range(n_batches):
                # Get batch indices
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                
                # Get batch data
                X_batch = X_train_tensor[batch_indices]
                y_batch = y_train_tensor[batch_indices]
                
                # Forward pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Calculate average training loss
            train_loss /= n_batches
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor).item()
                val_losses.append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
            
            # Print progress every few epochs
            if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            train_preds = model(X_train_tensor).cpu().numpy()
            test_preds = model(X_test_tensor).cpu().numpy()
        
        # Calculate metrics for each target
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        train_mae = mean_absolute_error(y_train, train_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        
        # Calculate metrics for individual targets
        target_metrics = []
        for i in range(output_size):
            target_name = target_names[i] if i < len(target_names) else f"target_{i}"
            target_train_rmse = np.sqrt(mean_squared_error(y_train[:, i], train_preds[:, i]))
            target_train_mae = mean_absolute_error(y_train[:, i], train_preds[:, i])
            target_test_rmse = np.sqrt(mean_squared_error(y_test[:, i], test_preds[:, i]))
            target_test_mae = mean_absolute_error(y_test[:, i], test_preds[:, i])
            
            target_metrics.append({
                'name': target_name,
                'train_rmse': float(target_train_rmse),
                'train_mae': float(target_train_mae),
                'test_rmse': float(target_test_rmse),
                'test_mae': float(target_test_mae)
            })
        
        logger.info(f"Overall metrics for {symbol}:")
        logger.info(f"Train RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}")
        logger.info(f"Test RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}")
        
        # Create output directory
        model_dir = os.path.join(output_dir, symbol)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"{symbol}_model.pt")
        torch.save(model.state_dict(), model_path)
        
        # Save model info
        model_info = {
            'symbol': symbol,
            'exchange': exchange,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size': output_size,
            'dropout': dropout,
            'features': feature_names,
            'targets': target_names,
            'metrics': {
                'train_rmse': float(train_rmse),
                'train_mae': float(train_mae),
                'test_rmse': float(test_rmse),
                'test_mae': float(test_mae),
                'best_val_loss': float(best_val_loss)
            },
            'target_metrics': target_metrics,
            'training_params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            },
            'training_time': training_time,
            'timestamp': datetime.now().isoformat(),
            'data_shapes': {
                'X_train': X_train.shape,
                'y_train': y_train.shape,
                'X_test': X_test.shape,
                'y_test': y_test.shape
            }
        }
        
        with open(os.path.join(model_dir, f"{symbol}_info.json"), 'w') as f:
            json.dump(model_info, f, indent=4)
        
        # Plot training history
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f"{symbol} Training History")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f"{symbol}_training_history.png"))
        plt.close()
        
        # Plot predictions for all targets
        for i in range(output_size):
            target_name = target_names[i] if i < len(target_names) else f"target_{i}"
            
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(y_train[:, i], label='True')
            plt.plot(train_preds[:, i], label='Predicted')
            plt.title(f"{symbol} - {target_name} - Training Set")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(y_test[:, i], label='True')
            plt.plot(test_preds[:, i], label='Predicted')
            plt.title(f"{symbol} - {target_name} - Test Set")
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f"{symbol}_predictions_{target_name}.png"))
            plt.close()
        
        logger.info(f"Model for {symbol} trained and saved successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error training model for {symbol}: {str(e)}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train LSTM models for cryptocurrency price prediction')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Directory with processed data')
    parser.add_argument('--output_dir', type=str, default='models/additional', help='Directory to save trained models')
    parser.add_argument('--exchange', type=str, default='uniswap', help='Exchange name')
    parser.add_argument('--symbols', type=str, nargs='+', help='Symbols to train models for')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of symbols to train models for
    if args.symbols:
        symbols = args.symbols
    else:
        # Find all symbols with lstm data
        symbols = []
        pattern = f"lstm_{args.exchange}_(.+)_X_train.npy"
        import re
        for file in os.listdir(args.data_dir):
            match = re.match(pattern, file)
            if match:
                symbols.append(match.group(1))
        
        # Use only a subset of symbols for testing if too many are found
        if len(symbols) > 10:
            logger.warning(f"Found {len(symbols)} symbols, using only the first 10 for training")
            symbols = symbols[:10]
    
    logger.info(f"Training models for {len(symbols)} symbols: {symbols}")
    
    # Train model for each symbol
    success_count = 0
    for symbol in symbols:
        logger.info(f"Training model for {symbol}")
        if train_model(
            symbol=symbol,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            exchange=args.exchange
        ):
            success_count += 1
    
    logger.info(f"Successfully trained {success_count} out of {len(symbols)} models")

if __name__ == "__main__":
    main()