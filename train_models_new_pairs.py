"""
Train LSTM models on new cryptocurrency pairs
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
from datetime import datetime

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

def train_lstm_model(X_train, y_train, X_test, y_test, model_params, training_params, symbol):
    """
    Train LSTM model
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        model_params: Dictionary of model parameters
        training_params: Dictionary of training parameters
        symbol: Symbol name
        
    Returns:
        Trained model, training history and evaluation metrics
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Create model
    model = PricePredictionLSTM(
        input_size=model_params['input_size'],
        hidden_size=model_params['hidden_size'],
        num_layers=model_params['num_layers'],
        output_size=model_params['output_size'],
        dropout=model_params['dropout']
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    
    # Training loop
    num_epochs = training_params['epochs']
    batch_size = training_params['batch_size']
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience = training_params.get('patience', 10)
    patience_counter = 0
    
    logger.info(f"Starting training for {symbol} with {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        # Create batches
        indices = np.random.permutation(len(X_train))
        n_batches = len(X_train) // batch_size
        
        for i in range(n_batches):
            # Get batch indices
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
            
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
        
        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train_tensor).cpu().numpy()
        test_preds = model(X_test_tensor).cpu().numpy()
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    train_mae = mean_absolute_error(y_train, train_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)
    
    logger.info(f"Training RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
    logger.info(f"Testing RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
    
    # Training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    # Metrics
    metrics = {
        'train_rmse': float(train_rmse),
        'train_mae': float(train_mae),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae)
    }
    
    return model, history, metrics

def save_model(model, symbol, output_dir, model_params, training_params, history, metrics):
    """
    Save model, parameters and history
    
    Args:
        model: Trained model
        symbol: Symbol name
        output_dir: Output directory
        model_params: Model parameters
        training_params: Training parameters
        history: Training history
        metrics: Evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f"{symbol}_model.pt")
    torch.save(model.state_dict(), model_path)
    
    # Save parameters and metrics
    info = {
        'model_params': model_params,
        'training_params': training_params,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, f"{symbol}_info.json"), 'w') as f:
        json.dump(info, f, indent=4)
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.title(f"{symbol} Training History")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{symbol}_history.png"))
    plt.close()
    
    logger.info(f"Model and information saved to {output_dir}")

def train_model_for_pair(pair, data_dir='data/processed', output_dir='models/new_pairs', exchange='uniswap'):
    """
    Train model for a specific pair
    
    Args:
        pair: Symbol/pair name
        data_dir: Directory with processed data
        output_dir: Output directory for models
        exchange: Exchange name
    """
    # Check if LSTM data exists
    X_train_path = os.path.join(data_dir, f"lstm_{exchange}_{pair}_X_train.npy")
    y_train_path = os.path.join(data_dir, f"lstm_{exchange}_{pair}_y_train.npy")
    X_test_path = os.path.join(data_dir, f"lstm_{exchange}_{pair}_X_test.npy")
    y_test_path = os.path.join(data_dir, f"lstm_{exchange}_{pair}_y_test.npy")
    
    if not all(os.path.exists(path) for path in [X_train_path, y_train_path, X_test_path, y_test_path]):
        logger.error(f"LSTM data not found for {pair}")
        return False
    
    try:
        # Load data
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        
        # Check data shapes
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"y_test shape: {y_test.shape}")
        
        # Check if we have enough data
        if len(X_train) < 50 or len(X_test) < 10:
            logger.error(f"Not enough data for {pair}")
            return False
        
        # Model parameters
        model_params = {
            'input_size': X_train.shape[2],  # Number of features
            'hidden_size': 64,  # Size of hidden layer
            'num_layers': 2,  # Number of LSTM layers
            'output_size': y_train.shape[1],  # Number of prediction targets
            'dropout': 0.2  # Dropout probability
        }
        
        # Training parameters
        training_params = {
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'patience': 10  # Early stopping patience
        }
        
        # Train model
        model, history, metrics = train_lstm_model(
            X_train, y_train, X_test, y_test,
            model_params, training_params, pair
        )
        
        # Save model
        model_dir = os.path.join(output_dir, pair)
        os.makedirs(model_dir, exist_ok=True)
        
        save_model(
            model, pair, model_dir,
            model_params, training_params, history, metrics
        )
        
        logger.info(f"Model for {pair} trained and saved successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error training model for {pair}: {str(e)}")
        return False

def main():
    """Train models for new cryptocurrency pairs"""
    # Pairs to train
    pairs = ['ATHEUR', 'DBREUR', 'CPOOLEUR', 'EURQEUR', 'EURQUSD', 'GTCEUR']
    
    # Create output directory
    output_dir = 'models/new_pairs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model for each pair
    for pair in pairs:
        logger.info(f"Training model for {pair}...")
        success = train_model_for_pair(pair)
        if success:
            logger.info(f"Model for {pair} trained successfully")
        else:
            logger.error(f"Failed to train model for {pair}")
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()