"""
Module for making predictions with trained ML models
"""

import os
import logging
import time
import json
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PricePredictionLSTM(torch.nn.Module):
    """LSTM model for price prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(PricePredictionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_size, output_size)
    
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

class ModelPredictor:
    """Class for making predictions with trained ML models"""
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize ModelPredictor
        
        Args:
            models_dir: Directory with trained models
        """
        self.models_dir = models_dir
        self.models = {}
        self.model_info = {}
        
        # Load available models
        self.load_available_models()
    
    def load_available_models(self):
        """Load all available models from models_dir"""
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory {self.models_dir} does not exist")
            return
            
        # Look for optimized models first (the ones with 10 features and 10 epochs)
        for file in os.listdir(self.models_dir):
            if file.endswith('_optimized_10f_10e.pth'):
                model_name = file.replace('.pth', '')
                symbol = file.replace('lstm_', '').replace('_optimized_10f_10e.pth', '')
                
                try:
                    model, info = self.load_model(os.path.join(self.models_dir, file))
                    if model is not None:
                        self.models[symbol] = model
                        self.model_info[symbol] = info
                        logger.info(f"Loaded optimized model for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading model for {symbol}: {e}")
        
        # Look for regular .pth models
        for file in os.listdir(self.models_dir):
            if file.endswith('.pth') and file.startswith('lstm_') and not file.endswith('_optimized_10f_10e.pth'):
                model_name = file.replace('.pth', '')
                symbol = file.replace('lstm_', '').replace('.pth', '')
                
                # Skip if model already loaded
                if symbol in self.models:
                    continue
                
                try:
                    model, info = self.load_model(os.path.join(self.models_dir, file))
                    if model is not None:
                        self.models[symbol] = model
                        self.model_info[symbol] = info
                        logger.info(f"Loaded model for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading model for {symbol}: {e}")
        
        # Look for .pt models (from newly trained models)
        for file in os.listdir(self.models_dir):
            if file.endswith('.pt') and file.startswith('lstm_'):
                model_name = file.replace('.pt', '')
                symbol = file.replace('lstm_', '').replace('.pt', '')
                
                # Skip if model already loaded
                if symbol in self.models:
                    continue
                
                try:
                    # Check if corresponding info file exists
                    info_file = os.path.join(self.models_dir, f"{model_name}_info.json")
                    if os.path.exists(info_file):
                        # Load model with info file
                        model, info = self.load_new_model(os.path.join(self.models_dir, file), info_file)
                    else:
                        # Try to load model directly
                        model, info = self.load_model(os.path.join(self.models_dir, file))
                        
                    if model is not None:
                        self.models[symbol] = model
                        self.model_info[symbol] = info
                        logger.info(f"Loaded new model for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading new model for {symbol}: {e}")
        
        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def load_model(self, model_path: str) -> Tuple[Optional[PricePredictionLSTM], Optional[Dict[str, Any]]]:
        """
        Load LSTM model from file
        
        Args:
            model_path: Path to model file
            
        Returns:
            Tuple of (model, model_info) or (None, None) if loading fails
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, weights_only=True)
            
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
            
            return model, checkpoint
        
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return None, None
            
    def load_new_model(self, model_path: str, info_path: str) -> Tuple[Optional[PricePredictionLSTM], Optional[Dict[str, Any]]]:
        """
        Load new LSTM model (.pt file) from file with separate info JSON
        
        Args:
            model_path: Path to model file (.pt)
            info_path: Path to model info file (.json)
            
        Returns:
            Tuple of (model, model_info) or (None, None) if loading fails
        """
        try:
            # Load model info
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            
            # Extract model parameters
            input_size = model_info.get('input_size', len(model_info.get('feature_columns', [])))
            hidden_size = model_info.get('hidden_size', 64)
            num_layers = model_info.get('num_layers', 2)
            output_size = model_info.get('output_size', len(model_info.get('target_columns', [])))
            dropout = model_info.get('dropout', 0.2)
            
            # Create model
            model = PricePredictionLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                dropout=dropout
            )
            
            # Load state dict
            model.load_state_dict(torch.load(model_path, weights_only=True))
            
            # Set to eval mode
            model.eval()
            
            # Add target names if not present
            if 'target_names' not in model_info:
                model_info['target_names'] = model_info.get('target_columns', 
                    [f'future_return_{h}' for h in [1, 3, 6, 12, 24]])
            
            # Add feature names if not present
            if 'feature_names' not in model_info:
                model_info['feature_names'] = model_info.get('feature_columns', [])
            
            return model, model_info
        
        except Exception as e:
            logger.error(f"Error loading new model from {model_path} with info {info_path}: {e}")
            return None, None
    
    def predict(self, symbol: str, features: np.ndarray) -> Optional[np.ndarray]:
        """
        Make prediction with model
        
        Args:
            symbol: Symbol to use model for
            features: Input features with shape (sequence_length, num_features)
            
        Returns:
            Predictions with shape (num_targets,) or None if prediction fails
        """
        if symbol not in self.models:
            logger.error(f"No model available for {symbol}")
            return None
            
        model = self.models[symbol]
        model_info = self.model_info[symbol]
        
        try:
            # Check if feature names are available
            feature_names = model_info.get('feature_names', [])
            
            # Ensure features have correct shape
            # LSTM expects (batch_size, seq_len, num_features)
            if len(features.shape) == 2:  # (seq_len, num_features)
                # Add batch dimension
                features = features.reshape(1, *features.shape)
            
            # Convert to PyTorch tensor
            features_tensor = torch.FloatTensor(features)
            
            # Make prediction
            with torch.no_grad():
                prediction = model(features_tensor).numpy()
            
            # Return first (and only) prediction
            return prediction[0]
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {e}")
            return None
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with available models"""
        return list(self.models.keys())
    
    def get_target_names(self, symbol: str) -> List[str]:
        """Get target names for symbol"""
        if symbol not in self.model_info:
            return []
            
        return self.model_info[symbol].get('target_names', [])
    
    def get_model_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get model metrics for symbol"""
        if symbol not in self.model_info:
            return {}
            
        return self.model_info[symbol].get('metrics', {})

def predict_price_movement(features: np.ndarray, symbol: str = 'uniswap_ETHWETH', target_idx: int = 0) -> Dict[str, Any]:
    """
    Predict price movement using trained model
    
    Args:
        features: Features for prediction with shape (sequence_length, num_features)
        symbol: Symbol to predict for
        target_idx: Index of target to predict (0 = 1h, 1 = 3h, 2 = 6h, 3 = 12h, 4 = 24h)
        
    Returns:
        Dictionary with prediction results
    """
    # Initialize predictor
    predictor = ModelPredictor()
    
    # Check if model is available
    if symbol not in predictor.get_available_symbols():
        logger.error(f"No model available for {symbol}")
        return {'error': f"No model available for {symbol}"}
    
    # Get target names
    target_names = predictor.get_target_names(symbol)
    if not target_names or target_idx >= len(target_names):
        logger.error(f"Invalid target index {target_idx} for {symbol}")
        return {'error': f"Invalid target index {target_idx} for {symbol}"}
    
    # Make prediction
    prediction = predictor.predict(symbol, features)
    if prediction is None:
        logger.error(f"Failed to make prediction for {symbol}")
        return {'error': f"Failed to make prediction for {symbol}"}
    
    # Get metrics for this model and target
    metrics = predictor.get_model_metrics(symbol)
    target_metrics = metrics.get(target_names[target_idx], {})
    
    # Create result
    result = {
        'symbol': symbol,
        'target': target_names[target_idx],
        'prediction': float(prediction[target_idx]),
        'metrics': target_metrics,
        'confidence': 1.0 - float(target_metrics.get('rmse', 0.5))  # Higher RMSE = lower confidence
    }
    
    return result

def predict_all_targets(features: np.ndarray, symbol: str = 'uniswap_ETHWETH') -> Dict[str, Any]:
    """
    Predict all targets using trained model
    
    Args:
        features: Features for prediction with shape (sequence_length, num_features)
        symbol: Symbol to predict for
        
    Returns:
        Dictionary with prediction results for all targets
    """
    # Initialize predictor
    predictor = ModelPredictor()
    
    # Check if model is available
    if symbol not in predictor.get_available_symbols():
        logger.error(f"No model available for {symbol}")
        return {'error': f"No model available for {symbol}"}
    
    # Get target names
    target_names = predictor.get_target_names(symbol)
    if not target_names:
        logger.error(f"No target names found for {symbol}")
        return {'error': f"No target names found for {symbol}"}
    
    # Make prediction
    prediction = predictor.predict(symbol, features)
    if prediction is None:
        logger.error(f"Failed to make prediction for {symbol}")
        return {'error': f"Failed to make prediction for {symbol}"}
    
    # Get metrics for this model
    metrics = predictor.get_model_metrics(symbol)
    
    # Create result
    results = {
        'symbol': symbol,
        'predictions': {}
    }
    
    for i, target_name in enumerate(target_names):
        target_metrics = metrics.get(target_name, {})
        results['predictions'][target_name] = {
            'value': float(prediction[i]),
            'metrics': target_metrics,
            'confidence': 1.0 - float(target_metrics.get('rmse', 0.5))  # Higher RMSE = lower confidence
        }
    
    return results

def detect_arbitrage_opportunity(predictions: Dict[str, Dict[str, Any]], threshold: float = 0.01) -> Dict[str, Any]:
    """
    Detect arbitrage opportunity from predictions
    
    Args:
        predictions: Dictionary of predictions by symbol
        threshold: Threshold for considering an opportunity
        
    Returns:
        Dictionary with opportunity details or None if no opportunity found
    """
    opportunities = []
    
    # For each symbol
    for symbol, prediction_data in predictions.items():
        # Get 1h prediction
        if 'predictions' not in prediction_data or 'return_1h' not in prediction_data['predictions']:
            continue
            
        pred_1h = prediction_data['predictions']['return_1h']
        
        # Check if prediction exceeds threshold (positive or negative)
        if abs(pred_1h['value']) > threshold:
            # Create opportunity
            opportunity = {
                'symbol': symbol,
                'type': 'long' if pred_1h['value'] > 0 else 'short',
                'predicted_return': pred_1h['value'],
                'confidence': pred_1h['confidence'],
                'time_frame': '1h',
                'threshold': threshold
            }
            
            opportunities.append(opportunity)
    
    return {
        'found': len(opportunities) > 0,
        'opportunities': opportunities,
        'count': len(opportunities)
    }

def main():
    """Test model prediction functionality"""
    # Get available models to understand what we're testing
    predictor = ModelPredictor()
    available_symbols = predictor.get_available_symbols()
    
    print(f"Found {len(available_symbols)} available models: {available_symbols}")
    
    # Test each model with appropriate feature dimensions
    for symbol in available_symbols:
        # Get model info to determine input dimensions
        model_info = predictor.model_info.get(symbol, {})
        input_size = model_info.get('input_size', 10)  # Default to 10 features for newer models
        
        # Create appropriate test features for this model
        test_features = np.random.rand(24, input_size)  # 24 time steps, model-specific features
        
        # Test prediction for this symbol
        print(f"\nTesting predictions for {symbol}:")
        result = predict_all_targets(test_features, symbol)
        print(json.dumps(result, indent=2))
    
    # Test specifically our newly trained model
    if 'binance_ADAUSDC' in available_symbols:
        print("\nTesting predictions for our newly trained ADAUSDC model:")
        # ADAUSDC model uses 10 features
        test_features = np.random.rand(24, 10)
        result = predict_all_targets(test_features, 'binance_ADAUSDC')
        print(json.dumps(result, indent=2))
    
    # Test opportunity detection with available models
    print("\nTesting arbitrage opportunity detection:")
    all_predictions = {}
    
    for symbol in available_symbols:
        model_info = predictor.model_info.get(symbol, {})
        input_size = model_info.get('input_size', 10)
        test_features = np.random.rand(24, input_size)
        
        try:
            all_predictions[symbol] = predict_all_targets(test_features, symbol)
        except Exception as e:
            print(f"Error predicting for {symbol}: {e}")
    
    # Detect arbitrage opportunities
    opportunities = detect_arbitrage_opportunity(all_predictions, threshold=0.005)
    print(json.dumps(opportunities, indent=2))

if __name__ == "__main__":
    main()
