"""
Advanced ML models for MEV opportunity prediction
Implements ensemble techniques combining LSTM, Transformer, and gradient boosting
"""

import os
import logging
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

# Import base LSTM model
from ml_model.lstm_predictor import LSTMModel, LSTMPredictor, OpportunityData

# Setup logger
logger = logging.getLogger(__name__)

class TransformerModel(nn.Module):
    """
    Transformer model for time series prediction
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, nhead=4, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = nn.Dropout(dropout)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim*4,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # If input is only 2D (for single samples), add sequence dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        batch_size, seq_len, _ = x.shape
        
        # Transform input to embedding dimension
        x = self.input_embedding(x)  # [batch_size, seq_len, hidden_dim]
        
        # Transpose for transformer: [seq_len, batch_size, hidden_dim]
        x = x.transpose(0, 1)
        
        # Apply positional encoding and transformer
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Use the output of the last sequence element
        x = x[-1, :, :]  # [batch_size, hidden_dim]
        
        # Final output layer
        x = self.output_layer(x)  # [batch_size, output_dim]
        
        return x

class EnsemblePredictor:
    """
    Advanced predictor using ensemble of models for better predictions
    """
    def __init__(self, db=None):
        self.db = db
        self.lstm_model = None
        self.transformer_model = None
        self.gbm_model = None
        self.rf_model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.model_path = os.environ.get('MODEL_PATH', 'ml_model/ensemble_models.pkl')
        self.lstm_predictor = LSTMPredictor(db)  # Base LSTM predictor as fallback
        
        # Model weights for ensemble
        self.model_weights = {
            'lstm': 0.3,
            'transformer': 0.3,
            'gbm': 0.25,
            'rf': 0.15
        }
        
        # Historical data cache for online learning
        self.historical_data = []
        self.last_retrain_time = None
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load all trained models for the ensemble"""
        try:
            # Try to load the ensemble model package
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.lstm_model = model_data.get('lstm_model')
                self.transformer_model = model_data.get('transformer_model')
                self.gbm_model = model_data.get('gbm_model')
                self.rf_model = model_data.get('rf_model')
                self.feature_scaler = model_data.get('feature_scaler')
                self.target_scaler = model_data.get('target_scaler')
                self.model_weights = model_data.get('model_weights', self.model_weights)
                self.feature_names = model_data.get('feature_names', [])
                self.hyperparams = model_data.get('hyperparams', {})
                
                logger.info(f"Loaded ensemble models from {self.model_path}")
                
                # Load base LSTM as fallback
                self.lstm_predictor.load_model()
                
                # Get feature names from LSTM if not in ensemble
                if not self.feature_names and hasattr(self.lstm_predictor, 'feature_names'):
                    self.feature_names = self.lstm_predictor.feature_names
                
            else:
                logger.warning(f"Ensemble model file not found at {self.model_path}. Using LSTM model only.")
                self.lstm_predictor.load_model()
                self.feature_names = self.lstm_predictor.feature_names if hasattr(self.lstm_predictor, 'feature_names') else []
                
        except Exception as e:
            logger.error(f"Error loading ensemble models: {e}")
            # Make sure the base LSTM is loaded as fallback
            self.lstm_predictor.load_model()
    
    def extract_features(self, opportunity: Dict) -> np.ndarray:
        """
        Extract features for prediction using the expanded feature set
        
        Args:
            opportunity: Opportunity dictionary
            
        Returns:
            Numpy array of features
        """
        # Use the base LSTM feature extraction and extend it
        base_features = self.lstm_predictor.extract_features(opportunity)
        
        # Get more market data for improved predictions
        try:
            from utils.web3_helpers import get_web3_provider, get_eth_price_usd, get_token_price_usd
            web3 = get_web3_provider()
            
            # Additional features for the ensemble model
            # We'll extend the base vector with additional features
            additional_features = np.zeros(5)  # 5 more features
            
            # 1. Block time trend (normalized)
            current_block = web3.eth.block_number
            
            # We can infer block time from recent blocks
            try:
                recent_block = web3.eth.get_block(current_block)
                prev_block = web3.eth.get_block(current_block - 10)
                
                if recent_block and prev_block:
                    avg_block_time = (recent_block['timestamp'] - prev_block['timestamp']) / 10
                    # Normalize: 12-15 seconds is normal, <10 is fast, >20 is slow
                    normalized_block_time = min(1.0, max(0.0, (avg_block_time - 5) / 20))
                    additional_features[0] = normalized_block_time
            except:
                additional_features[0] = 0.5  # Default
            
            # 2. Network congestion (based on pending transaction count)
            try:
                pending_count = web3.eth.get_transaction_count('pending') - web3.eth.get_transaction_count('latest')
                # Normalize: 0-50k pending transactions
                normalized_congestion = min(1.0, max(0.0, pending_count / 50000))
                additional_features[1] = normalized_congestion
            except:
                additional_features[1] = 0.5  # Default
            
            # 3. Recent opportunity success rate (from historical data)
            if self.db:
                try:
                    from models import Transaction
                    recent_time = datetime.utcnow() - timedelta(hours=1)
                    
                    # Query recent transactions
                    with self.db.session.no_autoflush:
                        total_recent = Transaction.query.filter(
                            Transaction.created_at >= recent_time
                        ).count()
                        
                        successful_recent = Transaction.query.filter(
                            Transaction.created_at >= recent_time,
                            Transaction.status == 'confirmed'
                        ).count()
                    
                    success_rate = successful_recent / max(1, total_recent)
                    additional_features[2] = success_rate
                except Exception as e:
                    logger.debug(f"Error calculating success rate: {e}")
                    additional_features[2] = 0.5  # Default
            else:
                additional_features[2] = 0.5  # Default
            
            # 4. Gas price volatility (normalized)
            # This indicates how much gas prices are changing
            try:
                current_gas_price = web3.eth.gas_price
                # We'd ideally track this over time, but for now use a placeholder
                additional_features[3] = 0.5  # Default
                
                # If opportunity has gas_price_volatility, use it
                if 'gas_price_volatility' in opportunity:
                    additional_features[3] = min(1.0, max(0.0, opportunity['gas_price_volatility']))
            except:
                additional_features[3] = 0.5  # Default
            
            # 5. Strategy-specific feature
            # For arbitrage: complexity score
            # For sandwich: probability of victim tx being included
            strategy_type = opportunity.get('strategy_type', 'arbitrage')
            
            if strategy_type == 'arbitrage':
                arb_type = opportunity.get('arbitrage_type', 'direct')
                if arb_type == 'triangular':
                    complexity = 0.8  # More complex
                else:
                    complexity = 0.3  # Less complex
                additional_features[4] = complexity
            elif strategy_type == 'sandwich':
                victim_gas_price = opportunity.get('victim_gas_price', 0)
                current_gas_gwei = web3.from_wei(web3.eth.gas_price, 'gwei')
                if victim_gas_price > 0:
                    # If victim's gas price is competitive, higher chance of inclusion
                    relative_gas = victim_gas_price / current_gas_gwei
                    inclusion_probability = min(1.0, max(0.0, relative_gas))
                    additional_features[4] = inclusion_probability
                else:
                    additional_features[4] = 0.5  # Default
            else:
                additional_features[4] = 0.5  # Default
            
            # Combine base and additional features
            extended_features = np.concatenate([base_features.flatten(), additional_features])
            
            return extended_features.reshape(1, -1)  # Return as 2D array with 1 row
            
        except Exception as e:
            logger.error(f"Error extracting extended features: {e}")
            # Return just the base features if there's an error
            return base_features
    
    def predict_with_lstm(self, features_tensor) -> Tuple[float, float]:
        """
        Make a prediction using the LSTM model
        
        Args:
            features_tensor: Feature tensor
            
        Returns:
            Tuple of (profit_adjustment, confidence_score)
        """
        if self.lstm_model is None:
            # Use the backup LSTM model
            return None, None
        
        try:
            # Create sequence for LSTM if needed
            seq_length = self.hyperparams.get('sequence_length', 1)
            if seq_length > 1 and len(features_tensor.shape) == 2:
                # Replicate feature vector to make a sequence
                features_tensor = features_tensor.unsqueeze(0).repeat(1, seq_length, 1)
            elif len(features_tensor.shape) == 2:
                features_tensor = features_tensor.unsqueeze(0)  # Add batch dimension
            
            # Get prediction
            with torch.no_grad():
                self.lstm_model.eval()
                outputs = self.lstm_model(features_tensor)
                
            # Process outputs
            profit_adjustment = outputs[0][0].item()
            confidence_score = min(max(outputs[0][1].item(), 0.0), 1.0)  # Clamp to [0, 1]
            
            return profit_adjustment, confidence_score
            
        except Exception as e:
            logger.error(f"Error predicting with LSTM: {e}")
            return None, None
    
    def predict_with_transformer(self, features_tensor) -> Tuple[float, float]:
        """
        Make a prediction using the Transformer model
        
        Args:
            features_tensor: Feature tensor
            
        Returns:
            Tuple of (profit_adjustment, confidence_score)
        """
        if self.transformer_model is None:
            return None, None
        
        try:
            # Create sequence for Transformer if needed
            with torch.no_grad():
                self.transformer_model.eval()
                outputs = self.transformer_model(features_tensor)
                
            # Process outputs
            profit_adjustment = outputs[0][0].item()
            confidence_score = min(max(outputs[0][1].item(), 0.0), 1.0)  # Clamp to [0, 1]
            
            return profit_adjustment, confidence_score
            
        except Exception as e:
            logger.error(f"Error predicting with Transformer: {e}")
            return None, None
    
    def predict_with_gbm(self, features_np) -> Tuple[float, float]:
        """
        Make a prediction using the Gradient Boosting model
        
        Args:
            features_np: Feature numpy array
            
        Returns:
            Tuple of (profit_adjustment, confidence_score)
        """
        if self.gbm_model is None:
            return None, None
        
        try:
            # Gradient Boosting models expect 2D array
            if len(features_np.shape) == 3:
                # If we have a sequence, use the last timestep
                features_np = features_np[0, -1, :].reshape(1, -1)
            
            # Predict
            predictions = self.gbm_model.predict(features_np)
            
            # GBM model outputs both profit adjustment and confidence
            if len(predictions[0]) >= 2:
                profit_adjustment = predictions[0][0]
                confidence_score = min(max(predictions[0][1], 0.0), 1.0)  # Clamp to [0, 1]
            else:
                profit_adjustment = predictions[0]
                confidence_score = 0.7  # Default confidence
            
            return profit_adjustment, confidence_score
            
        except Exception as e:
            logger.error(f"Error predicting with GBM: {e}")
            return None, None
    
    def predict_with_rf(self, features_np) -> Tuple[float, float]:
        """
        Make a prediction using the Random Forest model
        
        Args:
            features_np: Feature numpy array
            
        Returns:
            Tuple of (profit_adjustment, confidence_score)
        """
        if self.rf_model is None:
            return None, None
        
        try:
            # RF models expect 2D array
            if len(features_np.shape) == 3:
                # If we have a sequence, use the last timestep
                features_np = features_np[0, -1, :].reshape(1, -1)
            
            # Predict
            predictions = self.rf_model.predict(features_np)
            
            # RF model outputs both profit adjustment and confidence
            if len(predictions[0]) >= 2:
                profit_adjustment = predictions[0][0]
                confidence_score = min(max(predictions[0][1], 0.0), 1.0)  # Clamp to [0, 1]
            else:
                profit_adjustment = predictions[0]
                confidence_score = 0.7  # Default confidence
            
            return profit_adjustment, confidence_score
            
        except Exception as e:
            logger.error(f"Error predicting with Random Forest: {e}")
            return None, None
    
    def predict_arbitrage_opportunity(self, opportunity: Dict) -> Tuple[float, float]:
        """
        Predict profitability and confidence for an opportunity using ensemble
        
        Args:
            opportunity: Opportunity dictionary
            
        Returns:
            Tuple of (adjusted_profit, confidence_score)
        """
        # Extract features for all models
        features = self.extract_features(opportunity)
        
        # Store original expected profit
        expected_profit = opportunity.get('expected_profit', 0) or opportunity.get('estimated_profit', 0)
        
        # If no models loaded, fall back to the LSTM predictor
        if (self.lstm_model is None and 
            self.transformer_model is None and 
            self.gbm_model is None and 
            self.rf_model is None):
            return self.lstm_predictor.predict_arbitrage_opportunity(opportunity)
        
        # Track predictions from each model
        predictions = {}
        
        # Scale features if scaler exists
        if self.feature_scaler:
            features_scaled = self.feature_scaler.transform(features)
        else:
            features_scaled = features
        
        # Convert to tensor for deep learning models
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        
        # Get predictions from each model
        lstm_pred = self.predict_with_lstm(features_tensor)
        if lstm_pred[0] is not None:
            predictions['lstm'] = lstm_pred
        
        transformer_pred = self.predict_with_transformer(features_tensor)
        if transformer_pred[0] is not None:
            predictions['transformer'] = transformer_pred
        
        gbm_pred = self.predict_with_gbm(features_scaled)
        if gbm_pred[0] is not None:
            predictions['gbm'] = gbm_pred
        
        rf_pred = self.predict_with_rf(features_scaled)
        if rf_pred[0] is not None:
            predictions['rf'] = rf_pred
        
        # If we don't have any predictions, use the base LSTM
        if not predictions:
            return self.lstm_predictor.predict_arbitrage_opportunity(opportunity)
        
        # Calculate weighted average for profit adjustment and confidence
        total_weight = 0
        weighted_profit_adjustment = 0
        weighted_confidence = 0
        
        for model_name, (profit_adj, confidence) in predictions.items():
            weight = self.model_weights.get(model_name, 0)
            weighted_profit_adjustment += profit_adj * weight
            weighted_confidence += confidence * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            profit_adjustment = weighted_profit_adjustment / total_weight
            confidence_score = weighted_confidence / total_weight
        else:
            # Fallback to default
            profit_adjustment = 0
            confidence_score = 0.5
        
        # Apply adjustment to expected profit
        adjusted_profit = expected_profit * (1.0 + profit_adjustment)
        
        # Ensure adjusted profit is not negative
        adjusted_profit = max(0.0, adjusted_profit)
        
        logger.debug(f"Ensemble prediction: expected={expected_profit}, adjusted={adjusted_profit}, confidence={confidence_score}")
        
        # Store this opportunity for potential online learning
        self.historical_data.append({
            'features': features,
            'opportunity': opportunity,
            'prediction': (adjusted_profit, confidence_score),
            'timestamp': datetime.now()
        })
        
        # Periodically clean old historical data
        self._clean_historical_data()
        
        return adjusted_profit, confidence_score
    
    def should_execute_opportunity(self, opportunity: Dict) -> bool:
        """
        Determine if an opportunity should be executed
        
        Args:
            opportunity: Opportunity dictionary
            
        Returns:
            Boolean indicating if opportunity should be executed
        """
        from config import ML_CONFIG
        
        # Get prediction
        adjusted_profit, confidence = self.predict_arbitrage_opportunity(opportunity)
        
        # Get threshold from config
        confidence_threshold = ML_CONFIG.get('CONFIDENCE_THRESHOLD', 0.7)
        
        # Calculate minimum profitable gas price
        from utils.web3_helpers import get_web3_provider
        web3 = get_web3_provider()
        current_gas_price_gwei = web3.from_wei(web3.eth.gas_price, 'gwei')
        
        # Estimate gas usage based on strategy type
        if opportunity.get('strategy_type') == 'sandwich':
            estimated_gas = 600000  # Higher for sandwich (front + back run)
        elif opportunity.get('arbitrage_type') == 'triangular':
            estimated_gas = 500000  # Higher for triangular (more complex)
        else:
            estimated_gas = 300000  # Lower for direct arbitrage
        
        # Convert gas cost to ETH
        gas_cost_eth = (estimated_gas * web3.eth.gas_price) / 1e18
        
        # Check if expected profit exceeds gas cost with margin
        profitable = adjusted_profit > gas_cost_eth * 1.3  # 30% margin over gas
        
        # Adjust confidence threshold based on network conditions
        adjusted_threshold = confidence_threshold
        
        # If gas prices are very high, require higher confidence
        if current_gas_price_gwei > 150:  # Very high gas
            adjusted_threshold = confidence_threshold + 0.1
        elif current_gas_price_gwei < 50:  # Low gas
            adjusted_threshold = confidence_threshold - 0.05
        
        # Decide based on confidence and profitability
        execute = profitable and confidence >= adjusted_threshold
        
        logger.info(f"Execution decision: profitable={profitable}, confidence={confidence}, threshold={adjusted_threshold}, execute={execute}")
        
        return execute
    
    def _clean_historical_data(self):
        """Clean old historical data to prevent memory bloat"""
        try:
            # Keep only last 1000 entries
            if len(self.historical_data) > 1000:
                self.historical_data = self.historical_data[-1000:]
            
            # Remove entries older than 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.historical_data = [
                entry for entry in self.historical_data 
                if entry['timestamp'] > cutoff_time
            ]
        except Exception as e:
            logger.error(f"Error cleaning historical data: {e}")
    
    def update_from_transaction_result(self, opportunity: Dict, transaction_result: Dict) -> None:
        """
        Update model weights based on transaction results (online learning)
        
        Args:
            opportunity: Original opportunity dictionary
            transaction_result: Result of the transaction execution
        """
        # This would implement online learning to adjust model weights
        # Not fully implemented in this version
        pass

def get_enhanced_predictor(db=None) -> EnsemblePredictor:
    """
    Get an enhanced predictor instance
    
    Args:
        db: Database connection
        
    Returns:
        Enhanced predictor instance
    """
    return EnsemblePredictor(db)
