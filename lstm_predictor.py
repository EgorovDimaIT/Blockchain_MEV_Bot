import os
import logging
import numpy as np
import pandas as pd
import torch
import json
import requests
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from web3 import Web3
from ml_model.lstm_model import PricePredictionLSTM, get_model_trainer
from ml_model.transformer_model import TransformerModel, get_transformer_trainer
from ml_model.data_processor import get_data_processor
from data_collection.data_downloader import get_data_downloader
from models import MLModel, Transaction, ArbitrageOpportunity
from utils.web3_helpers import get_web3_provider

logger = logging.getLogger(__name__)

class PredictorException(Exception):
    """Exception raised for errors in the predictor."""
    pass

class ArbitragePredictor:
    """
    ML predictor for arbitrage opportunities
    
    This class uses LSTM and Transformer models to predict
    the profitability and confidence of arbitrage opportunities.
    """
    
    def __init__(self, db=None):
        """
        Initialize predictor
        
        Args:
            db: Database connection
        """
        self.db = db
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = {}
        self.model_weights = {
            'lstm': 0.6,
            'transformer': 0.4
        }
        self.feature_columns = None
        self.feature_scaler = None
        self.target_scaler = None
        self.market_state = 'normal'  # normal, volatile, trending_up, trending_down
        self.last_retrain = datetime.now() - timedelta(days=8)  # Force initial training
        self.last_market_update = datetime.now() - timedelta(hours=2)  # Force initial update
        self.default_sequence_length = 60
        self.models_dir = 'ml_model/saved_models'
        
        # Web3 provider for blockchain queries
        self.web3 = get_web3_provider()
        
        # Token price cache to minimize API calls
        self.token_price_cache = {}
        self.token_price_cache_expiry = 60  # seconds
        self.token_price_last_updated = {}
        
        # DEX liquidity cache
        self.dex_liquidity_cache = {}
        self.dex_liquidity_cache_expiry = 300  # seconds
        self.dex_liquidity_last_updated = {}
        
        # Initialize models directory
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize models
        self._load_or_train_models()
    
    def _load_or_train_models(self) -> bool:
        """
        Load existing models or train new ones if not available
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # First try to load models from database
            if self.db:
                from flask import current_app
                with current_app.app_context():
                    models = MLModel.query.filter_by(is_active=True).all()
                    
                    if models:
                        for model in models:
                            self._load_model_from_db(model)
                        
                        if self.models:
                            logger.info(f"Loaded {len(self.models)} active models from database")
                            return True
            
            # If no active models in DB or loading failed, load from filesystem
            lstm_files = [f for f in os.listdir(self.models_dir) if f.startswith('lstm_model') and f.endswith('.pt')]
            transformer_files = [f for f in os.listdir(self.models_dir) if f.startswith('transformer_model') and f.endswith('.pt')]
            
            if lstm_files:
                # Sort by timestamp (newest first)
                lstm_files.sort(reverse=True)
                lstm_trainer = get_model_trainer()
                lstm_trainer.load_model(os.path.join(self.models_dir, lstm_files[0]))
                self.models['lstm'] = lstm_trainer.model
                logger.info(f"Loaded LSTM model from {lstm_files[0]}")
            
            if transformer_files:
                # Sort by timestamp (newest first)
                transformer_files.sort(reverse=True)
                transformer_trainer = get_transformer_trainer()
                transformer_trainer.load_model(os.path.join(self.models_dir, transformer_files[0]))
                self.models['transformer'] = transformer_trainer.model
                logger.info(f"Loaded Transformer model from {transformer_files[0]}")
            
            # If models were loaded from filesystem
            if self.models:
                return True
            
            # If no models found, train new ones
            logger.info("No existing models found, training new models...")
            return self.train_models()
            
        except Exception as e:
            logger.error(f"Error loading or training models: {e}")
            return False
    
    def _load_model_from_db(self, model_record: MLModel) -> bool:
        """
        Load model from database record
        
        Args:
            model_record: MLModel record from database
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = model_record.file_path
            model_type = model_record.model_type.lower()
            
            if model_type == 'lstm':
                lstm_trainer = get_model_trainer()
                success = lstm_trainer.load_model(model_path)
                if success:
                    self.models['lstm'] = lstm_trainer.model
                    logger.info(f"Loaded LSTM model {model_record.name} from database")
                    return True
                    
            elif model_type == 'transformer':
                transformer_trainer = get_transformer_trainer()
                success = transformer_trainer.load_model(model_path)
                if success:
                    self.models['transformer'] = transformer_trainer.model
                    logger.info(f"Loaded Transformer model {model_record.name} from database")
                    return True
            
            logger.warning(f"Failed to load model {model_record.name} of type {model_type}")
            return False
            
        except Exception as e:
            logger.error(f"Error loading model {model_record.name} from database: {e}")
            return False
    
    def train_models(self, days: int = 7) -> bool:
        """
        Train new models with recent and historical data
        
        Args:
            days: Number of days of recent data to include
            
        Returns:
            True if training succeeded, False otherwise
        """
        try:
            logger.info(f"Training models with {days} days of recent data")
            
            # Download and process data
            data_downloader = get_data_downloader()
            data_dict = {}
            symbol = 'ETHUSDT'  
            data_dict[symbol] = data_downloader.download_historical_data(symbol=symbol, days=days, source='binance')
            
            if not data_dict:
                logger.error("No data available for training")
                return False
            
            # Process data
            data_processor = get_data_processor()
            train_data, metadata = data_processor.prepare_training_data(
                data_dict, 
                sequence_length=self.default_sequence_length
            )
            
            if not train_data or not metadata:
                logger.error("Failed to prepare training data")
                return False
            
            # Store metadata for later use
            self.feature_columns = metadata.get('feature_columns')
            if 'scaler_path' in metadata and os.path.exists(metadata['scaler_path']):
                import pickle
                with open(metadata['scaler_path'], 'rb') as f:
                    self.feature_scaler = pickle.load(f)
            
            # Train LSTM model
            lstm_trainer = get_model_trainer()
            if 'lstm' not in self.models or self.models['lstm'] is None:
                input_dim = train_data['X_train'].shape[2]
                lstm_trainer.create_model(
                    input_dim=input_dim,
                    hidden_dim=128,
                    num_layers=2,
                    output_dim=1,
                    dropout=0.2
                )
            else:
                lstm_trainer.model = self.models['lstm']
                
            lstm_results = lstm_trainer.train(
                train_data,
                epochs=50,
                batch_size=64,
                learning_rate=0.001,
                early_stopping=5
            )
            
            # Update model
            self.models['lstm'] = lstm_trainer.model
            
            # Save model
            lstm_path = lstm_trainer.save_model('lstm_model')
            
            # Train Transformer model
            transformer_trainer = get_transformer_trainer()
            if 'transformer' not in self.models or self.models['transformer'] is None:
                input_dim = train_data['X_train'].shape[2]
                transformer_trainer.create_model(
                    input_dim=input_dim,
                    d_model=64,
                    nhead=4,
                    num_encoder_layers=2,
                    dim_feedforward=256,
                    output_dim=1,
                    dropout=0.1
                )
            else:
                transformer_trainer.model = self.models['transformer']
                
            transformer_results = transformer_trainer.train(
                train_data,
                epochs=50,
                batch_size=64,
                learning_rate=0.001,
                early_stopping=5
            )
            
            # Update model
            self.models['transformer'] = transformer_trainer.model
            
            # Save model
            transformer_path = transformer_trainer.save_model('transformer_model')
            
            # Save models to database if available
            if self.db:
                self._save_models_to_db(lstm_path, transformer_path, train_data)
            
            self.last_retrain = datetime.now()
            logger.info("Models trained and saved successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def _save_models_to_db(self, lstm_path: str, transformer_path: str, train_data: Dict) -> None:
        """
        Save trained models to database
        
        Args:
            lstm_path: Path to LSTM model
            transformer_path: Path to Transformer model
            train_data: Training data dict
        """
        if not self.db:
            return
            
        try:
            from flask import current_app
            with current_app.app_context():
                # Deactivate old models
                old_models = MLModel.query.filter_by(is_active=True).all()
                for model in old_models:
                    model.is_active = False
                
                # Calculate accuracy metrics
                lstm_trainer = get_model_trainer()
                lstm_metrics = lstm_trainer.evaluate(train_data)
                
                transformer_trainer = get_transformer_trainer()
                transformer_metrics = transformer_trainer.evaluate(train_data)
                
                # Create new LSTM model record
                lstm_model = MLModel(
                    name=f"LSTM_{datetime.now().strftime('%Y%m%d')}",
                    model_type='LSTM',
                    file_path=lstm_path,
                    created_at=datetime.now(),
                    accuracy=float(lstm_metrics.get('directional_accuracy', 0.5)),
                    is_active=True
                )
                lstm_model.set_hyperparameters({
                    'hidden_dim': 128,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'loss': float(lstm_metrics.get('test_loss', 0)),
                    'rmse': float(lstm_metrics.get('rmse', 0))
                })
                
                # Create new Transformer model record
                transformer_model = MLModel(
                    name=f"Transformer_{datetime.now().strftime('%Y%m%d')}",
                    model_type='Transformer',
                    file_path=transformer_path,
                    created_at=datetime.now(),
                    accuracy=float(transformer_metrics.get('directional_accuracy', 0.5)),
                    is_active=True
                )
                transformer_model.set_hyperparameters({
                    'd_model': 64,
                    'nhead': 4,
                    'num_encoder_layers': 2,
                    'dim_feedforward': 256,
                    'dropout': 0.1,
                    'loss': float(transformer_metrics.get('test_loss', 0)),
                    'rmse': float(transformer_metrics.get('rmse', 0))
                })
                
                # Save to database
                self.db.session.add(lstm_model)
                self.db.session.add(transformer_model)
                self.db.session.commit()
                
                logger.info(f"Saved models to database: {lstm_model.name}, {transformer_model.name}")
                
        except Exception as e:
            logger.error(f"Error saving models to database: {e}")
            if self.db.session.is_active:
                self.db.session.rollback()
    
    def predict_arbitrage_opportunity(self, opportunity: Dict) -> Tuple[float, float]:
        """
        Predict if an arbitrage opportunity is worth taking
        
        Args:
            opportunity: Dictionary with arbitrage opportunity data
            
        Returns:
            Tuple of (adjusted_profit, confidence_score)
        """
        try:
            # Check if retraining is needed (weekly)
            if (datetime.now() - self.last_retrain).days >= 7:
                logger.info("Weekly retraining triggered")
                self.train_models()
            
            # Update market state if needed
            if (datetime.now() - self.last_market_update).seconds > 3600:  # hourly update
                self._update_market_state()
            
            # Enrich opportunity with real DEX data
            try:
                enriched_opportunity = self._enrich_opportunity(opportunity)
            except Exception as e:
                logger.error(f"Error enriching opportunity: {e}")
                enriched_opportunity = opportunity  # Fallback to original opportunity
            
            # Get predictions from all models
            predictions = {}
            confidence_scores = {}
            
            # LSTM prediction
            if 'lstm' in self.models and self.models['lstm'] is not None:
                try:
                    lstm_features = self._prepare_features_for_lstm(enriched_opportunity)
                    lstm_trainer = get_model_trainer()
                    lstm_trainer.model = self.models['lstm']
                    lstm_prediction = lstm_trainer.predict(lstm_features)
                    
                    # Convert prediction to profit adjustment factor
                    lstm_profit_adj = self._prediction_to_profit_adjustment(lstm_prediction[0])
                    lstm_confidence = self._calculate_confidence(lstm_prediction[0], enriched_opportunity)
                    
                    predictions['lstm'] = lstm_profit_adj
                    confidence_scores['lstm'] = lstm_confidence
                    
                except Exception as e:
                    logger.error(f"Error in LSTM prediction: {e}")
            
            # Transformer prediction
            if 'transformer' in self.models and self.models['transformer'] is not None:
                try:
                    transformer_features = self._prepare_features_for_transformer(enriched_opportunity)
                    transformer_trainer = get_transformer_trainer()
                    transformer_trainer.model = self.models['transformer']
                    transformer_prediction = transformer_trainer.predict(transformer_features)
                    
                    # Convert prediction to profit adjustment factor
                    transformer_profit_adj = self._prediction_to_profit_adjustment(transformer_prediction[0])
                    transformer_confidence = self._calculate_confidence(transformer_prediction[0], enriched_opportunity)
                    
                    predictions['transformer'] = transformer_profit_adj
                    confidence_scores['transformer'] = transformer_confidence
                    
                except Exception as e:
                    logger.error(f"Error in Transformer prediction: {e}")
            
            # If no predictions available
            if not predictions:
                logger.warning("No predictions available, using default values")
                return float(opportunity.get('expected_profit', 0.0)), 0.5
            
            # Calculate weighted average
            weighted_profit_adj = 0.0
            weighted_confidence = 0.0
            total_weight = 0.0
            
            for model_name in predictions:
                weight = self.model_weights.get(model_name, 0.5)
                weighted_profit_adj += predictions[model_name] * weight
                weighted_confidence += confidence_scores[model_name] * weight
                total_weight += weight
            
            if total_weight > 0:
                final_profit_adj = weighted_profit_adj / total_weight
                final_confidence = weighted_confidence / total_weight
            else:
                final_profit_adj = 1.0
                final_confidence = 0.5
            
            # Apply market state adjustments
            final_profit_adj = self._adjust_for_market_state(final_profit_adj)
            
            # Calculate adjusted profit
            expected_profit = float(opportunity.get('expected_profit', 0.0))
            adjusted_profit = expected_profit * final_profit_adj
            
            # Ensure profit is never negative
            adjusted_profit = max(0.0, adjusted_profit)
            
            return adjusted_profit, final_confidence
            
        except Exception as e:
            logger.error(f"Error predicting arbitrage opportunity: {e}")
            # Default fallback - use expected profit with 50% confidence
            return float(opportunity.get('expected_profit', 0.0)), 0.5
    
    def _enrich_opportunity(self, opportunity: Dict) -> Dict:
        """
        Enrich arbitrage opportunity with real DEX data
        
        Args:
            opportunity: Original opportunity dictionary
            
        Returns:
            Enriched opportunity with additional data
        """
        # Make a copy to avoid modifying the original
        enriched = opportunity.copy()
        
        # Get token addresses
        token_in_address = opportunity.get('token_in', '').lower()
        token_out_address = opportunity.get('token_out', '').lower()
        token_mid_address = opportunity.get('token_mid', '').lower()
        
        # Skip if no token addresses provided
        if not token_in_address or not token_out_address:
            logger.warning("Missing token addresses, skipping enrichment")
            return enriched
        
        # Tokens to analyze
        tokens_to_analyze = [token_in_address, token_out_address]
        if token_mid_address:
            tokens_to_analyze.append(token_mid_address)
            
        # Prepare token data
        token_data = {}
        eth_price_usd = self._get_eth_price_usd()
        enriched['eth_price_usd'] = eth_price_usd
        
        # DEX names
        dex1 = opportunity.get('dex_1', '').lower()
        dex2 = opportunity.get('dex_2', '').lower()
        dex3 = opportunity.get('dex_3', '').lower()
        
        dexes = [dex1, dex2]
        if dex3:
            dexes.append(dex3)
        
        # Get network congestion
        gas_price = self._get_gas_price_gwei()
        enriched['gas_price_gwei'] = gas_price
        
        # Analysis for each token
        for token_address in tokens_to_analyze:
            if not token_address or token_address == '0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee':
                # ETH itself
                token_data[token_address] = {
                    'price_usd': eth_price_usd,
                    'liquidity': 1000000000.0,  # ETH has high liquidity
                    'volatility': 0.02  # ETH volatility (placeholder, can be updated)
                }
                continue
                
            # Get token price
            token_price = self._get_token_price(token_address)
            
            # Get token liquidity
            token_liquidity = {}
            for dex in dexes:
                liq = self._get_token_liquidity_in_dex(token_address, dex)
                if liq > 0:
                    token_liquidity[dex] = liq
            
            # Calculate volatility
            volatility = self._get_token_volatility(token_address)
            
            # Store token data
            token_data[token_address] = {
                'price_usd': token_price,
                'liquidity': sum(token_liquidity.values()) if token_liquidity else 0,
                'liquidity_by_dex': token_liquidity,
                'volatility': volatility
            }
        
        # Get market state
        enriched['market_state'] = {
            'eth_price_change_24h': self._get_eth_price_change_24h(),
            'market_volatility': self._get_market_volatility(),
            'gas_price_trend': self._get_gas_price_trend(),
            'network_congestion': self._get_network_congestion()
        }
        
        # Add token data to opportunity
        enriched['token_data'] = token_data
        
        # Add historical performance for similar opportunities
        if self.db:
            try:
                from flask import current_app
                with current_app.app_context():
                    # Find similar arbitrage opportunities
                    similar_opps = ArbitrageOpportunity.query.filter(
                        ArbitrageOpportunity.token_in == token_in_address,
                        ArbitrageOpportunity.token_out == token_out_address,
                        ArbitrageOpportunity.executed == True
                    ).limit(10).all()
                    
                    if similar_opps:
                        # Calculate success rate
                        success_count = sum(1 for opp in similar_opps if opp.transaction and opp.transaction.profit_eth and opp.transaction.profit_eth > 0)
                        success_rate = success_count / len(similar_opps)
                        
                        # Calculate average profit
                        avg_profit = sum(opp.transaction.profit_eth or 0 for opp in similar_opps if opp.transaction) / len(similar_opps)
                        
                        enriched['historical_performance'] = {
                            'success_rate': success_rate,
                            'avg_profit': avg_profit,
                            'similar_opportunities_count': len(similar_opps)
                        }
            except Exception as e:
                logger.error(f"Error getting historical performance: {e}")
        
        return enriched
    
    def _prepare_features_for_lstm(self, opportunity: Dict) -> np.ndarray:
        """
        Prepare features for LSTM model
        
        Args:
            opportunity: Enriched opportunity dictionary
            
        Returns:
            Numpy array with features
        """
        # Get data processor
        data_processor = get_data_processor()
        
        # Prepare features
        features = data_processor.prepare_arbitrage_features(opportunity)
        
        # Reshape for LSTM input (batch_size, sequence_length, features)
        # For prediction, we use a batch size of 1 and sequence length of 1
        features_reshaped = features.reshape(1, 1, -1)
        
        return features_reshaped
    
    def _prepare_features_for_transformer(self, opportunity: Dict) -> np.ndarray:
        """
        Prepare features for Transformer model
        
        Args:
            opportunity: Enriched opportunity dictionary
            
        Returns:
            Numpy array with features
        """
        # Use same feature preparation as LSTM
        return self._prepare_features_for_lstm(opportunity)
    
    def _prediction_to_profit_adjustment(self, prediction: float) -> float:
        """
        Convert model prediction to profit adjustment factor
        
        Args:
            prediction: Raw model prediction
            
        Returns:
            Profit adjustment factor (multiply expected profit by this)
        """
        # Prediction is typically a price change percentage
        # Convert to a profit adjustment factor
        
        # For positive predictions, increase expected profit
        if prediction > 0.1:
            return 1.1 + min(prediction, 0.5)
        # For very small predictions, slightly reduce profit
        elif prediction > -0.05:
            return 1.0
        # For negative predictions, reduce expected profit
        else:
            return max(0.5, 1.0 + prediction)
    
    def _calculate_confidence(self, prediction: float, opportunity: Dict) -> float:
        """
        Calculate confidence score for prediction
        
        Args:
            prediction: Model prediction
            opportunity: Opportunity dictionary
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence
        base_confidence = 0.7
        
        # Adjust based on prediction magnitude
        prediction_magnitude = abs(prediction)
        if prediction_magnitude > 0.2:
            magnitude_adjustment = 0.1
        elif prediction_magnitude > 0.1:
            magnitude_adjustment = 0.05
        else:
            magnitude_adjustment = 0
            
        # Adjust based on token liquidity
        token_in = opportunity.get('token_in', '')
        token_out = opportunity.get('token_out', '')
        token_data = opportunity.get('token_data', {})
        
        liquidity_adjustment = 0
        if token_data:
            # Check both tokens have good liquidity
            in_liquidity = token_data.get(token_in, {}).get('liquidity', 0)
            out_liquidity = token_data.get(token_out, {}).get('liquidity', 0)
            
            if in_liquidity > 1000000 and out_liquidity > 1000000:
                liquidity_adjustment = 0.1
            elif in_liquidity > 500000 and out_liquidity > 500000:
                liquidity_adjustment = 0.05
            elif in_liquidity < 50000 or out_liquidity < 50000:
                liquidity_adjustment = -0.1
                
        # Adjust based on volatility
        volatility_adjustment = 0
        if token_data:
            in_volatility = token_data.get(token_in, {}).get('volatility', 0)
            out_volatility = token_data.get(token_out, {}).get('volatility', 0)
            
            avg_volatility = (in_volatility + out_volatility) / 2
            if avg_volatility > 0.1:
                volatility_adjustment = -0.1
            elif avg_volatility > 0.05:
                volatility_adjustment = -0.05
                
        # Adjust based on historical performance
        historical_adjustment = 0
        if 'historical_performance' in opportunity:
            historical = opportunity['historical_performance']
            success_rate = historical.get('success_rate', 0)
            
            if success_rate > 0.8:
                historical_adjustment = 0.1
            elif success_rate > 0.6:
                historical_adjustment = 0.05
            elif success_rate < 0.4:
                historical_adjustment = -0.1
        
        # Calculate final confidence
        confidence = base_confidence + magnitude_adjustment + liquidity_adjustment + volatility_adjustment + historical_adjustment
        
        # Ensure confidence is between 0 and 1
        return max(0.1, min(0.95, confidence))
    
    def _update_market_state(self) -> None:
        """Update market state based on current conditions"""
        try:
            eth_price_change = self._get_eth_price_change_24h()
            market_volatility = self._get_market_volatility()
            
            # Determine market state
            if market_volatility > 0.05:
                self.market_state = 'volatile'
            elif eth_price_change > 0.05:
                self.market_state = 'trending_up'
            elif eth_price_change < -0.05:
                self.market_state = 'trending_down'
            else:
                self.market_state = 'normal'
                
            logger.info(f"Updated market state: {self.market_state}")
            
            # Update timestamp
            self.last_market_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating market state: {e}")
    
    def _adjust_for_market_state(self, profit_adjustment: float) -> float:
        """
        Adjust profit based on market state
        
        Args:
            profit_adjustment: Current profit adjustment
            
        Returns:
            Adjusted profit factor
        """
        if self.market_state == 'volatile':
            # More cautious in volatile markets
            return profit_adjustment * 0.9
        elif self.market_state == 'trending_up':
            # More aggressive in bull markets
            return profit_adjustment * 1.1
        elif self.market_state == 'trending_down':
            # More cautious in bear markets
            return profit_adjustment * 0.95
        else:
            # Normal market
            return profit_adjustment
    
    def _get_token_price(self, token_address: str) -> float:
        """
        Get token price from API with caching
        
        Args:
            token_address: Token address
            
        Returns:
            Token price in USD
        """
        # Check cache first
        now = time.time()
        if token_address in self.token_price_cache:
            last_updated = self.token_price_last_updated.get(token_address, 0)
            if now - last_updated < self.token_price_cache_expiry:
                return self.token_price_cache[token_address]
        
        try:
            # First try CoinGecko API
            try:
                price = self._get_token_price_coingecko(token_address)
                if price > 0:
                    self.token_price_cache[token_address] = price
                    self.token_price_last_updated[token_address] = now
                    return price
            except Exception as e:
                logger.warning(f"CoinGecko price error for {token_address}: {e}")
            
            # Then try 1Inch API
            try:
                price = self._get_token_price_1inch(token_address)
                if price > 0:
                    self.token_price_cache[token_address] = price
                    self.token_price_last_updated[token_address] = now
                    return price
            except Exception as e:
                logger.warning(f"1Inch price error for {token_address}: {e}")
            
            # Then try Uniswap price
            try:
                price = self._get_token_price_uniswap(token_address)
                if price > 0:
                    self.token_price_cache[token_address] = price
                    self.token_price_last_updated[token_address] = now
                    return price
            except Exception as e:
                logger.warning(f"Uniswap price error for {token_address}: {e}")
            
            # Fallback to a default price
            logger.warning(f"Could not get price for {token_address}, using fallback")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting token price for {token_address}: {e}")
            return 0.0
    
    def _get_token_price_coingecko(self, token_address: str) -> float:
        """Get token price from CoinGecko API"""
        url = f"https://api.coingecko.com/api/v3/simple/token_price/ethereum"
        params = {
            'contract_addresses': token_address,
            'vs_currencies': 'usd'
        }
        
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if token_address in data and 'usd' in data[token_address]:
                return float(data[token_address]['usd'])
        
        raise PredictorException(f"CoinGecko API failed: {response.status_code}")
    
    def _get_token_price_1inch(self, token_address: str) -> float:
        """Get token price from 1Inch API"""
        # USDC as quote currency
        usdc_address = '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48'
        
        url = "https://api.1inch.io/v5.0/1/quote"
        params = {
            'fromTokenAddress': token_address,
            'toTokenAddress': usdc_address,
            'amount': '1000000000000000000'  # 1 token in wei
        }
        
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            if 'toAmount' in data:
                # Convert USDC amount (6 decimals) to USD
                usdc_amount = int(data['toAmount']) / 1e6
                return usdc_amount
        
        raise PredictorException(f"1Inch API failed: {response.status_code}")
    
    def _get_token_price_uniswap(self, token_address: str) -> float:
        """Get token price from Uniswap"""
        if not self.web3 or not self.web3.is_connected():
            raise PredictorException("Web3 not connected")
            
        try:
            # WETH address
            weth_address = '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'
            
            # Uniswap V2 Router contract
            uniswap_router_address = '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D'
            
            # ABI for the router's getAmountsOut function
            abi = [{
                "inputs": [
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"}
                ],
                "name": "getAmountsOut",
                "outputs": [
                    {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}
                ],
                "stateMutability": "view",
                "type": "function"
            }]
            
            # Create contract
            contract = self.web3.eth.contract(address=uniswap_router_address, abi=abi)
            
            # Get price to ETH
            amount_in = 10**18  # 1 token
            path = [self.web3.to_checksum_address(token_address), self.web3.to_checksum_address(weth_address)]
            
            try:
                amounts = contract.functions.getAmountsOut(amount_in, path).call()
                eth_amount = amounts[1] / 10**18
                
                # Convert ETH to USD
                eth_price_usd = self._get_eth_price_usd()
                return eth_amount * eth_price_usd
            except Exception as e:
                logger.warning(f"Direct path failed, trying reverse path: {e}")
                
                # Try reverse path (ETH to token)
                amount_in = 10**18  # 1 ETH
                path = [self.web3.to_checksum_address(weth_address), self.web3.to_checksum_address(token_address)]
                
                amounts = contract.functions.getAmountsOut(amount_in, path).call()
                token_amount = amounts[1] / 10**18
                
                # Token price = 1 ETH / token_amount
                eth_price_usd = self._get_eth_price_usd()
                if token_amount > 0:
                    return eth_price_usd / token_amount
                else:
                    raise PredictorException(f"Zero token amount returned")
                
        except Exception as e:
            raise PredictorException(f"Uniswap price calculation failed: {e}")
    
    def _get_token_liquidity_in_dex(self, token_address: str, dex: str) -> float:
        """
        Get token liquidity in a specific DEX
        
        Args:
            token_address: Token address
            dex: DEX name (uniswap, sushiswap, etc.)
            
        Returns:
            Liquidity in USD
        """
        # Cache key
        cache_key = f"{token_address}_{dex}"
        
        # Check cache first
        now = time.time()
        if cache_key in self.dex_liquidity_cache:
            last_updated = self.dex_liquidity_last_updated.get(cache_key, 0)
            if now - last_updated < self.dex_liquidity_cache_expiry:
                return self.dex_liquidity_cache[cache_key]
        
        try:
            liquidity = 0.0
            
            if 'uniswap' in dex.lower():
                liquidity = self._get_uniswap_liquidity(token_address)
            elif 'sushiswap' in dex.lower():
                liquidity = self._get_sushiswap_liquidity(token_address)
            elif 'curve' in dex.lower():
                liquidity = self._get_curve_liquidity(token_address)
            elif 'balancer' in dex.lower():
                liquidity = self._get_balancer_liquidity(token_address)
            else:
                # Default to Uniswap if DEX not recognized
                liquidity = self._get_uniswap_liquidity(token_address)
            
            # Cache result
            self.dex_liquidity_cache[cache_key] = liquidity
            self.dex_liquidity_last_updated[cache_key] = now
            
            return liquidity
            
        except Exception as e:
            logger.error(f"Error getting liquidity for {token_address} in {dex}: {e}")
            return 0.0
    
    def _get_uniswap_liquidity(self, token_address: str) -> float:
        """Get token liquidity in Uniswap"""
        try:
            # Use Uniswap GraphQL API
            url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"
            query = """
            {
              token(id: "%s") {
                totalLiquidity
                derivedETH
              }
            }
            """ % token_address.lower()
            
            response = requests.post(url, json={'query': query}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and 'data' in data and 'token' in data['data'] and data['data']['token']:
                    token_data = data['data']['token']
                    liquidity = float(token_data['totalLiquidity'])
                    eth_price = float(token_data['derivedETH'])
                    
                    # Get ETH price in USD
                    eth_price_usd = self._get_eth_price_usd()
                    
                    # Calculate liquidity in USD
                    return liquidity * eth_price * eth_price_usd
            
            # Fallback to a direct contract call if GraphQL API fails
            return self._get_uniswap_liquidity_direct(token_address)
            
        except Exception as e:
            logger.error(f"Error getting Uniswap liquidity for {token_address}: {e}")
            return self._get_uniswap_liquidity_direct(token_address)
    
    def _get_uniswap_liquidity_direct(self, token_address: str) -> float:
        """Get Uniswap liquidity using direct contract calls"""
        if not self.web3 or not self.web3.is_connected():
            return 0.0
            
        try:
            # WETH address
            weth_address = '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'
            
            # Uniswap V2 Factory contract
            factory_address = '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f'
            factory_abi = [{
                "inputs": [
                    {"internalType": "address", "name": "tokenA", "type": "address"},
                    {"internalType": "address", "name": "tokenB", "type": "address"}
                ],
                "name": "getPair",
                "outputs": [{"internalType": "address", "name": "pair", "type": "address"}],
                "stateMutability": "view",
                "type": "function"
            }]
            
            # Pair contract ABI (for getReserves)
            pair_abi = [{
                "inputs": [],
                "name": "getReserves",
                "outputs": [
                    {"internalType": "uint112", "name": "_reserve0", "type": "uint112"},
                    {"internalType": "uint112", "name": "_reserve1", "type": "uint112"},
                    {"internalType": "uint32", "name": "_blockTimestampLast", "type": "uint32"}
                ],
                "stateMutability": "view",
                "type": "function"
            }]
            
            # Create factory contract
            factory = self.web3.eth.contract(address=factory_address, abi=factory_abi)
            
            # Get pair address
            pair_address = factory.functions.getPair(
                self.web3.to_checksum_address(token_address),
                self.web3.to_checksum_address(weth_address)
            ).call()
            
            if pair_address == '0x0000000000000000000000000000000000000000':
                return 0.0
            
            # Create pair contract
            pair = self.web3.eth.contract(address=pair_address, abi=pair_abi)
            
            # Get reserves
            reserves = pair.functions.getReserves().call()
            
            # Determine which reserve is which
            token_address_checksum = self.web3.to_checksum_address(token_address)
            weth_address_checksum = self.web3.to_checksum_address(weth_address)
            
            if token_address_checksum.lower() < weth_address_checksum.lower():
                token_reserve = reserves[0]
                eth_reserve = reserves[1]
            else:
                token_reserve = reserves[1]
                eth_reserve = reserves[0]
            
            # Get token decimals (assume 18 if not available)
            token_decimals = 18
            try:
                erc20_abi = [{
                    "inputs": [],
                    "name": "decimals",
                    "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
                    "stateMutability": "view",
                    "type": "function"
                }]
                token_contract = self.web3.eth.contract(address=token_address_checksum, abi=erc20_abi)
                token_decimals = token_contract.functions.decimals().call()
            except Exception as e:
                logger.warning(f"Could not get decimals for {token_address}, using default: {e}")
            
            # Calculate token price in ETH
            if token_reserve > 0:
                token_price_eth = eth_reserve / (token_reserve / (10**(token_decimals - 18)))
            else:
                token_price_eth = 0
            
            # Get ETH price in USD
            eth_price_usd = self._get_eth_price_usd()
            
            # Calculate liquidity in USD
            eth_liquidity_usd = (eth_reserve / 10**18) * eth_price_usd
            token_liquidity_usd = (token_reserve / 10**token_decimals) * token_price_eth * eth_price_usd
            
            return eth_liquidity_usd + token_liquidity_usd
            
        except Exception as e:
            logger.error(f"Error getting Uniswap liquidity direct for {token_address}: {e}")
            return 0.0
    
    def _get_sushiswap_liquidity(self, token_address: str) -> float:
        """Get token liquidity in SushiSwap"""
        try:
            # SushiSwap GraphQL API
            url = "https://api.thegraph.com/subgraphs/name/sushiswap/exchange"
            query = """
            {
              token(id: "%s") {
                totalLiquidity
                derivedETH
              }
            }
            """ % token_address.lower()
            
            response = requests.post(url, json={'query': query}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and 'data' in data and 'token' in data['data'] and data['data']['token']:
                    token_data = data['data']['token']
                    liquidity = float(token_data['totalLiquidity'])
                    eth_price = float(token_data['derivedETH'])
                    
                    # Get ETH price in USD
                    eth_price_usd = self._get_eth_price_usd()
                    
                    # Calculate liquidity in USD
                    return liquidity * eth_price * eth_price_usd
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting SushiSwap liquidity for {token_address}: {e}")
            return 0.0
    
    def _get_curve_liquidity(self, token_address: str) -> float:
        """Get token liquidity in Curve"""
        # Curve is complex with different pools
        # Simplified implementation for common stablecoins
        stablecoins = {
            '0x6b175474e89094c44da98b954eedeac495271d0f': 'dai',
            '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48': 'usdc',
            '0xdac17f958d2ee523a2206206994597c13d831ec7': 'usdt',
            '0x0000000000085d4780b73119b644ae5ecd22b376': 'tusd',
            '0x57ab1ec28d129707052df4df418d58a2d46d5f51': 'susd'
        }
        
        if token_address.lower() in stablecoins:
            # Use CoinGecko for market data
            coin_id = stablecoins[token_address.lower()]
            try:
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    market_data = data.get('market_data', {})
                    total_volume = market_data.get('total_volume', {}).get('usd', 0)
                    
                    # Estimate liquidity as a fraction of daily volume
                    # This is a very rough approximation
                    return total_volume * 0.1  # Assume 10% of daily volume is in Curve
                    
            except Exception as e:
                logger.error(f"Error getting Curve liquidity for {token_address}: {e}")
        
        # For non-stablecoins, return 0 as they're less likely to be in Curve
        return 0.0
    
    def _get_balancer_liquidity(self, token_address: str) -> float:
        """Get token liquidity in Balancer"""
        try:
            # Balancer GraphQL API
            url = "https://api.thegraph.com/subgraphs/name/balancer-labs/balancer-v2"
            query = """
            {
              tokenGetPoolData(address: "%s") {
                pools {
                  totalLiquidity
                }
              }
            }
            """ % token_address.lower()
            
            response = requests.post(url, json={'query': query}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and 'data' in data and 'tokenGetPoolData' in data['data'] and data['data']['tokenGetPoolData']:
                    pools = data['data']['tokenGetPoolData'].get('pools', [])
                    total_liquidity = sum(float(pool.get('totalLiquidity', 0)) for pool in pools)
                    return total_liquidity
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting Balancer liquidity for {token_address}: {e}")
            return 0.0
    
    def _get_token_volatility(self, token_address: str) -> float:
        """
        Calculate token price volatility
        
        Args:
            token_address: Token address
            
        Returns:
            Volatility (standard deviation of daily returns)
        """
        try:
            # Use CoinGecko API for historical prices
            # Extract coin ID from token address (simplified)
            url = f"https://api.coingecko.com/api/v3/coins/ethereum/contract/{token_address.lower()}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': 7
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                prices = data.get('prices', [])
                
                if len(prices) > 1:
                    # Calculate daily returns
                    price_series = [price[1] for price in prices]
                    returns = np.diff(price_series) / price_series[:-1]
                    
                    # Calculate volatility (standard deviation of returns)
                    volatility = np.std(returns)
                    return float(volatility)
            
            # Fallback to default volatility values
            # Higher for tokens, lower for stablecoins
            if token_address.lower() in [
                '0x6b175474e89094c44da98b954eedeac495271d0f',  # DAI
                '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',  # USDC
                '0xdac17f958d2ee523a2206206994597c13d831ec7'   # USDT
            ]:
                return 0.005  # Stablecoins have low volatility
            else:
                return 0.03  # Default volatility for other tokens
                
        except Exception as e:
            logger.error(f"Error calculating volatility for {token_address}: {e}")
            return 0.03  # Default fallback volatility
    
    def _get_eth_price_usd(self) -> float:
        """Get current ETH price in USD"""
        try:
            # Check if price was fetched recently
            now = time.time()
            if 'eth_price' in self.token_price_cache:
                last_updated = self.token_price_last_updated.get('eth_price', 0)
                if now - last_updated < self.token_price_cache_expiry:
                    return self.token_price_cache['eth_price']
            
            # Try CoinGecko API
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': 'ethereum',
                'vs_currencies': 'usd'
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'ethereum' in data and 'usd' in data['ethereum']:
                    price = float(data['ethereum']['usd'])
                    self.token_price_cache['eth_price'] = price
                    self.token_price_last_updated['eth_price'] = now
                    return price
            
            # Try alternative API
            url = "https://api.coinbase.com/v2/prices/ETH-USD/spot"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'amount' in data['data']:
                    price = float(data['data']['amount'])
                    self.token_price_cache['eth_price'] = price
                    self.token_price_last_updated['eth_price'] = now
                    return price
            
            # If both APIs fail, use a fallback value
            # or try a blockchain oracle
            if self.web3 and self.web3.is_connected():
                try:
                    # Chainlink ETH/USD price feed
                    chainlink_eth_usd = '0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419'
                    abi = [{
                        "inputs": [],
                        "name": "latestAnswer",
                        "outputs": [{"internalType": "int256", "name": "", "type": "int256"}],
                        "stateMutability": "view",
                        "type": "function"
                    }]
                    
                    contract = self.web3.eth.contract(address=chainlink_eth_usd, abi=abi)
                    price_wei = contract.functions.latestAnswer().call()
                    price = price_wei / 10**8  # Chainlink uses 8 decimals
                    
                    self.token_price_cache['eth_price'] = price
                    self.token_price_last_updated['eth_price'] = now
                    return price
                except Exception as e:
                    logger.error(f"Error getting ETH price from Chainlink: {e}")
            
            # Last resort fallback
            return 3000.0
            
        except Exception as e:
            logger.error(f"Error getting ETH price: {e}")
            return 3000.0
    
    def _get_eth_price_change_24h(self) -> float:
        """Get ETH price change in the last 24 hours"""
        try:
            # Try CoinGecko API
            url = "https://api.coingecko.com/api/v3/coins/ethereum"
            params = {'localization': 'false', 'tickers': 'false', 'community_data': 'false', 'developer_data': 'false'}
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'market_data' in data and 'price_change_percentage_24h' in data['market_data']:
                    return float(data['market_data']['price_change_percentage_24h']) / 100
            
            # Fallback: Calculate manually
            if self.web3 and self.web3.is_connected():
                # Get current price
                current_price = self._get_eth_price_usd()
                
                # Get average price of last 100 blocks
                current_block = self.web3.eth.block_number
                day_ago_block = current_block - 6500  # ~6500 blocks per day
                
                # Chainlink ETH/USD price feed
                chainlink_eth_usd = '0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419'
                abi = [{
                    "inputs": [],
                    "name": "latestAnswer",
                    "outputs": [{"internalType": "int256", "name": "", "type": "int256"}],
                    "stateMutability": "view",
                    "type": "function"
                }]
                
                # Try to get price from a day ago using archive node
                try:
                    contract = self.web3.eth.contract(address=chainlink_eth_usd, abi=abi)
                    price_day_ago_wei = contract.functions.latestAnswer().call(block_identifier=day_ago_block)
                    price_day_ago = price_day_ago_wei / 10**8
                    
                    if price_day_ago > 0:
                        return (current_price - price_day_ago) / price_day_ago
                except Exception as e:
                    logger.warning(f"Error getting historical ETH price: {e}")
            
            # Default to a small positive change
            return 0.005
            
        except Exception as e:
            logger.error(f"Error calculating ETH price change: {e}")
            return 0.005
    
    def _get_market_volatility(self) -> float:
        """Get overall market volatility"""
        try:
            # Try CoinGecko API for ETH volatility
            url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
            params = {'vs_currency': 'usd', 'days': 7}
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                prices = data.get('prices', [])
                
                if len(prices) > 1:
                    # Calculate hourly returns
                    price_series = [price[1] for price in prices]
                    returns = np.diff(price_series) / price_series[:-1]
                    
                    # Calculate volatility (standard deviation of returns)
                    volatility = np.std(returns)
                    return float(volatility)
            
            # Default volatility
            return 0.02
            
        except Exception as e:
            logger.error(f"Error calculating market volatility: {e}")
            return 0.02
    
    def _get_gas_price_gwei(self) -> float:
        """Get current gas price in Gwei"""
        try:
            if self.web3 and self.web3.is_connected():
                gas_price_wei = self.web3.eth.gas_price
                gas_price_gwei = gas_price_wei / 10**9
                return float(gas_price_gwei)
            
            # Fallback to Etherscan API
            api_key = os.environ.get('ETHERSCAN_API_KEY', '')
            url = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={api_key}"
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == '1' and 'result' in data:
                    return float(data['result']['ProposeGasPrice'])
            
            # Default value
            return 50.0
            
        except Exception as e:
            logger.error(f"Error getting gas price: {e}")
            return 50.0
    
    def _get_gas_price_trend(self) -> float:
        """Get gas price trend (-1 to 1)"""
        try:
            if not self.web3 or not self.web3.is_connected():
                return 0.0
                
            # Get current gas price
            current_gas_price = self.web3.eth.gas_price
            
            # Get average gas price of last 10 blocks
            current_block = self.web3.eth.block_number
            gas_prices = []
            
            for block_num in range(current_block - 10, current_block):
                try:
                    block = self.web3.eth.get_block(block_num)
                    if 'baseFeePerGas' in block:
                        gas_prices.append(block['baseFeePerGas'])
                except Exception:
                    continue
            
            if not gas_prices:
                return 0.0
                
            avg_gas_price = sum(gas_prices) / len(gas_prices)
            
            # Calculate trend (-1 to 1)
            trend = (current_gas_price - avg_gas_price) / avg_gas_price
            
            # Limit to range -1 to 1
            return max(-1.0, min(1.0, float(trend)))
            
        except Exception as e:
            logger.error(f"Error calculating gas price trend: {e}")
            return 0.0
    
    def _get_network_congestion(self) -> float:
        """Get network congestion level (0-1)"""
        try:
            if not self.web3 or not self.web3.is_connected():
                return 0.5
                
            # Get current gas price
            current_gas_price = self.web3.eth.gas_price / 10**9  # Convert to Gwei
            
            # Calculate congestion based on gas price thresholds
            if current_gas_price < 30:
                congestion = 0.2  # Low congestion
            elif current_gas_price < 60:
                congestion = 0.4  # Moderate congestion
            elif current_gas_price < 100:
                congestion = 0.6  # High congestion
            elif current_gas_price < 200:
                congestion = 0.8  # Very high congestion
            else:
                congestion = 0.95  # Extreme congestion
                
            return congestion
            
        except Exception as e:
            logger.error(f"Error calculating network congestion: {e}")
            return 0.5

# Singleton instance
_predictor = None

def get_arbitrage_predictor(db=None) -> ArbitragePredictor:
    """
    Get singleton arbitrage predictor instance
    
    Args:
        db: Database connection
        
    Returns:
        ArbitragePredictor instance
    """
    global _predictor
    if _predictor is None:
        _predictor = ArbitragePredictor(db)
    return _predictor
