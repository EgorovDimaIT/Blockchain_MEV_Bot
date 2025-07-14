"""
Script to find arbitrage opportunities using trained ML models
"""

import os
import logging
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Any, Optional, Tuple
from model_predictor import ModelPredictor, predict_all_targets, detect_arbitrage_opportunity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_latest_data(symbol: str, exchange: str = 'binance', timeframe: str = '1h') -> pd.DataFrame:
    """
    Load latest data for a symbol
    
    Args:
        symbol: Symbol to load data for
        exchange: Exchange name
        timeframe: Timeframe to load
        
    Returns:
        DataFrame with latest data
    """
    try:
        # First check if we have a file in attached_assets (raw data format)
        raw_file_path = f'attached_assets/{symbol}.csv'
        if os.path.exists(raw_file_path):
            logger.info(f"Loading data from raw file: {raw_file_path}")
            df = pd.read_csv(raw_file_path)
            
            # If we have a raw data file with 3 columns (timestamp, price, volume)
            if len(df.columns) == 3 and 'datetime' not in df.columns:
                # Rename columns
                df.columns = ['timestamp', 'close', 'volume']
                
                # Convert timestamp to datetime
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Add high, low, open (using close as approximation for missing values)
                df['high'] = df['close']
                df['low'] = df['close']
                df['open'] = df['close']
                
                # Sort by datetime
                df = df.sort_values('datetime')
                
                logger.info(f"Loaded {len(df)} records from raw file {raw_file_path}")
                return df
        
        # If no raw file or raw file has different format, try to load processed data
        file_path = f'data/{exchange}_{symbol}_{timeframe}.csv'
        if not os.path.exists(file_path):
            logger.error(f"Data file not found: {file_path}")
            return pd.DataFrame()
            
        df = pd.read_csv(file_path)
        
        # Check if we have a different format (old format with named columns)
        if 'datetime' in df.columns:
            # Convert datetime
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            logger.warning(f"Unexpected column format in {file_path}, attempting to adapt")
            # Try to adapt to other formats
            if len(df.columns) >= 3:
                # Assume first column is timestamp, second is price, third is volume
                df.columns = ['timestamp', 'close', 'volume'] + [f'col_{i}' for i in range(3, len(df.columns))]
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['high'] = df['close']
                df['low'] = df['close']
                df['open'] = df['close']
        
        # Sort by datetime
        df = df.sort_values('datetime')
        
        logger.info(f"Loaded {len(df)} records from {file_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return pd.DataFrame()

def prepare_features_for_prediction(df: pd.DataFrame, sequence_length: int = 24) -> np.ndarray:
    """
    Prepare features for prediction
    
    Args:
        df: DataFrame with data
        sequence_length: Length of input sequence
        
    Returns:
        Features array with shape (sequence_length, num_features)
    """
    try:
        # Add technical indicators
        df_with_indicators = df.copy()
        
        # Add timestamp
        df_with_indicators['timestamp'] = df_with_indicators['datetime'].astype(np.int64) // 10**9
        
        # Add moving averages
        for period in [5, 10, 20, 50, 100]:
            # Simple Moving Average
            df_with_indicators[f'sma_{period}'] = df_with_indicators['close'].rolling(window=period).mean()
            
            # Exponential Moving Average
            df_with_indicators[f'ema_{period}'] = df_with_indicators['close'].ewm(span=period, adjust=False).mean()
        
        # Add Bollinger Bands
        for period in [20]:
            # Calculate middle band (SMA)
            middle_band = df_with_indicators['close'].rolling(window=period).mean()
            
            # Calculate standard deviation
            std_dev = df_with_indicators['close'].rolling(window=period).std()
            
            # Upper and lower bands
            df_with_indicators[f'bb_upper'] = middle_band + 2 * std_dev
            df_with_indicators[f'bb_lower'] = middle_band - 2 * std_dev
        
        # Add RSI
        for period in [14]:
            # Calculate price changes
            delta = df_with_indicators['close'].diff()
            
            # Calculate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            df_with_indicators[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Add MACD
        fast_period = 12
        slow_period = 26
        signal_period = 9
        
        # Calculate EMAs
        ema_fast = df_with_indicators['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df_with_indicators['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        df_with_indicators['macd'] = ema_fast - ema_slow
        
        # Calculate signal line
        df_with_indicators['macd_signal'] = df_with_indicators['macd'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate MACD histogram
        df_with_indicators['macd_hist'] = df_with_indicators['macd'] - df_with_indicators['macd_signal']
        
        # Add high-low ratio
        df_with_indicators['high_low_ratio'] = df_with_indicators['high'] / df_with_indicators['low']
        
        # Add price changes and returns
        for period in [1, 3, 6, 12, 24]:
            # Price change
            df_with_indicators[f'price_change_{period}'] = df_with_indicators['close'].diff(periods=period)
        
        # Calculate volume indicators
        # Volume SMA
        for period in [5, 10, 20]:
            df_with_indicators[f'volume_sma_{period}'] = df_with_indicators['volume'].rolling(window=period).mean()
        
        # Volume change
        df_with_indicators['volume_change'] = df_with_indicators['volume'].pct_change()
        
        # Drop rows with NaN values
        df_with_indicators = df_with_indicators.dropna()
        
        # Get the last sequence_length rows
        if len(df_with_indicators) < sequence_length:
            logger.warning(f"Not enough data for prediction. Need {sequence_length} rows, but only have {len(df_with_indicators)}.")
            return np.array([])
            
        features_df = df_with_indicators.iloc[-sequence_length:]
        
        # Create feature list (all except datetime and target columns)
        feature_cols = [col for col in features_df.columns if col not in ['datetime', 'return_1h', 'return_3h', 'return_6h', 'return_12h', 'return_24h']]
        
        # Convert to numpy array
        features = features_df[feature_cols].values
        
        return features
        
    except Exception as e:
        logger.error(f"Error preparing features for prediction: {e}")
        return np.array([])

def analyze_triangular_arbitrage(predictions: Dict[str, Dict[str, Any]], base_currency: str = 'ETH') -> List[Dict[str, Any]]:
    """
    Analyze triangular arbitrage opportunities
    
    Args:
        predictions: Dictionary of predictions by symbol
        base_currency: Base currency for triangular arbitrage
        
    Returns:
        List of triangular arbitrage opportunities
    """
    opportunities = []
    
    # Dictionary to hold predicted price movements
    price_movements = {}
    
    # Extract price movement predictions for each pair
    for symbol, prediction_data in predictions.items():
        if 'predictions' not in prediction_data:
            continue
            
        if 'return_1h' not in prediction_data['predictions']:
            continue
            
        price_movements[symbol] = prediction_data['predictions']['return_1h']['value']
    
    # Define pairs for triangular arbitrage with ETH as base
    triangle_pairs = [
        # ETH -> TOKEN1 -> TOKEN2 -> ETH
        ['ETHUSDT', 'USDTGBP', 'ETHUSDC'],  # Example: ETH -> USDT -> GBP -> ETH
        ['ETHWETH', 'UNIETH', 'ETHUSDT'],   # Example: ETH -> WETH -> UNI -> ETH
        ['ETHWETH', 'AAVEETH', 'LINKETH'],  # Example: ETH -> WETH -> AAVE -> ETH
        ['UNIETH', 'AAVEETH', 'LINKETH'],   # Example: ETH -> UNI -> AAVE -> ETH
        
        # New triangular arbitrage pairs using our newly trained models
        ['ADAUSDC', 'VANRYUSD', 'KUJIEUR'],  # Example: ADA -> USDC -> VANRY -> KUJI -> EUR
        ['FILAUD', 'KUJIEUR', 'EURQUSD'],    # Example: FIL -> AUD -> KUJI -> EUR -> EURQ -> USD
        ['VANRYUSD', 'EURQUSD', 'ADAUSDC'],  # Example: VANRY -> USD -> EURQ -> USD -> ADA -> USDC
    ]
    
    # Check each triangle
    for triangle in triangle_pairs:
        # Skip if any pair is missing
        if not all(pair in price_movements for pair in triangle):
            continue
        
        # Calculate expected combined return
        combined_return = 1.0
        for pair in triangle:
            combined_return *= (1.0 + price_movements[pair])
        
        # Subtract 1 to get the percentage return
        combined_return -= 1.0
        
        # If the combined return exceeds the threshold (min profit $0.2), it's an opportunity
        if combined_return > 0.002 and combined_return * 100 >= 0.2:  # 0.2% threshold AND min $0.2 profit
            opportunity = {
                'type': 'triangular_arbitrage',
                'pairs': triangle,
                'predicted_return': combined_return,
                'confidence': min([predictions[pair]['predictions']['return_1h']['confidence'] for pair in triangle]),
                'timestamp': time.time()
            }
            opportunities.append(opportunity)
    
    return opportunities

def check_direct_arbitrage(predictions: Dict[str, Dict[str, Any]], threshold: float = 0.005) -> List[Dict[str, Any]]:
    """
    Check for direct arbitrage opportunities
    
    Args:
        predictions: Dictionary of predictions by symbol
        threshold: Threshold for considering an opportunity
        
    Returns:
        List of direct arbitrage opportunities
    """
    opportunities = []
    
    # For each symbol
    for symbol, prediction_data in predictions.items():
        # Get 1h prediction
        if 'predictions' not in prediction_data or 'return_1h' not in prediction_data['predictions']:
            continue
            
        pred_1h = prediction_data['predictions']['return_1h']
        
        # Check if prediction exceeds threshold (positive or negative) and min profit $0.2
        if abs(pred_1h['value']) > threshold and abs(pred_1h['value'] * 100) >= 0.2:
            # Create opportunity
            opportunity = {
                'type': 'direct_arbitrage',
                'symbol': symbol,
                'direction': 'long' if pred_1h['value'] > 0 else 'short',
                'predicted_return': pred_1h['value'],
                'confidence': pred_1h['confidence'],
                'time_frame': '1h',
                'threshold': threshold,
                'timestamp': time.time()
            }
            
            opportunities.append(opportunity)
    
    return opportunities

def main():
    """Find arbitrage opportunities using trained ML models"""
    logger.info("Starting arbitrage opportunity detection")
    
    # Define pairs to analyze
    pairs_to_analyze = [
        {'symbol': 'ETHUSDT', 'exchange': 'binance'},
        {'symbol': 'USDTGBP', 'exchange': 'binance'},
        {'symbol': 'ETHUSDC', 'exchange': 'binance'},
        {'symbol': 'ETHWETH', 'exchange': 'uniswap'},
        {'symbol': 'UNIETH', 'exchange': 'uniswap'},
        {'symbol': 'AAVEETH', 'exchange': 'uniswap'},
        {'symbol': 'LINKETH', 'exchange': 'uniswap'},
        # Add newly trained models
        {'symbol': 'ADAUSDC', 'exchange': 'binance'},
        {'symbol': 'KUJIEUR', 'exchange': 'binance'},
        {'symbol': 'FILAUD', 'exchange': 'binance'},
        {'symbol': 'VANRYUSD', 'exchange': 'binance'},
        {'symbol': 'EURQUSD', 'exchange': 'binance'}
    ]
    
    # Load predictor
    predictor = ModelPredictor()
    
    # Dictionary to store predictions
    all_predictions = {}
    
    # For each pair
    for pair_info in pairs_to_analyze:
        symbol = pair_info['symbol']
        exchange = pair_info['exchange']
        symbol_name = f"{exchange}_{symbol}"
        
        logger.info(f"Analyzing {symbol_name}")
        
        try:
            # Check if model is available
            if symbol_name not in predictor.get_available_symbols():
                logger.warning(f"No model available for {symbol_name}")
                continue
            
            # Load latest data
            df = load_latest_data(symbol, exchange)
            
            if len(df) == 0:
                logger.warning(f"No data available for {symbol_name}")
                continue
            
            # Prepare features
            features = prepare_features_for_prediction(df)
            
            if len(features) == 0:
                logger.warning(f"Failed to prepare features for {symbol_name}")
                continue
            
            # Make predictions
            predictions = predict_all_targets(features, symbol_name)
            
            if 'error' in predictions:
                logger.warning(f"Failed to make predictions for {symbol_name}: {predictions['error']}")
                continue
            
            all_predictions[symbol_name] = predictions
            
            logger.info(f"Predictions for {symbol_name}: {predictions['predictions']['return_1h']['value']:.6f} (1h)")
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol_name}: {e}")
    
    # Detect arbitrage opportunities
    logger.info("Detecting direct arbitrage opportunities")
    direct_opportunities = check_direct_arbitrage(all_predictions)
    
    logger.info("Detecting triangular arbitrage opportunities")
    triangular_opportunities = analyze_triangular_arbitrage(all_predictions)
    
    # Combined opportunities
    all_opportunities = {
        'direct': direct_opportunities,
        'triangular': triangular_opportunities,
        'timestamp': time.time()
    }
    
    # Print results
    logger.info(f"Found {len(direct_opportunities)} direct arbitrage opportunities")
    for opp in direct_opportunities:
        logger.info(f"  {opp['symbol']}: {opp['direction']} with predicted return {opp['predicted_return']:.6f} and confidence {opp['confidence']:.6f}")
    
    logger.info(f"Found {len(triangular_opportunities)} triangular arbitrage opportunities")
    for opp in triangular_opportunities:
        logger.info(f"  {' -> '.join(opp['pairs'])}: predicted return {opp['predicted_return']:.6f} and confidence {opp['confidence']:.6f}")
    
    # Save results
    os.makedirs('opportunities', exist_ok=True)
    file_name = f"opportunities_{int(time.time())}.json"
    file_path = os.path.join('opportunities', file_name)
    
    with open(file_path, 'w') as f:
        json.dump(all_opportunities, f, indent=2)
    
    logger.info(f"Saved opportunities to {file_path}")

if __name__ == "__main__":
    main()