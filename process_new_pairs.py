"""
Script to process newly provided cryptocurrency pairs data and create ML datasets
"""

import os
import logging
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        # Load data with appropriate column names
        df = pd.read_csv(file_path, header=None, names=['timestamp', 'price', 'volume'])
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Sort by datetime
        df = df.sort_values('datetime')
        
        logger.info(f"Loaded {len(df)} records from {file_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

def create_timeframes(df: pd.DataFrame, pair_name: str, exchange: str = 'binance') -> Dict[str, pd.DataFrame]:
    """
    Create timeframe DataFrames from raw data
    
    Args:
        df: DataFrame with raw data
        pair_name: Name of trading pair
        exchange: Name of exchange
        
    Returns:
        Dictionary with timeframe DataFrames
    """
    timeframes = {}
    
    try:
        # Create directory for data if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Create OHLCV for different timeframes
        # 1h timeframe
        df_1h = df.set_index('datetime').price.resample('1H').ohlc()
        df_1h['volume'] = df.set_index('datetime').volume.resample('1H').sum()
        timeframes['1h'] = df_1h.reset_index()
        
        # Save 1h timeframe
        file_path = f'data/{exchange}_{pair_name}_1h.csv'
        timeframes['1h'].to_csv(file_path, index=False)
        logger.info(f"Saved {len(timeframes['1h'])} records to {file_path}")
        
        # 15m timeframe
        df_15m = df.set_index('datetime').price.resample('15T').ohlc()
        df_15m['volume'] = df.set_index('datetime').volume.resample('15T').sum()
        timeframes['15m'] = df_15m.reset_index()
        
        # Save 15m timeframe
        file_path = f'data/{exchange}_{pair_name}_15m.csv'
        timeframes['15m'].to_csv(file_path, index=False)
        logger.info(f"Saved {len(timeframes['15m'])} records to {file_path}")
        
        # 5m timeframe
        df_5m = df.set_index('datetime').price.resample('5T').ohlc()
        df_5m['volume'] = df.set_index('datetime').volume.resample('5T').sum()
        timeframes['5m'] = df_5m.reset_index()
        
        # Save 5m timeframe
        file_path = f'data/{exchange}_{pair_name}_5m.csv'
        timeframes['5m'].to_csv(file_path, index=False)
        logger.info(f"Saved {len(timeframes['5m'])} records to {file_path}")
        
        # 1m timeframe
        df_1m = df.set_index('datetime').price.resample('1T').ohlc()
        df_1m['volume'] = df.set_index('datetime').volume.resample('1T').sum()
        timeframes['1m'] = df_1m.reset_index()
        
        # Save 1m timeframe
        file_path = f'data/{exchange}_{pair_name}_1m.csv'
        timeframes['1m'].to_csv(file_path, index=False)
        logger.info(f"Saved {len(timeframes['1m'])} records to {file_path}")
        
        return timeframes
        
    except Exception as e:
        logger.error(f"Error creating timeframes for {pair_name}: {e}")
        return {}

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with technical indicators
    """
    try:
        # Copy DataFrame
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
            
            # Percentage return (shifted to be used as target)
            df_with_indicators[f'return_{period}h'] = df_with_indicators['close'].pct_change(periods=period).shift(-period)
        
        # Calculate volume indicators
        # Volume SMA
        for period in [5, 10, 20]:
            df_with_indicators[f'volume_sma_{period}'] = df_with_indicators['volume'].rolling(window=period).mean()
        
        # Volume change
        df_with_indicators['volume_change'] = df_with_indicators['volume'].pct_change()
        
        # Drop rows with NaN values
        df_with_indicators = df_with_indicators.dropna()
        
        return df_with_indicators
        
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}")
        return df

def create_lstm_dataset(df: pd.DataFrame, exchange: str, pair_name: str, sequence_length: int = 24) -> None:
    """
    Create LSTM dataset from processed DataFrame
    
    Args:
        df: DataFrame with processed data
        exchange: Name of exchange
        pair_name: Name of trading pair
        sequence_length: Length of input sequence
    """
    try:
        # Create directory for processed data
        os.makedirs('data/processed', exist_ok=True)
        
        # Save processed data
        processed_file = f'data/processed/{exchange}_{pair_name}_processed.csv'
        df.to_csv(processed_file, index=False)
        logger.info(f"Saved {len(df)} processed records to {processed_file}")
        
        # Get feature columns (all except datetime and target columns)
        feature_cols = [col for col in df.columns if col not in ['datetime', 'return_1h', 'return_3h', 'return_6h', 'return_12h', 'return_24h']]
        
        # Get target columns
        target_cols = ['return_1h', 'return_3h', 'return_6h', 'return_12h', 'return_24h']
        
        # Create sequences
        X, y = [], []
        
        for i in range(len(df) - sequence_length):
            # Get sequence of features
            X.append(df[feature_cols].iloc[i:i+sequence_length].values)
            
            # Get targets (prices at the end of the sequence)
            y.append(df[target_cols].iloc[i+sequence_length-1].values)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Split into training and testing sets (80/20)
        train_size = int(len(X) * 0.8)
        
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Save datasets
        base_path = f'data/processed/lstm_{exchange}_{pair_name}'
        
        np.save(f'{base_path}_X_train.npy', X_train)
        np.save(f'{base_path}_y_train.npy', y_train)
        np.save(f'{base_path}_X_test.npy', X_test)
        np.save(f'{base_path}_y_test.npy', y_test)
        
        # Save feature and target names
        with open(f'{base_path}_features.txt', 'w') as f:
            f.write('\n'.join(feature_cols))
            
        with open(f'{base_path}_targets.txt', 'w') as f:
            f.write('\n'.join(target_cols))
        
        logger.info(f"Created LSTM dataset with {len(X)} sequences: {len(X_train)} training, {len(X_test)} testing, {X.shape[2]} features, {y.shape[1]} targets")
        logger.info(f"Saved LSTM datasets to {base_path}_*.npy")
        
    except Exception as e:
        logger.error(f"Error in create_lstm_dataset for {pair_name}: {str(e)}")

def process_pair(file_path: str, pair_name: str, exchange: str = 'binance') -> None:
    """
    Process a single trading pair
    
    Args:
        file_path: Path to CSV file
        pair_name: Name of trading pair
        exchange: Name of exchange
    """
    try:
        logger.info(f"Processing {pair_name} data from {exchange}")
        
        # Load data
        df = load_data(file_path)
        
        if len(df) == 0:
            logger.error(f"No data loaded for {pair_name}")
            return
        
        # Create timeframes
        timeframes = create_timeframes(df, pair_name, exchange)
        
        if not timeframes:
            logger.error(f"Failed to create timeframes for {pair_name}")
            return
        
        # Process 1h timeframe for ML
        logger.info(f"Processing {pair_name} for ML training")
        df_1h_processed = add_technical_indicators(timeframes['1h'])
        
        # Create LSTM dataset
        create_lstm_dataset(df_1h_processed, exchange, pair_name)
        
    except Exception as e:
        logger.error(f"Error processing {pair_name}: {e}")

def main():
    """Main function to process all trading pairs"""
    start_time = time.time()
    logger.info("Starting data processing for new pairs")
    
    # Define pairs to process
    pairs_to_process = [
        {'file': 'attached_assets/ETH-USDT.csv', 'pair': 'ETHUSDT', 'exchange': 'binance'},
        {'file': 'attached_assets/USDT-GBP.csv', 'pair': 'USDTGBP', 'exchange': 'binance'},
        {'file': 'attached_assets/ETH-USDC.csv', 'pair': 'ETHUSDC', 'exchange': 'binance'},
        {'file': 'attached_assets/ETHWETH.csv', 'pair': 'ETHWETH', 'exchange': 'uniswap'},
        {'file': 'attached_assets/UNIETH.csv', 'pair': 'UNIETH', 'exchange': 'uniswap'},
        {'file': 'attached_assets/AAVEETH1.csv', 'pair': 'AAVEETH', 'exchange': 'uniswap'},
        {'file': 'attached_assets/LINKETH.csv', 'pair': 'LINKETH', 'exchange': 'uniswap'}
    ]
    
    # Process each pair
    for pair_info in pairs_to_process:
        try:
            process_pair(pair_info['file'], pair_info['pair'], pair_info['exchange'])
        except Exception as e:
            logger.error(f"Failed to process {pair_info['pair']}: {str(e)}")
    
    logger.info(f"All data processing completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()