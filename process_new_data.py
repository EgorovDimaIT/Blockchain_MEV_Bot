"""
Script to process additional crypto data files
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_crypto_data(file_name, exchange_name):
    """Process cryptocurrency data from CSV file"""
    logger.info(f"Processing {file_name} data from {exchange_name}")
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    
    # Path to data file
    file_path = f'data/{file_name}.csv'
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return
    
    try:
        # Read data
        df = pd.read_csv(file_path, header=None)
        
        # Add column names
        df.columns = ['timestamp', 'price', 'volume']
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Set timestamp as index for resampling
        df.set_index('timestamp', inplace=True)
        
        # Process data for different timeframes
        timeframes = {
            '1h': '1h',
            '15m': '15min',
            '5m': '5min',
            '1m': '1min'
        }
        
        for tf_name, tf_code in timeframes.items():
            logger.info(f"Creating {tf_name} timeframe data for {file_name}")
            
            # Resample to the specified timeframe
            resampled = df.resample(tf_code).agg({
                'price': ['first', 'max', 'min', 'last'],
                'volume': 'sum'
            })
            
            # Flatten the multi-index columns
            resampled.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Reset index to make timestamp a column again
            resampled = resampled.reset_index()
            
            # Save to data directory
            output_path = f'data/{exchange_name}_{file_name}_{tf_name}.csv'
            resampled.to_csv(output_path, index=False)
            logger.info(f"Saved {len(resampled)} records to {output_path}")
            
            # If 1h data, process further for ML
            if tf_name == '1h':
                process_for_ml(resampled, exchange_name, file_name)
    except Exception as e:
        logger.error(f"Error processing {file_name}: {str(e)}")

def process_for_ml(df, exchange_name, file_name):
    """Process data for ML training"""
    logger.info(f"Processing {file_name} for ML training")
    
    try:
        # Add technical indicators
        # Simple Moving Averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        
        # Exponential Moving Averages
        for window in [5, 10, 20, 50]:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Moving Average Convergence Divergence (MACD)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        std_dev = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (std_dev * 2)
        df['bb_lower'] = df['bb_middle'] - (std_dev * 2)
        
        # Add price ratios
        df['close_open_ratio'] = df['close'] / df['open']
        df['high_low_ratio'] = df['high'] / df['low']
        
        # Normalize volume
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Calculate returns for different time periods
        for period in [1, 3, 6, 12, 24]:  # different periods in hours
            df[f'return_{period}h'] = df['close'].pct_change(periods=period).shift(-period)
        
        # Add day of week and hour of day
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['hour_of_day'] = df['timestamp'].dt.hour
        
        # Drop NaN values
        df = df.dropna()
        
        # Save processed data
        symbol = f"{exchange_name}_{file_name}"
        output_path = f'data/processed/{symbol}_processed.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} processed records to {output_path}")
        
        # Create LSTM dataset
        create_lstm_dataset(df, exchange_name, file_name)
    except Exception as e:
        logger.error(f"Error in process_for_ml for {file_name}: {str(e)}")

def create_lstm_dataset(df, exchange_name, file_name, sequence_length: int = 24):
    """Create LSTM dataset from processed DataFrame"""
    logger.info(f"Creating LSTM dataset for {file_name}")
    
    try:
        symbol = f"{exchange_name}_{file_name}"
        
        # Target columns (future returns)
        target_columns = [col for col in df.columns if col.startswith('return_')]
        
        # Features for LSTM (excluding timestamp and targets)
        feature_columns = [col for col in df.columns if col not in ['timestamp'] + target_columns]
        
        # Prepare sequences
        sequences = []
        targets = []
        
        for i in range(len(df) - sequence_length):
            # Extract sequence
            seq = df[feature_columns].iloc[i:i+sequence_length].values
            # Extract targets
            target = df[target_columns].iloc[i+sequence_length-1].values
            
            sequences.append(seq)
            targets.append(target)
        
        # Convert to numpy arrays
        X = np.array(sequences)
        y = np.array(targets)
        
        # Split into train/test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        logger.info(f"Created LSTM dataset with {len(X)} sequences: " +
                    f"{X_train.shape[0]} training, {X_test.shape[0]} testing, " +
                    f"{X_train.shape[2]} features, {y_train.shape[1]} targets")
        
        # Save datasets
        base_path = f'data/processed/lstm_{symbol}'
        
        np.save(f'{base_path}_X_train.npy', X_train)
        np.save(f'{base_path}_y_train.npy', y_train)
        np.save(f'{base_path}_X_test.npy', X_test)
        np.save(f'{base_path}_y_test.npy', y_test)
        
        # Save feature and target names
        with open(f'{base_path}_features.txt', 'w') as f:
            f.write('\n'.join(feature_columns))
        
        with open(f'{base_path}_targets.txt', 'w') as f:
            f.write('\n'.join(target_columns))
        
        logger.info(f"Saved LSTM datasets to {base_path}_*.npy")
    except Exception as e:
        logger.error(f"Error in create_lstm_dataset for {file_name}: {str(e)}")

def main():
    """Main function to process all data files"""
    start_time = time.time()
    logger.info("Starting data processing")
    
    # Define mappings of file names to exchanges
    file_exchange_mappings = {
        'ATHUSD': 'binance',
        'ATLASEUR': 'kraken',
        'FISEUR': 'kraken',
        'FISUSD': 'kraken',
        'FLOKIEUR': 'binance',
        'FLOKIUSD': 'binance',
        'ETHWEUR': 'uniswap',
        'ETHWUSD': 'uniswap',
        'BICOEUR': 'binance',
        'BICOUSD': 'binance',
        'BIGTIMEEUR': 'binance',
        'BIGTIMEUSD': 'binance',
        'BITEUR': 'binance',
        'BITUSD': 'binance',
        'BLUREUR': 'binance'
    }
    
    # Process only selected files to avoid timeout
    files_to_process = ['ETHWEUR', 'BICOEUR', 'BIGTIMEEUR']
    
    # Process each file
    for file_name in files_to_process:
        try:
            exchange_name = file_exchange_mappings[file_name]
            logger.info(f"Processing {file_name} data from {exchange_name}")
            process_crypto_data(file_name, exchange_name)
        except Exception as e:
            logger.error(f"Failed to process {file_name}: {str(e)}")
    
    logger.info(f"Selected data processing completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()