"""
Process additional cryptocurrency pairs for model training
"""
import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DataProcessor')

def load_csv(filepath):
    """
    Load CSV file with cryptocurrency data
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with loaded data or None if loading fails
    """
    try:
        # Get symbol name from filename
        filename = os.path.basename(filepath)
        symbol = os.path.splitext(filename)[0]
        
        # Load CSV data
        df = pd.read_csv(filepath, header=None, names=['timestamp', 'price', 'volume'])
        
        # Convert timestamp to datetime
        if df['timestamp'].dtype == np.int64 or df['timestamp'].dtype == np.float64:
            if df['timestamp'].iloc[0] > 1e12:  # milliseconds
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:  # seconds
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        logger.info(f"Loaded {len(df)} rows for {symbol}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        return None

def preprocess_data(df):
    """
    Preprocess data for model training
    
    Args:
        df: DataFrame with raw data
        
    Returns:
        DataFrame with preprocessed data
    """
    try:
        # Drop duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.dropna()
        
        # Make sure price and volume are numeric
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Drop rows with invalid values
        df = df.dropna()
        
        # Calculate features
        # Returns
        df['return'] = df['price'].pct_change(fill_method=None)
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        
        # Price changes
        df['price_change_1'] = df['price'].pct_change(1, fill_method=None)
        df['price_change_6'] = df['price'].pct_change(6, fill_method=None)
        df['price_change_12'] = df['price'].pct_change(12, fill_method=None)
        df['price_change_24'] = df['price'].pct_change(24, fill_method=None)
        
        # Moving averages
        df['sma_5'] = df['price'].rolling(window=5).mean()
        df['sma_10'] = df['price'].rolling(window=10).mean()
        df['sma_20'] = df['price'].rolling(window=20).mean()
        
        # Exponential moving averages
        df['ema_5'] = df['price'].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df['price'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['price'].ewm(span=20, adjust=False).mean()
        
        # Volume features
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        
        # Price volatility
        df['volatility_5'] = df['return'].rolling(window=5).std()
        df['volatility_10'] = df['return'].rolling(window=10).std()
        
        # Normalized price and volume
        min_price = df['price'].min()
        max_price = df['price'].max()
        if max_price > min_price:
            df['price_normalized'] = (df['price'] - min_price) / (max_price - min_price)
        else:
            df['price_normalized'] = df['price']
        
        max_volume = df['volume'].max()
        if max_volume > 0:
            df['volume_normalized'] = df['volume'] / max_volume
        else:
            df['volume_normalized'] = df['volume']
        
        # Time features
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['weekday'] = df.index.weekday
        
        # RSI (Relative Strength Index)
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        return df

def resample_data(df, timeframe):
    """
    Resample data to specific timeframe
    
    Args:
        df: DataFrame with data
        timeframe: Timeframe to resample to (e.g., '1h', '4h', '1d')
        
    Returns:
        Resampled DataFrame
    """
    try:
        # Resample data
        df_resampled = df.resample(timeframe).agg({
            'price': 'last',
            'volume': 'sum',
            'return': 'sum',
            'log_return': 'sum',
            'price_change_1': 'last',
            'price_change_6': 'last',
            'price_change_12': 'last',
            'price_change_24': 'last',
            'sma_5': 'last',
            'sma_10': 'last',
            'sma_20': 'last',
            'ema_5': 'last',
            'ema_10': 'last',
            'ema_20': 'last',
            'volume_sma_5': 'last',
            'volatility_5': 'last',
            'volatility_10': 'last',
            'price_normalized': 'last',
            'volume_normalized': 'mean',
            'rsi_14': 'last',
            'hour': 'first',
            'day': 'first',
            'month': 'first',
            'weekday': 'first'
        })
        
        # Drop rows with NaN values
        df_resampled = df_resampled.dropna()
        
        logger.info(f"Resampled to {timeframe} with {len(df_resampled)} rows")
        return df_resampled
    
    except Exception as e:
        logger.error(f"Error resampling data: {str(e)}")
        return None

def create_lstm_sequences(df, sequence_length=24, prediction_horizons=[1, 3, 6, 12, 24]):
    """
    Create sequences for LSTM model training
    
    Args:
        df: DataFrame with processed data
        sequence_length: Length of input sequences
        prediction_horizons: List of time horizons to predict returns for
        
    Returns:
        X: Input sequences
        y: Target values
        feature_cols: List of feature column names
        target_cols: List of target column names
    """
    try:
        # Check if we have enough data
        if len(df) < sequence_length + max(prediction_horizons):
            logger.warning(f"Not enough data for LSTM sequences (need {sequence_length + max(prediction_horizons)}, got {len(df)})")
            return None, None, None, None
        
        # Calculate future returns for prediction horizons
        for horizon in prediction_horizons:
            df[f'future_return_{horizon}'] = df['price'].pct_change(periods=horizon, fill_method=None).shift(-horizon)
        
        # Drop rows with NaN in target columns
        target_cols = [f'future_return_{h}' for h in prediction_horizons]
        df = df.dropna(subset=target_cols)
        
        # Select features
        feature_cols = ['price', 'volume', 'return', 'log_return', 'price_change_1', 
                        'price_change_6', 'price_change_12', 'price_change_24',
                        'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'ema_20',
                        'volume_sma_5', 'volatility_5', 'volatility_10',
                        'price_normalized', 'volume_normalized', 'rsi_14',
                        'hour', 'day', 'month', 'weekday']
        
        # Prepare sequences
        X = []
        y = []
        
        for i in range(len(df) - sequence_length):
            X.append(df[feature_cols].iloc[i:i+sequence_length].values)
            y.append(df[target_cols].iloc[i+sequence_length-1].values)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        return X, y, feature_cols, target_cols
    
    except Exception as e:
        logger.error(f"Error creating LSTM sequences: {str(e)}")
        return None, None, None, None

def process_and_save_pair(filepath, output_dir, exchange='uniswap', sequence_length=24):
    """
    Process and save data for a cryptocurrency pair
    
    Args:
        filepath: Path to CSV file
        output_dir: Output directory
        exchange: Name of exchange
        sequence_length: Length of LSTM sequences
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        # Get symbol name from filename
        filename = os.path.basename(filepath)
        symbol = os.path.splitext(filename)[0]
        logger.info(f"Processing {symbol} from {exchange}")
        
        # Load data
        df = load_csv(filepath)
        if df is None or len(df) == 0:
            logger.error(f"Failed to load data for {symbol}")
            return False
        
        # Preprocess data
        df = preprocess_data(df)
        if df is None or len(df) == 0:
            logger.error(f"Failed to preprocess data for {symbol}")
            return False
        
        # Resample to different timeframes
        timeframes = ['1h', '4h', '1d']
        resampled_data = {}
        
        for tf in timeframes:
            df_resampled = resample_data(df, tf)
            if df_resampled is not None and len(df_resampled) > 0:
                resampled_data[tf] = df_resampled
                
                # Save resampled data
                output_file = os.path.join(output_dir, f"{exchange}_{symbol}_{tf}_processed.csv")
                df_resampled.to_csv(output_file)
                logger.info(f"Saved processed data to {output_file}")
                
                # Create plot
                plt.figure(figsize=(12, 6))
                plt.plot(df_resampled['price'])
                plt.title(f"{symbol} Price ({exchange}, {tf})")
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'plots', f"{exchange}_{symbol}_{tf}_price.png"))
                plt.close()
        
        # Create LSTM sequences
        if '1h' in resampled_data and len(resampled_data['1h']) >= 100:
            X, y, feature_cols, target_cols = create_lstm_sequences(
                resampled_data['1h'], sequence_length=sequence_length
            )
            
            if X is not None and y is not None and len(X) > 0:
                # Split into train and test sets (80/20)
                train_size = int(0.8 * len(X))
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                # Save LSTM data
                np.save(os.path.join(output_dir, f"lstm_{exchange}_{symbol}_X_train.npy"), X_train)
                np.save(os.path.join(output_dir, f"lstm_{exchange}_{symbol}_y_train.npy"), y_train)
                np.save(os.path.join(output_dir, f"lstm_{exchange}_{symbol}_X_test.npy"), X_test)
                np.save(os.path.join(output_dir, f"lstm_{exchange}_{symbol}_y_test.npy"), y_test)
                
                # Save feature and target names
                with open(os.path.join(output_dir, f"lstm_{exchange}_{symbol}_features.json"), 'w') as f:
                    json.dump(feature_cols, f)
                
                with open(os.path.join(output_dir, f"lstm_{exchange}_{symbol}_targets.json"), 'w') as f:
                    json.dump(target_cols, f)
                
                logger.info(f"Saved LSTM data for {symbol}")
        else:
            logger.warning(f"Not enough data to create LSTM sequences for {symbol}")
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'exchange': exchange,
            'original_rows': len(df),
            'resampled_rows': {tf: len(data) for tf, data in resampled_data.items()},
            'has_lstm_data': X is not None and y is not None and len(X) > 0 if 'X' in locals() else False,
            'processed_date': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, f"{exchange}_{symbol}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Successfully processed {symbol}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Process cryptocurrency data')
    parser.add_argument('--input_dir', type=str, default='attached_assets', help='Input directory with CSV files')
    parser.add_argument('--output_dir', type=str, default='data/processed', help='Output directory for processed data')
    parser.add_argument('--exchange', type=str, default='uniswap', help='Exchange name')
    parser.add_argument('--symbols', type=str, nargs='+', help='Symbols to process (if not specified, all files in input_dir will be processed)')
    parser.add_argument('--sequence_length', type=int, default=24, help='Length of LSTM sequences')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of files to process
    if args.symbols:
        files = [os.path.join(args.input_dir, f"{symbol}.csv") for symbol in args.symbols]
        files = [f for f in files if os.path.exists(f)]
        if not files:
            logger.error(f"No matching files found for specified symbols")
            return
    else:
        files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) 
                if f.endswith('.csv') and os.path.isfile(os.path.join(args.input_dir, f))]
    
    logger.info(f"Found {len(files)} files to process")
    
    # Process each file
    success_count = 0
    for filepath in files:
        if process_and_save_pair(filepath, args.output_dir, args.exchange, args.sequence_length):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count} out of {len(files)} files")

if __name__ == "__main__":
    main()