"""
Simplified script to process new cryptocurrency data files for ML model training
"""
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import math
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DataProcessor')

def load_and_preprocess_csv(filepath, exchange='kraken'):
    """
    Load a cryptocurrency CSV file and preprocess it into a pandas DataFrame
    
    Args:
        filepath: Path to the CSV file
        exchange: Exchange name (affects how we interpret the data format)
        
    Returns:
        Preprocessed DataFrame or None if processing fails
    """
    try:
        # Get symbol name from filename
        filename = os.path.basename(filepath)
        symbol = os.path.splitext(filename)[0]
        logger.info(f"Processing {symbol} from {exchange}")
        
        # Load CSV data
        # Different exchanges have different CSV formats
        try:
            # Format: timestamp,price,volume
            df = pd.read_csv(filepath, header=None, names=['timestamp', 'price', 'volume'])
            
            # Check if we have the required columns
            if not all(col in df.columns for col in ['timestamp', 'price', 'volume']):
                # Try to infer which columns might be timestamp, price, volume
                cols = df.columns.tolist()
                if len(cols) >= 3:
                    df = df.rename(columns={cols[0]: 'timestamp', cols[1]: 'price', cols[2]: 'volume'})
                    logger.info(f"Renamed columns for {symbol}: {cols} -> ['timestamp', 'price', 'volume']")
                else:
                    logger.error(f"Not enough columns in {filepath}, expected at least 3 columns")
                    return None
        except Exception as e:
            logger.error(f"Failed to load CSV {filepath}: {str(e)}")
            return None
        
        # Convert timestamp to datetime
        if df['timestamp'].dtype == 'object':
            # Try various timestamp formats
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                except:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    except Exception as e:
                        logger.error(f"Failed to parse timestamp for {symbol}: {str(e)}")
                        return None
        else:
            # Numeric timestamp in seconds or milliseconds
            if df['timestamp'].iloc[0] > 1e12:  # Milliseconds
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:  # Seconds
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Sort by timestamp to ensure chronological order
        df = df.sort_values('timestamp')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        # Ensure numeric price and volume
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Handle missing values
        df = df.dropna()
        
        # Calculate basic features
        df['return'] = df['price'].pct_change(fill_method=None)
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        
        # Calculate price changes
        df['price_change_1'] = df['price'].pct_change(1, fill_method=None)
        df['price_change_6'] = df['price'].pct_change(6, fill_method=None)
        df['price_change_12'] = df['price'].pct_change(12, fill_method=None)
        df['price_change_24'] = df['price'].pct_change(24, fill_method=None)
        
        # Simple moving averages
        df['sma_5'] = df['price'].rolling(window=5).mean()
        df['sma_10'] = df['price'].rolling(window=10).mean()
        df['sma_20'] = df['price'].rolling(window=20).mean()
        
        # Exponential moving averages
        df['ema_5'] = df['price'].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df['price'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['price'].ewm(span=20, adjust=False).mean()
        
        # Volume moving average
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        
        # Simple RSI
        delta = df['price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Normalized volume
        max_vol = df['volume'].max()
        if max_vol > 0:
            df['volume_normalized'] = df['volume'] / max_vol
        else:
            df['volume_normalized'] = df['volume']
        
        # Add time features
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['weekday'] = df.index.weekday
        
        # Handle missing values
        df = df.dropna()
        
        logger.info(f"Loaded {len(df)} rows for {symbol}")
        return df
    
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return None

def resample_to_timeframes(df, timeframes=['1h', '4h', '1d']):
    """
    Resample the DataFrame to different timeframes
    
    Args:
        df: DataFrame with timestamp index
        timeframes: List of timeframes to resample to
        
    Returns:
        Dictionary of resampled DataFrames
    """
    resampled = {}
    
    try:
        for tf in timeframes:
            # Resample using mean for price and sum for volume
            df_resampled = df.resample(tf).agg({
                'price': 'last',
                'volume': 'sum',
                'return': 'sum',
                'log_return': 'sum',
                'volume_normalized': 'mean',
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
                'rsi_14': 'last',
                'hour': 'first',
                'day': 'first',
                'month': 'first',
                'weekday': 'first'
            })
            
            # Store resampled DataFrame
            resampled[tf] = df_resampled
            
            # Drop NaN values
            resampled[tf] = resampled[tf].dropna()
            
            logger.info(f"Resampled to {tf} with {len(df_resampled)} rows")
    
    except Exception as e:
        logger.error(f"Error resampling data: {str(e)}")
    
    return resampled

def prepare_lstm_sequences(df, sequence_length=24, prediction_horizons=[1, 3, 6, 12, 24]):
    """
    Prepare sequences for LSTM training
    
    Args:
        df: DataFrame with features
        sequence_length: Length of input sequences
        prediction_horizons: List of future horizons to predict returns for
        
    Returns:
        X, y and feature names
    """
    if len(df) < sequence_length + max(prediction_horizons):
        logger.warning(f"Not enough data for LSTM sequences (need {sequence_length + max(prediction_horizons)} rows, got {len(df)})")
        return None, None, None, None
    
    try:
        # Calculate future returns for each prediction horizon
        for horizon in prediction_horizons:
            df[f'return_{horizon}h'] = df['price'].pct_change(periods=horizon, fill_method=None).shift(-horizon)
        
        # Drop rows with NaN in target columns
        target_cols = [f'return_{h}h' for h in prediction_horizons]
        df = df.dropna(subset=target_cols)
        
        # Get all numeric columns for features
        feature_cols = ['price', 'volume', 'return', 'log_return', 'volume_normalized', 
                       'price_change_1', 'price_change_6', 'price_change_12', 'price_change_24',
                       'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'ema_20', 
                       'volume_sma_5', 'rsi_14', 'hour', 'day', 'month', 'weekday']
        
        # Create sequences
        X = []
        y = []
        
        for i in range(len(df) - sequence_length):
            X.append(df[feature_cols].iloc[i:i+sequence_length].values)
            y.append(df[target_cols].iloc[i+sequence_length-1].values)
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y, feature_cols, target_cols
    
    except Exception as e:
        logger.error(f"Error preparing LSTM sequences: {str(e)}")
        return None, None, None, None

def process_and_save_data(filepath, output_dir, exchange='kraken'):
    """
    Process a cryptocurrency data file and save the processed data
    
    Args:
        filepath: Path to the CSV file
        output_dir: Directory to save processed data
        exchange: Exchange name
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get symbol name from filename
        filename = os.path.basename(filepath)
        symbol = os.path.splitext(filename)[0]
        logger.info(f"Processing {symbol} from {exchange}")
        
        # Load and preprocess data
        df = load_and_preprocess_csv(filepath, exchange)
        if df is None or len(df) == 0:
            logger.error(f"Failed to load and preprocess {filepath}")
            return False
        
        # Resample to different timeframes
        timeframes = ['1h', '4h', '1d']
        resampled = resample_to_timeframes(df, timeframes)
        
        # Save processed data for each timeframe
        for tf, df_resampled in resampled.items():
            if len(df_resampled) > 0:
                # Save to CSV
                output_file = os.path.join(output_dir, f"{exchange}_{symbol}_{tf}_processed.csv")
                df_resampled.to_csv(output_file)
                logger.info(f"Saved processed data to {output_file}")
        
        # Prepare LSTM sequences for 1h timeframe
        df_1h = resampled.get('1h', None)
        X, y, feature_cols, target_cols = None, None, None, None
        
        if df_1h is not None and len(df_1h) >= 100:  # Ensure we have enough data
            X, y, feature_cols, target_cols = prepare_lstm_sequences(df_1h)
        
        if X is not None and y is not None and len(X) > 0 and len(y) > 0:
            # Split into train and test sets (80/20)
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Save LSTM data
            lstm_dir = os.path.join(output_dir, 'lstm')
            os.makedirs(lstm_dir, exist_ok=True)
            
            np.save(os.path.join(output_dir, f"lstm_{exchange}_{symbol}_X_train.npy"), X_train)
            np.save(os.path.join(output_dir, f"lstm_{exchange}_{symbol}_y_train.npy"), y_train)
            np.save(os.path.join(output_dir, f"lstm_{exchange}_{symbol}_X_test.npy"), X_test)
            np.save(os.path.join(output_dir, f"lstm_{exchange}_{symbol}_y_test.npy"), y_test)
            
            # Save feature and target names
            with open(os.path.join(output_dir, f"lstm_{exchange}_{symbol}_features.txt"), 'w') as f:
                f.write('\n'.join(feature_cols))
            
            with open(os.path.join(output_dir, f"lstm_{exchange}_{symbol}_targets.txt"), 'w') as f:
                f.write('\n'.join(target_cols))
            
            logger.info(f"Saved LSTM data for {symbol}")
            
            # Create a simple plot of price
            plt.figure(figsize=(12, 6))
            plt.plot(df_1h['price'])
            plt.title(f"{symbol} Price ({exchange})")
            plt.xlabel('Time')
            plt.ylabel('Price')
            
            plot_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f"{exchange}_{symbol}_price.png"))
            plt.close()
        else:
            logger.warning(f"Not enough data to prepare LSTM sequences for {symbol}")
        
        # Save metadata about the processing
        metadata = {
            'symbol': symbol,
            'exchange': exchange,
            'original_rows': len(df),
            'processed_rows': {tf: len(df_tf) for tf, df_tf in resampled.items()},
            'processed_date': datetime.now().isoformat(),
            'timeframes': timeframes,
            'has_lstm_data': X is not None and y is not None and len(X) > 0 and len(y) > 0
        }
        
        with open(os.path.join(output_dir, f"{exchange}_{symbol}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return False

def main():
    """Main function to process selected pairs"""
    # Pairs to process
    pairs = ['ATHEUR', 'DBREUR', 'CPOOLEUR', 'EURQEUR', 'EURQUSD', 'GTCEUR']
    
    input_dir = 'attached_assets'
    output_dir = 'data/processed'
    exchange = 'uniswap'
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'lstm'), exist_ok=True)
    
    # Process each pair
    for pair in pairs:
        filepath = os.path.join(input_dir, f"{pair}.csv")
        if os.path.exists(filepath):
            logger.info(f"Processing {pair}...")
            success = process_and_save_data(filepath, output_dir, exchange)
            if success:
                logger.info(f"Successfully processed {pair}")
            else:
                logger.error(f"Failed to process {pair}")
        else:
            logger.error(f"File not found: {filepath}")
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()