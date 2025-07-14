"""
Script to process new cryptocurrency data files for ML model training
"""
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import math
import json
from tqdm import tqdm
import argparse

# Define TA functions to replace talib
def SMA(series, timeperiod):
    """Simple Moving Average"""
    return series.rolling(window=timeperiod).mean()

def EMA(series, timeperiod):
    """Exponential Moving Average"""
    return series.ewm(span=timeperiod, adjust=False).mean()

def RSI(series, timeperiod=14):
    """Relative Strength Index"""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    ma_up = up.rolling(window=timeperiod).mean()
    ma_down = down.rolling(window=timeperiod).mean()
    
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi

def BBANDS(series, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
    """Bollinger Bands"""
    middle = series.rolling(window=timeperiod).mean()
    std_dev = series.rolling(window=timeperiod).std()
    
    upper = middle + std_dev * nbdevup
    lower = middle - std_dev * nbdevdn
    
    return upper, middle, lower

def MACD(series, fastperiod=12, slowperiod=26, signalperiod=9):
    """Moving Average Convergence/Divergence"""
    fast_ema = EMA(series, fastperiod)
    slow_ema = EMA(series, slowperiod)
    macd = fast_ema - slow_ema
    signal = EMA(macd, signalperiod)
    hist = macd - signal
    
    return macd, signal, hist

def STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
    """Stochastic"""
    highest_high = high.rolling(window=fastk_period).max()
    lowest_low = low.rolling(window=fastk_period).min()
    
    fastk = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    fastk = fastk.rolling(window=slowk_period).mean()
    slowd = fastk.rolling(window=slowd_period).mean()
    
    return fastk, slowd

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
            if exchange in ['kraken', 'binance', 'uniswap']:
                # Format: timestamp,price,volume
                df = pd.read_csv(filepath, header=None, names=['timestamp', 'price', 'volume'])
            else:
                # Try a format with headers
                df = pd.read_csv(filepath)
                
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
        
        # Calculate returns
        df['return'] = df['price'].pct_change()
        
        # Calculate log returns
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        
        # Handle missing values created by pct_change and log operations
        df = df.dropna()
        
        # Normalize volume
        max_vol = df['volume'].max()
        if max_vol > 0:
            df['volume_normalized'] = df['volume'] / max_vol
        else:
            df['volume_normalized'] = df['volume']
        
        logger.info(f"Loaded {len(df)} rows for {symbol}")
        return df
    
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return None

def add_technical_indicators(df):
    """
    Add technical indicators to the DataFrame
    
    Args:
        df: DataFrame with price and volume data
        
    Returns:
        DataFrame with added technical indicators
    """
    try:
        # Extract price series
        close_series = df['price']
        high_series = df['price']  # Using price for high (we only have OHLCV)
        low_series = df['price']   # Using price for low
        open_series = df['price']  # Using price for open
        volume_series = df['volume']
        
        # Calculate price changes
        df['price_change_1'] = df['price'].pct_change(1)
        df['price_change_6'] = df['price'].pct_change(6)
        df['price_change_12'] = df['price'].pct_change(12)
        df['price_change_24'] = df['price'].pct_change(24)
        
        # Moving averages
        df['sma_5'] = SMA(close_series, timeperiod=5)
        df['sma_10'] = SMA(close_series, timeperiod=10)
        df['sma_20'] = SMA(close_series, timeperiod=20)
        df['sma_50'] = SMA(close_series, timeperiod=50)
        df['sma_100'] = SMA(close_series, timeperiod=100)
        
        # Exponential moving averages
        df['ema_5'] = EMA(close_series, timeperiod=5)
        df['ema_10'] = EMA(close_series, timeperiod=10)
        df['ema_20'] = EMA(close_series, timeperiod=20)
        df['ema_50'] = EMA(close_series, timeperiod=50)
        df['ema_100'] = EMA(close_series, timeperiod=100)
        
        # Volume indicators
        df['volume_sma_5'] = SMA(volume_series, timeperiod=5)
        df['volume_sma_10'] = SMA(volume_series, timeperiod=10)
        df['volume_sma_20'] = SMA(volume_series, timeperiod=20)
        
        # RSI
        df['rsi_14'] = RSI(close_series, timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = MACD(close_series, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # Bollinger Bands
        upper, middle, lower = BBANDS(close_series, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle
        
        # Stochastic
        slowk, slowd = STOCH(high_series, low_series, close_series, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        # High/Low ratio
        df['high_low_ratio'] = high_series / low_series
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    except Exception as e:
        logger.error(f"Error adding technical indicators: {str(e)}")
        return df

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
                'volume_normalized': 'mean'
            })
            
            # Add technical indicators
            df_resampled = add_technical_indicators(df_resampled)
            
            # Store resampled DataFrame
            resampled[tf] = df_resampled
            
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
    try:
        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Add timestamp as a feature (cyclical encoding)
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['weekday'] = df.index.weekday
        df['timestamp'] = np.arange(len(df))  # Numeric timestamp for modeling
        
        # Calculate future returns for each prediction horizon
        for horizon in prediction_horizons:
            df[f'return_{horizon}h'] = df['price'].pct_change(periods=horizon).shift(-horizon)
        
        # Drop rows with NaN in target columns
        target_cols = [f'return_{h}h' for h in prediction_horizons]
        df = df.dropna(subset=target_cols)
        
        # Select features (all numeric columns except targets)
        features = [col for col in numeric_cols if col not in target_cols and col != 'return' and col != 'log_return']
        features.append('timestamp')  # Add timestamp as a feature
        
        # Create sequences of length sequence_length
        X = []
        y = []
        
        for i in range(len(df) - sequence_length):
            X.append(df[features].iloc[i:i+sequence_length].values)
            y.append(df[target_cols].iloc[i+sequence_length-1])
        
        return np.array(X), np.array(y), features, target_cols
    
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
        if df is None:
            logger.error(f"Failed to load and preprocess {filepath}")
            return False
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Resample to different timeframes
        timeframes = ['1h', '4h', '1d']
        resampled = resample_to_timeframes(df, timeframes)
        
        # Save processed data for each timeframe
        for tf, df_resampled in resampled.items():
            # Save to CSV
            output_file = os.path.join(output_dir, f"{exchange}_{symbol}_{tf}_processed.csv")
            df_resampled.to_csv(output_file)
            logger.info(f"Saved processed data to {output_file}")
        
        # Prepare LSTM sequences for 1h timeframe
        df_1h = resampled.get('1h', None)
        if df_1h is not None and len(df_1h) > 100:  # Ensure we have enough data
            X, y, features, target_cols = prepare_lstm_sequences(df_1h)
            
            if X is not None and y is not None:
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
                    f.write('\n'.join(features))
                
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
        
        # Save metadata about the processing
        metadata = {
            'symbol': symbol,
            'exchange': exchange,
            'original_rows': len(df),
            'processed_rows': {tf: len(df_tf) for tf, df_tf in resampled.items()},
            'processed_date': datetime.now().isoformat(),
            'timeframes': timeframes,
            'has_lstm_data': X is not None and y is not None
        }
        
        with open(os.path.join(output_dir, f"{exchange}_{symbol}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return False

def main():
    """Main function for processing cryptocurrency data files"""
    parser = argparse.ArgumentParser(description='Process cryptocurrency data files')
    parser.add_argument('--input_dir', type=str, default='attached_assets', help='Input directory with CSV files')
    parser.add_argument('--output_dir', type=str, default='data/processed', help='Output directory for processed data')
    parser.add_argument('--exchange', type=str, default='uniswap', help='Exchange name')
    parser.add_argument('--symbol', type=str, default=None, help='Process only this symbol (optional)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of CSV files
    if args.symbol:
        # Process only the specified symbol
        filepath = os.path.join(args.input_dir, f"{args.symbol}.csv")
        if not os.path.exists(filepath):
            logger.error(f"File {filepath} not found")
            return
        
        process_and_save_data(filepath, args.output_dir, args.exchange)
    else:
        # Process all CSV files in the input directory
        files = [f for f in os.listdir(args.input_dir) if f.endswith('.csv')]
        logger.info(f"Found {len(files)} CSV files in {args.input_dir}")
        
        for file in tqdm(files, desc="Processing files"):
            filepath = os.path.join(args.input_dir, file)
            process_and_save_data(filepath, args.output_dir, args.exchange)
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()