"""
Data processing module for MEV bot
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import time
import datetime
import json
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Class for processing data for ML models"""
    
    def __init__(self, input_dir: str = 'data', output_dir: str = 'data/processed'):
        """
        Initialize data processor
        
        Args:
            input_dir: Directory with raw data
            output_dir: Directory to save processed data
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_ohlcv_data(self, file_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Process OHLCV data
        
        Args:
            file_path: Path to OHLCV CSV file
            output_path: Path to save processed data (optional)
            
        Returns:
            Processed DataFrame
        """
        try:
            # Load data
            df = pd.read_csv(file_path)
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
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
            
            # Average True Range (ATR)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(window=14).mean()
            
            # Add price ratios
            df['close_open_ratio'] = df['close'] / df['open']
            df['high_low_ratio'] = df['high'] / df['low']
            
            # Normalize volume
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # Calculate returns
            for period in [1, 3, 6, 12, 24]:  # different periods in hours
                df[f'return_{period}h'] = df['close'].pct_change(periods=period).shift(-period)
            
            # Add day of week and hour of day
            if 'timestamp' in df.columns:
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['hour_of_day'] = df['timestamp'].dt.hour
            
            # Drop NaN values
            df = df.dropna()
            
            # Save processed data if output_path is provided
            if output_path:
                df.to_csv(output_path, index=False)
                logger.info(f"Saved processed data to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing OHLCV data from {file_path}: {e}")
            return pd.DataFrame()
    
    def create_lstm_dataset(self, df: pd.DataFrame, sequence_length: int = 24,
                           target_columns: List[str] = None, symbol: str = None) -> Dict[str, Any]:
        """
        Create LSTM dataset from processed DataFrame
        
        Args:
            df: Processed DataFrame
            sequence_length: Length of sequences
            target_columns: Columns to use as targets (default is return_1h, return_3h, etc.)
            symbol: Symbol name for saving files
            
        Returns:
            Dictionary with LSTM datasets
        """
        try:
            if target_columns is None:
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
            
            # Save datasets if symbol is provided
            if symbol:
                base_path = os.path.join(self.output_dir, f'lstm_{symbol}')
                
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
            
            return {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'feature_columns': feature_columns,
                'target_columns': target_columns
            }
            
        except Exception as e:
            logger.error(f"Error creating LSTM dataset: {e}")
            return {}
    
    def process_arbitrage_data(self, file_path: str, arbitrage_type: str = 'direct',
                              test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Process arbitrage opportunity data
        
        Args:
            file_path: Path to arbitrage CSV file
            arbitrage_type: Type of arbitrage ('direct' or 'triangular')
            test_size: Size of test set
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with processed data
        """
        try:
            # Load data
            df = pd.read_csv(file_path)
            
            # Check if target column exists
            if 'target' not in df.columns:
                logger.warning(f"Target column not found in {file_path}")
                return {
                    'processed': None,
                    'train': None,
                    'test': None
                }
            
            # Prepare features
            if arbitrage_type == 'direct':
                # For direct arbitrage
                # Exclude non-feature columns
                excluded_cols = ['timestamp', 'datetime', 'target', 'symbol', 'exchange1', 'exchange2']
                feature_cols = [col for col in df.columns if col not in excluded_cols]
            else:
                # For triangular arbitrage
                excluded_cols = ['timestamp', 'datetime', 'target', 'path', 'exchange']
                feature_cols = [col for col in df.columns if col not in excluded_cols]
            
            # Split into train/test
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state, shuffle=False
            )
            
            # Save files
            train_output = os.path.join(self.output_dir, f'arbitrage_{arbitrage_type}_train.csv')
            test_output = os.path.join(self.output_dir, f'arbitrage_{arbitrage_type}_test.csv')
            
            train_df.to_csv(train_output, index=False)
            test_df.to_csv(test_output, index=False)
            
            logger.info(f"Saved processed arbitrage data to {train_output} and {test_output}")
            logger.info(f"Training set: {len(train_df)} records, Test set: {len(test_df)} records")
            
            return {
                'processed': df,
                'train': train_df,
                'test': test_df,
                'feature_cols': feature_cols,
                'train_output': train_output,
                'test_output': test_output
            }
            
        except Exception as e:
            logger.error(f"Error processing arbitrage data from {file_path}: {e}")
            return {
                'processed': None,
                'train': None,
                'test': None
            }
    
    def run(self) -> Dict[str, Any]:
        """
        Run the full data processing pipeline
        
        Returns:
            Dictionary with processing results
        """
        logger.info("Starting data processing")
        
        results = {
            'ohlcv': {
                'processed_files': [],
                'lstm_data': {}
            },
            'arbitrage': {
                'direct': {},
                'triangular': {}
            }
        }
        
        # Process OHLCV data
        ohlcv_files = glob.glob(os.path.join(self.input_dir, '*_1h.csv'))
        
        for file_path in ohlcv_files:
            try:
                filename = os.path.basename(file_path)
                symbol = filename.replace('_1h.csv', '')
                
                output_path = os.path.join(self.output_dir, filename)
                
                # Process OHLCV file
                df = self.process_ohlcv_data(file_path, output_path)
                
                if not df.empty:
                    results['ohlcv']['processed_files'].append(output_path)
                    
                    # Create LSTM dataset
                    lstm_data = self.create_lstm_dataset(df, sequence_length=24, symbol=symbol)
                    
                    if lstm_data:
                        results['ohlcv']['lstm_data'][symbol] = {
                            'X_train_shape': lstm_data['X_train'].shape,
                            'y_train_shape': lstm_data['y_train'].shape,
                            'X_test_shape': lstm_data['X_test'].shape,
                            'y_test_shape': lstm_data['y_test'].shape,
                            'num_features': len(lstm_data['feature_columns']),
                            'num_targets': len(lstm_data['target_columns'])
                        }
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        # Process direct arbitrage data
        direct_arb_path = os.path.join(self.input_dir, 'direct_arbitrage_opportunities.csv')
        if os.path.exists(direct_arb_path):
            direct_results = self.process_arbitrage_data(direct_arb_path, 'direct')
            results['arbitrage']['direct'] = direct_results
        
        # Process triangular arbitrage data
        tri_arb_path = os.path.join(self.input_dir, 'triangular_arbitrage_opportunities.csv')
        if os.path.exists(tri_arb_path):
            tri_results = self.process_arbitrage_data(tri_arb_path, 'triangular')
            results['arbitrage']['triangular'] = tri_results
        
        logger.info("Data processing completed")
        
        # Save processing summary
        summary = {
            'processed_files': len(results['ohlcv']['processed_files']),
            'lstm_datasets': len(results['ohlcv']['lstm_data']),
            'direct_arbitrage': results['arbitrage']['direct'].get('processed') is not None,
            'triangular_arbitrage': results['arbitrage']['triangular'].get('processed') is not None,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_path = os.path.join(self.output_dir, 'processing_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved processing summary to {summary_path}")
        
        return results

def get_data_processor(input_dir: str = 'data', output_dir: str = 'data/processed') -> DataProcessor:
    """
    Get data processor instance
    
    Args:
        input_dir: Directory with raw data
        output_dir: Directory to save processed data
        
    Returns:
        Data processor instance
    """
    return DataProcessor(input_dir, output_dir)

if __name__ == "__main__":
    # Test the data processor
    processor = get_data_processor()
    results = processor.run()