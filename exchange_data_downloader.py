"""
Exchange data downloader - downloads historical price/volume data 
from crypto exchanges for the last 7 days
"""

import os
import time
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import ccxt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExchangeDataDownloader:
    """Class for downloading market data from exchanges"""
    
    def __init__(self, days_to_fetch: int = 7):
        self.days_to_fetch = days_to_fetch
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Initialize exchanges
        self.exchanges = {
            'binance': ccxt.binance({'enableRateLimit': True}),
            'kraken': ccxt.kraken({'enableRateLimit': True}),
            'coinbase': ccxt.coinbase({'enableRateLimit': True}),
            'kucoin': ccxt.kucoin({'enableRateLimit': True})
        }
        
        # Top trading pairs to fetch
        self.pairs = {
            'binance': ['ETH/USDT', 'BTC/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 
                        'ADA/USDT', 'AVAX/USDT', 'MATIC/USDT', 'DOT/USDT', 'LINK/USDT'],
            'kraken': ['ETH/USD', 'BTC/USD', 'XRP/USD', 'SOL/USD', 'ADA/USD'],
            'coinbase': ['ETH/USD', 'BTC/USD', 'SOL/USD', 'XRP/USD', 'ADA/USD'],
            'kucoin': ['ETH/USDT', 'BTC/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT']
        }
    
    def download_ohlcv_data(self, exchange_name: str, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """
        Download OHLCV (Open, High, Low, Close, Volume) data from exchange
        
        Args:
            exchange_name: Name of exchange (binance, kraken, etc.)
            symbol: Trading pair symbol (ETH/USDT, BTC/USD, etc.)
            timeframe: Timeframe for data (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        if exchange_name not in self.exchanges:
            logger.error(f"Exchange {exchange_name} not supported")
            return pd.DataFrame()
            
        exchange = self.exchanges[exchange_name]
        
        if not exchange.has['fetchOHLCV']:
            logger.error(f"Exchange {exchange_name} does not support OHLCV data")
            return pd.DataFrame()
            
        try:
            # Calculate start time
            end_time = int(time.time() * 1000)  # Current time in milliseconds
            start_time = end_time - (self.days_to_fetch * 24 * 60 * 60 * 1000)  # days_to_fetch days ago
            
            logger.info(f"Downloading {timeframe} OHLCV data for {symbol} from {exchange_name}")
            
            # Initialize an empty list for all the OHLCV data
            all_ohlcv = []
            
            # Some exchanges limit the number of candles per request
            # Fetch in batches if necessary
            since = start_time
            
            while since < end_time:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                
                # Update since to the timestamp of the last candle plus 1
                since = ohlcv[-1][0] + 1
                
                # Add a small delay to avoid rate limits
                time.sleep(exchange.rateLimit / 1000)
                
                # If we got less than requested, we're at the end
                if len(ohlcv) < 100:  # Most exchanges return max 100 candles per request
                    break
                    
            # Convert to DataFrame
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Ensure the data is sorted
            df = df.sort_values('timestamp')
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp'])
            
            # Save to CSV
            symbol_file = symbol.replace('/', '_')
            filename = f'data/{exchange_name}_{symbol_file}_{timeframe}.csv'
            df.to_csv(filename, index=False)
            
            logger.info(f"Saved {len(df)} records to {filename}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol} from {exchange_name}: {e}")
            return pd.DataFrame()
    
    def download_orderbook_data(self, exchange_name: str, symbol: str, depth: int = 20) -> List[Dict]:
        """
        Download orderbook data from exchange
        
        Args:
            exchange_name: Name of exchange (binance, kraken, etc.)
            symbol: Trading pair symbol (ETH/USDT, BTC/USD, etc.)
            depth: Depth of orderbook to fetch
            
        Returns:
            List of orderbook snapshots
        """
        if exchange_name not in self.exchanges:
            logger.error(f"Exchange {exchange_name} not supported")
            return []
            
        exchange = self.exchanges[exchange_name]
        
        if not exchange.has['fetchOrderBook']:
            logger.error(f"Exchange {exchange_name} does not support orderbook data")
            return []
            
        try:
            logger.info(f"Downloading orderbook data for {symbol} from {exchange_name}")
            
            # Take several snapshots over time to get a representative sample
            snapshots = []
            num_snapshots = 10  # Number of snapshots to take
            interval = (24 * 60 * 60) // num_snapshots  # Interval between snapshots (in seconds)
            
            for i in range(num_snapshots):
                orderbook = exchange.fetch_order_book(symbol, depth)
                
                # Add timestamp
                orderbook['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Store snapshot
                snapshots.append(orderbook)
                
                # Wait for next interval
                if i < num_snapshots - 1:
                    time.sleep(5)  # Sleep for 5 seconds between snapshots
            
            # Save to JSON
            symbol_file = symbol.replace('/', '_')
            filename = f'data/{exchange_name}_{symbol_file}_orderbook.json'
            
            with open(filename, 'w') as f:
                json.dump(snapshots, f, indent=2, default=str)
                
            logger.info(f"Saved {len(snapshots)} orderbook snapshots to {filename}")
            
            return snapshots
            
        except Exception as e:
            logger.error(f"Error downloading orderbook data for {symbol} from {exchange_name}: {e}")
            return []
    
    def download_all_data(self, include_orderbooks: bool = False) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Download all data for configured pairs and exchanges
        
        Args:
            include_orderbooks: Whether to include orderbook data
            
        Returns:
            Dictionary of DataFrames by exchange and symbol
        """
        # Dictionary to hold all DataFrames
        all_data = {}
        
        # Download OHLCV data for all pairs
        for exchange_name, symbols in self.pairs.items():
            logger.info(f"Processing {exchange_name} data")
            
            all_data[exchange_name] = {}
            
            for symbol in symbols:
                # Get different timeframes
                timeframes = ['1h', '15m', '5m', '1m']
                
                for timeframe in timeframes:
                    df = self.download_ohlcv_data(exchange_name, symbol, timeframe)
                    
                    if not df.empty:
                        all_data[exchange_name][f"{symbol}_{timeframe}"] = df
                
                # Get orderbook data if requested
                if include_orderbooks:
                    snapshots = self.download_orderbook_data(exchange_name, symbol)
                    
                    if snapshots:
                        # Store in a different format since it's not a DataFrame
                        all_data[exchange_name][f"{symbol}_orderbook"] = snapshots
        
        return all_data
        
    def create_triangular_arbitrage_dataset(self) -> pd.DataFrame:
        """
        Create a dataset for triangular arbitrage from downloaded data
        
        Returns:
            DataFrame with potential triangular arbitrage opportunities
        """
        logger.info("Creating triangular arbitrage dataset")
        
        # Find all files with 1m timeframe data
        files = []
        for exchange_name in self.exchanges.keys():
            for filename in os.listdir('data'):
                if filename.startswith(exchange_name) and filename.endswith('_1m.csv'):
                    files.append(os.path.join('data', filename))
        
        if not files:
            logger.warning("No data files found for triangular arbitrage analysis")
            return pd.DataFrame()
            
        # Gather price data
        exchange_data = {}
        
        for file in files:
            try:
                df = pd.read_csv(file)
                
                # Extract exchange and symbol from filename
                parts = os.path.basename(file).split('_')
                exchange = parts[0]
                symbol = '_'.join(parts[1:-1])  # Symbol might contain '_' (e.g., BTC_USDT)
                
                if exchange not in exchange_data:
                    exchange_data[exchange] = {}
                    
                exchange_data[exchange][symbol] = df
                
            except Exception as e:
                logger.error(f"Error loading data from {file}: {e}")
        
        # Create triangular arbitrage dataset
        triangular_data = []
        
        # Look for triangular arbitrage within each exchange
        for exchange, symbols_data in exchange_data.items():
            # Find all unique symbols
            all_symbols = list(symbols_data.keys())
            
            # Find potential triangular paths
            for symbol1 in all_symbols:
                base1, quote1 = symbol1.split('_')
                
                for symbol2 in all_symbols:
                    if symbol2 == symbol1:
                        continue
                        
                    base2, quote2 = symbol2.split('_')
                    
                    # Check if they share a common currency
                    if quote1 == base2:
                        # Look for a third symbol to complete the triangle
                        for symbol3 in all_symbols:
                            if symbol3 == symbol1 or symbol3 == symbol2:
                                continue
                                
                            base3, quote3 = symbol3.split('_')
                            
                            if base3 == quote2 and quote3 == base1:
                                # Found a triangular path: base1/quote1 -> base2/quote2 -> base3/quote3
                                logger.info(f"Found triangular path in {exchange}: {base1}/{quote1} -> {base2}/{quote2} -> {base3}/{quote3}")
                                
                                # Merge data on timestamp
                                df1 = symbols_data[symbol1].copy()
                                df2 = symbols_data[symbol2].copy()
                                df3 = symbols_data[symbol3].copy()
                                
                                # Make sure we have datetime
                                if 'timestamp' in df1.columns and not pd.api.types.is_datetime64_any_dtype(df1['timestamp']):
                                    df1['timestamp'] = pd.to_datetime(df1['timestamp'])
                                if 'timestamp' in df2.columns and not pd.api.types.is_datetime64_any_dtype(df2['timestamp']):
                                    df2['timestamp'] = pd.to_datetime(df2['timestamp'])
                                if 'timestamp' in df3.columns and not pd.api.types.is_datetime64_any_dtype(df3['timestamp']):
                                    df3['timestamp'] = pd.to_datetime(df3['timestamp'])
                                
                                # Rename columns to avoid conflicts
                                df1 = df1.rename(columns={
                                    'open': f'open_{symbol1}',
                                    'high': f'high_{symbol1}',
                                    'low': f'low_{symbol1}',
                                    'close': f'close_{symbol1}',
                                    'volume': f'volume_{symbol1}'
                                })
                                
                                df2 = df2.rename(columns={
                                    'open': f'open_{symbol2}',
                                    'high': f'high_{symbol2}',
                                    'low': f'low_{symbol2}',
                                    'close': f'close_{symbol2}',
                                    'volume': f'volume_{symbol2}'
                                })
                                
                                df3 = df3.rename(columns={
                                    'open': f'open_{symbol3}',
                                    'high': f'high_{symbol3}',
                                    'low': f'low_{symbol3}',
                                    'close': f'close_{symbol3}',
                                    'volume': f'volume_{symbol3}'
                                })
                                
                                # Merge based on closest timestamp
                                df_merged = pd.merge_asof(
                                    df1.sort_values('timestamp'),
                                    df2.sort_values('timestamp'),
                                    on='timestamp',
                                    direction='nearest'
                                )
                                
                                df_merged = pd.merge_asof(
                                    df_merged.sort_values('timestamp'),
                                    df3.sort_values('timestamp'),
                                    on='timestamp',
                                    direction='nearest'
                                )
                                
                                # Calculate potential triangular arbitrage
                                df_merged['path'] = f"{base1}/{quote1}->{base2}/{quote2}->{base3}/{quote3}"
                                df_merged['exchange'] = exchange
                                
                                # Calculate theoretical rates
                                # For path A->B->C->A
                                # Rate1: How much B per A
                                # Rate2: How much C per B
                                # Rate3: How much A per C
                                df_merged['rate1'] = df_merged[f'close_{symbol1}']
                                df_merged['rate2'] = df_merged[f'close_{symbol2}']
                                df_merged['rate3'] = df_merged[f'close_{symbol3}']
                                
                                # Calculate profit ratio
                                # If we start with 1 A, we get rate1 B
                                # Then rate1 * rate2 C
                                # Then rate1 * rate2 * rate3 A
                                # Profit ratio: (final_amount / initial_amount) - 1
                                df_merged['profit_ratio'] = (df_merged['rate1'] * df_merged['rate2'] * df_merged['rate3']) - 1
                                
                                # Store data
                                triangular_data.append(df_merged)
        
        # Combine all triangular data
        if triangular_data:
            result = pd.concat(triangular_data, ignore_index=True)
            
            # Save to CSV
            result.to_csv('data/triangular_arbitrage_opportunities.csv', index=False)
            logger.info(f"Saved {len(result)} triangular arbitrage opportunities to data/triangular_arbitrage_opportunities.csv")
            
            return result
        else:
            logger.warning("No triangular arbitrage opportunities found")
            return pd.DataFrame()
    
    def create_direct_arbitrage_dataset(self) -> pd.DataFrame:
        """
        Create a dataset for direct arbitrage from downloaded data
        
        Returns:
            DataFrame with potential direct arbitrage opportunities
        """
        logger.info("Creating direct arbitrage dataset")
        
        # Find all files with 1m timeframe data
        files = []
        for exchange_name in self.exchanges.keys():
            for filename in os.listdir('data'):
                if filename.startswith(exchange_name) and filename.endswith('_1m.csv'):
                    files.append(os.path.join('data', filename))
        
        if not files:
            logger.warning("No data files found for direct arbitrage analysis")
            return pd.DataFrame()
            
        # Gather price data
        symbol_data = {}
        
        for file in files:
            try:
                df = pd.read_csv(file)
                
                # Extract exchange and symbol from filename
                parts = os.path.basename(file).split('_')
                exchange = parts[0]
                symbol = '_'.join(parts[1:-1])  # Symbol might contain '_' (e.g., BTC_USDT)
                
                if symbol not in symbol_data:
                    symbol_data[symbol] = {}
                    
                symbol_data[symbol][exchange] = df
                
            except Exception as e:
                logger.error(f"Error loading data from {file}: {e}")
        
        # Create direct arbitrage dataset
        arbitrage_data = []
        
        # Look for direct arbitrage opportunities across exchanges
        for symbol, exchanges_data in symbol_data.items():
            # Need at least 2 exchanges for arbitrage
            if len(exchanges_data) < 2:
                continue
                
            exchanges = list(exchanges_data.keys())
            
            # Compare each pair of exchanges
            for i in range(len(exchanges)):
                for j in range(i+1, len(exchanges)):
                    exchange1 = exchanges[i]
                    exchange2 = exchanges[j]
                    
                    logger.info(f"Analyzing {symbol} on {exchange1} vs {exchange2}")
                    
                    df1 = exchanges_data[exchange1].copy()
                    df2 = exchanges_data[exchange2].copy()
                    
                    # Make sure we have datetime
                    if 'timestamp' in df1.columns and not pd.api.types.is_datetime64_any_dtype(df1['timestamp']):
                        df1['timestamp'] = pd.to_datetime(df1['timestamp'])
                    if 'timestamp' in df2.columns and not pd.api.types.is_datetime64_any_dtype(df2['timestamp']):
                        df2['timestamp'] = pd.to_datetime(df2['timestamp'])
                    
                    # Rename columns to avoid conflicts
                    df1 = df1.rename(columns={
                        'open': f'open_{exchange1}',
                        'high': f'high_{exchange1}',
                        'low': f'low_{exchange1}',
                        'close': f'close_{exchange1}',
                        'volume': f'volume_{exchange1}'
                    })
                    
                    df2 = df2.rename(columns={
                        'open': f'open_{exchange2}',
                        'high': f'high_{exchange2}',
                        'low': f'low_{exchange2}',
                        'close': f'close_{exchange2}',
                        'volume': f'volume_{exchange2}'
                    })
                    
                    # Merge based on closest timestamp
                    df_merged = pd.merge_asof(
                        df1.sort_values('timestamp'),
                        df2.sort_values('timestamp'),
                        on='timestamp',
                        direction='nearest'
                    )
                    
                    # Calculate price difference and potential profit
                    df_merged['symbol'] = symbol
                    df_merged['exchange1'] = exchange1
                    df_merged['exchange2'] = exchange2
                    df_merged['price1'] = df_merged[f'close_{exchange1}']
                    df_merged['price2'] = df_merged[f'close_{exchange2}']
                    df_merged['price_diff'] = df_merged['price2'] - df_merged['price1']
                    df_merged['price_diff_pct'] = df_merged['price_diff'] / df_merged['price1']
                    
                    # Store data
                    arbitrage_data.append(df_merged)
        
        # Combine all arbitrage data
        if arbitrage_data:
            result = pd.concat(arbitrage_data, ignore_index=True)
            
            # Save to CSV
            result.to_csv('data/direct_arbitrage_opportunities.csv', index=False)
            logger.info(f"Saved {len(result)} direct arbitrage opportunities to data/direct_arbitrage_opportunities.csv")
            
            return result
        else:
            logger.warning("No direct arbitrage opportunities found")
            return pd.DataFrame()
            
    def run(self, include_orderbooks: bool = False, create_arbitrage_datasets: bool = True) -> Dict:
        """
        Run the full data download pipeline
        
        Args:
            include_orderbooks: Whether to include orderbook data
            create_arbitrage_datasets: Whether to create arbitrage datasets
            
        Returns:
            Dictionary with downloaded data
        """
        logger.info("Starting exchange data download")
        
        result = {
            'ohlcv_data': None,
            'triangular_arbitrage': None,
            'direct_arbitrage': None
        }
        
        # Download all OHLCV data
        result['ohlcv_data'] = self.download_all_data(include_orderbooks)
        
        # Create arbitrage datasets if requested
        if create_arbitrage_datasets:
            result['triangular_arbitrage'] = self.create_triangular_arbitrage_dataset()
            result['direct_arbitrage'] = self.create_direct_arbitrage_dataset()
        
        logger.info("Exchange data download complete")
        
        return result


if __name__ == "__main__":
    # Example usage
    downloader = ExchangeDataDownloader(days_to_fetch=7)
    data = downloader.run(include_orderbooks=False, create_arbitrage_datasets=True)
    
    # Report on downloaded data
    if data['ohlcv_data']:
        total_files = 0
        for exchange, symbols_data in data['ohlcv_data'].items():
            total_files += len(symbols_data)
        print(f"Downloaded OHLCV data: {total_files} files across {len(data['ohlcv_data'])} exchanges")
        
    if data['triangular_arbitrage'] is not None:
        print(f"Triangular arbitrage opportunities: {len(data['triangular_arbitrage'])} records")
        
    if data['direct_arbitrage'] is not None:
        print(f"Direct arbitrage opportunities: {len(data['direct_arbitrage'])} records")