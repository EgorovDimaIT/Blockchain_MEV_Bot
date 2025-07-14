"""
Download financial and cryptocurrency data from various sources
"""

import pandas as pd
import numpy as np
import logging
import os
import time
import datetime
import requests
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class CryptoDataDownloader:
    """
    Class for downloading cryptocurrency data from various sources
    """
    
    def __init__(self):
        self.api_rate_limit_delay = 1.0  # seconds between API calls
        self.data_cache = {}
        self.last_api_call = 0
        
        # Configure API keys
        self.binance_api_key = os.environ.get('BINANCE_API_KEY', '')
        self.coingecko_api_key = os.environ.get('COINGECKO_API_KEY', '')
        self.cryptocompare_api_key = os.environ.get('CRYPTOCOMPARE_API_KEY', '')
        
        # Create cache directory
        os.makedirs('data', exist_ok=True)
    
    def _get_days_ago_timestamp(self, days: int) -> int:
        """
        Get Unix timestamp from days ago
        
        Args:
            days: Number of days ago
            
        Returns:
            Unix timestamp in milliseconds
        """
        now = datetime.datetime.now()
        days_ago = now - datetime.timedelta(days=days)
        return int(days_ago.timestamp() * 1000)
    
    def _respect_rate_limit(self):
        """
        Ensure we don't exceed API rate limits
        """
        current_time = time.time()
        elapsed = current_time - self.last_api_call
        
        if elapsed < self.api_rate_limit_delay:
            time.sleep(self.api_rate_limit_delay - elapsed)
            
        self.last_api_call = time.time()
    
    def download_historical_data(self, symbol: str, interval: str = '1h', days: int = 7, source: str = 'binance') -> pd.DataFrame:
        """
        Download historical price data
        
        Args:
            symbol: Trading pair symbol (e.g., 'ETHUSDT')
            interval: Time interval ('1m', '5m', '15m', '1h', '4h', '1d')
            days: Number of days of data to download
            source: Data source ('binance', 'coingecko', 'cryptocompare')
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        cache_key = f"{symbol}_{interval}_{days}_{source}"
        if cache_key in self.data_cache:
            logger.info(f"Using cached data for {cache_key}")
            return self.data_cache[cache_key]
        
        # Check if we have a csv file cached
        cache_file = f"data/{cache_key}.csv"
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                logger.info(f"Loaded data from cache file {cache_file}")
                return df
            except Exception as e:
                logger.error(f"Error loading cache file: {e}")
        
        # Select download method based on source
        try:
            if source.lower() == 'binance':
                df = self._download_from_binance(symbol, interval, days)
            elif source.lower() == 'coingecko':
                df = self._download_from_coingecko(symbol, days)
            elif source.lower() == 'cryptocompare':
                df = self._download_from_cryptocompare(symbol, interval, days)
            else:
                logger.error(f"Unknown data source: {source}")
                return pd.DataFrame()
                
            # Cache the data in memory
            if not df.empty:
                self.data_cache[cache_key] = df
                
                # Save to CSV for future use
                df_to_save = df.reset_index()
                df_to_save.to_csv(cache_file, index=False)
                logger.info(f"Saved {len(df)} records to {cache_file}")
                
            return df
                
        except Exception as e:
            logger.error(f"Error downloading data for {symbol} from {source}: {e}")
            return pd.DataFrame()
    
    def _download_from_binance(self, symbol: str, interval: str = '1h', days: int = 7) -> pd.DataFrame:
        """
        Download historical data from Binance API
        
        Args:
            symbol: Trading pair symbol (e.g., 'ETHUSDT')
            interval: Time interval ('1m', '5m', '15m', '1h', '4h', '1d')
            days: Number of days of data to download
            
        Returns:
            DataFrame with OHLCV data
        """
        self._respect_rate_limit()
        
        # Convert days to milliseconds
        start_time = self._get_days_ago_timestamp(days)
        
        # Intervals in milliseconds
        interval_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        
        # Calculate number of candles to request (max 1000 per request)
        ms_diff = int(time.time() * 1000) - start_time
        num_candles = min(1000, ms_diff // interval_ms.get(interval, 3600000))
        
        # Build URL
        url = "https://api.binance.com/api/v3/klines"
        
        # Headers
        headers = {}
        if self.binance_api_key:
            headers['X-MBX-APIKEY'] = self.binance_api_key
        
        # Parameters
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'startTime': start_time,
            'limit': 1000  # Max limit
        }
        
        try:
            logger.info(f"Downloading {symbol} data from Binance for past {days} days")
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Binance API error: {response.text}")
                return pd.DataFrame()
                
            data = response.json()
            
            if not data:
                logger.warning(f"No data returned from Binance for {symbol}")
                return pd.DataFrame()
                
            # Parse the data
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                       'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                       'taker_buy_quote_asset_volume', 'ignored']
                       
            df = pd.DataFrame(data, columns=columns)
            
            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                              'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
                              
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
                
            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Drop unnecessary columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Downloaded {len(df)} candles for {symbol} from Binance")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading from Binance: {e}")
            return pd.DataFrame()
    
    def _download_from_coingecko(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """
        Download historical data from CoinGecko API
        
        Args:
            symbol: Trading pair symbol (e.g., 'ethereum', 'bitcoin')
            days: Number of days of data to download
            
        Returns:
            DataFrame with OHLCV data
        """
        self._respect_rate_limit()
        
        # Convert symbol to CoinGecko format
        symbol_parts = symbol.upper().split('USD')
        coin_id = None
        
        # Common mappings
        coin_map = {
            'ETH': 'ethereum',
            'BTC': 'bitcoin',
            'LINK': 'chainlink',
            'AAVE': 'aave',
            'UNI': 'uniswap',
            'MKR': 'maker',
            '1INCH': '1inch',
            'USDC': 'usd-coin',
            'USDT': 'tether',
            'DAI': 'dai',
            'WBTC': 'wrapped-bitcoin'
        }
        
        if len(symbol_parts) > 0 and symbol_parts[0] in coin_map:
            coin_id = coin_map[symbol_parts[0]]
        
        if not coin_id:
            # Try to extract from the symbol
            if 'ETH' in symbol:
                coin_id = 'ethereum'
            elif 'BTC' in symbol:
                coin_id = 'bitcoin'
            elif 'LINK' in symbol:
                coin_id = 'chainlink'
            elif symbol.upper() == '1INCHUSD' or symbol.upper() == '1INCHUSDT':
                coin_id = '1inch'
            else:
                logger.error(f"Could not map symbol {symbol} to CoinGecko coin ID")
                return pd.DataFrame()
        
        # Build URL
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        
        # Parameters
        params = {
            'vs_currency': 'usd',
            'days': str(days),
            'interval': 'hourly'
        }
        
        # Add API key if available
        if self.coingecko_api_key:
            params['x_cg_pro_api_key'] = self.coingecko_api_key
        
        try:
            logger.info(f"Downloading {coin_id} data from CoinGecko for past {days} days")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"CoinGecko API error: {response.text}")
                return pd.DataFrame()
                
            data = response.json()
            
            if not data or 'prices' not in data:
                logger.warning(f"No price data returned from CoinGecko for {coin_id}")
                return pd.DataFrame()
                
            # Extract price and volume data
            prices = data['prices']
            volumes = data['total_volumes']
            
            # Create DataFrame
            df_prices = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            
            # Merge the DataFrames
            df = pd.merge(df_prices, df_volumes, on='timestamp')
            
            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add other OHLC columns (only close prices available)
            df['open'] = df['close'].shift(1)
            df['high'] = df['close']
            df['low'] = df['close']
            
            # Fill first row
            df['open'].iloc[0] = df['close'].iloc[0]
            
            # Reorder columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Downloaded {len(df)} prices for {coin_id} from CoinGecko")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading from CoinGecko: {e}")
            return pd.DataFrame()
    
    def _download_from_cryptocompare(self, symbol: str, interval: str = '1h', days: int = 7) -> pd.DataFrame:
        """
        Download historical data from CryptoCompare API
        
        Args:
            symbol: Trading pair symbol (e.g., 'ETH', 'BTC')
            interval: Time interval ('1m', '5m', '15m', '1h', '4h', '1d')
            days: Number of days of data to download
            
        Returns:
            DataFrame with OHLCV data
        """
        self._respect_rate_limit()
        
        # Parse symbol to get base and quote currencies
        base_currency = symbol.upper().replace('USDT', '').replace('USD', '').replace('BTC', '')
        if not base_currency:
            base_currency = 'BTC'
            
        # Determine quote currency
        if 'USDT' in symbol.upper():
            quote_currency = 'USDT'
        elif 'USD' in symbol.upper():
            quote_currency = 'USD'
        elif 'BTC' in symbol.upper() and base_currency != 'BTC':
            quote_currency = 'BTC'
        else:
            quote_currency = 'USD'  # Default
        
        # Convert interval to CryptoCompare format
        interval_map = {
            '1m': 'minute',
            '5m': 'minute',
            '15m': 'minute',
            '1h': 'hour',
            '4h': 'hour',
            '1d': 'day'
        }
        
        interval_multiplier = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 1,
            '4h': 4,
            '1d': 1
        }
        
        # Build URL
        frequency = interval_map.get(interval, 'hour')
        aggregation = interval_multiplier.get(interval, 1)
        
        if frequency == 'minute':
            url = f"https://min-api.cryptocompare.com/data/v2/histo{frequency}"
        else:
            url = f"https://min-api.cryptocompare.com/data/v2/histo{frequency}"
        
        # Calculate number of data points
        if frequency == 'minute':
            limit = min(2000, days * 24 * 60 // aggregation)
        elif frequency == 'hour':
            limit = min(2000, days * 24 // aggregation)
        else:
            limit = min(2000, days)
        
        # Parameters
        params = {
            'fsym': base_currency,
            'tsym': quote_currency,
            'limit': limit,
            'aggregate': aggregation
        }
        
        # Add API key if available
        if self.cryptocompare_api_key:
            params['api_key'] = self.cryptocompare_api_key
        
        try:
            logger.info(f"Downloading {base_currency}/{quote_currency} data from CryptoCompare")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"CryptoCompare API error: {response.text}")
                return pd.DataFrame()
                
            data = response.json()
            
            if not data or 'Data' not in data or 'Data' not in data['Data']:
                logger.warning(f"No data returned from CryptoCompare for {base_currency}/{quote_currency}")
                return pd.DataFrame()
                
            # Extract data
            candles = data['Data']['Data']
            
            if not candles:
                logger.warning(f"Empty data returned from CryptoCompare for {base_currency}/{quote_currency}")
                return pd.DataFrame()
                
            # Create DataFrame
            df = pd.DataFrame(candles)
            
            # Convert types
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volumefrom'] = df['volumefrom'].astype(float)
            
            # Rename volume column
            df['volume'] = df['volumefrom']
            
            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            # Select columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Downloaded {len(df)} candles for {base_currency}/{quote_currency} from CryptoCompare")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading from CryptoCompare: {e}")
            return pd.DataFrame()
        
# Singleton instance
_downloader = None

def get_data_downloader() -> CryptoDataDownloader:
    """
    Get singleton instance of data downloader
    
    Returns:
        CryptoDataDownloader instance
    """
    global _downloader
    if _downloader is None:
        _downloader = CryptoDataDownloader()
    return _downloader