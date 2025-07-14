"""
Contract data collector - extracts data from Ethereum blockchain
for a specific contract address
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
from web3 import Web3

from utils.web3_helpers import get_web3, get_contract_abi, get_eth_price, get_gas_price
from utils.token_utils import get_token_info, get_token_balance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContractDataCollector:
    """Class for collecting data from a specific contract"""
    
    def __init__(self, contract_address: str, days_to_fetch: int = 7):
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.days_to_fetch = days_to_fetch
        self.web3 = get_web3()
        self.contract_abi = get_contract_abi(self.contract_address)
        self.etherscan_api_key = os.environ.get('ETHERSCAN_API_KEY')
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        if not self.web3 or not self.web3.is_connected():
            logger.error("Cannot connect to Ethereum network")
            raise ConnectionError("Cannot connect to Ethereum network")
            
        # Initialize contract object if ABI is available
        if self.contract_abi:
            self.contract = self.web3.eth.contract(address=self.contract_address, abi=self.contract_abi)
        else:
            logger.warning(f"Contract ABI not available for {contract_address}, limited functionality")
            self.contract = None
    
    def get_basic_contract_info(self) -> Dict:
        """Get basic information about the contract"""
        
        info = {
            'address': self.contract_address,
            'creation_block': None,
            'creation_date': None,
            'creator_address': None,
            'balance_eth': None,
            'eth_price_usd': get_eth_price(),
            'transaction_count': None,
            'is_token': False,
            'token_info': None
        }
        
        try:
            # Get contract balance
            balance_wei = self.web3.eth.get_balance(self.contract_address)
            info['balance_eth'] = self.web3.from_wei(balance_wei, 'ether')
            
            # Get transaction count
            tx_count = self.web3.eth.get_transaction_count(self.contract_address)
            info['transaction_count'] = tx_count
            
            # Try to get token info if it's an ERC20 token
            token_info = get_token_info(self.contract_address)
            if token_info:
                info['is_token'] = True
                info['token_info'] = token_info
                
            # Get contract creation info from Etherscan
            if self.etherscan_api_key:
                url = f"https://api.etherscan.io/api?module=contract&action=getcontractcreation&contractaddresses={self.contract_address}&apikey={self.etherscan_api_key}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == '1' and data['result']:
                        creation_info = data['result'][0]
                        info['creator_address'] = creation_info['contractCreator']
                        
                        # Get transaction to find creation block and date
                        tx_hash = creation_info.get('txHash')
                        if tx_hash:
                            tx = self.web3.eth.get_transaction(tx_hash)
                            if tx and tx.blockNumber:
                                info['creation_block'] = tx.blockNumber
                                block = self.web3.eth.get_block(tx.blockNumber)
                                info['creation_date'] = datetime.fromtimestamp(block.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        except Exception as e:
            logger.error(f"Error getting contract info: {e}")
        
        return info
    
    def get_transaction_history(self) -> pd.DataFrame:
        """Get transaction history for the contract"""
        
        logger.info(f"Fetching transaction history for {self.contract_address}")
        
        # Calculate start time (days_to_fetch days ago)
        end_time = int(time.time())
        start_time = end_time - (self.days_to_fetch * 24 * 60 * 60)
        
        transactions = []
        
        if not self.etherscan_api_key:
            logger.warning("Etherscan API key not available, using only web3")
            
            # Get current block number
            latest_block = self.web3.eth.block_number
            
            # Estimate blocks to go back (avg 13s per block)
            blocks_per_day = 24 * 60 * 60 // 13  # ~6646 blocks per day
            start_block = max(0, latest_block - (blocks_per_day * self.days_to_fetch))
            
            # Get all recent logs involving this contract
            logs = []
            
            try:
                # Filter for all events from this contract
                event_filter = {
                    'fromBlock': start_block,
                    'toBlock': 'latest',
                    'address': self.contract_address
                }
                
                logs = self.web3.eth.get_logs(event_filter)
                logger.info(f"Found {len(logs)} logs for contract")
                
                # Process logs to extract transaction hashes
                tx_hashes = set()
                for log in logs:
                    tx_hashes.add(log.transactionHash.hex())
                
                # Get transaction details
                for tx_hash in tx_hashes:
                    try:
                        tx = self.web3.eth.get_transaction(tx_hash)
                        receipt = self.web3.eth.get_transaction_receipt(tx_hash)
                        
                        # Get block for timestamp
                        block = self.web3.eth.get_block(tx.blockNumber)
                        
                        tx_record = {
                            'hash': tx.hash.hex(),
                            'block_number': tx.blockNumber,
                            'timestamp': datetime.fromtimestamp(block.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                            'from': tx['from'],
                            'to': tx.to,
                            'value_eth': self.web3.from_wei(tx.value, 'ether'),
                            'gas_price_gwei': self.web3.from_wei(tx.gasPrice, 'gwei'),
                            'gas_used': receipt.gasUsed,
                            'status': receipt.status
                        }
                        
                        transactions.append(tx_record)
                    except Exception as e:
                        logger.error(f"Error processing transaction {tx_hash}: {e}")
                
            except Exception as e:
                logger.error(f"Error getting logs: {e}")
            
        else:
            # Use Etherscan API for more efficient data retrieval
            # Normal transactions
            try:
                url = f"https://api.etherscan.io/api?module=account&action=txlist&address={self.contract_address}&startblock=0&endblock=99999999&sort=desc&apikey={self.etherscan_api_key}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == '1' and 'result' in data:
                        for tx in data['result']:
                            timestamp = int(tx['timeStamp'])
                            if timestamp >= start_time:
                                tx_record = {
                                    'hash': tx['hash'],
                                    'block_number': int(tx['blockNumber']),
                                    'timestamp': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                                    'from': tx['from'],
                                    'to': tx['to'],
                                    'value_eth': float(self.web3.from_wei(int(tx['value']), 'ether')),
                                    'gas_price_gwei': float(self.web3.from_wei(int(tx['gasPrice']), 'gwei')),
                                    'gas_used': int(tx['gasUsed']),
                                    'status': 1 if tx['txreceipt_status'] == '1' else 0
                                }
                                transactions.append(tx_record)
            except Exception as e:
                logger.error(f"Error getting normal transactions from Etherscan: {e}")
            
            # Internal transactions
            try:
                url = f"https://api.etherscan.io/api?module=account&action=txlistinternal&address={self.contract_address}&startblock=0&endblock=99999999&sort=desc&apikey={self.etherscan_api_key}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == '1' and 'result' in data:
                        for tx in data['result']:
                            timestamp = int(tx['timeStamp'])
                            if timestamp >= start_time:
                                # Some internal txs don't have all fields
                                tx_record = {
                                    'hash': tx['hash'],
                                    'block_number': int(tx['blockNumber']),
                                    'timestamp': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                                    'from': tx['from'],
                                    'to': tx['to'],
                                    'value_eth': float(self.web3.from_wei(int(tx['value']), 'ether')),
                                    'gas_price_gwei': 0,  # Not always available for internal txs
                                    'gas_used': 0,  # Not always available for internal txs
                                    'status': 1,  # Assume success for internal txs
                                    'is_internal': True
                                }
                                transactions.append(tx_record)
            except Exception as e:
                logger.error(f"Error getting internal transactions from Etherscan: {e}")
            
            # Token transfers if it's a token contract
            if self.contract.get('is_token', False):
                try:
                    url = f"https://api.etherscan.io/api?module=account&action=tokentx&address={self.contract_address}&startblock=0&endblock=99999999&sort=desc&apikey={self.etherscan_api_key}"
                    response = requests.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data['status'] == '1' and 'result' in data:
                            for tx in data['result']:
                                timestamp = int(tx['timeStamp'])
                                if timestamp >= start_time:
                                    tx_record = {
                                        'hash': tx['hash'],
                                        'block_number': int(tx['blockNumber']),
                                        'timestamp': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                                        'from': tx['from'],
                                        'to': tx['to'],
                                        'value_eth': 0,  # Token transfers don't have ETH value
                                        'token_value': float(int(tx['value']) / (10 ** int(tx['tokenDecimal']))),
                                        'token_symbol': tx['tokenSymbol'],
                                        'gas_price_gwei': float(self.web3.from_wei(int(tx['gasPrice']), 'gwei')),
                                        'gas_used': int(tx['gasUsed']),
                                        'status': 1,  # Assume success for token transfers
                                        'is_token_transfer': True
                                    }
                                    transactions.append(tx_record)
                except Exception as e:
                    logger.error(f"Error getting token transfers from Etherscan: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Sort by timestamp
        if not df.empty and 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('datetime')
            
            # Calculate time between transactions
            df['seconds_since_prev'] = df['datetime'].diff().dt.total_seconds()
            df.loc[df['seconds_since_prev'].isna(), 'seconds_since_prev'] = 0
            
            # Save to CSV
            df.to_csv(f'data/{self.contract_address}_transactions.csv', index=False)
            logger.info(f"Saved {len(df)} transactions to data/{self.contract_address}_transactions.csv")
        else:
            logger.warning("No transactions found or missing timestamp column")
            # Create empty DataFrame with expected columns
            df = pd.DataFrame(columns=['hash', 'block_number', 'timestamp', 'datetime', 'from', 'to', 
                                      'value_eth', 'gas_price_gwei', 'gas_used', 'status', 'seconds_since_prev'])
        
        return df
        
    def get_contract_events(self) -> Dict[str, pd.DataFrame]:
        """Get events emitted by the contract"""
        
        logger.info(f"Fetching events for {self.contract_address}")
        event_dataframes = {}
        
        if not self.contract or not self.contract_abi:
            logger.warning("Contract ABI not available, cannot fetch specific events")
            return event_dataframes
        
        # Extract event definitions from ABI
        events = [item for item in self.contract_abi if item.get('type') == 'event']
        logger.info(f"Found {len(events)} event types in contract ABI")
        
        # Get current block number
        latest_block = self.web3.eth.block_number
        
        # Estimate blocks to go back (avg 13s per block)
        blocks_per_day = 24 * 60 * 60 // 13  # ~6646 blocks per day
        start_block = max(0, latest_block - (blocks_per_day * self.days_to_fetch))
        
        # Fetch each event type
        for event_def in events:
            event_name = event_def.get('name')
            if not event_name:
                continue
                
            logger.info(f"Fetching events of type {event_name}")
            
            try:
                # Get event object
                event_obj = getattr(self.contract.events, event_name)
                
                # Create filter
                event_filter = event_obj.create_filter(fromBlock=start_block, toBlock='latest')
                
                # Get all entries
                entries = event_filter.get_all_entries()
                logger.info(f"Found {len(entries)} {event_name} events")
                
                if entries:
                    # Process event data
                    event_data = []
                    for entry in entries:
                        # Basic event info
                        record = {
                            'transaction_hash': entry.transactionHash.hex(),
                            'block_number': entry.blockNumber,
                            'log_index': entry.logIndex
                        }
                        
                        # Add args
                        for arg_name, arg_value in dict(entry.args).items():
                            # Convert bytes to hex strings
                            if isinstance(arg_value, bytes):
                                record[arg_name] = arg_value.hex()
                            # Convert large integers to strings to avoid overflow
                            elif isinstance(arg_value, int) and (arg_value > 9223372036854775807 or arg_value < -9223372036854775808):
                                record[arg_name] = str(arg_value)
                            else:
                                record[arg_name] = arg_value
                        
                        # Get block timestamp
                        try:
                            block = self.web3.eth.get_block(entry.blockNumber)
                            record['timestamp'] = datetime.fromtimestamp(block.timestamp).strftime('%Y-%m-%d %H:%M:%S')
                        except Exception as e:
                            logger.error(f"Error getting block info: {e}")
                            record['timestamp'] = None
                            
                        event_data.append(record)
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(event_data)
                    
                    # Sort by block number and log index
                    if not df.empty and 'block_number' in df.columns and 'log_index' in df.columns:
                        df = df.sort_values(['block_number', 'log_index'])
                        
                        # Add datetime column
                        if 'timestamp' in df.columns:
                            df['datetime'] = pd.to_datetime(df['timestamp'])
                            
                        # Save to CSV
                        df.to_csv(f'data/{self.contract_address}_{event_name}_events.csv', index=False)
                        logger.info(f"Saved {len(df)} {event_name} events to data/{self.contract_address}_{event_name}_events.csv")
                        
                        # Store DataFrame
                        event_dataframes[event_name] = df
                    
            except Exception as e:
                logger.error(f"Error processing {event_name} events: {e}")
        
        return event_dataframes
        
    def get_price_data(self) -> pd.DataFrame:
        """
        Get price data if the contract is a token or interacts with major tokens
        """
        logger.info(f"Fetching price data related to {self.contract_address}")
        
        # Check if it's a token
        token_info = get_token_info(self.contract_address)
        
        if not token_info:
            logger.info(f"Contract {self.contract_address} is not a token, skipping price data")
            return pd.DataFrame()  # Return empty DataFrame
            
        # Get price data from CoinGecko
        try:
            symbol = token_info.get('symbol', '').lower()
            address = self.contract_address.lower()
            
            # Try to get price history from CoinGecko
            days = str(self.days_to_fetch)
            url = f"https://api.coingecko.com/api/v3/coins/ethereum/contract/{address}/market_chart/?vs_currency=usd&days={days}"
            
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Process price data
                if 'prices' in data:
                    prices = data['prices']  # List of [timestamp, price] pairs
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(prices, columns=['timestamp', 'price_usd'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Get additional data if available
                    if 'market_caps' in data:
                        market_caps = data['market_caps']  # List of [timestamp, market_cap] pairs
                        df_mc = pd.DataFrame(market_caps, columns=['timestamp', 'market_cap_usd'])
                        df_mc['timestamp'] = pd.to_datetime(df_mc['timestamp'], unit='ms')
                        
                        # Merge with price data
                        df = pd.merge_asof(df, df_mc, on='timestamp')
                        
                    if 'total_volumes' in data:
                        volumes = data['total_volumes']  # List of [timestamp, volume] pairs
                        df_vol = pd.DataFrame(volumes, columns=['timestamp', 'volume_usd'])
                        df_vol['timestamp'] = pd.to_datetime(df_vol['timestamp'], unit='ms')
                        
                        # Merge with price data
                        df = pd.merge_asof(df, df_vol, on='timestamp')
                    
                    # Save to CSV
                    df.to_csv(f'data/{address}_price_data.csv', index=False)
                    logger.info(f"Saved price data to data/{address}_price_data.csv")
                    
                    return df
                else:
                    logger.warning(f"No price data found for {symbol} ({address})")
            else:
                logger.warning(f"Error fetching price data: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error getting price data: {e}")
            
        return pd.DataFrame()  # Return empty DataFrame if no data found
    
    def collect_all_data(self) -> Dict:
        """Collect all data for the contract"""
        
        result = {
            'contract_info': None,
            'transaction_data': None,
            'event_data': None,
            'price_data': None
        }
        
        try:
            logger.info(f"Starting data collection for contract {self.contract_address}")
            
            # Get basic contract info
            result['contract_info'] = self.get_basic_contract_info()
            
            # Get transaction history
            result['transaction_data'] = self.get_transaction_history()
            
            # Get contract events
            result['event_data'] = self.get_contract_events()
            
            # Get price data if it's a token
            result['price_data'] = self.get_price_data()
            
            logger.info(f"Completed data collection for contract {self.contract_address}")
            
            # Save contract info to JSON
            with open(f'data/{self.contract_address}_info.json', 'w') as f:
                json.dump(result['contract_info'], f, indent=2, default=str)
                
            return result
            
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
            return result


if __name__ == "__main__":
    # Example usage
    contract_address = "0xE7abdBc456DacfED3653118ab3223320185B8662"  # Replace with actual contract address
    collector = ContractDataCollector(contract_address, days_to_fetch=7)
    data = collector.collect_all_data()
    
    print(f"Contract info: {data['contract_info']}")
    print(f"Transactions: {len(data['transaction_data'])} records")
    print(f"Events: {sum(len(df) for df in data['event_data'].values())} records across {len(data['event_data'])} event types")
    print(f"Price data: {len(data['price_data'])} records")