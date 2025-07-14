import os
import time
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from threading import Lock
import requests

from web3 import Web3
from web3.providers import HTTPProvider
from web3.exceptions import TransactionNotFound

# Initialize logging
logger = logging.getLogger(__name__)

# Cache for web3 connections and token prices
_web3_instances = {}
_web3_locks = {}
_token_price_cache = {}
_token_price_timestamp = {}
_price_cache_max_age = 60  # in seconds

def get_web3_provider(network: str = 'mainnet', infura_key: str = None) -> Optional[Web3]:
    """
    Get a Web3 instance for the specified network.
    Caches instances for reuse.
    
    Args:
        network: Network name ('mainnet', 'goerli', 'sepolia', etc.)
        infura_key: Infura API key (will use env var if not provided)
        
    Returns:
        Web3 instance or None if connection failed
    """
    global _web3_instances, _web3_locks
    
    if network not in _web3_locks:
        _web3_locks[network] = Lock()
        
    # Check cache first (without lock for speed)
    if network in _web3_instances and _web3_instances[network]:
        if _web3_instances[network].is_connected():
            return _web3_instances[network]
    
    # Acquire lock to prevent multiple threads from creating instances
    with _web3_locks[network]:
        # Check again in case another thread created it while we were waiting
        if network in _web3_instances and _web3_instances[network]:
            if _web3_instances[network].is_connected():
                return _web3_instances[network]
        
        # Get Infura key from args or env
        if not infura_key:
            infura_key = os.environ.get('INFURA_API_KEY')
            
        if not infura_key:
            logger.error(f"Cannot connect to {network}: No Infura API key provided")
            return None
            
        # Create new web3 instance
        try:
            endpoint = f"https://{network}.infura.io/v3/{infura_key}"
            provider = HTTPProvider(endpoint)
            w3 = Web3(provider)
            
            if w3.is_connected():
                logger.info(f"Connected to Ethereum {network} network via Infura")
                _web3_instances[network] = w3
                return w3
            else:
                logger.error(f"Failed to connect to {network} via Infura")
                
                # Try fallback to ChainStack if env var is present
                chainstack_url = os.environ.get('CHAINSTACK_URL')
                if chainstack_url:
                    try:
                        provider = HTTPProvider(chainstack_url)
                        w3 = Web3(provider)
                        
                        if w3.is_connected():
                            logger.info(f"Connected to Ethereum {network} network via ChainStack (fallback)")
                            _web3_instances[network] = w3
                            return w3
                        else:
                            logger.error(f"Failed to connect to {network} via ChainStack fallback")
                    except Exception as e:
                        logger.error(f"Error connecting to ChainStack: {e}")
                
        except Exception as e:
            logger.error(f"Error connecting to {network}: {e}")
            
    return None

def get_token_price(token_address: str) -> Optional[float]:
    """
    Get the current USD price of a token from CoinGecko
    Caches results for performance
    
    Args:
        token_address: Token contract address
        
    Returns:
        Token price in USD or None if unavailable
    """
    global _token_price_cache, _token_price_timestamp
    
    if not token_address:
        return None
        
    # Normalize address
    token_address = token_address.lower()
    
    # Check cache
    current_time = time.time()
    if token_address in _token_price_cache and token_address in _token_price_timestamp:
        if current_time - _token_price_timestamp[token_address] < _price_cache_max_age:
            return _token_price_cache[token_address]
    
    # WETH special case (use ETH price)
    if token_address.lower() == "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2":
        return get_eth_price()
    
    # Try to get from CoinGecko
    try:
        url = f"https://api.coingecko.com/api/v3/simple/token_price/ethereum?contract_addresses={token_address}&vs_currencies=usd"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if token_address in data and 'usd' in data[token_address]:
                price = float(data[token_address]['usd'])
                
                # Cache result
                _token_price_cache[token_address] = price
                _token_price_timestamp[token_address] = current_time
                
                return price
    except Exception as e:
        logger.error(f"Error getting token price for {token_address}: {e}")
    
    # Return cached value even if it's old, or None if not in cache
    return _token_price_cache.get(token_address)

def get_eth_price_usd() -> Optional[float]:
    """
    Get the current price of ETH in USD
    Alias for get_eth_price for backwards compatibility
    
    Returns:
        ETH price in USD or None if unavailable
    """
    return get_eth_price()

def get_eth_price() -> Optional[float]:
    """
    Get the current USD price of ETH from CoinGecko
    Caches results for performance
    
    Returns:
        ETH price in USD or None if unavailable
    """
    global _token_price_cache, _token_price_timestamp
    
    # Use special key for ETH
    eth_key = "ethereum"
    
    # Check cache
    current_time = time.time()
    if eth_key in _token_price_cache and eth_key in _token_price_timestamp:
        if current_time - _token_price_timestamp[eth_key] < _price_cache_max_age:
            return _token_price_cache[eth_key]
    
    # Try to get from CoinGecko
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if 'ethereum' in data and 'usd' in data['ethereum']:
                price = float(data['ethereum']['usd'])
                
                # Cache result
                _token_price_cache[eth_key] = price
                _token_price_timestamp[eth_key] = current_time
                
                return price
    except Exception as e:
        logger.error(f"Error getting ETH price: {e}")
    
    # Return cached value even if it's old, or None if not in cache
    return _token_price_cache.get(eth_key)

def get_web3(network: str = 'mainnet', infura_key: str = None) -> Optional[Web3]:
    """
    Alias for get_web3_provider to maintain backward compatibility.
    
    Args:
        network: Network name ('mainnet', 'goerli', 'sepolia', etc.)
        infura_key: Infura API key (will use env var if not provided)
        
    Returns:
        Web3 instance or None if connection failed
    """
    return get_web3_provider(network, infura_key)

def get_gas_price(web3: Optional[Web3] = None) -> Tuple[int, int, int]:
    """
    Get current gas prices (standard, fast, fastest)
    
    Args:
        web3: Web3 instance (will create one if None)
        
    Returns:
        Tuple of (standard, fast, fastest) gas prices in wei
    """
    if web3 is None:
        web3 = get_web3_provider()
        
    if not web3 or not web3.is_connected():
        logger.error("Cannot get gas price: No web3 connection")
        return (20 * 10**9, 25 * 10**9, 30 * 10**9)  # Default values
        
    try:
        # Get current gas price from the node
        base_fee = web3.eth.get_block('latest').baseFeePerGas
        
        # Calculate prices based on base fee
        standard = int(base_fee * 1.1)  # 10% premium
        fast = int(base_fee * 1.2)      # 20% premium
        fastest = int(base_fee * 1.5)   # 50% premium
        
        return (standard, fast, fastest)
    except Exception as e:
        logger.error(f"Error getting gas price: {e}")
        return (20 * 10**9, 25 * 10**9, 30 * 10**9)  # Default values

def get_block_info(block_number: Union[int, str] = 'latest', web3: Optional[Web3] = None) -> Optional[Dict]:
    """
    Get block information
    
    Args:
        block_number: Block number or 'latest'
        web3: Web3 instance (will create one if None)
        
    Returns:
        Block info dictionary or None on error
    """
    if web3 is None:
        web3 = get_web3_provider()
        
    if not web3 or not web3.is_connected():
        logger.error("Cannot get block info: No web3 connection")
        return None
        
    try:
        # Get block
        block = web3.eth.get_block(block_number, full_transactions=False)
        
        # Convert to dictionary
        block_dict = dict(block)
        
        # Convert bytes to hex
        for key, value in block_dict.items():
            if isinstance(value, bytes):
                block_dict[key] = value.hex()
                
        return block_dict
    except Exception as e:
        logger.error(f"Error getting block info for {block_number}: {e}")
        return None

def get_transaction_info(tx_hash: str, web3: Optional[Web3] = None) -> Optional[Dict]:
    """
    Get transaction information
    
    Args:
        tx_hash: Transaction hash
        web3: Web3 instance (will create one if None)
        
    Returns:
        Transaction info dictionary or None on error
    """
    if web3 is None:
        web3 = get_web3_provider()
        
    if not web3 or not web3.is_connected():
        logger.error("Cannot get transaction info: No web3 connection")
        return None
        
    try:
        # Get transaction
        tx = web3.eth.get_transaction(tx_hash)
        
        # Convert to dictionary
        tx_dict = dict(tx)
        
        # Convert bytes to hex
        for key, value in tx_dict.items():
            if isinstance(value, bytes):
                tx_dict[key] = value.hex()
                
        # Get transaction receipt (for gas used, status, etc.)
        try:
            receipt = web3.eth.get_transaction_receipt(tx_hash)
            receipt_dict = dict(receipt)
            
            # Convert bytes to hex
            for key, value in receipt_dict.items():
                if isinstance(value, bytes):
                    receipt_dict[key] = value.hex()
                    
            tx_dict['receipt'] = receipt_dict
        except TransactionNotFound:
            # Transaction not yet confirmed
            tx_dict['receipt'] = None
            
        return tx_dict
    except Exception as e:
        logger.error(f"Error getting transaction info for {tx_hash}: {e}")
        return None

def get_contract_abi(address: str, web3: Optional[Web3] = None) -> Optional[List]:
    """
    Get contract ABI from Etherscan API
    
    Args:
        address: Contract address
        web3: Web3 instance (will create one if None)
        
    Returns:
        Contract ABI as list or None on error
    """
    if not address:
        return None
        
    # Check cache in data/abis directory
    cache_dir = "data/abis"
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"{address.lower()}.json")
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cached ABI for {address}: {e}")
    
    # If not in cache, try to get from Etherscan
    etherscan_api_key = os.environ.get('ETHERSCAN_API_KEY')
    if not etherscan_api_key:
        logger.error("Cannot get contract ABI: No Etherscan API key")
        return None
        
    try:
        url = f"https://api.etherscan.io/api?module=contract&action=getabi&address={address}&apikey={etherscan_api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == '1' and 'result' in data:
                abi = json.loads(data['result'])
                
                # Cache the ABI
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(abi, f)
                except Exception as e:
                    logger.error(f"Error caching ABI for {address}: {e}")
                
                return abi
            else:
                logger.error(f"Etherscan API error: {data.get('message', 'unknown error')}")
    except Exception as e:
        logger.error(f"Error getting contract ABI for {address}: {e}")
        
    return None
