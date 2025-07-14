"""
Utilities for working with ERC20 tokens and DEX interactions
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from decimal import Decimal

from web3 import Web3
from web3.exceptions import ContractLogicError

from utils.web3_helpers import get_web3, get_contract_abi, get_eth_price_usd

# Initialize logging
logger = logging.getLogger(__name__)

# Common token addresses (Ethereum mainnet)
WETH_ADDRESS = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
DAI_ADDRESS = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
USDT_ADDRESS = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
WBTC_ADDRESS = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"

# DEX router addresses (Ethereum mainnet)
UNISWAP_V2_ROUTER = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
SUSHISWAP_ROUTER = "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F"
UNISWAP_V3_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"

# ABI snippets for common functions
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "payable": False,
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "payable": False,
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    }
]

# Cache for token info
_token_info_cache = {}

def get_token_info(token_address: str, web3: Optional[Web3] = None) -> Optional[Dict]:
    """
    Get token information (name, symbol, decimals)
    Caches results for performance
    
    Args:
        token_address: Token contract address
        web3: Web3 instance (will create one if None)
        
    Returns:
        Dictionary with token info or None on error
    """
    global _token_info_cache
    
    if not token_address:
        return None
        
    # Normalize address
    token_address = token_address.lower()
    
    # Check cache
    if token_address in _token_info_cache:
        return _token_info_cache[token_address]
    
    if web3 is None:
        web3 = get_web3()
        
    if not web3 or not web3.is_connected():
        logger.error("Cannot get token info: No web3 connection")
        return None
        
    try:
        # Use checksum address for contract
        checksum_address = web3.to_checksum_address(token_address)
        
        # Create contract instance
        contract = web3.eth.contract(address=checksum_address, abi=ERC20_ABI)
        
        # Get token info
        name = contract.functions.name().call()
        symbol = contract.functions.symbol().call()
        decimals = contract.functions.decimals().call()
        
        # Get USD price
        price_usd = None
        try:
            price_usd = get_token_price(token_address)
        except Exception as e:
            logger.warning(f"Error getting price for {token_address}: {e}")
        
        # Store in cache
        token_info = {
            'address': token_address,
            'name': name,
            'symbol': symbol,
            'decimals': decimals,
            'price_usd': price_usd
        }
        
        _token_info_cache[token_address] = token_info
        
        return token_info
    except Exception as e:
        logger.error(f"Error getting token info for {token_address}: {e}")
        return None
        
def get_token_balance(token_address: str, wallet_address: str, web3: Optional[Web3] = None) -> Optional[int]:
    """
    Get token balance for a wallet
    
    Args:
        token_address: Token contract address
        wallet_address: Wallet address
        web3: Web3 instance (will create one if None)
        
    Returns:
        Token balance in smallest units (wei, gwei, etc.) or None on error
    """
    if not token_address or not wallet_address:
        return None
        
    if web3 is None:
        web3 = get_web3()
        
    if not web3 or not web3.is_connected():
        logger.error("Cannot get token balance: No web3 connection")
        return None
        
    try:
        # Special case for ETH (not an ERC20 token)
        if token_address.lower() == "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" or token_address.lower() == "eth":
            balance = web3.eth.get_balance(wallet_address)
            return balance
            
        # Use checksum addresses
        checksum_token = web3.to_checksum_address(token_address)
        checksum_wallet = web3.to_checksum_address(wallet_address)
        
        # Create contract instance
        contract = web3.eth.contract(address=checksum_token, abi=ERC20_ABI)
        
        # Get balance
        balance = contract.functions.balanceOf(checksum_wallet).call()
        
        return balance
    except Exception as e:
        logger.error(f"Error getting balance of {token_address} for {wallet_address}: {e}")
        return None
        
def get_token_price(token_address: str) -> Optional[float]:
    """
    Get token price in USD
    
    Args:
        token_address: Token contract address
        
    Returns:
        Token price in USD or None if unavailable
    """
    from utils.web3_helpers import get_token_price as get_price
    
    try:
        return get_price(token_address)
    except Exception as e:
        logger.error(f"Error getting token price for {token_address}: {e}")
        return None
        
def estimate_token_swap_output(
    router_address: str,
    token_in: str,
    token_out: str,
    amount_in: int,
    slippage: float = 0.005,
    web3: Optional[Web3] = None
) -> Tuple[int, bool]:
    """
    Estimate output amount for a token swap on a DEX
    
    Args:
        router_address: DEX router address
        token_in: Input token address
        token_out: Output token address
        amount_in: Input amount in smallest units
        slippage: Slippage tolerance as decimal (e.g., 0.005 for 0.5%)
        web3: Web3 instance (will create one if None)
        
    Returns:
        Tuple of (estimated output amount, success flag)
    """
    if not router_address or not token_in or not token_out or amount_in <= 0:
        return 0, False
        
    if web3 is None:
        web3 = get_web3()
        
    if not web3 or not web3.is_connected():
        logger.error("Cannot estimate swap output: No web3 connection")
        return 0, False
        
    try:
        # Use checksum addresses
        router = web3.to_checksum_address(router_address)
        token_in = web3.to_checksum_address(token_in)
        token_out = web3.to_checksum_address(token_out)
        
        # Determine which router (Uniswap v2, Sushiswap, etc.)
        if router.lower() == UNISWAP_V2_ROUTER.lower() or router.lower() == SUSHISWAP_ROUTER.lower():
            # UniswapV2/Sushiswap style router
            router_abi = get_contract_abi(router)
            if not router_abi:
                # Use minimal ABI if can't get full ABI
                router_abi = [{
                    "inputs": [
                        {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                        {"internalType": "address[]", "name": "path", "type": "address[]"}
                    ],
                    "name": "getAmountsOut",
                    "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                    "stateMutability": "view",
                    "type": "function"
                }]
                
            router_contract = web3.eth.contract(address=router, abi=router_abi)
            
            # Path for the swap
            path = [token_in, token_out]
            
            # Try direct path first
            try:
                amounts = router_contract.functions.getAmountsOut(amount_in, path).call()
                amount_out = amounts[-1]
                return amount_out, True
            except ContractLogicError:
                # If direct path fails, try with WETH in the middle
                if token_in.lower() != WETH_ADDRESS.lower() and token_out.lower() != WETH_ADDRESS.lower():
                    try:
                        path = [token_in, WETH_ADDRESS, token_out]
                        amounts = router_contract.functions.getAmountsOut(amount_in, path).call()
                        amount_out = amounts[-1]
                        return amount_out, True
                    except Exception as e:
                        logger.error(f"Error estimating swap via WETH: {e}")
                        return 0, False
                else:
                    logger.error("Error estimating direct swap output")
                    return 0, False
                    
        elif router.lower() == UNISWAP_V3_ROUTER.lower():
            # UniswapV3 style router
            try:
                # Use Quoter contract to get quote
                quoter_address = '0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6'
                quoter_abi = [{
                    "inputs": [
                        {"internalType": "address", "name": "tokenIn", "type": "address"},
                        {"internalType": "address", "name": "tokenOut", "type": "address"},
                        {"internalType": "uint24", "name": "fee", "type": "uint24"},
                        {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                        {"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"}
                    ],
                    "name": "quoteExactInputSingle",
                    "outputs": [{"internalType": "uint256", "name": "amountOut", "type": "uint256"}],
                    "stateMutability": "nonpayable",
                    "type": "function"
                }]
                
                quoter_contract = web3.eth.contract(
                    address=web3.to_checksum_address(quoter_address),
                    abi=quoter_abi
                )
                
                # Try different fee tiers (most common first)
                for fee in [3000, 500, 10000]:
                    try:
                        amount_out = quoter_contract.functions.quoteExactInputSingle(
                            token_in,
                            token_out,
                            fee,
                            amount_in,
                            0  # No price limit
                        ).call()
                        return amount_out, True
                    except ContractLogicError:
                        continue  # Try next fee tier
                
                logger.error("Failed to get quote for any fee tier")
                return 0, False
                
            except Exception as e:
                logger.error(f"Error estimating UniswapV3 swap: {e}")
                return 0, False
            
        else:
            logger.error(f"Unknown router: {router}")
            return 0, False
            
    except Exception as e:
        logger.error(f"Error estimating swap output: {e}")
        return 0, False
        
def approve_token_spend(
    token_address: str,
    spender_address: str,
    amount: int,
    wallet_address: str,
    private_key: str,
    web3: Optional[Web3] = None
) -> Optional[str]:
    """
    Approve a spender to use tokens
    
    Args:
        token_address: Token contract address
        spender_address: Address to approve (usually a DEX router)
        amount: Amount to approve in smallest units
        wallet_address: Wallet address
        private_key: Private key for signing transaction
        web3: Web3 instance (will create one if None)
        
    Returns:
        Transaction hash or None on error
    """
    if not token_address or not spender_address or amount <= 0:
        return None
        
    if web3 is None:
        web3 = get_web3()
        
    if not web3 or not web3.is_connected():
        logger.error("Cannot approve token spend: No web3 connection")
        return None
        
    try:
        # Use checksum addresses
        token = web3.to_checksum_address(token_address)
        spender = web3.to_checksum_address(spender_address)
        
        # Create contract instance
        contract = web3.eth.contract(address=token, abi=ERC20_ABI)
        
        # Build approval transaction
        tx = contract.functions.approve(spender, amount).build_transaction({
            'from': wallet_address,
            'nonce': web3.eth.get_transaction_count(wallet_address),
            'gas': 100000,  # Gas limit
            'gasPrice': web3.eth.gas_price,
            'chainId': web3.eth.chain_id
        })
        
        # Sign and send transaction
        signed_tx = web3.eth.account.sign_transaction(tx, private_key)
        tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Return transaction hash
        return tx_hash.hex()
    except Exception as e:
        logger.error(f"Error approving {token_address} for {spender_address}: {e}")
        return None
        
def get_token_allowance(
    token_address: str,
    owner_address: str,
    spender_address: str,
    web3: Optional[Web3] = None
) -> Optional[int]:
    """
    Get token allowance for a spender
    
    Args:
        token_address: Token contract address
        owner_address: Token owner address
        spender_address: Spender address
        web3: Web3 instance (will create one if None)
        
    Returns:
        Allowance amount in smallest units or None on error
    """
    if not token_address or not owner_address or not spender_address:
        return None
        
    if web3 is None:
        web3 = get_web3()
        
    if not web3 or not web3.is_connected():
        logger.error("Cannot get token allowance: No web3 connection")
        return None
        
    try:
        # Use checksum addresses
        token = web3.to_checksum_address(token_address)
        owner = web3.to_checksum_address(owner_address)
        spender = web3.to_checksum_address(spender_address)
        
        # Create contract instance
        contract = web3.eth.contract(address=token, abi=ERC20_ABI)
        
        # Get allowance
        allowance = contract.functions.allowance(owner, spender).call()
        
        return allowance
    except Exception as e:
        logger.error(f"Error getting allowance of {token_address} for {spender_address}: {e}")
        return None
