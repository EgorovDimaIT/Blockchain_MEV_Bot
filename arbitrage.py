"""
Arbitrage strategy implementation for MEV bot
"""

import os
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from decimal import Decimal
import json
from web3 import Web3

from utils.web3_helpers import get_web3, get_token_price
from utils.token_utils import (
    get_token_info, get_token_balance, estimate_token_swap_output,
    UNISWAP_V2_ROUTER, SUSHISWAP_ROUTER, UNISWAP_V3_ROUTER,
    WETH_ADDRESS, DAI_ADDRESS, USDC_ADDRESS, USDT_ADDRESS, WBTC_ADDRESS
)

logger = logging.getLogger(__name__)

# Common token pairs for arbitrage
COMMON_TOKEN_PAIRS = [
    (WETH_ADDRESS, USDC_ADDRESS),
    (WETH_ADDRESS, DAI_ADDRESS),
    (WETH_ADDRESS, USDT_ADDRESS),
    (WETH_ADDRESS, WBTC_ADDRESS),
    (USDC_ADDRESS, DAI_ADDRESS),
    (USDC_ADDRESS, USDT_ADDRESS),
    (WBTC_ADDRESS, USDT_ADDRESS),
    (WBTC_ADDRESS, USDC_ADDRESS)
]

# Common DEX routers for arbitrage
COMMON_DEXES = [
    UNISWAP_V2_ROUTER,
    SUSHISWAP_ROUTER,
    UNISWAP_V3_ROUTER
]

def find_direct_arbitrage_opportunities(
    web3: Optional[Web3] = None,
    min_profit_threshold: float = 0.002,
    token_pairs: Optional[List[Tuple[str, str]]] = None,
    dexes: Optional[List[str]] = None
) -> List[Dict]:
    """
    Find direct arbitrage opportunities between different DEXes
    
    Args:
        web3: Web3 instance
        min_profit_threshold: Minimum profit threshold as decimal (e.g., 0.002 for 0.2%)
        token_pairs: List of token pairs to check (defaults to common pairs)
        dexes: List of DEX routers to check (defaults to common DEXes)
        
    Returns:
        List of arbitrage opportunity dictionaries
    """
    if web3 is None:
        web3 = get_web3()
        
    if not web3 or not web3.is_connected():
        logger.error("Cannot find arbitrage opportunities: No web3 connection")
        return []
        
    if token_pairs is None:
        token_pairs = COMMON_TOKEN_PAIRS
        
    if dexes is None:
        dexes = COMMON_DEXES
        
    opportunities = []
    
    # Iterate over all token pairs
    for token_a, token_b in token_pairs:
        # Get token info
        token_a_info = get_token_info(token_a)
        token_b_info = get_token_info(token_b)
        
        if not token_a_info or not token_b_info:
            logger.warning(f"Skipping pair ({token_a}, {token_b}): Failed to get token info")
            continue
            
        token_a_decimals = token_a_info.get('decimals', 18)
        token_b_decimals = token_b_info.get('decimals', 18)
        
        # Standard amount for price check (0.1 ETH worth)
        amount_in = 10 ** token_a_decimals // 10  # 0.1 units of token A
        
        logger.info(f"Checking arbitrage for {token_a_info.get('symbol')} <-> {token_b_info.get('symbol')}")
        
        # Get prices on different DEXes
        dex_prices = {}
        
        for dex in dexes:
            try:
                # Forward swap (A -> B)
                amount_out, success = estimate_token_swap_output(
                    dex, token_a, token_b, amount_in
                )
                
                if success and amount_out > 0:
                    # Calculate effective price
                    price = Decimal(amount_out) / Decimal(amount_in) * Decimal(10 ** (token_a_decimals - token_b_decimals))
                    
                    # Store in dex_prices
                    dex_name = "unknown"
                    if dex.lower() == UNISWAP_V2_ROUTER.lower():
                        dex_name = "uniswap_v2"
                    elif dex.lower() == SUSHISWAP_ROUTER.lower():
                        dex_name = "sushiswap"
                    elif dex.lower() == UNISWAP_V3_ROUTER.lower():
                        dex_name = "uniswap_v3"
                        
                    dex_prices[dex_name] = {
                        'price': float(price),
                        'router': dex,
                        'amount_out': amount_out
                    }
            except Exception as e:
                logger.error(f"Error checking price on DEX {dex}: {e}")
                
        # Find arbitrage opportunities if we have prices from at least 2 DEXes
        if len(dex_prices) >= 2:
            # Find best buy and sell prices
            best_buy = min(dex_prices.items(), key=lambda x: x[1]['price'])
            best_sell = max(dex_prices.items(), key=lambda x: x[1]['price'])
            
            # Calculate potential profit
            buy_price = best_buy[1]['price']
            sell_price = best_sell[1]['price']
            
            price_diff = (sell_price - buy_price) / buy_price
            
            # If profit is above threshold, record opportunity
            if price_diff > min_profit_threshold:
                # Get token prices in USD for calculating profit
                token_a_usd = get_token_price(token_a) or 100  # Default to $100 if price not available
                
                # Calculate profit in token units and USD
                profit_token = amount_in * price_diff
                profit_usd = profit_token * token_a_usd / (10 ** token_a_decimals)
                
                logger.info(f"Found arbitrage opportunity: Buy {token_a_info.get('symbol')} on {best_buy[0]}, "
                           f"Sell on {best_sell[0]}, Profit: {price_diff:.2%}, ${profit_usd:.2f}")
                
                opportunity = {
                    'arbitrage_type': 'direct',
                    'token_in': token_a,
                    'token_out': token_b,
                    'token_in_symbol': token_a_info.get('symbol'),
                    'token_out_symbol': token_b_info.get('symbol'),
                    'buy_dex': best_buy[0],
                    'sell_dex': best_sell[0],
                    'buy_router': best_buy[1]['router'],
                    'sell_router': best_sell[1]['router'],
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'price_difference': price_diff,
                    'expected_profit': float(price_diff),  # As a decimal
                    'expected_profit_usd': float(profit_usd),
                    'confidence': 0.8,  # Default confidence
                    'timestamp': int(time.time())
                }
                
                opportunities.append(opportunity)
                
    # Sort opportunities by expected profit
    opportunities.sort(key=lambda x: x['expected_profit'], reverse=True)
    
    return opportunities

def find_triangular_arbitrage_opportunities(
    web3: Optional[Web3] = None,
    min_profit_threshold: float = 0.002,
    base_tokens: Optional[List[str]] = None,
    mid_tokens: Optional[List[str]] = None,
    dexes: Optional[List[str]] = None
) -> List[Dict]:
    """
    Find triangular arbitrage opportunities on a single DEX
    
    Args:
        web3: Web3 instance
        min_profit_threshold: Minimum profit threshold as decimal (e.g., 0.002 for 0.2%)
        base_tokens: List of base tokens to use (defaults to common tokens)
        mid_tokens: List of mid tokens to use (defaults to common tokens)
        dexes: List of DEX routers to check (defaults to common DEXes)
        
    Returns:
        List of arbitrage opportunity dictionaries
    """
    if web3 is None:
        web3 = get_web3()
        
    if not web3 or not web3.is_connected():
        logger.error("Cannot find triangular arbitrage opportunities: No web3 connection")
        return []
        
    # Default base tokens (tokens we start and end with)
    if base_tokens is None:
        base_tokens = [WETH_ADDRESS, USDC_ADDRESS, WBTC_ADDRESS]
        
    # Default mid tokens (intermediate tokens)
    if mid_tokens is None:
        mid_tokens = [USDC_ADDRESS, DAI_ADDRESS, USDT_ADDRESS, WETH_ADDRESS, WBTC_ADDRESS]
        
    if dexes is None:
        dexes = COMMON_DEXES
        
    opportunities = []
    
    # Iterate over all base tokens, mid tokens, and DEXes
    for base_token in base_tokens:
        base_token_info = get_token_info(base_token)
        if not base_token_info:
            continue
            
        base_decimals = base_token_info.get('decimals', 18)
        
        # Standard amount for price check (0.1 token)
        amount_in = 10 ** base_decimals // 10  # 0.1 units
        
        for mid_token in mid_tokens:
            # Skip if base and mid are the same
            if base_token.lower() == mid_token.lower():
                continue
                
            mid_token_info = get_token_info(mid_token)
            if not mid_token_info:
                continue
                
            for dex in dexes:
                dex_name = "unknown"
                if dex.lower() == UNISWAP_V2_ROUTER.lower():
                    dex_name = "uniswap_v2"
                elif dex.lower() == SUSHISWAP_ROUTER.lower():
                    dex_name = "sushiswap"
                elif dex.lower() == UNISWAP_V3_ROUTER.lower():
                    dex_name = "uniswap_v3"
                
                try:
                    # Step 1: Base -> Mid
                    amount_mid, success1 = estimate_token_swap_output(
                        dex, base_token, mid_token, amount_in
                    )
                    
                    if not success1 or amount_mid <= 0:
                        continue
                        
                    # Step 2: Mid -> Base (complete the triangle)
                    amount_out, success2 = estimate_token_swap_output(
                        dex, mid_token, base_token, amount_mid
                    )
                    
                    if not success2 or amount_out <= 0:
                        continue
                        
                    # Calculate profit ratio
                    profit_ratio = Decimal(amount_out) / Decimal(amount_in) - 1
                    
                    # If profitable, record opportunity
                    if profit_ratio > min_profit_threshold:
                        # Get token prices in USD for calculating profit
                        base_token_usd = get_token_price(base_token) or 100  # Default to $100 if price not available
                        
                        # Calculate profit in base token units and USD
                        profit_token = amount_in * float(profit_ratio)
                        profit_usd = profit_token * base_token_usd / (10 ** base_decimals)
                        
                        logger.info(f"Found triangular arbitrage: {base_token_info.get('symbol')} -> "
                                  f"{mid_token_info.get('symbol')} -> {base_token_info.get('symbol')} "
                                  f"on {dex_name}, Profit: {float(profit_ratio):.2%}, ${profit_usd:.2f}")
                        
                        opportunity = {
                            'arbitrage_type': 'triangular',
                            'base_token': base_token,
                            'mid_token': mid_token,
                            'base_token_symbol': base_token_info.get('symbol'),
                            'mid_token_symbol': mid_token_info.get('symbol'),
                            'dex': dex_name,
                            'router': dex,
                            'profit_ratio': float(profit_ratio),
                            'expected_profit': float(profit_ratio),  # As a decimal
                            'expected_profit_usd': float(profit_usd),
                            'amount_in': amount_in,
                            'amount_mid': amount_mid,
                            'amount_out': amount_out,
                            'confidence': 0.75,  # Default confidence
                            'timestamp': int(time.time())
                        }
                        
                        opportunities.append(opportunity)
                        
                except Exception as e:
                    logger.error(f"Error checking triangular arbitrage for {base_token_info.get('symbol')} -> "
                              f"{mid_token_info.get('symbol')} on {dex_name}: {e}")
                    
    # Sort opportunities by expected profit
    opportunities.sort(key=lambda x: x['expected_profit'], reverse=True)
    
    return opportunities

def execute_arbitrage(opportunity: Dict, wallet_address: str, private_key: str) -> Dict:
    """
    Execute an arbitrage opportunity
    
    Args:
        opportunity: Arbitrage opportunity dictionary
        wallet_address: Wallet address
        private_key: Private key for signing transactions
        
    Returns:
        Dictionary with execution results
    """
    # Not implemented yet - would contain actual transaction execution logic
    logger.warning("Arbitrage execution not yet implemented")
    
    return {
        'success': False,
        'reason': 'Not implemented',
        'opportunity': opportunity
    }