"""
Strategy engine for MEV bot - manages different MEV strategies
"""

import os
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from web3 import Web3

from utils.web3_helpers import get_web3
from utils.token_utils import get_token_info

# Import token price function
from utils.web3_helpers import get_token_price
from .arbitrage import find_direct_arbitrage_opportunities, find_triangular_arbitrage_opportunities

logger = logging.getLogger(__name__)

# Global variables
strategies = {}
running = False
stop_event = threading.Event()
last_scan_time = {}
opportunity_cache = {}

def init_strategy_engine():
    """
    Initialize the strategy engine
    """
    global strategies, running, stop_event
    
    # Reset state
    strategies = {
        'direct_arbitrage': {
            'enabled': True,
            'min_profit_threshold': 0.002,  # 0.2%
            'scan_interval': 60,  # seconds
            'execution_enabled': False
        },
        'triangular_arbitrage': {
            'enabled': True,
            'min_profit_threshold': 0.002,  # 0.2%
            'scan_interval': 60,  # seconds
            'execution_enabled': False
        },
        'sandwich': {
            'enabled': False,
            'min_profit_threshold': 0.005,  # 0.5%
            'scan_interval': 10,  # seconds
            'execution_enabled': False
        },
        'liquidation': {
            'enabled': False,
            'min_collateralization_ratio': 1.1,  # 110%
            'scan_interval': 60,  # seconds
            'execution_enabled': False
        }
    }
    
    running = False
    stop_event = threading.Event()
    last_scan_time.clear()
    opportunity_cache.clear()
    
    logger.info("Strategy engine initialized")
    
def start_engine():
    """
    Start the strategy engine
    """
    global running
    
    if running:
        logger.warning("Strategy engine already running")
        return False
        
    logger.info("Starting strategy engine")
    
    # Reset stop event
    stop_event.clear()
    
    # Start background thread
    running = True
    threading.Thread(target=_engine_loop, daemon=True).start()
    
    return True
    
def stop_engine():
    """
    Stop the strategy engine
    """
    global running
    
    if not running:
        logger.warning("Strategy engine not running")
        return False
        
    logger.info("Stopping strategy engine")
    
    # Set stop event
    stop_event.set()
    
    # Wait for thread to stop
    running = False
    
    return True
    
def get_strategies():
    """
    Get all configured strategies
    
    Returns:
        Dictionary of strategies and their configuration
    """
    return strategies
    
def update_strategy_config(strategy_name: str, config_updates: Dict):
    """
    Update strategy configuration
    
    Args:
        strategy_name: Name of the strategy to update
        config_updates: Dictionary of configuration updates
        
    Returns:
        True if successful, False otherwise
    """
    if strategy_name not in strategies:
        logger.error(f"Unknown strategy: {strategy_name}")
        return False
        
    # Update the strategy configuration
    for key, value in config_updates.items():
        if key in strategies[strategy_name]:
            strategies[strategy_name][key] = value
        else:
            logger.warning(f"Unknown configuration key: {key} for strategy: {strategy_name}")
            
    logger.info(f"Updated strategy configuration for {strategy_name}")
    return True
    
def get_cached_opportunities(strategy_type: str = None) -> List[Dict]:
    """
    Get cached opportunities
    
    Args:
        strategy_type: Type of opportunities to get (None for all)
        
    Returns:
        List of opportunity dictionaries
    """
    if strategy_type:
        return opportunity_cache.get(strategy_type, [])
    else:
        all_opportunities = []
        for opps in opportunity_cache.values():
            all_opportunities.extend(opps)
        return all_opportunities
    
def scan_opportunities(strategy_type: str = None) -> Dict[str, List[Dict]]:
    """
    Scan for opportunities across all strategies
    
    Args:
        strategy_type: Type of strategy to scan (None for all enabled)
        
    Returns:
        Dictionary of strategy types to lists of opportunity dictionaries
    """
    web3 = get_web3()
    
    if not web3 or not web3.is_connected():
        logger.error("Cannot scan for opportunities: No web3 connection")
        return {}
        
    results = {}
    
    if strategy_type is None or strategy_type == 'direct_arbitrage':
        if strategies['direct_arbitrage']['enabled']:
            try:
                logger.info("Scanning for direct arbitrage opportunities")
                opps = find_direct_arbitrage_opportunities(
                    web3=web3,
                    min_profit_threshold=strategies['direct_arbitrage']['min_profit_threshold']
                )
                results['direct_arbitrage'] = opps
                opportunity_cache['direct_arbitrage'] = opps
                last_scan_time['direct_arbitrage'] = time.time()
                logger.info(f"Found {len(opps)} direct arbitrage opportunities")
            except Exception as e:
                logger.error(f"Error scanning for direct arbitrage: {e}")
                
    if strategy_type is None or strategy_type == 'triangular_arbitrage':
        if strategies['triangular_arbitrage']['enabled']:
            try:
                logger.info("Scanning for triangular arbitrage opportunities")
                opps = find_triangular_arbitrage_opportunities(
                    web3=web3,
                    min_profit_threshold=strategies['triangular_arbitrage']['min_profit_threshold']
                )
                results['triangular_arbitrage'] = opps
                opportunity_cache['triangular_arbitrage'] = opps
                last_scan_time['triangular_arbitrage'] = time.time()
                logger.info(f"Found {len(opps)} triangular arbitrage opportunities")
            except Exception as e:
                logger.error(f"Error scanning for triangular arbitrage: {e}")
                
    if strategy_type is None or strategy_type == 'sandwich':
        if strategies['sandwich']['enabled']:
            # Not implemented yet
            logger.warning("Sandwich strategy not implemented yet")
            
    if strategy_type is None or strategy_type == 'liquidation':
        if strategies['liquidation']['enabled']:
            # Not implemented yet
            logger.warning("Liquidation strategy not implemented yet")
            
    return results
    
def _engine_loop():
    """
    Main strategy engine loop (runs in background thread)
    """
    logger.info("Strategy engine loop started")
    
    while running and not stop_event.is_set():
        try:
            # Check each strategy
            for strategy_name, config in strategies.items():
                if not config['enabled']:
                    continue
                    
                # Check if it's time to scan
                last_time = last_scan_time.get(strategy_name, 0)
                if time.time() - last_time < config['scan_interval']:
                    continue
                    
                # Scan for opportunities
                logger.debug(f"Scanning for {strategy_name} opportunities")
                scan_opportunities(strategy_name)
                
        except Exception as e:
            logger.error(f"Error in strategy engine loop: {e}")
            
        # Short sleep to prevent CPU hogging
        time.sleep(1)
        
    logger.info("Strategy engine loop stopped")