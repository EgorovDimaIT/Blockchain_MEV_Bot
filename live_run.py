import os
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import threading
import traceback
import json
from web3 import Web3
from eth_account import Account
from eth_account.signers.local import LocalAccount

from data_collection.mempool_listener import get_mempool_listener
from data_collection.data_downloader import get_data_downloader
from ml_model.lstm_predictor import get_arbitrage_predictor
from utils.web3_helpers import get_web3_provider, get_eth_price_usd
from models import ArbitrageOpportunity, SandwichOpportunity, Transaction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('live_run.log')
    ]
)
logger = logging.getLogger('mev_bot_live_run')

# Global services dictionary
services = {}

# Exit flags for main loop
running = False
stop_requested = False

# Safety controls
safety_controls = {
    'emergency_stop': False,
    'max_gas_price_gwei': 200,  # Maximum gas price to pay in Gwei
    'min_wallet_balance_eth': 0.05,  # Minimum wallet balance to maintain in ETH
    'max_daily_tx_count': 500,  # Maximum number of transactions per day
    'max_daily_loss_eth': 0.1,  # Maximum daily loss in ETH
    'max_slippage_percent': 2.0,  # Maximum slippage percentage
    'transaction_timeout_sec': 120,  # Transaction timeout in seconds
    'reverts_before_pause': 3  # Number of reverts before pausing operations
}

# Opportunity execution thresholds
MIN_PROFIT_USD = 0.2  # $0.2 minimum profit
CONFIDENCE_THRESHOLD = 0.7  # 70% confidence minimum for live trading

# Transaction statistics
tx_stats = {
    'daily_tx_count': 0,
    'daily_profit_eth': 0.0,
    'daily_loss_eth': 0.0,
    'revert_count': 0,
    'last_reset': datetime.now()
}

def initialize_services() -> Dict:
    """
    Initialize and connect to required services
    
    Returns:
        Dictionary of service connections
    """
    global services
    
    logger.info("Initializing services...")
    
    try:
        # Connect to Web3 provider
        web3 = get_web3_provider()
        if not web3 or not web3.is_connected():
            logger.error("Failed to connect to Ethereum node")
            return {}
            
        logger.info(f"Connected to Ethereum node, chain ID: {web3.eth.chain_id}")
        services['web3'] = web3
        
        # Connect to database
        try:
            from app import db
            services['db'] = db
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return {}
        
        # Initialize data downloader
        try:
            data_downloader = get_data_downloader()
            services['data_downloader'] = data_downloader
            logger.info("Data downloader initialized")
        except Exception as e:
            logger.error(f"Error initializing data downloader: {e}")
        
        # Initialize ML predictor
        try:
            predictor = get_arbitrage_predictor(db)
            services['ml_predictor'] = predictor
            logger.info("ML predictor initialized")
        except Exception as e:
            logger.error(f"Error initializing ML predictor: {e}")
        
        # Start mempool listener
        try:
            mempool_listener = get_mempool_listener(db)
            success = mempool_listener.start()
            if success:
                services['mempool_listener'] = mempool_listener
                logger.info("Mempool listener started")
            else:
                logger.error("Failed to start mempool listener")
        except Exception as e:
            logger.error(f"Error starting mempool listener: {e}")
        
        # Initialize trading wallet
        private_key = os.environ.get('PRIVATE_KEY')
        if not private_key:
            logger.error("Private key not found in environment variables")
            return {}
            
        try:
            account: LocalAccount = Account.from_key(private_key)
            services['wallet'] = account
            wallet_address = account.address
            
            # Check wallet balance
            balance_wei = web3.eth.get_balance(wallet_address)
            balance_eth = web3.from_wei(balance_wei, 'ether')
            
            logger.info(f"Trading wallet initialized with address: {wallet_address[:6]}...{wallet_address[-4:]}")
            logger.info(f"Wallet balance: {balance_eth:.6f} ETH")
            
            if balance_eth < safety_controls['min_wallet_balance_eth']:
                logger.warning(f"Wallet balance below minimum threshold of {safety_controls['min_wallet_balance_eth']} ETH")
            
        except Exception as e:
            logger.error(f"Error initializing trading wallet: {e}")
            return {}
        
        # Initialize contract ABIs
        services['contract_abis'] = load_contract_abis()
        
        return services
        
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        return {}

def load_contract_abis() -> Dict:
    """
    Load contract ABIs from files
    
    Returns:
        Dictionary mapping contract names to ABIs
    """
    abis = {}
    
    # Define common ABIs inline if files not available
    
    # ERC20 token ABI
    abis['erc20'] = [
        {"inputs": [{"name": "_spender", "type": "address"}, {"name": "_value", "type": "uint256"}], "name": "approve", "outputs": [{"name": "", "type": "bool"}], "stateMutability": "nonpayable", "type": "function"},
        {"inputs": [{"name": "_owner", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}], "stateMutability": "view", "type": "function"},
        {"inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "stateMutability": "view", "type": "function"},
        {"inputs": [{"name": "_to", "type": "address"}, {"name": "_value", "type": "uint256"}], "name": "transfer", "outputs": [{"name": "", "type": "bool"}], "stateMutability": "nonpayable", "type": "function"}
    ]
    
    # Uniswap V2 Router ABI
    abis['uniswap_v2_router'] = [
        {"inputs": [{"name": "amountOutMin", "type": "uint256"}, {"name": "path", "type": "address[]"}, {"name": "to", "type": "address"}, {"name": "deadline", "type": "uint256"}], "name": "swapExactETHForTokens", "outputs": [{"name": "amounts", "type": "uint256[]"}], "stateMutability": "payable", "type": "function"},
        {"inputs": [{"name": "amountIn", "type": "uint256"}, {"name": "amountOutMin", "type": "uint256"}, {"name": "path", "type": "address[]"}, {"name": "to", "type": "address"}, {"name": "deadline", "type": "uint256"}], "name": "swapExactTokensForETH", "outputs": [{"name": "amounts", "type": "uint256[]"}], "stateMutability": "nonpayable", "type": "function"},
        {"inputs": [{"name": "amountIn", "type": "uint256"}, {"name": "amountOutMin", "type": "uint256"}, {"name": "path", "type": "address[]"}, {"name": "to", "type": "address"}, {"name": "deadline", "type": "uint256"}], "name": "swapExactTokensForTokens", "outputs": [{"name": "amounts", "type": "uint256[]"}], "stateMutability": "nonpayable", "type": "function"},
        {"inputs": [{"name": "amountOut", "type": "uint256"}, {"name": "path", "type": "address[]"}, {"name": "to", "type": "address"}, {"name": "deadline", "type": "uint256"}], "name": "swapETHForExactTokens", "outputs": [{"name": "amounts", "type": "uint256[]"}], "stateMutability": "payable", "type": "function"},
        {"inputs": [{"name": "amountA", "type": "uint256"}, {"name": "reserveA", "type": "uint256"}, {"name": "reserveB", "type": "uint256"}], "name": "getAmountOut", "outputs": [{"name": "amountB", "type": "uint256"}], "stateMutability": "pure", "type": "function"},
        {"inputs": [{"name": "amountOut", "type": "uint256"}, {"name": "reserveA", "type": "uint256"}, {"name": "reserveB", "type": "uint256"}], "name": "getAmountIn", "outputs": [{"name": "amountA", "type": "uint256"}], "stateMutability": "pure", "type": "function"},
        {"inputs": [{"name": "amountIn", "type": "uint256"}, {"name": "path", "type": "address[]"}], "name": "getAmountsOut", "outputs": [{"name": "amounts", "type": "uint256[]"}], "stateMutability": "view", "type": "function"}
    ]
    
    # Uniswap V2 Pair ABI
    abis['uniswap_v2_pair'] = [
        {"inputs": [], "name": "getReserves", "outputs": [{"name": "_reserve0", "type": "uint112"}, {"name": "_reserve1", "type": "uint112"}, {"name": "_blockTimestampLast", "type": "uint32"}], "stateMutability": "view", "type": "function"},
        {"inputs": [], "name": "token0", "outputs": [{"name": "", "type": "address"}], "stateMutability": "view", "type": "function"},
        {"inputs": [], "name": "token1", "outputs": [{"name": "", "type": "address"}], "stateMutability": "view", "type": "function"}
    ]
    
    # Uniswap V3 Router ABI
    abis['uniswap_v3_router'] = [
        {"inputs": [{"components": [{"internalType": "address", "name": "tokenIn", "type": "address"}, {"internalType": "address", "name": "tokenOut", "type": "address"}, {"internalType": "uint24", "name": "fee", "type": "uint24"}, {"internalType": "address", "name": "recipient", "type": "address"}, {"internalType": "uint256", "name": "deadline", "type": "uint256"}, {"internalType": "uint256", "name": "amountIn", "type": "uint256"}, {"internalType": "uint256", "name": "amountOutMinimum", "type": "uint256"}, {"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"}], "internalType": "struct ISwapRouter.ExactInputSingleParams", "name": "params", "type": "tuple"}], "name": "exactInputSingle", "outputs": [{"internalType": "uint256", "name": "amountOut", "type": "uint256"}], "stateMutability": "payable", "type": "function"},
        {"inputs": [{"components": [{"internalType": "bytes", "name": "path", "type": "bytes"}, {"internalType": "address", "name": "recipient", "type": "address"}, {"internalType": "uint256", "name": "deadline", "type": "uint256"}, {"internalType": "uint256", "name": "amountIn", "type": "uint256"}, {"internalType": "uint256", "name": "amountOutMinimum", "type": "uint256"}], "internalType": "struct ISwapRouter.ExactInputParams", "name": "params", "type": "tuple"}], "name": "exactInput", "outputs": [{"internalType": "uint256", "name": "amountOut", "type": "uint256"}], "stateMutability": "payable", "type": "function"}
    ]
    
    # Curve stETH-ETH pool ABI
    abis['curve_steth_eth'] = [
        {"name": "exchange", "outputs": [{"type": "uint256", "name": ""}], "inputs": [{"type": "int128", "name": "i"}, {"type": "int128", "name": "j"}, {"type": "uint256", "name": "dx"}, {"type": "uint256", "name": "min_dy"}], "stateMutability": "payable", "type": "function"},
        {"name": "get_dy", "outputs": [{"type": "uint256", "name": ""}], "inputs": [{"type": "int128", "name": "i"}, {"type": "int128", "name": "j"}, {"type": "uint256", "name": "dx"}], "stateMutability": "view", "type": "function"}
    ]
    
    # AAVE Flash Loan ABI
    abis['aave_lending_pool'] = [
        {"inputs": [{"internalType": "address[]", "name": "assets", "type": "address[]"}, {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}, {"internalType": "uint256[]", "name": "premiums", "type": "uint256[]"}, {"internalType": "address", "name": "initiator", "type": "address"}, {"internalType": "bytes", "name": "params", "type": "bytes"}], "name": "executeOperation", "outputs": [{"internalType": "bool", "name": "", "type": "bool"}], "stateMutability": "nonpayable", "type": "function"},
        {"inputs": [{"internalType": "address[]", "name": "assets", "type": "address[]"}, {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}, {"internalType": "uint256[]", "name": "modes", "type": "uint256[]"}, {"internalType": "address", "name": "onBehalfOf", "type": "address"}, {"internalType": "bytes", "name": "params", "type": "bytes"}, {"internalType": "uint16", "name": "referralCode", "type": "uint16"}], "name": "flashLoan", "outputs": [], "stateMutability": "nonpayable", "type": "function"}
    ]
    
    return abis

def scan_for_opportunities(services: Dict) -> List[Dict]:
    """
    Scan for arbitrage and sandwich opportunities
    
    Args:
        services: Dictionary of service connections
        
    Returns:
        List of opportunity dictionaries
    """
    try:
        logger.info("Scanning for arbitrage opportunities...")
        
        # Use Web3 instance directly to look for opportunities
        web3 = services.get('web3')
        
        if not web3 or not web3.is_connected():
            logger.error("Web3 not connected")
            return []
        
        # Use higher minimum profit threshold for live trading
        min_profit_threshold = 0.005  # 0.5% minimum for live trading
        
        # Get both direct and triangular arbitrage opportunities
        from strategy.arbitrage import find_direct_arbitrage_opportunities, find_triangular_arbitrage_opportunities
        arb_direct = find_direct_arbitrage_opportunities(web3, min_profit_threshold=min_profit_threshold)
        arb_triangular = find_triangular_arbitrage_opportunities(web3, min_profit_threshold=min_profit_threshold)
        
        # Combine all arbitrage opportunities
        arb_opportunities = arb_direct + arb_triangular
        
        # Look for sandwich opportunities if specifically configured
        sandwich_opportunities = []
        if not safety_controls.get('disable_sandwich_attacks', True):
            from strategy.sandwich import find_sandwich_attack_opportunities
            sandwich_opportunities = find_sandwich_attack_opportunities(web3, min_profit_threshold=min_profit_threshold)
        
        # Filter opportunities
        eth_price_usd = get_eth_price_usd()
        
        filtered_opportunities = []
        
        # Process arbitrage opportunities
        for opp in arb_opportunities:
            # Convert ETH profit to USD
            profit_usd = opp.get('expected_profit', 0) * eth_price_usd
            
            if profit_usd >= MIN_PROFIT_USD:
                opp['profit_usd'] = profit_usd
                filtered_opportunities.append({
                    'type': 'arbitrage',
                    'subtype': opp.get('arbitrage_type', 'direct'),
                    'data': opp
                })
        
        # Process sandwich opportunities
        for opp in sandwich_opportunities:
            # Convert ETH profit to USD
            profit_usd = opp.get('estimated_profit', 0) * eth_price_usd
            
            if profit_usd >= MIN_PROFIT_USD:
                opp['profit_usd'] = profit_usd
                filtered_opportunities.append({
                    'type': 'sandwich',
                    'data': opp
                })
        
        num_opportunities = len(filtered_opportunities)
        if num_opportunities > 0:
            logger.info(f"Found {num_opportunities} profitable opportunities")
        else:
            logger.info("No profitable opportunities found")
            
        return filtered_opportunities
        
    except Exception as e:
        logger.error(f"Error scanning for opportunities: {e}")
        traceback.print_exc()
        return []

def evaluate_opportunity(opportunity: Dict, services: Dict) -> Tuple[bool, float, float]:
    """
    Evaluate if an opportunity should be executed
    
    Args:
        opportunity: Opportunity dictionary
        services: Dictionary of service connections
        
    Returns:
        Tuple of (should_execute, adjusted_profit, confidence)
    """
    try:
        opp_type = opportunity.get('type', '')
        opp_data = opportunity.get('data', {})
        
        # Get ML predictor
        predictor = services.get('ml_predictor')
        
        if not predictor:
            logger.warning("ML predictor not available, using simple evaluation")
            return _simple_evaluate_opportunity(opportunity)
        
        # Evaluate based on opportunity type
        if opp_type == 'arbitrage':
            adjusted_profit, confidence = predictor.predict_arbitrage_opportunity(opp_data)
            
            # Calculate USD profit
            eth_price_usd = get_eth_price_usd()
            profit_usd = adjusted_profit * eth_price_usd
            
            # Check gas price constraints
            current_gas_price = services['web3'].eth.gas_price
            gas_price_gwei = services['web3'].from_wei(current_gas_price, 'gwei')
            
            if gas_price_gwei > safety_controls['max_gas_price_gwei']:
                logger.warning(f"Gas price too high: {gas_price_gwei} gwei > {safety_controls['max_gas_price_gwei']} gwei")
                return False, adjusted_profit, confidence
            
            # Log evaluation
            logger.info(f"Arbitrage evaluation: Expected profit ${opp_data.get('profit_usd', 0):.2f}, "
                        f"Adjusted profit ${profit_usd:.2f}, Confidence {confidence:.2f}")
            
            # Make execution decision
            should_execute = (profit_usd >= MIN_PROFIT_USD and 
                              confidence >= CONFIDENCE_THRESHOLD)
            
            return should_execute, adjusted_profit, confidence
            
        elif opp_type == 'sandwich':
            # For sandwich attacks, use a simple evaluation for now
            # TODO: Implement ML prediction for sandwich attacks
            return _simple_evaluate_opportunity(opportunity)
        
        else:
            logger.warning(f"Unknown opportunity type: {opp_type}")
            return False, 0.0, 0.0
        
    except Exception as e:
        logger.error(f"Error evaluating opportunity: {e}")
        traceback.print_exc()
        return False, 0.0, 0.0

def _simple_evaluate_opportunity(opportunity: Dict) -> Tuple[bool, float, float]:
    """
    Simple evaluation for opportunities without ML
    
    Args:
        opportunity: Opportunity dictionary
        
    Returns:
        Tuple of (should_execute, adjusted_profit, confidence)
    """
    opp_type = opportunity.get('type', '')
    opp_data = opportunity.get('data', {})
    
    if opp_type == 'arbitrage':
        expected_profit = opp_data.get('expected_profit', 0.0)
        profit_usd = opp_data.get('profit_usd', 0.0)
        
        # Use higher confidence for live trading
        confidence = 0.75
        
        # Apply a conservative discount factor for live trading
        adjusted_profit = expected_profit * 0.85  # 15% discount for risks
        
        # Make execution decision - use higher thresholds for live trading
        should_execute = (profit_usd >= MIN_PROFIT_USD and 
                          confidence >= CONFIDENCE_THRESHOLD)
        
        return should_execute, adjusted_profit, confidence
        
    elif opp_type == 'sandwich':
        estimated_profit = opp_data.get('estimated_profit', 0.0)
        profit_usd = opp_data.get('profit_usd', 0.0)
        
        # Sandwich attacks have more risk, use lower confidence
        confidence = 0.6
        
        # Apply a larger discount factor for sandwich attacks
        adjusted_profit = estimated_profit * 0.7  # 30% discount for risks
        
        # Make execution decision - use higher thresholds for live trading
        should_execute = (profit_usd >= MIN_PROFIT_USD and 
                          confidence >= CONFIDENCE_THRESHOLD)
        
        return should_execute, adjusted_profit, confidence
    
    else:
        logger.warning(f"Unknown opportunity type in simple evaluation: {opp_type}")
        return False, 0.0, 0.0

def store_opportunity(opportunity: Dict, evaluation: Tuple[bool, float, float], services: Dict) -> bool:
    """
    Store opportunity in database
    
    Args:
        opportunity: Opportunity dictionary
        evaluation: Evaluation tuple (should_execute, adjusted_profit, confidence)
        services: Dictionary of service connections
        
    Returns:
        True if stored successfully, False otherwise
    """
    try:
        db = services.get('db')
        if not db:
            logger.warning("Database not available, cannot store opportunity")
            return False
            
        should_execute, adjusted_profit, confidence = evaluation
        opp_type = opportunity.get('type', '')
        opp_data = opportunity.get('data', {})
        
        # Create database record based on opportunity type
        if opp_type == 'arbitrage':
            from flask import current_app
            with current_app.app_context():
                arb = ArbitrageOpportunity(
                    token_in=opp_data.get('token_in', ''),
                    token_out=opp_data.get('token_out', ''),
                    dex_1=opp_data.get('dex_1', ''),
                    dex_2=opp_data.get('dex_2', ''),
                    amount_in=str(opp_data.get('amount_in', 0)),
                    expected_profit=float(opp_data.get('expected_profit', 0)),
                    confidence_score=float(confidence),
                    detected_at=datetime.now(),
                    executed=False
                )
                
                # Add additional fields if available
                if opp_data.get('arbitrage_type'):
                    arb.arbitrage_type = opp_data.get('arbitrage_type')
                
                if opp_data.get('token_mid'):
                    arb.token_mid = opp_data.get('token_mid')
                
                if opp_data.get('dex_3'):
                    arb.dex_3 = opp_data.get('dex_3')
                
                if opp_data.get('flash_loan_source'):
                    arb.flash_loan = True
                    arb.flash_loan_source = opp_data.get('flash_loan_source')
                
                db.session.add(arb)
                db.session.commit()
                
                logger.info(f"Stored arbitrage opportunity (ID: {arb.id})")
                return True
                
        elif opp_type == 'sandwich':
            from flask import current_app
            with current_app.app_context():
                sandwich = SandwichOpportunity(
                    target_tx_hash=opp_data.get('target_tx_hash', ''),
                    token_address=opp_data.get('token_address', ''),
                    front_run_amount=float(opp_data.get('front_run_amount', 0)),
                    target_amount=float(opp_data.get('target_amount', 0)),
                    estimated_profit=float(opp_data.get('estimated_profit', 0)),
                    confidence_score=float(confidence),
                    dex=opp_data.get('dex', ''),
                    detected_at=datetime.now(),
                    executed=False
                )
                
                # Add additional fields if available
                if opp_data.get('mev_share_bundle_id'):
                    sandwich.mev_share_bundle_id = opp_data.get('mev_share_bundle_id')
                
                if opp_data.get('mev_share_builder'):
                    sandwich.mev_share_builder = opp_data.get('mev_share_builder')
                
                db.session.add(sandwich)
                db.session.commit()
                
                logger.info(f"Stored sandwich opportunity (ID: {sandwich.id})")
                return True
        
        else:
            logger.warning(f"Unknown opportunity type for storage: {opp_type}")
            return False
        
    except Exception as e:
        logger.error(f"Error storing opportunity: {e}")
        if services.get('db') and services['db'].session.is_active:
            services['db'].session.rollback()
        return False

def execute_arbitrage(opportunity: Dict, services: Dict) -> Dict:
    """
    Execute arbitrage opportunity
    
    Args:
        opportunity: Opportunity dictionary
        services: Dictionary of service connections
        
    Returns:
        Execution result dictionary
    """
    try:
        opp_data = opportunity.get('data', {})
        arbitrage_type = opp_data.get('arbitrage_type', 'direct')
        
        # Check if it's a flash loan arbitrage
        use_flash_loan = opp_data.get('flash_loan_source') is not None
        
        if arbitrage_type == 'direct':
            if use_flash_loan:
                return _execute_direct_arbitrage_with_flash_loan(opp_data, services)
            else:
                return _execute_direct_arbitrage(opp_data, services)
        elif arbitrage_type == 'triangular':
            if use_flash_loan:
                return _execute_triangular_arbitrage_with_flash_loan(opp_data, services)
            else:
                return _execute_triangular_arbitrage(opp_data, services)
        else:
            logger.error(f"Unknown arbitrage type: {arbitrage_type}")
            return {
                'success': False,
                'tx_hash': None,
                'executed_at': datetime.now(),
                'profit_eth': 0.0,
                'gas_used': 0,
                'gas_price': 0,
                'error': f"Unknown arbitrage type: {arbitrage_type}"
            }
    
    except Exception as e:
        logger.error(f"Error executing arbitrage: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'tx_hash': None,
            'executed_at': datetime.now(),
            'profit_eth': 0.0,
            'gas_used': 0,
            'gas_price': 0,
            'error': str(e)
        }

def _execute_direct_arbitrage(opp_data: Dict, services: Dict) -> Dict:
    """
    Execute direct arbitrage opportunity
    
    Args:
        opp_data: Opportunity data
        services: Service connections
        
    Returns:
        Execution result dictionary
    """
    try:
        web3 = services.get('web3')
        wallet = services.get('wallet')
        abis = services.get('contract_abis', {})
        
        if not web3 or not wallet:
            return {'success': False, 'error': 'Web3 or wallet not available'}
        
        token_in_address = opp_data.get('token_in')
        token_out_address = opp_data.get('token_out')
        amount_in = int(opp_data.get('amount_in', 0))
        dex_1 = opp_data.get('dex_1')
        dex_2 = opp_data.get('dex_2')
        
        # Check if we have ETH or need to approve token transfers
        is_eth_in = token_in_address.lower() in ['0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee', 
                                                'eth', 
                                                '0x0000000000000000000000000000000000000000']
        
        # Check wallet balance
        wallet_address = wallet.address
        if is_eth_in:
            balance = web3.eth.get_balance(wallet_address)
            if balance < amount_in:
                return {'success': False, 'error': f'Insufficient ETH balance: {web3.from_wei(balance, "ether")} ETH'}
        else:
            # Check ERC20 token balance
            token_contract = web3.eth.contract(address=token_in_address, abi=abis['erc20'])
            balance = token_contract.functions.balanceOf(wallet_address).call()
            if balance < amount_in:
                return {'success': False, 'error': f'Insufficient token balance: {balance / 10**18} tokens'}
            
            # Approve token transfers if necessary
            for dex in [dex_1, dex_2]:
                if dex.lower() == 'uniswap_v2':
                    router_address = '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D'
                    # Check allowance
                    allowance = token_contract.functions.allowance(wallet_address, router_address).call()
                    if allowance < amount_in:
                        # Approve
                        tx = token_contract.functions.approve(
                            router_address, 
                            2**256 - 1  # Unlimited approval
                        ).build_transaction({
                            'from': wallet_address,
                            'nonce': web3.eth.get_transaction_count(wallet_address),
                            'gas': 100000,
                            'gasPrice': web3.eth.gas_price
                        })
                        
                        # Sign and send transaction
                        signed_tx = web3.eth.account.sign_transaction(tx, wallet.key)
                        tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                        logger.info(f"Approval transaction sent: {tx_hash.hex()}")
                        
                        # Wait for transaction to be mined
                        receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
                        if receipt['status'] != 1:
                            return {'success': False, 'error': 'Token approval failed'}
        
        # Execute the arbitrage based on DEXes
        # Here we'll implement the Uniswap V2 direct arbitrage as an example
        if dex_1.lower() == 'uniswap_v2' and dex_2.lower() == 'uniswap_v2':
            return _execute_uniswap_v2_arbitrage(opp_data, services)
        else:
            return {'success': False, 'error': f'Unsupported DEX combination: {dex_1} and {dex_2}'}
    
    except Exception as e:
        logger.error(f"Error executing direct arbitrage: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def _execute_uniswap_v2_arbitrage(opp_data: Dict, services: Dict) -> Dict:
    """
    Execute Uniswap V2 arbitrage
    
    Args:
        opp_data: Opportunity data
        services: Service connections
        
    Returns:
        Execution result dict
    """
    try:
        web3 = services.get('web3')
        wallet = services.get('wallet')
        abis = services.get('contract_abis', {})
        
        token_in_address = opp_data.get('token_in')
        token_out_address = opp_data.get('token_out')
        amount_in = int(opp_data.get('amount_in', 0))
        
        # Uniswap V2 router address
        router_address = '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D'
        router = web3.eth.contract(address=router_address, abi=abis['uniswap_v2_router'])
        
        # Create transaction
        wallet_address = wallet.address
        deadline = int(time.time()) + 300  # 5 minutes
        
        # Get minimum amount out with slippage
        path = [web3.to_checksum_address(token_in_address), web3.to_checksum_address(token_out_address)]
        amounts_out = router.functions.getAmountsOut(amount_in, path).call()
        amount_out_min = int(amounts_out[1] * (1 - safety_controls['max_slippage_percent'] / 100))
        
        # Check if ETH is involved
        is_eth_in = token_in_address.lower() in ['0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee', 
                                                'eth', 
                                                '0x0000000000000000000000000000000000000000']
        is_eth_out = token_out_address.lower() in ['0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee', 
                                                   'eth', 
                                                   '0x0000000000000000000000000000000000000000']
        
        # Create appropriate transaction based on token types
        if is_eth_in:
            # ETH to Token
            tx = router.functions.swapExactETHForTokens(
                amount_out_min,
                path,
                wallet_address,
                deadline
            ).build_transaction({
                'from': wallet_address,
                'value': amount_in,
                'nonce': web3.eth.get_transaction_count(wallet_address),
                'gas': 250000,
                'gasPrice': web3.eth.gas_price
            })
        elif is_eth_out:
            # Token to ETH
            tx = router.functions.swapExactTokensForETH(
                amount_in,
                amount_out_min,
                path,
                wallet_address,
                deadline
            ).build_transaction({
                'from': wallet_address,
                'nonce': web3.eth.get_transaction_count(wallet_address),
                'gas': 250000,
                'gasPrice': web3.eth.gas_price
            })
        else:
            # Token to Token
            tx = router.functions.swapExactTokensForTokens(
                amount_in,
                amount_out_min,
                path,
                wallet_address,
                deadline
            ).build_transaction({
                'from': wallet_address,
                'nonce': web3.eth.get_transaction_count(wallet_address),
                'gas': 250000,
                'gasPrice': web3.eth.gas_price
            })
        
        # Sign and send transaction
        signed_tx = web3.eth.account.sign_transaction(tx, wallet.key)
        tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        logger.info(f"Arbitrage transaction sent: {tx_hash.hex()}")
        
        # Wait for transaction to be mined with timeout
        try:
            receipt = web3.eth.wait_for_transaction_receipt(
                tx_hash, 
                timeout=safety_controls['transaction_timeout_sec']
            )
            
            # Process transaction result
            if receipt['status'] == 1:
                gas_used = receipt['gasUsed']
                gas_price = web3.eth.get_transaction(tx_hash)['gasPrice']
                gas_cost_eth = web3.from_wei(gas_used * gas_price, 'ether')
                
                # Calculate actual profit
                # For ETH to Token to ETH, we measure balance change
                balance_after = web3.eth.get_balance(wallet_address)
                profit_eth = web3.from_wei(balance_after, 'ether') - opp_data.get('initial_balance_eth', 0)
                profit_eth -= gas_cost_eth
                
                return {
                    'success': True,
                    'tx_hash': tx_hash.hex(),
                    'executed_at': datetime.now(),
                    'profit_eth': float(profit_eth),
                    'gas_used': gas_used,
                    'gas_price': gas_price,
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'tx_hash': tx_hash.hex(),
                    'executed_at': datetime.now(),
                    'profit_eth': 0.0,
                    'gas_used': receipt['gasUsed'],
                    'gas_price': web3.eth.get_transaction(tx_hash)['gasPrice'],
                    'error': 'Transaction failed'
                }
                
        except TimeoutError:
            return {
                'success': False,
                'tx_hash': tx_hash.hex(),
                'executed_at': datetime.now(),
                'profit_eth': 0.0,
                'gas_used': 0,
                'gas_price': 0,
                'error': 'Transaction timeout'
            }
    
    except Exception as e:
        logger.error(f"Error executing Uniswap V2 arbitrage: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def _execute_direct_arbitrage_with_flash_loan(opp_data: Dict, services: Dict) -> Dict:
    """
    Execute direct arbitrage using a flash loan
    
    Args:
        opp_data: Opportunity data
        services: Service connections
        
    Returns:
        Execution result dictionary
    """
    # This is a complex implementation that would involve:
    # 1. Creating a flash loan contract
    # 2. Encoding the arbitrage logic in the contract
    # 3. Calling the flash loan provider
    
    # For now, return a placeholder implementation
    logger.warning("Flash loan arbitrage execution not fully implemented yet")
    return {
        'success': False,
        'tx_hash': None,
        'executed_at': datetime.now(),
        'profit_eth': 0.0,
        'gas_used': 0,
        'gas_price': 0,
        'error': 'Flash loan arbitrage not implemented'
    }

def _execute_triangular_arbitrage(opp_data: Dict, services: Dict) -> Dict:
    """
    Execute triangular arbitrage
    
    Args:
        opp_data: Opportunity data
        services: Service connections
        
    Returns:
        Execution result dictionary
    """
    # This would be similar to direct arbitrage but with 3 tokens
    logger.warning("Triangular arbitrage execution not fully implemented yet")
    return {
        'success': False,
        'tx_hash': None,
        'executed_at': datetime.now(),
        'profit_eth': 0.0,
        'gas_used': 0,
        'gas_price': 0,
        'error': 'Triangular arbitrage not implemented'
    }

def _execute_triangular_arbitrage_with_flash_loan(opp_data: Dict, services: Dict) -> Dict:
    """
    Execute triangular arbitrage with flash loan
    
    Args:
        opp_data: Opportunity data
        services: Service connections
        
    Returns:
        Execution result dictionary
    """
    # This is even more complex than direct flash loan arbitrage
    logger.warning("Triangular flash loan arbitrage execution not fully implemented yet")
    return {
        'success': False,
        'tx_hash': None,
        'executed_at': datetime.now(),
        'profit_eth': 0.0,
        'gas_used': 0,
        'gas_price': 0,
        'error': 'Triangular flash loan arbitrage not implemented'
    }

def execute_sandwich(opportunity: Dict, services: Dict) -> Dict:
    """
    Execute sandwich attack
    
    Args:
        opportunity: Opportunity dictionary
        services: Service connections
        
    Returns:
        Execution result dictionary
    """
    # Sandwich attacks are complex and ethically questionable
    # This is just a placeholder for completeness
    logger.warning("Sandwich attack execution not implemented")
    return {
        'success': False,
        'tx_hash': None,
        'executed_at': datetime.now(),
        'profit_eth': 0.0,
        'gas_used': 0,
        'gas_price': 0,
        'error': 'Sandwich attacks not enabled'
    }

def store_execution_result(opportunity: Dict, result: Dict, services: Dict) -> bool:
    """
    Store execution result in database
    
    Args:
        opportunity: Opportunity dictionary
        result: Execution result dictionary
        services: Dictionary of service connections
        
    Returns:
        True if stored successfully, False otherwise
    """
    try:
        db = services.get('db')
        if not db:
            logger.warning("Database not available, cannot store execution result")
            return False
            
        opp_type = opportunity.get('type', '')
        opp_data = opportunity.get('data', {})
        
        from flask import current_app
        with current_app.app_context():
            # Create transaction record
            tx = Transaction(
                tx_hash=result.get('tx_hash'),
                strategy_type=opp_type,
                status='confirmed' if result.get('success') else 'failed',
                created_at=datetime.now(),
                executed_at=result.get('executed_at'),
                gas_used=result.get('gas_used'),
                gas_price=result.get('gas_price'),
                profit_eth=result.get('profit_eth'),
                profit_usd=result.get('profit_eth', 0) * get_eth_price_usd(),
                error_message=result.get('error')
            )
            
            # Set tokens involved
            tokens = []
            if opp_type == 'arbitrage':
                tokens.append(opp_data.get('token_in', ''))
                tokens.append(opp_data.get('token_out', ''))
                if opp_data.get('token_mid'):
                    tokens.append(opp_data.get('token_mid'))
                    
                # Set arbitrage-specific fields
                tx.arbitrage_type = opp_data.get('arbitrage_type', 'direct')
                tx.flash_loan_used = bool(opp_data.get('flash_loan_source'))
                tx.flash_loan_source = opp_data.get('flash_loan_source')
                
            elif opp_type == 'sandwich':
                tokens.append(opp_data.get('token_address', ''))
                
                # Set sandwich-specific fields
                tx.front_tx_hash = result.get('front_tx_hash')
                tx.back_tx_hash = result.get('back_tx_hash')
            
            tx.set_tokens_involved(tokens)
            
            # Add to database
            db.session.add(tx)
            db.session.flush()  # Get the transaction ID
            
            # Update corresponding opportunity
            if opp_type == 'arbitrage':
                arb = ArbitrageOpportunity.query.filter_by(
                    token_in=opp_data.get('token_in', ''),
                    token_out=opp_data.get('token_out', ''),
                    executed=False
                ).order_by(ArbitrageOpportunity.detected_at.desc()).first()
                
                if arb:
                    arb.executed = True
                    arb.transaction_id = tx.id
                    
            elif opp_type == 'sandwich':
                sandwich = SandwichOpportunity.query.filter_by(
                    target_tx_hash=opp_data.get('target_tx_hash', ''),
                    executed=False
                ).first()
                
                if sandwich:
                    sandwich.executed = True
                    sandwich.transaction_id = tx.id
            
            # Commit changes
            db.session.commit()
            
            logger.info(f"Stored execution result for {opp_type} (TX ID: {tx.id})")
            return True
            
    except Exception as e:
        logger.error(f"Error storing execution result: {e}")
        if services.get('db') and services['db'].session.is_active:
            services['db'].session.rollback()
        return False

def update_stats(result: Dict):
    """
    Update transaction statistics
    
    Args:
        result: Execution result dictionary
    """
    global tx_stats
    
    # Check if we need to reset daily stats
    now = datetime.now()
    if (now - tx_stats['last_reset']).days > 0:
        # Reset daily stats
        tx_stats['daily_tx_count'] = 0
        tx_stats['daily_profit_eth'] = 0.0
        tx_stats['daily_loss_eth'] = 0.0
        tx_stats['last_reset'] = now
    
    # Update stats
    tx_stats['daily_tx_count'] += 1
    
    if result.get('success', False):
        profit_eth = result.get('profit_eth', 0.0)
        if profit_eth > 0:
            tx_stats['daily_profit_eth'] += profit_eth
        else:
            tx_stats['daily_loss_eth'] += abs(profit_eth)
    else:
        # Count failed transactions separately
        tx_stats['revert_count'] += 1

def check_safety_thresholds() -> bool:
    """
    Check if any safety thresholds have been exceeded
    
    Returns:
        True if bot should continue, False if it should stop
    """
    global tx_stats, safety_controls
    
    # Check if emergency stop is requested
    if safety_controls['emergency_stop']:
        logger.warning("Emergency stop requested")
        return False
    
    # Check wallet balance
    try:
        web3 = services.get('web3')
        wallet = services.get('wallet')
        
        if web3 and wallet:
            balance_wei = web3.eth.get_balance(wallet.address)
            balance_eth = web3.from_wei(balance_wei, 'ether')
            
            if balance_eth < safety_controls['min_wallet_balance_eth']:
                logger.warning(f"Wallet balance below threshold: {balance_eth} ETH")
                return False
    except Exception as e:
        logger.error(f"Error checking wallet balance: {e}")
    
    # Check daily transaction count
    if tx_stats['daily_tx_count'] >= safety_controls['max_daily_tx_count']:
        logger.warning(f"Maximum daily transaction count reached: {tx_stats['daily_tx_count']}")
        return False
    
    # Check daily loss
    if tx_stats['daily_loss_eth'] >= safety_controls['max_daily_loss_eth']:
        logger.warning(f"Maximum daily loss reached: {tx_stats['daily_loss_eth']} ETH")
        return False
    
    # Check consecutive reverts
    if tx_stats['revert_count'] >= safety_controls['reverts_before_pause']:
        logger.warning(f"Too many consecutive reverts: {tx_stats['revert_count']}")
        return False
    
    return True

def main_loop():
    """Main execution loop"""
    global running, stop_requested, services, tx_stats
    
    try:
        while running and not stop_requested:
            try:
                # Check services
                if not services.get('web3') or not services['web3'].is_connected():
                    logger.error("Web3 connection lost, attempting to reconnect...")
                    services['web3'] = get_web3_provider()
                    if not services['web3'] or not services['web3'].is_connected():
                        logger.error("Failed to reconnect to Web3, retrying in 30 seconds")
                        time.sleep(30)
                        continue
                
                # Check safety thresholds
                if not check_safety_thresholds():
                    logger.warning("Safety threshold exceeded, pausing operations")
                    time.sleep(300)  # Pause for 5 minutes
                    continue
                
                # Scan for opportunities
                opportunities = scan_for_opportunities(services)
                
                # Process each opportunity
                for opp in opportunities:
                    try:
                        # Skip if stop requested
                        if stop_requested:
                            break
                            
                        # Evaluate opportunity
                        evaluation = evaluate_opportunity(opp, services)
                        should_execute, adjusted_profit, confidence = evaluation
                        
                        # Store opportunity in database
                        store_opportunity(opp, evaluation, services)
                        
                        # Execute if profitable enough
                        if should_execute:
                            opp_type = opp.get('type', '')
                            
                            if opp_type == 'arbitrage':
                                # Execute arbitrage
                                result = execute_arbitrage(opp, services)
                            elif opp_type == 'sandwich':
                                # Execute sandwich attack
                                result = execute_sandwich(opp, services)
                            else:
                                logger.warning(f"Unknown opportunity type: {opp_type}")
                                continue
                            
                            # Store execution result
                            store_execution_result(opp, result, services)
                            
                            # Update statistics
                            update_stats(result)
                            
                            # Reset revert count if transaction succeeded
                            if result.get('success', False):
                                tx_stats['revert_count'] = 0
                    
                    except Exception as e:
                        logger.error(f"Error processing opportunity: {e}")
                        traceback.print_exc()
                
                # Wait before next scan (shorter interval for live trading)
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                traceback.print_exc()
                time.sleep(30)  # Wait longer on error
                
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        running = False
        logger.info("Main loop stopped")

def start_bot():
    """Start the bot"""
    global running, stop_requested, services, tx_stats
    
    if running:
        logger.warning("Bot already running")
        return
    
    # Connect to services
    if not services:
        services = initialize_services()
        
    if not services:
        logger.error("Failed to initialize required services")
        return
    
    # Reset stats
    tx_stats = {
        'daily_tx_count': 0,
        'daily_profit_eth': 0.0,
        'daily_loss_eth': 0.0,
        'revert_count': 0,
        'last_reset': datetime.now()
    }
    
    # Reset safety controls
    safety_controls['emergency_stop'] = False
    
    # Set running flag
    running = True
    stop_requested = False
    
    # Start main loop in a separate thread
    thread = threading.Thread(target=main_loop)
    thread.daemon = True
    thread.start()
    
    logger.info("Bot started in live mode")
    
    return thread

def stop_bot():
    """Stop the bot"""
    global running, stop_requested
    
    if not running:
        logger.warning("Bot not running")
        return
    
    # Set stop flag
    stop_requested = True
    
    # Stop mempool listener
    if services.get('mempool_listener'):
        services['mempool_listener'].stop()
    
    logger.info("Bot stop requested")

if __name__ == "__main__":
    try:
        # Initialize services
        services = initialize_services()
        
        if not services:
            logger.error("Failed to initialize required services")
            exit(1)
        
        # Start the bot
        bot_thread = start_bot()
        
        # Keep main thread alive
        while running and not stop_requested:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down...")
                stop_bot()
                break
        
        # Wait for bot thread to finish
        if bot_thread:
            bot_thread.join(timeout=30)
        
        logger.info("Bot shutdown complete")
        
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        traceback.print_exc()
        exit(1)