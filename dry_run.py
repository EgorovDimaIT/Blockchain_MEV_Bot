import os
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import threading
from dotenv import load_dotenv
import traceback
import os

from data_collection.mempool_listener import get_mempool_listener
from data_collection.data_downloader import get_data_downloader
from ml_model.lstm_predictor import get_arbitrage_predictor
from utils.web3_helpers import get_web3_provider
from models import ArbitrageOpportunity, SandwichOpportunity, Transaction

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dry_run.log')
    ]
)
logger = logging.getLogger('mev_bot_dry_run')

# Global services dictionary
services = {}

# Exit flag for main loop
running = False
stop_requested = False

# Opportunity scanning interval (seconds)
SCAN_INTERVAL = 10

# Opportunity execution thresholds
MIN_PROFIT_USD = 0.2  # $0.2 minimum profit
CONFIDENCE_THRESHOLD = 0.6  # 60% confidence minimum

def connect_to_services() -> Dict:
    """
    Connect to required services
    
    Returns:
        Dictionary of service connections
    """
    global services
    
    logger.info("Connecting to services...")
    
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
            mempool_listener.start_listening()
            services['mempool_listener'] = mempool_listener
            logger.info("Mempool listener started")
        except Exception as e:
            logger.error(f"Error starting mempool listener: {e}")
        
        return services
        
    except Exception as e:
        logger.error(f"Error connecting to services: {e}")
        return {}

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
        
        # First try direct arbitrage with standard threshold
        from strategy.arbitrage import find_direct_arbitrage_opportunities, find_triangular_arbitrage_opportunities
        arb_direct = find_direct_arbitrage_opportunities(web3, min_profit_threshold=0.002)
        
        if not arb_direct:
            # If no direct opportunities found, try with lower threshold for testing
            logger.info("No direct arbitrage opportunities found with 0.2% threshold, trying with 0.1%...")
            arb_direct = find_direct_arbitrage_opportunities(web3, min_profit_threshold=0.001)
        
        # Try triangular arbitrage
        arb_triangular = find_triangular_arbitrage_opportunities(web3, min_profit_threshold=0.002)
        
        if not arb_triangular:
            # If no triangular opportunities found, try with lower threshold for testing
            logger.info("No triangular arbitrage opportunities found with 0.2% threshold, trying with 0.1%...")
            arb_triangular = find_triangular_arbitrage_opportunities(web3, min_profit_threshold=0.001)
        
        # Combine all arbitrage opportunities
        arb_opportunities = arb_direct + arb_triangular
        
        # Look for sandwich opportunities
        from strategy.sandwich import find_sandwich_attack_opportunities
        sandwich_opportunities = find_sandwich_attack_opportunities(web3, min_profit_threshold=0.002)
        
        # Filter out low profit opportunities
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
            
            # Log evaluation
            logger.info(f"Arbitrage evaluation: Expected profit ${opp_data.get('profit_usd', 0):.2f}, "
                        f"Adjusted profit ${profit_usd:.2f}, Confidence {confidence:.2f}")
            
            # Make execution decision
            should_execute = (profit_usd >= MIN_PROFIT_USD and confidence >= CONFIDENCE_THRESHOLD)
            
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
        
        # For direct arbitrage, use a higher confidence
        subtype = opportunity.get('subtype', 'direct')
        if subtype == 'direct':
            confidence = 0.8
        else:
            confidence = 0.7
        
        # Apply a simple discount factor
        adjusted_profit = expected_profit * 0.9  # 10% discount for risks
        
        # Make execution decision
        should_execute = (profit_usd >= MIN_PROFIT_USD)
        
        return should_execute, adjusted_profit, confidence
        
    elif opp_type == 'sandwich':
        estimated_profit = opp_data.get('estimated_profit', 0.0)
        profit_usd = opp_data.get('profit_usd', 0.0)
        
        # Sandwich attacks have more risk
        confidence = 0.6
        
        # Apply a larger discount factor
        adjusted_profit = estimated_profit * 0.8  # 20% discount for risks
        
        # Make execution decision
        should_execute = (profit_usd >= MIN_PROFIT_USD)
        
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

def simulate_execution(opportunity: Dict, services: Dict) -> Dict:
    """
    Simulate execution of an opportunity (dry run)
    
    Args:
        opportunity: Opportunity dictionary
        services: Dictionary of service connections
        
    Returns:
        Execution result dictionary
    """
    try:
        opp_type = opportunity.get('type', '')
        opp_data = opportunity.get('data', {})
        
        # Record transaction start
        now = datetime.now()
        
        # Simulate blockchain interaction with a delay
        logger.info(f"Simulating execution of {opp_type} opportunity...")
        time.sleep(2)  # Simulate blockchain delay
        
        # Set up result
        result = {
            'success': True,
            'tx_hash': f"0xsimulated{now.timestamp():.0f}",
            'executed_at': now,
            'profit_eth': 0.0,
            'gas_used': 150000,  # Simulated gas used
            'gas_price': 50 * 10**9,  # 50 Gwei
            'error': None
        }
        
        # Randomly simulate success/failure (90% success rate)
        import random
        success = random.random() < 0.9
        
        if success:
            if opp_type == 'arbitrage':
                # Use expected profit with a slight variation
                expected_profit = opp_data.get('expected_profit', 0.0)
                # Simulate actual profit (80-120% of expected)
                actual_profit = expected_profit * (0.8 + random.random() * 0.4)
                result['profit_eth'] = actual_profit
                
                logger.info(f"Simulated arbitrage executed successfully: "
                           f"{actual_profit:.6f} ETH profit")
                
            elif opp_type == 'sandwich':
                # Use estimated profit with a slight variation
                estimated_profit = opp_data.get('estimated_profit', 0.0)
                # Simulate actual profit (70-110% of estimated)
                actual_profit = estimated_profit * (0.7 + random.random() * 0.4)
                result['profit_eth'] = actual_profit
                
                # For sandwich attacks, we need front and back transaction hashes
                result['front_tx_hash'] = f"0xfront{now.timestamp():.0f}"
                result['back_tx_hash'] = f"0xback{now.timestamp():.0f}"
                
                logger.info(f"Simulated sandwich attack executed successfully: "
                           f"{actual_profit:.6f} ETH profit")
        else:
            # Simulate failure
            result['success'] = False
            result['error'] = "Transaction reverted: insufficient liquidity"
            result['profit_eth'] = 0.0
            
            logger.warning(f"Simulated {opp_type} execution failed: {result['error']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error simulating execution: {e}")
        return {
            'success': False,
            'tx_hash': None,
            'executed_at': datetime.now(),
            'profit_eth': 0.0,
            'gas_used': 0,
            'gas_price': 0,
            'error': str(e)
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

def main_loop():
    """Main bot execution loop"""
    global running, stop_requested, services
    
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
                            # Simulate execution (dry run)
                            result = simulate_execution(opp, services)
                            
                            # Store execution result
                            store_execution_result(opp, result, services)
                    
                    except Exception as e:
                        logger.error(f"Error processing opportunity: {e}")
                        traceback.print_exc()
                
                # Wait before next scan
                time.sleep(SCAN_INTERVAL)
                
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
    global running, stop_requested, services
    
    if running:
        logger.warning("Bot already running")
        return
    
    # Connect to services
    if not services:
        services = connect_to_services()
        
    if not services:
        logger.error("Failed to connect to required services")
        return
    
    # Set running flag
    running = True
    stop_requested = False
    
    # Start main loop in a separate thread
    thread = threading.Thread(target=main_loop)
    thread.daemon = True
    thread.start()
    
    logger.info("Bot started in dry run mode")
    
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
        services['mempool_listener'].stop_listening()
    
    logger.info("Bot stop requested")

if __name__ == "__main__":
    try:
        # Connect to services
        services = connect_to_services()
        
        if not services:
            logger.error("Failed to connect to required services")
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
