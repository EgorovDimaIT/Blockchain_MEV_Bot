"""
Main app file for the MEV bot web interface
"""

import os
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json

from flask import Flask, render_template, jsonify, request, redirect, url_for

from utils.web3_helpers import get_web3, get_eth_price, get_gas_price
from strategy.strategy_engine import init_strategy_engine, start_engine, stop_engine, get_strategies, get_cached_opportunities, scan_opportunities, update_strategy_config
# Import the database instance from the database module
from database import db, init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")  # Replace with actual secret key in production

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:postgres@localhost:5432/mev_bot"
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the database with the app
init_db(app)

# Status variables
bot_status = {
    'running': False,
    'status': 'stopped',
    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'network': 'mainnet',
    'wallet_balance_eth': 0.0,
    'wallet_balance_usd': 0.0,
    'transactions_count': 0,
    'opportunities_found': 0,
    'current_block': 0,
    'current_gas_price': 0.0,
    'eth_price_usd': 0.0,
    'errors': []
}

performance_metrics = {
    'total_profit_eth': 0.0,
    'total_profit_usd': 0.0,
    'average_profit_per_tx_eth': 0.0,
    'average_profit_per_tx_usd': 0.0,
    'success_rate': 0.0,
    'transactions_per_hour': 0.0,
    'gas_used_total': 0.0,
    'gas_used_average': 0.0,
    'strategy_performance': {},
    'total_opportunities': 0,
    'executed_opportunities': 0,
    'successful_transactions': 0,
}

# Background status update function
def update_bot_status():
    """Update bot status with latest blockchain info"""
    global bot_status
    
    try:
        # Get web3 connection
        web3 = get_web3()
        if not web3 or not web3.is_connected():
            bot_status['errors'] = ["Cannot connect to Ethereum node. Check your Infura API key."]
            return
            
        # Update ETH price
        eth_price = get_eth_price() or 0.0
        bot_status['eth_price_usd'] = eth_price
        
        # Update gas price
        gas_prices = get_gas_price(web3)
        bot_status['current_gas_price'] = web3.from_wei(gas_prices[1], 'gwei')  # Use 'fast' gas price
        
        # Update current block
        bot_status['current_block'] = web3.eth.block_number
        
        # Update wallet balance if wallet address is set
        wallet_address = os.environ.get('WALLET_ADDRESS')
        if wallet_address:
            try:
                balance_wei = web3.eth.get_balance(wallet_address)
                balance_eth = web3.from_wei(balance_wei, 'ether')
                bot_status['wallet_balance_eth'] = float(balance_eth)
                bot_status['wallet_balance_usd'] = float(balance_eth) * eth_price
            except Exception as e:
                logger.error(f"Error getting wallet balance: {e}")
                
        # Update network
        chain_id = web3.eth.chain_id
        if chain_id == 1:
            bot_status['network'] = 'mainnet'
        elif chain_id == 5:
            bot_status['network'] = 'goerli'
        elif chain_id == 11155111:
            bot_status['network'] = 'sepolia'
        else:
            bot_status['network'] = f'chain_id_{chain_id}'
            
        # Update last update time
        bot_status['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Clear errors if successful
        bot_status['errors'] = []
    except Exception as e:
        logger.error(f"Error updating bot status: {e}")
        bot_status['errors'] = [str(e)]
        
def update_performance_metrics():
    """Update performance metrics from database"""
    global performance_metrics
    
    try:
        # Import models here to avoid circular import
        from models import Transaction
        
        # Use app context to ensure proper database connection
        with app.app_context():
            try:
                # Calculate total profit
                successful_txs = Transaction.query.filter_by(status='confirmed').all()
                
                if not successful_txs:
                    return
                    
                total_profit_eth = sum(tx.profit_eth or 0 for tx in successful_txs)
                total_eth_spent = sum(
                    (tx.gas_used or 0) * (tx.gas_price or 0) / 1e18 
                    for tx in successful_txs if tx.gas_used and tx.gas_price
                )
                
                net_profit_eth = total_profit_eth - total_eth_spent
                eth_price = bot_status['eth_price_usd']
                
                # Calculate performance metrics
                performance_metrics['total_profit_eth'] = net_profit_eth
                performance_metrics['total_profit_usd'] = net_profit_eth * eth_price
                
                tx_count = len(successful_txs)
                if tx_count > 0:
                    performance_metrics['average_profit_per_tx_eth'] = net_profit_eth / tx_count
                    performance_metrics['average_profit_per_tx_usd'] = (net_profit_eth / tx_count) * eth_price
                    
                    # Gas stats
                    gas_used_txs = [tx for tx in successful_txs if tx.gas_used]
                    if gas_used_txs:
                        performance_metrics['gas_used_total'] = sum(tx.gas_used or 0 for tx in gas_used_txs)
                        performance_metrics['gas_used_average'] = performance_metrics['gas_used_total'] / len(gas_used_txs)
                    
                # Success rate calculation
                all_txs = Transaction.query.all()
                if all_txs:
                    performance_metrics['success_rate'] = len(successful_txs) / len(all_txs) * 100
                
                # Calculate transactions per hour (based on last 24 hours)
                day_ago = datetime.utcnow() - timedelta(days=1)
                txs_24h = Transaction.query.filter(Transaction.created_at >= day_ago).all()
                
                if txs_24h:
                    performance_metrics['transactions_per_hour'] = len(txs_24h) / 24
                    
                # Strategy performance
                arbitrage_txs = Transaction.query.filter_by(strategy_type='arbitrage', status='confirmed').all()
                sandwich_txs = Transaction.query.filter_by(strategy_type='sandwich', status='confirmed').all()
                
                if arbitrage_txs:
                    arbitrage_profit = sum(tx.profit_eth or 0 for tx in arbitrage_txs)
                    performance_metrics['strategy_performance']['arbitrage'] = {
                        'count': len(arbitrage_txs),
                        'profit_eth': arbitrage_profit,
                        'profit_usd': arbitrage_profit * eth_price
                    }
                    
                if sandwich_txs:
                    sandwich_profit = sum(tx.profit_eth or 0 for tx in sandwich_txs)
                    performance_metrics['strategy_performance']['sandwich'] = {
                        'count': len(sandwich_txs),
                        'profit_eth': sandwich_profit,
                        'profit_usd': sandwich_profit * eth_price
                    }
            except Exception as e:
                logger.error(f"Database query error in update_performance_metrics: {e}")
    except Exception as e:
        logger.error(f"Error updating performance metrics: {e}")
        
def status_updater():
    """Background thread to update bot status and metrics"""
    while True:
        # Wrap everything in a try-except to make sure the thread doesn't die
        try:
            # Update status with Ethereum info
            try:
                update_bot_status()
            except Exception as e:
                logger.error(f"Error updating bot status: {e}")
                bot_status['errors'].append(str(e))
            
            # Update database metrics
            try:
                with app.app_context():
                    update_performance_metrics()
            except Exception as e:
                logger.error(f"Error updating performance metrics: {e}")
                bot_status['errors'].append(str(e))
        except Exception as e:
            logger.error(f"Critical error in status updater thread: {e}")
            
        # Sleep for 15 seconds
        time.sleep(15)
        
# Import models after database initialization
from models import Transaction, ArbitrageOpportunity, SandwichOpportunity, Setting, MLModel, init_default_settings

with app.app_context():
    db.create_all()
    init_default_settings()
    
# Initialize strategy engine
init_strategy_engine()

# Start status updater thread with app context
app.app_context().push()  # Push app context to ensure database connection is available
status_thread = threading.Thread(target=status_updater, daemon=True)
status_thread.start()

# Define routes
@app.route('/')
def index():
    """Home page with bot status overview"""
    strategies = get_strategies()
    opportunities = get_cached_opportunities()
    return render_template(
        'index.html',
        bot_status=bot_status,
        performance_metrics=performance_metrics,
        strategies=strategies,
        opportunities=opportunities[:5] if opportunities else []
    )
    
@app.route('/dashboard')
def dashboard():
    """Detailed dashboard with performance metrics and charts"""
    # Models already imported at the top of the file
    
    try:
        recent_txs = Transaction.query.order_by(Transaction.created_at.desc()).limit(10).all()
        
        # Prepare data for charts
        profit_data = []
        for tx in Transaction.query.filter_by(status='confirmed').order_by(Transaction.executed_at).all():
            if tx.executed_at and tx.profit_eth:
                profit_data.append({
                    'date': tx.executed_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'profit_eth': float(tx.profit_eth),
                    'profit_usd': float(tx.profit_eth) * bot_status['eth_price_usd']
                })
        
        return render_template(
            'dashboard.html',
            bot_status=bot_status,
            performance_metrics=performance_metrics,
            recent_transactions=recent_txs,
            profit_data=json.dumps(profit_data)
        )
    except Exception as e:
        logger.error(f"Error in dashboard route: {e}")
        return render_template(
            'dashboard.html',
            bot_status=bot_status,
            performance_metrics=performance_metrics,
            recent_transactions=[],
            profit_data=json.dumps([]),
            error=str(e)
        )
    
@app.route('/opportunities')
def opportunities():
    """View arbitrage and sandwich opportunities"""
    # Get cached opportunities
    all_opportunities = get_cached_opportunities()
    direct_arb_opps = get_cached_opportunities('direct_arbitrage')
    triangular_arb_opps = get_cached_opportunities('triangular_arbitrage')
    sandwich_opps = get_cached_opportunities('sandwich')
    
    return render_template(
        'opportunities.html',
        bot_status=bot_status,
        all_opportunities=all_opportunities,
        direct_arbitrage=direct_arb_opps,
        triangular_arbitrage=triangular_arb_opps,
        sandwich=sandwich_opps
    )
    
@app.route('/transactions')
def transactions():
    """View all transactions"""
    # Models already imported at the top of the file
    
    try:
        txs = Transaction.query.order_by(Transaction.created_at.desc()).all()
        
        return render_template(
            'transactions.html',
            bot_status=bot_status,
            transactions=txs
        )
    except Exception as e:
        logger.error(f"Error in transactions route: {e}")
        return render_template(
            'transactions.html',
            bot_status=bot_status,
            transactions=[],
            error=str(e)
        )
    
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """View and update bot settings"""
    if request.method == 'POST':
        try:
            # Update strategy settings
            for strategy in get_strategies():
                if f"{strategy}_enabled" in request.form:
                    enabled = request.form.get(f"{strategy}_enabled") == 'on'
                    threshold = float(request.form.get(f"{strategy}_threshold", 0.002))
                    interval = int(request.form.get(f"{strategy}_interval", 60))
                    execution = request.form.get(f"{strategy}_execution") == 'on'
                    
                    update_strategy_config(strategy, {
                        'enabled': enabled,
                        'min_profit_threshold': threshold,
                        'scan_interval': interval,
                        'execution_enabled': execution
                    })
                    
            return redirect(url_for('settings'))
        except Exception as e:
            return render_template(
                'settings.html',
                bot_status=bot_status,
                strategies=get_strategies(),
                error=str(e)
            )
    
    return render_template(
        'settings.html',
        bot_status=bot_status,
        strategies=get_strategies(),
        error=None
    )
    
# API routes
@app.route('/api/start', methods=['POST'])
def start_bot():
    """API endpoint to start the bot"""
    global bot_status
    
    if bot_status['running']:
        return jsonify({'success': False, 'message': 'Bot is already running'})
        
    try:
        success = start_engine()
        if success:
            bot_status['running'] = True
            bot_status['status'] = 'running'
            return jsonify({'success': True, 'message': 'Bot started successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to start bot'})
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})
    
@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """API endpoint to stop the bot"""
    global bot_status
    
    if not bot_status['running']:
        return jsonify({'success': False, 'message': 'Bot is not running'})
        
    try:
        success = stop_engine()
        if success:
            bot_status['running'] = False
            bot_status['status'] = 'stopped'
            return jsonify({'success': True, 'message': 'Bot stopped successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to stop bot'})
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})
    
@app.route('/api/status', methods=['GET'])
def api_status():
    """API endpoint to get current bot status"""
    return jsonify({
        'bot_status': bot_status,
        'performance_metrics': performance_metrics
    })
    
@app.route('/api/scan', methods=['POST'])
def api_scan_opportunities():
    """API endpoint to manually scan for opportunities"""
    try:
        strategy_type = request.json.get('strategy_type') if request.is_json else None
        results = scan_opportunities(strategy_type)
        
        # Update opportunities count
        opportunity_count = sum(len(opps) for opps in results.values())
        if opportunity_count > 0:
            bot_status['opportunities_found'] += opportunity_count
        
        return jsonify({
            'success': True,
            'opportunities_found': opportunity_count,
            'results': {k: len(v) for k, v in results.items()}
        })
    except Exception as e:
        logger.error(f"Error scanning for opportunities: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})
    
# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404
    
@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # This block is only executed when running the file directly
    # It's not executed when the file is imported
    app.run(host='0.0.0.0', port=5000, debug=True)