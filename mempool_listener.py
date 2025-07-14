"""
Mempool listener for monitoring pending transactions
"""

import os
import logging
import threading
import time
import json
from typing import Dict, List, Optional, Any, Callable
from web3 import Web3
from web3.types import TxReceipt

from utils.web3_helpers import get_web3_provider

logger = logging.getLogger(__name__)

class MempoolListener:
    """
    Listens to the mempool for pending transactions
    """
    
    def __init__(self, db=None):
        """
        Initialize mempool listener
        
        Args:
            db: Database connection
        """
        self.web3 = None
        self.db = db
        self.running = False
        self.listener_thread = None
        self.callbacks = []
        self.monitored_tx_types = set(['swap', 'liquidity', 'large_transfer'])
        self.tx_history = {}  # Store transaction history
        self.tx_filters = {}  # Filters for transactions
        
        # Initialize web3 connection
        self._connect_to_node()
    
    def _connect_to_node(self) -> bool:
        """
        Connect to Ethereum node
        
        Returns:
            True if connection successful, False otherwise
        """
        self.web3 = get_web3_provider()
        if self.web3 and self.web3.is_connected():
            logger.info("Connected to Ethereum node")
            return True
        else:
            logger.error("Failed to connect to Ethereum node")
            return False
    
    def register_callback(self, callback: Callable[[Dict], None]):
        """
        Register callback for transaction events
        
        Args:
            callback: Callback function that takes a transaction dict
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)
            logger.info(f"Registered callback: {callback.__name__}")
    
    def unregister_callback(self, callback: Callable[[Dict], None]):
        """
        Unregister callback
        
        Args:
            callback: Callback function to unregister
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.info(f"Unregistered callback: {callback.__name__}")
    
    def start_listening(self):
        """Start listening to mempool"""
        if self.running:
            logger.warning("Mempool listener is already running")
            return
            
        if not self.web3 or not self.web3.is_connected():
            if not self._connect_to_node():
                logger.error("Cannot start mempool listener: Not connected to Ethereum node")
                return
        
        self.running = True
        
        # Start listening thread
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()
        
        logger.info("Started mempool listener")
    
    def stop_listening(self):
        """Stop listening to mempool"""
        if not self.running:
            logger.warning("Mempool listener is not running")
            return
            
        self.running = False
        
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=2.0)
            
        logger.info("Stopped mempool listener")
    
    def _listen_loop(self):
        """Main listening loop"""
        try:
            # Create new filter for pending transactions
            pending_filter = self.web3.eth.filter('pending')
            
            while self.running:
                try:
                    # Check for new pending transactions
                    new_pending = pending_filter.get_new_entries()
                    
                    for tx_hash in new_pending:
                        try:
                            # Get transaction details
                            tx = self.web3.eth.get_transaction(tx_hash)
                            
                            if tx:
                                # Process transaction
                                self._process_transaction(tx)
                                
                        except Exception as e:
                            logger.error(f"Error processing transaction {tx_hash}: {e}")
                            
                    # Sleep to avoid high CPU usage
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in mempool listener loop: {e}")
                    time.sleep(1.0)  # Longer sleep on error
                    
        except Exception as e:
            logger.error(f"Fatal error in mempool listener: {e}")
            self.running = False
    
    def _process_transaction(self, tx: Dict):
        """
        Process a transaction and notify callbacks if relevant
        
        Args:
            tx: Transaction dict from web3
        """
        try:
            # Skip if already processed
            tx_hash = tx.get('hash', '').hex()
            if tx_hash in self.tx_history:
                return
                
            # Store in history
            self.tx_history[tx_hash] = {
                'processed_at': time.time(),
                'status': 'pending'
            }
            
            # Check if transaction matches our filters
            tx_type = self._classify_transaction(tx)
            
            if tx_type and tx_type in self.monitored_tx_types:
                # Get additional data for relevant transaction
                tx_data = self._extract_transaction_data(tx, tx_type)
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(tx_data)
                    except Exception as e:
                        logger.error(f"Error in callback {callback.__name__}: {e}")
                
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
    
    def _classify_transaction(self, tx: Dict) -> Optional[str]:
        """
        Classify transaction type
        
        Args:
            tx: Transaction dict from web3
            
        Returns:
            Transaction type or None if not interesting
        """
        # Default to None (not interesting)
        tx_type = None
        
        # Check for empty transactions
        if not tx or not tx.get('input') or tx.get('input') == '0x':
            return None
            
        input_data = tx.get('input', '')
        
        # Check for swap transactions
        if input_data.startswith('0x38ed1739') or input_data.startswith('0x7ff36ab5'):
            # swapExactTokensForTokens or swapExactETHForTokens on Uniswap/Sushiswap
            tx_type = 'swap'
        elif input_data.startswith('0x18cbafe5') or input_data.startswith('0x4a25d94a'):
            # swapExactTokensForETH or swapTokensForExactETH
            tx_type = 'swap'
        elif input_data.startswith('0xf305d719') or input_data.startswith('0xe8e33700'):
            # addLiquidity or addLiquidityETH
            tx_type = 'liquidity'
        elif input_data.startswith('0xa9059cbb') or input_data.startswith('0x095ea7b3'):
            # transfer or approve
            if tx.get('value', 0) > Web3.to_wei(10, 'ether'):
                tx_type = 'large_transfer'
                
        return tx_type
    
    def _extract_transaction_data(self, tx: Dict, tx_type: str) -> Dict:
        """
        Extract detailed data from transaction
        
        Args:
            tx: Transaction dict from web3
            tx_type: Transaction type
            
        Returns:
            Detailed transaction data
        """
        tx_data = {
            'hash': tx.get('hash', '').hex(),
            'from': tx.get('from', ''),
            'to': tx.get('to', ''),
            'value': self.web3.from_wei(tx.get('value', 0), 'ether'),
            'gas': tx.get('gas', 0),
            'gas_price': self.web3.from_wei(tx.get('gasPrice', 0), 'gwei'),
            'nonce': tx.get('nonce', 0),
            'block_number': tx.get('blockNumber'),
            'type': tx_type,
            'timestamp': time.time()
        }
        
        # Add more details based on transaction type
        if tx_type == 'swap':
            # Extract swap details from input data
            tx_data['swap_details'] = self._extract_swap_details(tx)
        elif tx_type == 'liquidity':
            # Extract liquidity details from input data
            tx_data['liquidity_details'] = self._extract_liquidity_details(tx)
        elif tx_type == 'large_transfer':
            # Extract transfer details from input data
            tx_data['transfer_details'] = self._extract_transfer_details(tx)
            
        return tx_data
    
    def _extract_swap_details(self, tx: Dict) -> Dict:
        """
        Extract details from swap transaction
        
        Args:
            tx: Transaction dict from web3
            
        Returns:
            Swap details dict
        """
        # This is a simplified version - in real implementation,
        # you would decode the input data to extract exact token addresses,
        # amounts, and paths
        input_data = tx.get('input', '')
        
        if input_data.startswith('0x38ed1739'):
            # swapExactTokensForTokens
            method = 'swapExactTokensForTokens'
        elif input_data.startswith('0x7ff36ab5'):
            # swapExactETHForTokens
            method = 'swapExactETHForTokens'
        elif input_data.startswith('0x18cbafe5'):
            # swapExactTokensForETH
            method = 'swapExactTokensForETH'
        elif input_data.startswith('0x4a25d94a'):
            # swapTokensForExactETH
            method = 'swapTokensForExactETH'
        else:
            method = 'unknown'
        
        return {
            'method': method,
            'input_data': input_data,
            'dex': self._identify_dex(tx.get('to', ''))
        }
    
    def _extract_liquidity_details(self, tx: Dict) -> Dict:
        """
        Extract details from liquidity transaction
        
        Args:
            tx: Transaction dict from web3
            
        Returns:
            Liquidity details dict
        """
        input_data = tx.get('input', '')
        
        if input_data.startswith('0xf305d719'):
            # addLiquidity
            method = 'addLiquidity'
        elif input_data.startswith('0xe8e33700'):
            # addLiquidityETH
            method = 'addLiquidityETH'
        else:
            method = 'unknown'
        
        return {
            'method': method,
            'input_data': input_data,
            'dex': self._identify_dex(tx.get('to', ''))
        }
    
    def _extract_transfer_details(self, tx: Dict) -> Dict:
        """
        Extract details from transfer transaction
        
        Args:
            tx: Transaction dict from web3
            
        Returns:
            Transfer details dict
        """
        input_data = tx.get('input', '')
        
        if input_data.startswith('0xa9059cbb'):
            # transfer
            method = 'transfer'
        elif input_data.startswith('0x095ea7b3'):
            # approve
            method = 'approve'
        else:
            method = 'unknown'
        
        return {
            'method': method,
            'input_data': input_data,
            'value': self.web3.from_wei(tx.get('value', 0), 'ether')
        }
    
    def _identify_dex(self, address: str) -> str:
        """
        Identify DEX from router address
        
        Args:
            address: Router address
            
        Returns:
            DEX name or 'unknown'
        """
        if not address:
            return 'unknown'
            
        address = address.lower()
        
        # Common DEX router addresses
        dex_routers = {
            '0x7a250d5630b4cf539739df2c5dacb4c659f2488d': 'uniswap_v2',
            '0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f': 'sushiswap',
            '0xe592427a0aece92de3edee1f18e0157c05861564': 'uniswap_v3',
            '0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45': 'uniswap_v3_router2',
            '0x1111111254fb6c44bac0bed2854e76f90643097d': 'oneinch',
            '0xdef171fe48cf0115b1d80b88dc8eab59176fee57': 'paraswap',
            '0x11111112542d85b3ef69ae05771c2dccff4faa26': 'oneinch_v2',
            '0xf164fc0ec4e93095b804a4795bbe1e041497b92a': 'oneswap',
            '0x881d40237659c251811cec9c364ef91dc08d300c': 'metamask_swap'
        }
        
        return dex_routers.get(address, 'unknown')

# Singleton instance
_listener = None

def get_mempool_listener(db=None) -> MempoolListener:
    """
    Get singleton instance of mempool listener
    
    Args:
        db: Database connection
        
    Returns:
        MempoolListener instance
    """
    global _listener
    if _listener is None:
        _listener = MempoolListener(db)
    return _listener

def start_listening():
    """Start listening to mempool"""
    listener = get_mempool_listener()
    listener.start_listening()
    
def stop_listening():
    """Stop listening to mempool"""
    if _listener:
        _listener.stop_listening()