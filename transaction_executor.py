"""
Transaction execution engine for MEV bot.
Handles transaction building, gas optimization, simulations, and execution.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from web3 import Web3
from eth_account import Account
from web3.exceptions import ContractLogicError

# Setup logger
logger = logging.getLogger(__name__)

def build_arbitrage_tx(web3: Web3, opportunity: Dict, wallet_address: str) -> Dict:
    """
    Build transaction data for arbitrage execution
    
    Args:
        web3: Web3 instance
        opportunity: Arbitrage opportunity dict
        wallet_address: Wallet address
        
    Returns:
        Transaction data
    """
    from utils.token_utils import get_token_approval, ERC20_ABI
    from config import DEX_ADDRESSES
    
    # Uniswap V2 Router ABI (partial, only swap function)
    router_abi = [
        {
            "inputs": [
                {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                {"internalType": "address[]", "name": "path", "type": "address[]"},
                {"internalType": "address", "name": "to", "type": "address"},
                {"internalType": "uint256", "name": "deadline", "type": "uint256"}
            ],
            "name": "swapExactTokensForTokens",
            "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
            "stateMutability": "nonpayable",
            "type": "function"
        }
    ]
    
    try:
        if opportunity['arbitrage_type'] == 'direct':
            # For direct arbitrage, we do two swaps
            
            # Get token addresses and amounts
            token_in = opportunity['token_in']
            token_out = opportunity['token_out']
            amount_in = int(opportunity['amount_in'])
            
            # Get router addresses
            dex1_name = opportunity['dex_1']
            dex2_name = opportunity['dex_2']
            
            if dex1_name == 'Uniswap V2':
                router1_address = DEX_ADDRESSES['Uniswap V2 Router']
            elif dex1_name == 'Uniswap V3':
                router1_address = DEX_ADDRESSES['Uniswap V3 Router']
            elif dex1_name == 'Sushiswap':
                router1_address = DEX_ADDRESSES['Sushiswap Router']
            else:
                raise ValueError(f"Unsupported DEX: {dex1_name}")
                
            if dex2_name == 'Uniswap V2':
                router2_address = DEX_ADDRESSES['Uniswap V2 Router']
            elif dex2_name == 'Uniswap V3':
                router2_address = DEX_ADDRESSES['Uniswap V3 Router']
            elif dex2_name == 'Sushiswap':
                router2_address = DEX_ADDRESSES['Sushiswap Router']
            else:
                raise ValueError(f"Unsupported DEX: {dex2_name}")
            
            # Create router contracts
            router1 = web3.eth.contract(address=router1_address, abi=router_abi)
            router2 = web3.eth.contract(address=router2_address, abi=router_abi)
            
            # Check token approval for DEX 1
            token_contract = web3.eth.contract(address=token_in, abi=ERC20_ABI)
            allowance = get_token_approval(web3, token_in, wallet_address, router1_address)
            
            tx_data = {}
            
            if allowance < amount_in:
                # Need to approve first
                approve_data = token_contract.functions.approve(
                    router1_address, 
                    amount_in
                ).build_transaction({
                    'from': wallet_address,
                    'nonce': web3.eth.get_transaction_count(wallet_address),
                    'gas': 100000,
                    'gasPrice': web3.eth.gas_price
                })
                
                tx_data['approve_tx'] = approve_data
            
            # Calculate minimum amount out with 1% slippage
            path1 = [token_in, token_out]
            amount_out = int(opportunity['amount_out_1'])
            amount_out_min = int(amount_out * 0.99)  # 1% slippage
            deadline = int(time.time() + 300)  # 5 minutes
            
            # Build first swap transaction
            swap1_data = router1.functions.swapExactTokensForTokens(
                amount_in,
                amount_out_min,
                path1,
                wallet_address,  # Receive tokens to same wallet
                deadline
            ).build_transaction({
                'from': wallet_address,
                'nonce': web3.eth.get_transaction_count(wallet_address) + (1 if 'approve_tx' in tx_data else 0),
                'gas': 300000,
                'gasPrice': web3.eth.gas_price
            })
            
            tx_data['swap1_tx'] = swap1_data
            
            # We need to approve DEX 2 to spend token_out if needed
            allowance_out = get_token_approval(web3, token_out, wallet_address, router2_address)
            
            if allowance_out < amount_out:
                # Need to approve second router
                token_out_contract = web3.eth.contract(address=token_out, abi=ERC20_ABI)
                approve_out_data = token_out_contract.functions.approve(
                    router2_address, 
                    amount_out
                ).build_transaction({
                    'from': wallet_address,
                    'nonce': web3.eth.get_transaction_count(wallet_address) + 
                            (1 if 'approve_tx' in tx_data else 0) + 
                            1,  # After swap1
                    'gas': 100000,
                    'gasPrice': web3.eth.gas_price
                })
                
                tx_data['approve_out_tx'] = approve_out_data
            
            # Path for second swap
            path2 = [token_out, token_in]
            final_amount_min = int(amount_in * 1.001)  # Ensure profitable return
            
            # Build second swap transaction
            swap2_data = router2.functions.swapExactTokensForTokens(
                amount_out,
                final_amount_min,
                path2,
                wallet_address,
                deadline
            ).build_transaction({
                'from': wallet_address,
                'nonce': web3.eth.get_transaction_count(wallet_address) + 
                        (1 if 'approve_tx' in tx_data else 0) + 
                        1 +  # After swap1
                        (1 if 'approve_out_tx' in tx_data else 0),
                'gas': 300000,
                'gasPrice': web3.eth.gas_price
            })
            
            tx_data['swap2_tx'] = swap2_data
            
            return tx_data
            
        elif opportunity['arbitrage_type'] == 'triangular':
            # For triangular arbitrage, we need a flash loan contract
            # This is more complex and would need a dedicated contract
            # Simplified here to just return the data
            
            tx_data = {
                'arbitrage_type': 'triangular',
                'token_in': opportunity['token_in'],
                'token_mid': opportunity['token_mid'],
                'token_out': opportunity['token_out'],
                'amount_in': opportunity['amount_in'],
                'dex_1': opportunity['dex_1'],
                'dex_2': opportunity['dex_2'],
                'dex_3': opportunity['dex_3'],
                'flash_loan_source': opportunity['flash_loan_source'],
                'wallet_address': wallet_address
            }
            
            return tx_data
            
        else:
            raise ValueError(f"Unsupported arbitrage type: {opportunity['arbitrage_type']}")
            
    except Exception as e:
        logger.error(f"Error building arbitrage transaction: {e}")
        raise

def simulate_transaction(web3: Web3, tx_data: Dict) -> Dict:
    """
    Simulate a transaction to check if it will succeed
    
    Args:
        web3: Web3 instance
        tx_data: Transaction data dict
        
    Returns:
        Result dict with success flag
    """
    result = {
        'success': False,
        'error': None,
        'gas_estimate': 0
    }
    
    try:
        # If this is a transaction object
        if all(k in tx_data for k in ['to', 'from', 'gas', 'gasPrice']):
            # Use eth_call to simulate
            call_result = web3.eth.call({
                'to': tx_data['to'],
                'from': tx_data['from'],
                'data': tx_data['data'],
                'value': tx_data.get('value', 0)
            })
            
            # Estimate gas to make sure it will execute
            gas_estimate = web3.eth.estimate_gas({
                'to': tx_data['to'],
                'from': tx_data['from'],
                'data': tx_data['data'],
                'value': tx_data.get('value', 0)
            })
            
            result['success'] = True
            result['gas_estimate'] = gas_estimate
            return result
            
        # If this is a set of transactions (like arbitrage)
        elif 'approve_tx' in tx_data or 'swap1_tx' in tx_data:
            # Simulate each transaction in order
            total_gas = 0
            
            if 'approve_tx' in tx_data:
                approve_call = web3.eth.call({
                    'to': tx_data['approve_tx']['to'],
                    'from': tx_data['approve_tx']['from'],
                    'data': tx_data['approve_tx']['data']
                })
                
                gas_estimate = web3.eth.estimate_gas({
                    'to': tx_data['approve_tx']['to'],
                    'from': tx_data['approve_tx']['from'],
                    'data': tx_data['approve_tx']['data']
                })
                
                total_gas += gas_estimate
            
            if 'swap1_tx' in tx_data:
                swap1_call = web3.eth.call({
                    'to': tx_data['swap1_tx']['to'],
                    'from': tx_data['swap1_tx']['from'],
                    'data': tx_data['swap1_tx']['data']
                })
                
                gas_estimate = web3.eth.estimate_gas({
                    'to': tx_data['swap1_tx']['to'],
                    'from': tx_data['swap1_tx']['from'],
                    'data': tx_data['swap1_tx']['data']
                })
                
                total_gas += gas_estimate
            
            if 'approve_out_tx' in tx_data:
                approve_out_call = web3.eth.call({
                    'to': tx_data['approve_out_tx']['to'],
                    'from': tx_data['approve_out_tx']['from'],
                    'data': tx_data['approve_out_tx']['data']
                })
                
                gas_estimate = web3.eth.estimate_gas({
                    'to': tx_data['approve_out_tx']['to'],
                    'from': tx_data['approve_out_tx']['from'],
                    'data': tx_data['approve_out_tx']['data']
                })
                
                total_gas += gas_estimate
            
            if 'swap2_tx' in tx_data:
                swap2_call = web3.eth.call({
                    'to': tx_data['swap2_tx']['to'],
                    'from': tx_data['swap2_tx']['from'],
                    'data': tx_data['swap2_tx']['data']
                })
                
                gas_estimate = web3.eth.estimate_gas({
                    'to': tx_data['swap2_tx']['to'],
                    'from': tx_data['swap2_tx']['from'],
                    'data': tx_data['swap2_tx']['data']
                })
                
                total_gas += gas_estimate
            
            result['success'] = True
            result['gas_estimate'] = total_gas
            return result
            
        # If this is a triangular arbitrage or other special case
        else:
            # For complex cases, just return a basic success
            # In a real implementation, a more detailed simulation would be needed
            result['success'] = True
            result['gas_estimate'] = 800000  # Approximate high estimate
            return result
            
    except ContractLogicError as e:
        error_message = str(e)
        result['error'] = f"Contract logic error: {error_message}"
        return result
    except Exception as e:
        result['error'] = f"Simulation error: {str(e)}"
        return result

def execute_transaction(web3: Web3, tx_data: Dict, private_key: str, options: Dict = None) -> Dict:
    """
    Execute a transaction or set of transactions
    
    Args:
        web3: Web3 instance
        tx_data: Transaction data dict
        private_key: Private key for signing
        options: Additional options
        
    Returns:
        Result dict with success flag and transaction details
    """
    options = options or {}
    max_gas_price = options.get('max_gas_price', 100)  # Default 100 Gwei
    
    result = {
        'success': False,
        'tx_hash': None,
        'gas_used': 0,
        'gas_price': 0,
        'error': None
    }
    
    try:
        # If this is a single transaction
        if all(k in tx_data for k in ['to', 'from', 'gas', 'gasPrice']):
            # Make sure gas price doesn't exceed maximum
            if web3.from_wei(tx_data['gasPrice'], 'gwei') > max_gas_price:
                tx_data['gasPrice'] = web3.to_wei(max_gas_price, 'gwei')
            
            # Sign and send
            signed_tx = web3.eth.account.sign_transaction(tx_data, private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            
            result['success'] = receipt.status == 1
            result['tx_hash'] = tx_hash.hex()
            result['gas_used'] = receipt.gasUsed
            result['gas_price'] = tx_data['gasPrice']
            
            if receipt.status != 1:
                result['error'] = "Transaction failed"
            
            return result
            
        # If this is arbitrage with multiple transactions
        elif 'swap1_tx' in tx_data:
            total_gas_used = 0
            tx_hashes = []
            
            # First approval if needed
            if 'approve_tx' in tx_data:
                # Cap gas price
                if web3.from_wei(tx_data['approve_tx']['gasPrice'], 'gwei') > max_gas_price:
                    tx_data['approve_tx']['gasPrice'] = web3.to_wei(max_gas_price, 'gwei')
                
                signed_tx = web3.eth.account.sign_transaction(tx_data['approve_tx'], private_key)
                tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
                
                if receipt.status != 1:
                    result['error'] = "Approval transaction failed"
                    return result
                
                total_gas_used += receipt.gasUsed
                tx_hashes.append(tx_hash.hex())
                
                # Update nonce for next transaction
                new_nonce = web3.eth.get_transaction_count(tx_data['approve_tx']['from'])
                tx_data['swap1_tx']['nonce'] = new_nonce
            
            # Execute first swap
            if web3.from_wei(tx_data['swap1_tx']['gasPrice'], 'gwei') > max_gas_price:
                tx_data['swap1_tx']['gasPrice'] = web3.to_wei(max_gas_price, 'gwei')
            
            signed_tx = web3.eth.account.sign_transaction(tx_data['swap1_tx'], private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status != 1:
                result['error'] = "First swap transaction failed"
                return result
            
            total_gas_used += receipt.gasUsed
            tx_hashes.append(tx_hash.hex())
            
            # Update nonce for next transaction
            new_nonce = web3.eth.get_transaction_count(tx_data['swap1_tx']['from'])
            
            # Second approval if needed
            if 'approve_out_tx' in tx_data:
                tx_data['approve_out_tx']['nonce'] = new_nonce
                
                if web3.from_wei(tx_data['approve_out_tx']['gasPrice'], 'gwei') > max_gas_price:
                    tx_data['approve_out_tx']['gasPrice'] = web3.to_wei(max_gas_price, 'gwei')
                
                signed_tx = web3.eth.account.sign_transaction(tx_data['approve_out_tx'], private_key)
                tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
                
                if receipt.status != 1:
                    result['error'] = "Second approval transaction failed"
                    return result
                
                total_gas_used += receipt.gasUsed
                tx_hashes.append(tx_hash.hex())
                
                # Update nonce for next transaction
                new_nonce = web3.eth.get_transaction_count(tx_data['approve_out_tx']['from'])
                tx_data['swap2_tx']['nonce'] = new_nonce
            else:
                tx_data['swap2_tx']['nonce'] = new_nonce
            
            # Execute second swap
            if web3.from_wei(tx_data['swap2_tx']['gasPrice'], 'gwei') > max_gas_price:
                tx_data['swap2_tx']['gasPrice'] = web3.to_wei(max_gas_price, 'gwei')
            
            signed_tx = web3.eth.account.sign_transaction(tx_data['swap2_tx'], private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status != 1:
                result['error'] = "Second swap transaction failed"
                return result
            
            total_gas_used += receipt.gasUsed
            tx_hashes.append(tx_hash.hex())
            
            # Success
            result['success'] = True
            result['tx_hash'] = tx_hashes[-1]  # Last transaction hash
            result['all_tx_hashes'] = tx_hashes
            result['gas_used'] = total_gas_used
            result['gas_price'] = int(tx_data['swap2_tx']['gasPrice'])
            
            return result
            
        # If this is triangular arbitrage
        elif 'arbitrage_type' in tx_data and tx_data['arbitrage_type'] == 'triangular':
            # This would use a specialized contract for triangular arbitrage
            # Placeholder for actual implementation
            result['success'] = False
            result['error'] = "Triangular arbitrage requires a flash loan contract"
            return result
            
        else:
            result['error'] = "Unsupported transaction format"
            return result
            
    except Exception as e:
        logger.error(f"Error executing transaction: {e}")
        result['error'] = str(e)
        return result
