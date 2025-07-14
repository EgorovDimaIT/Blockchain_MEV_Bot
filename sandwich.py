"""
Sandwich attack strategy implementation for MEV bot.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from web3 import Web3
from decimal import Decimal

# Setup logger
logger = logging.getLogger(__name__)

# Import utilities
from utils.web3_helpers import get_token_price_usd, get_eth_price_usd
from utils.token_utils import (
    get_token_balance, 
    get_token_decimals, 
    format_token_amount, 
    get_token_symbol,
    ERC20_ABI
)

# DEX Router ABIs for sandwich attacks
UNISWAP_V2_ROUTER_ABI = [
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
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
            {"internalType": "address[]", "name": "path", "type": "address[]"}
        ],
        "name": "getAmountsOut",
        "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function"
    }
]

# Uniswap V2 Pair ABI for getting reserves
UNISWAP_V2_PAIR_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"internalType": "uint112", "name": "_reserve0", "type": "uint112"},
            {"internalType": "uint112", "name": "_reserve1", "type": "uint112"},
            {"internalType": "uint32", "name": "_blockTimestampLast", "type": "uint32"}
        ],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "token0",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "token1",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    }
]

def find_sandwich_attack_opportunities(web3: Web3, min_profit_threshold: float = 0.002, pending_txs: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Find sandwich attack opportunities in pending transactions
    
    Args:
        web3: Web3 instance
        min_profit_threshold: Minimum profit threshold as percentage
        pending_txs: Optional list of pending transactions to analyze
        
    Returns:
        List of sandwich opportunity dictionaries
    """
    from config import TOKEN_ADDRESSES, DEX_ADDRESSES
    
    opportunities = []
    
    # Get pending transactions from mempool if not provided
    if not pending_txs:
        try:
            pending_txs = web3.eth.get_pending_transactions()
        except Exception as e:
            logger.error(f"Error getting pending transactions: {e}")
            return opportunities
    
    # Define DEX router contracts to check
    dex_routers = {
        'Uniswap V2': {
            'address': DEX_ADDRESSES['Uniswap V2 Router'],
            'contract': web3.eth.contract(address=DEX_ADDRESSES['Uniswap V2 Router'], abi=UNISWAP_V2_ROUTER_ABI)
        },
        'Sushiswap': {
            'address': DEX_ADDRESSES['Sushiswap Router'],
            'contract': web3.eth.contract(address=DEX_ADDRESSES['Sushiswap Router'], abi=UNISWAP_V2_ROUTER_ABI)
        }
    }
    
    # Define common tokens to look for in swaps
    target_tokens = [
        TOKEN_ADDRESSES['WETH'],
        TOKEN_ADDRESSES['USDC'],
        TOKEN_ADDRESSES['DAI'],
        TOKEN_ADDRESSES['USDT'],
        TOKEN_ADDRESSES['WBTC']
    ]
    
    # Analyze each pending transaction
    for tx in pending_txs:
        tx_hash = tx.get('hash', '').hex() if hasattr(tx.get('hash', ''), 'hex') else tx.get('hash', '')
        to_address = tx.get('to', '')
        from_address = tx.get('from', '')
        input_data = tx.get('input', '')
        gas_price = tx.get('gasPrice', 0)
        
        # Skip transactions with low gas price (unlikely to be included soon)
        if gas_price < web3.eth.gas_price * 0.9:
            continue
            
        # Check if transaction is to a DEX router
        target_dex = None
        for dex_name, dex_info in dex_routers.items():
            if to_address.lower() == dex_info['address'].lower():
                target_dex = dex_name
                router = dex_info['contract']
                break
        
        if not target_dex:
            continue
        
        # Try to decode the transaction input data as a swap
        try:
            # Decode function call
            func_obj, func_params = router.decode_function_input(input_data)
            
            # Check if it's a swap function
            if 'swap' in func_obj.fn_name.lower():
                # Get swap parameters
                path = func_params.get('path', [])
                
                # Skip if path is too short
                if len(path) < 2:
                    continue
                
                # Get token addresses from path
                token_in = path[0]
                token_out = path[-1]
                
                # Skip if not swapping a target token
                if not (token_in.lower() in [t.lower() for t in target_tokens] or
                        token_out.lower() in [t.lower() for t in target_tokens]):
                    continue
                
                amount_in = func_params.get('amountIn', 0)
                
                # Skip very small transactions
                if amount_in == 0:
                    continue
                
                # Get token details
                token_in_symbol = get_token_symbol(web3, token_in)
                token_out_symbol = get_token_symbol(web3, token_out)
                token_in_decimals = get_token_decimals(web3, token_in)
                
                # Format amount for logging
                formatted_amount = amount_in / 10**token_in_decimals
                
                logger.info(f"Found potential sandwich target: {tx_hash} on {target_dex}, swapping {formatted_amount} {token_in_symbol} for {token_out_symbol}")
                
                # Now calculate potential profit from sandwich attack
                # We need to find the Uniswap pair contract for these tokens
                uniswap_factory_address = '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f'  # Uniswap V2 Factory
                
                try:
                    # Create factory contract
                    factory_abi = [{"inputs":[{"internalType":"address","name":"tokenA","type":"address"},{"internalType":"address","name":"tokenB","type":"address"}],"name":"getPair","outputs":[{"internalType":"address","name":"pair","type":"address"}],"stateMutability":"view","type":"function"}]
                    factory = web3.eth.contract(address=uniswap_factory_address, abi=factory_abi)
                    
                    # Get pair address
                    pair_address = factory.functions.getPair(token_in, token_out).call()
                    
                    if pair_address and pair_address != '0x0000000000000000000000000000000000000000':
                        # Create pair contract
                        pair = web3.eth.contract(address=pair_address, abi=UNISWAP_V2_PAIR_ABI)
                        
                        # Get current reserves
                        reserves = pair.functions.getReserves().call()
                        token0 = pair.functions.token0().call()
                        
                        # Determine which reserve corresponds to which token
                        if token0.lower() == token_in.lower():
                            reserve_in = reserves[0]
                            reserve_out = reserves[1]
                        else:
                            reserve_in = reserves[1]
                            reserve_out = reserves[0]
                        
                        # Calculate price impact of victim transaction
                        price_impact = (amount_in * 997 * reserve_out) / (reserve_in * 1000 + amount_in * 997)
                        price_impact_percentage = price_impact / reserve_out
                        
                        # Skip if price impact is too small
                        if price_impact_percentage < 0.002:  # 0.2%
                            continue
                        
                        # Calculate front-run and back-run amounts
                        front_run_amount = amount_in * 0.5  # 50% of victim's amount
                        
                        # Simulate front-run transaction
                        front_run_out = router.functions.getAmountsOut(
                            front_run_amount,
                            path
                        ).call()[1]
                        
                        # Calculate new reserves after front-run
                        new_reserve_in = reserve_in + front_run_amount
                        new_reserve_out = reserve_out - front_run_out
                        
                        # Simulate victim transaction with new reserves
                        victim_out = (amount_in * 997 * new_reserve_out) / (new_reserve_in * 1000 + amount_in * 997)
                        
                        # Calculate new reserves after victim transaction
                        final_reserve_in = new_reserve_in + amount_in
                        final_reserve_out = new_reserve_out - victim_out
                        
                        # Calculate optimal back-run amount (same tokens we received in front-run)
                        back_run_in = front_run_out
                        
                        # Simulate back-run transaction
                        back_run_out = (back_run_in * 997 * final_reserve_in) / (final_reserve_out * 1000 + back_run_in * 997)
                        
                        # Calculate profit
                        profit = back_run_out - front_run_amount
                        profit_percentage = profit / front_run_amount
                        
                        # Convert profit to ETH
                        if token_in.lower() == TOKEN_ADDRESSES['WETH'].lower():
                            profit_eth = profit / 10**token_in_decimals
                        else:
                            token_in_price = get_token_price_usd(token_in)
                            eth_price = get_eth_price_usd()
                            profit_eth = (profit / 10**token_in_decimals) * token_in_price / eth_price if eth_price > 0 else 0
                        
                        # Check if profit exceeds threshold
                        if profit_percentage >= min_profit_threshold:
                            # Create opportunity
                            opportunity = {
                                'strategy_type': 'sandwich',
                                'target_tx_hash': tx_hash,
                                'dex': target_dex,
                                'token_address': token_in,
                                'path': [addr for addr in path],
                                'token_symbol': token_in_symbol,
                                'target_amount': formatted_amount,
                                'front_run_amount': front_run_amount / 10**token_in_decimals,
                                'back_run_amount': back_run_in / 10**token_in_decimals,
                                'price_impact_percentage': float(price_impact_percentage),
                                'estimated_profit': float(profit_eth),
                                'profit_percentage': float(profit_percentage),
                                'confidence_score': min(0.8, 0.6 + profit_percentage * 100),  # Lower confidence than arbitrage
                                'gas_price': gas_price,
                                'victim_gas_price': web3.from_wei(gas_price, 'gwei')
                            }
                            
                            logger.info(f"Found sandwich opportunity: {opportunity}")
                            opportunities.append(opportunity)
                            
                except Exception as e:
                    logger.error(f"Error analyzing pair for sandwich attack: {e}")
                
        except Exception as e:
            logger.debug(f"Error decoding transaction input: {e}")
    
    return opportunities

def execute_sandwich_attack(web3: Web3, opportunity: Dict, wallet_address: str, private_key: str, options: Dict = None) -> Dict:
    """
    Execute a sandwich attack
    
    Args:
        web3: Web3 instance
        opportunity: Sandwich opportunity dictionary
        wallet_address: Wallet address to execute from
        private_key: Private key for transaction signing
        options: Additional options
        
    Returns:
        Result dictionary
    """
    from config import DEX_ADDRESSES
    from execution.transaction_executor import execute_transaction, simulate_transaction
    
    result = {
        'success': False,
        'front_tx_hash': None,
        'back_tx_hash': None,
        'total_gas_used': 0,
        'avg_gas_price': 0,
        'profit_eth': 0,
        'error': None
    }
    
    options = options or {}
    max_gas_price = options.get('max_gas_price', 100)  # Gwei
    
    try:
        # Get DEX router contract
        dex_name = opportunity['dex']
        if dex_name == 'Uniswap V2':
            router_address = DEX_ADDRESSES['Uniswap V2 Router']
        elif dex_name == 'Sushiswap':
            router_address = DEX_ADDRESSES['Sushiswap Router']
        else:
            result['error'] = f"Unsupported DEX: {dex_name}"
            return result
        
        router = web3.eth.contract(address=router_address, abi=UNISWAP_V2_ROUTER_ABI)
        
        # Get token details
        token_address = opportunity['token_address']
        token = web3.eth.contract(address=token_address, abi=ERC20_ABI)
        token_decimals = token.functions.decimals().call()
        
        # Calculate front-run amount in smallest unit
        front_run_amount = int(opportunity['front_run_amount'] * 10**token_decimals)
        
        # Check if we have enough tokens
        token_balance = token.functions.balanceOf(wallet_address).call()
        if token_balance < front_run_amount:
            result['error'] = f"Insufficient token balance: have {token_balance}, need {front_run_amount}"
            return result
        
        # Check token allowance
        allowance = token.functions.allowance(wallet_address, router_address).call()
        if allowance < front_run_amount:
            # Approve router to spend tokens
            approve_tx = token.functions.approve(
                router_address,
                front_run_amount * 2  # Approve enough for both transactions
            ).build_transaction({
                'from': wallet_address,
                'nonce': web3.eth.get_transaction_count(wallet_address),
                'gas': 100000,
                'gasPrice': min(web3.eth.gas_price * 1.1, Web3.to_wei(max_gas_price, 'gwei'))
            })
            
            # Sign and send approval transaction
            signed_tx = web3.eth.account.sign_transaction(approve_tx, private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status != 1:
                result['error'] = "Approval transaction failed"
                return result
        
        # Get victim transaction details
        target_tx_hash = opportunity['target_tx_hash']
        path = opportunity['path']
        
        # Calculate victim gas price
        victim_gas_price = web3.to_wei(opportunity['victim_gas_price'], 'gwei')
        
        # Front-run: set gas price higher than victim
        front_run_gas_price = min(int(victim_gas_price * 1.2), Web3.to_wei(max_gas_price, 'gwei'))
        
        # Prepare front-run transaction data
        deadline = int(time.time() + 120)  # 2 minutes from now
        
        # Make sure we get at least a minimum amount out
        amounts_out = router.functions.getAmountsOut(front_run_amount, path).call()
        front_run_min_out = int(amounts_out[1] * 0.95)  # Allow 5% slippage
        
        front_run_tx = router.functions.swapExactTokensForTokens(
            front_run_amount,
            front_run_min_out,
            path,
            wallet_address,
            deadline
        ).build_transaction({
            'from': wallet_address,
            'nonce': web3.eth.get_transaction_count(wallet_address),
            'gas': 300000,
            'gasPrice': front_run_gas_price
        })
        
        # Simulate front-run transaction
        simulation = simulate_transaction(web3, front_run_tx)
        if not simulation['success']:
            result['error'] = f"Front-run simulation failed: {simulation['error']}"
            return result
        
        # Sign and send front-run transaction
        signed_tx = web3.eth.account.sign_transaction(front_run_tx, private_key)
        front_tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        front_receipt = web3.eth.wait_for_transaction_receipt(front_tx_hash)
        
        if front_receipt.status != 1:
            result['error'] = "Front-run transaction failed"
            return result
        
        # Save front-run tx hash and gas used
        result['front_tx_hash'] = front_tx_hash.hex()
        front_gas_used = front_receipt.gasUsed
        
        # Wait for victim transaction to be mined
        try:
            web3.eth.wait_for_transaction_receipt(Web3.to_hex(target_tx_hash) if isinstance(target_tx_hash, bytes) else target_tx_hash, timeout=120)
        except Exception as e:
            logger.warning(f"Victim transaction not mined within timeout: {e}")
            # Continue anyway, the victim transaction might be dropped or delayed
        
        # Prepare back-run transaction
        
        # Get current token balances
        token_out = path[-1]
        token_out_contract = web3.eth.contract(address=token_out, abi=ERC20_ABI)
        back_run_amount = token_out_contract.functions.balanceOf(wallet_address).call()
        
        # Check if we have tokens to back-run with
        if back_run_amount <= 0:
            result['error'] = f"No tokens received from front-run to execute back-run"
            return result
        
        # Check if token out needs approval
        back_run_allowance = token_out_contract.functions.allowance(wallet_address, router_address).call()
        if back_run_allowance < back_run_amount:
            # Approve router to spend tokens
            approve_tx = token_out_contract.functions.approve(
                router_address,
                back_run_amount * 2
            ).build_transaction({
                'from': wallet_address,
                'nonce': web3.eth.get_transaction_count(wallet_address),
                'gas': 100000,
                'gasPrice': min(web3.eth.gas_price * 1.1, Web3.to_wei(max_gas_price, 'gwei'))
            })
            
            # Sign and send approval transaction
            signed_tx = web3.eth.account.sign_transaction(approve_tx, private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status != 1:
                result['error'] = "Back-run approval transaction failed"
                return result
        
        # Prepare back-run path (reverse of front-run)
        back_run_path = list(reversed(path))
        
        # Check the expected output of the back-run
        amounts_out = router.functions.getAmountsOut(back_run_amount, back_run_path).call()
        back_run_min_out = int(amounts_out[1] * 0.95)  # Allow 5% slippage
        
        # Prepare back-run transaction
        back_run_tx = router.functions.swapExactTokensForTokens(
            back_run_amount,
            back_run_min_out,
            back_run_path,
            wallet_address,
            deadline
        ).build_transaction({
            'from': wallet_address,
            'nonce': web3.eth.get_transaction_count(wallet_address),
            'gas': 300000,
            'gasPrice': min(web3.eth.gas_price * 1.1, Web3.to_wei(max_gas_price, 'gwei'))
        })
        
        # Simulate back-run transaction
        simulation = simulate_transaction(web3, back_run_tx)
        if not simulation['success']:
            result['error'] = f"Back-run simulation failed: {simulation['error']}"
            return result
        
        # Sign and send back-run transaction
        signed_tx = web3.eth.account.sign_transaction(back_run_tx, private_key)
        back_tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        back_receipt = web3.eth.wait_for_transaction_receipt(back_tx_hash)
        
        if back_receipt.status != 1:
            result['error'] = "Back-run transaction failed"
            return result
        
        # Save back-run tx hash and gas used
        result['back_tx_hash'] = back_tx_hash.hex()
        back_gas_used = back_receipt.gasUsed
        
        # Calculate total gas used
        total_gas_used = front_gas_used + back_gas_used
        avg_gas_price = (front_run_gas_price + web3.eth.gas_price) / 2
        
        # Calculate actual profit
        initial_balance = token_balance
        final_balance = token.functions.balanceOf(wallet_address).call()
        actual_profit = final_balance - initial_balance
        
        # Convert to ETH value
        token_symbol = get_token_symbol(web3, token_address)
        if token_symbol == 'WETH':
            profit_eth = actual_profit / 10**token_decimals
        else:
            token_price = get_token_price_usd(token_address)
            eth_price = get_eth_price_usd()
            profit_eth = (actual_profit / 10**token_decimals) * token_price / eth_price if eth_price > 0 else 0
        
        # Calculate gas cost in ETH
        gas_cost_wei = total_gas_used * avg_gas_price
        gas_cost_eth = web3.from_wei(gas_cost_wei, 'ether')
        
        # Net profit after gas
        net_profit_eth = profit_eth - gas_cost_eth
        
        # Update result with success and metrics
        result['success'] = True
        result['total_gas_used'] = total_gas_used
        result['avg_gas_price'] = avg_gas_price
        result['gas_cost_eth'] = float(gas_cost_eth)
        result['profit_eth'] = float(net_profit_eth)
        
        return result
        
    except Exception as e:
        logger.error(f"Error executing sandwich attack: {e}")
        result['error'] = str(e)
        return result
