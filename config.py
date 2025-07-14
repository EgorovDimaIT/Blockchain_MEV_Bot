"""
Configuration settings for the MEV bot
"""

import os

# API keys and provider configuration
API_KEYS = {
    'INFURA_API_KEY': os.environ.get('INFURA_API_KEY', ''),
    'DRPC_API_KEY': os.environ.get('DRPC_API_KEY', ''),
    'ETHERSCAN_API_KEY': os.environ.get('ETHERSCAN_API_KEY', ''),
    'ALCHEMY_API_KEY': os.environ.get('ALCHEMY_API_KEY', '')
}

# Chainstack configuration
CHAINSTACK_CONFIG = {
    'HTTP_ENDPOINT': os.environ.get('CHAINSTACK_HTTP_ENDPOINT', ''),
    'WSS_ENDPOINT': os.environ.get('CHAINSTACK_WSS_ENDPOINT', '')
}

# Wallet configuration
WALLET_CONFIG = {
    'PRIVATE_KEY': os.environ.get('PRIVATE_KEY', ''),
    'WALLET_ADDRESS': os.environ.get('WALLET_ADDRESS', '')
}

# MEV-specific configuration
MEV_CONFIG = {
    'MIN_PROFIT_THRESHOLD': float(os.environ.get('MIN_PROFIT_THRESHOLD', '0.002')),
    'MAX_GAS_PRICE': int(os.environ.get('MAX_GAS_PRICE', '100')),
    'FLASHBOTS_ENABLED': os.environ.get('FLASHBOTS_ENABLED', 'true').lower() == 'true',
    'MAX_TRANSACTIONS_PER_HOUR': int(os.environ.get('MAX_TRANSACTIONS_PER_HOUR', '10')),
    'MAX_TRANSACTION_VALUE_ETH': float(os.environ.get('MAX_TRANSACTION_VALUE_ETH', '1.0'))
}

# Common token addresses on Ethereum mainnet
TOKEN_ADDRESSES = {
    'ETH': '0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE',  # Special address for ETH
    'WETH': '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
    'DAI': '0x6b175474e89094c44da98b954eedeac495271d0f',
    'USDC': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
    'USDT': '0xdac17f958d2ee523a2206206994597c13d831ec7',
    'WBTC': '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599',
    'LINK': '0x514910771af9ca656af840dff83e8264ecf986ca',
    'UNI': '0x1f9840a85d5af5bf1d1762f925bdaddc4201f984',
    'AAVE': '0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9',
    'SNX': '0xc011a73ee8576fb46f5e1c5751ca3b9fe0af2a6f',
    'CRV': '0xd533a949740bb3306d119cc777fa900ba034cd52',
    'BAL': '0xba100000625a3754423978a60c9317c58a424e3d',
    'COMP': '0xc00e94cb662c3520282e6f5717214004a7f26888',
    'MKR': '0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2',
    'YFI': '0x0bc529c00c6401aef6d220be8c6ea1667f6ad93e',
    'SUSHI': '0x6b3595068778dd592e39a122f4f5a5cf09c90fe2',
    'stETH': '0xae7ab96520de3a18e5e111b5eaab095312d7fe84'
}

# DEX addresses on Ethereum mainnet
DEX_ADDRESSES = {
    'Uniswap V2 Router': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
    'Uniswap V3 Router': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
    'Sushiswap Router': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
    'Curve 3pool': '0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7',
    'Balancer V2 Vault': '0xBA12222222228d8Ba445958a75a0704d566BF2C8',
    '1inch Router': '0x1111111254fb6c44bAC0beD2854e76F90643097d'
}

# Flash loan providers
FLASH_LOAN_PROVIDERS = {
    'Aave V2': '0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9',
    'Aave V3': '0x87870Bca3F3fD6335C3F4ce8392D69350B4fa4E2',
    'dYdX Solo': '0x1E0447b19BB6EcFdAe1e4AE1694b0C3659614e4e',
    'Balancer V2': '0xBA12222222228d8Ba445958a75a0704d566BF2C8'
}

# Liquidation bots configuration
LIQUIDATION_CONFIG = {
    'PROTOCOLS': ['Aave', 'Compound', 'MakerDAO'],
    'MIN_LIQUIDATION_VALUE': 1.0,  # ETH
    'MAX_SLIPPAGE': 0.05  # 5%
}

# ML model configuration
ML_CONFIG = {
    'MODEL_DIR': 'ml_models',
    'DEFAULT_MODEL': 'lstm_model.pkl',
    'CONFIDENCE_THRESHOLD': 0.7,
    'RETRAIN_INTERVAL_HOURS': 24
}
