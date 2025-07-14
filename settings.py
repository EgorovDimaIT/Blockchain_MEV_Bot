"""
Default settings for the MEV bot
"""

DEFAULT_SETTINGS = [
    {
        'key': 'min_profit_threshold',
        'value': '0.002',
        'description': 'Minimum profit threshold as decimal (e.g. 0.002 for 0.2%)'
    },
    {
        'key': 'max_gas_price_gwei',
        'value': '100',
        'description': 'Maximum gas price in Gwei for transactions'
    },
    {
        'key': 'max_tx_value_eth',
        'value': '0.5',
        'description': 'Maximum transaction value in ETH'
    },
    {
        'key': 'arbitrage_enabled',
        'value': 'true',
        'description': 'Enable arbitrage strategy'
    },
    {
        'key': 'triangular_arbitrage_enabled',
        'value': 'true',
        'description': 'Enable triangular arbitrage strategy'
    },
    {
        'key': 'sandwich_enabled',
        'value': 'false',
        'description': 'Enable sandwich attack strategy'
    },
    {
        'key': 'use_flashloans',
        'value': 'false',
        'description': 'Use flash loans for strategies'
    },
    {
        'key': 'use_ml_prediction',
        'value': 'true',
        'description': 'Use machine learning models for prediction'
    },
    {
        'key': 'ml_confidence_threshold',
        'value': '0.7',
        'description': 'Minimum ML confidence score to execute'
    },
    {
        'key': 'safety_cooldown_seconds',
        'value': '30',
        'description': 'Minimum seconds between transactions'
    }
]