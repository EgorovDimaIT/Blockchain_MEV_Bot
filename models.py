"""
Database models for the MEV bot
"""

import os
import logging
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship

# Import the db instance from database.py instead of app.py
from database import db

logger = logging.getLogger(__name__)

class Transaction(db.Model):
    """Model for transactions executed by the bot"""
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    tx_hash = Column(String(66), unique=True)
    strategy_type = Column(String(50), nullable=False)  # 'arbitrage', 'sandwich', etc.
    status = Column(String(20), nullable=False)  # 'pending', 'confirmed', 'failed'
    profit_eth = Column(Float)
    gas_used = Column(Integer)
    gas_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime)
    block_number = Column(Integer)
    details = Column(JSON)  # JSON with transaction details
    
    def __repr__(self):
        return f"<Transaction {self.tx_hash}>"

class ArbitrageOpportunity(db.Model):
    """Model for arbitrage opportunities detected by the bot"""
    __tablename__ = 'arbitrage_opportunities'
    
    id = Column(Integer, primary_key=True)
    opportunity_id = Column(String(100), unique=True)
    type = Column(String(50), nullable=False)  # 'direct', 'triangular', etc.
    token_in = Column(String(42))
    token_out = Column(String(42))
    token_mid = Column(String(42))  # For triangular arbitrage
    exchange_1 = Column(String(100))
    exchange_2 = Column(String(100))
    exchange_3 = Column(String(100))  # For triangular arbitrage
    potential_profit_eth = Column(Float)
    potential_profit_usd = Column(Float)
    gas_estimate_eth = Column(Float)
    confidence_score = Column(Float)  # ML confidence
    detected_at = Column(DateTime, default=datetime.utcnow)
    executed = Column(Boolean, default=False)
    transaction_id = Column(Integer, ForeignKey('transactions.id'), nullable=True)
    details = Column(JSON)  # JSON with opportunity details
    
    transaction = relationship("Transaction", backref="arbitrage_opportunity")
    
    def __repr__(self):
        return f"<ArbitrageOpportunity {self.opportunity_id}>"

class SandwichOpportunity(db.Model):
    """Model for sandwich opportunities detected by the bot"""
    __tablename__ = 'sandwich_opportunities'
    
    id = Column(Integer, primary_key=True)
    opportunity_id = Column(String(100), unique=True)
    victim_tx_hash = Column(String(66))
    token_in = Column(String(42))
    token_out = Column(String(42))
    exchange = Column(String(100))
    potential_profit_eth = Column(Float)
    potential_profit_usd = Column(Float)
    gas_estimate_eth = Column(Float)
    confidence_score = Column(Float)  # ML confidence
    detected_at = Column(DateTime, default=datetime.utcnow)
    executed = Column(Boolean, default=False)
    front_run_tx_id = Column(Integer, ForeignKey('transactions.id'), nullable=True)
    back_run_tx_id = Column(Integer, ForeignKey('transactions.id'), nullable=True)
    details = Column(JSON)  # JSON with opportunity details
    
    front_run_tx = relationship("Transaction", foreign_keys=[front_run_tx_id])
    back_run_tx = relationship("Transaction", foreign_keys=[back_run_tx_id])
    
    def __repr__(self):
        return f"<SandwichOpportunity {self.opportunity_id}>"

class Setting(db.Model):
    """Model for bot settings"""
    __tablename__ = 'settings'
    
    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(String(500))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Setting {self.key}>"

class MLModel(db.Model):
    """Model for machine learning model metadata"""
    __tablename__ = 'ml_models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # 'lstm', 'transformer', etc.
    version = Column(String(20))
    file_path = Column(String(255))
    accuracy = Column(Float)
    rmse = Column(Float)
    mae = Column(Float)
    training_date = Column(DateTime)
    is_active = Column(Boolean, default=False)
    description = Column(Text)
    parameters = Column(JSON)  # JSON with model parameters
    
    def __repr__(self):
        return f"<MLModel {self.name} v{self.version}>"

def init_default_settings():
    """Initialize default settings in the database"""
    with db.session.begin():
        # Only initialize if settings table is empty
        if Setting.query.count() == 0:
            logger.info("Initializing default settings")
            
            default_settings = [
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
            
            for setting in default_settings:
                db.session.add(Setting(**setting))
            
            logger.info("Default settings initialized")