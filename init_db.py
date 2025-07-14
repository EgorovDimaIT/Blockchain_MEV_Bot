"""
Initialize the database tables.
"""
import os
from flask import Flask

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")

# Configure the database
# Ensure data directory exists
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Use absolute path for database
db_path = os.path.join(data_dir, "mev_bot.db")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", f"sqlite:///{db_path}")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Import SQLAlchemy after configuring app
from flask_sqlalchemy import SQLAlchemy

# Initialize db with app
db = SQLAlchemy(app)

# Create models directly with db.Model
class Transaction(db.Model):
    """Model for transactions executed by the bot"""
    __tablename__ = 'transactions'
    
    id = db.Column(db.Integer, primary_key=True)
    tx_hash = db.Column(db.String(66), unique=True)
    strategy_type = db.Column(db.String(50), nullable=False)  # 'arbitrage', 'sandwich', etc.
    status = db.Column(db.String(20), nullable=False)  # 'pending', 'confirmed', 'failed'
    profit_eth = db.Column(db.Float)
    gas_used = db.Column(db.Integer)
    gas_price = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=db.func.now())
    executed_at = db.Column(db.DateTime)
    block_number = db.Column(db.Integer)
    details = db.Column(db.JSON)  # JSON with transaction details

class ArbitrageOpportunity(db.Model):
    """Model for arbitrage opportunities detected by the bot"""
    __tablename__ = 'arbitrage_opportunities'
    
    id = db.Column(db.Integer, primary_key=True)
    opportunity_id = db.Column(db.String(100), unique=True)
    type = db.Column(db.String(50), nullable=False)  # 'direct', 'triangular', etc.
    token_in = db.Column(db.String(42))
    token_out = db.Column(db.String(42))
    token_mid = db.Column(db.String(42))  # For triangular arbitrage
    exchange_1 = db.Column(db.String(100))
    exchange_2 = db.Column(db.String(100))
    exchange_3 = db.Column(db.String(100))  # For triangular arbitrage
    potential_profit_eth = db.Column(db.Float)
    potential_profit_usd = db.Column(db.Float)
    gas_estimate_eth = db.Column(db.Float)
    confidence_score = db.Column(db.Float)  # ML confidence
    detected_at = db.Column(db.DateTime, default=db.func.now())
    executed = db.Column(db.Boolean, default=False)
    transaction_id = db.Column(db.Integer, db.ForeignKey('transactions.id'), nullable=True)
    details = db.Column(db.JSON)  # JSON with opportunity details
    
    transaction = db.relationship("Transaction", backref="arbitrage_opportunity")

class SandwichOpportunity(db.Model):
    """Model for sandwich opportunities detected by the bot"""
    __tablename__ = 'sandwich_opportunities'
    
    id = db.Column(db.Integer, primary_key=True)
    opportunity_id = db.Column(db.String(100), unique=True)
    victim_tx_hash = db.Column(db.String(66))
    token_in = db.Column(db.String(42))
    token_out = db.Column(db.String(42))
    exchange = db.Column(db.String(100))
    potential_profit_eth = db.Column(db.Float)
    potential_profit_usd = db.Column(db.Float)
    gas_estimate_eth = db.Column(db.Float)
    confidence_score = db.Column(db.Float)  # ML confidence
    detected_at = db.Column(db.DateTime, default=db.func.now())
    executed = db.Column(db.Boolean, default=False)
    front_run_tx_id = db.Column(db.Integer, db.ForeignKey('transactions.id'), nullable=True)
    back_run_tx_id = db.Column(db.Integer, db.ForeignKey('transactions.id'), nullable=True)
    details = db.Column(db.JSON)  # JSON with opportunity details
    
    front_run_tx = db.relationship("Transaction", foreign_keys=[front_run_tx_id])
    back_run_tx = db.relationship("Transaction", foreign_keys=[back_run_tx_id])

class Setting(db.Model):
    """Model for bot settings"""
    __tablename__ = 'settings'
    
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.String(500))
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=db.func.now())
    updated_at = db.Column(db.DateTime, default=db.func.now(), onupdate=db.func.now())

class MLModel(db.Model):
    """Model for machine learning model metadata"""
    __tablename__ = 'ml_models'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)  # 'lstm', 'transformer', etc.
    version = db.Column(db.String(20))
    file_path = db.Column(db.String(255))
    accuracy = db.Column(db.Float)
    rmse = db.Column(db.Float)
    mae = db.Column(db.Float)
    training_date = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=False)
    description = db.Column(db.Text)
    parameters = db.Column(db.JSON)  # JSON with model parameters

def init_default_settings():
    """Initialize default settings in the database"""
    # Only initialize if settings table is empty
    if Setting.query.count() == 0:
        print("Initializing default settings")
        
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
        
        db.session.commit()
        print("Default settings initialized")

# Create tables and initialize settings
with app.app_context():
    # Create all tables
    db.create_all()
    
    # Initialize default settings
    init_default_settings()
    
    print("Database initialized successfully!")

if __name__ == "__main__":
    print("Database setup complete.")
