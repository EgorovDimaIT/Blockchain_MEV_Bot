"""
Database configuration module to avoid circular imports
"""

from flask_sqlalchemy import SQLAlchemy

# Create a database instance to be shared across modules
db = SQLAlchemy()

def init_db(app):
    """Initialize the database with the Flask app"""
    db.init_app(app)
