"""
Script to download historical data from exchanges
"""

import os
import logging
from data_collection.exchange_data_downloader import ExchangeDataDownloader
from ml_model.data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to download and process exchange data"""
    
    logger.info("Starting download of exchange data")
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Step 1: Download data from exchanges
    downloader = ExchangeDataDownloader(days_to_fetch=7)
    data = downloader.run(include_orderbooks=False, create_arbitrage_datasets=True)
    
    # Log downloaded data
    if data['ohlcv_data']:
        total_files = 0
        for exchange, symbols_data in data['ohlcv_data'].items():
            total_files += len(symbols_data)
        logger.info(f"Downloaded OHLCV data: {total_files} files across {len(data['ohlcv_data'])} exchanges")
        
    if data['triangular_arbitrage'] is not None:
        logger.info(f"Triangular arbitrage opportunities: {len(data['triangular_arbitrage'])} records")
        
    if data['direct_arbitrage'] is not None:
        logger.info(f"Direct arbitrage opportunities: {len(data['direct_arbitrage'])} records")
    
    # Step 2: Process data
    processor = DataProcessor(input_dir='data', output_dir='data/processed')
    processed_data = processor.run()
    
    # Log processed data
    if processed_data['ohlcv']:
        logger.info(f"Processed OHLCV files: {len(processed_data['ohlcv']['processed_files'])}")
        logger.info(f"LSTM datasets created: {len(processed_data['ohlcv']['lstm_data'])}")
        
    if processed_data['arbitrage']:
        direct = processed_data['arbitrage']['direct']
        triangular = processed_data['arbitrage']['triangular']
        
        logger.info(f"Direct arbitrage data processed: {direct['processed'] is not None}")
        logger.info(f"Triangular arbitrage data processed: {triangular['processed'] is not None}")
    
    logger.info("Exchange data download and processing completed")
    
    return {
        'data': data,
        'processed_data': processed_data
    }

if __name__ == "__main__":
    main()