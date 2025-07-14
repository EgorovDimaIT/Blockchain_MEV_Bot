"""
Test script for model predictions and profitability analysis
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

from model_predictor import ModelPredictor, predict_all_targets
from utils.web3_helpers import get_web3, get_eth_price

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_data(symbol: str, exchange: str = 'binance', timeframe: str = '1h', test_rows: int = 100) -> pd.DataFrame:
    """
    Load test data for a symbol
    
    Args:
        symbol: Symbol to load data for
        exchange: Exchange name
        timeframe: Timeframe to load
        test_rows: Number of rows to test
        
    Returns:
        DataFrame with latest data
    """
    try:
        # Load data
        file_path = f'data/{exchange}_{symbol}_{timeframe}.csv'
        if not os.path.exists(file_path):
            file_path = f'attached_assets/{symbol}.csv'
            if not os.path.exists(file_path):
                logger.error(f"Data file not found: {file_path}")
                return pd.DataFrame()
        
        # For files from attached_assets with 3-column format (timestamp, price, volume)
        if 'attached_assets' in file_path:
            df = pd.read_csv(file_path, header=None, names=['timestamp', 'price', 'volume'])
            
            # Convert timestamp to datetime
            if df['timestamp'].dtype == np.int64 or df['timestamp'].dtype == np.float64:
                if df['timestamp'].iloc[0] > 1e12:  # milliseconds
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:  # seconds
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                df['datetime'] = pd.to_datetime(df['timestamp'])
                
            # Add close column (same as price for this format)
            df['close'] = df['price']
            df['open'] = df['price']
            df['high'] = df['price']
            df['low'] = df['price']
            
        else:  # Standard format with named columns
            df = pd.read_csv(file_path)
            
            # Convert datetime if needed
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Sort by datetime
        df = df.sort_values('datetime') if 'datetime' in df.columns else df
        
        logger.info(f"Loaded {len(df)} records from {file_path}")
        
        # Get last test_rows rows for testing
        if len(df) > test_rows:
            df = df.iloc[-test_rows:]
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return pd.DataFrame()

def prepare_features(df: pd.DataFrame, sequence_length: int = 24) -> List[np.ndarray]:
    """
    Prepare features for prediction
    
    Args:
        df: DataFrame with data
        sequence_length: Length of input sequence
        
    Returns:
        List of feature arrays, each with shape (sequence_length, num_features)
    """
    try:
        # Add technical indicators
        df_with_indicators = df.copy()
        
        # Add timestamp
        if 'datetime' in df_with_indicators.columns:
            df_with_indicators['timestamp'] = df_with_indicators['datetime'].astype(np.int64) // 10**9
        
        # Add moving averages
        for period in [5, 10, 20]:
            # Simple Moving Average
            df_with_indicators[f'sma_{period}'] = df_with_indicators['close'].rolling(window=period).mean()
            
            # Exponential Moving Average
            df_with_indicators[f'ema_{period}'] = df_with_indicators['close'].ewm(span=period, adjust=False).mean()
        
        # For simplicity, we'll drop rows with NaN values
        df_with_indicators = df_with_indicators.dropna()
        
        # Create feature arrays
        feature_arrays = []
        
        # Get feature columns (we'll use the model's input_size to determine how many columns to use)
        predictor = ModelPredictor()
        
        # Get feature columns (don't include datetime or target columns)
        feature_cols = [col for col in df_with_indicators.columns 
                        if col not in ['datetime', 'return_1h', 'return_3h', 'return_6h', 'return_12h', 'return_24h']]
        
        # Ensure we have enough rows
        if len(df_with_indicators) < sequence_length:
            logger.warning(f"Not enough data after adding indicators. Need {sequence_length} rows but only have {len(df_with_indicators)}")
            return []
        
        # Create sliding windows for feature extraction
        for i in range(len(df_with_indicators) - sequence_length + 1):
            window = df_with_indicators.iloc[i:i+sequence_length]
            
            # Get features
            features = window[feature_cols].values
            
            # Limit to the top 10 features (to match our optimized models)
            if features.shape[1] > 10:
                features = features[:, :10]
                
            feature_arrays.append(features)
        
        return feature_arrays
        
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return []

def calculate_actual_returns(df: pd.DataFrame) -> Dict[str, List[float]]:
    """
    Calculate actual returns from data
    
    Args:
        df: DataFrame with price data
        
    Returns:
        Dictionary with actual returns for different time horizons
    """
    returns = {
        'return_1h': [],
        'return_3h': [],
        'return_6h': [],
        'return_12h': [],
        'return_24h': []
    }
    
    try:
        # Calculate returns
        for period, name in [(1, 'return_1h'), (3, 'return_3h'), (6, 'return_6h'), (12, 'return_12h'), (24, 'return_24h')]:
            df[name] = df['close'].pct_change(periods=period).shift(-period)
        
        # Remove rows with NaN values
        df_clean = df.dropna(subset=list(returns.keys()))
        
        # Extract returns
        for name in returns.keys():
            returns[name] = df_clean[name].tolist()
        
        return returns
        
    except Exception as e:
        logger.error(f"Error calculating actual returns: {e}")
        return returns

def analyze_model_accuracy(symbol: str, exchange: str = 'binance') -> Dict[str, Any]:
    """
    Analyze model accuracy for a symbol
    
    Args:
        symbol: Symbol to analyze
        exchange: Exchange name
        
    Returns:
        Dictionary with accuracy metrics
    """
    results = {
        'symbol': f"{exchange}_{symbol}",
        'accuracy': {},
        'profitability': {},
        'successful_predictions': 0,
        'total_predictions': 0
    }
    
    try:
        # Load test data
        df = load_test_data(symbol, exchange)
        
        if len(df) == 0:
            logger.error(f"No data available for {exchange}_{symbol}")
            return results
        
        # Prepare features
        feature_arrays = prepare_features(df)
        
        if len(feature_arrays) == 0:
            logger.error(f"Failed to prepare features for {exchange}_{symbol}")
            return results
        
        # Calculate actual returns
        actual_returns = calculate_actual_returns(df)
        
        # Make predictions
        predictions = []
        for features in feature_arrays:
            pred = predict_all_targets(features, f"{exchange}_{symbol}")
            if 'error' not in pred:
                predictions.append(pred)
        
        # Skip if no valid predictions
        if len(predictions) == 0:
            logger.error(f"No valid predictions for {exchange}_{symbol}")
            return results
        
        # Calculate accuracy metrics
        correct_direction = {key: 0 for key in actual_returns.keys()}
        total_preds = min(len(predictions), len(list(actual_returns.values())[0]))
        
        logger.info(f"Analyzing {total_preds} predictions for {exchange}_{symbol}")
        
        # For each prediction
        for i in range(total_preds):
            for target in actual_returns.keys():
                if i >= len(actual_returns[target]):
                    continue
                    
                # Get predicted and actual return
                pred_return = predictions[i]['predictions'].get(target, {}).get('value', 0)
                actual_return = actual_returns[target][i]
                
                # Check if direction is correct
                if (pred_return > 0 and actual_return > 0) or (pred_return < 0 and actual_return < 0):
                    correct_direction[target] += 1
        
        # Calculate accuracy as percentage of correct direction predictions
        for target in actual_returns.keys():
            if total_preds > 0:
                results['accuracy'][target] = correct_direction[target] / total_preds
            else:
                results['accuracy'][target] = 0.0
        
        # Calculate overall accuracy
        total_correct = sum(correct_direction.values())
        total_predictions = total_preds * len(actual_returns.keys())
        
        if total_predictions > 0:
            results['overall_accuracy'] = total_correct / total_predictions
        else:
            results['overall_accuracy'] = 0.0
        
        # Calculate profitability metrics
        # For each time horizon, what would be the profit if we executed based on predictions?
        for target in actual_returns.keys():
            # Simulate trading based on predictions
            capital = 1000.0  # Starting capital
            position = None
            entry_price = 0.0
            
            for i in range(total_preds):
                if i >= len(actual_returns[target]):
                    continue
                    
                pred_return = predictions[i]['predictions'].get(target, {}).get('value', 0)
                actual_return = actual_returns[target][i]
                
                # If we have a position and reached the target time frame, close it
                if position is not None:
                    # Calculate profit/loss
                    if position == 'long':
                        profit = capital * actual_return
                    else:  # 'short'
                        profit = capital * -actual_return
                    
                    # Update capital
                    capital += profit
                    position = None
                
                # If no position and prediction is significant, enter a position
                if position is None and abs(pred_return) > 0.005:  # 0.5% threshold
                    position = 'long' if pred_return > 0 else 'short'
            
            # Calculate final ROI
            roi = (capital - 1000.0) / 1000.0
            results['profitability'][target] = roi
        
        # Calculate overall profitability
        results['overall_profitability'] = sum(results['profitability'].values()) / len(results['profitability'])
        
        # Calculate success rate of predictions that led to profitable trades
        successful_predictions = 0
        
        for i in range(total_preds):
            for target in actual_returns.keys():
                if i >= len(actual_returns[target]):
                    continue
                    
                pred_return = predictions[i]['predictions'].get(target, {}).get('value', 0)
                actual_return = actual_returns[target][i]
                
                # If prediction is significant and direction is correct
                if abs(pred_return) > 0.005 and ((pred_return > 0 and actual_return > 0) or (pred_return < 0 and actual_return < 0)):
                    successful_predictions += 1
        
        results['successful_predictions'] = successful_predictions
        results['total_predictions'] = total_predictions
        results['success_rate'] = successful_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing model accuracy for {exchange}_{symbol}: {e}")
        return results

def simulate_arbitrage_strategy(capital: float = 1000.0, days: int = 30, opportunities_per_day: int = 40) -> Dict[str, Any]:
    """
    Simulate profitability of arbitrage strategy
    
    Args:
        capital: Initial capital
        days: Number of days to simulate
        opportunities_per_day: Number of opportunities per day
        
    Returns:
        Dictionary with simulation results
    """
    results = {
        'initial_capital': capital,
        'final_capital': capital,
        'profit': 0.0,
        'roi': 0.0,
        'successful_trades': 0,
        'failed_trades': 0,
        'avg_profit_per_trade': 0.0,
        'success_rate': 0.0
    }
    
    try:
        # Get predictor
        predictor = ModelPredictor()
        available_models = predictor.get_available_symbols()
        
        if len(available_models) == 0:
            logger.error("No models available for simulation")
            return results
        
        # Get ETH price for USD conversion
        eth_price = get_eth_price() or 2000.0  # Default to $2000 if can't get price
        
        # Simulate trades
        successful_trades = 0
        failed_trades = 0
        total_profit = 0.0
        
        # For each day
        for day in range(days):
            # For each opportunity
            for opp in range(opportunities_per_day):
                # Randomly select a symbol
                symbol = np.random.choice(available_models)
                
                # Load random test data
                try:
                    if '_' in symbol:
                        exchange, pair = symbol.split('_', 1)
                    else:
                        exchange = 'binance'
                        pair = symbol
                        
                    df = load_test_data(pair, exchange, test_rows=1)
                    
                    if len(df) == 0:
                        continue
                    
                    # Prepare features
                    features = prepare_features(df)[0] if prepare_features(df) else None
                    
                    if features is None:
                        continue
                    
                    # Get prediction
                    prediction = predict_all_targets(features, symbol)
                    
                    if 'error' in prediction:
                        continue
                    
                    # Get 1h prediction
                    pred_1h = prediction['predictions'].get('return_1h', {}).get('value', 0)
                    
                    # If prediction is significant
                    if abs(pred_1h) > 0.005:  # 0.5% threshold
                        # Simulate trade
                        # In real system, we'd execute actual trade logic here
                        
                        # For simulation, we'll assume:
                        # - 70% chance of prediction being correct (based on model metrics)
                        # - Average gas cost of 0.005 ETH per trade
                        # - Asset exposure of 80% of capital per trade
                        
                        # Calculate trade parameters
                        trade_size = capital * 0.8
                        gas_cost = 0.005 * eth_price  # Convert to USD
                        
                        # Determine if prediction is correct (70% chance)
                        prediction_correct = np.random.random() < 0.7
                        
                        if prediction_correct:
                            # Calculate profit (predicted return minus gas costs)
                            profit = trade_size * abs(pred_1h) - gas_cost
                            successful_trades += 1
                        else:
                            # Calculate loss (gas costs plus slippage)
                            profit = -gas_cost - (trade_size * 0.001)  # 0.1% slippage
                            failed_trades += 1
                        
                        # Update capital
                        capital += profit
                        total_profit += profit
                
                # Skip any errors in simulation
                except Exception as e:
                    logger.error(f"Error in simulation: {e}")
                    continue
        
        # Calculate results
        total_trades = successful_trades + failed_trades
        
        results['final_capital'] = capital
        results['profit'] = capital - results['initial_capital']
        results['roi'] = results['profit'] / results['initial_capital'] if results['initial_capital'] > 0 else 0.0
        results['successful_trades'] = successful_trades
        results['failed_trades'] = failed_trades
        results['total_trades'] = total_trades
        results['avg_profit_per_trade'] = total_profit / total_trades if total_trades > 0 else 0.0
        results['success_rate'] = successful_trades / total_trades if total_trades > 0 else 0.0
        
        return results
        
    except Exception as e:
        logger.error(f"Error simulating arbitrage strategy: {e}")
        return results

def main():
    """Test model predictions and analyze profitability"""
    logger.info("Starting model predictions test")
    
    # Define models to test
    models_to_test = [
        {'symbol': 'ADAUSDC', 'exchange': 'binance'},
        {'symbol': 'NTRNEUR', 'exchange': 'binance'},
        {'symbol': 'KUJIEUR', 'exchange': 'binance'},
        {'symbol': 'FILAUD', 'exchange': 'binance'},
        {'symbol': 'ETHWETH', 'exchange': 'uniswap'}
    ]
    
    # Analyze accuracy for each model
    accuracy_results = {}
    
    for model_info in models_to_test:
        symbol = model_info['symbol']
        exchange = model_info['exchange']
        
        logger.info(f"Analyzing accuracy for {exchange}_{symbol}")
        accuracy = analyze_model_accuracy(symbol, exchange)
        accuracy_results[f"{exchange}_{symbol}"] = accuracy
        
        logger.info(f"Results for {exchange}_{symbol}:")
        logger.info(f"  Overall accuracy: {accuracy.get('overall_accuracy', 0):.4f}")
        logger.info(f"  Overall profitability: {accuracy.get('overall_profitability', 0):.4f}")
        logger.info(f"  Success rate: {accuracy.get('success_rate', 0):.4f}")
    
    # Simulate arbitrage strategy
    logger.info("Simulating arbitrage strategy")
    simulation_results = simulate_arbitrage_strategy()
    
    logger.info("Arbitrage strategy simulation results:")
    logger.info(f"  Initial capital: ${simulation_results['initial_capital']:.2f}")
    logger.info(f"  Final capital: ${simulation_results['final_capital']:.2f}")
    logger.info(f"  Profit: ${simulation_results['profit']:.2f}")
    logger.info(f"  ROI: {simulation_results['roi'] * 100:.2f}%")
    logger.info(f"  Success rate: {simulation_results['success_rate'] * 100:.2f}%")
    logger.info(f"  Average profit per trade: ${simulation_results['avg_profit_per_trade']:.2f}")
    
    # Save results
    results = {
        'accuracy_results': accuracy_results,
        'simulation_results': simulation_results
    }
    
    with open('model_profitability_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info("Analysis complete. Results saved to model_profitability_analysis.json")

if __name__ == "__main__":
    main()