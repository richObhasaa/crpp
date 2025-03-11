import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
from utils.api import get_timeframe_data, get_coin_history, get_top_cryptocurrencies

def calculate_technical_indicators(df):
    """Calculate technical indicators for a given dataframe"""
    if isinstance(df, dict) and "error" in df:
        return df
    
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"error": "Invalid or empty data for technical analysis"}
    
    try:
        # Ensure we have enough data points for calculations
        if len(df) < 26:  # Need at least 26 points for MACD
            # If not enough data points, return the original dataframe with warning
            print(f"Warning: Not enough data points for technical analysis. Got {len(df)}, need at least 26.")
            # Add empty indicator columns to make UI display work
            df_ta = df.copy()
            required_indicators = ['SMA_7', 'SMA_25', 'SMA_99', 'EMA_7', 'EMA_25', 
                                 'RSI', 'BB_middle', 'BB_std', 'BB_upper', 'BB_lower',
                                 'MACD_line', 'MACD_signal', 'MACD_histogram', 
                                 'price_change_pct', 'volume_change_pct', 'volatility']
            
            # Fill with NaN where we don't have enough data
            for indicator in required_indicators:
                df_ta[indicator] = np.nan
                
            return df_ta
        
        # Make a copy to avoid modifying the original dataframe
        df_ta = df.copy()
        
        # Ensure price and volume columns exist and are numeric
        for col in ['price', 'volume']:
            if col not in df_ta.columns:
                return {"error": f"Required column '{col}' missing from data"}
            
            # Convert to numeric if not already
            df_ta[col] = pd.to_numeric(df_ta[col], errors='coerce')
            
            # Check if conversion resulted in all NaN values
            if df_ta[col].isna().all():
                return {"error": f"Column '{col}' contains invalid numeric data"}
        
        # Calculate moving averages with adaptive windows
        window_7 = min(7, len(df_ta))
        window_25 = min(25, len(df_ta))
        window_99 = min(99, len(df_ta))
        
        df_ta['SMA_7'] = df_ta['price'].rolling(window=window_7).mean()
        df_ta['SMA_25'] = df_ta['price'].rolling(window=window_25).mean()
        df_ta['SMA_99'] = df_ta['price'].rolling(window=window_99).mean()
        
        # Calculate exponential moving averages
        df_ta['EMA_7'] = df_ta['price'].ewm(span=window_7, adjust=False).mean()
        df_ta['EMA_25'] = df_ta['price'].ewm(span=window_25, adjust=False).mean()
        
        # Calculate Relative Strength Index (RSI)
        delta = df_ta['price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use adaptive window for RSI too
        rsi_window = min(14, len(df_ta) - 1)
        
        avg_gain = gain.rolling(window=rsi_window).mean()
        avg_loss = loss.rolling(window=rsi_window).mean()
        
        # Calculate RSI - handle division by zero
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        df_ta['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        bb_window = min(20, len(df_ta))
        df_ta['BB_middle'] = df_ta['price'].rolling(window=bb_window).mean()
        df_ta['BB_std'] = df_ta['price'].rolling(window=bb_window).std()
        df_ta['BB_upper'] = df_ta['BB_middle'] + 2 * df_ta['BB_std']
        df_ta['BB_lower'] = df_ta['BB_middle'] - 2 * df_ta['BB_std']
        
        # Calculate MACD with adaptive windows
        short_window = min(12, len(df_ta) - 1)
        long_window = min(26, len(df_ta) - 1)
        signal_window = min(9, len(df_ta) - 1)
        
        df_ta['MACD_line'] = df_ta['price'].ewm(span=short_window, adjust=False).mean() - df_ta['price'].ewm(span=long_window, adjust=False).mean()
        df_ta['MACD_signal'] = df_ta['MACD_line'].ewm(span=signal_window, adjust=False).mean()
        df_ta['MACD_histogram'] = df_ta['MACD_line'] - df_ta['MACD_signal']
        
        # Calculate price change percentage
        df_ta['price_change_pct'] = df_ta['price'].pct_change() * 100
        
        # Calculate volume change percentage
        df_ta['volume_change_pct'] = df_ta['volume'].pct_change() * 100
        
        # Add price volatility (standard deviation of price changes over 7 periods)
        vol_window = min(7, len(df_ta) - 1)
        df_ta['volatility'] = df_ta['price_change_pct'].rolling(window=vol_window).std()
        
        return df_ta
    except Exception as e:
        return {"error": f"Error calculating technical indicators: {str(e)}"}

def calculate_performance_metrics(df):
    """Calculate performance metrics for a given dataframe"""
    if isinstance(df, dict) and "error" in df:
        return df
    
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"error": "Invalid or empty data for performance metrics"}
    
    try:
        # Start and end prices
        start_price = df['price'].iloc[0]
        end_price = df['price'].iloc[-1]
        
        # Calculate metrics
        price_change = end_price - start_price
        price_change_pct = (price_change / start_price) * 100 if start_price > 0 else 0
        
        max_price = df['price'].max()
        min_price = df['price'].min()
        price_range = max_price - min_price
        price_range_pct = (price_range / min_price) * 100 if min_price > 0 else 0
        
        # Volatility (standard deviation of daily returns)
        daily_returns = df['price'].pct_change().dropna()
        volatility = daily_returns.std() * 100
        
        # Volume metrics
        avg_volume = df['volume'].mean()
        max_volume = df['volume'].max()
        
        # Statistical metrics
        mean_price = df['price'].mean()
        median_price = df['price'].median()
        std_dev = df['price'].std()
        
        # Coefficient of variation
        cv = (std_dev / mean_price) * 100 if mean_price > 0 else 0
        
        # Skewness and Kurtosis
        skewness = df['price'].skew()
        kurtosis = df['price'].kurtosis()
        
        # Risk-adjusted return (Sharpe ratio approximation - without risk-free rate)
        sharpe_ratio = (price_change_pct / volatility) if volatility > 0 else 0
        
        return {
            "start_price": start_price,
            "end_price": end_price,
            "price_change": price_change,
            "price_change_pct": price_change_pct,
            "max_price": max_price,
            "min_price": min_price,
            "price_range": price_range,
            "price_range_pct": price_range_pct,
            "volatility": volatility,
            "avg_volume": avg_volume,
            "max_volume": max_volume,
            "mean_price": mean_price,
            "median_price": median_price,
            "std_dev": std_dev,
            "cv": cv,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "sharpe_ratio": sharpe_ratio
        }
    except Exception as e:
        return {"error": f"Error calculating performance metrics: {str(e)}"}

def calculate_profit_loss(entry_price, exit_price, investment_amount, is_long=True):
    """Calculate profit/loss for a given trade"""
    try:
        if entry_price <= 0 or exit_price < 0 or investment_amount <= 0:
            return {"error": "Invalid input values"}
        
        if is_long:
            # Long position: Buy low, sell high
            quantity = investment_amount / entry_price
            exit_value = quantity * exit_price
            profit_loss = exit_value - investment_amount
            profit_loss_pct = (profit_loss / investment_amount) * 100
        else:
            # Short position: Sell high, buy low
            quantity = investment_amount / entry_price
            exit_cost = quantity * exit_price
            profit_loss = investment_amount - exit_cost
            profit_loss_pct = (profit_loss / investment_amount) * 100
        
        return {
            "entry_price": entry_price,
            "exit_price": exit_price,
            "investment_amount": investment_amount,
            "quantity": quantity,
            "profit_loss": profit_loss,
            "profit_loss_pct": profit_loss_pct,
            "is_profit": profit_loss > 0
        }
    except Exception as e:
        return {"error": f"Error calculating profit/loss: {str(e)}"}

def predict_price(coin_id, timeframe="5m"):
    """Predict price for the next 1-5 minutes using ARIMA model"""
    try:
        # Get historical data for the coin
        data = get_timeframe_data(coin_id, timeframe)
        
        if isinstance(data, dict) and "error" in data:
            return data
        
        if not isinstance(data, pd.DataFrame) or data.empty:
            return {"error": "Insufficient data for prediction"}
        
        # Use only the required data points for prediction
        # For 1-5m predictions, we'll use the last 60 data points
        if len(data) > 60:
            data = data.iloc[-60:].copy()
        
        # Prepare data for ARIMA model
        prices = data['price'].values
        
        # Fit ARIMA model - parameters (p,d,q) = (5,1,0)
        # These parameters may need adjustment based on the specific cryptocurrency
        model = ARIMA(prices, order=(5, 1, 0))
        model_fit = model.fit()
        
        # Forecast next 5 minutes
        forecast = model_fit.forecast(steps=5)
        
        # Calculate confidence intervals (95%)
        confidence = 1.96 * np.sqrt(model_fit.prediction_variance)
        
        # Calculate accuracy as the inverse of the normalized confidence interval width
        accuracy = 1.0 - (confidence / np.mean(prices))
        accuracy = max(0, min(accuracy, 1)) * 100  # Convert to percentage and clip to 0-100%
        
        # Calculate percentage change from current price
        current_price = prices[-1]
        price_changes = [(forecast[i] - current_price) / current_price * 100 for i in range(len(forecast))]
        
        prediction_results = {
            "current_price": current_price,
            "predicted_prices": forecast.tolist(),
            "confidence_interval": confidence.tolist() if isinstance(confidence, np.ndarray) else [confidence],
            "price_change_pct": price_changes,
            "accuracy": accuracy,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return prediction_results
    except Exception as e:
        return {"error": f"Error predicting price: {str(e)}"}

def calculate_comparison_statistics(coin_ids):
    """Calculate detailed statistical comparison between multiple coins"""
    try:
        # Get top cryptocurrencies for reference
        top_cryptos = get_top_cryptocurrencies()
        
        if top_cryptos.empty:
            return {"error": "Failed to retrieve cryptocurrency data"}
        
        # Filter the dataframe to include only the requested coins
        comparison_df = top_cryptos[top_cryptos['id'].isin(coin_ids)]
        
        if comparison_df.empty:
            return {"error": "None of the specified cryptocurrencies were found"}
        
        # Calculate additional statistics for each coin
        comparison_stats = {}
        
        for coin_id in coin_ids:
            coin_data = comparison_df[comparison_df['id'] == coin_id]
            
            if not coin_data.empty:
                # Get historical data for advanced metrics
                hist_data = get_coin_history(coin_id, days="30")
                
                if isinstance(hist_data, pd.DataFrame) and not hist_data.empty:
                    # Calculate statistical metrics
                    price_data = hist_data['price']
                    mean = price_data.mean()
                    median = price_data.median()
                    std_dev = price_data.std()
                    cv = (std_dev / mean) * 100 if mean > 0 else 0
                    skewness = price_data.skew()
                    kurtosis = price_data.kurtosis()
                    
                    # Calculate daily returns for volatility and other metrics
                    daily_returns = price_data.pct_change().dropna()
                    volatility = daily_returns.std() * 100
                    
                    # Calculate drawdown
                    rolling_max = price_data.cummax()
                    drawdown = (price_data - rolling_max) / rolling_max
                    max_drawdown = drawdown.min() * 100
                    
                    # Extract basic metrics from the top cryptocurrencies dataframe
                    market_cap = coin_data['market_cap'].values[0]
                    current_price = coin_data['current_price'].values[0]
                    volume = coin_data['total_volume'].values[0]
                    price_change_24h = coin_data['price_change_percentage_24h'].values[0]
                    
                    # Store all metrics in the comparison stats dictionary
                    comparison_stats[coin_id] = {
                        "name": coin_data['name'].values[0],
                        "symbol": coin_data['symbol'].values[0],
                        "market_cap": market_cap,
                        "current_price": current_price,
                        "volume": volume,
                        "price_change_24h": price_change_24h,
                        "mean": mean,
                        "median": median,
                        "std_dev": std_dev,
                        "cv": cv,
                        "skewness": skewness,
                        "kurtosis": kurtosis,
                        "volatility": volatility,
                        "max_drawdown": max_drawdown
                    }
                else:
                    # Use basic metrics if historical data is not available
                    comparison_stats[coin_id] = {
                        "name": coin_data['name'].values[0],
                        "symbol": coin_data['symbol'].values[0],
                        "market_cap": coin_data['market_cap'].values[0],
                        "current_price": coin_data['current_price'].values[0],
                        "volume": coin_data['total_volume'].values[0],
                        "price_change_24h": coin_data['price_change_percentage_24h'].values[0],
                        "error": "Historical data not available for advanced metrics"
                    }
            else:
                comparison_stats[coin_id] = {"error": f"Data for {coin_id} not found"}
        
        return comparison_stats
    except Exception as e:
        return {"error": f"Error calculating comparison statistics: {str(e)}"}

def find_best_entry_points(coin_id, timeframe="1d", lookback_days="30"):
    """Find potentially good entry points based on historical data"""
    try:
        # Get historical data
        hist_data = get_coin_history(coin_id, days=lookback_days)
        
        if isinstance(hist_data, dict) and "error" in hist_data:
            return hist_data
        
        if not isinstance(hist_data, pd.DataFrame) or hist_data.empty:
            return {"error": "Insufficient data for analysis"}
        
        # Calculate technical indicators
        data = calculate_technical_indicators(hist_data)
        
        if isinstance(data, dict) and "error" in data:
            return data
        
        # Strategy 1: RSI oversold conditions (RSI < 30)
        rsi_entry_points = data[data['RSI'] < 30][['timestamp', 'price', 'RSI']]
        
        # Strategy 2: Price below lower Bollinger Band
        bb_entry_points = data[data['price'] < data['BB_lower']][['timestamp', 'price', 'BB_lower']]
        
        # Strategy 3: Golden Cross (shorter MA crosses above longer MA)
        data['golden_cross'] = (data['SMA_7'] > data['SMA_25']) & (data['SMA_7'].shift(1) <= data['SMA_25'].shift(1))
        golden_cross_points = data[data['golden_cross']][['timestamp', 'price', 'SMA_7', 'SMA_25']]
        
        # Strategy 4: MACD line crosses above signal line
        data['macd_cross_above'] = (data['MACD_line'] > data['MACD_signal']) & (data['MACD_line'].shift(1) <= data['MACD_signal'].shift(1))
        macd_entry_points = data[data['macd_cross_above']][['timestamp', 'price', 'MACD_line', 'MACD_signal']]
        
        # Strategy 5: High volume with price increase
        data['high_volume'] = data['volume'] > data['volume'].rolling(window=20).mean() * 1.5
        data['price_increase'] = data['price'] > data['price'].shift(1)
        volume_price_points = data[data['high_volume'] & data['price_increase']][['timestamp', 'price', 'volume']]
        
        # Combine and analyze the entry points
        entry_strategies = {
            "rsi_oversold": rsi_entry_points.to_dict(orient='records'),
            "bollinger_band_lower": bb_entry_points.to_dict(orient='records'),
            "golden_cross": golden_cross_points.to_dict(orient='records'),
            "macd_cross_above": macd_entry_points.to_dict(orient='records'),
            "high_volume_price_increase": volume_price_points.to_dict(orient='records')
        }
        
        # Calculate success rate for each strategy by checking subsequent price action
        # (This is a simplified version, would need more complex logic for real trading)
        
        # For each entry point, check if price increased by at least 3% within the next 7 periods
        success_rates = {}
        
        for strategy, points in entry_strategies.items():
            if len(points) > 0:
                success_count = 0
                for point in points:
                    # Get the index of this entry point
                    try:
                        entry_idx = data[data['timestamp'] == point['timestamp']].index[0]
                        
                        # Check if we have enough data after this point
                        if entry_idx + 7 < len(data):
                            # Get the entry price
                            entry_price = data.loc[entry_idx, 'price']
                            
                            # Get the highest price in the next 7 periods
                            future_prices = data.loc[entry_idx+1:entry_idx+7, 'price']
                            highest_future_price = future_prices.max()
                            
                            # Check if price increased by at least 3%
                            if (highest_future_price - entry_price) / entry_price >= 0.03:
                                success_count += 1
                    except (IndexError, KeyError):
                        continue
                
                success_rate = (success_count / len(points)) * 100 if len(points) > 0 else 0
                success_rates[strategy] = success_rate
            else:
                success_rates[strategy] = 0
        
        # Determine the best strategy based on success rate
        best_strategy = max(success_rates.items(), key=lambda x: x[1]) if success_rates else None
        
        return {
            "entry_strategies": entry_strategies,
            "success_rates": success_rates,
            "best_strategy": best_strategy,
            "lookback_period": lookback_days,
            "timeframe": timeframe
        }
    except Exception as e:
        return {"error": f"Error finding entry points: {str(e)}"}
