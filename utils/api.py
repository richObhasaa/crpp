import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import json
from functools import wraps
from utils.constants import (
    COINGECKO_API_BASE_URL, 
    TOP_CRYPTO_COUNT, 
    CRYPTO_NEWS_API_URL, 
    NEWS_API_URL, 
    CRYPTO_PANIC_API_URL
)

# In-memory cache dictionary
_cache = {}

def cache_result(expire_seconds=300):
    """Cache decorator with expiration time for API requests
    Args:
        expire_seconds (int): Number of seconds before cache expires (default: 5 minutes)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key based on function name and arguments
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check if we have a cached response and it's still valid
            if key in _cache:
                result, timestamp = _cache[key]
                if (datetime.now() - timestamp).total_seconds() < expire_seconds:
                    return result
            
            # Call the original function
            result = func(*args, **kwargs)
            
            # Cache the result along with the current timestamp
            _cache[key] = (result, datetime.now())
            
            return result
        return wrapper
    return decorator

# Constants for API rate limiting
COINGECKO_RATE_LIMIT_DELAY = 6.0  # seconds between requests to avoid rate limiting (free tier is limited)

def handle_api_error(response):
    """Handle API error responses"""
    if response.status_code == 429:
        return {"error": "Rate limit exceeded. Please try again later."}
    elif response.status_code == 404:
        return {"error": "Resource not found."}
    else:
        return {"error": f"API error: {response.status_code} - {response.text}"}

@cache_result(expire_seconds=300)  # Cache for 5 minutes
def get_global_market_data():
    """Get global cryptocurrency market data"""
    try:
        # Wait to avoid rate limiting
        time.sleep(COINGECKO_RATE_LIMIT_DELAY)
        
        response = requests.get(f"{COINGECKO_API_BASE_URL}/global")
        
        if response.status_code == 200:
            try:
                data = response.json()
                if "data" not in data:
                    return {"error": "Incomplete global market data received"}
                return data["data"]
            except Exception as inner_e:
                return {"error": f"Error processing global market data: {str(inner_e)}"}
        elif response.status_code == 429:
            return {"error": "Rate limit exceeded for global market data. Please try again later."}
        else:
            error_response = handle_api_error(response)
            return {"error": f"API error: {error_response.get('error', 'Unknown error')}"}
    except Exception as e:
        return {"error": f"Error fetching global market data: {str(e)}"}

@cache_result(expire_seconds=300)  # Cache for 5 minutes
def get_top_cryptocurrencies(limit=TOP_CRYPTO_COUNT):
    """Get top cryptocurrencies by market cap"""
    try:
        # Wait to avoid rate limiting
        time.sleep(COINGECKO_RATE_LIMIT_DELAY)
        
        response = requests.get(
            f"{COINGECKO_API_BASE_URL}/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": limit,
                "page": 1,
                "sparkline": False,
                "price_change_percentage": "1h,24h,7d"
            }
        )
        
        if response.status_code == 200:
            try:
                data = response.json()
                if not data or len(data) == 0:
                    print("Warning: No cryptocurrency data received from API")
                    return pd.DataFrame()
                
                df = pd.DataFrame(data)
                if df.empty:
                    print("Warning: Empty DataFrame created from cryptocurrency data")
                    return pd.DataFrame()
                
                return df
            except Exception as inner_e:
                print(f"Error processing top cryptocurrencies data: {str(inner_e)}")
                return pd.DataFrame()
        elif response.status_code == 429:
            print(f"Rate limit exceeded for top cryptocurrencies. Please try again later.")
            return pd.DataFrame()
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching top cryptocurrencies: {str(e)}")
        return pd.DataFrame()

@cache_result(expire_seconds=300)  # Cache for 5 minutes
def get_coin_history(coin_id, days="7", interval="daily"):
    """Get historical price data for a specific coin"""
    try:
        # Wait to avoid rate limiting
        time.sleep(COINGECKO_RATE_LIMIT_DELAY)
        
        url = f"{COINGECKO_API_BASE_URL}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": interval if days != "max" and int(days) > 1 else None
        }
        
        # Make the API request
        response = requests.get(url, params={k: v for k, v in params.items() if v is not None})
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                # Check if the required data exists
                if "prices" not in data or "market_caps" not in data or "total_volumes" not in data:
                    return {"error": f"Incomplete data received for {coin_id}"}
                
                # Create DataFrames for prices, market caps, and volumes
                df_prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
                df_market_caps = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])
                df_volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
                
                # Check if dataframes are empty
                if df_prices.empty or df_market_caps.empty or df_volumes.empty:
                    return {"error": f"Empty data received for {coin_id}"}
                
                # Convert timestamps to datetime
                df_prices["timestamp"] = pd.to_datetime(df_prices["timestamp"], unit="ms")
                df_market_caps["timestamp"] = pd.to_datetime(df_market_caps["timestamp"], unit="ms")
                df_volumes["timestamp"] = pd.to_datetime(df_volumes["timestamp"], unit="ms")
                
                # Merge the DataFrames
                df = pd.merge(df_prices, df_market_caps, on="timestamp")
                df = pd.merge(df, df_volumes, on="timestamp")
                
                if not df.empty:
                    return df
                else:
                    return {"error": f"Failed to merge data for {coin_id}"}
            except Exception as inner_e:
                return {"error": f"Error processing data for {coin_id}: {str(inner_e)}"}
        elif response.status_code == 429:
            # Special handling for rate limit errors
            return {"error": f"Rate limit exceeded for {coin_id}. Please try again later or reduce the number of requests."}
        else:
            error_response = handle_api_error(response)
            return {"error": f"API error for {coin_id}: {error_response.get('error', 'Unknown error')}"}
    except Exception as e:
        return {"error": f"Error fetching coin history for {coin_id}: {str(e)}"}

@cache_result(expire_seconds=300)  # Cache for 5 minutes
def get_coin_details(coin_id):
    """Get detailed information about a specific coin"""
    try:
        # Wait to avoid rate limiting
        time.sleep(COINGECKO_RATE_LIMIT_DELAY)
        
        response = requests.get(f"{COINGECKO_API_BASE_URL}/coins/{coin_id}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                if not data:
                    return {"error": f"No details data available for {coin_id}"}
                return data
            except Exception as inner_e:
                return {"error": f"Error processing details data for {coin_id}: {str(inner_e)}"}
        elif response.status_code == 429:
            return {"error": f"Rate limit exceeded for {coin_id}. Please try again later or reduce the number of requests."}
        else:
            error_response = handle_api_error(response)
            return {"error": f"API error for {coin_id}: {error_response.get('error', 'Unknown error')}"}
    except Exception as e:
        return {"error": f"Error fetching coin details for {coin_id}: {str(e)}"}

@cache_result(expire_seconds=300)  # Cache for 5 minutes
def get_coin_ohlc(coin_id, days="7"):
    """Get OHLC (Open, High, Low, Close) data for a specific coin"""
    try:
        # Wait to avoid rate limiting
        time.sleep(COINGECKO_RATE_LIMIT_DELAY)
        
        url = f"{COINGECKO_API_BASE_URL}/coins/{coin_id}/ohlc"
        params = {
            "vs_currency": "usd",
            "days": days
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            try:
                data = response.json()
                if not data or len(data) == 0:
                    return {"error": f"No OHLC data available for {coin_id}"}
                
                df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                
                if not df.empty:
                    return df
                else:
                    return {"error": f"Empty OHLC data received for {coin_id}"}
            except Exception as inner_e:
                return {"error": f"Error processing OHLC data for {coin_id}: {str(inner_e)}"}
        elif response.status_code == 429:
            return {"error": f"Rate limit exceeded for {coin_id}. Please try again later or reduce the number of requests."}
        else:
            error_response = handle_api_error(response)
            return {"error": f"API error for {coin_id}: {error_response.get('error', 'Unknown error')}"}
    except Exception as e:
        return {"error": f"Error fetching OHLC data for {coin_id}: {str(e)}"}

@cache_result(expire_seconds=300)  # Cache for 5 minutes
def get_market_metrics():
    """Get comprehensive market metrics"""
    try:
        global_data = get_global_market_data()
        top_coins = get_top_cryptocurrencies(limit=TOP_CRYPTO_COUNT)
        
        if isinstance(global_data, dict) and "error" not in global_data and not top_coins.empty:
            market_cap_usd = global_data["total_market_cap"]["usd"]
            volume_usd = global_data["total_volume"]["usd"]
            
            # Calculate metrics from top coins
            if not top_coins.empty:
                largest_market_cap = top_coins["market_cap"].max()
                average_market_cap = top_coins["market_cap"].mean()
                median_market_cap = top_coins["market_cap"].median()
                average_volume = top_coins["total_volume"].mean()
                median_volume = top_coins["total_volume"].median()
                mode_price_change = top_coins["price_change_percentage_24h"].mode()[0] if not top_coins["price_change_percentage_24h"].isna().all() else 0
                std_dev_market_cap = top_coins["market_cap"].std()
                
                return {
                    "total_market_cap_usd": market_cap_usd,
                    "total_volume_usd": volume_usd,
                    "largest_market_cap": largest_market_cap,
                    "average_market_cap": average_market_cap,
                    "median_market_cap": median_market_cap,
                    "average_volume": average_volume,
                    "median_volume": median_volume,
                    "mode_price_change_24h": mode_price_change,
                    "std_dev_market_cap": std_dev_market_cap,
                    "btc_dominance": global_data["market_cap_percentage"]["btc"],
                    "eth_dominance": global_data["market_cap_percentage"]["eth"],
                    "active_cryptocurrencies": global_data["active_cryptocurrencies"]
                }
            else:
                return {"error": "Failed to retrieve top cryptocurrencies data"}
        else:
            error_msg = global_data.get("error", "Unknown error") if isinstance(global_data, dict) else "Unknown error"
            return {"error": f"Failed to retrieve market data: {error_msg}"}
    except Exception as e:
        return {"error": f"Error calculating market metrics: {str(e)}"}

def get_coin_comparison_data(coin_ids, vs_currency="usd"):
    """Get comparison data for multiple coins"""
    try:
        comparison_data = {}
        
        for coin_id in coin_ids:
            time.sleep(COINGECKO_RATE_LIMIT_DELAY)  # Avoid rate limiting
            coin_data = get_coin_details(coin_id)
            
            if isinstance(coin_data, dict) and "error" not in coin_data:
                # Extract relevant comparison metrics
                comparison_data[coin_id] = {
                    "name": coin_data.get("name", "Unknown"),
                    "symbol": coin_data.get("symbol", "Unknown"),
                    "market_cap": coin_data.get("market_data", {}).get("market_cap", {}).get(vs_currency, 0),
                    "price": coin_data.get("market_data", {}).get("current_price", {}).get(vs_currency, 0),
                    "24h_change": coin_data.get("market_data", {}).get("price_change_percentage_24h", 0),
                    "7d_change": coin_data.get("market_data", {}).get("price_change_percentage_7d", 0),
                    "30d_change": coin_data.get("market_data", {}).get("price_change_percentage_30d", 0),
                    "volume": coin_data.get("market_data", {}).get("total_volume", {}).get(vs_currency, 0),
                    "circulating_supply": coin_data.get("market_data", {}).get("circulating_supply", 0),
                    "total_supply": coin_data.get("market_data", {}).get("total_supply", 0),
                    "max_supply": coin_data.get("market_data", {}).get("max_supply", 0),
                    "all_time_high": coin_data.get("market_data", {}).get("ath", {}).get(vs_currency, 0),
                    "all_time_high_date": coin_data.get("market_data", {}).get("ath_date", {}).get(vs_currency, ""),
                }
            else:
                comparison_data[coin_id] = {"error": f"Failed to retrieve data for {coin_id}"}
        
        return comparison_data
    except Exception as e:
        return {"error": f"Error fetching comparison data: {str(e)}"}

def get_news_api_news(category="cryptocurrency", items=20):
    """Get cryptocurrency news from News API"""
    try:
        # Use environment variable for API key
        api_key = os.environ.get("NEWS_API_KEY", "")
        
        if not api_key:
            return {"error": "News API key not found. Please set the NEWS_API_KEY environment variable."}
        
        # Convert category to appropriate query
        query = "cryptocurrency"
        if category and category.lower() != "all news":
            query = category.lower()
        
        # Add Bitcoin to query if not already there for more relevant results
        if "bitcoin" not in query.lower() and "cryptocurrency" not in query.lower():
            query = f"{query} cryptocurrency"
        
        params = {
            "q": query,
            "apiKey": api_key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": items
        }
        
        response = requests.get(NEWS_API_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Transform News API response to match our expected format
            articles = []
            for article in data.get("articles", []):
                # Basic sentiment analysis - this could be improved with NLP later
                title = article.get("title", "")
                sentiment = "Neutral"
                if any(word in title.lower() for word in ["surge", "soar", "rally", "jump", "gain", "bull", "up"]):
                    sentiment = "Positive"
                elif any(word in title.lower() for word in ["crash", "drop", "plunge", "sink", "fall", "bear", "down"]):
                    sentiment = "Negative"
                
                # Extract mentioned cryptocurrencies - basic approach
                tickers = []
                if "bitcoin" in title.lower() or "btc" in title.lower():
                    tickers.append("BTC")
                if "ethereum" in title.lower() or "eth" in title.lower():
                    tickers.append("ETH")
                if "ripple" in title.lower() or "xrp" in title.lower():
                    tickers.append("XRP")
                if "litecoin" in title.lower() or "ltc" in title.lower():
                    tickers.append("LTC")
                
                articles.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "date": article.get("publishedAt", "").replace("T", " ").replace("Z", ""),
                    "sentiment": sentiment,
                    "tickers": tickers
                })
            
            return {"articles": articles}
        else:
            return handle_api_error(response)
    except Exception as e:
        return {"error": f"Error fetching news from News API: {str(e)}"}

def get_crypto_panic_news(category="", items=50):
    """Get cryptocurrency news from Crypto Panic API"""
    try:
        # Use environment variable for API key
        api_key = os.environ.get("CRYPTO_PANIC_API_KEY", "")
        
        if not api_key:
            return {"error": "Crypto Panic API key not found. Please set the CRYPTO_PANIC_API_KEY environment variable."}
        
        params = {
            "auth_token": api_key,
            "public": "true",
            "kind": "news",
        }
        
        # Map category to Crypto Panic format
        if category and category.lower() != "all news":
            if category.lower() == "bitcoin":
                params["currencies"] = "BTC"
            elif category.lower() == "ethereum":
                params["currencies"] = "ETH"
            elif category.lower() == "altcoins":
                params["currencies"] = "not:BTC,ETH"
            elif category.lower() == "defi":
                params["filter"] = "defi"
            elif category.lower() == "nft":
                params["filter"] = "nft"
            elif category.lower() == "regulation":
                params["filter"] = "regulation"
        
        response = requests.get(CRYPTO_PANIC_API_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Transform Crypto Panic response to match our expected format
            articles = []
            for post in data.get("results", [])[:items]:  # Limit to requested number of items
                # Get sentiment from votes
                votes = post.get("votes", {})
                positive = votes.get("positive", 0)
                negative = votes.get("negative", 0)
                
                if positive > negative:
                    sentiment = "Positive"
                elif negative > positive:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
                
                # Get mentioned currencies
                currencies = post.get("currencies", [])
                tickers = [currency.get("code", "") for currency in currencies if "code" in currency]
                
                # Format the date
                date_str = post.get("published_at", "")
                try:
                    date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    formatted_date = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    formatted_date = date_str
                
                articles.append({
                    "title": post.get("title", ""),
                    "description": post.get("description", ""),
                    "url": post.get("url", ""),
                    "source": post.get("source", {}).get("domain", "Unknown"),
                    "date": formatted_date,
                    "sentiment": sentiment,
                    "tickers": tickers
                })
            
            return {"articles": articles}
        else:
            return handle_api_error(response)
    except Exception as e:
        return {"error": f"Error fetching news from Crypto Panic API: {str(e)}"}

def get_crypto_news(category="", items=20):
    """Get cryptocurrency news from multiple sources"""
    try:
        # First try Crypto Panic API for specialized crypto news
        crypto_panic_news = get_crypto_panic_news(category, items)
        
        # If successful, return that
        if isinstance(crypto_panic_news, dict) and "error" not in crypto_panic_news and "articles" in crypto_panic_news:
            return crypto_panic_news
        
        # Fallback to News API for general news
        news_api_news = get_news_api_news(category, items)
        
        # If successful, return that
        if isinstance(news_api_news, dict) and "error" not in news_api_news and "articles" in news_api_news:
            return news_api_news
        
        # If both failed, return error with details
        cp_error = crypto_panic_news.get("error", "Unknown error") if isinstance(crypto_panic_news, dict) else "Unknown error"
        na_error = news_api_news.get("error", "Unknown error") if isinstance(news_api_news, dict) else "Unknown error"
        
        return {"error": f"Failed to fetch news from both sources. Crypto Panic: {cp_error}. News API: {na_error}"}
    except Exception as e:
        return {"error": f"Error fetching crypto news: {str(e)}"}

def get_timeframe_data(coin_id, timeframe):
    """Get data for a specific timeframe"""
    try:
        # For short timeframes (< 1 day), we'll get hourly or minute data
        if timeframe in ["1m", "5m", "15m", "30m", "45m"]:
            # For minute-level data, we need to get 1-day data with minute intervals
            # Note: CoinGecko free API has limited support for minute-level data
            days = "1"
            interval = "minute"
        elif timeframe in ["1h", "3h", "6h", "12h"]:
            days = "7"  # Get a week of data for hourly analysis
            interval = "hourly"
        else:
            # Convert timeframe to days
            if timeframe == "1d":
                days = "1"
            elif timeframe == "7d":
                days = "7"
            elif timeframe == "14d":
                days = "14"
            elif timeframe == "30d":
                days = "30"
            elif timeframe == "90d":
                days = "90"
            elif timeframe == "180d":
                days = "180"
            elif timeframe == "365d":
                days = "365"
            else:
                days = "7"  # Default to 7 days
            
            interval = "daily"
        
        data = get_coin_history(coin_id, days=days, interval=interval)
        
        if isinstance(data, pd.DataFrame) and not data.empty:
            return data
        else:
            return {"error": "Failed to retrieve timeframe data"}
    except Exception as e:
        return {"error": f"Error fetching timeframe data: {str(e)}"}
