# Application constants
APP_TITLE = "CryptoAnalytics Dashboard"
APP_DESCRIPTION = """
A comprehensive cryptocurrency analysis platform providing real-time market data visualization, 
token comparison, technical analysis, and AI-powered whitepaper insights.
"""

# API endpoints
COINGECKO_API_BASE_URL = "https://api.coingecko.com/api/v3"
CRYPTO_NEWS_API_URL = "https://cryptonews-api.com/api/v1"
NEWS_API_URL = "https://newsapi.org/v2/everything"
CRYPTO_PANIC_API_URL = "https://cryptopanic.com/api/v1/posts"

# Timeframes in minutes for data fetching and display
TIMEFRAMES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "45m": 45,
    "1h": 60,
    "3h": 180,
    "6h": 360,
    "12h": 720,
    "1d": 1440,
    "7d": 10080,
    "14d": 20160,
    "30d": 43200,
    "90d": 129600,
    "180d": 259200,
    "365d": 525600,
}

# Display timeframes for UI
DISPLAY_TIMEFRAMES = [
    "1m", "5m", "15m", "30m", "45m", "1h", "3h", 
    "6h", "12h", "1d", "7d", "14d", "30d", "90d", "365d"
]

# Chart colors
CHART_COLORS = {
    "price_up": "#00C853",
    "price_down": "#FF3D00",
    "volume": "#3366CC",
    "market_cap": "#FB8C00",
    "dominance": "#8E24AA",
    "comparison": [
        "#1E88E5", "#D81B60", "#8E24AA", "#43A047", 
        "#FB8C00", "#546E7A", "#6D4C41", "#039BE5"
    ],
    "distribution": [
        "#1E88E5", "#D81B60", "#8E24AA", "#43A047", 
        "#FB8C00", "#546E7A", "#6D4C41", "#039BE5",
        "#00ACC1", "#3949AB", "#7CB342", "#FFB300"
    ]
}

# Number of top cryptocurrencies to fetch
TOP_CRYPTO_COUNT = 100

# Default cryptocurrencies for comparison
DEFAULT_COMPARISON_CRYPTOS = ["bitcoin", "ethereum", "binancecoin", "solana", "ripple"]

# News categories
NEWS_CATEGORIES = ["All News", "Bitcoin", "Ethereum", "Altcoins", "DeFi", "NFT", "Regulation"]
