import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from utils.api import (
    get_global_market_data, 
    get_top_cryptocurrencies, 
    get_coin_history,
    get_market_metrics,
    get_timeframe_data
)
from utils.visualization import (
    create_price_chart, 
    create_market_cap_chart,
    create_market_distribution_chart,
    create_market_dominance_chart,
    create_price_vs_volume_chart,
    create_crypto_bubble_chart
)
from utils.analysis import (
    calculate_technical_indicators,
    find_best_entry_points
)
from utils.styles import apply_styles
from utils.constants import DISPLAY_TIMEFRAMES, CHART_COLORS

# Set page config
st.set_page_config(
    page_title="Market Overview | CryptoAnalytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom styles
apply_styles()

def main():
    # Header
    st.title("📈 Market Overview")
    st.markdown(
        """
        Comprehensive overview of the cryptocurrency market with real-time data, 
        interactive charts, and key metrics across multiple timeframes.
        """
    )
    st.markdown("---")
    
    # Sidebar filters
    st.sidebar.header("Market Overview Filters")
    
    # Timeframe selection
    selected_timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        options=DISPLAY_TIMEFRAMES,
        index=DISPLAY_TIMEFRAMES.index("1d")
    )
    
    # Coin selection for detailed view
    try:
        top_coins_df = get_top_cryptocurrencies(limit=100)
        
        if not top_coins_df.empty:
            coin_options = [(row["id"], f"{row['name']} ({row['symbol'].upper()})") 
                            for _, row in top_coins_df.iterrows()]
            
            selected_coin_tuple = st.sidebar.selectbox(
                "Select Coin for Detailed View",
                options=coin_options,
                format_func=lambda x: x[1]
            )
            
            selected_coin_id = selected_coin_tuple[0]
            selected_coin_name = selected_coin_tuple[1]
        else:
            st.sidebar.warning("Unable to load coin list. Please try again later.")
            selected_coin_id = "bitcoin"
            selected_coin_name = "Bitcoin (BTC)"
    except Exception as e:
        st.sidebar.error(f"Error loading coin list: {str(e)}")
        selected_coin_id = "bitcoin"
        selected_coin_name = "Bitcoin (BTC)"
    
    # Auto refresh option
    auto_refresh = st.sidebar.checkbox("Auto refresh data (30s)", value=False)
    
    # Main content
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Total Market Cap", 
        "📈 Key Market Metrics", 
        "🔍 Market Visualization", 
        "💹 Detailed Market Data"
    ])
    
    with tab1:
        st.header("Total Market Cap")
        
        # Create two columns
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Get global market data
            global_market_data = get_global_market_data()
            
            if global_market_data and "error" not in global_market_data:
                # Display total market cap chart
                with st.spinner("Loading market cap data..."):
                    # Get Bitcoin data as a proxy for total market cap trend
                    market_cap_df = get_timeframe_data("bitcoin", selected_timeframe)
                    
                    if isinstance(market_cap_df, pd.DataFrame) and not market_cap_df.empty:
                        fig = create_market_cap_chart(market_cap_df, "Total Crypto Market", selected_timeframe)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Unable to load market cap chart data.")
            else:
                st.error("Unable to fetch global market data. Please try again later.")
        
        with col2:
            # Display current market stats
            st.subheader("Current Market Stats")
            
            if global_market_data and "error" not in global_market_data:
                total_market_cap = global_market_data.get("total_market_cap", {}).get("usd", 0)
                total_volume = global_market_data.get("total_volume", {}).get("usd", 0)
                btc_dominance = global_market_data.get("market_cap_percentage", {}).get("btc", 0)
                eth_dominance = global_market_data.get("market_cap_percentage", {}).get("eth", 0)
                
                st.metric(
                    "Total Market Cap", 
                    f"${total_market_cap:,.0f}", 
                    f"{global_market_data.get('market_cap_change_percentage_24h_usd', 0):.2f}%"
                )
                st.metric("Total 24h Volume", f"${total_volume:,.0f}")
                st.metric("BTC Dominance", f"{btc_dominance:.2f}%")
                st.metric("ETH Dominance", f"{eth_dominance:.2f}%")
                st.metric("Active Cryptocurrencies", f"{global_market_data.get('active_cryptocurrencies', 0):,}")
            else:
                st.warning("Market stats unavailable. Please try again later.")
        
        # Best Entry Points Analysis
        st.subheader("Best Entry Points Based on Historical Data")
        
        with st.spinner("Analyzing entry points..."):
            entry_points = find_best_entry_points(selected_coin_id, timeframe=selected_timeframe)
            
            if isinstance(entry_points, dict) and "error" not in entry_points:
                # Display summary of best strategies
                success_rates = entry_points.get("success_rates", {})
                best_strategy = entry_points.get("best_strategy", None)
                
                if best_strategy:
                    strategy_name, success_rate = best_strategy
                    
                    st.info(f"📊 **Best Entry Strategy for {selected_coin_name}**: "
                           f"_{strategy_name.replace('_', ' ').title()}_ with "
                           f"**{success_rate:.1f}%** historical success rate")
                    
                    # Display entry points for the best strategy
                    best_points = entry_points.get("entry_strategies", {}).get(strategy_name, [])
                    
                    if best_points:
                        st.markdown(f"**Historical entry points using this strategy:**")
                        
                        # Create a dataframe for the points
                        entry_df = pd.DataFrame(best_points)
                        
                        if not entry_df.empty:
                            # Format the dataframe
                            entry_df["timestamp"] = pd.to_datetime(entry_df["timestamp"])
                            entry_df["date"] = entry_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
                            
                            # Select and rename columns based on the strategy
                            if strategy_name == "rsi_oversold":
                                display_df = entry_df[["date", "price", "RSI"]]
                                display_df = display_df.rename(columns={"RSI": "RSI Value"})
                            elif strategy_name == "bollinger_band_lower":
                                display_df = entry_df[["date", "price", "BB_lower"]]
                                display_df = display_df.rename(columns={"BB_lower": "Lower Band"})
                            elif strategy_name == "golden_cross":
                                display_df = entry_df[["date", "price", "SMA_7", "SMA_25"]]
                                display_df = display_df.rename(columns={"SMA_7": "SMA (7)", "SMA_25": "SMA (25)"})
                            elif strategy_name == "macd_cross_above":
                                display_df = entry_df[["date", "price", "MACD_line", "MACD_signal"]]
                                display_df = display_df.rename(columns={"MACD_line": "MACD Line", "MACD_signal": "Signal Line"})
                            elif strategy_name == "high_volume_price_increase":
                                display_df = entry_df[["date", "price", "volume"]]
                                display_df = display_df.rename(columns={"volume": "Volume"})
                            else:
                                display_df = entry_df[["date", "price"]]
                            
                            # Format price column
                            if "price" in display_df.columns:
                                display_df["price"] = display_df["price"].apply(lambda x: f"${x:,.2f}")
                                display_df = display_df.rename(columns={"price": "Price"})
                            
                            # Format volume column if it exists
                            if "Volume" in display_df.columns:
                                display_df["Volume"] = display_df["Volume"].apply(lambda x: f"${x:,.0f}")
                            
                            # Display the dataframe
                            st.dataframe(display_df, use_container_width=True)
                        else:
                            st.warning("No specific entry points found for this strategy.")
                    else:
                        st.warning("No specific entry points found for this strategy.")
                    
                    # Show success rates for all strategies
                    st.markdown("**Success rates of all strategies:**")
                    
                    # Create a dataframe for success rates
                    success_df = pd.DataFrame({
                        "Strategy": [k.replace("_", " ").title() for k in success_rates.keys()],
                        "Success Rate (%)": [f"{v:.1f}%" for v in success_rates.values()]
                    })
                    
                    st.dataframe(success_df, use_container_width=True)
                else:
                    st.warning("Unable to determine the best entry strategy with available data.")
            else:
                error_msg = entry_points.get("error", "Unknown error") if isinstance(entry_points, dict) else "Unknown error"
                st.error(f"Error analyzing entry points: {error_msg}")
    
    with tab2:
        st.header("Key Market Metrics")
        
        # Get market metrics
        with st.spinner("Loading market metrics..."):
            metrics = get_market_metrics()
            
            if isinstance(metrics, dict) and "error" not in metrics:
                # Create 3 columns for metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Market Size")
                    st.metric("Total Market Cap", f"${metrics['total_market_cap_usd']:,.0f}")
                    st.metric("Largest Market Cap", f"${metrics['largest_market_cap']:,.0f}")
                    st.metric("Active Cryptocurrencies", f"{metrics['active_cryptocurrencies']:,}")
                
                with col2:
                    st.subheader("Market Averages")
                    st.metric("Average Market Cap", f"${metrics['average_market_cap']:,.0f}")
                    st.metric("Median Market Cap", f"${metrics['median_market_cap']:,.0f}")
                    st.metric("Average 24h Volume", f"${metrics['average_volume']:,.0f}")
                
                with col3:
                    st.subheader("Market Dominance")
                    st.metric("Bitcoin Dominance", f"{metrics['btc_dominance']:.2f}%")
                    st.metric("Ethereum Dominance", f"{metrics['eth_dominance']:.2f}%")
                    st.metric("Standard Deviation", f"${metrics['std_dev_market_cap']:,.0f}")
                
                # Additional metrics in expandable sections
                with st.expander("Trading Volume Statistics"):
                    vol_col1, vol_col2 = st.columns(2)
                    
                    with vol_col1:
                        st.metric("Total 24h Volume", f"${metrics['total_volume_usd']:,.0f}")
                        st.metric("Average Volume", f"${metrics['average_volume']:,.0f}")
                    
                    with vol_col2:
                        st.metric("Median Volume", f"${metrics['median_volume']:,.0f}")
                        st.metric("Volume to Market Cap Ratio", f"{(metrics['total_volume_usd'] / metrics['total_market_cap_usd']) * 100:.2f}%")
                
                with st.expander("Market Distribution Analysis"):
                    # Show market dominance as a pie chart
                    global_data = get_global_market_data()
                    if global_data and "error" not in global_data:
                        fig = create_market_distribution_chart(global_data)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Unable to create market distribution chart.")
                    else:
                        st.warning("Market distribution data unavailable.")
                
                with st.expander("Price Change Statistics"):
                    st.metric(
                        "Mode Price Change 24h", 
                        f"{metrics['mode_price_change_24h']:.2f}%"
                    )
                    
                    # Get top gainers and losers
                    if not top_coins_df.empty:
                        top_gainers = top_coins_df.nlargest(5, 'price_change_percentage_24h')
                        top_losers = top_coins_df.nsmallest(5, 'price_change_percentage_24h')
                        
                        gain_col1, gain_col2 = st.columns(2)
                        
                        with gain_col1:
                            st.subheader("Top Gainers (24h)")
                            for _, coin in top_gainers.iterrows():
                                st.metric(
                                    f"{coin['name']} ({coin['symbol'].upper()})",
                                    f"${coin['current_price']:,.8f}",
                                    f"{coin['price_change_percentage_24h']:.2f}%"
                                )
                        
                        with gain_col2:
                            st.subheader("Top Losers (24h)")
                            for _, coin in top_losers.iterrows():
                                st.metric(
                                    f"{coin['name']} ({coin['symbol'].upper()})",
                                    f"${coin['current_price']:,.8f}",
                                    f"{coin['price_change_percentage_24h']:.2f}%"
                                )
            else:
                error_msg = metrics.get("error", "Unknown error") if isinstance(metrics, dict) else "Unknown error"
                st.error(f"Error loading market metrics: {error_msg}")
    
    with tab3:
        st.header("Market Visualization")
        
        # Create tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "Market Distribution", 
            "Market Dominance", 
            "Price vs Volume", 
            "Crypto Bubble"
        ])
        
        with viz_tab1:
            st.subheader("Market Distribution")
            
            # Get global market data for distribution
            global_data = get_global_market_data()
            
            if global_data and "error" not in global_data:
                fig = create_market_distribution_chart(global_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Unable to create market distribution chart.")
            else:
                st.warning("Market distribution data unavailable.")
            
            # Add description
            st.markdown("""
            **About This Chart:**
            
            The cryptocurrency market distribution shows the relative market capitalization of major cryptocurrencies. 
            Bitcoin and Ethereum typically dominate the market, with the rest of the market distributed among altcoins.
            
            This visualization helps understand the market concentration and identify which cryptocurrencies have the most significant influence on the overall market.
            """)
        
        with viz_tab2:
            st.subheader("Top Coins Relative Performance")
            
            # Get top 5 cryptocurrencies data for comparison
            with st.spinner("Loading comparative performance data..."):
                if not top_coins_df.empty:
                    # Get top 5 coins for comparison
                    top_5_coins = top_coins_df.head(5)
                    top_5_ids = top_5_coins['id'].tolist()
                    
                    from utils.api import get_coin_comparison_data
                    comparison_data = get_coin_comparison_data(top_5_ids, "usd")
                    
                    if "error" not in comparison_data:
                        from utils.visualization import create_comparison_chart
                        fig = create_comparison_chart(comparison_data, metric="price", timeframe=selected_timeframe)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Unable to create comparison chart.")
                    else:
                        st.error(f"Error fetching comparison data: {comparison_data.get('error', 'Unknown error')}")
                else:
                    st.error("Unable to load top cryptocurrencies for comparison.")
            
            st.markdown("""
            **About Relative Performance:**
            
            This chart shows the relative price performance of the top 5 cryptocurrencies by market capitalization, 
            normalized to show percentage change over the selected time period.
            
            This comparison helps identify which cryptocurrencies are outperforming or underperforming relative to their peers.
            """)
        
        with viz_tab3:
            st.subheader("Price vs Volume Analysis")
            
            # Get data for selected coin
            with st.spinner("Loading price and volume data..."):
                price_volume_data = get_timeframe_data(selected_coin_id, selected_timeframe)
                
                if isinstance(price_volume_data, pd.DataFrame) and not price_volume_data.empty:
                    fig = create_price_vs_volume_chart(price_volume_data, selected_coin_name.split(" (")[0], selected_timeframe)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Unable to create price vs volume chart.")
                else:
                    error_msg = price_volume_data.get("error", "Unknown error") if isinstance(price_volume_data, dict) else "Unknown error"
                    st.error(f"Error loading price and volume data: {error_msg}")
            
            # Add description
            st.markdown("""
            **Understanding Price vs Volume:**
            
            This scatter plot shows the relationship between price changes and volume changes, which can help identify patterns in trading behavior.
            
            **Key insights:**
            - **High volume + price increase**: Potential strong bullish signal
            - **High volume + price decrease**: Potential strong bearish signal
            - **Low volume + price change**: Potentially less significant movement
            - **Clusters**: Identify common trading patterns
            
            The relationship between price and volume is a key technical analysis concept that can help forecast potential price movements.
            """)
        
        with viz_tab4:
            st.subheader("Cryptocurrency Bubble Chart")
            
            # Get top cryptocurrencies for bubble chart
            with st.spinner("Loading data for bubble chart..."):
                if not top_coins_df.empty:
                    # Limit to top 50 for better visualization
                    bubble_data = top_coins_df.head(50)
                    
                    fig = create_crypto_bubble_chart(bubble_data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Unable to create crypto bubble chart.")
                else:
                    st.error("Unable to load data for bubble chart.")
            
            # Add description
            st.markdown("""
            **About The Bubble Chart:**
            
            This bubble chart visualizes three dimensions of data:
            1. **X-axis**: 24-hour price change percentage
            2. **Y-axis**: 24-hour trading volume (logarithmic scale)
            3. **Bubble size**: Market capitalization
            
            This multi-dimensional view helps identify cryptocurrencies with high market cap, significant trading activity, and notable price movements all in one visualization.
            
            Cryptocurrencies in the upper right quadrant (high volume, positive price change) might represent bullish momentum, while those in the upper left (high volume, negative price change) might indicate bearish pressure.
            """)
    
    with tab4:
        st.header("Detailed Market Cap Data")
        
        # Create tabs for different data views
        data_tab1, data_tab2 = st.tabs(["Top 100 Cryptocurrencies", "Detailed Coin Analysis"])
        
        with data_tab1:
            st.subheader("Top 100 Cryptocurrencies by Market Cap")
            
            # Display top cryptocurrencies table
            if not top_coins_df.empty:
                # Format the dataframe for display
                display_df = top_coins_df.copy()
                
                # Select and rename columns
                display_df = display_df[[
                    'market_cap_rank', 'name', 'symbol', 'current_price', 
                    'market_cap', 'total_volume', 'price_change_percentage_24h'
                ]]
                
                display_df = display_df.rename(columns={
                    'market_cap_rank': 'Rank',
                    'name': 'Name',
                    'symbol': 'Symbol',
                    'current_price': 'Price (USD)',
                    'market_cap': 'Market Cap',
                    'total_volume': '24h Volume',
                    'price_change_percentage_24h': '24h Change %'
                })
                
                # Format values
                display_df['Symbol'] = display_df['Symbol'].str.upper()
                display_df['Price (USD)'] = display_df['Price (USD)'].apply(lambda x: f"${x:,.8f}" if x < 1 else f"${x:,.2f}")
                display_df['Market Cap'] = display_df['Market Cap'].apply(lambda x: f"${x:,.0f}")
                display_df['24h Volume'] = display_df['24h Volume'].apply(lambda x: f"${x:,.0f}")
                display_df['24h Change %'] = display_df['24h Change %'].apply(lambda x: f"{x:.2f}%")
                
                # Apply styling based on price change
                def highlight_change(val):
                    val_num = float(val.replace('%', ''))
                    if val_num > 0:
                        return 'color: #00C853; font-weight: bold'
                    elif val_num < 0:
                        return 'color: #FF3D00; font-weight: bold'
                    else:
                        return ''
                
                # Apply the styling
                styled_df = display_df.style.map(highlight_change, subset=['24h Change %'])
                
                # Display the table
                st.dataframe(styled_df, use_container_width=True)
                
                # Add download button
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Table as CSV",
                    csv,
                    "cryptocurrency_market_data.csv",
                    "text/csv",
                    key='download-csv'
                )
            else:
                st.error("Unable to load cryptocurrency data. Please try again later.")
        
        with data_tab2:
            st.subheader(f"Detailed Analysis: {selected_coin_name}")
            
            # Get detailed data for the selected coin
            with st.spinner(f"Loading detailed data for {selected_coin_name}..."):
                coin_data = get_timeframe_data(selected_coin_id, selected_timeframe)
                
                if isinstance(coin_data, pd.DataFrame) and not coin_data.empty:
                    # Create tabs for Technical Analysis, Performance Metrics, and Price Prediction
                    analysis_tabs = st.tabs(["Technical Indicators", "Performance Metrics", "Price Prediction"])
                    
                    # Calculate technical indicators
                    technical_data = calculate_technical_indicators(coin_data)
                    
                    with analysis_tabs[0]:  # Technical Indicators Tab
                        if isinstance(technical_data, pd.DataFrame) and not technical_data.empty:
                            # Create price chart with technical indicators
                            with st.spinner("Generating technical indicators chart..."):
                                fig = create_price_chart(technical_data, selected_coin_name.split(" (")[0], selected_timeframe)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Unable to create detailed price chart.")
                            
                            # Display technical indicators in an expander
                            with st.expander("Current Technical Indicators", expanded=True):
                                # Select the most recent data point for current indicators
                                latest_data = technical_data.iloc[-1]
                                
                                # Create columns for different indicator groups
                                ind_col1, ind_col2, ind_col3 = st.columns(3)
                                
                                with ind_col1:
                                    st.subheader("Moving Averages")
                                    
                                    # Check if values are NaN before formatting
                                    sma7_val = latest_data['SMA_7']
                                    sma7_display = f"${sma7_val:.2f}" if pd.notna(sma7_val) else "Insufficient data"
                                    
                                    sma25_val = latest_data['SMA_25']
                                    sma25_display = f"${sma25_val:.2f}" if pd.notna(sma25_val) else "Insufficient data"
                                    
                                    ema7_val = latest_data['EMA_7']
                                    ema7_display = f"${ema7_val:.2f}" if pd.notna(ema7_val) else "Insufficient data"
                                    
                                    ema25_val = latest_data['EMA_25']
                                    ema25_display = f"${ema25_val:.2f}" if pd.notna(ema25_val) else "Insufficient data"
                                    
                                    st.metric("SMA (7)", sma7_display)
                                    st.metric("SMA (25)", sma25_display)
                                    st.metric("EMA (7)", ema7_display)
                                    st.metric("EMA (25)", ema25_display)
                                
                                with ind_col2:
                                    st.subheader("Oscillators")
                                    
                                    # RSI with color coding
                                    rsi_value = latest_data['RSI']
                                    
                                    if pd.notna(rsi_value):
                                        rsi_color = "normal"
                                        if rsi_value >= 70:
                                            rsi_color = "inverse"  # Overbought - inverse means red (bad)
                                        elif rsi_value <= 30:
                                            rsi_color = "normal"   # Oversold - normal means green (good buying opportunity)
                                        
                                        st.metric("RSI (14)", f"{rsi_value:.2f}", delta_color=rsi_color)
                                    else:
                                        st.metric("RSI (14)", "Insufficient data")
                                    
                                    # MACD
                                    macd_line = latest_data['MACD_line']
                                    macd_signal = latest_data['MACD_signal']
                                    macd_hist = latest_data['MACD_histogram']
                                    
                                    if pd.notna(macd_line) and pd.notna(macd_signal):
                                        macd_color = "normal" if macd_line > macd_signal else "inverse"
                                        
                                        st.metric("MACD Line", f"{macd_line:.6f}")
                                        st.metric("MACD Signal", f"{macd_signal:.6f}")
                                        st.metric("MACD Histogram", f"{macd_hist:.6f}", delta_color=macd_color)
                                    else:
                                        st.metric("MACD Line", "Insufficient data")
                                        st.metric("MACD Signal", "Insufficient data")
                                        st.metric("MACD Histogram", "Insufficient data")
                                
                                with ind_col3:
                                    st.subheader("Bollinger Bands")
                                    
                                    # Calculate position within Bollinger Bands as percentage
                                    bb_upper = latest_data['BB_upper']
                                    bb_lower = latest_data['BB_lower']
                                    bb_middle = latest_data['BB_middle']
                                    price = latest_data['price']
                                    
                                    if pd.notna(bb_upper) and pd.notna(bb_lower) and pd.notna(price):
                                        # Position within bands (0% = at lower band, 100% = at upper band)
                                        if bb_upper - bb_lower > 0:
                                            bb_position = ((price - bb_lower) / (bb_upper - bb_lower)) * 100
                                            bb_position_text = f"{bb_position:.1f}%"
                                            
                                            # Determine color based on position
                                            bb_color = "normal"
                                            if bb_position >= 80:
                                                bb_color = "inverse"  # Close to upper band (potentially overbought)
                                            elif bb_position <= 20:
                                                bb_color = "normal"   # Close to lower band (potentially oversold)
                                        else:
                                            bb_position_text = "N/A"
                                            bb_color = "normal"
                                        
                                        st.metric("Upper Band", f"${bb_upper:.2f}")
                                        st.metric("Middle Band", f"${bb_middle:.2f}")
                                        st.metric("Lower Band", f"${bb_lower:.2f}")
                                        st.metric("Position within Bands", bb_position_text, delta_color=bb_color)
                                    else:
                                        st.metric("Upper Band", "Insufficient data")
                                        st.metric("Middle Band", "Insufficient data")
                                        st.metric("Lower Band", "Insufficient data")
                                        st.metric("Position within Bands", "Insufficient data")
                            
                            # Display recent price data table
                            with st.expander("Recent Price Data"):
                                # Get the last 20 data points
                                recent_data = technical_data.tail(20).copy()
                                
                                # Format for display
                                recent_data['timestamp'] = recent_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                                recent_data = recent_data[['timestamp', 'price', 'volume', 'market_cap', 'RSI', 'MACD_line', 'MACD_signal']]
                                recent_data.columns = ['Timestamp', 'Price (USD)', 'Volume (USD)', 'Market Cap (USD)', 'RSI', 'MACD Line', 'MACD Signal']
                                
                                # Format numeric columns safely
                                recent_data['Price (USD)'] = recent_data['Price (USD)'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
                                recent_data['Volume (USD)'] = recent_data['Volume (USD)'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                                recent_data['Market Cap (USD)'] = recent_data['Market Cap (USD)'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                                
                                # Display the table
                                st.dataframe(recent_data, use_container_width=True)
                        else:
                            error_msg = technical_data.get("error", "Unknown error") if isinstance(technical_data, dict) else "Unknown error"
                            st.error(f"Error calculating technical indicators: {error_msg}")
                    
                    with analysis_tabs[1]:  # Performance Metrics Tab
                        with st.spinner("Calculating performance metrics..."):
                            from utils.analysis import calculate_performance_metrics
                            performance = calculate_performance_metrics(coin_data)
                            
                            if isinstance(performance, dict) and "error" not in performance:
                                # Display performance metrics in organized columns
                                st.subheader("Price Performance")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    price_change = performance["price_change"]
                                    price_change_pct = performance["price_change_pct"]
                                    
                                    price_color = "normal"
                                    if price_change_pct > 0:
                                        price_change_text = f"+${price_change:.2f} (+{price_change_pct:.2f}%)"
                                        price_color = "normal"
                                    else:
                                        price_change_text = f"-${abs(price_change):.2f} ({price_change_pct:.2f}%)"
                                        price_color = "inverse"
                                    
                                    st.metric(
                                        "Price Change", 
                                        f"${performance['end_price']:.4f}", 
                                        delta=price_change_text,
                                        delta_color=price_color
                                    )
                                    
                                    st.metric("Start Price", f"${performance['start_price']:.4f}")
                                    st.metric("End Price", f"${performance['end_price']:.4f}")
                                
                                with col2:
                                    st.metric("High", f"${performance['max_price']:.4f}")
                                    st.metric("Low", f"${performance['min_price']:.4f}")
                                    st.metric("Range", f"${performance['price_range']:.4f} ({performance['price_range_pct']:.2f}%)")
                                
                                with col3:
                                    st.metric("Volatility", f"{performance['volatility']:.2f}%")
                                    st.metric("Mean Price", f"${performance['mean_price']:.4f}")
                                    st.metric("Median Price", f"${performance['median_price']:.4f}")
                                
                                # Volume metrics
                                st.subheader("Volume Metrics")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Average Volume", f"${performance['avg_volume']:,.0f}")
                                
                                with col2:
                                    st.metric("Max Volume", f"${performance['max_volume']:,.0f}")
                                
                                # Statistical metrics
                                st.subheader("Statistical Analysis")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Standard Deviation", f"${performance['std_dev']:.4f}")
                                
                                with col2:
                                    st.metric("Coefficient of Variation", f"{performance['cv']:.2f}%")
                                
                                with col3:
                                    # Use a simpler sharpe ratio explanation for non-technical users
                                    sharpe_color = "normal" if performance['sharpe_ratio'] > 1 else "inverse"
                                    st.metric(
                                        "Risk-Adjusted Return", 
                                        f"{performance['sharpe_ratio']:.2f}", 
                                        delta="Good" if performance['sharpe_ratio'] > 1 else "Poor",
                                        delta_color=sharpe_color
                                    )
                            else:
                                error_msg = performance.get("error", "Unknown error") if isinstance(performance, dict) else "Unknown error"
                                st.error(f"Error calculating performance metrics: {error_msg}")
                    
                    with analysis_tabs[2]:  # Price Prediction Tab
                        # Add tabs for different prediction methods
                        pred_tab1, pred_tab2 = st.tabs(["Basic Prediction", "ML Prediction"])
                        
                        with pred_tab1:
                            with st.spinner("Calculating basic price prediction..."):
                                from utils.analysis import predict_price
                                
                                # Only show prediction for smaller timeframes
                                valid_prediction_timeframes = ["1m", "5m", "15m", "30m", "1h", "2h"]
                                
                                if selected_timeframe in valid_prediction_timeframes:
                                    prediction = predict_price(selected_coin_id, selected_timeframe)
                                    
                                    if isinstance(prediction, dict) and "error" not in prediction:
                                        st.subheader("Short-term Price Prediction (ARIMA)")
                                        
                                        # Display current price and prediction
                                        current_price = prediction["current_price"]
                                        predicted_prices = prediction["predicted_prices"]
                                        
                                        # Calculate average prediction
                                        avg_prediction = sum(predicted_prices) / len(predicted_prices)
                                        avg_change_pct = ((avg_prediction - current_price) / current_price) * 100
                                        
                                        # Display the prediction with color coding
                                        price_color = "normal" if avg_change_pct >= 0 else "inverse"
                                        st.metric(
                                            "Average Predicted Price", 
                                            f"${avg_prediction:.4f}", 
                                            delta=f"{avg_change_pct:+.2f}%",
                                            delta_color=price_color
                                        )
                                        
                                        # Display accuracy
                                        accuracy = prediction.get("accuracy", 0)
                                        st.metric("Prediction Confidence", f"{accuracy:.1f}%")
                                        
                                        # Warning about predictions
                                        st.warning(
                                            "⚠️ Price predictions are estimates based on historical patterns and should not be used as the sole basis for investment decisions. " +
                                            "Markets are volatile and past performance does not guarantee future results."
                                        )
                                        
                                        # Show predicted prices for next few time periods
                                        st.subheader("Predicted Price Trajectory")
                                    
                                    else:
                                        error_msg = prediction.get("error", "Unknown error") if isinstance(prediction, dict) else "Unknown error"
                                        st.error(f"Error calculating price prediction: {error_msg}")
                                else:
                                    st.info(
                                        "Short-term price predictions are only available for timeframes of 1 hour or less. " +
                                        "Please select a smaller timeframe (1m, 5m, 15m, 30m, or 1h) to see price predictions."
                                    )
                        
                        with pred_tab2:
                            with st.spinner("Loading machine learning prediction..."):
                                try:
                                    from utils.ml_models import predict_prices
                                    
                                    # Check if we have enough data for ML prediction
                                    if isinstance(coin_data, pd.DataFrame) and not coin_data.empty and len(coin_data) >= 48:
                                        # Only predict for a shorter period
                                        prediction_periods = 12
                                        
                                        # For ML predictions, we can use more timeframes
                                        ml_prediction = predict_prices(
                                            selected_coin_id, 
                                            selected_timeframe, 
                                            coin_data, 
                                            periods=prediction_periods
                                        )
                                        
                                        if "error" in ml_prediction:
                                            st.info(f"ML prediction not available: {ml_prediction['error']}")
                                            st.info("For detailed ML-based predictions, please visit the ML Price Prediction page.")
                                        else:
                                            st.subheader(f"ML-based Price Prediction ({ml_prediction['model_used']})")
                                            
                                            # Create prediction chart
                                            fig = go.Figure()
                                            
                                            # Add historical data
                                            hist_len = min(24, len(coin_data))
                                            historical_data = coin_data.tail(hist_len).copy()
                                            
                                            fig.add_trace(
                                                go.Scatter(
                                                    x=historical_data.index,
                                                    y=historical_data['price'],
                                                    mode='lines',
                                                    name='Historical Price',
                                                    line=dict(color='#1E88E5', width=2)
                                                )
                                            )
                                            
                                            # Add predicted data
                                            fig.add_trace(
                                                go.Scatter(
                                                    x=ml_prediction['timestamps'],
                                                    y=ml_prediction['predictions'],
                                                    mode='lines+markers',
                                                    name='ML Prediction',
                                                    line=dict(color='#00C853', width=3, dash='dash'),
                                                    marker=dict(size=6)
                                                )
                                            )
                                            
                                            # Update layout
                                            fig.update_layout(
                                                title=f"ML Price Prediction ({prediction_periods} periods ahead)",
                                                height=400,
                                                template="plotly_dark",
                                                hovermode="x unified",
                                                margin=dict(l=20, r=20, t=50, b=20),
                                                paper_bgcolor="#0A192F",
                                                plot_bgcolor="#172A46"
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Show prediction metrics
                                            current_price = ml_prediction['current_price']
                                            final_price = ml_prediction['predictions'][-1]
                                            change_pct = ((final_price / current_price) - 1) * 100
                                            
                                            price_color = "normal" if change_pct >= 0 else "inverse"
                                            st.metric(
                                                "Final Predicted Price", 
                                                f"${final_price:.4f}", 
                                                delta=f"{change_pct:+.2f}%",
                                                delta_color=price_color
                                            )
                                            
                                            st.info("For more detailed ML predictions and model information, visit the ML Price Prediction page.")
                                    else:
                                        st.info(
                                            "Machine learning predictions require more data. " +
                                            "Please select a longer timeframe or visit the ML Price Prediction page for more options."
                                        )
                                except Exception as e:
                                    st.error(f"Error loading ML prediction: {str(e)}")
                                    st.info("For detailed ML-based predictions, please visit the ML Price Prediction page.")
                                    
                            if "pred_tab1" in locals():
                                with pred_tab1:
                                    if "prediction" in locals() and isinstance(prediction, dict) and "error" not in prediction:
                                        # Show predicted prices for next few time periods
                                        if "predicted_prices" in prediction and "price_change_pct" in prediction:
                                            pred_df = pd.DataFrame({
                                                "Time Period": [f"+{i+1} periods" for i in range(len(prediction['predicted_prices']))],
                                                "Predicted Price": [f"${price:.4f}" for price in prediction['predicted_prices']],
                                                "Change %": [f"{pct:+.2f}%" for pct in prediction["price_change_pct"]]
                                            })
                                            st.dataframe(pred_df, use_container_width=True)
                else:
                    error_msg = coin_data.get("error", "Unknown error") if isinstance(coin_data, dict) else "Unknown error"
                    st.error(f"Error loading detailed coin data: {error_msg}")

    # Auto-refresh mechanism
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
