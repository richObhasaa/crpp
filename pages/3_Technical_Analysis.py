import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.api import (
    get_top_cryptocurrencies,
    get_timeframe_data
)
from utils.visualization import (
    create_price_chart,
    create_technical_indicator_chart
)
from utils.analysis import (
    calculate_technical_indicators,
    calculate_performance_metrics,
    calculate_profit_loss,
    predict_price
)
from utils.styles import apply_styles
from utils.constants import DISPLAY_TIMEFRAMES, CHART_COLORS

# Set page config
st.set_page_config(
    page_title="Technical Analysis | CryptoAnalytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom styles
apply_styles()

def main():
    # Header
    st.title("📊 Technical Analysis")
    st.markdown(
        """
        Advanced technical analysis tools for cryptocurrency price movements with multiple 
        timeframes and performance calculator.
        """
    )
    st.markdown("---")
    
    # Load top cryptocurrencies for selection
    with st.spinner("Loading cryptocurrency data..."):
        top_cryptos_df = get_top_cryptocurrencies(limit=100)
        
        if top_cryptos_df.empty:
            st.error("Unable to load cryptocurrency data. Please check your internet connection and try again.")
            return
            
        # Create a dict of id:name pairs for the selection widget
        crypto_options = [(row["id"], f"{row['name']} ({row['symbol'].upper()})") 
                        for _, row in top_cryptos_df.iterrows()]
    
    # Sidebar filters
    st.sidebar.header("Technical Analysis Options")
    
    # Cryptocurrency selection
    selected_crypto_tuple = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=crypto_options,
        index=0,
        format_func=lambda x: x[1]
    )
    
    selected_crypto_id = selected_crypto_tuple[0]
    selected_crypto_name = selected_crypto_tuple[1]
    
    # Timeframe selection
    selected_timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        options=DISPLAY_TIMEFRAMES,
        index=DISPLAY_TIMEFRAMES.index("1d")
    )
    
    # Load data for the selected cryptocurrency
    with st.spinner(f"Loading data for {selected_crypto_name}..."):
        crypto_data = get_timeframe_data(selected_crypto_id, selected_timeframe)
        
        if isinstance(crypto_data, dict) and "error" in crypto_data:
            st.error(f"Error loading data: {crypto_data.get('error')}")
            return
        
        if not isinstance(crypto_data, pd.DataFrame) or crypto_data.empty:
            st.error("No data available for the selected cryptocurrency and timeframe.")
            return
        
        # Calculate technical indicators
        technical_data = calculate_technical_indicators(crypto_data)
        
        if isinstance(technical_data, dict) and "error" in technical_data:
            st.error(f"Error calculating technical indicators: {technical_data.get('error')}")
            return
    
    # Main content
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Price Analysis", 
        "📉 Performance Metrics", 
        "🧮 Profit/Loss Calculator", 
        "🔮 Price Prediction"
    ])
    
    with tab1:
        st.header("Price Analysis")
        
        # Display price chart with technical indicators
        st.subheader(f"{selected_crypto_name} Price Chart ({selected_timeframe})")
        
        # Create tabs for different technical indicators
        chart_tab1, chart_tab2, chart_tab3, chart_tab4, chart_tab5 = st.tabs([
            "Price & Volume", 
            "RSI", 
            "MACD", 
            "Bollinger Bands",
            "Volume Analysis"
        ])
        
        with chart_tab1:
            # Main price and volume chart
            fig = create_price_chart(technical_data, selected_crypto_name.split(" (")[0], selected_timeframe)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Unable to create price chart.")
        
        with chart_tab2:
            # RSI chart
            fig = create_technical_indicator_chart(technical_data, "rsi", selected_crypto_name.split(" (")[0], selected_timeframe)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Unable to create RSI chart.")
            
            # Add RSI explanation
            st.markdown("""
            **Relative Strength Index (RSI)**
            
            The RSI is a momentum oscillator that measures the speed and change of price movements on a scale from 0 to 100:
            
            - **RSI > 70**: The asset is potentially overbought (overvalued)
            - **RSI < 30**: The asset is potentially oversold (undervalued)
            - **RSI Divergence**: When price makes a new high/low but RSI doesn't, it can signal a potential reversal
            
            RSI can be used to identify potential buy opportunities during oversold conditions and sell opportunities during overbought conditions.
            """)
        
        with chart_tab3:
            # MACD chart
            fig = create_technical_indicator_chart(technical_data, "macd", selected_crypto_name.split(" (")[0], selected_timeframe)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Unable to create MACD chart.")
            
            # Add MACD explanation
            st.markdown("""
            **Moving Average Convergence Divergence (MACD)**
            
            The MACD is a trend-following momentum indicator that shows the relationship between two moving averages:
            
            - **MACD Line**: The difference between the 12-period and 26-period EMAs
            - **Signal Line**: The 9-period EMA of the MACD Line
            - **Histogram**: The difference between the MACD Line and Signal Line
            
            Key signals include:
            - **MACD Line crosses above Signal Line**: Potential buy signal
            - **MACD Line crosses below Signal Line**: Potential sell signal
            - **MACD Line crosses above/below zero**: Trend change indication
            - **Divergence**: When price makes a new high/low but MACD doesn't, it can signal a potential reversal
            """)
        
        with chart_tab4:
            # Bollinger Bands chart
            fig = create_technical_indicator_chart(technical_data, "bollinger", selected_crypto_name.split(" (")[0], selected_timeframe)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Unable to create Bollinger Bands chart.")
            
            # Add Bollinger Bands explanation
            st.markdown("""
            **Bollinger Bands**
            
            Bollinger Bands consist of three lines that help identify volatility and potential overbought/oversold conditions:
            
            - **Middle Band**: A 20-period simple moving average (SMA)
            - **Upper Band**: Middle Band + (2 × standard deviation of price)
            - **Lower Band**: Middle Band - (2 × standard deviation of price)
            
            Key signals include:
            - **Price touching Upper Band**: Potentially overbought
            - **Price touching Lower Band**: Potentially oversold
            - **Bands narrowing**: Low volatility, often precedes a significant price move
            - **Bands widening**: High volatility
            - **Price moving from one band to the middle**: Potential reversal signal
            """)
        
        with chart_tab5:
            # Volume analysis chart
            fig = create_technical_indicator_chart(technical_data, "volume", selected_crypto_name.split(" (")[0], selected_timeframe)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Unable to create volume analysis chart.")
            
            # Add volume analysis explanation
            st.markdown("""
            **Volume Analysis**
            
            Volume represents the total amount of trading activity and can provide important confirmation signals:
            
            - **High volume + price increase**: Strong bullish signal
            - **High volume + price decrease**: Strong bearish signal
            - **Low volume + price change**: Less significant movement, may not be sustainable
            - **Volume increasing over time**: Growing market interest
            - **Volume decreasing over time**: Waning market interest
            
            Volume often precedes price, so a spike in volume can indicate an upcoming price movement, especially if it's accompanied by a breakout from a consolidation pattern.
            """)
        
        # Key price levels section
        st.subheader("Key Price Levels")
        
        # Calculate key price levels
        if isinstance(technical_data, pd.DataFrame) and not technical_data.empty:
            # Get the latest data point
            latest_data = technical_data.iloc[-1]
            current_price = latest_data["price"]
            
            # Calculate support and resistance levels
            min_price = technical_data["price"].min()
            max_price = technical_data["price"].max()
            
            resistance_1 = current_price * 1.05  # 5% above current price
            resistance_2 = current_price * 1.10  # 10% above current price
            resistance_3 = max_price if max_price > current_price else current_price * 1.15  # 15% above or historical max
            
            support_1 = current_price * 0.95  # 5% below current price
            support_2 = current_price * 0.90  # 10% below current price
            support_3 = min_price if min_price < current_price else current_price * 0.85  # 15% below or historical min
            
            # Calculate moving averages if they exist
            sma_7 = latest_data.get("SMA_7", None)
            sma_25 = latest_data.get("SMA_25", None)
            sma_99 = latest_data.get("SMA_99", None)
            
            # Create three columns for price levels display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Resistance Levels**")
                st.metric("Resistance 1", f"${resistance_1:.2f}", f"{((resistance_1 / current_price) - 1) * 100:.2f}%")
                st.metric("Resistance 2", f"${resistance_2:.2f}", f"{((resistance_2 / current_price) - 1) * 100:.2f}%")
                st.metric("Resistance 3", f"${resistance_3:.2f}", f"{((resistance_3 / current_price) - 1) * 100:.2f}%")
            
            with col2:
                st.markdown("**Current Price**")
                st.metric("Price", f"${current_price:.2f}")
                
                if sma_7 is not None and sma_25 is not None:
                    sma_7_pct = ((current_price / sma_7) - 1) * 100
                    sma_25_pct = ((current_price / sma_25) - 1) * 100
                    
                    st.metric("SMA (7)", f"${sma_7:.2f}", f"{sma_7_pct:.2f}%")
                    st.metric("SMA (25)", f"${sma_25:.2f}", f"{sma_25_pct:.2f}%")
                
                if sma_99 is not None:
                    sma_99_pct = ((current_price / sma_99) - 1) * 100
                    st.metric("SMA (99)", f"${sma_99:.2f}", f"{sma_99_pct:.2f}%")
            
            with col3:
                st.markdown("**Support Levels**")
                st.metric("Support 1", f"${support_1:.2f}", f"{((support_1 / current_price) - 1) * 100:.2f}%")
                st.metric("Support 2", f"${support_2:.2f}", f"{((support_2 / current_price) - 1) * 100:.2f}%")
                st.metric("Support 3", f"${support_3:.2f}", f"{((support_3 / current_price) - 1) * 100:.2f}%")
        else:
            st.warning("Unable to calculate price levels. Insufficient data.")
    
    with tab2:
        st.header("Performance Metrics")
        
        # Calculate performance metrics
        with st.spinner("Calculating performance metrics..."):
            performance = calculate_performance_metrics(crypto_data)
            
            if isinstance(performance, dict) and "error" not in performance:
                # Create three columns for performance metrics display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Price Performance")
                    
                    # Determine color for price change
                    price_change_color = "normal"
                    if performance["price_change_pct"] > 0:
                        price_change_color = "normal"
                    elif performance["price_change_pct"] < 0:
                        price_change_color = "inverse"
                    
                    st.metric(
                        "Price Change", 
                        f"${performance['price_change']:.2f}", 
                        f"{performance['price_change_pct']:.2f}%",
                        delta_color=price_change_color
                    )
                    
                    st.metric("Start Price", f"${performance['start_price']:.2f}")
                    st.metric("End Price", f"${performance['end_price']:.2f}")
                    st.metric("Max Price", f"${performance['max_price']:.2f}")
                    st.metric("Min Price", f"${performance['min_price']:.2f}")
                    
                    # Calculate percentage from all-time high
                    current_price = performance['end_price']
                    ath_price = performance['max_price']
                    pct_from_ath = ((current_price - ath_price) / ath_price) * 100 if ath_price > 0 else 0
                    
                    st.metric(
                        "% From Period High", 
                        f"{pct_from_ath:.2f}%",
                        delta_color="inverse" if pct_from_ath < 0 else "normal"
                    )
                
                with col2:
                    st.subheader("Statistical Metrics")
                    
                    st.metric("Mean Price", f"${performance['mean_price']:.2f}")
                    st.metric("Median Price", f"${performance['median_price']:.2f}")
                    st.metric("Standard Deviation", f"${performance['std_dev']:.2f}")
                    
                    # Format statistical metrics
                    st.metric("Coefficient of Variation", f"{performance['cv']:.2f}%")
                    st.metric("Skewness", f"{performance['skewness']:.2f}")
                    st.metric("Kurtosis", f"{performance['kurtosis']:.2f}")
                    
                    # Add explanation for skewness and kurtosis
                    with st.expander("About Statistical Metrics"):
                        st.markdown("""
                        **Coefficient of Variation (CV)** measures the relative variability of price. Higher CV indicates more volatility relative to the mean price.
                        
                        **Skewness** measures the asymmetry of the price distribution:
                        - **Positive skew**: More frequent small losses but occasional extreme gains
                        - **Negative skew**: More frequent small gains but occasional extreme losses
                        - **Zero**: Symmetric distribution
                        
                        **Kurtosis** measures the "tailedness" of the distribution:
                        - **High kurtosis**: More extreme outliers (fat tails)
                        - **Low kurtosis**: Fewer extreme values (thin tails)
                        - **Normal distribution**: Kurtosis = 3
                        """)
                
                with col3:
                    st.subheader("Risk Metrics")
                    
                    st.metric("Volatility", f"{performance['volatility']:.2f}%")
                    st.metric("Price Range", f"${performance['price_range']:.2f}")
                    st.metric("Price Range %", f"{performance['price_range_pct']:.2f}%")
                    
                    # Add risk-adjusted return
                    st.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")
                    
                    # Volume metrics
                    st.metric("Average Volume", f"${performance['avg_volume']:,.0f}")
                    st.metric("Max Volume", f"${performance['max_volume']:,.0f}")
                    
                    # Add explanation for risk metrics
                    with st.expander("About Risk Metrics"):
                        st.markdown("""
                        **Volatility** measures the degree of variation in price over time, expressed as a percentage. Higher volatility indicates larger price swings and potentially higher risk.
                        
                        **Sharpe Ratio** measures risk-adjusted return. Higher values indicate better returns for the level of risk taken.
                        - **Sharpe > 1**: Generally considered good
                        - **Sharpe > 2**: Very good
                        - **Sharpe < 0**: Negative returns relative to risk
                        
                        **Note**: This is a simplified Sharpe calculation without a risk-free rate. It compares the return of the asset to its volatility.
                        """)
                
                # Performance visualization section
                st.subheader("Performance Visualization")
                
                # Create a more detailed performance chart
                if isinstance(crypto_data, pd.DataFrame) and not crypto_data.empty:
                    # Calculate returns
                    returns_df = crypto_data.copy()
                    returns_df["return"] = returns_df["price"].pct_change()
                    returns_df["cumulative_return"] = (1 + returns_df["return"]).cumprod() - 1
                    
                    # Plot cumulative returns
                    fig = go.Figure()
                    
                    # Add cumulative returns line
                    fig.add_trace(
                        go.Scatter(
                            x=returns_df["timestamp"],
                            y=returns_df["cumulative_return"] * 100,
                            mode="lines",
                            name="Cumulative Return",
                            line=dict(
                                color=CHART_COLORS["price_up"],
                                width=2
                            ),
                            fill="tozeroy",
                            fillcolor=f"rgba({int(CHART_COLORS['price_up'][1:3], 16)}, {int(CHART_COLORS['price_up'][3:5], 16)}, {int(CHART_COLORS['price_up'][5:7], 16)}, 0.2)"
                        )
                    )
                    
                    # Add reference line at 0%
                    fig.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color="white",
                        opacity=0.5
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{selected_crypto_name} Cumulative Return ({selected_timeframe})",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return (%)",
                        height=500,
                        template="plotly_dark",
                        hovermode="x unified",
                        paper_bgcolor="#0A192F",
                        plot_bgcolor="#172A46",
                        font=dict(family="Arial, sans-serif", size=12, color="#F8F9FA")
                    )
                    
                    # Update hover template
                    fig.update_traces(
                        hovertemplate="%{x}<br>Return: %{y:.2f}%<extra></extra>"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Daily returns distribution chart
                    # Filter out NaN values
                    daily_returns = returns_df["return"].dropna() * 100
                    
                    if not daily_returns.empty:
                        fig = go.Figure()
                        
                        # Add histogram
                        fig.add_trace(
                            go.Histogram(
                                x=daily_returns,
                                marker_color=CHART_COLORS["volume"],
                                opacity=0.7,
                                nbinsx=30,
                                name="Returns Distribution"
                            )
                        )
                        
                        # Add vertical line at 0
                        fig.add_vline(
                            x=0,
                            line_dash="dash",
                            line_color="white",
                            opacity=0.7
                        )
                        
                        # Add a normal distribution curve for comparison
                        mean = daily_returns.mean()
                        std = daily_returns.std()
                        
                        x = np.linspace(daily_returns.min(), daily_returns.max(), 100)
                        y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(x - mean) ** 2 / (2 * std ** 2))
                        y = y * (len(daily_returns) * (daily_returns.max() - daily_returns.min()) / 30)  # Scale to match histogram
                        
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=y,
                                mode="lines",
                                line=dict(color="white", width=2, dash="dash"),
                                name="Normal Distribution"
                            )
                        )
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{selected_crypto_name} Daily Returns Distribution ({selected_timeframe})",
                            xaxis_title="Daily Return (%)",
                            yaxis_title="Frequency",
                            height=400,
                            template="plotly_dark",
                            paper_bgcolor="#0A192F",
                            plot_bgcolor="#172A46",
                            font=dict(family="Arial, sans-serif", size=12, color="#F8F9FA"),
                            bargap=0.1
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                        **Returns Distribution Analysis**
                        
                        This histogram shows the distribution of daily returns. The dashed white line represents what a normal distribution with the same mean and standard deviation would look like.
                        
                        Comparing the actual distribution to the normal curve can reveal:
                        
                        - **Fat tails**: More extreme returns than expected in a normal distribution
                        - **Skewness**: Asymmetry in the distribution (more positive or negative returns)
                        - **Kurtosis**: The "peakedness" of the distribution
                        
                        Cryptocurrency returns typically show higher kurtosis (fatter tails) than traditional financial assets, indicating more frequent extreme price movements.
                        """)
            else:
                error_msg = performance.get("error", "Unknown error") if isinstance(performance, dict) else "Unknown error"
                st.error(f"Error calculating performance metrics: {error_msg}")
    
    with tab3:
        st.header("Profit/Loss Calculator")
        
        # Get the current price for the default entry price
        current_price = 0
        if isinstance(technical_data, pd.DataFrame) and not technical_data.empty:
            current_price = technical_data["price"].iloc[-1]
        
        # Create calculator interface
        st.subheader("Investment Simulator")
        
        # Create columns for calculator inputs
        col1, col2 = st.columns(2)
        
        with col1:
            # Entry strategy
            position_type = st.radio(
                "Position Type",
                options=["Long (Buy Low, Sell High)", "Short (Sell High, Buy Low)"],
                index=0
            )
            
            is_long = position_type.startswith("Long")
            
            # Investment amount
            investment_amount = st.number_input(
                "Investment Amount (USD)",
                min_value=10.0,
                max_value=1000000.0,
                value=1000.0,
                step=100.0
            )
            
            # Entry price
            entry_price = st.number_input(
                "Entry Price (USD)",
                min_value=0.000001,
                max_value=1000000.0,
                value=current_price,
                format="%.8f" if current_price < 1 else "%.2f"
            )
        
        with col2:
            # Exit price scenarios
            st.markdown("**Exit Price Scenarios**")
            
            # Default exit prices based on entry price
            default_exit_1 = entry_price * 1.1 if is_long else entry_price * 0.9
            default_exit_2 = entry_price * 0.9 if is_long else entry_price * 1.1
            
            # Exit prices
            exit_price_profit = st.number_input(
                "Profit Target (USD)",
                min_value=0.000001,
                max_value=1000000.0,
                value=default_exit_1,
                format="%.8f" if default_exit_1 < 1 else "%.2f"
            )
            
            exit_price_loss = st.number_input(
                "Stop Loss (USD)",
                min_value=0.000001,
                max_value=1000000.0,
                value=default_exit_2,
                format="%.8f" if default_exit_2 < 1 else "%.2f"
            )
            
            custom_exit_price = st.number_input(
                "Custom Exit Price (USD)",
                min_value=0.000001,
                max_value=1000000.0,
                value=current_price,
                format="%.8f" if current_price < 1 else "%.2f"
            )
        
        # Calculate results
        profit_target_result = calculate_profit_loss(entry_price, exit_price_profit, investment_amount, is_long)
        stop_loss_result = calculate_profit_loss(entry_price, exit_price_loss, investment_amount, is_long)
        custom_result = calculate_profit_loss(entry_price, custom_exit_price, investment_amount, is_long)
        
        # Display results
        st.subheader("Investment Results")
        
        # Create columns for results
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.markdown("**Profit Target Scenario**")
            
            if isinstance(profit_target_result, dict) and "error" not in profit_target_result:
                profit_loss = profit_target_result["profit_loss"]
                profit_loss_pct = profit_target_result["profit_loss_pct"]
                
                # Determine color based on profit/loss
                color = "normal" if profit_loss > 0 else "inverse" if profit_loss < 0 else "normal"
                
                st.metric(
                    "Profit/Loss", 
                    f"${profit_loss:.2f}", 
                    f"{profit_loss_pct:.2f}%",
                    delta_color=color
                )
                
                st.metric("Exit Price", f"${exit_price_profit:.8f}" if exit_price_profit < 1 else f"${exit_price_profit:.2f}")
                st.metric("Quantity", f"{profit_target_result['quantity']:.8f}" if profit_target_result['quantity'] < 1 else f"{profit_target_result['quantity']:.4f}")
            else:
                error_msg = profit_target_result.get("error", "Invalid input") if isinstance(profit_target_result, dict) else "Calculation error"
                st.error(error_msg)
        
        with res_col2:
            st.markdown("**Stop Loss Scenario**")
            
            if isinstance(stop_loss_result, dict) and "error" not in stop_loss_result:
                profit_loss = stop_loss_result["profit_loss"]
                profit_loss_pct = stop_loss_result["profit_loss_pct"]
                
                # Determine color based on profit/loss
                color = "normal" if profit_loss > 0 else "inverse" if profit_loss < 0 else "normal"
                
                st.metric(
                    "Profit/Loss", 
                    f"${profit_loss:.2f}", 
                    f"{profit_loss_pct:.2f}%",
                    delta_color=color
                )
                
                st.metric("Exit Price", f"${exit_price_loss:.8f}" if exit_price_loss < 1 else f"${exit_price_loss:.2f}")
                st.metric("Quantity", f"{stop_loss_result['quantity']:.8f}" if stop_loss_result['quantity'] < 1 else f"{stop_loss_result['quantity']:.4f}")
            else:
                error_msg = stop_loss_result.get("error", "Invalid input") if isinstance(stop_loss_result, dict) else "Calculation error"
                st.error(error_msg)
        
        with res_col3:
            st.markdown("**Custom Exit Scenario**")
            
            if isinstance(custom_result, dict) and "error" not in custom_result:
                profit_loss = custom_result["profit_loss"]
                profit_loss_pct = custom_result["profit_loss_pct"]
                
                # Determine color based on profit/loss
                color = "normal" if profit_loss > 0 else "inverse" if profit_loss < 0 else "normal"
                
                st.metric(
                    "Profit/Loss", 
                    f"${profit_loss:.2f}", 
                    f"{profit_loss_pct:.2f}%",
                    delta_color=color
                )
                
                st.metric("Exit Price", f"${custom_exit_price:.8f}" if custom_exit_price < 1 else f"${custom_exit_price:.2f}")
                st.metric("Quantity", f"{custom_result['quantity']:.8f}" if custom_result['quantity'] < 1 else f"{custom_result['quantity']:.4f}")
            else:
                error_msg = custom_result.get("error", "Invalid input") if isinstance(custom_result, dict) else "Calculation error"
                st.error(error_msg)
        
        # Risk-Reward Analysis
        if (isinstance(profit_target_result, dict) and "error" not in profit_target_result and
            isinstance(stop_loss_result, dict) and "error" not in stop_loss_result):
            
            st.subheader("Risk-Reward Analysis")
            
            profit = abs(profit_target_result["profit_loss"])
            loss = abs(stop_loss_result["profit_loss"])
            
            if loss > 0:
                risk_reward_ratio = profit / loss
                
                # Determine if the risk-reward ratio is favorable
                ratio_assessment = "Favorable" if risk_reward_ratio >= 2 else "Unfavorable"
                ratio_color = "normal" if risk_reward_ratio >= 2 else "inverse"
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Risk-Reward Ratio", 
                        f"{risk_reward_ratio:.2f}",
                        ratio_assessment,
                        delta_color=ratio_color
                    )
                
                with col2:
                    # Calculate break-even probability
                    break_even_prob = loss / (profit + loss) * 100
                    
                    st.metric(
                        "Break-Even Win Rate", 
                        f"{break_even_prob:.2f}%",
                        f"You need to win {break_even_prob:.0f}% of similar trades to break even"
                    )
                
                # Add explanation
                st.markdown(f"""
                **Risk-Reward Analysis Explanation**
                
                Your risk-reward ratio is **{risk_reward_ratio:.2f}**. This means for every $1 you risk, you stand to gain ${risk_reward_ratio:.2f}.
                
                A ratio of 2.0 or higher is generally considered favorable for most trading strategies. With your current setup:
                
                - Potential gain: **${profit:.2f}** ({profit_target_result["profit_loss_pct"]:.2f}%)
                - Potential loss: **${loss:.2f}** ({abs(stop_loss_result["profit_loss_pct"]):.2f}%)
                - To break even, you need to win at least **{break_even_prob:.0f}%** of similar trades
                
                {
                    "This is a favorable risk-reward setup." if risk_reward_ratio >= 2 else 
                    "This risk-reward ratio is unfavorable. Consider adjusting your profit target or stop loss."
                }
                """)
            else:
                st.warning("Cannot calculate risk-reward ratio because one of the scenarios doesn't have a loss.")
        
        # Add historical performance for context
        with st.expander("Historical Performance Context"):
            if isinstance(technical_data, pd.DataFrame) and not technical_data.empty:
                # Calculate win rate based on historical data
                price_changes = technical_data["price"].pct_change().dropna()
                
                if len(price_changes) > 0:
                    # For long positions
                    long_win_rate = (price_changes > 0).mean() * 100
                    
                    # For short positions
                    short_win_rate = (price_changes < 0).mean() * 100
                    
                    # Average daily change
                    avg_daily_change = price_changes.mean() * 100
                    
                    # Average gain on winning days
                    avg_gain = price_changes[price_changes > 0].mean() * 100 if any(price_changes > 0) else 0
                    
                    # Average loss on losing days
                    avg_loss = price_changes[price_changes < 0].mean() * 100 if any(price_changes < 0) else 0
                    
                    st.markdown(f"""
                    **Historical Performance Metrics**
                    
                    Based on the selected timeframe ({selected_timeframe}), {selected_crypto_name} has the following historical metrics:
                    
                    - **Long Position Win Rate**: {long_win_rate:.2f}% (percentage of periods with price increases)
                    - **Short Position Win Rate**: {short_win_rate:.2f}% (percentage of periods with price decreases)
                    - **Average Period Change**: {avg_daily_change:.2f}%
                    - **Average Gain (Winning Periods)**: {avg_gain:.2f}%
                    - **Average Loss (Losing Periods)**: {avg_loss:.2f}%
                    
                    These metrics can provide context for evaluating your trade setup, but remember that past performance doesn't guarantee future results.
                    """)
                else:
                    st.warning("Insufficient historical data to calculate performance metrics.")
            else:
                st.warning("Historical data not available for context.")
    
    with tab4:
        st.header("Price Prediction")
        
        # Information about price prediction
        st.info("""
        This feature uses an ARIMA model to make short-term price predictions. 
        
        **Important Disclaimer**: This is a simplified prediction based on historical patterns and should not be used as the sole basis for investment decisions. Cryptocurrency prices are highly volatile and influenced by many external factors that this model cannot account for.
        """)
        
        # Check if we have enough data for prediction
        if isinstance(technical_data, pd.DataFrame) and len(technical_data) >= 60:
            # Button to generate prediction
            if st.button("Generate Price Prediction (1-5 minutes)"):
                with st.spinner("Calculating price prediction..."):
                    prediction = predict_price(selected_crypto_id, timeframe="5m")
                    
                    if isinstance(prediction, dict) and "error" not in prediction:
                        # Display prediction results
                        st.subheader("Price Prediction Results")
                        
                        # Current price
                        current_price = prediction["current_price"]
                        
                        # Create a table for predictions
                        prediction_data = []
                        
                        for i, price in enumerate(prediction["predicted_prices"]):
                            prediction_data.append({
                                "Timeframe": f"{i+1} minute{'s' if i > 0 else ''}",
                                "Predicted Price": price,
                                "Change %": prediction["price_change_pct"][i]
                            })
                        
                        prediction_df = pd.DataFrame(prediction_data)
                        
                        # Create columns for display
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # Display current price and accuracy
                            st.metric(
                                "Current Price", 
                                f"${current_price:.8f}" if current_price < 1 else f"${current_price:.2f}"
                            )
                            
                            st.metric(
                                "Prediction Accuracy", 
                                f"{prediction['accuracy']:.2f}%"
                            )
                            
                            # Generate timestamp
                            st.text(f"Generated at: {prediction['timestamp']}")
                        
                        with col2:
                            # Format prediction table
                            styled_prediction_df = prediction_df.copy()
                            
                            # Format price and change columns
                            styled_prediction_df["Predicted Price"] = styled_prediction_df["Predicted Price"].apply(
                                lambda x: f"${x:.8f}" if x < 1 else f"${x:.2f}"
                            )
                            
                            styled_prediction_df["Change %"] = styled_prediction_df["Change %"].apply(
                                lambda x: f"{x:.2f}%"
                            )
                            
                            # Display the table
                            st.dataframe(styled_prediction_df, use_container_width=True)
                        
                        # Create prediction chart
                        fig = go.Figure()
                        
                        # Add prediction line
                        minutes = list(range(len(prediction["predicted_prices"])))
                        times = [f"{i+1}m" for i in minutes]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=times,
                                y=prediction["predicted_prices"],
                                mode="lines+markers",
                                name="Predicted Price",
                                line=dict(color=CHART_COLORS["price_up"], width=2),
                                marker=dict(size=8)
                            )
                        )
                        
                        # Add confidence intervals
                        upper_bound = [p + c for p, c in zip(prediction["predicted_prices"], prediction["confidence_interval"])]
                        lower_bound = [p - c for p, c in zip(prediction["predicted_prices"], prediction["confidence_interval"])]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=times,
                                y=upper_bound,
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False
                            )
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=times,
                                y=lower_bound,
                                mode="lines",
                                line=dict(width=0),
                                fill="tonexty",
                                fillcolor="rgba(0, 200, 83, 0.2)",
                                name="Confidence Interval"
                            )
                        )
                        
                        # Add current price reference line
                        fig.add_hline(
                            y=current_price,
                            line_dash="dash",
                            line_color="white",
                            opacity=0.7,
                            annotation_text="Current Price",
                            annotation_position="bottom right"
                        )
                        
                        # Update layout
                        fig.update_layout(
                            title=f"Price Prediction for {selected_crypto_name} (Next 5 Minutes)",
                            xaxis_title="Time Horizon",
                            yaxis_title="Predicted Price (USD)",
                            height=400,
                            template="plotly_dark",
                            hovermode="x unified",
                            paper_bgcolor="#0A192F",
                            plot_bgcolor="#172A46",
                            font=dict(family="Arial, sans-serif", size=12, color="#F8F9FA")
                        )
                        
                        # Update hover template
                        fig.update_traces(
                            hovertemplate="%{x}<br>Price: $%{y:.8f}<extra></extra>" if current_price < 1 else
                                         "%{x}<br>Price: $%{y:.2f}<extra></extra>"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add explanation of the prediction
                        st.markdown("""
                        **Understanding the Prediction**
                        
                        The chart above shows the predicted price for the next 5 minutes, with the shaded area representing the confidence interval. This prediction is based on recent price patterns and uses an ARIMA (AutoRegressive Integrated Moving Average) model.
                        
                        **Key points to understand**:
                        
                        1. **Accuracy**: The stated accuracy is based on the model's confidence in its prediction, not a guarantee of future performance.
                        
                        2. **Confidence Interval**: The shaded area shows the range where the price is likely to fall with 95% confidence. Wider intervals indicate less certainty.
                        
                        3. **Short-Term Only**: This model is designed for very short-term predictions and becomes increasingly unreliable beyond a few minutes.
                        
                        4. **Technical Limitations**: The model only uses price history and does not account for news, market sentiment, or other external factors.
                        
                        **Always combine this prediction with other forms of analysis before making any trading decisions.**
                        """)
                    else:
                        error_msg = prediction.get("error", "Unknown error") if isinstance(prediction, dict) else "Unknown error"
                        st.error(f"Error generating price prediction: {error_msg}")
            else:
                st.markdown("Click the button above to generate a short-term price prediction.")
        else:
            st.warning("Insufficient data for price prediction. This feature requires at least 60 data points with the selected timeframe.")

if __name__ == "__main__":
    main()
