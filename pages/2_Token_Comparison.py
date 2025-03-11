import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.api import (
    get_top_cryptocurrencies,
    get_coin_history,
    get_coin_comparison_data
)
from utils.visualization import (
    create_comparison_chart
)
from utils.analysis import (
    calculate_comparison_statistics
)
from utils.styles import apply_styles
from utils.constants import CHART_COLORS, DISPLAY_TIMEFRAMES, DEFAULT_COMPARISON_CRYPTOS

# Set page config
st.set_page_config(
    page_title="Token Comparison | CryptoAnalytics Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom styles
apply_styles()

def main():
    # Header
    st.title("🔍 Token Comparison")
    st.markdown(
        """
        Compare cryptocurrencies side by side with detailed metrics and visualizations 
        to make informed investment decisions.
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
    st.sidebar.header("Token Comparison Options")
    
    # Timeframe selection
    selected_timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        options=DISPLAY_TIMEFRAMES,
        index=DISPLAY_TIMEFRAMES.index("7d")
    )
    
    # Cryptocurrency selection
    st.sidebar.subheader("Select Cryptocurrencies to Compare")
    
    # Get default selection indices
    default_indices = []
    for default_crypto in DEFAULT_COMPARISON_CRYPTOS:
        for i, (crypto_id, _) in enumerate(crypto_options):
            if crypto_id == default_crypto:
                default_indices.append(i)
                break
    
    # Multi-select for cryptocurrencies
    selected_crypto_tuples = st.sidebar.multiselect(
        "Choose up to 5 cryptocurrencies",
        options=crypto_options,
        default=[crypto_options[i] for i in default_indices if i < len(crypto_options)],
        format_func=lambda x: x[1],
        max_selections=5
    )
    
    selected_crypto_ids = [crypto[0] for crypto in selected_crypto_tuples]
    selected_crypto_names = [crypto[1] for crypto in selected_crypto_tuples]
    
    if not selected_crypto_ids:
        st.warning("Please select at least one cryptocurrency to continue.")
        return
    
    # Main content
    # Tabs for different comparison views
    tab1, tab2, tab3 = st.tabs([
        "📊 Price Comparison", 
        "📈 Statistical Comparison", 
        "🧮 Metrics Comparison"
    ])
    
    with tab1:
        st.header("Price Comparison")
        
        # Controls for chart type
        chart_metric = st.radio(
            "Select comparison metric",
            options=["Price", "Market Cap", "Volume"],
            horizontal=True,
            key="price_comparison_metric"
        )
        
        # Load and display comparison chart
        with st.spinner("Loading comparison data..."):
            comparison_data = {}
            
            for crypto_id in selected_crypto_ids:
                # Get historical data for this crypto
                hist_data = get_coin_history(crypto_id, days=selected_timeframe)
                
                if isinstance(hist_data, pd.DataFrame) and not hist_data.empty:
                    # Extract the crypto name from the selected options
                    crypto_name = next((name for id, name in crypto_options if id == crypto_id), crypto_id)
                    
                    # Prepare data for the comparison chart
                    price_data = []
                    market_cap_data = []
                    volume_data = []
                    
                    for _, row in hist_data.iterrows():
                        price_data.append({
                            "timestamp": row["timestamp"],
                            "price": row["price"]
                        })
                        
                        market_cap_data.append({
                            "timestamp": row["timestamp"],
                            "market_cap": row["market_cap"]
                        })
                        
                        volume_data.append({
                            "timestamp": row["timestamp"],
                            "volume": row["volume"]
                        })
                    
                    comparison_data[crypto_id] = {
                        "name": crypto_name.split(" (")[0],
                        "price_data": price_data,
                        "market_cap_data": market_cap_data,
                        "volume_data": volume_data
                    }
                else:
                    st.warning(f"Unable to load data for {crypto_id}. This cryptocurrency will be excluded from the comparison.")
            
            if comparison_data:
                # Convert the selected metric to lowercase for the function
                metric = chart_metric.lower()
                
                # Create and display the comparison chart
                fig = create_comparison_chart(comparison_data, metric=metric, timeframe=selected_timeframe)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Unable to create comparison chart. Please try different cryptocurrencies or timeframe.")
            else:
                st.error("No data available for comparison. Please select different cryptocurrencies.")
        
        # Description of the chart
        st.markdown("""
        **About This Chart:**
        
        This chart shows the percentage change in price, market cap, or volume for the selected cryptocurrencies over the chosen timeframe.
        
        All values are normalized to start from the same point (0%) to allow for easier comparison of relative performance regardless of absolute price differences.
        
        This visualization helps identify which cryptocurrencies outperformed or underperformed relative to each other during the selected period.
        """)
        
        # Current price comparison table
        st.subheader("Current Price Comparison")
        
        with st.spinner("Loading current price data..."):
            # Filter the top_cryptos_df to only include the selected cryptocurrencies
            if not top_cryptos_df.empty:
                current_data = top_cryptos_df[top_cryptos_df['id'].isin(selected_crypto_ids)].copy()
                
                if not current_data.empty:
                    # Select and rename columns for display
                    display_df = current_data[[
                        'name', 'symbol', 'current_price', 'price_change_percentage_24h', 
                        'market_cap', 'total_volume', 'circulating_supply'
                    ]]
                    
                    display_df = display_df.rename(columns={
                        'name': 'Name',
                        'symbol': 'Symbol',
                        'current_price': 'Price (USD)',
                        'price_change_percentage_24h': '24h Change %',
                        'market_cap': 'Market Cap',
                        'total_volume': '24h Volume',
                        'circulating_supply': 'Circulating Supply'
                    })
                    
                    # Format values
                    display_df['Symbol'] = display_df['Symbol'].str.upper()
                    display_df['Price (USD)'] = display_df['Price (USD)'].apply(lambda x: f"${x:,.8f}" if x < 1 else f"${x:,.2f}")
                    display_df['24h Change %'] = display_df['24h Change %'].apply(lambda x: f"{x:.2f}%")
                    display_df['Market Cap'] = display_df['Market Cap'].apply(lambda x: f"${x:,.0f}")
                    display_df['24h Volume'] = display_df['24h Volume'].apply(lambda x: f"${x:,.0f}")
                    display_df['Circulating Supply'] = display_df['Circulating Supply'].apply(lambda x: f"{x:,.0f}")
                    
                    # Apply styling based on price change
                    def highlight_change(val):
                        if 'N/A' in val:
                            return ''
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
                else:
                    st.warning("No current data available for the selected cryptocurrencies.")
            else:
                st.warning("No cryptocurrency data available.")
    
    with tab2:
        st.header("Statistical Comparison")
        
        with st.spinner("Calculating statistical metrics..."):
            # Calculate detailed statistics for the selected cryptocurrencies
            stats = calculate_comparison_statistics(selected_crypto_ids)
            
            if isinstance(stats, dict) and "error" not in stats:
                # Create a dataframe from the statistics for easier comparison
                stats_data = []
                
                for crypto_id, crypto_stats in stats.items():
                    if "error" not in crypto_stats:
                        stats_data.append({
                            "Cryptocurrency": crypto_stats.get("name", crypto_id),
                            "Symbol": crypto_stats.get("symbol", "").upper(),
                            "Market Cap": crypto_stats.get("market_cap", 0),
                            "Current Price": crypto_stats.get("current_price", 0),
                            "Mean Price": crypto_stats.get("mean", 0),
                            "Median Price": crypto_stats.get("median", 0),
                            "Standard Deviation": crypto_stats.get("std_dev", 0),
                            "CV (%)": crypto_stats.get("cv", 0),
                            "Skewness": crypto_stats.get("skewness", 0),
                            "Kurtosis": crypto_stats.get("kurtosis", 0),
                            "Volatility (%)": crypto_stats.get("volatility", 0),
                            "Max Drawdown (%)": crypto_stats.get("max_drawdown", 0),
                        })
                    else:
                        st.warning(f"Unable to calculate statistics for {crypto_id}: {crypto_stats.get('error')}")
                
                if stats_data:
                    # Create dataframe
                    stats_df = pd.DataFrame(stats_data)
                    
                    # Format columns for display
                    stats_df["Market Cap"] = stats_df["Market Cap"].apply(lambda x: f"${x:,.0f}")
                    stats_df["Current Price"] = stats_df["Current Price"].apply(lambda x: f"${x:,.8f}" if x < 1 else f"${x:,.2f}")
                    stats_df["Mean Price"] = stats_df["Mean Price"].apply(lambda x: f"${x:,.8f}" if x < 1 else f"${x:,.2f}")
                    stats_df["Median Price"] = stats_df["Median Price"].apply(lambda x: f"${x:,.8f}" if x < 1 else f"${x:,.2f}")
                    stats_df["Standard Deviation"] = stats_df["Standard Deviation"].apply(lambda x: f"${x:,.8f}" if x < 1 else f"${x:,.2f}")
                    stats_df["CV (%)"] = stats_df["CV (%)"].apply(lambda x: f"{x:.2f}%")
                    stats_df["Skewness"] = stats_df["Skewness"].apply(lambda x: f"{x:.2f}")
                    stats_df["Kurtosis"] = stats_df["Kurtosis"].apply(lambda x: f"{x:.2f}")
                    stats_df["Volatility (%)"] = stats_df["Volatility (%)"].apply(lambda x: f"{x:.2f}%")
                    stats_df["Max Drawdown (%)"] = stats_df["Max Drawdown (%)"].apply(lambda x: f"{x:.2f}%")
                    
                    # Display the statistics table
                    st.dataframe(stats_df, use_container_width=True)
                else:
                    st.warning("No statistical data available for the selected cryptocurrencies.")
                
                # Create visualizations for the statistical data
                st.subheader("Statistical Visualizations")
                
                # Create tabs for different visualizations
                viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                    "Volatility Comparison", 
                    "Risk vs. Return", 
                    "Distribution Metrics"
                ])
                
                with viz_tab1:
                    # Create volatility bar chart
                    volatility_data = []
                    
                    for crypto_id, crypto_stats in stats.items():
                        if "error" not in crypto_stats:
                            volatility_data.append({
                                "Cryptocurrency": crypto_stats.get("name", crypto_id),
                                "Symbol": crypto_stats.get("symbol", "").upper(),
                                "Volatility (%)": crypto_stats.get("volatility", 0)
                            })
                    
                    if volatility_data:
                        volatility_df = pd.DataFrame(volatility_data)
                        
                        # Sort by volatility
                        volatility_df = volatility_df.sort_values("Volatility (%)", ascending=False)
                        
                        # Create bar chart
                        fig = px.bar(
                            volatility_df,
                            x="Cryptocurrency",
                            y="Volatility (%)",
                            title="Volatility Comparison (30-day)",
                            color="Volatility (%)",
                            color_continuous_scale=px.colors.sequential.Bluered,
                            labels={"Volatility (%)": "30-day Volatility (%)"},
                            text="Symbol"
                        )
                        
                        # Update layout
                        fig.update_layout(
                            height=500,
                            template="plotly_dark",
                            paper_bgcolor="#0A192F",
                            plot_bgcolor="#172A46",
                            font=dict(family="Arial, sans-serif", size=12, color="#F8F9FA"),
                            yaxis_title="Volatility (%)",
                            xaxis_title="",
                            coloraxis_showscale=False
                        )
                        
                        # Update text position
                        fig.update_traces(
                            textposition='inside',
                            textfont=dict(color="white", size=12)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No volatility data available for visualization.")
                    
                    st.markdown("""
                    **Understanding Volatility:**
                    
                    Volatility measures the degree of variation in a cryptocurrency's price over time. Higher volatility indicates larger price swings, which can present both higher risk and potential for returns.
                    
                    This chart shows the 30-day annualized volatility for each selected cryptocurrency, calculated as the standard deviation of daily returns.
                    
                    **Key insights:**
                    - Higher bars indicate more volatile assets
                    - More volatile cryptocurrencies typically carry higher risk
                    - Lower volatility may indicate more stable price action, but possibly lower potential returns
                    """)
                
                with viz_tab2:
                    # Create risk vs. return scatter plot
                    risk_return_data = []
                    
                    for crypto_id, crypto_stats in stats.items():
                        if "error" not in crypto_stats:
                            # Calculate return as price change percentage
                            current_price = crypto_stats.get("current_price", 0)
                            mean_price = crypto_stats.get("mean", 0)
                            
                            if mean_price > 0:
                                returns = ((current_price - mean_price) / mean_price) * 100
                            else:
                                returns = 0
                            
                            risk_return_data.append({
                                "Cryptocurrency": crypto_stats.get("name", crypto_id),
                                "Symbol": crypto_stats.get("symbol", "").upper(),
                                "Return (%)": returns,
                                "Volatility (%)": crypto_stats.get("volatility", 0),
                                "Market Cap": crypto_stats.get("market_cap", 0)
                            })
                    
                    if risk_return_data:
                        risk_return_df = pd.DataFrame(risk_return_data)
                        
                        # Create scatter plot
                        fig = px.scatter(
                            risk_return_df,
                            x="Volatility (%)",
                            y="Return (%)",
                            size="Market Cap",
                            color="Cryptocurrency",
                            hover_name="Cryptocurrency",
                            text="Symbol",
                            title="Risk vs. Return Analysis",
                            labels={
                                "Volatility (%)": "Risk (Volatility %)",
                                "Return (%)": "30-day Return (%)"
                            },
                            size_max=50
                        )
                        
                        # Add quadrant lines
                        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                        fig.add_vline(x=risk_return_df["Volatility (%)"].median(), 
                                    line_dash="dash", 
                                    line_color="white", 
                                    opacity=0.5)
                        
                        # Add annotations for quadrants
                        fig.add_annotation(
                            x=risk_return_df["Volatility (%)"].min() * 1.1,
                            y=risk_return_df["Return (%)"].max() * 0.9,
                            text="Low Risk, High Return",
                            showarrow=False,
                            font=dict(color="green")
                        )
                        
                        fig.add_annotation(
                            x=risk_return_df["Volatility (%)"].max() * 0.9,
                            y=risk_return_df["Return (%)"].max() * 0.9,
                            text="High Risk, High Return",
                            showarrow=False,
                            font=dict(color="orange")
                        )
                        
                        fig.add_annotation(
                            x=risk_return_df["Volatility (%)"].min() * 1.1,
                            y=risk_return_df["Return (%)"].min() * 0.9,
                            text="Low Risk, Low Return",
                            showarrow=False,
                            font=dict(color="yellow")
                        )
                        
                        fig.add_annotation(
                            x=risk_return_df["Volatility (%)"].max() * 0.9,
                            y=risk_return_df["Return (%)"].min() * 0.9,
                            text="High Risk, Low Return",
                            showarrow=False,
                            font=dict(color="red")
                        )
                        
                        # Update layout
                        fig.update_layout(
                            height=600,
                            template="plotly_dark",
                            paper_bgcolor="#0A192F",
                            plot_bgcolor="#172A46",
                            font=dict(family="Arial, sans-serif", size=12, color="#F8F9FA"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        # Update text position
                        fig.update_traces(
                            textposition='top center',
                            textfont=dict(color="white", size=10)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No risk-return data available for visualization.")
                    
                    st.markdown("""
                    **Understanding Risk vs. Return:**
                    
                    This scatter plot positions each cryptocurrency according to its risk (volatility) and return, with bubble size representing market capitalization.
                    
                    **Key insights:**
                    - **Top-left quadrant**: Low risk, high return (ideal investments)
                    - **Top-right quadrant**: High risk, high return (speculative opportunities)
                    - **Bottom-left quadrant**: Low risk, low return (stable but less profitable)
                    - **Bottom-right quadrant**: High risk, low return (investments to potentially avoid)
                    
                    The chart helps identify which cryptocurrencies offer the best risk-adjusted returns in the selected timeframe.
                    """)
                
                with viz_tab3:
                    # Create distribution metrics comparison
                    dist_data = []
                    
                    for crypto_id, crypto_stats in stats.items():
                        if "error" not in crypto_stats:
                            dist_data.append({
                                "Cryptocurrency": crypto_stats.get("name", crypto_id),
                                "Symbol": crypto_stats.get("symbol", "").upper(),
                                "CV (%)": crypto_stats.get("cv", 0),
                                "Skewness": crypto_stats.get("skewness", 0),
                                "Kurtosis": crypto_stats.get("kurtosis", 0)
                            })
                    
                    if dist_data:
                        dist_df = pd.DataFrame(dist_data)
                        
                        # Create subplots
                        fig = make_subplots(
                            rows=1, 
                            cols=3,
                            subplot_titles=("Coefficient of Variation", "Skewness", "Kurtosis")
                        )
                        
                        # Add bars for each metric
                        for i, crypto in enumerate(dist_df["Cryptocurrency"]):
                            # Coefficient of Variation
                            fig.add_trace(
                                go.Bar(
                                    x=[crypto],
                                    y=[dist_df.iloc[i]["CV (%)"]],
                                    name=crypto,
                                    text=dist_df.iloc[i]["Symbol"],
                                    showlegend=i == 0
                                ),
                                row=1, col=1
                            )
                            
                            # Skewness
                            fig.add_trace(
                                go.Bar(
                                    x=[crypto],
                                    y=[dist_df.iloc[i]["Skewness"]],
                                    name=crypto,
                                    text=dist_df.iloc[i]["Symbol"],
                                    showlegend=False
                                ),
                                row=1, col=2
                            )
                            
                            # Kurtosis
                            fig.add_trace(
                                go.Bar(
                                    x=[crypto],
                                    y=[dist_df.iloc[i]["Kurtosis"]],
                                    name=crypto,
                                    text=dist_df.iloc[i]["Symbol"],
                                    showlegend=False
                                ),
                                row=1, col=3
                            )
                        
                        # Update layout
                        fig.update_layout(
                            height=500,
                            title_text="Distribution Metrics Comparison",
                            template="plotly_dark",
                            paper_bgcolor="#0A192F",
                            plot_bgcolor="#172A46",
                            font=dict(family="Arial, sans-serif", size=12, color="#F8F9FA"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        # Update y-axis titles
                        fig.update_yaxes(title_text="Coefficient of Variation (%)", row=1, col=1)
                        fig.update_yaxes(title_text="Skewness Value", row=1, col=2)
                        fig.update_yaxes(title_text="Kurtosis Value", row=1, col=3)
                        
                        # Update text position
                        fig.update_traces(
                            textposition='inside',
                            textfont=dict(color="white", size=10)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No distribution metrics available for visualization.")
                    
                    st.markdown("""
                    **Understanding Distribution Metrics:**
                    
                    These charts compare statistical properties of price distributions for the selected cryptocurrencies:
                    
                    1. **Coefficient of Variation (CV)**: Measures relative volatility - higher CV means more dispersion relative to the mean.
                    
                    2. **Skewness**: Indicates asymmetry in the price distribution:
                       - **Positive skew**: More frequent small losses but occasional extreme gains
                       - **Negative skew**: More frequent small gains but occasional extreme losses
                       - **Zero**: Symmetric distribution
                    
                    3. **Kurtosis**: Measures the "tailedness" of the distribution:
                       - **High kurtosis**: More extreme outliers (fat tails)
                       - **Low kurtosis**: Fewer extreme values (thin tails)
                       - **Normal distribution**: Kurtosis = 3
                    
                    These metrics help understand the pattern of price movements and potential for extreme events.
                    """)
            else:
                error_msg = stats.get("error", "Unknown error") if isinstance(stats, dict) else "Unknown error"
                st.error(f"Error calculating statistical metrics: {error_msg}")
    
    with tab3:
        st.header("Metrics Comparison")
        
        # Fetch detailed metrics for comparison
        with st.spinner("Loading detailed metrics..."):
            comparison_data = get_coin_comparison_data(selected_crypto_ids)
            
            if isinstance(comparison_data, dict) and "error" not in comparison_data:
                # Convert the comparison data to a more usable format
                metrics_data = []
                
                for crypto_id, data in comparison_data.items():
                    if "error" not in data:
                        metrics_data.append({
                            "Cryptocurrency": data.get("name", crypto_id),
                            "Symbol": data.get("symbol", "").upper(),
                            "Price (USD)": data.get("price", 0),
                            "Market Cap (USD)": data.get("market_cap", 0),
                            "24h Change (%)": data.get("24h_change", 0),
                            "7d Change (%)": data.get("7d_change", 0),
                            "30d Change (%)": data.get("30d_change", 0),
                            "24h Volume (USD)": data.get("volume", 0),
                            "Circulating Supply": data.get("circulating_supply", 0),
                            "Total Supply": data.get("total_supply", 0),
                            "Max Supply": data.get("max_supply", 0),
                            "All-Time High (USD)": data.get("all_time_high", 0),
                            "ATH Date": data.get("all_time_high_date", "")
                        })
                    else:
                        st.warning(f"Unable to fetch detailed metrics for {crypto_id}: {data.get('error')}")
                
                if metrics_data:
                    # Create dataframe
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    # Format columns for display
                    metrics_df["Price (USD)"] = metrics_df["Price (USD)"].apply(lambda x: f"${x:,.8f}" if x < 1 else f"${x:,.2f}")
                    metrics_df["Market Cap (USD)"] = metrics_df["Market Cap (USD)"].apply(lambda x: f"${x:,.0f}")
                    metrics_df["24h Change (%)"] = metrics_df["24h Change (%)"].apply(lambda x: f"{x:.2f}%")
                    metrics_df["7d Change (%)"] = metrics_df["7d Change (%)"].apply(lambda x: f"{x:.2f}%")
                    metrics_df["30d Change (%)"] = metrics_df["30d Change (%)"].apply(lambda x: f"{x:.2f}%")
                    metrics_df["24h Volume (USD)"] = metrics_df["24h Volume (USD)"].apply(lambda x: f"${x:,.0f}")
                    metrics_df["Circulating Supply"] = metrics_df["Circulating Supply"].apply(lambda x: f"{x:,.0f}")
                    metrics_df["Total Supply"] = metrics_df["Total Supply"].apply(lambda x: f"{x:,.0f}" if x else "∞")
                    metrics_df["Max Supply"] = metrics_df["Max Supply"].apply(lambda x: f"{x:,.0f}" if x else "∞")
                    metrics_df["All-Time High (USD)"] = metrics_df["All-Time High (USD)"].apply(lambda x: f"${x:,.8f}" if x < 1 else f"${x:,.2f}")
                    
                    # Format date
                    metrics_df["ATH Date"] = metrics_df["ATH Date"].apply(lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")).strftime("%Y-%m-%d") if x else "N/A")
                    
                    # Apply styling based on price changes
                    def highlight_change(val):
                        if 'N/A' in val:
                            return ''
                        val_num = float(val.replace('%', ''))
                        if val_num > 0:
                            return 'color: #00C853; font-weight: bold'
                        elif val_num < 0:
                            return 'color: #FF3D00; font-weight: bold'
                        else:
                            return ''
                    
                    # Apply the styling
                    styled_df = metrics_df.style.map(highlight_change, subset=['24h Change (%)', '7d Change (%)', '30d Change (%)'])
                    
                    # Display the table
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Create visualization tabs
                    viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
                        "Price Changes", 
                        "Market Cap", 
                        "Volume", 
                        "Supply Metrics",
                        "ATH Analysis"
                    ])
                    
                    with viz_tab1:
                        # Create price changes chart
                        # Convert back to numeric values for charting
                        chart_data = pd.DataFrame({
                            "Cryptocurrency": [d["Cryptocurrency"] for d in metrics_data],
                            "24h": [float(d["24h Change (%)"].replace("%", "")) for d in metrics_data],
                            "7d": [float(d["7d Change (%)"].replace("%", "")) for d in metrics_data],
                            "30d": [float(d["30d Change (%)"].replace("%", "")) for d in metrics_data]
                        })
                        
                        # Reshape data for plotting
                        plot_data = []
                        for i, row in enumerate(metrics_data):
                            # Extract values from formatted strings
                            h24_change = float(row["24h Change (%)"].replace("%", "")) if isinstance(row["24h Change (%)"], str) else row["24h Change (%)"]
                            d7_change = float(row["7d Change (%)"].replace("%", "")) if isinstance(row["7d Change (%)"], str) else row["7d Change (%)"]
                            d30_change = float(row["30d Change (%)"].replace("%", "")) if isinstance(row["30d Change (%)"], str) else row["30d Change (%)"]
                            
                            plot_data.extend([
                                {"Cryptocurrency": row["Cryptocurrency"], "Period": "24h", "Change": h24_change, "Symbol": row["Symbol"]},
                                {"Cryptocurrency": row["Cryptocurrency"], "Period": "7d", "Change": d7_change, "Symbol": row["Symbol"]},
                                {"Cryptocurrency": row["Cryptocurrency"], "Period": "30d", "Change": d30_change, "Symbol": row["Symbol"]}
                            ])
                        
                        plot_df = pd.DataFrame(plot_data)
                        
                        # Create grouped bar chart
                        fig = px.bar(
                            plot_df,
                            x="Cryptocurrency",
                            y="Change",
                            color="Period",
                            barmode="group",
                            title="Price Change Comparison",
                            text="Symbol",
                            color_discrete_sequence=["#1E88E5", "#D81B60", "#8E24AA"],
                            labels={"Change": "Price Change (%)"}
                        )
                        
                        # Add a zero line for reference
                        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                        
                        # Update layout
                        fig.update_layout(
                            height=500,
                            template="plotly_dark",
                            paper_bgcolor="#0A192F",
                            plot_bgcolor="#172A46",
                            font=dict(family="Arial, sans-serif", size=12, color="#F8F9FA"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        # Update text position
                        fig.update_traces(
                            textposition='inside',
                            textfont=dict(color="white", size=10)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_tab2:
                        # Create market cap comparison chart
                        market_cap_data = []
                        
                        for d in metrics_data:
                            # Convert the formatted string back to numeric
                            market_cap_str = d.get("Market Cap (USD)", "$0")
                            market_cap = float(market_cap_str.replace("$", "").replace(",", ""))
                            
                            market_cap_data.append({
                                "Cryptocurrency": d["Cryptocurrency"],
                                "Symbol": d["Symbol"],
                                "Market Cap": market_cap
                            })
                        
                        market_cap_df = pd.DataFrame(market_cap_data)
                        
                        # Sort by market cap
                        market_cap_df = market_cap_df.sort_values("Market Cap", ascending=False)
                        
                        # Create bar chart
                        fig = px.bar(
                            market_cap_df,
                            x="Cryptocurrency",
                            y="Market Cap",
                            title="Market Capitalization Comparison",
                            text="Symbol",
                            color="Market Cap",
                            color_continuous_scale=px.colors.sequential.Viridis,
                            labels={"Market Cap": "Market Cap (USD)"}
                        )
                        
                        # Update layout
                        fig.update_layout(
                            height=500,
                            template="plotly_dark",
                            paper_bgcolor="#0A192F",
                            plot_bgcolor="#172A46",
                            font=dict(family="Arial, sans-serif", size=12, color="#F8F9FA"),
                            coloraxis_showscale=False
                        )
                        
                        # Update y-axis to use log scale
                        fig.update_yaxes(type="log", title_text="Market Cap (USD, log scale)")
                        
                        # Update text position
                        fig.update_traces(
                            textposition='inside',
                            textfont=dict(color="white", size=12)
                        )
                        
                        # Update hover template to show formatted values
                        fig.update_traces(
                            hovertemplate='<b>%{x}</b><br>Market Cap: $%{y:,.0f}<extra></extra>'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_tab3:
                        # Create volume comparison chart
                        volume_data = []
                        
                        for d in metrics_data:
                            # Convert the formatted string back to numeric
                            volume_str = d.get("24h Volume (USD)", "$0")
                            volume = float(volume_str.replace("$", "").replace(",", ""))
                            
                            # Get price for calculation
                            price_str = d.get("Price (USD)", "$0")
                            price = float(price_str.replace("$", "").replace(",", ""))
                            
                            # Calculate volume/price ratio
                            vol_price_ratio = volume / price if price > 0 else 0
                            
                            volume_data.append({
                                "Cryptocurrency": d["Cryptocurrency"],
                                "Symbol": d["Symbol"],
                                "24h Volume": volume,
                                "Volume/Price Ratio": vol_price_ratio
                            })
                        
                        volume_df = pd.DataFrame(volume_data)
                        
                        # Sort by volume
                        volume_df = volume_df.sort_values("24h Volume", ascending=False)
                        
                        # Create bar chart
                        fig = px.bar(
                            volume_df,
                            x="Cryptocurrency",
                            y="24h Volume",
                            title="24h Trading Volume Comparison",
                            text="Symbol",
                            color="Volume/Price Ratio",
                            color_continuous_scale=px.colors.sequential.Plasma,
                            labels={"24h Volume": "24h Volume (USD)"}
                        )
                        
                        # Update layout
                        fig.update_layout(
                            height=500,
                            template="plotly_dark",
                            paper_bgcolor="#0A192F",
                            plot_bgcolor="#172A46",
                            font=dict(family="Arial, sans-serif", size=12, color="#F8F9FA")
                        )
                        
                        # Update y-axis to use log scale
                        fig.update_yaxes(type="log", title_text="24h Volume (USD, log scale)")
                        
                        # Update text position
                        fig.update_traces(
                            textposition='inside',
                            textfont=dict(color="white", size=12)
                        )
                        
                        # Update hover template to show formatted values
                        fig.update_traces(
                            hovertemplate='<b>%{x}</b><br>24h Volume: $%{y:,.0f}<br>Volume/Price Ratio: %{marker.color:,.0f}<extra></extra>'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                        **About Volume/Price Ratio:**
                        
                        The Volume/Price Ratio indicates how many units of currency are being traded relative to the price of the asset. A higher ratio suggests more liquidity relative to price. This can be a useful indicator of market interest and potential for price movement.
                        """)
                    
                    with viz_tab4:
                        # Create supply metrics comparison
                        supply_data = []
                        
                        for d in metrics_data:
                            # Convert the formatted strings back to numeric
                            circ_supply_str = d.get("Circulating Supply", "0")
                            circ_supply = float(circ_supply_str.replace(",", "")) if circ_supply_str != "∞" else 0
                            
                            total_supply_str = d.get("Total Supply", "0")
                            total_supply = float(total_supply_str.replace(",", "")) if total_supply_str != "∞" else 0
                            
                            max_supply_str = d.get("Max Supply", "0")
                            max_supply = float(max_supply_str.replace(",", "")) if max_supply_str != "∞" else 0
                            
                            # Calculate supply percentage
                            if max_supply > 0:
                                supply_percentage = (circ_supply / max_supply) * 100
                            elif total_supply > 0:
                                supply_percentage = (circ_supply / total_supply) * 100
                            else:
                                supply_percentage = 100  # If no max or total supply, assume 100%
                            
                            supply_data.append({
                                "Cryptocurrency": d["Cryptocurrency"],
                                "Symbol": d["Symbol"],
                                "Circulating Supply": circ_supply,
                                "Total Supply": total_supply,
                                "Max Supply": max_supply,
                                "Supply Percentage": supply_percentage
                            })
                        
                        supply_df = pd.DataFrame(supply_data)
                        
                        # Create subplots: one for supply comparison, one for percentage
                        fig = make_subplots(
                            rows=2, 
                            cols=1,
                            subplot_titles=("Supply Comparison", "Percentage of Maximum Supply in Circulation"),
                            row_heights=[0.7, 0.3],
                            vertical_spacing=0.15
                        )
                        
                        # For each cryptocurrency, add bars for different supply types
                        for i, crypto in enumerate(supply_df["Cryptocurrency"]):
                            # Add circulating supply bar
                            fig.add_trace(
                                go.Bar(
                                    x=[crypto],
                                    y=[supply_df.iloc[i]["Circulating Supply"]],
                                    name="Circulating Supply",
                                    marker_color="#1E88E5",
                                    text=supply_df.iloc[i]["Symbol"],
                                    showlegend=i == 0
                                ),
                                row=1, col=1
                            )
                            
                            # Add total supply bar
                            if supply_df.iloc[i]["Total Supply"] > 0:
                                fig.add_trace(
                                    go.Bar(
                                        x=[crypto],
                                        y=[supply_df.iloc[i]["Total Supply"] - supply_df.iloc[i]["Circulating Supply"]],
                                        name="Remaining Total Supply",
                                        marker_color="#D81B60",
                                        text=supply_df.iloc[i]["Symbol"],
                                        showlegend=i == 0
                                    ),
                                    row=1, col=1
                                )
                            
                            # Add max supply bar (if different from total)
                            if supply_df.iloc[i]["Max Supply"] > supply_df.iloc[i]["Total Supply"]:
                                fig.add_trace(
                                    go.Bar(
                                        x=[crypto],
                                        y=[supply_df.iloc[i]["Max Supply"] - supply_df.iloc[i]["Total Supply"]],
                                        name="Remaining Max Supply",
                                        marker_color="#8E24AA",
                                        text=supply_df.iloc[i]["Symbol"],
                                        showlegend=i == 0
                                    ),
                                    row=1, col=1
                                )
                            
                            # Add supply percentage bar
                            fig.add_trace(
                                go.Bar(
                                    x=[crypto],
                                    y=[supply_df.iloc[i]["Supply Percentage"]],
                                    name="Supply Percentage",
                                    marker_color="#43A047",
                                    text=f"{supply_df.iloc[i]['Supply Percentage']:.1f}%",
                                    showlegend=False
                                ),
                                row=2, col=1
                            )
                        
                        # Update layout
                        fig.update_layout(
                            height=700,
                            title_text="Cryptocurrency Supply Metrics",
                            template="plotly_dark",
                            paper_bgcolor="#0A192F",
                            plot_bgcolor="#172A46",
                            font=dict(family="Arial, sans-serif", size=12, color="#F8F9FA"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            barmode="stack"
                        )
                        
                        # Update y-axis titles
                        fig.update_yaxes(title_text="Supply (Tokens)", row=1, col=1)
                        fig.update_yaxes(title_text="Percentage (%)", row=2, col=1)
                        
                        # Update hover templates
                        fig.update_traces(
                            hovertemplate='<b>%{x}</b><br>%{data.name}: %{y:,.0f}<extra></extra>',
                            row=1, col=1
                        )
                        
                        fig.update_traces(
                            hovertemplate='<b>%{x}</b><br>Circulating: %{y:.1f}%<extra></extra>',
                            row=2, col=1
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                        **Understanding Supply Metrics:**
                        
                        This chart shows three important supply metrics for each cryptocurrency:
                        
                        1. **Circulating Supply**: The number of coins currently available and in public hands
                        2. **Total Supply**: All coins that currently exist (minted or mined)
                        3. **Maximum Supply**: The maximum number of coins that will ever exist
                        
                        The bottom chart shows what percentage of the maximum (or total) supply is currently in circulation. A lower percentage may indicate more potential for inflation as more coins enter the market.
                        """)
                    
                    with viz_tab5:
                        # Create ATH analysis chart
                        ath_data = []
                        
                        for d in metrics_data:
                            # Convert the formatted strings back to numeric
                            ath_str = d.get("All-Time High (USD)", "$0")
                            ath = float(ath_str.replace("$", "").replace(",", ""))
                            
                            price_str = d.get("Price (USD)", "$0")
                            price = float(price_str.replace("$", "").replace(",", ""))
                            
                            # Calculate percentage from ATH
                            if ath > 0:
                                pct_from_ath = ((price - ath) / ath) * 100
                            else:
                                pct_from_ath = 0
                            
                            # Calculate days since ATH
                            ath_date_str = d.get("ATH Date", "")
                            if ath_date_str != "N/A":
                                ath_date = datetime.strptime(ath_date_str, "%Y-%m-%d")
                                days_since_ath = (datetime.now() - ath_date).days
                            else:
                                days_since_ath = 0
                            
                            ath_data.append({
                                "Cryptocurrency": d["Cryptocurrency"],
                                "Symbol": d["Symbol"],
                                "All-Time High": ath,
                                "Current Price": price,
                                "% From ATH": pct_from_ath,
                                "Days Since ATH": days_since_ath
                            })
                        
                        ath_df = pd.DataFrame(ath_data)
                        
                        # Create scatter plot for ATH analysis
                        fig = px.scatter(
                            ath_df,
                            x="Days Since ATH",
                            y="% From ATH",
                            size="All-Time High",
                            color="% From ATH",
                            hover_name="Cryptocurrency",
                            text="Symbol",
                            title="All-Time High Analysis",
                            color_continuous_scale=px.colors.diverging.RdBu_r,
                            labels={
                                "Days Since ATH": "Days Since All-Time High",
                                "% From ATH": "% Below All-Time High"
                            },
                            size_max=50,
                            range_color=[-100, 0]
                        )
                        
                        # Update layout
                        fig.update_layout(
                            height=600,
                            template="plotly_dark",
                            paper_bgcolor="#0A192F",
                            plot_bgcolor="#172A46",
                            font=dict(family="Arial, sans-serif", size=12, color="#F8F9FA")
                        )
                        
                        # Update hover template
                        fig.update_traces(
                            hovertemplate='<b>%{hovertext}</b><br>Symbol: %{text}<br>Days Since ATH: %{x}<br>% Below ATH: %{y:.2f}%<br>ATH Price: $%{marker.size:,.2f}<extra></extra>',
                            textposition='top center',
                            textfont=dict(color="white", size=12)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                        **Understanding ATH Analysis:**
                        
                        This chart shows how far each cryptocurrency is trading below its all-time high (ATH) and how long it has been since reaching that ATH.
                        
                        **Key insights:**
                        - **Y-axis**: Percentage below ATH (all values are typically negative in a non-bull market)
                        - **X-axis**: Number of days since the ATH was reached
                        - **Bubble size**: The ATH price in USD
                        
                        Cryptocurrencies closer to the top left may be considered stronger performers as they are either:
                        1. Closer to their all-time highs, or
                        2. Have set their all-time highs more recently
                        
                        This analysis can help identify cryptocurrencies with strong historical performance or those that may have more recovery potential.
                        """)
                else:
                    st.warning("No metrics data available for the selected cryptocurrencies.")
            else:
                error_msg = comparison_data.get("error", "Unknown error") if isinstance(comparison_data, dict) else "Unknown error"
                st.error(f"Error loading detailed metrics: {error_msg}")

if __name__ == "__main__":
    main()
