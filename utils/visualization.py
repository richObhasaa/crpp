import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.constants import CHART_COLORS

def create_price_chart(df, coin_name, timeframe, include_volume=True):
    """Create a price chart with technical indicators"""
    if isinstance(df, dict) and "error" in df:
        return None
    
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    
    try:
        # Create subplot with two rows - price and volume
        if include_volume:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.7, 0.3],
                                subplot_titles=(f"{coin_name} Price ({timeframe})", "Volume"))
        else:
            fig = make_subplots(rows=1, cols=1, subplot_titles=(f"{coin_name} Price ({timeframe})",))
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], 
                y=df["price"],
                mode='lines',
                name='Price',
                line=dict(color=CHART_COLORS["price_up"], width=2),
                hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add technical indicators if they exist in the dataframe
        if 'SMA_7' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["SMA_7"],
                    mode='lines',
                    name='SMA (7)',
                    line=dict(color='#17BECF', width=1.5, dash='dot'),
                    hovertemplate='%{x}<br>SMA (7): $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        if 'SMA_25' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["SMA_25"],
                    mode='lines',
                    name='SMA (25)',
                    line=dict(color='#7F7F7F', width=1.5, dash='dot'),
                    hovertemplate='%{x}<br>SMA (25): $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["BB_upper"],
                    mode='lines',
                    name='Upper BB',
                    line=dict(color='rgba(255, 144, 14, 0.5)', width=1),
                    hovertemplate='%{x}<br>Upper BB: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["BB_lower"],
                    mode='lines',
                    name='Lower BB',
                    line=dict(color='rgba(255, 144, 14, 0.5)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(255, 144, 14, 0.1)',
                    hovertemplate='%{x}<br>Lower BB: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add volume chart
        if include_volume:
            fig.add_trace(
                go.Bar(
                    x=df["timestamp"],
                    y=df["volume"],
                    name='Volume',
                    marker=dict(color=CHART_COLORS["volume"]),
                    hovertemplate='%{x}<br>Volume: $%{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=600,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="#F8F9FA"
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified",
            paper_bgcolor="#0A192F",
            plot_bgcolor="#172A46"
        )
        
        # Update x-axis and y-axis styles
        fig.update_xaxes(
            title_text="Date",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor="#293B5F",
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="Price (USD)",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            tickprefix="$",
            gridcolor="#293B5F",
            zeroline=False,
            row=1, col=1
        )
        
        if include_volume:
            fig.update_xaxes(
                title_text="Date",
                title_font=dict(size=12),
                tickfont=dict(size=10),
                gridcolor="#293B5F",
                row=2, col=1
            )
            
            fig.update_yaxes(
                title_text="Volume",
                title_font=dict(size=12),
                tickfont=dict(size=10),
                gridcolor="#293B5F",
                zeroline=False,
                row=2, col=1
            )
        
        return fig
    except Exception as e:
        print(f"Error creating price chart: {str(e)}")
        return None

def create_market_cap_chart(df, coin_name, timeframe):
    """Create a market cap chart"""
    if isinstance(df, dict) and "error" in df:
        return None
    
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    
    try:
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["market_cap"],
                mode='lines',
                name='Market Cap',
                line=dict(color=CHART_COLORS["market_cap"], width=2),
                fill='tozeroy',
                fillcolor=f'rgba({int(CHART_COLORS["market_cap"][1:3], 16)}, {int(CHART_COLORS["market_cap"][3:5], 16)}, {int(CHART_COLORS["market_cap"][5:7], 16)}, 0.2)',
                hovertemplate='%{x}<br>Market Cap: $%{y:,.0f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title=f"{coin_name} Market Cap ({timeframe})",
            height=400,
            template="plotly_dark",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="#F8F9FA"
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified",
            paper_bgcolor="#0A192F",
            plot_bgcolor="#172A46"
        )
        
        fig.update_xaxes(
            title_text="Date",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor="#293B5F"
        )
        
        fig.update_yaxes(
            title_text="Market Cap (USD)",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            tickprefix="$",
            gridcolor="#293B5F",
            zeroline=False,
        )
        
        return fig
    except Exception as e:
        print(f"Error creating market cap chart: {str(e)}")
        return None

def create_market_distribution_chart(market_data):
    """Create a market distribution chart (pie or treemap)"""
    if "error" in market_data:
        return None
    
    try:
        # Extract market cap percentages
        market_percentages = market_data.get("market_cap_percentage", {})
        
        if not market_percentages:
            return None
        
        # Create dataframe from the market cap percentages
        data = []
        for coin, percentage in market_percentages.items():
            data.append({
                "coin": coin.upper(),
                "percentage": percentage,
                "market_cap": market_data.get("total_market_cap", {}).get(coin.lower(), 0)
            })
        
        df = pd.DataFrame(data)
        
        # Sort by percentage (descending)
        df = df.sort_values("percentage", ascending=False)
        
        # Add "Others" category for coins with small percentages
        threshold = 1.0  # Coins with less than 1% go into "Others"
        others_df = df[df["percentage"] < threshold]
        
        if not others_df.empty:
            others_sum = others_df["percentage"].sum()
            others_market_cap = others_df["market_cap"].sum()
            
            # Remove small coins and add "Others"
            df = df[df["percentage"] >= threshold]
            df = pd.concat([df, pd.DataFrame([{
                "coin": "Others",
                "percentage": others_sum,
                "market_cap": others_market_cap
            }])], ignore_index=True)
        
        # Create pie chart
        fig = px.pie(
            df, 
            values="percentage", 
            names="coin",
            title="Cryptocurrency Market Distribution",
            hover_data=["market_cap"],
            color_discrete_sequence=CHART_COLORS["distribution"],
            labels={"percentage": "Market Dominance (%)", "coin": "Cryptocurrency", "market_cap": "Market Cap (USD)"}
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Market Dominance: %{value:.2f}%<br>Market Cap: $%{customdata[0]:,.0f}<extra></extra>',
            textinfo='label+percent'
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            template="plotly_dark",
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="#F8F9FA"
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#0A192F"
        )
        
        return fig
    except Exception as e:
        print(f"Error creating market distribution chart: {str(e)}")
        return None

def create_market_dominance_chart(historical_data, timeframe):
    """Create a market dominance chart over time"""
    if "error" in historical_data:
        return None
    
    try:
        # Process historical dominance data
        # This assumes historical_data contains dominance time series
        if not isinstance(historical_data, dict) or "dominance" not in historical_data:
            return None
        
        dominance_data = historical_data["dominance"]
        
        # Create figure
        fig = go.Figure()
        
        # Add trace for each major cryptocurrency
        for coin, data in dominance_data.items():
            if isinstance(data, list) and len(data) > 0:
                dates = [entry.get("date") for entry in data]
                values = [entry.get("value", 0) for entry in data]
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=values,
                        mode='lines',
                        name=coin.upper(),
                        hovertemplate='%{x}<br>' + coin.upper() + ': %{y:.2f}%<extra></extra>'
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=f"Cryptocurrency Market Dominance Over Time ({timeframe})",
            height=500,
            template="plotly_dark",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="#F8F9FA"
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified",
            paper_bgcolor="#0A192F",
            plot_bgcolor="#172A46"
        )
        
        fig.update_xaxes(
            title_text="Date",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor="#293B5F"
        )
        
        fig.update_yaxes(
            title_text="Market Dominance (%)",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            ticksuffix="%",
            gridcolor="#293B5F",
            zeroline=False,
        )
        
        return fig
    except Exception as e:
        print(f"Error creating market dominance chart: {str(e)}")
        return None

def create_price_vs_volume_chart(df, coin_name, timeframe):
    """Create a price vs volume scatter plot"""
    if isinstance(df, dict) and "error" in df:
        return None
    
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    
    try:
        # Ensure we have enough data points
        if len(df) < 3:
            return None

        # Calculate percentage changes
        df_pct = df.copy()
        df_pct["price_pct_change"] = df_pct["price"].pct_change() * 100
        df_pct["volume_pct_change"] = df_pct["volume"].pct_change() * 100
        
        # Remove NaN values and extreme outliers
        df_pct = df_pct.dropna()
        
        # Apply capping to extreme values for better visualization
        # Cap at 5 standard deviations
        for col in ["price_pct_change", "volume_pct_change"]:
            mean = df_pct[col].mean()
            std = df_pct[col].std()
            
            if not np.isnan(std) and std > 0:
                upper_bound = mean + 5 * std
                lower_bound = mean - 5 * std
                df_pct[col] = df_pct[col].clip(lower_bound, upper_bound)
        
        # Format timestamp for better display
        df_pct["formatted_timestamp"] = df_pct["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        
        # Create a color column based on price change
        df_pct["color"] = np.where(df_pct["price_pct_change"] >= 0, CHART_COLORS["price_up"], CHART_COLORS["price_down"])
        
        # Create scatter plot
        fig = px.scatter(
            df_pct,
            x="volume_pct_change",
            y="price_pct_change",
            color_discrete_map={True: CHART_COLORS["price_up"], False: CHART_COLORS["price_down"]},
            color=df_pct["price_pct_change"] >= 0,
            title=f"{coin_name} Price Change vs Volume Change ({timeframe})",
            labels={
                "volume_pct_change": "Volume Change (%)",
                "price_pct_change": "Price Change (%)"
            },
            hover_data=["formatted_timestamp", "price", "volume"]
        )
        
        # Add a zero line for reference
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        
        # Update hover template
        fig.update_traces(
            hovertemplate='<b>%{customdata[0]}</b><br>Price Change: %{y:.2f}%<br>Volume Change: %{x:.2f}%<br>Price: $%{customdata[1]:.2f}<br>Volume: $%{customdata[2]:,.0f}<extra></extra>',
            marker=dict(
                size=10,
                opacity=0.7,
                line=dict(width=1, color="#172A46")
            )
        )
        
        # Add quadrant labels
        fig.add_annotation(
            x=max(df_pct["volume_pct_change"])*0.7, 
            y=max(df_pct["price_pct_change"])*0.7,
            text="Strong Bullish<br>(High Volume, Price Increase)",
            showarrow=False,
            font=dict(color="#00C853", size=10)
        )
        
        fig.add_annotation(
            x=min(df_pct["volume_pct_change"])*0.7, 
            y=max(df_pct["price_pct_change"])*0.7,
            text="Weak Bullish<br>(Low Volume, Price Increase)",
            showarrow=False,
            font=dict(color="#64B5F6", size=10)
        )
        
        fig.add_annotation(
            x=min(df_pct["volume_pct_change"])*0.7, 
            y=min(df_pct["price_pct_change"])*0.7,
            text="Weak Bearish<br>(Low Volume, Price Decrease)",
            showarrow=False,
            font=dict(color="#FFB74D", size=10)
        )
        
        fig.add_annotation(
            x=max(df_pct["volume_pct_change"])*0.7, 
            y=min(df_pct["price_pct_change"])*0.7,
            text="Strong Bearish<br>(High Volume, Price Decrease)",
            showarrow=False,
            font=dict(color="#FF3D00", size=10)
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            template="plotly_dark",
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="#F8F9FA"
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#0A192F",
            plot_bgcolor="#172A46",
            legend_title_text="Price Change",
            showlegend=True
        )
        
        fig.update_xaxes(
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor="#293B5F",
            zeroline=True,
            zerolinecolor="#FFFFFF",
            zerolinewidth=1
        )
        
        fig.update_yaxes(
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor="#293B5F",
            zeroline=True,
            zerolinecolor="#FFFFFF",
            zerolinewidth=1
        )
        
        return fig
    except Exception as e:
        print(f"Error creating price vs volume chart: {str(e)}")
        return None

def create_comparison_chart(comparison_data, metric="price", timeframe="7d"):
    """Create a comparison chart for multiple cryptocurrencies"""
    if "error" in comparison_data:
        return None
    
    try:
        # Create figure
        fig = go.Figure()
        
        # Process comparison data for each coin
        for coin_id, data in comparison_data.items():
            if "error" in data:
                continue
            
            if metric == "price":
                # For price comparison, we need to normalize the data
                prices = data.get("price_data", [])
                if not prices:
                    continue
                
                # Extract timestamps and prices
                timestamps = [entry.get("timestamp") for entry in prices]
                values = [entry.get("price") for entry in prices]
                
                # Normalize prices to percentage change from starting point
                if values and len(values) > 0 and values[0] != 0:
                    start_value = values[0]
                    normalized_values = [(val / start_value - 1) * 100 for val in values]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=normalized_values,
                            mode='lines',
                            name=data.get("name", coin_id),
                            hovertemplate='%{x}<br>' + data.get("name", coin_id) + ': %{y:.2f}%<extra></extra>'
                        )
                    )
            elif metric == "market_cap":
                # Similar approach for market cap
                market_caps = data.get("market_cap_data", [])
                if not market_caps:
                    continue
                
                timestamps = [entry.get("timestamp") for entry in market_caps]
                values = [entry.get("market_cap") for entry in market_caps]
                
                if values and len(values) > 0 and values[0] != 0:
                    start_value = values[0]
                    normalized_values = [(val / start_value - 1) * 100 for val in values]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=normalized_values,
                            mode='lines',
                            name=data.get("name", coin_id),
                            hovertemplate='%{x}<br>' + data.get("name", coin_id) + ': %{y:.2f}%<extra></extra>'
                        )
                    )
            elif metric == "volume":
                # Similar approach for volume
                volumes = data.get("volume_data", [])
                if not volumes:
                    continue
                
                timestamps = [entry.get("timestamp") for entry in volumes]
                values = [entry.get("volume") for entry in volumes]
                
                if values and len(values) > 0 and values[0] != 0:
                    start_value = values[0]
                    normalized_values = [(val / start_value - 1) * 100 for val in values]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=normalized_values,
                            mode='lines',
                            name=data.get("name", coin_id),
                            hovertemplate='%{x}<br>' + data.get("name", coin_id) + ': %{y:.2f}%<extra></extra>'
                        )
                    )
        
        # Update layout
        metric_title = "Price" if metric == "price" else "Market Cap" if metric == "market_cap" else "Volume"
        fig.update_layout(
            title=f"Cryptocurrency {metric_title} Comparison ({timeframe})",
            height=500,
            template="plotly_dark",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="#F8F9FA"
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified",
            paper_bgcolor="#0A192F",
            plot_bgcolor="#172A46"
        )
        
        fig.update_xaxes(
            title_text="Date",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor="#293B5F"
        )
        
        fig.update_yaxes(
            title_text=f"% Change from Start ({timeframe})",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            ticksuffix="%",
            gridcolor="#293B5F",
            zeroline=True,
            zerolinecolor="white",
            zerolinewidth=1
        )
        
        return fig
    except Exception as e:
        print(f"Error creating comparison chart: {str(e)}")
        return None

def create_crypto_bubble_chart(market_data):
    """Create a bubble chart of cryptocurrencies"""
    try:
        if market_data is None or isinstance(market_data, dict) and "error" in market_data:
            return None
        
        if isinstance(market_data, pd.DataFrame) and not market_data.empty:
            df = market_data.copy()
            
            # Create bubble chart
            fig = px.scatter(
                df,
                x="price_change_percentage_24h",
                y="total_volume",
                size="market_cap",
                color="price_change_percentage_24h",
                color_continuous_scale=["#FF3D00", "#FFFFFF", "#00C853"],
                range_color=[-10, 10],
                hover_name="name",
                text="symbol",
                log_y=True,
                size_max=60,
                title="Cryptocurrency Bubble Chart (Size = Market Cap)",
                labels={
                    "price_change_percentage_24h": "24h Price Change (%)",
                    "total_volume": "24h Trading Volume (USD)",
                    "market_cap": "Market Cap (USD)"
                }
            )
            
            # Update hover template
            fig.update_traces(
                hovertemplate='<b>%{hovertext}</b><br>Symbol: %{text}<br>Price: $%{customdata}<br>24h Change: %{x:.2f}%<br>Volume: $%{y:,.0f}<br>Market Cap: $%{marker.size:,.0f}<extra></extra>',
                customdata=df["current_price"]
            )
            
            # Update layout
            fig.update_layout(
                height=600,
                template="plotly_dark",
                font=dict(
                    family="Arial, sans-serif",
                    size=12,
                    color="#F8F9FA"
                ),
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="#0A192F",
                plot_bgcolor="#172A46"
            )
            
            fig.update_xaxes(
                title_font=dict(size=12),
                tickfont=dict(size=10),
                gridcolor="#293B5F",
                zeroline=True,
                zerolinecolor="#FFFFFF",
                zerolinewidth=1
            )
            
            fig.update_yaxes(
                title_font=dict(size=12),
                tickfont=dict(size=10),
                gridcolor="#293B5F"
            )
            
            return fig
        else:
            return None
    except Exception as e:
        print(f"Error creating crypto bubble chart: {str(e)}")
        return None

def create_technical_indicator_chart(df, indicator, coin_name, timeframe):
    """Create a chart for a specific technical indicator"""
    if isinstance(df, dict) and "error" in df:
        return None
    
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    
    try:
        fig = go.Figure()
        
        if indicator == "rsi":
            # RSI Chart
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["RSI"],
                    mode='lines',
                    name='RSI',
                    line=dict(color="#E91E63", width=2),
                    hovertemplate='%{x}<br>RSI: %{y:.2f}<extra></extra>'
                )
            )
            
            # Add overbought and oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", annotation_position="top right")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", annotation_position="bottom right")
            
            fig.update_layout(
                title=f"{coin_name} RSI Indicator ({timeframe})",
                yaxis_title="RSI Value"
            )
            
        elif indicator == "macd":
            # MACD Chart
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["MACD_line"],
                    mode='lines',
                    name='MACD Line',
                    line=dict(color="#2196F3", width=2),
                    hovertemplate='%{x}<br>MACD Line: %{y:.6f}<extra></extra>'
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["MACD_signal"],
                    mode='lines',
                    name='Signal Line',
                    line=dict(color="#FF9800", width=2),
                    hovertemplate='%{x}<br>Signal Line: %{y:.6f}<extra></extra>'
                )
            )
            
            # Add MACD Histogram
            colors = np.where(df["MACD_histogram"] >= 0, CHART_COLORS["price_up"], CHART_COLORS["price_down"])
            
            fig.add_trace(
                go.Bar(
                    x=df["timestamp"],
                    y=df["MACD_histogram"],
                    name='Histogram',
                    marker_color=colors,
                    hovertemplate='%{x}<br>Histogram: %{y:.6f}<extra></extra>'
                )
            )
            
            fig.update_layout(
                title=f"{coin_name} MACD Indicator ({timeframe})",
                yaxis_title="MACD Value"
            )
            
        elif indicator == "bollinger":
            # Bollinger Bands Chart
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["price"],
                    mode='lines',
                    name='Price',
                    line=dict(color=CHART_COLORS["price_up"], width=2),
                    hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>'
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["BB_upper"],
                    mode='lines',
                    name='Upper Band',
                    line=dict(color='rgba(173, 216, 230, 0.8)', width=1),
                    hovertemplate='%{x}<br>Upper Band: $%{y:.2f}<extra></extra>'
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["BB_middle"],
                    mode='lines',
                    name='Middle Band (SMA 20)',
                    line=dict(color='rgba(173, 216, 230, 0.8)', width=1, dash='dash'),
                    hovertemplate='%{x}<br>Middle Band: $%{y:.2f}<extra></extra>'
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["BB_lower"],
                    mode='lines',
                    name='Lower Band',
                    line=dict(color='rgba(173, 216, 230, 0.8)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(173, 216, 230, 0.1)',
                    hovertemplate='%{x}<br>Lower Band: $%{y:.2f}<extra></extra>'
                )
            )
            
            fig.update_layout(
                title=f"{coin_name} Bollinger Bands ({timeframe})",
                yaxis_title="Price (USD)"
            )
            
        elif indicator == "volume":
            # Volume Chart
            colors = np.where(df["price"].diff() >= 0, CHART_COLORS["price_up"], CHART_COLORS["price_down"])
            
            fig.add_trace(
                go.Bar(
                    x=df["timestamp"],
                    y=df["volume"],
                    name='Volume',
                    marker_color=colors,
                    hovertemplate='%{x}<br>Volume: $%{y:,.0f}<extra></extra>'
                )
            )
            
            # Add moving average of volume
            if len(df) >= 20:
                df['volume_ma'] = df['volume'].rolling(window=20).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["volume_ma"],
                        mode='lines',
                        name='Volume MA (20)',
                        line=dict(color='white', width=2),
                        hovertemplate='%{x}<br>Volume MA (20): $%{y:,.0f}<extra></extra>'
                    )
                )
            
            fig.update_layout(
                title=f"{coin_name} Volume ({timeframe})",
                yaxis_title="Volume (USD)"
            )
        
        # Common layout updates
        fig.update_layout(
            height=400,
            template="plotly_dark",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="#F8F9FA"
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified",
            paper_bgcolor="#0A192F",
            plot_bgcolor="#172A46"
        )
        
        fig.update_xaxes(
            title_text="Date",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor="#293B5F"
        )
        
        fig.update_yaxes(
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor="#293B5F",
            zeroline=False,
        )
        
        return fig
    except Exception as e:
        print(f"Error creating technical indicator chart: {str(e)}")
        return None
