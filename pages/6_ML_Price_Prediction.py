import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

from utils.api import get_top_cryptocurrencies, get_timeframe_data, get_coin_history
from utils.ml_models import train_prediction_models, predict_prices, get_feature_importance
from utils.styles import apply_styles

# Apply custom styling
apply_styles()

def main():
    st.title("🤖 ML Price Prediction")
    
    # Page description
    st.markdown("""
    This page uses advanced machine learning models to predict cryptocurrency prices with higher accuracy.
    Models include Random Forest, XGBoost, and Prophet to provide multiple prediction perspectives.
    """)
    
    # Set up the sidebar
    with st.sidebar:
        st.header("Prediction Settings")
        
        # Get list of top cryptocurrencies for the dropdown
        try:
            top_coins_df = get_top_cryptocurrencies(limit=100)
            if not isinstance(top_coins_df, pd.DataFrame) or top_coins_df.empty:
                st.error("Unable to fetch cryptocurrency data. Please refresh the page.")
                coin_options = [("bitcoin", "Bitcoin (BTC)")]
            else:
                coin_options = [(row['id'], f"{row['name']} ({row['symbol'].upper()})") for _, row in top_coins_df.iterrows()]
        except Exception as e:
            st.error(f"Error loading cryptocurrencies: {str(e)}")
            coin_options = [("bitcoin", "Bitcoin (BTC)")]
        
        # Cryptocurrency selection
        selected_coin_id, selected_coin_name = coin_options[0]
        selected_coin = st.selectbox(
            "Select Cryptocurrency",
            options=[name for _, name in coin_options],
            index=0
        )
        
        # Map the selected name back to the ID
        for coin_id, coin_name in coin_options:
            if coin_name == selected_coin:
                selected_coin_id = coin_id
                selected_coin_name = coin_name
                break
        
        # Historical data timeframe selection
        st.subheader("Training Data")
        timeframe_options = {
            "1h": "1 Hour",
            "4h": "4 Hours",
            "1d": "1 Day"
        }
        
        selected_timeframe = st.selectbox(
            "Data Resolution",
            options=list(timeframe_options.keys()),
            format_func=lambda x: timeframe_options[x],
            index=0
        )
        
        # Historical time period selection
        period_options = {
            "7": "1 Week",
            "14": "2 Weeks",
            "30": "1 Month",
            "90": "3 Months",
            "180": "6 Months",
            "365": "1 Year"
        }
        
        selected_period = st.selectbox(
            "Historical Period",
            options=list(period_options.keys()),
            format_func=lambda x: period_options[x],
            index=2  # Default to 1 month
        )
        
        # Prediction horizon
        st.subheader("Prediction Settings")
        prediction_horizon_options = {
            "6": "6 hours/days",
            "12": "12 hours/days",
            "24": "24 hours/days (1 day/week)",
            "48": "48 hours/days (2 days/weeks)",
            "72": "72 hours/days (3 days/weeks)"
        }
        
        prediction_horizon = st.selectbox(
            "Prediction Horizon",
            options=list(prediction_horizon_options.keys()),
            format_func=lambda x: prediction_horizon_options[x],
            index=2  # Default to 24 hours/days
        )
        
        # Model selection
        model_options = {
            "best": "Best Performing Model",
            "xgboost": "XGBoost (Best for Short-term)",
            "random_forest": "Random Forest (Balanced)",
            "prophet": "Prophet (Best for Seasonality)"
        }
        
        selected_model = st.selectbox(
            "Prediction Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )
        
        # Button to run prediction
        run_prediction = st.button("Generate Prediction", type="primary")
        
        # Display info about model retraining
        st.info(
            "Models are trained on the selected historical data. For best results, use more historical data "
            "for long-term predictions and recent data for short-term predictions."
        )
    
    # Main content area
    st.header(f"Price Prediction for {selected_coin_name.split(' (')[0]}")
    
    # Tab for prediction results and model details
    tab1, tab2, tab3 = st.tabs(["Prediction Results", "Model Details", "Feature Importance"])
    
    with tab1:
        if run_prediction:
            with st.spinner(f"Fetching historical data for {selected_coin_name}..."):
                # Get historical data
                if selected_timeframe == "1h":
                    # For hourly data, we'll use a different API endpoint
                    days_to_fetch = min(90, int(selected_period))  # API limitation
                    hist_data = get_coin_history(selected_coin_id, days=str(days_to_fetch), interval="hourly")
                else:
                    # For daily data
                    hist_data = get_coin_history(selected_coin_id, days=selected_period)
            
            if not isinstance(hist_data, pd.DataFrame) or hist_data.empty:
                st.error(f"Unable to fetch historical data for {selected_coin_name}. Please try another cryptocurrency or timeframe.")
            else:
                with st.spinner("Training machine learning models and generating predictions..."):
                    # Convert prediction horizon to int
                    horizon = int(prediction_horizon)
                    
                    # Choose the model name based on selection
                    model_name = None if selected_model == "best" else selected_model
                    
                    # Make predictions
                    predictions = predict_prices(
                        selected_coin_id, 
                        selected_timeframe, 
                        hist_data, 
                        periods=horizon,
                        model_name=model_name
                    )
                    
                    if "error" in predictions:
                        st.error(f"Error generating predictions: {predictions['error']}")
                    else:
                        st.success(f"Successfully generated predictions using {predictions['model_used']} model!")
                        
                        # Display the prediction chart
                        fig = go.Figure()
                        
                        # Display historical data (last 30 points)
                        hist_len = min(30, len(hist_data))
                        historical_data = hist_data.tail(hist_len).copy()
                        
                        # Add historical price line
                        fig.add_trace(
                            go.Scatter(
                                x=historical_data.index,
                                y=historical_data['price'],
                                mode='lines',
                                name='Historical Price',
                                line=dict(color='#1E88E5', width=2)
                            )
                        )
                        
                        # Add predicted price line
                        fig.add_trace(
                            go.Scatter(
                                x=predictions['timestamps'],
                                y=predictions['predictions'],
                                mode='lines+markers',
                                name='Predicted Price',
                                line=dict(color='#00C853', width=3, dash='dash'),
                                marker=dict(size=8, symbol='circle')
                            )
                        )
                        
                        # Add confidence interval if available
                        if 'lower_bound' in predictions and 'upper_bound' in predictions:
                            fig.add_trace(
                                go.Scatter(
                                    x=predictions['timestamps'] + predictions['timestamps'][::-1],
                                    y=predictions['upper_bound'] + predictions['lower_bound'][::-1],
                                    fill='toself',
                                    fillcolor='rgba(0, 200, 83, 0.2)',
                                    line=dict(color='rgba(255, 255, 255, 0)'),
                                    name='95% Confidence Interval'
                                )
                            )
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{selected_coin_name.split(' (')[0]} Price Prediction ({horizon} periods)",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            height=500,
                            template="plotly_dark",
                            hovermode="x unified",
                            font=dict(family="Arial, sans-serif", size=12, color="#F8F9FA"),
                            margin=dict(l=20, r=20, t=50, b=20),
                            paper_bgcolor="#0A192F",
                            plot_bgcolor="#172A46"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display prediction summary
                        st.subheader("Prediction Summary")
                        
                        # Create summary metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            current_price = predictions['current_price']
                            last_predicted = predictions['predictions'][-1]
                            change_pct = ((last_predicted / current_price) - 1) * 100
                            
                            delta_color = "normal" if change_pct >= 0 else "inverse"
                            
                            st.metric(
                                "Predicted Final Price", 
                                f"${last_predicted:.4f}", 
                                f"{change_pct:+.2f}%",
                                delta_color=delta_color
                            )
                        
                        with col2:
                            max_price = max(predictions['predictions'])
                            max_change = ((max_price / current_price) - 1) * 100
                            
                            st.metric(
                                "Maximum Predicted Price",
                                f"${max_price:.4f}",
                                f"{max_change:+.2f}%",
                                delta_color="normal"
                            )
                        
                        with col3:
                            min_price = min(predictions['predictions'])
                            min_change = ((min_price / current_price) - 1) * 100
                            
                            st.metric(
                                "Minimum Predicted Price",
                                f"${min_price:.4f}",
                                f"{min_change:+.2f}%",
                                delta_color="inverse" if min_change < 0 else "normal"
                            )
                        
                        # Display table of predictions
                        st.subheader("Detailed Predictions")
                        
                        # Create dataframe for predictions
                        pred_df = pd.DataFrame({
                            "Timestamp": [ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else ts for ts in predictions['timestamps']],
                            "Predicted Price": [f"${p:.4f}" for p in predictions['predictions']],
                            "Change (%)": [f"{p:+.2f}%" for p in predictions['price_change_pct']]
                        })
                        
                        st.dataframe(pred_df, use_container_width=True)
                        
                        # Warning about predictions
                        st.warning(
                            "⚠️ These predictions are based on historical patterns and should not be used as the sole basis "
                            "for investment decisions. Cryptocurrency markets are highly volatile and unpredictable."
                        )
        else:
            st.info("Click the 'Generate Prediction' button to run the machine learning models and generate price predictions.")
    
    with tab2:
        if run_prediction and 'predictions' in locals() and 'error' not in predictions:
            st.subheader("Model Information")
            
            model_used = predictions.get('model_used', 'Unknown')
            
            # Display model information
            st.markdown(f"### Model: {model_used}")
            
            if model_used == 'random_forest':
                st.markdown("""
                **Random Forest Regressor** is an ensemble learning method that operates by constructing multiple decision trees 
                during training and outputting the average prediction of the individual trees. Key characteristics:
                
                - 🔹 Handles both numerical and categorical features
                - 🔹 Resistant to overfitting
                - 🔹 Provides good balance between accuracy and interpretability
                - 🔹 Best suited for datasets with complex relationships
                """)
                
            elif model_used == 'xgboost':
                st.markdown("""
                **XGBoost (Extreme Gradient Boosting)** is an optimized gradient boosting algorithm that sequentially builds 
                new models to correct the errors made by existing models. Key characteristics:
                
                - 🔹 Often achieves superior performance compared to other algorithms
                - 🔹 Handles missing values automatically
                - 🔹 Implements regularization to prevent overfitting
                - 🔹 Particularly effective for short-term price predictions
                """)
                
            elif model_used == 'prophet':
                st.markdown("""
                **Prophet** is a procedure for forecasting time series data developed by Facebook. It's particularly effective at:
                
                - 🔹 Capturing seasonal effects (daily, weekly, yearly patterns)
                - 🔹 Handling missing data and outliers well
                - 🔹 Automatically detecting trend changes
                - 🔹 Particularly suited for longer-term forecasts with seasonal patterns
                """)
            
            # Display model confidence
            confidence = predictions.get('confidence', 0) * 100
            st.metric("Model Confidence", f"{confidence:.1f}%")
            
            # Display training data info
            st.subheader("Training Data Information")
            
            if 'hist_data' in locals() and isinstance(hist_data, pd.DataFrame) and not hist_data.empty:
                st.markdown(f"""
                - **Time Period**: {period_options[selected_period]}
                - **Resolution**: {timeframe_options[selected_timeframe]}
                - **Data Points**: {len(hist_data)}
                - **Date Range**: {hist_data.index.min().strftime('%Y-%m-%d')} to {hist_data.index.max().strftime('%Y-%m-%d')}
                """)
            else:
                st.info("Training data information will be displayed after generating predictions.")
        else:
            st.info("Model details will be displayed after generating predictions.")
    
    with tab3:
        if run_prediction and 'predictions' in locals() and 'error' not in predictions:
            st.subheader("Feature Importance")
            
            model_used = predictions.get('model_used', 'Unknown')
            
            # Only show feature importance for tree-based models
            if model_used in ['random_forest', 'xgboost']:
                with st.spinner("Calculating feature importance..."):
                    feature_importance = get_feature_importance(selected_coin_id, selected_timeframe, model_used)
                    
                    if 'error' in feature_importance:
                        st.error(f"Error calculating feature importance: {feature_importance['error']}")
                    else:
                        # Create horizontal bar chart of feature importance
                        importance_dict = feature_importance['feature_importance']
                        
                        # Take top 15 features
                        top_features = dict(list(importance_dict.items())[:15])
                        
                        # Convert to dataframe for plotting
                        df_importance = pd.DataFrame({
                            'Feature': list(top_features.keys()),
                            'Importance': list(top_features.values())
                        })
                        
                        # Sort by importance
                        df_importance = df_importance.sort_values('Importance', ascending=True)
                        
                        # Create bar chart
                        fig = px.bar(
                            df_importance, 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            title=f"Top 15 Features for {model_used.capitalize()} Model",
                            color='Importance',
                            color_continuous_scale=['#64B5F6', '#1E88E5', '#0D47A1'],
                            template="plotly_dark"
                        )
                        
                        fig.update_layout(
                            height=500,
                            font=dict(family="Arial, sans-serif", size=12, color="#F8F9FA"),
                            margin=dict(l=20, r=20, t=50, b=20),
                            paper_bgcolor="#0A192F",
                            plot_bgcolor="#172A46"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Explain feature importance
                        st.markdown("""
                        **Understanding Feature Importance:**
                        
                        Feature importance indicates how much each feature contributes to the model's predictions. 
                        Higher values indicate greater importance.
                        
                        - **Price-related features**: Historical prices and their derivatives (like moving averages, lag values)
                        - **Time-based features**: Hours, days, months that capture cyclical patterns
                        - **Volume-related features**: Trading volume indicates market activity
                        """)
                        
                        # Display most important features
                        st.subheader("Most Influential Factors")
                        
                        # Take top 5 features and explain them
                        top_5 = list(importance_dict.items())[:5]
                        
                        for i, (feature, importance) in enumerate(top_5, 1):
                            display_name = feature.replace('_', ' ').title()
                            st.markdown(f"**{i}. {display_name}** - Importance: {importance:.4f}")
            else:
                st.info(
                    "Feature importance analysis is only available for tree-based models (Random Forest and XGBoost). "
                    "Prophet uses a different approach based on time series decomposition."
                )
        else:
            st.info("Feature importance information will be displayed after generating predictions with a tree-based model.")

if __name__ == "__main__":
    main()