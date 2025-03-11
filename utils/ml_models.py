import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from prophet import Prophet
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Suppress warnings
warnings.filterwarnings('ignore')

class CryptoPredictor:
    """Class for training and making predictions using various ML models"""
    
    def __init__(self, coin_id, timeframe="1h"):
        """Initialize the predictor with the coin ID and timeframe"""
        self.coin_id = coin_id
        self.timeframe = timeframe
        self.models = {}
        self.scalers = {}
        self.features = []
        self.target = 'price'
        self.prediction_horizon = 24  # Default: predict 24 periods ahead
        
    def prepare_data(self, df):
        """Prepare data for ML models by creating features and scaling"""
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Invalid or empty dataframe provided")
            return None
        
        try:
            # Make a copy to avoid modifying the original
            data = df.copy()
            
            # Ensure we have datetime index
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
            
            # Required columns
            required_cols = ['price', 'volume', 'market_cap']
            if not all(col in data.columns for col in required_cols):
                logger.error(f"Required columns missing. Needed: {required_cols}, Found: {data.columns}")
                return None
            
            # Add time-based features
            data['hour'] = data.index.hour
            data['day_of_week'] = data.index.dayofweek
            data['day_of_month'] = data.index.day
            data['month'] = data.index.month
            data['year'] = data.index.year
            
            # Add lag features for price
            for lag in [1, 3, 6, 12, 24]:
                if len(data) > lag:
                    data[f'price_lag_{lag}'] = data['price'].shift(lag)
            
            # Add rolling statistics
            for window in [6, 12, 24, 48]:
                if len(data) > window:
                    data[f'price_rolling_mean_{window}'] = data['price'].rolling(window=window).mean()
                    data[f'price_rolling_std_{window}'] = data['price'].rolling(window=window).std()
                    data[f'volume_rolling_mean_{window}'] = data['volume'].rolling(window=window).mean()
            
            # Add percentage changes
            data['price_pct_change'] = data['price'].pct_change()
            data['volume_pct_change'] = data['volume'].pct_change()
            
            # Add VWAP (Volume Weighted Average Price)
            data['vwap'] = (data['price'] * data['volume']).rolling(window=24).sum() / data['volume'].rolling(window=24).sum()
            
            # Drop NaN values
            data.dropna(inplace=True)
            
            # Save the feature names
            self.features = [col for col in data.columns if col != self.target]
            
            return data
        
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None
    
    def train_random_forest(self, data, test_size=0.2):
        """Train a Random Forest model"""
        try:
            # Prepare features and target
            X = data[self.features]
            y = data[self.target]
            
            # Scale the features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['random_forest'] = scaler
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, shuffle=False)
            
            # Create and train the model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Save the model
            self.models['random_forest'] = model
            
            # Save model to disk
            model_path = os.path.join(MODELS_DIR, f'{self.coin_id}_{self.timeframe}_rf.joblib')
            scaler_path = os.path.join(MODELS_DIR, f'{self.coin_id}_{self.timeframe}_rf_scaler.joblib')
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            return {
                'model': 'Random Forest',
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'model_path': model_path
            }
        
        except Exception as e:
            logger.error(f"Error training Random Forest model: {str(e)}")
            return {'error': str(e)}
    
    def train_xgboost(self, data, test_size=0.2):
        """Train an XGBoost model"""
        try:
            # Prepare features and target
            X = data[self.features]
            y = data[self.target]
            
            # Scale the features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['xgboost'] = scaler
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, shuffle=False)
            
            # Create and train the model
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Save the model
            self.models['xgboost'] = model
            
            # Save model to disk
            model_path = os.path.join(MODELS_DIR, f'{self.coin_id}_{self.timeframe}_xgb.joblib')
            scaler_path = os.path.join(MODELS_DIR, f'{self.coin_id}_{self.timeframe}_xgb_scaler.joblib')
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            return {
                'model': 'XGBoost',
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'model_path': model_path
            }
        
        except Exception as e:
            logger.error(f"Error training XGBoost model: {str(e)}")
            return {'error': str(e)}
    
    def train_prophet(self, data):
        """Train a Prophet model"""
        try:
            # Prophet requires specific column names: 'ds' for date and 'y' for target
            prophet_data = data.reset_index()
            prophet_data = prophet_data.rename(columns={'timestamp': 'ds', 'price': 'y'})
            
            # Create and train the model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            # Add volume as a regressor if available
            if 'volume' in prophet_data.columns:
                model.add_regressor('volume')
            
            model.fit(prophet_data[['ds', 'y', 'volume'] if 'volume' in prophet_data.columns else ['ds', 'y']])
            
            # Make predictions for the training data to evaluate
            future = model.make_future_dataframe(periods=0, freq='H')
            if 'volume' in prophet_data.columns:
                future['volume'] = prophet_data['volume'].values
            
            forecast = model.predict(future)
            
            # Evaluate the model
            y_true = prophet_data['y'].values
            y_pred = forecast['yhat'].values[:len(y_true)]
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Save the model
            self.models['prophet'] = model
            
            # Save model to disk
            model_path = os.path.join(MODELS_DIR, f'{self.coin_id}_{self.timeframe}_prophet.joblib')
            joblib.dump(model, model_path)
            
            return {
                'model': 'Prophet',
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'model_path': model_path
            }
        
        except Exception as e:
            logger.error(f"Error training Prophet model: {str(e)}")
            return {'error': str(e)}
    
    def train_all_models(self, df):
        """Train all available models and return evaluation metrics"""
        prepared_data = self.prepare_data(df)
        
        if prepared_data is None:
            return {'error': 'Failed to prepare data for training'}
        
        results = {}
        
        # Train Random Forest
        rf_result = self.train_random_forest(prepared_data)
        results['random_forest'] = rf_result
        
        # Train XGBoost
        xgb_result = self.train_xgboost(prepared_data)
        results['xgboost'] = xgb_result
        
        # Train Prophet if we have at least 2 days of data
        if len(prepared_data) >= 48:  # Assuming hourly data
            prophet_result = self.train_prophet(prepared_data)
            results['prophet'] = prophet_result
        else:
            results['prophet'] = {'error': 'Not enough data for Prophet model (need at least 48 data points)'}
        
        # Determine best model based on RMSE
        valid_models = {k: v for k, v in results.items() if 'error' not in v}
        if valid_models:
            best_model = min(valid_models.items(), key=lambda x: x[1]['rmse'])[0]
            results['best_model'] = best_model
        else:
            results['best_model'] = None
        
        return results
    
    def predict_with_model(self, model_name, data, periods=24):
        """Make predictions using the specified model"""
        if model_name not in self.models:
            return {'error': f'Model {model_name} not trained or loaded'}
        
        try:
            model = self.models[model_name]
            
            if model_name in ['random_forest', 'xgboost']:
                # ML models need the latest data point as a starting point
                latest_data = data.iloc[-1:].copy()
                predictions = []
                timestamps = []
                
                # Get the scaler
                scaler = self.scalers.get(model_name)
                if scaler is None:
                    return {'error': f'Scaler for {model_name} not found'}
                
                # Make predictions for each period
                for i in range(periods):
                    # Prepare features for prediction
                    pred_features = latest_data[self.features].values
                    pred_features_scaled = scaler.transform(pred_features)
                    
                    # Make prediction
                    prediction = model.predict(pred_features_scaled)[0]
                    predictions.append(prediction)
                    
                    # Create timestamp for this prediction
                    if i == 0:
                        next_timestamp = data.index[-1] + pd.Timedelta(hours=1)
                    else:
                        next_timestamp = timestamps[-1] + pd.Timedelta(hours=1)
                    timestamps.append(next_timestamp)
                    
                    # Update latest data for next prediction
                    latest_data = latest_data.copy()
                    latest_data['price'] = prediction
                    
                    # Update lag features, rolling stats, etc. (simplified)
                    for col in self.features:
                        if col not in ['hour', 'day_of_week', 'day_of_month', 'month', 'year']:
                            latest_data[col] = 0  # Simplification
                    
                    # Update time features
                    latest_data['hour'] = next_timestamp.hour
                    latest_data['day_of_week'] = next_timestamp.dayofweek
                    latest_data['day_of_month'] = next_timestamp.day
                    latest_data['month'] = next_timestamp.month
                    latest_data['year'] = next_timestamp.year
                
                # Create prediction results
                return {
                    'model': model_name,
                    'predictions': predictions,
                    'timestamps': timestamps,
                    'current_price': data['price'].iloc[-1]
                }
            
            elif model_name == 'prophet':
                # Prophet uses its own prediction method
                model = self.models['prophet']
                
                # Create future dataframe
                future = model.make_future_dataframe(periods=periods, freq='H')
                
                # Add volume regressor if it was used in training
                if model.extra_regressors and 'volume' in model.extra_regressors:
                    # Use the last volume value for future predictions (simple approach)
                    last_volume = data['volume'].iloc[-1]
                    for i in range(periods):
                        if len(future) > len(data) + i:
                            future.loc[len(data) + i, 'volume'] = last_volume
                
                # Make predictions
                forecast = model.predict(future)
                
                # Extract the future predictions
                predictions = forecast['yhat'].iloc[-periods:].values.tolist()
                timestamps = forecast['ds'].iloc[-periods:].tolist()
                
                return {
                    'model': 'Prophet',
                    'predictions': predictions,
                    'timestamps': timestamps,
                    'current_price': data['price'].iloc[-1],
                    'lower_bound': forecast['yhat_lower'].iloc[-periods:].values.tolist(),
                    'upper_bound': forecast['yhat_upper'].iloc[-periods:].values.tolist()
                }
            
            else:
                return {'error': f'Prediction method for {model_name} not implemented'}
        
        except Exception as e:
            logger.error(f"Error making predictions with {model_name}: {str(e)}")
            return {'error': str(e)}
    
    def predict_with_best_model(self, data, best_model=None, periods=24):
        """Make predictions using the best model or a specified model"""
        if not best_model and 'best_model' in self.models:
            best_model = self.models['best_model']
        
        if not best_model:
            # Try each model in order of typical performance
            for model_name in ['xgboost', 'random_forest', 'prophet']:
                if model_name in self.models:
                    best_model = model_name
                    break
        
        if best_model:
            return self.predict_with_model(best_model, data, periods)
        else:
            return {'error': 'No trained models available for prediction'}
    
    def load_model(self, model_name):
        """Load a previously saved model"""
        try:
            model_path = os.path.join(MODELS_DIR, f'{self.coin_id}_{self.timeframe}_{model_name}.joblib')
            scaler_path = os.path.join(MODELS_DIR, f'{self.coin_id}_{self.timeframe}_{model_name}_scaler.joblib')
            
            if not os.path.exists(model_path):
                return {'error': f'Model file not found: {model_path}'}
            
            # Load the model
            model = joblib.load(model_path)
            self.models[model_name] = model
            
            # Load the scaler if it exists (not needed for Prophet)
            if model_name in ['random_forest', 'xgboost'] and os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                self.scalers[model_name] = scaler
            
            return {'success': f'Successfully loaded {model_name} model'}
        
        except Exception as e:
            logger.error(f"Error loading {model_name} model: {str(e)}")
            return {'error': str(e)}
    
    def load_all_available_models(self):
        """Load all available models for this coin and timeframe"""
        results = {}
        
        for model_name in ['random_forest', 'xgboost', 'prophet']:
            result = self.load_model(model_name)
            results[model_name] = result
        
        return results

def train_prediction_models(coin_id, timeframe, historical_data):
    """Train prediction models for a specific coin and timeframe"""
    predictor = CryptoPredictor(coin_id, timeframe)
    results = predictor.train_all_models(historical_data)
    return results

def predict_prices(coin_id, timeframe, historical_data, periods=24, model_name=None):
    """Make price predictions for a specific coin and timeframe"""
    predictor = CryptoPredictor(coin_id, timeframe)
    
    # Try to load existing models first
    load_results = predictor.load_all_available_models()
    
    # Check if we have any successfully loaded models
    loaded_models = [k for k, v in load_results.items() if 'success' in v]
    
    # If no models loaded or specified model not loaded, train new models
    if not loaded_models or (model_name and model_name not in loaded_models):
        train_results = predictor.train_all_models(historical_data)
        best_model = train_results.get('best_model')
    else:
        # Use the specified model or choose the best from loaded ones
        if model_name and model_name in loaded_models:
            best_model = model_name
        else:
            # Prefer XGBoost, then Random Forest, then Prophet if available
            for preferred in ['xgboost', 'random_forest', 'prophet']:
                if preferred in loaded_models:
                    best_model = preferred
                    break
            else:
                best_model = loaded_models[0]
    
    # Make predictions
    predictions = predictor.predict_with_best_model(historical_data, best_model, periods)
    
    if 'error' in predictions:
        logger.error(f"Prediction error: {predictions['error']}")
        return {
            'error': predictions['error'],
            'current_price': historical_data['price'].iloc[-1] if 'price' in historical_data.columns else None
        }
    
    # Add percentage changes
    current_price = predictions['current_price']
    predictions['price_change_pct'] = [(p / current_price - 1) * 100 for p in predictions['predictions']]
    
    # Calculate accuracy metrics
    predictions['model_used'] = predictions.get('model', best_model)
    
    # Add confidence level (simplified approach)
    # In a real application, you would use prediction intervals
    predictions['confidence'] = 0.85  # Default value
    
    return predictions

def get_feature_importance(coin_id, timeframe, model_name='random_forest'):
    """Get feature importance from a trained model"""
    try:
        model_path = os.path.join(MODELS_DIR, f'{coin_id}_{timeframe}_{model_name}.joblib')
        
        if not os.path.exists(model_path):
            return {'error': f'Model file not found: {model_path}'}
        
        # Load the model
        model = joblib.load(model_path)
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            # Create a predictor to get feature names
            predictor = CryptoPredictor(coin_id, timeframe)
            
            # Load the model to get feature names
            result = predictor.load_model(model_name)
            if 'error' in result:
                return {'error': result['error']}
                
            # Get the feature names
            feature_names = predictor.features
            
            # Create a dictionary of feature importances
            importances = model.feature_importances_
            feature_importance = dict(zip(feature_names, importances))
            
            # Sort by importance
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return {
                'feature_importance': sorted_importance,
                'top_features': list(sorted_importance.keys())[:10]
            }
        else:
            return {'error': f'Model {model_name} does not provide feature importance'}
    
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        return {'error': str(e)}