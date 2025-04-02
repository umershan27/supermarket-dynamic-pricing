import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import pickle
import pandas as pd

class PricingModel:
    def __init__(self):
        self.model = None
        self.model_path = '/app/data/xgboost_model.pkl'
        self.model_features = None
        self._metrics = {
            "rmse": 0.0,
            "mae": 0.0,
            "r2": 0.0,
            "accuracy_within_5_percent": 0.0,
            "accuracy_within_10_percent": 0.0,
            "accuracy_within_20_percent": 0.0
        }

    @property
    def metrics(self):
        return self._metrics

    def train(self, combined_df, features, target, validation_size=None):
        try:
            print(f"Training model with features: {features}")
            print(f"Target: {target}")
            print(f"Combined DF shape: {combined_df.shape}")
            print(f"Combined DF columns: {combined_df.columns.tolist()}")
            
            # Store the features used in training
            self.model_features = features
            
            # Check if all features exist in dataframe
            missing_features = [f for f in features if f not in combined_df.columns]
            if missing_features:
                print(f"Warning: Missing features: {missing_features}")
                # Ensure all features exist
                for feat in missing_features:
                    combined_df[feat] = 0
            
            # Check if target exists
            if target not in combined_df.columns:
                print(f"Warning: Target {target} not in columns. Adding dummy values.")
                combined_df[target] = 0.0
            
            X = combined_df[features].copy()
            y = combined_df[target].copy()

            # Clean data
            X = X.fillna(0)
            y = y.fillna(0)
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            X = X.replace([np.inf, -np.inf], 0).astype('float32')
            y = y.astype('float32')

            # Determine validation size - use larger validation set for better evaluation
            if validation_size is None or validation_size <= 0:
                validation_size = min(int(len(X) * 0.2), 20000)  # Default to 20% with max of 20K samples
            else:
                validation_size = min(validation_size, len(X) - 1000)  # Ensure we have enough training data
            
            print(f"Using validation size: {validation_size}")
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_size, random_state=42
            )

            # Enhanced parameters for better accuracy
            fixed_params = {
                'learning_rate': 0.03,        # Reduced from 0.05 for better generalization
                'max_depth': 6,               # Increased from 5 for more complex patterns
                'subsample': 0.85,            # Increased from 0.8 for better stability
                'colsample_bytree': 0.8,      # Increased from 0.7 for better feature usage
                'n_estimators': 150,          # Increased from 100 for better convergence
                'reg_alpha': 0.3,             # Reduced from 0.5 for less L1 regularization
                'reg_lambda': 2.0,            # Adjusted from 2.5 for less L2 regularization
                'min_child_weight': 3,        # Added to prevent overfitting
                'gamma': 0.1                  # Added to prevent overfitting
            }

            print("Training XGBoost model with enhanced hyperparameters...")
            self.model = xgb.XGBRegressor(objective='reg:squarederror',
                                      tree_method='hist',
                                      random_state=42,
                                      **fixed_params)
            self.model.fit(X_train, y_train, 
                          eval_set=[(X_val, y_val)], 
                          early_stopping_rounds=15,
                          verbose=True)

            # Calculate performance metrics
            print("Calculating model metrics...")
            
            # Validation predictions
            y_pred_val = self.model.predict(X_val)
            
            # RMSE
            rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
            mae_val = mean_absolute_error(y_val, y_pred_val)
            r2_val = r2_score(y_val, y_pred_val)
            print(f"Validation RMSE: {rmse_val:.4f}")
            print(f"Validation MAE: {mae_val:.4f}")
            print(f"Validation R²: {r2_val:.4f}")
            
            # Calculate percentage of predictions within X% of actual values
            within_5_percent = np.mean(np.abs(y_pred_val - y_val) <= 0.05 * y_val)
            within_10_percent = np.mean(np.abs(y_pred_val - y_val) <= 0.10 * y_val)
            within_20_percent = np.mean(np.abs(y_pred_val - y_val) <= 0.20 * y_val)
            
            print(f"Accuracy within 5% of true value: {within_5_percent:.2%}")
            print(f"Accuracy within 10% of true value: {within_10_percent:.2%}")
            print(f"Accuracy within 20% of true value: {within_20_percent:.2%}")
            
            # Store metrics
            self._metrics = {
                "rmse": float(rmse_val),
                "mae": float(mae_val),
                "r2": float(r2_val),
                "accuracy_within_5_percent": float(within_5_percent),
                "accuracy_within_10_percent": float(within_10_percent),
                "accuracy_within_20_percent": float(within_20_percent),
                "validation_size": validation_size,
                "training_size": len(X_train)
            }
            
            # Get feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                self._metrics["feature_importance"] = {
                    features[i]: float(importance[i])
                    for i in range(len(features))
                }
            
            # Save the model and features
            self.save_model(self.model_path)
            
        except Exception as e:
            print(f"Error during training: {e}")
            # Create a dummy model that returns the mean value
            print("Creating a fallback model")
            from sklearn.dummy import DummyRegressor
            self.model = DummyRegressor(strategy='mean')
            self.model.fit(np.array([[0, 0]]), np.array([10]))
            self.model_features = ['dummy1', 'dummy2']
            self._metrics = {"error": str(e)}

    def predict(self, X_dynamic):
        try:
            if self.model is None:
                print("No model available. Creating dummy model.")
                from sklearn.dummy import DummyRegressor
                self.model = DummyRegressor(strategy='mean')
                self.model.fit(np.array([[0, 0]]), np.array([10]))
                self.model_features = ['dummy1', 'dummy2']
                
            print(f"Input shape: {X_dynamic.shape}")
            print(f"Input columns: {X_dynamic.columns.tolist()}")
            
            # Make sure we have all required features with the right names
            if self.model_features is not None:
                # Map features if names don't match but they're clearly the same thing
                feature_mapping = {
                    'godown_code': 'godown_code_encoded',
                    'godown_code_encoded': 'godown_code'
                }
                
                # Create a copy to avoid modifying the original
                X_prepared = X_dynamic.copy()
                
                # Handle missing or mismatched columns
                for feat in self.model_features:
                    if feat not in X_prepared.columns:
                        # Check if there's a differently named column that means the same
                        mapped_name = None
                        for original, mapped in feature_mapping.items():
                            if mapped == feat and original in X_prepared.columns:
                                mapped_name = original
                                break
                        
                        if mapped_name:
                            # Rename the column to match what the model expects
                            X_prepared[feat] = X_prepared[mapped_name]
                            print(f"Mapped {mapped_name} to {feat} for model compatibility")
                        else:
                            # If no mapping found, add a zero column
                            print(f"Adding missing feature {feat} with zeros")
                            X_prepared[feat] = 0
                
                # Reorder columns to match the model's expectations
                X_prepared = X_prepared[self.model_features]
            else:
                X_prepared = X_dynamic
            
            # Clean input data
            X_clean = X_prepared.fillna(0)
            for col in X_clean.columns:
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce').fillna(0)
            X_clean = X_clean.replace([np.inf, -np.inf], 0).astype('float32')
                
            predicted_prices = self.model.predict(X_clean)
            
            # Apply business rules - convert to numpy arrays first
            if 'cost_rate' in X_dynamic.columns and 'mrp' in X_dynamic.columns:
                cost_rates = X_dynamic['cost_rate'].values
                mrps = X_dynamic['mrp'].values
                
                # Apply element-wise clipping
                min_prices = cost_rates * 1.05  # Minimum 5% margin
                max_prices = mrps * 0.95  # Max 95% of MRP
                
                for i in range(len(predicted_prices)):
                    if predicted_prices[i] < min_prices[i]:
                        predicted_prices[i] = min_prices[i]
                    elif predicted_prices[i] > max_prices[i]:
                        predicted_prices[i] = max_prices[i]
                
            return predicted_prices
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Return a reasonable fallback based on retail_rate
            if 'retail_rate' in X_dynamic.columns:
                return X_dynamic['retail_rate'].values * np.random.uniform(0.95, 1.05, size=len(X_dynamic))
            return np.array([10.0] * len(X_dynamic))

    def save_model(self, model_path):
        try:
            # Save model, features, and metrics
            with open(model_path, "wb") as f:
                pickle.dump({
                    'model': self.model,
                    'features': self.model_features,
                    'metrics': self._metrics
                }, f)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, "rb") as f:
                    model_data = pickle.load(f)
                    if isinstance(model_data, dict):
                        self.model = model_data.get('model')
                        self.model_features = model_data.get('features')
                        self._metrics = model_data.get('metrics', {})
                    else:
                        # Compatibility with older saved models
                        self.model = model_data
                        self._metrics = {}
                print("Model loaded successfully")
                return True
            else:
                print(f"Model file {self.model_path} does not exist")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def train_and_save(self, X, y):
        try:
            # Store features
            self.model_features = X.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Initialize XGBoost model
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )

            # Train model
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10,
                verbose=True
            )

            # Calculate RMSE
            predictions = self.model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            print(f"Model RMSE: {rmse:.2f}")

            # Save model
            self.save_model(self.model_path)
            return True

        except Exception as e:
            print(f"Error training model: {e}")
            return False

    def predict_from_file(self, X):
        try:
            # Load model if not loaded
            self.load_model()

            if self.model is None:
                raise Exception("Model not trained")

            # Make predictions
            predictions = self.model.predict(X)
            
            # Apply business rules
            predictions = np.clip(
                predictions,
                X['cost_rate'] * 1.1,  # Minimum 10% margin
                X['mrp']  # Maximum MRP
            )
            
            return predictions

        except Exception as e:
            print(f"Error making predictions: {e}")
            return None 