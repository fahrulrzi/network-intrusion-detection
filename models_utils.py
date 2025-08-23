"""
Model utilities and custom classes for network intrusion detection
"""

import os
import json
import sys
import joblib
import pickle
import logging
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold

logger = logging.getLogger(__name__)

# ================================================================================================
# CUSTOM TARGET ENCODER CLASSES
# ================================================================================================

class TargetEncoder(BaseEstimator, TransformerMixin):
    """Binary target encoder with cross-validation to prevent target leakage"""
    
    def __init__(self, smoothing=10.0, cv_folds=5):
        self.smoothing = smoothing
        self.cv_folds = cv_folds
        self.encodings = {}  # For test data (from entire training data)
        self.global_mean = None

    def fit(self, X, y):
        """
        Fit encoder on training data
        This will create encoding for test data later
        """
        self.global_mean = y.mean()

        # Calculate encoding based on ENTIRE training data (for test data)
        for column in X.columns:
            if X[column].dtype == 'object':
                encoding_stats = y.groupby(X[column]).agg(['mean', 'count'])
                encoding_stats.columns = ['mean', 'count']

                # Apply smoothing
                smoothed_encoding = (
                    encoding_stats['mean'] * encoding_stats['count'] +
                    self.global_mean * self.smoothing
                ) / (encoding_stats['count'] + self.smoothing)

                self.encodings[column] = smoothed_encoding.to_dict()

        return self

    def transform(self, X):
        """
        Transform data using fitted encodings
        This is for TEST DATA - using encoding from entire training data
        """
        X_encoded = X.copy()

        for column in X.columns:
            if column in self.encodings:
                X_encoded[f'{column}_encoded'] = X[column].map(
                    self.encodings[column]
                ).fillna(self.global_mean)
                X_encoded = X_encoded.drop(column, axis=1)

        return X_encoded

    def fit_transform_with_cv(self, X, y):
        """
        Fit and transform with CV for TRAINING DATA
        This prevents target leakage on training data
        """
        # First fit for test data later
        self.fit(X, y)

        # CV encoding for training data
        X_encoded = X.copy()
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        for column in X.columns:
            if X[column].dtype == 'object':
                encoded_column = np.full(len(X), self.global_mean, dtype=float)

                # Cross-validation encoding for training
                for train_idx, val_idx in kf.split(X):
                    # Data for this fold (NOT including validation fold)
                    X_train_fold = X.iloc[train_idx]
                    y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]

                    # Calculate encoding ONLY from train fold (not validation fold)
                    fold_encoding = y_train_fold.groupby(X_train_fold[column]).agg(['mean', 'count'])
                    fold_encoding.columns = ['mean', 'count']

                    # Apply smoothing
                    smoothed_fold_encoding = (
                        fold_encoding['mean'] * fold_encoding['count'] +
                        self.global_mean * self.smoothing
                    ) / (fold_encoding['count'] + self.smoothing)

                    # Apply encoding to VALIDATION fold
                    for idx in val_idx:
                        category = X.iloc[idx][column]
                        if category in smoothed_fold_encoding:
                            encoded_column[idx] = smoothed_fold_encoding[category]
                        else:
                            encoded_column[idx] = self.global_mean

                X_encoded[f'{column}_encoded'] = encoded_column
                X_encoded = X_encoded.drop(column, axis=1)

        return X_encoded


class MulticlassTargetEncoder(BaseEstimator, TransformerMixin):
    """Multiclass target encoder with probability-based encoding per class"""
    
    def __init__(self, smoothing=10.0, cv_folds=5, random_state=42):
        self.smoothing = smoothing
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.encodings = {}  # For test data
        self.global_probs = {}  # Global probability per class
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit encoder with multi-column probability encoding per target class
        """
        # Save unique classes
        self.classes_ = np.unique(y)

        # Calculate global probability for each class (for smoothing)
        total_count = len(y)
        for class_name in self.classes_:
            self.global_probs[class_name] = np.sum(y == class_name) / total_count

        # Calculate encoding based on ENTIRE training data (for test data)
        for column in X.columns:
            if X[column].dtype == 'object':
                self.encodings[column] = {}

                # For each category in column
                for category in X[column].unique():
                    category_mask = X[column] == category
                    category_targets = y[category_mask]
                    category_count = len(category_targets)

                    # Calculate probability encoding for each class
                    self.encodings[column][category] = {}

                    for class_name in self.classes_:
                        class_count_in_category = np.sum(category_targets == class_name)

                        # Apply smoothing: (actual_count + smoothing * global_prob) / (total_count + smoothing)
                        smoothed_prob = (
                            class_count_in_category + self.smoothing * self.global_probs[class_name]
                        ) / (category_count + self.smoothing)

                        self.encodings[column][category][class_name] = smoothed_prob

        return self

    def transform(self, X):
        """
        Transform data with multi-column probability encoding
        """
        X_encoded = X.copy()

        for column in X.columns:
            if column in self.encodings:
                # Create new column for each class
                for class_name in self.classes_:
                    new_column_name = f'{column}_prob_{class_name}'

                    # Map each category to probability for this class
                    def map_to_prob(cat):
                        if cat in self.encodings[column]:
                            return self.encodings[column][cat][class_name]
                        else:
                            # For unseen categories, use global probability
                            return self.global_probs[class_name]

                    X_encoded[new_column_name] = X[column].apply(map_to_prob)

                # Drop original column
                X_encoded = X_encoded.drop(column, axis=1)

        return X_encoded

    def fit_transform_with_cv(self, X, y):
        """
        Fit and transform with Stratified CV for TRAINING DATA
        Using multi-column probability encoding with smoothing
        """
        # First fit for test data later
        self.fit(X, y)

        # CV encoding for training data with Stratified KFold
        X_encoded = X.copy()
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        for column in X.columns:
            if X[column].dtype == 'object':
                # Initialize columns for each class
                encoded_columns = {}
                for class_name in self.classes_:
                    encoded_columns[class_name] = np.full(len(X), self.global_probs[class_name], dtype=float)

                # Stratified Cross-validation encoding
                for train_idx, val_idx in skf.split(X, y):
                    # Data for this fold (NOT including validation fold)
                    X_train_fold = X.iloc[train_idx]
                    y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]

                    # Calculate fold-specific encoding for each category
                    fold_encodings = {}

                    for category in X_train_fold[column].unique():
                        category_mask = X_train_fold[column] == category
                        category_targets = y_train_fold[category_mask]
                        category_count = len(category_targets)

                        fold_encodings[category] = {}

                        # Calculate probability for each class in this category
                        for class_name in self.classes_:
                            class_count_in_category = np.sum(category_targets == class_name)

                            # Apply smoothing
                            smoothed_prob = (
                                class_count_in_category + self.smoothing * self.global_probs[class_name]
                            ) / (category_count + self.smoothing)

                            fold_encodings[category][class_name] = smoothed_prob

                    # Apply encoding to VALIDATION fold
                    for idx in val_idx:
                        category = X.iloc[idx][column]

                        for class_name in self.classes_:
                            if category in fold_encodings:
                                encoded_columns[class_name][idx] = fold_encodings[category][class_name]
                            else:
                                # Category not found in this fold, use global prob
                                encoded_columns[class_name][idx] = self.global_probs[class_name]

                # Add encoded columns to dataframe
                for class_name in self.classes_:
                    X_encoded[f'{column}_prob_{class_name}'] = encoded_columns[class_name]

                # Drop original column
                X_encoded = X_encoded.drop(column, axis=1)

        return X_encoded


# ================================================================================================
# CUSTOM UNPICKLER FOR HANDLING MODULE IMPORT ISSUES
# ================================================================================================

class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler to handle module path issues when loading models"""
    
    def find_class(self, module, name):
        logger.debug(f"CustomUnpickler.find_class called with module='{module}', name='{name}'")
        
        # Handle models trained in __main__ context
        if module == "__main__":
            # Look for the class in current module (models_utils)
            current_module = sys.modules[__name__]
            if hasattr(current_module, name):
                logger.debug(f"Found {name} in current module (__main__ -> models_utils)")
                return getattr(current_module, name)
        
        # Handle gunicorn context issues
        if "gunicorn" in module:
            # Try to find the class in models_utils module
            try:
                current_module = sys.modules[__name__]
                if hasattr(current_module, name):
                    logger.debug(f"Found {name} in current module (gunicorn -> models_utils)")
                    return getattr(current_module, name)
            except Exception as e:
                logger.debug(f"Failed to find {name} in current module: {e}")
        
        # Try to map known class names to current module regardless of original module
        if name in ['TargetEncoder', 'MulticlassTargetEncoder']:
            try:
                current_module = sys.modules[__name__]
                if hasattr(current_module, name):
                    logger.debug(f"Found {name} in current module (forced mapping)")
                    return getattr(current_module, name)
            except Exception as e:
                logger.debug(f"Forced mapping failed for {name}: {e}")
        
        # Fallback to default behavior
        try:
            result = super().find_class(module, name)
            logger.debug(f"Default find_class succeeded for {module}.{name}")
            return result
        except Exception as e:
            logger.error(f"Default find_class failed for {module}.{name}: {e}")
            raise


def load_model_with_fix(filepath):
    """
    Load model/encoder files with proper unpickling for custom classes
    Uses CustomUnpickler for .pkl files and joblib.load for .joblib files
    """
    logger.debug(f"Loading file: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.endswith('.pkl'):
        # Use custom unpickler for .pkl files (usually encoders/custom classes)
        logger.debug(f"Using CustomUnpickler for .pkl file: {filepath}")
        
        # APPROACH 1: Try CustomUnpickler
        try:
            with open(filepath, 'rb') as f:
                obj = CustomUnpickler(f).load()
            logger.debug(f"SUCCESS: Loaded .pkl file with CustomUnpickler: {filepath}")
            logger.debug(f"Loaded object type: {type(obj)}")
            return obj
        except Exception as e:
            logger.warning(f"CustomUnpickler failed for {filepath}: {e}")
        
        # APPROACH 2: Try with module injection
        logger.debug(f"Trying module injection approach for: {filepath}")
        try:
            # Temporarily inject our classes into __main__ and gunicorn modules
            import sys
            current_module = sys.modules[__name__]
            
            # Get references to our classes
            target_encoder_class = getattr(current_module, 'TargetEncoder', None)
            multiclass_encoder_class = getattr(current_module, 'MulticlassTargetEncoder', None)
            
            # Inject into __main__
            if '__main__' in sys.modules:
                if target_encoder_class:
                    setattr(sys.modules['__main__'], 'TargetEncoder', target_encoder_class)
                if multiclass_encoder_class:
                    setattr(sys.modules['__main__'], 'MulticlassTargetEncoder', multiclass_encoder_class)
            
            # Inject into gunicorn.__main__
            for module_name in sys.modules:
                if 'gunicorn' in module_name and '__main__' in module_name:
                    gunicorn_main = sys.modules[module_name]
                    if target_encoder_class:
                        setattr(gunicorn_main, 'TargetEncoder', target_encoder_class)
                    if multiclass_encoder_class:
                        setattr(gunicorn_main, 'MulticlassTargetEncoder', multiclass_encoder_class)
            
            # Try loading with regular pickle after injection
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
            logger.debug(f"SUCCESS: Loaded .pkl file with module injection: {filepath}")
            logger.debug(f"Loaded object type: {type(obj)}")
            return obj
            
        except Exception as e2:
            logger.warning(f"Module injection approach failed for {filepath}: {e2}")
        
        # APPROACH 3: Regular pickle as last resort
        logger.debug(f"Trying regular pickle as last resort for: {filepath}")
        try:
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
            logger.debug(f"SUCCESS: Loaded .pkl file with regular pickle: {filepath}")
            logger.debug(f"Loaded object type: {type(obj)}")
            return obj
        except Exception as e3:
            logger.error(f"All pickle methods failed for {filepath}")
            logger.error(f"CustomUnpickler error: {e}")
            logger.error(f"Module injection error: {e2}")
            logger.error(f"Regular pickle error: {e3}")
            raise e3
            
    else:
        # Use joblib.load for other files (usually models .joblib)
        logger.debug(f"Using joblib for non-.pkl file: {filepath}")
        try:
            obj = joblib.load(filepath)
            logger.debug(f"SUCCESS: Loaded with joblib: {filepath}")
            logger.debug(f"Loaded object type: {type(obj)}")
            return obj
        except Exception as e:
            logger.error(f"Joblib loading failed for {filepath}: {e}")
            logger.error(f"Joblib full traceback: ", exc_info=True)
            raise e


# ================================================================================================
# MODEL LOADING FUNCTIONS
# ================================================================================================

def load_model_ecosystem(model_dir, model_type='binary'):
    """
    Load all components needed for inference
    Uses load_model_with_fix for proper unpickling of custom classes
    """
    logger.info(f"Loading {model_type} model from: {model_dir}")
    
    if not os.path.exists(model_dir):
        logger.error(f"Model directory not found: {model_dir}")
        return None
    
    components = {}
    
    # Set filenames based on model type
    if model_type == 'binary':
        model_filename = 'binary_classifiers_model.joblib'
        target_encoder_filename = 'target_encoder.pkl'
        config_filename = 'feature_config.json'
    else:  # multiclass
        model_filename = 'multiclass_gradientboosting_model.joblib'
        target_encoder_filename = 'multiclass_target_encoder.pkl'
        config_filename = None  # We'll create config from features
    
    # Load main model
    try:
        model_path = f"{model_dir}/{model_filename}"
        logger.debug(f"Attempting to load model from: {model_path}")
        
        # Early guard: detect Git LFS pointer files (not yet checked out)
        try:
            with open(model_path, 'rb') as _f:
                head = _f.read(120)
            if head.startswith(b'version https://git-lfs.github.com/spec/v1'):
                logger.error(
                    "Model file appears to be a Git LFS pointer, not the real binary. "
                    "Run 'git lfs pull' and 'git lfs checkout' in the repository root to fetch large files."
                )
                return None
        except Exception as e_head:
            logger.debug(f"Could not check LFS pointer header: {e_head}")
        
        # Use load_model_with_fix for consistent loading
        components['model'] = load_model_with_fix(model_path)
        logger.info(f"SUCCESS: {model_type} model loaded from {model_path}")
                
    except Exception as e:
        logger.error(f"ERROR: Loading {model_type} model failed - {e}")
        return None
    
    # Load target encoder using load_model_with_fix
    try:
        encoder_path = f"{model_dir}/{target_encoder_filename}"
        logger.debug(f"Attempting to load target encoder from: {encoder_path}")
        components['target_encoder'] = load_model_with_fix(encoder_path)
        logger.info(f"SUCCESS: {model_type} target encoder loaded")
    except Exception as e:
        logger.error(f"ERROR: Loading {model_type} target encoder failed - {e}")
        logger.error(f"Full traceback: ", exc_info=True)
        return None
    
    # Load or create feature config
    if config_filename and os.path.exists(f"{model_dir}/{config_filename}"):
        try:
            with open(f"{model_dir}/{config_filename}", 'r') as f:
                components['config'] = json.load(f)
            logger.info(f"SUCCESS: {model_type} feature configuration loaded")
        except Exception as e:
            logger.error(f"ERROR: Loading {model_type} feature config failed - {e}")
            return None
    else:
        # For multiclass, try to load the feature_config.json file that should exist
        if model_type == 'multiclass':
            config_path = f"{model_dir}/feature_config.json"
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        components['config'] = json.load(f)
                    logger.info(f"SUCCESS: Loaded multiclass feature configuration from {config_path}")
                except Exception as e:
                    logger.error(f"ERROR: Loading multiclass feature config failed - {e}")
                    return None
            else:
                logger.error(f"ERROR: Multiclass feature config not found at {config_path}")
                return None
        else:
            # Create config from predefined features for binary
            from config import Config
            features = Config.BINARY_FEATURES
            numeric_features = [f for f in features if f not in ['proto', 'service', 'state']]
            categorical_features = ['service', 'state']
            
            # Add proto_encoded to numeric features (since proto gets encoded)
            if 'proto' in features:
                numeric_features.append('proto_encoded')
            
            components['config'] = {
                'numeric_features': numeric_features,
                'categorical_features': categorical_features,
                'original_columns': features
            }
            logger.info(f"SUCCESS: Created {model_type} feature configuration")
    
    # Load label encoder for multiclass if available
    if model_type == 'multiclass':
        try:
            label_encoder_path = f"{model_dir}/label_encoder.pkl"
            if os.path.exists(label_encoder_path):
                components['label_encoder'] = load_model_with_fix(label_encoder_path)
                logger.info("SUCCESS: Label encoder loaded")
        except Exception as e:
            logger.warning(f"Label encoder not found or failed to load: {e}")
        
        # Load preprocessor for multiclass if available
        try:
            preprocessor_path = f"{model_dir}/preprocessor.joblib"
            if os.path.exists(preprocessor_path):
                components['preprocessor'] = load_model_with_fix(preprocessor_path)
                logger.info("SUCCESS: Preprocessor loaded")
            else:
                logger.warning("Preprocessor not found - categorical features may need manual encoding")
        except Exception as e:
            logger.warning(f"Preprocessor loading failed: {e}")
    
    return components


def preprocess_data_for_model(new_data, target_encoder, config, model_type='binary', preprocessor=None):
    """
    Preprocess new data for prediction using target encoder and feature config
    Different preprocessing for binary vs multiclass models
    """
    logger.debug(f"Preprocessing data for {model_type} model")
    
    import pandas as pd
    
    # Ensure we have a DataFrame
    if not isinstance(new_data, pd.DataFrame):
        new_data = pd.DataFrame(new_data)
    
    # Copy data to avoid modifying original
    processed_data = new_data.copy()
    
    if model_type == 'binary':
        # BINARY MODEL PREPROCESSING
        # Get required features from config
        if 'original_columns' in config:
            required_features = config['original_columns']
        else:
            required_features = config.get('numeric_features', []) + config.get('categorical_features', [])
            required_features = [f for f in required_features if not f.endswith('_encoded')]
        
        logger.debug(f"Required features: {required_features}")
        logger.debug(f"Input data columns: {list(processed_data.columns)}")
        
        # Ensure 'sloss' is not used in binary path
        if 'sloss' in processed_data.columns:
            logger.debug("Dropping 'sloss' for binary prediction")
            processed_data = processed_data.drop('sloss', axis=1)

        # Check for missing features
        missing_features = set(required_features) - set(processed_data.columns)
        if missing_features:
            logger.warning(f"WARNING: Missing features: {missing_features}")
            # Fill missing features with defaults
            defaults = {
                'dur': 0.0, 'sloss': 0, 'dloss': 0, 'sinpkt': 0.0, 'dinpkt': 0.0,
                'sjit': 0.0, 'djit': 0.0, 'swin': 0, 'dwin': 0, 'stcpb': 0,
                'dtcpb': 0, 'tcprtt': 0.0, 'synack': 0.0, 'ackdat': 0.0,
                'ct_state_ttl': 1, 'ct_flw_http_mthd': 0, 'ct_ftp_cmd': 0,
                'is_ftp_login': 0, 'is_sm_ips_ports': 0
            }
            
            for feature in missing_features:
                processed_data[feature] = defaults.get(feature, 0.0)
            
            logger.debug("SUCCESS: Missing features filled with defaults")
        
        # Apply target encoding to 'proto' feature only
        if 'proto' in processed_data.columns:
            logger.debug(f"Applying target encoding to 'proto' feature...")
            try:
                proto_df = processed_data[['proto']].copy()
                encoded_df = target_encoder.transform(proto_df)
                
                # Add encoded columns and remove original
                for col in encoded_df.columns:
                    processed_data[col] = encoded_df[col]
                processed_data = processed_data.drop('proto', axis=1)
                
                logger.debug(f"SUCCESS: 'proto' encoded successfully")
            except Exception as e:
                logger.warning(f"Proto encoding failed for binary: {e}")
                processed_data['proto_encoded'] = 0.0  # neutral fallback
                processed_data = processed_data.drop('proto', axis=1, errors='ignore')
        
        # Reorder columns to match expected order
        final_features = required_features.copy()
        if 'proto' in final_features:
            final_features.remove('proto')
            final_features.append('proto_encoded')
        
        available_features = [f for f in final_features if f in processed_data.columns]
        processed_data = processed_data[available_features]
        
    else:
        # MULTICLASS MODEL PREPROCESSING
        logger.debug("Applying multiclass preprocessing pipeline...")
        
        # Debug: Show initial state
        logger.debug(f"Initial columns: {list(processed_data.columns)}")
        
        # Step 1: Apply target encoding to 'proto' if present
        if 'proto' in processed_data.columns:
            logger.debug("Applying target encoding to 'proto'...")
            try:
                proto_df = processed_data[['proto']].copy()
                logger.debug(f"Proto data to encode: {proto_df.values.flatten()}")
                
                encoded_df = target_encoder.transform(proto_df)
                
                # Remove original proto and add encoded columns
                processed_data = processed_data.drop('proto', axis=1)
                for col in encoded_df.columns:
                    processed_data[col] = encoded_df[col]
                
                logger.debug(f"SUCCESS: Proto encoded to {len(encoded_df.columns)} probability columns")
                logger.debug(f"Added columns: {list(encoded_df.columns)}")
                
            except Exception as e:
                logger.error(f"ERROR: Failed to encode proto: {e}")
                # Add default probability columns if encoding fails
                if hasattr(target_encoder, 'classes_'):
                    logger.debug(f"Adding default probability columns for classes: {target_encoder.classes_}")
                    for class_name in target_encoder.classes_:
                        processed_data[f'proto_prob_{class_name}'] = 1.0 / len(target_encoder.classes_)
                processed_data = processed_data.drop('proto', axis=1, errors='ignore')
        else:
            # Add default proto probability columns even if proto column is missing
            logger.debug("Proto column not found - adding default probability columns")
            if hasattr(target_encoder, 'classes_'):
                for class_name in target_encoder.classes_:
                    processed_data[f'proto_prob_{class_name}'] = 1.0 / len(target_encoder.classes_)
                logger.debug(f"Added default probability columns for: {target_encoder.classes_}")
        
        # Debug: Show state after proto encoding
        logger.debug(f"After proto encoding - columns ({len(processed_data.columns)}): {list(processed_data.columns)}")
        
        # Step 2: Add missing sloss column if needed
        if 'sloss' not in processed_data.columns:
            # Check if sloss is in the expected features
            expected_numeric = config.get('numeric_features', [])
            if 'sloss' in expected_numeric:
                processed_data['sloss'] = 0.0  # Default value
                logger.debug("Added missing 'sloss' column with default value 0.0")

        # Ensure 'is_sm_ips_ports' is not used in multiclass path
        if 'is_sm_ips_ports' in processed_data.columns:
            logger.debug("Dropping 'is_sm_ips_ports' for multiclass prediction")
            processed_data = processed_data.drop('is_sm_ips_ports', axis=1)
        
        # Step 3: Apply preprocessor if available
        if preprocessor is not None:
            try:
                logger.debug("Applying sklearn preprocessor...")
                
                # Get expected features from config
                if 'numeric_features' in config and 'categorical_features' in config:
                    expected_numeric = config['numeric_features']
                    expected_categorical = config['categorical_features']
                    expected_features = expected_numeric + expected_categorical
                else:
                    # Fallback - use current columns
                    expected_features = list(processed_data.columns)
                
                logger.debug(f"Expected features ({len(expected_features)}): {expected_features}")
                logger.debug(f"Available features ({len(processed_data.columns)}): {list(processed_data.columns)}")
                
                # Add missing features with defaults
                for feature in expected_features:
                    if feature not in processed_data.columns:
                        if feature in expected_categorical or feature in ['service', 'state', 'is_ftp_login']:
                            processed_data[feature] = 'unknown'  # Default for categorical
                            logger.debug(f"Added missing categorical feature '{feature}' = 'unknown'")
                        else:
                            processed_data[feature] = 0.0  # Default for numeric
                            logger.debug(f"Added missing numeric feature '{feature}' = 0.0")
                
                # Reorder columns to match preprocessor expectations
                processed_data_ordered = processed_data[expected_features]
                
                logger.debug(f"Data ready for preprocessor. Shape: {processed_data_ordered.shape}")
                
                # Apply preprocessor (scaling + one-hot encoding)
                processed_array = preprocessor.transform(processed_data_ordered)
                
                # Convert back to DataFrame for consistency
                processed_data = pd.DataFrame(processed_array)
                
                logger.debug(f"SUCCESS: Preprocessor applied. Shape: {processed_data.shape}")
                
            except Exception as e:
                logger.error(f"ERROR: Preprocessor failed: {e}")
                logger.warning("Falling back to manual categorical encoding...")
                
                # Manual fallback - simple label encoding for categorical features
                from sklearn.preprocessing import LabelEncoder
                categorical_cols = ['service', 'state', 'is_ftp_login']
                
                for col in categorical_cols:
                    if col in processed_data.columns:
                        le = LabelEncoder()
                        # Use a predefined set of common values
                        if col == 'service':
                            le.fit(['http', 'ftp', 'smtp', 'dns', '-', 'unknown'])
                        elif col == 'state':
                            le.fit(['FIN', 'CON', 'RST', 'REQ', 'INT', 'unknown'])
                        else:  # is_ftp_login
                            le.fit([0, 1])
                        
                        try:
                            processed_data[col] = le.transform(processed_data[col])
                        except ValueError:
                            # Handle unseen categories
                            processed_data[col] = processed_data[col].map(
                                lambda x: le.transform(['unknown'])[0] if x not in le.classes_ else le.transform([x])[0]
                            )
                        
                        logger.debug(f"Manual encoding applied to '{col}'")
        else:
            logger.warning("No preprocessor available - manual encoding may be needed")
    
    logger.debug(f"Final processed data shape: {processed_data.shape}")
    logger.debug(f"Final features: {list(processed_data.columns)}")
    
    return processed_data