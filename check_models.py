import os
import pickle
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin

# ================================================================================================
# CUSTOM TARGET ENCODER CLASS - MUST BE DEFINED BEFORE LOADING PICKLE
# ================================================================================================

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=10.0, cv_folds=5):
        self.smoothing = smoothing
        self.cv_folds = cv_folds
        self.encodings = {}  # Untuk test data (dari seluruh training data)
        self.global_mean = None

    def fit(self, X, y):
        """
        Fit encoder pada training data
        Ini akan membuat encoding untuk test data nanti
        """
        self.global_mean = y.mean()

        # Hitung encoding berdasarkan SELURUH training data (untuk test data)
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
        Transform data menggunakan fitted encodings
        Ini untuk TEST DATA - menggunakan encoding dari seluruh training data
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
        Fit dan transform dengan CV untuk TRAINING DATA
        Ini mencegah target leakage pada training data
        """
        # Pertama fit untuk test data nanti
        self.fit(X, y)

        # CV encoding untuk training data
        X_encoded = X.copy()
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)  # Fixed seed

        for column in X.columns:
            if X[column].dtype == 'object':
                encoded_column = np.full(len(X), self.global_mean, dtype=float)

                # Cross-validation encoding untuk training
                for train_idx, val_idx in kf.split(X):
                    # Data untuk fold ini (TIDAK termasuk validation fold)
                    X_train_fold = X.iloc[train_idx]
                    y_train_fold = y.iloc[train_idx]

                    # Hitung encoding HANYA dari train fold (bukan validation fold)
                    fold_encoding = y_train_fold.groupby(X_train_fold[column]).agg(['mean', 'count'])
                    fold_encoding.columns = ['mean', 'count']

                    # Apply smoothing
                    smoothed_fold_encoding = (
                        fold_encoding['mean'] * fold_encoding['count'] +
                        self.global_mean * self.smoothing
                    ) / (fold_encoding['count'] + self.smoothing)

                    # Apply encoding ke VALIDATION fold
                    for idx in val_idx:
                        category = X.iloc[idx][column]
                        if category in smoothed_fold_encoding:
                            encoded_column[idx] = smoothed_fold_encoding[category]
                        else:
                            encoded_column[idx] = self.global_mean

                X_encoded[f'{column}_encoded'] = encoded_column
                X_encoded = X_encoded.drop(column, axis=1)

        return X_encoded

def check_model_files():
    """Check if all model files exist and can be loaded"""
    
    model_dir = 'models'
    
    print("="*60)
    print("MODEL FILES DIAGNOSTIC")
    print("="*60)
    
    # Check if directory exists
    if not os.path.exists(model_dir):
        print(f"❌ Model directory '{model_dir}' does not exist")
        return False
    
    print(f"✅ Model directory '{model_dir}' exists")
    print(f"Files in {model_dir}: {os.listdir(model_dir)}")
    
    # Check each required file
    required_files = [
        'binary_classifiers_model.joblib',
        'target_encoder.pkl', 
        'feature_config.json'
    ]
    
    all_files_ok = True
    
    for filename in required_files:
        filepath = os.path.join(model_dir, filename)
        print(f"\nChecking: {filename}")
        
        if not os.path.exists(filepath):
            print(f"❌ File {filename} does not exist")
            all_files_ok = False
            continue
        
        print(f"✅ File {filename} exists ({os.path.getsize(filepath)} bytes)")
        
        # Try to load each file
        try:
            if filename.endswith('.joblib'):
                model = joblib.load(filepath)
                print(f"✅ Successfully loaded {filename}")
                print(f"   Model type: {type(model)}")
                if hasattr(model, 'named_steps'):
                    print(f"   Pipeline steps: {list(model.named_steps.keys())}")
                    
            elif filename.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    encoder = pickle.load(f)
                print(f"✅ Successfully loaded {filename}")
                print(f"   Encoder type: {type(encoder)}")
                if hasattr(encoder, 'encodings'):
                    print(f"   Encodings keys: {list(encoder.encodings.keys())}")
                    if 'proto' in encoder.encodings:
                        proto_encodings = encoder.encodings['proto']
                        print(f"   Proto categories: {len(proto_encodings)} categories")
                        # Show first few encodings
                        sample_encodings = list(proto_encodings.items())[:5]
                        for cat, enc in sample_encodings:
                            print(f"     {cat}: {enc:.4f}")
                if hasattr(encoder, 'global_mean'):
                    print(f"   Global mean: {encoder.global_mean:.4f}")
                    
            elif filename.endswith('.json'):
                with open(filepath, 'r') as f:
                    config = json.load(f)
                print(f"✅ Successfully loaded {filename}")
                print(f"   Config keys: {list(config.keys())}")
                if 'numeric_features' in config:
                    print(f"   Numeric features: {len(config['numeric_features'])}")
                if 'categorical_features' in config:
                    print(f"   Categorical features: {len(config['categorical_features'])}")
                if 'model_info' in config:
                    print(f"   Model algorithm: {config['model_info'].get('algorithm', 'Unknown')}")
                
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            all_files_ok = False
    
    print("\n" + "="*60)
    if all_files_ok:
        print("✅ ALL MODEL FILES OK - Ready for deployment")
        print("\n🎯 NEXT STEPS:")
        print("1. Run: python app.py")
        print("2. Open browser: http://localhost:5000")
        print("3. Test predictions with sample data")
    else:
        print("❌ SOME MODEL FILES HAVE ISSUES - Check above errors")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Make sure TargetEncoder class is defined before loading pickle")
        print("2. Verify all model files are present and not corrupted")
        print("3. Check if you need to retrain and save the model")
    
    return all_files_ok

def test_prediction_pipeline():
    """Test the complete prediction pipeline"""
    print("\n" + "="*60)
    print("TESTING PREDICTION PIPELINE")
    print("="*60)
    
    try:
        # Load all components
        model = joblib.load('models/binary_classifiers_model.joblib')
        with open('models/target_encoder.pkl', 'rb') as f:
            target_encoder = pickle.load(f)
        with open('models/feature_config.json', 'r') as f:
            config = json.load(f)
        
        print("✅ All model components loaded successfully")
        
        # Create test data
        test_data = pd.DataFrame([{
            'dur': 0.12, 'proto': 'tcp', 'service': 'http', 'state': 'FIN',
            'spkts': 8, 'dpkts': 6, 'sbytes': 490, 'dbytes': 1337, 'rate': 66.67,
            'sttl': 254, 'dttl': 63, 'sload': 4083.33, 'dload': 11141.67, 'dloss': 0,
            'sinpkt': 65, 'dinpkt': 65, 'sjit': 0.01, 'djit': 0.03, 'swin': 255,
            'stcpb': 0, 'dtcpb': 0, 'dwin': 255, 'tcprtt': 0.15, 'synack': 0.07,
            'ackdat': 0.05, 'smean': 61, 'dmean': 222, 'trans_depth': 2,
            'response_body_len': 671, 'ct_srv_src': 1, 'ct_state_ttl': 1,
            'ct_dst_ltm': 1, 'ct_src_dport_ltm': 1, 'ct_dst_sport_ltm': 1,
            'ct_dst_src_ltm': 1, 'is_ftp_login': 0, 'ct_ftp_cmd': 0,
            'ct_flw_http_mthd': 7, 'ct_src_ltm': 1, 'ct_srv_dst': 1, 'is_sm_ips_ports': 0
        }])
        
        print("✅ Test data created")
        
        # Apply target encoding
        if 'proto' in test_data.columns:
            proto_df = test_data[['proto']]
            proto_encoded = target_encoder.transform(proto_df)
            test_data = test_data.drop('proto', axis=1)
            test_data['proto_encoded'] = proto_encoded['proto_encoded']
            print("✅ Target encoding applied")
        
        # Remove sloss if present
        if 'sloss' in test_data.columns:
            test_data = test_data.drop('sloss', axis=1)
        
        # Select required features
        required_features = config['numeric_features'] + config['categorical_features']
        missing_features = set(required_features) - set(test_data.columns)
        
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
            for feature in missing_features:
                if feature in config['numeric_features']:
                    test_data[feature] = 0
                else:
                    test_data[feature] = 'unknown'
        
        test_data = test_data[required_features]
        print(f"✅ Data preprocessed - shape: {test_data.shape}")
        
        # Make prediction
        prediction = model.predict(test_data)[0]
        probabilities = model.predict_proba(test_data)[0]
        
        result = "🚨 ATTACK DETECTED" if prediction == 1 else "✅ NORMAL TRAFFIC"
        confidence = probabilities[prediction]
        
        print(f"\n🎯 PREDICTION RESULT:")
        print(f"   Prediction: {result}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Probabilities: Normal={probabilities[0]:.3f}, Attack={probabilities[1]:.3f}")
        
        print("\n✅ PREDICTION PIPELINE TEST SUCCESSFUL")
        return True
        
    except Exception as e:
        print(f"❌ Prediction pipeline test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    # Check model files
    files_ok = check_model_files()
    
    # If files are OK, test prediction pipeline
    if files_ok:
        test_prediction_pipeline()