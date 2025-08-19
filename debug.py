#!/usr/bin/env python3
"""
Debug script for Cyber Attack Detection System
Helps identify and troubleshoot issues with the model and application
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import traceback
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """Check Python environment and dependencies"""
    print("=" * 60)
    print("🔍 ENVIRONMENT CHECK")
    print("=" * 60)
    
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.executable}")
    
    # Check required packages
    required_packages = [
        'flask', 'pandas', 'numpy', 'scikit-learn', 
        'xgboost', 'joblib'
    ]
    
    print("\n📦 Package versions:")
    for package in required_packages:
        try:
            if package == 'scikit-learn':
                import sklearn
                print(f"  ✅ {package}: {sklearn.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'Unknown')
                print(f"  ✅ {package}: {version}")
        except ImportError as e:
            print(f"  ❌ {package}: Not installed ({e})")
    print()

def check_files():
    """Check if required files exist"""
    print("=" * 60)
    print("📁 FILE CHECK")
    print("=" * 60)
    
    required_files = [
        'app.py',
        'templates/index.html',
        'requirements.txt',
        'xgboost_model_20250722_030754.pkl'
    ]
    
    current_files = []
    try:
        current_files = os.listdir('.')
        if 'templates' in current_files:
            template_files = os.listdir('templates')
            current_files.extend([f'templates/{f}' for f in template_files])
    except Exception as e:
        print(f"Error listing files: {e}")
    
    print("📋 Required files:")
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  ✅ {file} ({size:,} bytes)")
        else:
            print(f"  ❌ {file} (missing)")
    
    print(f"\n📋 All files in directory:")
    for file in sorted(current_files):
        if os.path.isfile(file):
            size = os.path.getsize(file)
            print(f"  📄 {file} ({size:,} bytes)")
        elif os.path.isdir(file):
            print(f"  📁 {file}/")
    print()

def test_model_loading():
    """Test model loading and basic functionality"""
    print("=" * 60)
    print("🧠 MODEL LOADING TEST")
    print("=" * 60)
    
    model_path = 'xgboost_model_20250722_030754.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        
        # Check for alternative model files
        alt_models = ['xgboost_model_fixed.pkl', 'xgboost_model_20250722_030754.joblib']
        for alt_model in alt_models:
            if os.path.exists(alt_model):
                print(f"🔄 Found alternative model: {alt_model}")
                model_path = alt_model
                break
        else:
            return None
    
    try:
        print(f"📂 Loading model from: {model_path}")
        
        # Try with warnings suppressed first
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model type: {type(model)}")
        
        # Check if it's a pipeline
        if hasattr(model, 'named_steps'):
            print(f"   Pipeline steps: {list(model.named_steps.keys())}")
            
            # Check preprocessor
            if 'preprocessor' in model.named_steps:
                preprocessor = model.named_steps['preprocessor']
                print(f"   Preprocessor: {type(preprocessor)}")
                if hasattr(preprocessor, 'transformers'):
                    print(f"   Transformers: {[t[0] for t in preprocessor.transformers]}")
            
            # Check classifier
            if 'classifier' in model.named_steps:
                classifier = model.named_steps['classifier']
                print(f"   Classifier: {type(classifier)}")
        
        # Test basic methods
        print("\n🧪 Testing model methods:")
        if hasattr(model, 'predict'):
            print("   ✅ predict method available")
        else:
            print("   ❌ predict method not available")
            
        if hasattr(model, 'predict_proba'):
            print("   ✅ predict_proba method available")
        else:
            print("   ❌ predict_proba method not available")
        
        return model
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error loading model: {e}")
        
        # Check for specific sklearn version issues
        if "InconsistentVersionWarning" in error_msg or "_RemainderColsList" in error_msg:
            print(f"\n🔧 SKLEARN VERSION ISSUE DETECTED!")
            print(f"   This is a scikit-learn version compatibility problem.")
            print(f"   The model was saved with a different version of scikit-learn.")
            print(f"\n   SOLUTIONS:")
            print(f"   1. Run: pip install scikit-learn==1.6.1")
            print(f"   2. Or regenerate model: python fix_model.py")
            print(f"   3. Or use the app anyway (may have issues)")
            
        print(f"   Traceback: {traceback.format_exc()}")
        return None

def create_test_data():
    """Create test data for prediction"""
    print("\n🔬 Creating test data...")
    
    # Sample normal traffic data
    test_data = {
        'dur': 0.12,
        'proto': 'tcp',
        'service': 'http',
        'state': 'FIN',
        'spkts': 8,
        'dpkts': 6,
        'sbytes': 490,
        'dbytes': 1337,
        'rate': 66.67,
        'sttl': 254,
        'dttl': 63,
        'sload': 4083.33,
        'dload': 11141.67,
        'sloss': 0,
        'dloss': 0,
        'sinpkt': 65,
        'dinpkt': 65,
        'sjit': 0.01,
        'djit': 0.03,
        'swin': 255,
        'dwin': 255,
        'stcpb': 0,
        'dtcpb': 0,
        'smean': 61,
        'dmean': 222,
        'trans_depth': 2,
        'response_body_len': 671,
        'tcprtt': 0.15,
        'synack': 0.07,
        'ackdat': 0.05,
        'ct_state_ttl': 1,
        'ct_flw_http_mthd': 7,
        'is_ftp_login': 0,
        'ct_ftp_cmd': 0,
        'ct_srv_src': 1,
        'ct_srv_dst': 1,
        'ct_dst_ltm': 1,
        'ct_src_ltm': 1,
        'ct_src_dport_ltm': 1,
        'ct_dst_sport_ltm': 1,
        'ct_dst_src_ltm': 1,
        'is_sm_ips_ports': 0
    }
    
    print(f"   Test data created with {len(test_data)} features")
    return test_data

def feature_engineering(df):
    """Apply the same feature engineering as in the notebook"""
    try:
        print("   🔧 Applying feature engineering...")
        
        # Add Count column (always 1 for single record)
        df['Count'] = 1
        
        # Ratios
        df['byte_ratio'] = df['sbytes'] / (df['dbytes'] + 1)
        df['pkt_ratio'] = df['spkts'] / (df['dpkts'] + 1)
        df['load_ratio'] = df['sload'] / (df['dload'] + 1)
        df['jit_ratio'] = df['sjit'] / (df['djit'] + 1)
        df['tcp_setup_ratio'] = df['tcprtt'] / (df['synack'] + df['ackdat'] + 1)

        # Aggregate Features
        df['total_bytes'] = df['sbytes'] + df['dbytes']
        df['total_pkts'] = df['spkts'] + df['dpkts']
        df['total_load'] = df['sload'] + df['dload']
        df['total_jitter'] = df['sjit'] + df['djit']
        df['total_tcp_setup'] = df['tcprtt'] + df['synack'] + df['ackdat']

        # Interaction Features
        df['byte_pkt_interaction_src'] = df['sbytes'] * df['spkts']
        df['byte_pkt_interaction_dst'] = df['dbytes'] * df['dpkts']
        df['load_jit_interaction_src'] = df['sload'] * df['sjit']
        df['load_jit_interaction_dst'] = df['dload'] * df['djit']
        df['pkt_jit_interaction_src'] = df['spkts'] * df['sjit']
        df['pkt_jit_interaction_dst'] = df['dpkts'] * df['djit']

        # Statistical Features
        df['tcp_seq_diff'] = df['stcpb'] - df['dtcpb']
        
        print(f"   ✅ Feature engineering completed. Shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"   ❌ Error in feature engineering: {e}")
        raise e

def test_prediction(model):
    """Test model prediction with sample data"""
    if model is None:
        print("❌ Cannot test prediction - model not loaded")
        return False
    
    print("\n🎯 PREDICTION TEST")
    print("-" * 40)
    
    try:
        # Create test data
        test_data = create_test_data()
        
        # Convert to DataFrame
        print("   📊 Converting to DataFrame...")
        input_df = pd.DataFrame([test_data])
        print(f"   DataFrame shape: {input_df.shape}")
        print(f"   DataFrame columns: {len(input_df.columns)}")
        
        # Apply feature engineering
        input_df = feature_engineering(input_df)
        print(f"   After feature engineering: {input_df.shape}")
        
        # Check for issues
        if input_df.isnull().any().any():
            print("   ⚠️  DataFrame contains NaN values")
            nan_cols = input_df.columns[input_df.isnull().any()].tolist()
            print(f"      NaN columns: {nan_cols}")
            input_df = input_df.fillna(0)
            print("   🔧 Filled NaN values with 0")
        
        if np.isinf(input_df.select_dtypes(include=[np.number])).any().any():
            print("   ⚠️  DataFrame contains infinite values")
            input_df = input_df.replace([np.inf, -np.inf], 0)
            print("   🔧 Replaced infinite values with 0")
        
        # Make prediction
        print("   🎯 Making prediction...")
        prediction = model.predict(input_df)[0]
        print(f"   Prediction: {prediction}")
        
        prediction_proba = model.predict_proba(input_df)[0]
        print(f"   Probabilities: {prediction_proba}")
        
        result_text = 'Attack Detected' if prediction == 1 else 'Normal Traffic'
        confidence = max(prediction_proba)
        
        print(f"   ✅ Result: {result_text} (confidence: {confidence:.3f})")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in prediction: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_flask_import():
    """Test Flask and related imports"""
    print("=" * 60)
    print("🌐 FLASK IMPORT TEST")
    print("=" * 60)
    
    try:
        from flask import Flask, render_template, request, jsonify
        print("✅ Flask imports successful")
        
        # Test creating Flask app
        test_app = Flask(__name__)
        print("✅ Flask app creation successful")
        
        return True
    except Exception as e:
        print(f"❌ Flask import error: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🛡️  CYBER ATTACK DETECTION SYSTEM - DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Run all tests
    check_environment()
    check_files()
    test_flask_import()
    model = test_model_loading()
    test_prediction(model)
    
    print("=" * 60)
    print("🏁 DIAGNOSTIC COMPLETE")
    print("=" * 60)
    
    if model is not None:
        print("✅ All systems appear to be working correctly!")
        print("   You can now run: python app.py")
    else:
        print("❌ Issues detected. Please fix the problems above before running the app.")
    
    print("\n💡 TROUBLESHOOTING TIPS:")
    print("   - Make sure all required files are in the correct location")
    print("   - Check that the model file is not corrupted")
    print("   - Verify all dependencies are installed: pip install -r requirements.txt")
    print("   - Check the app.log file for detailed error messages")
    print("   - If you're still having issues, check the specific error messages above")

if __name__ == "__main__":
    main()
