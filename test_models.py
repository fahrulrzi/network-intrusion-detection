#!/usr/bin/env python3
"""
Test script for checking model loading and compatibility
"""

import os
import sys
import traceback
import joblib
import pickle
import json
import numpy as np
import pandas as pd

# Import the target encoder classes to make them available for pickle loading
from models_utils import TargetEncoder, MulticlassTargetEncoder

def test_model_loading():
    """Test loading both binary and multiclass models"""
    
    print("🔍 Testing Model Loading...")
    print("=" * 50)
    
    # Test binary model
    binary_dir = "models/binary"
    print(f"\n📁 Testing Binary Model from: {binary_dir}")
    
    if os.path.exists(binary_dir):
        try:
            # Test model loading
            model_path = f"{binary_dir}/binary_classifiers_model.joblib"
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                print(f"✅ Binary model loaded successfully: {type(model)}")
            else:
                print(f"❌ Binary model file not found: {model_path}")
            
            # Test target encoder loading
            encoder_path = f"{binary_dir}/target_encoder.pkl"
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    encoder = pickle.load(f)
                print(f"✅ Binary target encoder loaded successfully: {type(encoder)}")
            else:
                print(f"❌ Binary target encoder not found: {encoder_path}")
            
            # Test config loading
            config_path = f"{binary_dir}/feature_config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"✅ Binary config loaded successfully")
                print(f"   Numeric features: {len(config.get('numeric_features', []))}")
                print(f"   Categorical features: {len(config.get('categorical_features', []))}")
            else:
                print(f"❌ Binary config not found: {config_path}")
                
        except Exception as e:
            print(f"❌ Error loading binary model: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
    else:
        print(f"❌ Binary model directory not found: {binary_dir}")
    
    # Test multiclass model
    multiclass_dir = "models/multiclass"
    print(f"\n📁 Testing Multiclass Model from: {multiclass_dir}")
    
    if os.path.exists(multiclass_dir):
        try:
            # List files in directory
            files = os.listdir(multiclass_dir)
            print(f"   Files in directory: {files}")
            
            # Test model loading with different methods
            model_path = f"{multiclass_dir}/multiclass_gradientboosting_model.joblib"
            if os.path.exists(model_path):
                try:
                    # Try joblib first
                    model = joblib.load(model_path)
                    print(f"✅ Multiclass model loaded with joblib: {type(model)}")
                except Exception as e1:
                    print(f"⚠️ joblib loading failed: {e1}")
                    try:
                        # Try pickle
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        print(f"✅ Multiclass model loaded with pickle: {type(model)}")
                    except Exception as e2:
                        print(f"❌ Both joblib and pickle failed")
                        print(f"   joblib error: {e1}")
                        print(f"   pickle error: {e2}")
            else:
                print(f"❌ Multiclass model file not found: {model_path}")
            
            # Test target encoder loading
            encoder_path = f"{multiclass_dir}/multiclass_target_encoder.pkl"
            if os.path.exists(encoder_path):
                try:
                    encoder = joblib.load(encoder_path)
                    print(f"✅ Multiclass target encoder loaded with joblib: {type(encoder)}")
                except Exception as e1:
                    try:
                        with open(encoder_path, 'rb') as f:
                            encoder = pickle.load(f)
                        print(f"✅ Multiclass target encoder loaded with pickle: {type(encoder)}")
                    except Exception as e2:
                        print(f"❌ Multiclass target encoder loading failed")
                        print(f"   joblib error: {e1}")
                        print(f"   pickle error: {e2}")
            else:
                print(f"❌ Multiclass target encoder not found: {encoder_path}")
            
            # Test label encoder loading
            label_encoder_path = f"{multiclass_dir}/label_encoder.pkl"
            if os.path.exists(label_encoder_path):
                try:
                    label_encoder = joblib.load(label_encoder_path)
                    print(f"✅ Label encoder loaded with joblib: {type(label_encoder)}")
                    print(f"   Classes: {label_encoder.classes_}")
                except Exception as e1:
                    try:
                        with open(label_encoder_path, 'rb') as f:
                            label_encoder = pickle.load(f)
                        print(f"✅ Label encoder loaded with pickle: {type(label_encoder)}")
                        print(f"   Classes: {label_encoder.classes_}")
                    except Exception as e2:
                        print(f"❌ Label encoder loading failed")
                        print(f"   joblib error: {e1}")
                        print(f"   pickle error: {e2}")
            else:
                print(f"❌ Label encoder not found: {label_encoder_path}")
                
        except Exception as e:
            print(f"❌ Error testing multiclass model: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
    else:
        print(f"❌ Multiclass model directory not found: {multiclass_dir}")
    
    print("\n" + "=" * 50)
    print("🏁 Model Loading Test Complete")

def test_prediction_pipeline():
    """Test a simple prediction to ensure everything works"""
    
    print("\n🧪 Testing Prediction Pipeline...")
    print("=" * 50)
    
    # Create sample data
    sample_data = {
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
    }
    
    # Try to import and run the app functions
    try:
        from models_utils import load_model_ecosystem, preprocess_data_for_model
        from config import Config
        
        print("✅ Successfully imported app functions")
        
        # Test binary model loading
        print("\n🔧 Testing binary model ecosystem...")
        binary_components = load_model_ecosystem("models/binary", "binary")
        if binary_components:
            print("✅ Binary model ecosystem loaded successfully")
            
            # Test preprocessing
            df = pd.DataFrame([sample_data])
            processed_data = preprocess_data_for_model(
                df, 
                binary_components['target_encoder'], 
                binary_components['config'], 
                'binary'
            )
            print(f"✅ Binary preprocessing successful. Shape: {processed_data.shape}")
            
            # Test prediction
            pred = binary_components['model'].predict(processed_data)
            proba = binary_components['model'].predict_proba(processed_data)
            print(f"✅ Binary prediction successful. Result: {pred[0]}, Confidence: {max(proba[0]):.3f}")
        else:
            print("❌ Binary model ecosystem failed to load")
        
        # Test multiclass model loading
        print("\n🔧 Testing multiclass model ecosystem...")
        multiclass_components = load_model_ecosystem("models/multiclass", "multiclass")
        if multiclass_components:
            print("✅ Multiclass model ecosystem loaded successfully")
            
            # Test preprocessing
            df = pd.DataFrame([sample_data])
            processed_data = preprocess_data_for_model(
                df, 
                multiclass_components['target_encoder'], 
                multiclass_components['config'], 
                'multiclass',
                preprocessor=multiclass_components.get('preprocessor')
            )
            print(f"✅ Multiclass preprocessing successful. Shape: {processed_data.shape}")
            
            # Test prediction
            pred = multiclass_components['model'].predict(processed_data)
            proba = multiclass_components['model'].predict_proba(processed_data)
            
            # Get class name
            if 'label_encoder' in multiclass_components:
                class_name = multiclass_components['label_encoder'].inverse_transform([pred[0]])[0]
            else:
                class_name = Config.ATTACK_TYPES.get(pred[0], f"Unknown_{pred[0]}")
            
            print(f"✅ Multiclass prediction successful. Result: {class_name}, Confidence: {max(proba[0]):.3f}")
        else:
            print("❌ Multiclass model ecosystem failed to load")
            
    except Exception as e:
        print(f"❌ Error in prediction pipeline test: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
    
    print("\n" + "=" * 50)
    print("🏁 Prediction Pipeline Test Complete")

if __name__ == "__main__":
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("🚀 Starting Model Test Suite")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version}")
    print(f"📦 NumPy version: {np.__version__}")
    print(f"📊 Pandas version: {pd.__version__}")
    
    # Run tests
    test_model_loading()
    test_prediction_pipeline()
    
    print("\n✅ All tests completed!")
