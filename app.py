"""
Network Intrusion Detection System - Main Application
Enhanced with multi-model pipeline and LLM integration
"""

import os
import sys
import warnings
import logging
from flask import Flask
from flask_cors import CORS

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='Trying to unpickle estimator.*')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import custom modules
from models_utils import load_model_ecosystem, TargetEncoder, MulticlassTargetEncoder
from routes import register_routes

# ================================================================================================
# GLOBAL MODEL COMPONENTS
# ================================================================================================

binary_model_components = None
multiclass_model_components = None

# ================================================================================================
# APPLICATION INITIALIZATION
# ================================================================================================

def initialize_models():
    """Initialize all model components"""
    global binary_model_components, multiclass_model_components
    
    logger.info("Starting application...")
    
    # Load binary model
    binary_model_components = load_model_ecosystem("models/binary", "binary")
    if binary_model_components:
        logger.info("Binary model ecosystem loaded successfully")
    else:
        logger.error("Failed to load binary model ecosystem")
    
    # Load multiclass model
    multiclass_model_components = load_model_ecosystem("models/multiclass", "multiclass")
    if multiclass_model_components:
        logger.info("Multiclass model ecosystem loaded successfully")
    else:
        logger.error("Failed to load multiclass model ecosystem")


def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    CORS(app, resources={
        r"/*": {
            "origins": "*", # Untuk production, lebih baik ganti "*" dengan domain frontend Anda, misal: "https://aplikasi-nextjs-anda.vercel.app"
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"] # Tambahkan header lain jika perlu
        }
    })
    app.config['SECRET_KEY'] = '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'
    
    # Initialize models
    initialize_models()
    
    # Store model components in app config
    app.config['BINARY_MODEL'] = binary_model_components
    app.config['MULTICLASS_MODEL'] = multiclass_model_components
    
    # Register routes
    register_routes(app)
    
    # Log application status
    logger.info("Starting Flask application...")
    logger.info(f"Binary model loaded: {binary_model_components is not None}")
    logger.info(f"Multiclass model loaded: {multiclass_model_components is not None}")
    
    if not binary_model_components or not multiclass_model_components:
        logger.warning("Application starting without complete model pipeline - predictions may fail")
    else:
        logger.info("Application ready with complete model pipeline")
    
    return app


# ================================================================================================
# MAIN EXECUTION
# ================================================================================================

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5001, debug=True)
