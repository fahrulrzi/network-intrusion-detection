"""
Flask routes for network intrusion detection web application
"""

import pandas as pd
import logging
from flask import render_template, request, jsonify, current_app
from predictions import perform_full_prediction
from sample_data import SAMPLE_REPO

logger = logging.getLogger(__name__)

def register_routes(app):
    """Register all application routes"""

    @app.route('/')
    def home():
        """Home page - Landing page"""
        logger.info("Landing page accessed")
        return render_template('pages/cyberguard-landing.html')

    @app.route('/predict')
    def predict_page():
        """Prediction interface page"""
        logger.info("Prediction page accessed")
        return render_template('index.html')

    @app.route('/instructions')
    def instructions():
        """Instructions page"""
        logger.info("Instructions page accessed")
        return render_template('pages/instructions.html')

    @app.route('/metodologi')
    def metodologi():
        """Methodology and Innovation page"""
        logger.info("Methodology page accessed")
        return render_template('pages/metodologi.html')

    @app.route('/inovasi')
    def inovasi():
        """Innovation Technology page"""
        logger.info("Innovation Technology page accessed")
        return render_template('pages/inovasi.html')

    # Component routes for dynamic navbar/footer loading
    @app.route('/components/navbar.html')
    def component_navbar():
        logger.debug("Serving navbar component")
        return render_template('components/navbar.html')

    @app.route('/components/footer.html')
    def component_footer():
        logger.debug("Serving footer component")
        return render_template('components/footer.html')

    @app.route('/get_sample_data/<data_type>')
    def get_sample_data(data_type):
        """Get sample data for testing"""
        logger.info(f"Sample data requested: {data_type}")
        
        # Use centralized sample repository
        kind = (data_type or '').lower()
        if kind not in {"normal", "attack"}:
            return jsonify({'error': 'Invalid data type'}), 400
        sample = SAMPLE_REPO.get_random(kind)
        logger.debug(f"Returning {kind} sample data with {len(sample)} features")
        return jsonify(sample)

    @app.route('/predict', methods=['POST'])
    def predict_api():
        """Main prediction endpoint"""
        try:
            logger.info("Prediction request received")
            
            # Get data from request
            if request.is_json:
                input_data = request.get_json()
            else:
                input_data = request.form.to_dict()
            
            # Remove fields that are not part of prediction input if provided accidentally
            for drop_key in ("id", "label", "attack_cat"):
                if drop_key in input_data:
                    input_data.pop(drop_key, None)

            logger.debug(f"Received form data: {list(input_data.keys())}")
            
            # Convert numeric fields
            numeric_fields = [
                'dur', 'rate', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'sttl', 'dttl',
                'sload', 'dload', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
                'swin', 'dwin', 'stcpb', 'dtcpb', 'tcprtt', 'synack', 'ackdat',
                'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src',
                'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
                'ct_dst_src_ltm', 'ct_src_ltm', 'ct_srv_dst', 'is_ftp_login',
                'ct_ftp_cmd', 'ct_flw_http_mthd', 'is_sm_ips_ports', 'sloss'
            ]
            
            for field in numeric_fields:
                if field in input_data:
                    try:
                        # Normalize decimal comma to dot just in case
                        val = input_data[field]
                        if isinstance(val, str):
                            val = val.replace(',', '.')
                        input_data[field] = float(val)
                    except (ValueError, TypeError):
                        input_data[field] = 0.0
            
            # Create DataFrame
            df = pd.DataFrame([input_data])
            logger.debug(f"Input data shape: {df.shape}")
            
            # Get model components from Flask app config
            binary_model = current_app.config.get('BINARY_MODEL')
            multiclass_model = current_app.config.get('MULTICLASS_MODEL')
            
            # Perform full prediction pipeline
            result = perform_full_prediction(df, binary_model, multiclass_model)
            
            # Format response for UI compatibility
            ui_response = {
                'binary_prediction': result.get('binary_prediction', 'Unknown'),
                'binary_confidence': result.get('binary_confidence', 0) * 100,  # Convert to percentage
                'is_attack': result.get('binary_prediction') == 'Attack',
                'multiclass_prediction': result.get('multiclass_prediction'),
                'multiclass_confidence': result.get('multiclass_confidence', 0) * 100 if result.get('multiclass_confidence') else None,
                'multiclass_available': multiclass_model is not None and result.get('pipeline_status') != 'partial',
                'attack_type': result.get('attack_type'),
                'mitigation': result.get('mitigation'),
                'pipeline_status': result.get('pipeline_status', 'unknown'),
                'notes': result.get('notes'),
                'timestamp': result.get('timestamp') or pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"Prediction completed: {result.get('binary_prediction', 'Unknown')}")
            return jsonify(ui_response)
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            logger.error(f"Traceback: ", exc_info=True)
            return jsonify({
                'error': str(e),
                'binary_prediction': 'Error',
                'pipeline_status': 'error'
            }), 500

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors"""
        logger.warning(f"404 error: {request.url}")
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors"""
        logger.error(f"500 error: {error}")
        return jsonify({'error': 'Internal server error'}), 500
