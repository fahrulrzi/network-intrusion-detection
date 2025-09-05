"""
Prediction pipeline and LLM integration for network intrusion detection
"""

import requests
import logging
from config import Config

logger = logging.getLogger(__name__)

# ================================================================================================
# LLM INTEGRATION
# ================================================================================================

def get_llm_mitigation(attack_type, input_data=None):
    """
    Get LLM-powered mitigation recommendations using Groq API
    """
    try:
        # Check if API key is available
        if not Config.GROQ_API_KEY:
            logger.info("Groq API key not configured - using fallback mitigation")
            return None
        
        # Prepare network data summary if available
        network_summary = ""
        if input_data is not None and len(input_data) > 0:
            row = input_data.iloc[0]
            network_summary = f"""
            Data lalu lintas jaringan yang dianalisis:
            - Durasi koneksi: {row.get('dur', 'N/A')} detik
            - Paket yang dikirim: {row.get('spkts', 'N/A')} paket
            - Bytes yang ditransfer: {row.get('sbytes', 'N/A')} bytes
            - Protocol: {row.get('proto', 'N/A')}
            - Service: {row.get('service', 'N/A')}
            """
        
        prompt = f"""Anda adalah ahli keamanan siber. Analisis hasil deteksi intrusi jaringan ini dan berikan rekomendasi mitigasi dalam bahasa Indonesia.
            HASIL DETEKSI:
            - Jenis Serangan: {attack_type}
            {network_summary}

            Berikan response dalam format markdown yang BENAR dengan struktur:

            ## 1. Penjelasan Serangan
            Jelaskan dalam 1-2 kalimat singkat jenis serangan ini.

            ## 2. TINDAKAN DARURAT
            - Langkah pertama yang harus dilakukan
            - Langkah kedua yang kritis  
            - Langkah ketiga jika diperlukan

            ## 3. Analisis Data Jaringan
            Analisis pola lalu lintas dalam 2-3 kalimat berdasarkan data yang terdeteksi. Sebutkan angka data usernya (agar terlihat bahwa LLM dapat membaca data user yg diinputkan tersebut)

            ## 4. LANGKAH INVESTIGASI
            - Periksa log sistem dan keamanan
            - Analisis traffic pattern lebih detail
            - Identifikasi sumber dan dampak serangan

            ## 5. PENCEGAHAN MASA DEPAN
            - Konfigurasi firewall dan filtering
            - Implementasi sistem monitoring
            - Update keamanan dan patching

            Gunakan format markdown yang benar dengan ## untuk header dan - untuk bullet points. Berikan rekomendasi yang praktis dan spesifik.
            Langsung berikan jawabannya tanpa perlu kalimat pembuka respons (seperti:Here is the analysis and recommendation in markdown format:)"""

        # Call Groq API
        headers = {
            'Authorization': f'Bearer {Config.GROQ_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'model': 'llama3-8b-8192',  # Updated to supported model
            'temperature': 0.3,
            'max_tokens': 1000
        }
        
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                mitigation_text = result['choices'][0]['message']['content'].strip()
                logger.info(f"LLM mitigation generated successfully for {attack_type}")
                return {
                    'success': True,
                    'mitigation': mitigation_text,
                    'source': 'LLM (Groq Llama3-8B)'
                }
        elif response.status_code == 401:
            logger.warning("LLM API authentication failed - check GROQ_API_KEY")
        else:
            logger.warning(f"LLM API call failed with status {response.status_code}")
        
        return None
        
    except Exception as e:
        logger.error(f"Error calling LLM API: {e}")
        return None


def get_fallback_mitigation(attack_type):
    """
    Get fallback mitigation recommendations from predefined templates
    """
    mitigation = Config.FALLBACK_MITIGATIONS.get(attack_type, Config.FALLBACK_MITIGATIONS['Generic'])
    
    return {
        'success': True,
        'mitigation': mitigation,
        'source': 'LLM'
    }



# ================================================================================================
# PREDICTION PIPELINE
# ================================================================================================

def perform_full_prediction(input_data, binary_model_components=None, multiclass_model_components=None):
    """
    Perform complete prediction pipeline: Binary -> Multiclass -> LLM
    """
    from models_utils import preprocess_data_for_model
    import pandas as pd
    
    logger.info("Starting binary classification...")
    
    # Use provided components or try to get from Flask context
    if binary_model_components is None or multiclass_model_components is None:
        try:
            from flask import current_app
            binary_model_components = current_app.config.get('BINARY_MODEL')
            multiclass_model_components = current_app.config.get('MULTICLASS_MODEL')
        except:
            # Fallback to global variables if no Flask context
            try:
                from app import binary_model_components as bin_comp, multiclass_model_components as multi_comp
                binary_model_components = bin_comp
                multiclass_model_components = multi_comp
            except ImportError:
                pass
    
    if not binary_model_components:
        raise Exception("Binary model not loaded")
    
    # Add timestamp
    timestamp = pd.Timestamp.now().isoformat()
    
    # Step 1: Binary Classification
    logger.debug("Preprocessing data for binary model")
    processed_data = preprocess_data_for_model(
        input_data, 
        binary_model_components['target_encoder'], 
        binary_model_components['config'], 
        'binary'
    )
    
    # Make binary prediction
    binary_pred = binary_model_components['model'].predict(processed_data)[0]
    binary_proba = binary_model_components['model'].predict_proba(processed_data)[0]
    binary_confidence = max(binary_proba)
    
    # Determine binary result
    binary_result = "Normal" if binary_pred == 0 else "Attack"
    logger.info(f"Binary prediction: {binary_result} (confidence: {binary_confidence:.2%})")
    
    result = {
        'binary_prediction': binary_result,
        'binary_confidence': binary_confidence,
        'binary_probabilities': {
            'normal': float(binary_proba[0]),
            'attack': float(binary_proba[1])
        },
        'timestamp': timestamp
    }
    
    # Step 2: Multiclass Classification (only if attack detected)
    if binary_pred == 1:  # Attack detected
        logger.info("Attack detected, performing multiclass classification...")
        
        if not multiclass_model_components:
            logger.error("Error in prediction pipeline: Multiclass model not loaded")
            # Return binary result with fallback mitigation
            attack_type = "Generic"
            mitigation = get_fallback_mitigation(attack_type)
            result.update({
                'multiclass_prediction': attack_type,
                'multiclass_confidence': None,
                'attack_type': attack_type,
                'mitigation': mitigation,
                'pipeline_status': 'partial',
                'source': mitigation['source'],
                'notes': 'Multiclass model unavailable - using generic recommendations'
            })
            return result
        
        try:
            # Preprocess for multiclass model
            multiclass_processed = preprocess_data_for_model(
                input_data,
                multiclass_model_components['target_encoder'],
                multiclass_model_components['config'],
                'multiclass',
                preprocessor=multiclass_model_components.get('preprocessor')
            )
            
            # Make multiclass prediction
            multiclass_pred = multiclass_model_components['model'].predict(multiclass_processed)[0]
            multiclass_proba = multiclass_model_components['model'].predict_proba(multiclass_processed)[0]
            multiclass_confidence = max(multiclass_proba)
            
            # Get attack type name
            if 'label_encoder' in multiclass_model_components:
                attack_type = multiclass_model_components['label_encoder'].inverse_transform([multiclass_pred])[0]
            else:
                attack_type = Config.ATTACK_TYPES.get(multiclass_pred, f"Unknown_{multiclass_pred}")
            
            logger.info(f"Multiclass prediction: {attack_type} (confidence: {multiclass_confidence:.2%})")
            
            result.update({
                'multiclass_prediction': attack_type,
                'multiclass_confidence': multiclass_confidence,
                'attack_type': attack_type
            })
            
        except Exception as e:
            logger.error(f"Multiclass prediction failed: {e}")
            # Fallback to generic attack type
            attack_type = "Generic"
            result.update({
                'multiclass_prediction': attack_type,
                'multiclass_confidence': None,
                'attack_type': attack_type,
                'pipeline_status': 'partial',
                'notes': 'Multiclass prediction failed - using generic classification'
            })
        
        # Step 3: Get Mitigation Recommendations
        logger.info("Generating mitigation recommendations...")
        
        # Try LLM first, fallback to template
        mitigation = get_llm_mitigation(
            result['attack_type'], 
            input_data
        )
        
        if not mitigation:
            logger.info("LLM unavailable, using fallback mitigation")
            mitigation = get_fallback_mitigation(result['attack_type'])
        
        result['mitigation'] = mitigation
        result['pipeline_status'] = result.get('pipeline_status', 'complete')
        
    else:
        # Normal traffic - no need for multiclass or mitigation
        result.update({
            'multiclass_prediction': None,
            'multiclass_confidence': None,
            'attack_type': None,
            'mitigation': None,
            'pipeline_status': 'normal'
        })
    
    return result
