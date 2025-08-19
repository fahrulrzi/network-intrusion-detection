# 🛡️ Advanced Cyber Attack Detection System

A sophisticated multi-model AI pipeline for network intrusion detection that combines binary classification, multiclass attack detection, and LLM-powered mitigation recommendations.

## � Features

### 1. **Binary Classification**
- **Model**: Random Forest Classifier
- **Purpose**: Determines if network traffic is normal or contains an attack
- **Features**: 41 optimized features (excludes 'sloss')
- **Accuracy**: High-performance detection with target encoding

### 2. **Multiclass Attack Detection**
- **Model**: Gradient Boosting Classifier
- **Purpose**: Classifies specific attack types when attacks are detected
- **Attack Types**: 9 categories
  - Analysis
  - Backdoor
  - DoS (Denial of Service)
  - Exploits
  - Fuzzers
  - Generic
  - Reconnaissance
  - Shellcode
  - Worms
- **Features**: 40 optimized features (excludes 'sloss' and 'is_sm_ips_ports')

### 3. **AI-Powered Mitigation Recommendations**
- **LLM**: Groq Llama-3.1-70B-Versatile
- **Purpose**: Generates specific, actionable mitigation strategies
- **Output Format**:
  - **Description**: Brief explanation of attack type (1-2 sentences)
  - **Traffic Analysis**: Analysis of network traffic patterns (3 sentences)
  - **Mitigation Steps**: Specific remediation recommendations (3 sentences)
- **Fallback**: Template-based responses when LLM is unavailable

## 🔄 Detection Pipeline

1. **Input Processing**: User provides network traffic features
2. **Binary Classification**: Determines Normal vs Attack
3. **Conditional Multiclass**: If attack detected → classify attack type
4. **AI Recommendations**: If attack detected → generate mitigation strategy
5. **Results Display**: Comprehensive analysis with actionable insights

## �️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd network-intrusion-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables (Optional for LLM features)**
   ```bash
   # Windows Command Prompt
   set GROQ_API_KEY=your_groq_api_key_here
   
   # Windows PowerShell
   $env:GROQ_API_KEY="your_groq_api_key_here"
   
   # Linux/Mac
   export GROQ_API_KEY="your_groq_api_key_here"
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the web interface**
   Open your browser and navigate to `http://localhost:5000`

## 🔧 Configuration

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key for LLM functionality (optional)
- `FLASK_DEBUG`: Enable/disable debug mode (default: True)
- `FLASK_HOST`: Host address (default: 0.0.0.0)
- `FLASK_PORT`: Port number (default: 5000)

### Getting Groq API Key

1. Visit [Groq Console](https://console.groq.com/keys)
2. Sign up or log in
3. Generate a new API key
4. Set the environment variable

**Note**: The system works without the API key but will use template-based mitigation recommendations instead of AI-generated ones.
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Model Files**
   Pastikan direktori model tersedia dalam salah satu format berikut:
   
   **Option 1: models/ directory (recommended)**
   ```
   models/
   ├── main_model.joblib           # Model Random Forest pipeline
   ├── target_encoder.pkl          # Custom target encoder untuk 'proto'
   ├── feature_config.json         # Konfigurasi fitur
   └── model_metrics.json          # Metrics performa model
   ```

   **Option 2: saved_models/ directory (fallback)**
   ```
   saved_models/
   └── cybersecurity_model_YYYYMMDD_HHMMSS/
       ├── main_model.joblib
       ├── target_encoder.pkl
       ├── feature_config.json
       ├── model_metrics.json
       └── README.md
   ```

5. **Jalankan Aplikasi**
   ```bash
   python app.py
   ```

6. **Akses Aplikasi**
   Buka browser dan kunjungi: `http://localhost:5000`

## 📋 Cara Penggunaan

### 1. Menggunakan Data Contoh
- Klik tombol **"Load Normal Traffic"** untuk memuat contoh data lalu lintas normal
- Klik tombol **"Load Attack Traffic"** untuk memuat contoh data serangan
- Klik **"Analyze Network Traffic"** untuk mendapatkan prediksi

### 2. Input Manual
Isi form dengan data network traffic yang ingin dianalisis:

#### Basic Connection Information
- **Duration (dur)**: Durasi koneksi dalam detik
- **Protocol (proto)**: Jenis protokol (tcp, udp, icmp, dll) - *akan di-encode otomatis*
- **Service**: Jenis layanan jaringan
- **State**: Status koneksi
- **Rate**: Rate koneksi

#### Packet Information
- **Source/Dest Packets**: Jumlah paket dari/ke source/destination
- **Source/Dest Bytes**: Jumlah bytes dari/ke source/destination

#### Network Metrics
- **TTL**: Time to Live source dan destination
- **Load**: Beban jaringan source dan destination
- **Jitter**: Variasi latensi source dan destination
- **Window Size**: Ukuran window TCP
- **Loss**: Paket yang hilang (dloss saja, sloss tidak digunakan)

#### Connection Counts
- Various connection count features untuk analisis pola traffic

### 3. Interpretasi Hasil
- **Normal Traffic**: Ditampilkan dengan warna hijau, aman
- **Attack Detected**: Ditampilkan dengan warna merah, memerlukan perhatian
- **Confidence Score**: Persentase keyakinan model (0-100%)
- **Recommendations**: Saran tindakan jika terdeteksi serangan

## 🔧 Struktur Project

```
cyber-attack-detection/
├── app.py                          # Aplikasi Flask utama
├── config.py                       # Konfigurasi aplikasi
├── requirements.txt                # Dependencies Python
├── README.md                      # Dokumentasi ini
├── templates/
│   ├── index.html                       # Template HTML dengan Tailwind CSS
│   └── instructions.html                # Halaman instruksi
├── models/                              # Model directory (primary)
│   ├── binary_classifier_model.joblib   # Random Forest pipeline
│   ├── target_encoder.pkl               # Target encoder
│   ├── feature_config.json              # Feature configuration
│   └── model_metrics.json               # Performance metrics
├── saved_models/                        # Fallback model directory
│   └── cybersecurity_model_*/           # Timestamped model saves
```

## 🛠️ Technical Details

### Model Architecture
- **Base Algorithm**: Random Forest Classifier
- **Hyperparameters** (optimized with Random Search):
  - n_estimators: Optimized through hyperparameter tuning
  - max_depth: Tuned for optimal performance
  - min_samples_split: Optimized value
  - min_samples_leaf: Tuned parameter
  - max_features: Selected optimal feature subset strategy

### Target Encoding System
Model menggunakan **Custom Target Encoder** untuk menangani fitur kategorikal 'proto':

**Target Encoding Process**:
```python
# Training phase
target_encoder = TargetEncoder()
target_encoder.fit(train_data[['proto']], train_labels)

# Inference phase
proto_encoded = target_encoder.transform(new_data[['proto']])
```

**Key Features**:
- Menghindari data leakage dengan proper train/validation split
- Handle unknown categories dengan global mean fallback
- Robust encoding yang konsisten antara training dan inference

### Feature Engineering
Model menggunakan fitur original dari dataset (43 features):

**Included Features**:
- Basic connection info (dur, proto, service, state)
- Packet metrics (spkts, dpkts, sbytes, dbytes, rate)
- Timing info (sttl, dttl, sload, dload, dloss)
- Network behavior (sinpkt, dinpkt, sjit, djit)
- TCP specifics (swin, dwin, stcpb, dtcpb, tcprtt, synack, ackdat)
- Statistical features (smean, dmean, trans_depth, response_body_len)
- Connection patterns (ct_* features)
- Behavioral flags (is_ftp_login, ct_ftp_cmd, etc.)

**Excluded Features**:
- `sloss`: Removed during feature selection for better performance

### Data Preprocessing Pipeline
- **Target Encoding**: Custom encoder untuk fitur 'proto'
- **Feature Scaling**: RobustScaler untuk fitur numerik (tahan outliers)
- **One-hot Encoding**: Untuk fitur kategorikal lainnya (service, state)
- **Missing Value Handling**: Smart default values
- **Feature Selection**: 43 dari 44 fitur original (exclude sloss)

### Model Ecosystem Components
Sistem deployment menggunakan 4 komponen utama:

1. **main_model.joblib**: Complete Random Forest pipeline
2. **target_encoder.pkl**: Custom target encoder dengan mapping
3. **feature_config.json**: Konfigurasi fitur dan metadata
4. **model_metrics.json**: Performance metrics dan info model

### Deployment Architecture
```python
# Load model ecosystem
components = load_model_ecosystem('models/')

# Preprocess new data
processed_data = preprocess_new_data(
    new_data, 
    components['target_encoder'], 
    components['config']
)

# Predict
prediction = components['model'].predict(processed_data)
```

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| Algorithm | Random Forest |
| Features | 43 (exclude sloss) |
| Target Encoding | Custom for 'proto' |
| Preprocessing | RobustScaler + OneHot |
| Optimization | Bayesian Optimization|

## 🔄 Model Usage Examples

### Encoding dan Prediksi dengan Pipeline Baru

#### 1. Load Model Ecosystem
```python
import joblib
import pickle
import json
import pandas as pd

# Load semua komponen
model = joblib.load('models/main_model.joblib')
with open('models/target_encoder.pkl', 'rb') as f:
    target_encoder = pickle.load(f)
with open('models/feature_config.json', 'r') as f:
    config = json.load(f)
```

#### 2. Prepare Data Baru
```python
# Data baru dari user atau monitoring system
new_data = pd.DataFrame({
    'dur': [0.12],
    'proto': ['tcp'],  # Akan di-encode otomatis
    'service': ['http'],
    'state': ['FIN'],
    'spkts': [8],
    'dpkts': [6],
    # ... semua 43 features (exclude sloss)
})
```

#### 3. Preprocessing dengan Target Encoding
```python
def preprocess_new_data(data, target_encoder, config):
    # Target encode 'proto'
    if 'proto' in data.columns:
        proto_encoded = target_encoder.transform(data[['proto']])
        data = data.drop('proto', axis=1)
        data['proto_encoded'] = proto_encoded['proto_encoded']
    
    # Remove 'sloss' if present
    if 'sloss' in data.columns:
        data = data.drop('sloss', axis=1)
    
    # Ensure all required features present
    required_features = config['numeric_features'] + config['categorical_features']
    data = data[required_features]
    
    return data

# Apply preprocessing
processed_data = preprocess_new_data(new_data, target_encoder, config)
```

#### 4. Prediksi
```python
# Predict
prediction = model.predict(processed_data)[0]
probability = model.predict_proba(processed_data)[0]

# Results
if prediction == 1:
    print(f"🚨 ATTACK DETECTED! Confidence: {probability[1]:.2%}")
else:
    print(f"✅ Normal Traffic. Confidence: {probability[0]:.2%}")
```

### Integration untuk Production

#### Web API Integration
```python
from flask import Flask, request, jsonify

@app.route('/predict', methods=['POST'])
def predict_attack():
    # Get data from request
    input_data = request.json
    
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Preprocess dengan target encoding
    processed_df = preprocess_new_data(df, target_encoder, config)
    
    # Predict
    prediction = model.predict(processed_df)[0]
    probability = model.predict_proba(processed_df)[0]
    
    return jsonify({
        'prediction': int(prediction),
        'confidence': {
            'normal': float(probability[0]),
            'attack': float(probability[1])
        }
    })
```

#### Batch Processing
```python
# For processing multiple records
def batch_predict(data_list):
    # Convert to DataFrame
    df = pd.DataFrame(data_list)
    
    # Preprocess semua data sekaligus
    processed_df = preprocess_new_data(df, target_encoder, config)
    
    # Batch prediction
    predictions = model.predict(processed_df)
    probabilities = model.predict_proba(processed_df)
    
    return predictions, probabilities
```

## 🚨 Troubleshooting

### Error: Model ecosystem tidak ditemukan
- Pastikan direktori `models/` ada dan berisi semua file required:
  - `main_model.joblib`
  - `target_encoder.pkl` 
  - `feature_config.json`
- Atau gunakan direktori fallback `saved_models/cybersecurity_model_*/`

### Error: Target encoding gagal
- Pastikan file `target_encoder.pkl` kompatibel dengan data input
- Check log aplikasi untuk error details encoding

### Error: ModuleNotFoundError
- Pastikan virtual environment aktif
- Install ulang dependencies: `pip install -r requirements.txt`
- Pastikan `imbalanced-learn` terinstall untuk SMOTE support

### Error: Feature mismatch
- Pastikan input data memiliki semua 43 required features
- Check `feature_config.json` untuk daftar fitur yang dibutuhkan
- Fitur 'sloss' harus dihapus dari input data

### Error: Port sudah digunakan
- Ubah port di `app.py`: `app.run(debug=True, host='0.0.0.0', port=5001)`
- Atau hentikan proses yang menggunakan port 5000

### Performance Issues
- Aplikasi berjalan optimal pada Python 3.8+
- Random Forest lebih cepat dan stabil dibanding XGBoost untuk inferensi
- Target encoding dilakukan sekali saat load model, sangat efisien

### Debugging Model Issues

#### Check Model Components
```python
import joblib
import pickle

# Check main model
model = joblib.load('models/main_model.joblib')
print(f"Model type: {type(model)}")
print(f"Pipeline steps: {model.named_steps.keys()}")

# Check target encoder
with open('models/target_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
print(f"Encoder type: {type(encoder)}")
print(f"Available encodings: {encoder.encodings.keys()}")
```

#### Test Prediction Pipeline
```python
# Test dengan data minimal
test_data = pd.DataFrame([{
    'dur': 0.12, 'proto': 'tcp', 'service': 'http', 'state': 'FIN',
    # ... isi semua 43 features
}])

# Test preprocessing
processed = preprocess_new_data(test_data, encoder, config)
print(f"Processed shape: {processed.shape}")

# Test prediction
pred = model.predict(processed)
prob = model.predict_proba(processed)
print(f"Prediction: {pred[0]}, Probability: {prob[0]}")
```

## 📚 Referensi

1. UNSW-NB15 Dataset: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Random Forest Documentation: https://scikit-learn.org/stable/modules/ensemble.html#forest
3. Target Encoding: https://contrib.scikit-learn.org/category_encoders/targetencoder.html
4. Scikit-learn Documentation: https://scikit-learn.org/
5. Flask Documentation: https://flask.palletsprojects.com/
6. Imbalanced-learn Documentation: https://imbalanced-learn.org/

## 📄 Lisensi

Proyek ini dibuat untuk tujuan edukasi dalam mata kuliah Kecerdasan Artifisial.

---

Untuk pertanyaan atau masalah, silakan buat issue atau hubungi tim pengembang.
