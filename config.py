# Configuration file for Network Intrusion Detection System

import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'cyber-attack-detection-secret-key-2025'
    
    # API Configuration
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL = "llama3-8b-8192"  # Updated to supported model
    
    # Model Paths
    BINARY_MODEL_DIR = 'models/binary'
    MULTICLASS_MODEL_DIR = 'models/multiclass'
    FALLBACK_MODEL_DIR = os.environ.get('FALLBACK_MODEL_DIR') or 'saved_models'
    
    # Flask configuration
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5000))
    
    # Security headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block'
    }
    
    # Attack Types for Multiclass Classification
    ATTACK_TYPES = {
        0: 'Normal',
        1: 'Analysis',
        2: 'Backdoor',
        3: 'DoS',
        4: 'Exploits',
        5: 'Fuzzers',
        6: 'Generic',
        7: 'Reconnaissance',
        8: 'Shellcode',
        9: 'Worms'
    }
    
    # Feature configurations for different models
    BINARY_FEATURES = [
        # Numerical features (exclude 'sloss')
        'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 
        'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 
        'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 
        'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
        'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 
        'ct_srv_dst', 'is_sm_ips_ports',
        # Categorical features ('proto' will be target encoded)
        'proto', 'service', 'state'
    ]
    
    MULTICLASS_FEATURES = [
        # Numerical features (exclude 'sloss' and 'is_sm_ips_ports')
        'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 
        'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 
        'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 
        'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
        'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 
        'ct_srv_dst',
        # Categorical features ('proto' will be target encoded)
        'proto', 'service', 'state'
    ]
    
    # Fallback mitigation templates for each attack type
    FALLBACK_MITIGATIONS = {
        'Analysis': {
            'description': 'Serangan Analysis melibatkan pengumpulan informasi sistem dan jaringan untuk mencari kerentanan.',
            'analysis': 'Traffic menunjukkan pola scanning atau probing yang mencurigakan dengan volume query tinggi ke berbagai port atau service. Aktivitas ini biasanya merupakan tahap reconnaissance sebelum serangan utama. Pattern komunikasi menunjukkan adanya upaya enumerasi sistem atau service discovery.',
            'mitigation': 'Aktifkan monitoring dan logging yang lebih ketat untuk mendeteksi pola scanning. Gunakan rate limiting untuk membatasi query berlebihan dari IP yang sama. Implementasikan network segmentation dan hide unnecessary services dari akses eksternal.'
        },
        'Backdoor': {
            'description': 'Serangan Backdoor menciptakan akses tersembunyi yang persisten ke dalam sistem untuk kontrol jangka panjang.',
            'analysis': 'Traffic menunjukkan koneksi outbound yang mencurigakan atau komunikasi dengan command & control server. Pola komunikasi mungkin menggunakan port non-standar atau protokol yang disamarkan. Aktivitas ini mengindikasikan adanya malware yang sudah terinstall dan berkomunikasi dengan penyerang.',
            'mitigation': 'Lakukan full system scan dengan antimalware terbaru dan isolasi sistem yang terinfeksi. Block komunikasi ke IP atau domain yang mencurigakan melalui firewall. Audit dan ganti semua credentials yang mungkin telah dikompromikan.'
        },
        'DoS': {
            'description': 'Serangan Denial of Service (DoS) bertujuan membuat layanan tidak tersedia dengan membanjiri sumber daya sistem.',
            'analysis': 'Traffic menunjukkan volume request yang sangat tinggi dalam waktu singkat yang dapat menghabiskan bandwidth atau resource server. Pattern menunjukkan flood attack seperti SYN flood, UDP flood, atau HTTP flood. Aktivitas ini dapat menyebabkan system overload dan service interruption.',
            'mitigation': 'Implementasikan rate limiting dan traffic shaping untuk membatasi volume request per IP. Gunakan DDoS protection service atau appliance untuk filtering traffic malicious. Aktifkan auto-scaling infrastructure untuk menangani traffic spike yang legitimate.'
        },
        'Exploits': {
            'description': 'Serangan Exploits memanfaatkan kerentanan software atau sistem untuk mendapatkan akses yang tidak sah.',
            'analysis': 'Traffic menunjukkan payload yang mencurigakan dalam request HTTP atau network packet yang mengindikasikan exploit attempt. Pattern komunikasi mungkin mengandung shellcode atau buffer overflow attempts. Aktivitas ini bertujuan mengeksploitasi vulnerabilities untuk mendapatkan system access.',
            'mitigation': 'Segera update dan patch semua software dan sistem operasi ke versi terbaru. Implementasikan Web Application Firewall (WAF) untuk filtering malicious payload. Lakukan vulnerability assessment dan penetration testing secara berkala.'
        },
        'Fuzzers': {
            'description': 'Serangan Fuzzers menggunakan input yang tidak valid atau random untuk mencari kerentanan dalam aplikasi.',
            'analysis': 'Traffic menunjukkan request dengan parameter atau payload yang tidak normal dan bervariasi secara acak. Pattern komunikasi mengandung data yang malformed atau unexpected untuk mencari crash atau error condition. Aktivitas ini merupakan automated testing untuk menemukan vulnerabilities.',
            'mitigation': 'Implementasikan input validation yang ketat pada semua aplikasi dan service. Gunakan rate limiting untuk mencegah automated fuzzing attempts. Monitor application logs untuk mendeteksi unusual request patterns dan block IP yang melakukan fuzzing.'
        },
        'Generic': {
            'description': 'Serangan Generic merupakan kategori umum untuk aktivitas malicious yang tidak terklasifikasi spesifik.',
            'analysis': 'Traffic menunjukkan pattern yang mencurigakan namun tidak masuk kategori serangan spesifik lainnya. Aktivitas ini mungkin merupakan kombinasi beberapa teknik serangan atau variant baru. Pattern komunikasi mengindikasikan aktivitas yang tidak legitimate dan berpotensi berbahaya.',
            'mitigation': 'Lakukan analisis mendalam terhadap traffic pattern untuk mengidentifikasi nature serangan. Implementasikan monitoring tambahan dan logging yang detail. Konsultasikan dengan security expert untuk threat analysis dan response strategy yang tepat.'
        },
        'Reconnaissance': {
            'description': 'Serangan Reconnaissance melakukan pemetaan dan pengumpulan informasi target sebagai persiapan serangan lanjutan.',
            'analysis': 'Traffic menunjukkan aktivitas scanning port, service enumeration, atau information gathering yang sistematis. Pattern komunikasi mengindikasikan upaya pemetaan network topology dan identifikasi target yang vulnerable. Aktivitas ini biasanya merupakan tahap awal dari serangan yang lebih kompleks.',
            'mitigation': 'Implementasikan network monitoring untuk mendeteksi scanning activities dan block source IP. Gunakan honeypots untuk mendeteksi dan mempelajari reconnaissance attempts. Minimize information disclosure melalui proper system hardening dan service configuration.'
        },
        'Shellcode': {
            'description': 'Serangan Shellcode melibatkan injeksi dan eksekusi kode malicious untuk mengambil alih kontrol sistem.',
            'analysis': 'Traffic mengandung payload binary yang mencurigakan atau encoded shellcode dalam network communication. Pattern menunjukkan upaya code injection melalui buffer overflow atau script injection. Aktivitas ini bertujuan mengeksekusi arbitrary code pada target system.',
            'mitigation': 'Implementasikan Data Execution Prevention (DEP) dan Address Space Layout Randomization (ASLR). Gunakan antimalware dengan real-time protection dan behavior analysis. Isolasi sistem yang terdeteksi shellcode dan lakukan forensic analysis untuk menentukan scope infection.'
        },
        'Worms': {
            'description': 'Serangan Worms merupakan malware yang dapat mereplikasi diri dan menyebar secara otomatis melalui jaringan.',
            'analysis': 'Traffic menunjukkan pattern propagasi yang khas dengan upaya koneksi ke multiple hosts dalam network. Aktivitas scanning dan exploitation terjadi secara otomatis untuk mencari host vulnerable. Communication pattern mengindikasikan self-replicating behavior dan network-based spreading.',
            'mitigation': 'Segera isolasi network segment yang terinfeksi dan block lateral movement. Update signature antimalware dan lakukan network-wide scanning untuk identifikasi infected hosts. Implementasikan network segmentation yang ketat untuk mencegah worm propagation.'
        }
    }

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
    
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
