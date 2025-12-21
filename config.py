"""
Configuration file for Nexus Omega Flask Application
Preserves all settings from Streamlit version
"""

import os

class Config:
    # Flask Settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'nexus-omega-secret-key-2025'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
    
    # Upload Settings
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'webm', 'ogg'}  # Added WebM/OGG for browser recordings
    
    # Model Settings
    WHISPER_MODEL_SIZES = ['base', 'small', 'medium']
    DEFAULT_WHISPER_MODEL = 'medium'
    
    # Language Settings - EXACT mapping from Streamlit
    WHISPER_LANGUAGES = {
        "Auto-Detect": None,
        "English": "en",
        "Hindi": "hi",
        "Kannada": "kn",
        "Tamil": "ta",
        "Telugu": "te",
        "Marathi": "mr",
        "Bengali": "bn",
        "Spanish": "es",
        "Chinese": "zh",
    }
    
    # Initial prompts for better Indian language recognition (PRESERVED)
    LANGUAGE_PROMPTS = {
        "hi": "नमस्ते, मैं हिंदी बोल रहा हूँ।",
        "en": "Hello, this is a standard English sentence.",
        "es": "Hola, esta es una frase en español.",
        "zh": "你好,这是中文句子。",
        "kn": "ನಮಸ್ಕಾರ, ಇದು ಕನ್ನಡ ವಾಕ್ಯ.",
        "ta": "வணக்கம், இது தமிழ் வாக்கியம்.",
        "te": "నమస్కారం, ఇది తెలుగు వాక్యం.",
        "mr": "नमस्कार, हे मराठी वाक्य आहे.",
        "bn": "নমস্কার, এটি বাংলা বাক্য।",
    }
    
    # NLLB Translation Language Codes (PRESERVED)
    NLLB_LANG_MAP = {
        "English": "eng_Latn",
        "Hindi": "hin_Deva",
        "Kannada": "kan_Knda",
        "Tamil": "tam_Taml",
        "Telugu": "tel_Telu",
        "Marathi": "mar_Deva",
        "Bengali": "ben_Beng",
        "Spanish": "spa_Latn",
        "German": "deu_Latn",
        "French": "fra_Latn",
        "Chinese": "zho_Hans"
    }
    
    # gTTS Language Codes (PRESERVED)
    GTTS_LANG_MAP = {
        "English": "en", "Hindi": "hi", "Kannada": "kn", 
        "Tamil": "ta", "Telugu": "te", "Marathi": "mr", 
        "Bengali": "bn", "Spanish": "es", "German": "de", 
        "French": "fr", "Chinese": "zh-cn"
    }
    
    # Topic Classification Labels (PRESERVED)
    TOPIC_LABELS = [
        "Business and Finance",
        "Technology and Computing",
        "Health and Wellness",
        "Politics and Society",
        "Sports and Entertainment",
        "Education and Learning",
        "Personal/General Conversation"
    ]
    
    # Audio Processing Settings (PRESERVED)
    AUDIO_SAMPLE_RATE = 16000
    BUTTER_FILTER_ORDER = 10
    BUTTER_FILTER_CUTOFF = 100  # Hz
    NOISE_ESTIMATION_DURATION = 0.25  # seconds
    SILENCE_TOP_DB = 30
    
    # Communication Score Weights (PRESERVED EXACTLY)
    SCORE_WEIGHT_NOISE = 30
    SCORE_WEIGHT_WPM = 30
    SCORE_WEIGHT_TOXICITY = 20
    SCORE_WEIGHT_EMOTION = 20
    OPTIMAL_WPM = 140
    
    # Grammar Settings (PRESERVED)
    GRAMMAR_MIN_SCORE = 40
    GRAMMAR_CHANGE_THRESHOLD = 0.03
    
    # Emotion Detection Settings (PRESERVED)
    EMOTION_ANGER_THRESHOLD = 0.60
    POSITIVE_EMOTIONS = ["neutral", "happy"]
    NEGATIVE_EMOTIONS = ["anger", "sadness"]