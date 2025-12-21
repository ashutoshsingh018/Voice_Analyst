"""
ML Models Loading & Management
EXACT COPY of Streamlit model loading logic - NO CHANGES
"""

import torch
import whisper
import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import datetime

class SystemLog:
    """Logging utility - PRESERVED from Streamlit"""
    @staticmethod
    def log(message, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [{level}] SYSTEM: {message}")

class ModelManager:
    """
    Singleton class to manage all AI models
    EXACT PRESERVATION of Streamlit's load_models() function
    """
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ModelManager._models_loaded:
            self._initialize_nltk()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.models = {}
            ModelManager._models_loaded = True
    
    def _initialize_nltk(self):
        """Initialize NLTK resources - PRESERVED"""
        for res in ['tokenizers/punkt_tab', 'tokenizers/punkt', 'corpora/stopwords']:
            try:
                if '/' in res:
                    nltk.data.find(res)
            except LookupError:
                nltk.download(res.split('/')[-1], quiet=True)
    
    def load_all_models(self, whisper_model_size='medium'):
        """
        Load all AI models - EXACT COPY from Streamlit
        NO LOGIC CHANGES, only structure for Flask
        """
        SystemLog.log(f"Loading Models on device: {self.device}")
        
        # 1. Whisper Model (PRESERVED)
        try:
            self.models['whisper'] = whisper.load_model(whisper_model_size, device=self.device)
            SystemLog.log(f"Whisper {whisper_model_size} loaded successfully")
        except Exception as e:
            SystemLog.log(f"Failed to load Whisper model: {e}", "ERROR")
            self.models['whisper'] = None
        
        # 2. Emotion Model (PRESERVED)
        self.models['emotion'] = self._safe_pipeline_load(
            "audio-classification", 
            "superb/wav2vec2-base-superb-er"
        )
        
        # 3. Sentiment Model (PRESERVED)
        self.models['sentiment'] = self._safe_pipeline_load(
            "sentiment-analysis",
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # 4. Toxicity Model (PRESERVED)
        self.models['toxicity'] = self._safe_pipeline_load(
            "text-classification",
            "unitary/unbiased-toxic-roberta"
        )
        
        # 5. Summary Model (PRESERVED)
        self.models['summary'] = self._safe_pipeline_load(
            "summarization",
            "facebook/bart-large-cnn"
        )
        
        # 6. Topic Model (PRESERVED)
        self.models['topic'] = self._safe_pipeline_load(
            "zero-shot-classification",
            "facebook/bart-large-mnli"
        )
        
        # 7. NLLB Translator - 1.3B (PRESERVED - Critical for Kannada)
        try:
            nllb_model_name = "facebook/nllb-200-distilled-1.3B"
            self.models['tokenizer'] = AutoTokenizer.from_pretrained(nllb_model_name)
            self.models['translator'] = AutoModelForSeq2SeqLM.from_pretrained(
                nllb_model_name
            ).to(self.device)
            SystemLog.log("NLLB 1.3B translator loaded successfully")
        except Exception as e:
            SystemLog.log(f"Failed to load NLLB translation models: {e}", "ERROR")
            self.models['tokenizer'], self.models['translator'] = None, None
        
        # 8. T5 Grammar Correction (PRESERVED)
        try:
            self.models['grammar_tokenizer'] = AutoTokenizer.from_pretrained(
                "vennify/t5-base-grammar-correction"
            )
            self.models['grammar_model'] = AutoModelForSeq2SeqLM.from_pretrained(
                "vennify/t5-base-grammar-correction"
            ).to(self.device)
            SystemLog.log("T5 grammar correction loaded successfully")
        except Exception as e:
            SystemLog.log(f"Failed to load T5 grammar models: {e}", "ERROR")
            self.models['grammar_tokenizer'], self.models['grammar_model'] = None, None
        
        return self.models
    
    def _safe_pipeline_load(self, task, model_name):
        """
        Safe pipeline loading - PRESERVED from Streamlit
        """
        try:
            model = pipeline(
                task, 
                model=model_name, 
                device=0 if self.device == "cuda" else -1
            )
            SystemLog.log(f"Loaded {model_name} for {task}")
            return model
        except Exception as e:
            SystemLog.log(f"Failed to load {model_name} for {task}: {e}", "ERROR")
            return None
    
    def get_model(self, model_name):
        """Get specific model"""
        return self.models.get(model_name)
    
    def get_device(self):
        """Get current device"""
        return self.device
    
    def are_models_ready(self):
        """Check if critical models are loaded"""
        critical_models = ['whisper', 'emotion', 'sentiment', 'toxicity', 
                          'summary', 'topic', 'translator', 'grammar_model']
        return all(self.models.get(m) is not None for m in critical_models)

# Global instance
model_manager = ModelManager()