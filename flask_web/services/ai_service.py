import os
import pickle
import logging
import numpy as np
# import tensorflow as tf # Commented out to avoid import errors if env is not set up, but logic remains
# from tensorflow import keras

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        # Define model paths relative to this file
        # flask_web/services/ai_service.py -> flask_web/models
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = os.path.join(base_dir, 'models')
        
        self.models = {
            'ner': None,
            'sentiment': None,
            'summarizer': None,
            'forecaster': None,
            'tokenizer': None
        }
        
        self._initialized = True
        logger.info(f"AIService initialized. Model directory: {self.model_dir}")

    def _load_keras_model(self, filename):
        """Helper to load Keras models with error handling"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            path = os.path.join(self.model_dir, filename)
            if os.path.exists(path):
                logger.info(f"Loading model: {filename}")
                return keras.models.load_model(path)
            else:
                logger.warning(f"Model file not found: {path}")
                return None
        except Exception as e:
            logger.error(f"Failed to load model {filename}: {str(e)}")
            return None

    def _load_pickle(self, filename):
        """Helper to load Pickle files with error handling"""
        try:
            path = os.path.join(self.model_dir, filename)
            if os.path.exists(path):
                logger.info(f"Loading pickle: {filename}")
                with open(path, 'rb') as f:
                    return pickle.load(f)
            else:
                logger.warning(f"Pickle file not found: {path}")
                return None
        except Exception as e:
            logger.error(f"Failed to load pickle {filename}: {str(e)}")
            return None

    def get_model(self, model_name):
        """Lazy loader for models"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model name: {model_name}")

        if self.models[model_name] is None:
            if model_name == 'ner':
                self.models['ner'] = self._load_keras_model('ner_bilstm.keras')
            elif model_name == 'sentiment':
                self.models['sentiment'] = self._load_keras_model('sentiment_lstm.keras')
            elif model_name == 'summarizer':
                self.models['summarizer'] = self._load_keras_model('summarizer_seq2seq.keras')
            elif model_name == 'forecaster':
                self.models['forecaster'] = self._load_keras_model('forecaster_lstm.keras')
            elif model_name == 'tokenizer':
                self.models['tokenizer'] = self._load_pickle('tokenizer.pkl')
        
        return self.models[model_name]

    def extract_entities(self, text):
        """M1: NER - Extract Hotel, Price, Date"""
        model = self.get_model('ner')
        tokenizer = self.get_model('tokenizer')
        
        if not model or not tokenizer:
            return {"error": "NER model or tokenizer not available"}
        
        # Mock inference logic
        return {"entities": [{"text": "Sample Hotel", "label": "HOTEL"}]}

    def analyze_sentiment(self, text):
        """M2: Sentiment Analysis"""
        model = self.get_model('sentiment')
        if not model:
            return {"error": "Sentiment model not available"}
        
        # Mock inference logic
        return {"sentiment": "positive", "score": 0.95}

    def summarize_request(self, text):
        """M3: Summarizer"""
        model = self.get_model('summarizer')
        if not model:
            return {"error": "Summarizer model not available"}
        
        # Mock inference logic
        return {"summary": "Customer requested a quote for Japan trip."}

    def forecast_price(self, date_range):
        """M4: Price Forecasting"""
        model = self.get_model('forecaster')
        if not model:
            return {"error": "Forecaster model not available"}
        
        # Mock inference logic
        return {"forecast": 150000}

# Singleton instance
ai_service = AIService()
