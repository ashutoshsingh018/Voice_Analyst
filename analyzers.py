"""
Speech Analysis Functions
EXACT COPY of all analysis logic from Streamlit - NO CHANGES
"""

import re
import io
import difflib
from collections import Counter
from gtts import gTTS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from audio_processor import clean_audio
from config import Config

def smart_emotion_detection(audio_path, model):
    """
    Smart emotion detection with heuristics
    PRESERVED EXACTLY - including anger threshold logic
    """
    if not model:
        return {"label": "Unavailable", "score": 0.0}
    
    clean_audio(audio_path)
    res = model(audio_path)
    res = sorted(res, key=lambda x: x["score"], reverse=True)
    top = res[0]
    
    # Heuristic: Downgrade low-confidence 'anger' to 'neutral' (PRESERVED)
    if top["label"] == "anger" and top["score"] < Config.EMOTION_ANGER_THRESHOLD:
        neutral = next((x for x in res if x["label"] == "neutral"), None)
        if neutral:
            neutral["score"] = 0.55
            return neutral
    
    return top

def analyze_sentiment(text, model):
    """
    Sentiment analysis
    PRESERVED EXACTLY
    """
    if not model:
        return {"label": "NEUTRAL", "score": 0.5}
    
    try:
        result = model(text)[0]
        return result
    except Exception as e:
        print(f"Sentiment analysis failed: {e}")
        return {"label": "NEUTRAL", "score": 0.5}

def analyze_toxicity(text, model):
    """
    Toxicity detection
    PRESERVED EXACTLY
    """
    if not model:
        return {"label": "safe", "score": 0.0}
    
    try:
        tox_raw = model(text)
        tox = tox_raw[0][0] if isinstance(tox_raw, list) else tox_raw[0]
        return tox
    except Exception as e:
        print(f"Toxicity analysis failed: {e}")
        return {"label": "safe", "score": 0.0}

def generate_summary(text, model):
    """
    Text summarization
    PRESERVED EXACTLY - including word count thresholds
    """
    if not model:
        return "Summarization model unavailable."
    
    wc = len(text.split())
    if wc < 20:
        return "Original text is too short to summarize."
    
    try:
        summary = model(
            text,
            max_length=int(wc * 0.8),
            min_length=5,
            do_sample=False
        )[0]["summary_text"]
        return summary
    except Exception as e:
        print(f"Summarization failed: {e}")
        return "Summary unavailable."

def extract_keywords(text, top_n=5):
    """
    Extract top keywords using NLTK
    PRESERVED EXACTLY
    """
    try:
        toks = word_tokenize(text.lower())
        stops = set(stopwords.words("english"))
        kws = [w for w in toks if w.isalnum() and w not in stops]
        top_kws = [k for k, v in Counter(kws).most_common(top_n)]
        return top_kws
    except Exception as e:
        print(f"Keyword extraction failed: {e}")
        return []

def classify_topic(text, model):
    """
    Zero-shot topic classification
    PRESERVED EXACTLY - including all candidate labels
    """
    if not model or len(text.split()) < 5:
        return {
            "label": "Too Short/Unavailable",
            "score": 0.0,
            "raw_results": []
        }
    
    try:
        result = model(text, Config.TOPIC_LABELS)
        
        top_topic = result['labels'][0]
        top_score = result['scores'][0]
        raw_results = list(zip(result['labels'], result['scores']))
        
        return {
            "label": top_topic,
            "score": round(top_score * 100, 1),
            "raw_results": raw_results
        }
    except Exception as e:
        print(f"Topic classification failed: {e}")
        return {
            "label": f"Error: {e.__class__.__name__}",
            "score": 0.0,
            "raw_results": []
        }

def advanced_grammar_check(text, model, tokenizer, device):
    """
    T5-based grammar correction
    PRESERVED EXACTLY - including scoring algorithm
    """
    if not all([text, model, tokenizer]):
        return text, 0, "", 100
    
    input_text = "grammar: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    try:
        # Beam search for better grammar optimization (PRESERVED)
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_beams=8,
            early_stopping=True,
            temperature=0.7,
            repetition_penalty=2.0
        )
        
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Compute difference ratio for scoring (PRESERVED)
        seq = difflib.SequenceMatcher(None, text, corrected_text)
        change_ratio = 1 - seq.ratio()
        
        # Levenshtein distance based score (40 is the minimum tolerance) (PRESERVED)
        grammar_score = max(Config.GRAMMAR_MIN_SCORE, 100 - (change_ratio * 100))
        
        # Highlight changed words (PRESERVED)
        diff = difflib.ndiff(text.split(), corrected_text.split())
        highlight = " ".join([
            f"**{token[2:]}**" if token.startswith("+ ") else token[2:]
            for token in diff if not token.startswith("- ")
        ])
        
        return corrected_text, change_ratio, highlight, round(grammar_score, 1)
    
    except Exception as e:
        print(f"T5 Grammar Check failed: {e.__class__.__name__}")
        return text, 0, f"Error: {e.__class__.__name__}", 100

def calculate_communication_score(noise_ratio, wpm, toxicity_score, emotion_label):
    """
    Calculate overall communication quality score
    PRESERVED EXACTLY - including all weights and formula
    """
    comm_score = (
        (1 - noise_ratio) * Config.SCORE_WEIGHT_NOISE +
        max(0, 1 - abs(wpm - Config.OPTIMAL_WPM) / Config.OPTIMAL_WPM) * Config.SCORE_WEIGHT_WPM +
        (1 - toxicity_score) * Config.SCORE_WEIGHT_TOXICITY +
        (0.5 if emotion_label in Config.POSITIVE_EMOTIONS else 0.2) * Config.SCORE_WEIGHT_EMOTION
    )
    return round(comm_score, 1)

def translate_text(text, target_lang, translator, tokenizer, device):
    """
    NLLB translation with beam search
    PRESERVED EXACTLY - including beam search parameters
    """
    if not all([translator, tokenizer]):
        return None, "Translator unavailable"
    
    try:
        # Tokenize input (PRESERVED)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        # Get Target Code (PRESERVED)
        target_code = Config.NLLB_LANG_MAP.get(target_lang)
        if not target_code:
            return None, "Invalid target language"
        
        # Generate translation with beam search (PRESERVED)
        translated_tokens = translator.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),
            max_length=512,
            num_beams=5,
            repetition_penalty=1.2
        )
        
        final_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return final_text, None
    
    except Exception as e:
        print(f"Translation failed: {e}")
        return None, str(e)

def text_to_speech_file(text, lang="en"):
    """
    Convert text to audio using gTTS
    PRESERVED EXACTLY
    """
    try:
        tts = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        print(f"TTS Error ({lang}): {e}")
        return None

def generate_wordcloud(text):
    """
    Generate word cloud from text
    PRESERVED EXACTLY - including error handling
    """
    if not text or len(text.strip()) == 0:
        return None
    try:
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        return wc.to_array()
    except ValueError:
        # Handles empty vocab error
        return None
    except Exception:
        return None