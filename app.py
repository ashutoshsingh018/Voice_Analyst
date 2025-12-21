"""
NEXUS OMEGA - Flask Application
Complete conversion from Streamlit while preserving ALL functionality
"""

import os
import re
import traceback
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify, session, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from config import Config
from models import model_manager
from audio_processor import (
    clean_audio, estimate_noise_level, calculate_wpm,
    detect_speaking_time, validate_audio_file, generate_emotion_timeline,
    convert_audio_format  # NEW: For browser-recorded audio
)
from analyzers import (
    smart_emotion_detection, analyze_sentiment, analyze_toxicity,
    generate_summary, extract_keywords, classify_topic,
    advanced_grammar_check, calculate_communication_score,
    translate_text, text_to_speech_file, generate_wordcloud
)
from report_generator import generate_professional_pdf

# Initialize Flask App
app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models at startup
print("=" * 60)
print("NEXUS OMEGA: Initializing AI Models...")
print("=" * 60)
model_manager.load_all_models(whisper_model_size=Config.DEFAULT_WHISPER_MODEL)
print("=" * 60)
print("Models loaded successfully!")
print("=" * 60)

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main interface"""
    session.clear()  # Clear previous session
    return render_template('index.html', languages=Config.WHISPER_LANGUAGES.keys())

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and initial processing
    Maps to Streamlit's file_uploader + analyze button
    
    UPDATED: Now handles both file uploads AND browser-recorded audio
    """
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['audio_file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file (supports WebM/OGG for browser recordings)
    filename = secure_filename(file.filename)
    is_valid_upload = validate_audio_file(file)
    is_browser_recording = filename.endswith(('.webm', '.ogg'))
    
    if not is_valid_upload and not is_browser_recording:
        return jsonify({'error': 'Invalid file type. Use WAV, MP3, FLAC, WebM, or OGG'}), 400
    
    # Save file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # NEW: Convert browser recordings to WAV format
    if is_browser_recording:
        print(f"[Flask] Browser recording detected: {filename}")
        converted_path = convert_audio_format(filepath)
        
        if converted_path != filepath:
            # Use converted file and clean up original
            filepath = converted_path
            print(f"[Flask] Using converted file: {filepath}")
    
    # Store in session
    session['audio_path'] = filepath
    session['filename'] = os.path.basename(filepath)
    
    return jsonify({'success': True, 'filename': os.path.basename(filepath)})

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint - EXACT COPY of Streamlit's analysis logic
    Preserves ALL processing steps
    """
    if 'audio_path' not in session:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    audio_path = session['audio_path']
    
    # Get parameters from request
    lang_choice = request.form.get('language', 'Auto-Detect')
    translate_direct = request.form.get('translate_direct') == 'true'
    
    # Get Whisper language code
    selected_lang = Config.WHISPER_LANGUAGES.get(lang_choice)
    initial_prompt = Config.LANGUAGE_PROMPTS.get(selected_lang, None)
    
    try:
        # Get models
        whisper_model = model_manager.get_model('whisper')
        emotion_model = model_manager.get_model('emotion')
        sentiment_model = model_manager.get_model('sentiment')
        toxicity_model = model_manager.get_model('toxicity')
        summary_model = model_manager.get_model('summary')
        topic_model = model_manager.get_model('topic')
        translator = model_manager.get_model('translator')
        tokenizer = model_manager.get_model('tokenizer')
        grammar_model = model_manager.get_model('grammar_model')
        grammar_tokenizer = model_manager.get_model('grammar_tokenizer')
        device = model_manager.get_device()
        
        # Check model availability
        if not whisper_model:
            return jsonify({'error': 'Whisper model unavailable'}), 500
        
        # 1. WHISPER TRANSCRIPTION (PRESERVED)
        task = "translate" if translate_direct else "transcribe"
        result = whisper_model.transcribe(
            audio_path,
            fp16=False,
            language=selected_lang,
            task=task,
            initial_prompt=initial_prompt,
        )
        
        text = result["text"].strip()
        # Clean up any residual bracketed text (PRESERVED)
        text = re.sub(r'\[.*?\]|\(.*?\)|\<.*?\>', '', text).strip()
        
        if not text:
            return jsonify({'error': 'No speech detected'}), 400
        
        # 2. EMOTION DETECTION (PRESERVED)
        emotion = smart_emotion_detection(audio_path, emotion_model)
        
        # 3. SENTIMENT ANALYSIS (PRESERVED)
        sentiment = analyze_sentiment(text, sentiment_model)
        
        # 4. TOXICITY DETECTION (PRESERVED)
        toxicity = analyze_toxicity(text, toxicity_model)
        
        # 5. SUMMARIZATION (PRESERVED)
        summary = generate_summary(text, summary_model)
        
        # 6. KEYWORD EXTRACTION (PRESERVED)
        keywords = extract_keywords(text)
        
        # 7. ACOUSTIC METRICS (PRESERVED)
        noise_label, noise_ratio = estimate_noise_level(audio_path)
        wpm = calculate_wpm(text, audio_path)
        speaking_time = detect_speaking_time(audio_path)
        
        # 8. GRAMMAR CHECK (PRESERVED)
        corrected_text, change_ratio, highlight_text, grammar_score = advanced_grammar_check(
            text, grammar_model, grammar_tokenizer, device
        )
        
        grammar_results = {
            "score": grammar_score,
            "changes": change_ratio,
            "status": "AI Corrected" if change_ratio > Config.GRAMMAR_CHANGE_THRESHOLD else "Excellent",
            "corrected_text": corrected_text,
            "highlight": highlight_text
        }
        
        # 9. TOPIC CLASSIFICATION (PRESERVED)
        topic_results = classify_topic(text, topic_model)
        
        # 10. COMMUNICATION SCORE (PRESERVED - exact formula)
        comm_score = calculate_communication_score(
            noise_ratio, wpm, toxicity["score"], emotion["label"]
        )
        
        # Compile results (PRESERVED structure)
        results = {
            "text": text,
            "emotion": emotion,
            "sentiment": sentiment,
            "toxicity": toxicity,
            "summary": summary,
            "keywords": keywords,
            "lang": lang_choice,
            "noise": (noise_label, noise_ratio),
            "wpm": wpm,
            "speaking_time": speaking_time,
            "score": comm_score,
            "grammar": grammar_results,
            "topic": topic_results,
        }
        
        # Store in session
        session['analysis_results'] = results
        
        return jsonify({'success': True, 'results': results})
    
    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)
        return jsonify({'error': 'Analysis failed', 'details': str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate():
    """
    Translation endpoint (NLLB + TTS)
    PRESERVED EXACTLY from Streamlit Tab 2
    """
    if 'analysis_results' not in session:
        return jsonify({'error': 'No analysis results available'}), 400
    
    results = session['analysis_results']
    target_lang = request.form.get('target_lang', 'English')
    
    translator = model_manager.get_model('translator')
    tokenizer = model_manager.get_model('tokenizer')
    device = model_manager.get_device()
    
    # Translate text (PRESERVED)
    translated_text, error = translate_text(
        results['text'], target_lang, translator, tokenizer, device
    )
    
    if error:
        return jsonify({'error': error}), 500
    
    # Generate TTS audio (PRESERVED)
    tts_lang_code = Config.GTTS_LANG_MAP.get(target_lang, "en")
    audio_stream = text_to_speech_file(translated_text, lang=tts_lang_code)
    
    if audio_stream:
        # Convert to base64 for JSON response
        audio_base64 = base64.b64encode(audio_stream.read()).decode('utf-8')
        return jsonify({
            'success': True,
            'translated_text': translated_text,
            'audio': audio_base64
        })
    else:
        return jsonify({
            'success': True,
            'translated_text': translated_text,
            'audio': None
        })

@app.route('/emotion_timeline')
def emotion_timeline():
    """
    Generate emotion timeline visualization
    PRESERVED EXACTLY from Streamlit Tab 3
    """
    if 'audio_path' not in session:
        return jsonify({'error': 'No audio file'}), 400
    
    audio_path = session['audio_path']
    emotion_model = model_manager.get_model('emotion')
    
    if not emotion_model:
        return jsonify({'error': 'Emotion model unavailable'}), 500
    
    # Generate timeline (PRESERVED)
    timeline = generate_emotion_timeline(audio_path, emotion_model)
    
    # Create plot (PRESERVED styling)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(timeline, linewidth=2, color="#3498db")
    ax.set_title("Emotional Intensity (Per Second)")
    ax.set_ylabel("Top Emotion Score")
    ax.set_xlabel("Time (s)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    # Convert to base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return jsonify({'success': True, 'image': img_base64})

@app.route('/generate_pdf')
def generate_pdf():
    """
    Generate and download PDF report
    PRESERVED EXACTLY from Streamlit Tab 4
    """
    if 'analysis_results' not in session:
        return jsonify({'error': 'No analysis results'}), 400
    
    results = session['analysis_results']
    
    try:
        pdf_path = generate_professional_pdf(results)
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name='Speech_Report.pdf',
            mimetype='application/pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/wordcloud')
def wordcloud():
    """
    Generate word cloud visualization
    PRESERVED EXACTLY from Streamlit Tab 5
    """
    if 'analysis_results' not in session:
        return jsonify({'error': 'No analysis results'}), 400
    
    results = session['analysis_results']
    img_array = generate_wordcloud(results['text'])
    
    if img_array is None:
        return jsonify({'error': 'Not enough words for word cloud'}), 400
    
    # Convert numpy array to image
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img_array, interpolation='bilinear')
    ax.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close()
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return jsonify({'success': True, 'image': img_base64})

@app.route('/results')
def results():
    """Results page"""
    if 'analysis_results' not in session:
        return redirect(url_for('index'))
    
    return render_template('results.html', 
                          results=session['analysis_results'],
                          target_languages=Config.NLLB_LANG_MAP.keys())

@app.route('/audio_test')
def audio_test():
    """
    Diagnostic test page for audio recording
    NEW: Helps debug microphone issues
    """
    return render_template('audio_test.html')

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)