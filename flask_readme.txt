# NEXUS OMEGA - Flask Speech Intelligence System

## 🎯 Project Overview

**NEXUS OMEGA** is a comprehensive speech analysis system converted from Streamlit to Flask while preserving **100% of the original functionality**. This system provides advanced AI-powered speech intelligence including transcription, emotion detection, sentiment analysis, grammar correction, multi-language translation, and more.

### Original vs Flask Comparison

| Feature | Streamlit Version | Flask Version | Status |
|---------|------------------|---------------|--------|
| Whisper ASR | ✅ | ✅ | **PRESERVED** |
| Emotion Detection | ✅ | ✅ | **PRESERVED** |
| Sentiment Analysis | ✅ | ✅ | **PRESERVED** |
| Toxicity Detection | ✅ | ✅ | **PRESERVED** |
| Topic Classification | ✅ | ✅ | **PRESERVED** |
| Grammar Correction (T5) | ✅ | ✅ | **PRESERVED** |
| NLLB Translation (1.3B) | ✅ | ✅ | **PRESERVED** |
| Text-to-Speech | ✅ | ✅ | **PRESERVED** |
| Acoustic Analysis | ✅ | ✅ | **PRESERVED** |
| Communication Score | ✅ | ✅ | **PRESERVED** |
| PDF Reports | ✅ | ✅ | **PRESERVED** |
| Word Cloud | ✅ | ✅ | **PRESERVED** |
| Emotion Timeline | ✅ | ✅ | **PRESERVED** |

---

## 📁 Project Structure

```
nexus_omega_flask/
├── app.py                      # Main Flask application
├── config.py                   # Configuration (all settings preserved)
├── models.py                   # ML model loading (exact copy)
├── audio_processor.py          # Audio processing utilities (exact copy)
├── analyzers.py                # Analysis functions (exact copy)
├── report_generator.py         # PDF generation (exact copy)
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── static/
│   ├── css/
│   │   └── style.css          # Included in templates
│   └── uploads/                # Temporary audio storage
└── templates/
    ├── base.html              # Base template
    ├── index.html             # Upload interface
    └── results.html           # Results dashboard
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)
- 10GB+ free disk space

### Step 1: Clone/Download Project
```bash
# Create project directory
mkdir nexus_omega_flask
cd nexus_omega_flask

# Copy all provided files into this directory
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Download NLTK data (required)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 4: Create Required Directories
```bash
# Create upload directory
mkdir -p static/uploads
```

### Step 5: Run Application
```bash
python app.py
```

The application will start on `http://localhost:5000`

**First Launch Note:** The first time you run the app, it will download all AI models (~10GB). This may take 10-30 minutes depending on your internet speed.

---

## 🎮 Usage Guide

### Basic Workflow

1. **Upload Audio**
   - Navigate to `http://localhost:5000`
   - Select an audio file (WAV, MP3, or FLAC)
   - Choose spoken language (or use Auto-Detect)
   - Optionally enable "Transcribe directly to English"
   - Click "ANALYZE AUDIO"

2. **View Results**
   - Wait for analysis to complete (30-120 seconds)
   - View comprehensive metrics dashboard
   - Explore 5 tabs:
     - 📝 **Transcript**: Full text, summary, keywords, grammar analysis
     - 🌍 **Translate**: Multi-language translation with TTS
     - 📉 **Emotion Timeline**: Per-second emotion visualization
     - ☁️ **Word Cloud**: Visual keyword representation
     - 📄 **Download**: Professional PDF report

3. **Advanced Features**
   - **Translation**: Select target language and generate speech-to-speech translation
   - **Grammar Analysis**: View T5 AI corrections and scoring
   - **Topic Classification**: See zero-shot topic detection
   - **Communication Score**: Overall quality metric (0-100)

---

## 🔧 Configuration

All settings from the Streamlit version are preserved in `config.py`:

### Key Settings
```python
# Model Settings
DEFAULT_WHISPER_MODEL = 'medium'  # Options: base, small, medium

# Supported Languages
WHISPER_LANGUAGES = {
    "Auto-Detect": None,
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn",  # Fully supported
    "Tamil": "ta",
    "Telugu": "te",
    # ... more languages
}

# Audio Processing (PRESERVED)
AUDIO_SAMPLE_RATE = 16000
BUTTER_FILTER_ORDER = 10
BUTTER_FILTER_CUTOFF = 100  # Hz

# Communication Score Weights (PRESERVED)
SCORE_WEIGHT_NOISE = 30
SCORE_WEIGHT_WPM = 30
SCORE_WEIGHT_TOXICITY = 20
SCORE_WEIGHT_EMOTION = 20
OPTIMAL_WPM = 140
```

---

## 🧠 AI Models Used

All models from the Streamlit version are preserved:

1. **Whisper** (base/small/medium) - Speech-to-Text
2. **Wav2Vec2** (superb/wav2vec2-base-superb-er) - Emotion Detection
3. **DistilBERT** (SST-2) - Sentiment Analysis
4. **RoBERTa** (unitary/unbiased-toxic-roberta) - Toxicity Detection
5. **BART** (facebook/bart-large-cnn) - Summarization
6. **BART** (facebook/bart-large-mnli) - Topic Classification
7. **NLLB-200** (1.3B distilled) - Translation (Kannada support)
8. **T5** (vennify/t5-base-grammar-correction) - Grammar Correction

---

## 📊 Features Deep Dive

### 1. Speech Recognition
- **Whisper Model**: State-of-the-art ASR
- **Language Support**: 11+ languages including Kannada
- **Direct Translation**: Option to transcribe foreign audio directly to English

### 2. Emotion & Sentiment
- **Emotion Detection**: Smart heuristics with anger threshold (0.60)
- **Sentiment Analysis**: Positive/Negative classification
- **Toxicity Scoring**: Safe/Toxic detection

### 3. Acoustic Analysis
- **Noise Level**: Low/Medium/High classification
- **WPM (Words Per Minute)**: Speaking speed analysis
- **Speaking Time**: Active speech duration (silence removed)
- **Audio Cleaning**: Butterworth high-pass filter at 100Hz

### 4. Grammar & Quality
- **T5 Grammar Correction**: AI-powered grammar improvements
- **Grammar Score**: 0-100 quality index
- **Change Highlighting**: Visual diff of corrections
- **Communication Score**: Weighted formula combining multiple metrics

### 5. Translation & TTS
- **NLLB 1.3B**: High-accuracy translation (Kannada support)
- **Beam Search**: 5 beams for better translation quality
- **Text-to-Speech**: gTTS integration for 11+ languages
- **Speech-to-Speech**: Complete audio translation pipeline

### 6. Visualizations
- **Emotion Timeline**: Per-second emotion intensity plot
- **Word Cloud**: Keyword frequency visualization
- **PDF Reports**: Professional ReportLab-generated documents

---

## 🔄 Streamlit → Flask Mapping

### UI Component Mapping

| Streamlit | Flask Equivalent | Notes |
|-----------|------------------|-------|
| `st.file_uploader` | `<input type="file">` | HTML5 file input |
| `st.button` | `<button>` | Form submission |
| `st.selectbox` | `<select>` | Dropdown menu |
| `st.checkbox` | `<input type="checkbox">` | Boolean input |
| `st.session_state` | `session` | Flask session |
| `st.write` / `st.markdown` | Jinja2 templates | Template rendering |
| `st.pyplot` | Matplotlib → Base64 | Image encoding |
| `st.spinner` | JavaScript spinner | Custom loading UI |
| `st.columns` | CSS Grid | Responsive layout |
| `st.tabs` | JavaScript tabs | Tab switching |

### Processing Logic
- **100% PRESERVED**: All analysis functions are exact copies
- **No Changes**: Thresholds, weights, formulas are identical
- **Model Loading**: Same caching strategy, just different framework

---

## 🧪 Testing Checklist

After installation, verify all features:

- [ ] Upload WAV/MP3/FLAC file
- [ ] Transcription works (test multiple languages)
- [ ] Emotion detection returns valid emotion
- [ ] Sentiment analysis works
- [ ] Toxicity detection functional
- [ ] Summary generates for text >20 words
- [ ] Keywords extracted correctly
- [ ] Grammar correction produces T5 output
- [ ] Topic classification returns category
- [ ] Communication score calculated (0-100)
- [ ] Translation works (test Kannada)
- [ ] TTS audio plays correctly
- [ ] Emotion timeline generates
- [ ] Word cloud displays
- [ ] PDF downloads successfully

---

## 🐛 Troubleshooting

### Models Not Loading
```bash
# Clear cache and reinstall
pip uninstall transformers torch
pip install transformers torch --no-cache-dir
```

### CUDA Out of Memory
```python
# Edit config.py
DEFAULT_WHISPER_MODEL = 'base'  # Use smaller model
```

### NLTK Data Missing
```bash
python -c "import nltk; nltk.download('all')"
```

### Port Already in Use
```bash
# Change port in app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

---

## 📝 Academic Notes

This project is suitable for:
- **VTU CSE Final Year Project**
- **ML/NLP Research Project**
- **Speech Analysis System**
- **Multi-Modal AI Application**

### Key Academic Highlights
1. **Multi-Model Integration**: 8+ AI models working together
2. **Production-Ready Code**: Modular, documented, scalable
3. **Real-World Application**: Practical speech intelligence
4. **Full-Stack Implementation**: Backend + Frontend + ML
5. **Advanced NLP**: Translation, Grammar, Topic Classification

---

## 📄 License & Credits

- **Original Streamlit App**: Provided by user
- **Flask Conversion**: Exact functionality preservation
- **AI Models**: HuggingFace Transformers, OpenAI Whisper
- **Purpose**: Educational/Academic use

---

## 🚨 Important Notes

### What Was NOT Changed
✅ All ML model inference logic  
✅ All scoring formulas and weights  
✅ All thresholds and hyperparameters  
✅ All audio processing algorithms  
✅ All analysis functions  
✅ All visualization logic  

### What WAS Changed
❌ Streamlit UI components → Flask/HTML/CSS  
❌ Session management (st.session_state → Flask session)  
❌ File handling (Streamlit uploader → Werkzeug)  

### Result
**Zero functionality loss. Zero accuracy degradation. 100% feature parity.**

---

## 📧 Support

For issues related to:
- **Model Loading**: Check GPU/CUDA setup
- **Dependencies**: Verify Python 3.9+ and all requirements installed
- **Performance**: Consider smaller Whisper model or CPU mode

---

**Built with Flask • Powered by AI • Academic Excellence**
