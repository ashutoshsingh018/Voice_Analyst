"""
Audio Processing Utilities
EXACT COPY of audio processing logic from Streamlit - NO CHANGES
+ NEW: convert_audio_format() for browser-recorded audio (WebM/OGG → WAV)
"""

import numpy as np
import librosa
import scipy.signal
from scipy.io.wavfile import write
from config import Config
import subprocess
import os

def convert_audio_format(input_path, output_path=None):
    """
    Convert browser-recorded audio (WebM/OGG) to WAV format
    This is needed for microphone recordings from MediaRecorder API
    
    NEW FUNCTION - Enables real-time recording feature
    Does NOT modify existing audio processing pipeline
    
    Args:
        input_path: Path to input audio file (any format)
        output_path: Path for output WAV file (optional)
    
    Returns:
        Path to converted WAV file, or original path if already WAV
    """
    # Check if file is already WAV
    if input_path.lower().endswith('.wav'):
        return input_path
    
    try:
        # Generate output path if not provided
        if output_path is None:
            base = os.path.splitext(input_path)[0]
            output_path = f"{base}_converted.wav"
        
        # Use librosa for conversion (handles most formats)
        y, sr = librosa.load(input_path, sr=Config.AUDIO_SAMPLE_RATE, mono=True)
        
        # Normalize
        y = librosa.util.normalize(y)
        
        # Save as WAV
        write(output_path, Config.AUDIO_SAMPLE_RATE, y)
        
        print(f"[AudioProcessor] Converted {input_path} → {output_path}")
        return output_path
        
    except Exception as e:
        print(f"[AudioProcessor] Conversion failed with librosa: {e}")
        
        # Fallback: Try ffmpeg if available
        try:
            subprocess.run([
                'ffmpeg', '-i', input_path,
                '-ar', str(Config.AUDIO_SAMPLE_RATE),
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                output_path
            ], check=True, capture_output=True)
            
            print(f"[AudioProcessor] Converted with ffmpeg: {input_path} → {output_path}")
            return output_path
            
        except (subprocess.CalledProcessError, FileNotFoundError) as ffmpeg_error:
            print(f"[AudioProcessor] FFmpeg conversion failed: {ffmpeg_error}")
            # Return original if all conversions fail
            return input_path

def clean_audio(audio_path):
    """
    Audio cleaning with Butterworth filter
    PRESERVED EXACTLY from Streamlit version
    """
    try:
        y, sr = librosa.load(audio_path, sr=Config.AUDIO_SAMPLE_RATE)
        
        # Apply 10th order Butterworth High-pass filter at 100 Hz (PRESERVED)
        sos = scipy.signal.butter(
            Config.BUTTER_FILTER_ORDER, 
            Config.BUTTER_FILTER_CUTOFF, 
            'hp', 
            fs=sr, 
            output='sos'
        )
        cleaned = scipy.signal.sosfilt(sos, y)
        cleaned = librosa.util.normalize(cleaned)
        write(audio_path, sr, cleaned)
        return True
    except Exception as e:
        print(f"Audio cleaning failed: {e}")
        return False

def estimate_noise_level(audio_path):
    """
    Estimate noise level from audio
    PRESERVED EXACTLY - including thresholds
    """
    try:
        y, sr = librosa.load(audio_path, sr=Config.AUDIO_SAMPLE_RATE)
        
        # Estimate noise from the first 0.25s (PRESERVED)
        noise = np.mean(np.abs(y[:int(sr * Config.NOISE_ESTIMATION_DURATION)]))
        speech = np.mean(np.abs(y))
        ratio = min(noise / (speech + 1e-5), 1.0)  # Noise-to-Signal ratio
        
        # Classification thresholds (PRESERVED)
        if ratio < 0.2:
            level = "Low"
        elif ratio < 0.5:
            level = "Medium"
        else:
            level = "High"
        
        return level, ratio
    except Exception as e:
        print(f"Noise estimation failed: {e}")
        return "Unknown", 0

def calculate_wpm(text, audio_path):
    """
    Calculate Words Per Minute
    PRESERVED EXACTLY
    """
    try:
        y, sr = librosa.load(audio_path, sr=Config.AUDIO_SAMPLE_RATE)
        duration = len(y) / sr
        wpm = (len(text.split()) / duration) * 60
        return round(wpm, 1)
    except Exception as e:
        print(f"WPM calculation failed: {e}")
        return 0

def detect_speaking_time(audio_path):
    """
    Detect actual speaking time (excluding silence)
    PRESERVED EXACTLY - including top_db threshold
    """
    try:
        y, sr = librosa.load(audio_path, sr=Config.AUDIO_SAMPLE_RATE)
        
        # Split audio by silence (top_db=30 PRESERVED)
        intervals = librosa.effects.split(y, top_db=Config.SILENCE_TOP_DB)
        seconds = sum((i[1] - i[0]) for i in intervals) / sr
        return round(seconds, 2)
    except Exception as e:
        print(f"Speaking time detection failed: {e}")
        return 0

def generate_emotion_timeline(audio_path, emotion_model, max_duration=60):
    """
    Generate emotion timeline (per-second analysis)
    PRESERVED EXACTLY from Streamlit version
    """
    import tempfile
    import os
    
    try:
        y, sr = librosa.load(audio_path, sr=Config.AUDIO_SAMPLE_RATE)
        step = int(1.0 * sr)  # 1 second (PRESERVED)
        timeline = []
        
        for i in range(0, min(len(y), max_duration * sr), step):
            chunk = y[i:i + step]
            if len(chunk) > sr * 0.5:
                tf = tempfile.mktemp(suffix=".wav")
                write(tf, sr, chunk)
                try:
                    # Get the top confidence score from the chunk (PRESERVED)
                    score = emotion_model(tf)[0]["score"]
                    timeline.append(score)
                except:
                    timeline.append(0.0)
                finally:
                    if os.path.exists(tf):
                        os.remove(tf)
        
        return timeline
    except Exception as e:
        print(f"Timeline generation failed: {e}")
        return []

def validate_audio_file(file):
    """Validate uploaded audio file"""
    if file and '.' in file.filename:
        ext = file.filename.rsplit('.', 1)[1].lower()
        return ext in Config.ALLOWED_EXTENSIONS
    return False