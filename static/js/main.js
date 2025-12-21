/**
 * NEXUS OMEGA - Real-Time Audio Recorder
 * 
 * This module adds the missing microphone recording functionality
 * that was present in the Streamlit version.
 * 
 * Architecture:
 * Browser MediaRecorder → Flask /record_audio → Existing ML Pipeline
 * 
 * PRESERVES: All existing functionality
 * ADDS: Real-time microphone input (matching Streamlit's st.audio_input)
 */

class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.stream = null;
        this.isRecording = false;
        this.recordingStartTime = null;
        this.timerInterval = null;
        
        // Audio constraints for optimal quality (matching Streamlit's 16kHz)
        this.constraints = {
            audio: {
                channelCount: 1,  // Mono (matching audio processing pipeline)
                sampleRate: 16000,  // 16kHz (matching AUDIO_SAMPLE_RATE in config.py)
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        };
    }

    /**
     * Initialize microphone access
     * Requests user permission for microphone
     */
    async initialize() {
        try {
            console.log('[AudioRecorder] Requesting microphone access...');
            this.stream = await navigator.mediaDevices.getUserMedia(this.constraints);
            console.log('[AudioRecorder] Microphone access granted');
            return true;
        } catch (error) {
            console.error('[AudioRecorder] Microphone access denied:', error);
            this.showError('Microphone access denied. Please allow microphone permissions.');
            return false;
        }
    }

    /**
     * Start recording audio
     * @param {number} maxDuration - Maximum recording duration in seconds (default: 60)
     */
    async startRecording(maxDuration = 60) {
        if (this.isRecording) {
            console.warn('[AudioRecorder] Already recording');
            return;
        }

        // Initialize if not already done
        if (!this.stream) {
            const initialized = await this.initialize();
            if (!initialized) return;
        }

        try {
            // Reset chunks
            this.audioChunks = [];

            // Determine MIME type (prefer WAV for compatibility)
            const mimeType = this.getSupportedMimeType();
            console.log(`[AudioRecorder] Using MIME type: ${mimeType}`);

            // Create MediaRecorder
            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType: mimeType,
                audioBitsPerSecond: 128000  // 128 kbps for good quality
            });

            // Handle data available
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                    console.log(`[AudioRecorder] Chunk received: ${event.data.size} bytes`);
                }
            };

            // Handle recording stop
            this.mediaRecorder.onstop = () => {
                console.log('[AudioRecorder] Recording stopped');
                this.processRecording();
            };

            // Handle errors
            this.mediaRecorder.onerror = (event) => {
                console.error('[AudioRecorder] Recording error:', event.error);
                this.showError('Recording error occurred. Please try again.');
                this.stopRecording();
            };

            // Start recording
            this.mediaRecorder.start(1000);  // Collect data every 1 second
            this.isRecording = true;
            this.recordingStartTime = Date.now();

            // Update UI
            this.updateUI('recording');
            this.startTimer();

            // Auto-stop after maxDuration
            setTimeout(() => {
                if (this.isRecording) {
                    console.log(`[AudioRecorder] Auto-stopping after ${maxDuration} seconds`);
                    this.stopRecording();
                }
            }, maxDuration * 1000);

            console.log(`[AudioRecorder] Recording started (max ${maxDuration}s)`);
        } catch (error) {
            console.error('[AudioRecorder] Failed to start recording:', error);
            this.showError('Failed to start recording. Please check your microphone.');
        }
    }

    /**
     * Stop recording audio
     */
    stopRecording() {
        if (!this.isRecording || !this.mediaRecorder) {
            console.warn('[AudioRecorder] Not recording');
            return;
        }

        try {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.stopTimer();
            this.updateUI('stopped');
            console.log('[AudioRecorder] Recording stopped by user');
        } catch (error) {
            console.error('[AudioRecorder] Error stopping recording:', error);
        }
    }

    /**
     * Process recorded audio and send to backend
     */
    async processRecording() {
        if (this.audioChunks.length === 0) {
            console.error('[AudioRecorder] No audio data recorded');
            this.showError('No audio data recorded. Please try again.');
            return;
        }

        try {
            // Create blob from chunks
            const mimeType = this.mediaRecorder.mimeType;
            const audioBlob = new Blob(this.audioChunks, { type: mimeType });
            
            const durationMs = Date.now() - this.recordingStartTime;
            const durationSec = (durationMs / 1000).toFixed(1);
            
            console.log(`[AudioRecorder] Recorded ${durationSec}s, size: ${(audioBlob.size / 1024).toFixed(2)} KB`);

            // Show preview (optional)
            this.showAudioPreview(audioBlob);

            // Send to backend for analysis
            await this.sendToBackend(audioBlob, mimeType);

        } catch (error) {
            console.error('[AudioRecorder] Error processing recording:', error);
            this.showError('Failed to process recording. Please try again.');
        }
    }

    /**
     * Send recorded audio to Flask backend
     * @param {Blob} audioBlob - Recorded audio blob
     * @param {string} mimeType - MIME type of audio
     */
    async sendToBackend(audioBlob, mimeType) {
        // Get analysis parameters from form (matching file upload flow)
        const language = document.getElementById('language')?.value || 'Auto-Detect';
        const translateDirect = document.getElementById('translateDirect')?.checked || false;

        // Create FormData
        const formData = new FormData();
        
        // Determine file extension from MIME type
        const extension = this.getExtensionFromMimeType(mimeType);
        const filename = `recording_${Date.now()}.${extension}`;
        
        formData.append('audio_file', audioBlob, filename);
        formData.append('language', language);
        formData.append('translate_direct', translateDirect);

        // Show loading
        this.updateUI('analyzing');

        try {
            console.log('[AudioRecorder] Sending audio to backend...');

            // Step 1: Upload audio (using existing /upload endpoint)
            const uploadResponse = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) {
                const error = await uploadResponse.json();
                throw new Error(error.error || 'Upload failed');
            }

            const uploadResult = await uploadResponse.json();
            console.log('[AudioRecorder] Upload successful:', uploadResult);

            // Step 2: Analyze (using existing /analyze endpoint)
            const analyzeFormData = new FormData();
            analyzeFormData.append('language', language);
            analyzeFormData.append('translate_direct', translateDirect);

            const analyzeResponse = await fetch('/analyze', {
                method: 'POST',
                body: analyzeFormData
            });

            if (!analyzeResponse.ok) {
                const error = await analyzeResponse.json();
                throw new Error(error.error || 'Analysis failed');
            }

            const analyzeResult = await analyzeResponse.json();
            console.log('[AudioRecorder] Analysis complete:', analyzeResult);

            // Redirect to results page (matching file upload flow)
            window.location.href = '/results';

        } catch (error) {
            console.error('[AudioRecorder] Backend error:', error);
            this.showError('Analysis failed: ' + error.message);
            this.updateUI('error');
        }
    }

    /**
     * Show audio preview player
     * @param {Blob} audioBlob - Audio blob to preview
     */
    showAudioPreview(audioBlob) {
        const previewContainer = document.getElementById('audioPreview');
        if (!previewContainer) return;

        // Create audio element
        const audioUrl = URL.createObjectURL(audioBlob);
        previewContainer.innerHTML = `
            <div class="audio-preview">
                <p style="color: var(--success-color); font-weight: 600; margin-bottom: 10px;">
                    ✓ Recording Complete
                </p>
                <audio controls style="width: 100%;">
                    <source src="${audioUrl}" type="${audioBlob.type}">
                    Your browser does not support audio playback.
                </audio>
                <p style="font-size: 12px; color: var(--text-muted); margin-top: 10px;">
                    Sending to AI for analysis...
                </p>
            </div>
        `;
        previewContainer.classList.remove('hidden');
    }

    /**
     * Update UI based on recorder state
     * @param {string} state - Current state: 'idle', 'recording', 'stopped', 'analyzing', 'error'
     */
    updateUI(state) {
        const startBtn = document.getElementById('startRecording');
        const stopBtn = document.getElementById('stopRecording');
        const statusText = document.getElementById('recordingStatus');
        const timerDisplay = document.getElementById('recordingTimer');

        if (!startBtn || !stopBtn) return;

        switch (state) {
            case 'recording':
                startBtn.disabled = true;
                startBtn.style.opacity = '0.5';
                stopBtn.disabled = false;
                stopBtn.style.opacity = '1';
                if (statusText) statusText.textContent = '🔴 Recording...';
                if (timerDisplay) timerDisplay.classList.remove('hidden');
                break;

            case 'stopped':
                startBtn.disabled = false;
                startBtn.style.opacity = '1';
                stopBtn.disabled = true;
                stopBtn.style.opacity = '0.5';
                if (statusText) statusText.textContent = '⏹️ Recording stopped';
                if (timerDisplay) timerDisplay.classList.add('hidden');
                break;

            case 'analyzing':
                startBtn.disabled = true;
                stopBtn.disabled = true;
                if (statusText) statusText.textContent = '🤖 AI analyzing...';
                break;

            case 'error':
                startBtn.disabled = false;
                startBtn.style.opacity = '1';
                stopBtn.disabled = true;
                stopBtn.style.opacity = '0.5';
                if (statusText) statusText.textContent = '❌ Error occurred';
                if (timerDisplay) timerDisplay.classList.add('hidden');
                break;

            default: // idle
                startBtn.disabled = false;
                startBtn.style.opacity = '1';
                stopBtn.disabled = true;
                stopBtn.style.opacity = '0.5';
                if (statusText) statusText.textContent = '🎙️ Ready to record';
                if (timerDisplay) timerDisplay.classList.add('hidden');
        }
    }

    /**
     * Start recording timer display
     */
    startTimer() {
        const timerDisplay = document.getElementById('recordingTimer');
        if (!timerDisplay) return;

        this.timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.recordingStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
            const seconds = (elapsed % 60).toString().padStart(2, '0');
            timerDisplay.textContent = `${minutes}:${seconds}`;
        }, 1000);
    }

    /**
     * Stop recording timer
     */
    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    /**
     * Get supported MIME type for recording
     * @returns {string} Supported MIME type
     */
    getSupportedMimeType() {
        const types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/mp4',
            'audio/mpeg'
        ];

        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }

        // Fallback
        return 'audio/webm';
    }

    /**
     * Get file extension from MIME type
     * @param {string} mimeType - MIME type
     * @returns {string} File extension
     */
    getExtensionFromMimeType(mimeType) {
        const mimeToExt = {
            'audio/webm': 'webm',
            'audio/ogg': 'ogg',
            'audio/mp4': 'mp4',
            'audio/mpeg': 'mp3',
            'audio/wav': 'wav'
        };

        for (const [mime, ext] of Object.entries(mimeToExt)) {
            if (mimeType.includes(mime)) {
                return ext;
            }
        }

        return 'webm';  // Default
    }

    /**
     * Show error message to user
     * @param {string} message - Error message
     */
    showError(message) {
        // Use existing showAlert function from base template
        if (typeof showAlert === 'function') {
            showAlert(message, 'danger');
        } else {
            alert(message);
        }
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
        }
        this.isRecording = false;
        this.audioChunks = [];
    }
}

// Global instance
let audioRecorder = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('[AudioRecorder] Initializing...');
    audioRecorder = new AudioRecorder();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (audioRecorder) {
        audioRecorder.cleanup();
    }
});