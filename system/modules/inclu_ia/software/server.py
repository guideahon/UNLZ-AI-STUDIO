import os
import sys
import threading
import time
import queue
import logging
import argparse
import socket

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from faster_whisper import WhisperModel
import speech_recognition as sr
import numpy as np

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IncluIA-Server")

# Absolute paths for assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "../web/templates")
STATIC_DIR = os.path.join(BASE_DIR, "../web/static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['SECRET_KEY'] = 'incluia_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global State
TRANSCRIPTION_QUEUE = queue.Queue()
STOP_EVENT = threading.Event()
MODEL_SIZE = "tiny"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

# --- Audio Capture & Transcription ---
def audio_worker(model_size, device):
    logger.info(f"Loading Whisper Model: {model_size} on {device}...")
    try:
        model = WhisperModel(model_size, device=device, compute_type="int8")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = 1000  # Adjust based on mic

    # We use SpeechRecognition for convenience in handling mics
    # But faster-whisper needs raw audio or file.
    # We will use SR to detect phrases and then pass buffer to Whisper.
    
    # Simpler approach for demo: Use SR to record, save to temp wav, transcribe?
    # Or use PyAudio stream directly. 
    # Let's use SR's listen_in_background if possible, but passing audio to faster-whisper is non-trivial without saving file.
    # Alternative: Record chunks.
    
    mic = sr.Microphone()
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        logger.info("Microphone ready. Listening...")
        
        while not STOP_EVENT.is_set():
            try:
                # Listen for a phrase
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                # Convert to wav bytes
                wav_data = audio.get_wav_data()
                
                # Write to temp file (fastest way to bridge sr -> faster_whisper reliably)
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(wav_data)
                    tmp_path = tmp.name
                
                # Transcribe
                segments, _ = model.transcribe(tmp_path, beam_size=5, language="es")
                text = " ".join([segment.text for segment in segments]).strip()
                
                # Clean up
                try: os.remove(tmp_path)
                except: pass
                
                if text:
                    logger.info(f"Transcribed: {text}")
                    socketio.emit('new_caption', {'text': text})
            
            except sr.WaitTimeoutError:
                pass # Silence
            except Exception as e:
                logger.error(f"Error in audio loop: {e}")
                time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    emit('status', {'msg': 'Connected to Inclu-IA Server'})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", type=str, default="tiny")
    args = parser.parse_args()
    
    # Start Audio Thread
    t = threading.Thread(target=audio_worker, args=(args.model, "cpu"), daemon=True)
    t.start()
    
    ip = get_local_ip()
    print(f"Server starting on http://{ip}:{args.port}")
    
    try:
        socketio.run(app, host='0.0.0.0', port=args.port, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        STOP_EVENT.set()
