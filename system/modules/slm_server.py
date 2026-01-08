import os
import sys
import logging
import argparse
import tempfile
import torch
from flask import Flask, request, jsonify, send_file
from TTS.api import TTS

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SLM-Server")

app = Flask(__name__)

# Global Model
tts_model = None
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "slm_outputs")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_model(model_path, device="cpu"):
    global tts_model
    logger.info(f"Loading XTTS model from {model_path} on {device}...")
    try:
        # XTTS-v2 usually loads by model_name or path. 
        # If model_path is a directory, we might need to specify config/vocab.
        # Allowing 'tts_models/multilingual/multi-dataset/xtts_v2' generic load 
        # but configured to use local path if possible.
        
        # For this implementation, we assume model_path points to the folder containing config.json
        # TTS api usually expects model_name for auto-download, or model_path/config_path for manual.
        # We will try initializing with model_path as model_path argument if it exists.
        
        # Simple init:
        tts_model = TTS(model_path=model_path, config_path=os.path.join(model_path, "config.json"), progress_bar=False, gpu=(device=="cuda"))
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Fallback to standard init if specific path fails, or exit
        sys.exit(1)

@app.route('/health', methods=['GET'])
def health():
    if tts_model:
        return jsonify({"status": "ok", "model": "loaded"}), 200
    return jsonify({"status": "error", "message": "model not loaded"}), 503

@app.route('/v1/audio/speech', methods=['POST'])
def speech():
    if not tts_model:
        return jsonify({"error": "Model not loaded"}), 503
    
    data = request.json
    if not data or 'input' not in data:
        return jsonify({"error": "Missing 'input' text"}), 400
    
    text = data['input']
    # Default speaker reference if none provided (XTTS needs a reference audio)
    # We should have a default speaker file in assets or model dir
    speaker_wav = data.get('speaker_wav', None) 
    language = data.get('language', 'es')
    
    if not speaker_wav:
        # Try to find a default speaker in the model dir
        # Or require it. For now, let's return error if no speaker provided and no default found.
        # But for ease of use, let's create a dummy or use a known path if available.
        # Ideally user provides path or we rely on pre-cloned voices.
        return jsonify({"error": "XTTS requires 'speaker_wav' path (reference audio)"}), 400

    try:
        output_path = os.path.join(OUTPUT_DIR, f"output_{os.urandom(4).hex()}.wav")
        
        logger.info(f"Generating speech for: '{text[:20]}...'")
        tts_model.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=output_path)
        
        return send_file(output_path, mimetype="audio/wav")

    except Exception as e:
        logger.error(f"TTS error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--model-path", type=str, required=True, help="Path to XTTS model directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    
    args = parser.parse_args()
    
    # Run
    load_model(args.model_path, args.device)
    app.run(host='0.0.0.0', port=args.port)
