import os
import sys
import logging
import argparse
import tempfile
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ALM-Server")

app = Flask(__name__)

# Global Model
model = None

def load_model(model_path, device="cpu", compute_type="int8"):
    global model
    logger.info(f"Loading Whisper model from {model_path} on {device} ({compute_type})...")
    try:
        model = WhisperModel(model_path, device=device, compute_type=compute_type)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

@app.route('/health', methods=['GET'])
def health():
    if model:
        return jsonify({"status": "ok", "model": "loaded"}), 200
    return jsonify({"status": "error", "message": "model not loaded"}), 503

@app.route('/v1/audio/transcriptions', methods=['POST'])
def transcribe():
    if not model:
        return jsonify({"error": "Model not loaded"}), 503
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save to temp file because faster-whisper needs path or file-like object
        # Using temp file is safer for various formats
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        logger.info(f"Transcribing {file.filename}...")
        segments, info = model.transcribe(tmp_path, beam_size=5)
        
        text = "".join([segment.text for segment in segments]).strip()
        
        # Cleanup
        os.remove(tmp_path)
        
        return jsonify({
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration
        })

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model-path", type=str, required=True, help="Path to faster-whisper model directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    
    args = parser.parse_args()
    
    # Run
    load_model(args.model_path, args.device)
    app.run(host='0.0.0.0', port=args.port)
