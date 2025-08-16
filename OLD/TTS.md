TTS: texto → voz (Kokoro o Piper)

Más simple y liviano: Piper binario; mejor calidad en inglés: Kokoro (hay modelos ES buenos también).

Opción rápida Piper (Windows)

Descargá Piper y una voz ES (por ej. es_ES-...) y ponelos en C:\piper\.

Servidor FastAPI tts_server.py:

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import subprocess, io, uvicorn

app = FastAPI()
PIPER = r"C:\piper\piper.exe"
VOICE = r"C:\piper\es_ES-...-medium.onnx"  # poné el nombre de tu voz

@app.post("/v1/audio/speech")
def tts(payload: dict):
    text = payload.get("input") or payload.get("text")
    if not text:
        return {"error": "Missing 'input' text"}
    # Piper escribe WAV a stdout
    proc = subprocess.Popen([PIPER, "-m", VOICE, "-f", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, _ = proc.communicate(text.encode("utf-8"))
    return StreamingResponse(io.BytesIO(out), media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7071)


Correr:

python .\tts_server.py