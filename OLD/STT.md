STT: voz → texto (faster-whisper, CUDA)
Instalar
pip install -U faster-whisper uvicorn fastapi python-multipart

Servidor FastAPI (guárdalo como stt_server.py)
from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel
import uvicorn, tempfile, shutil

app = FastAPI()
model = WhisperModel("medium", device="cuda", compute_type="float16")  # "small" si querés menos VRAM/latencia

@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), language: str = "es"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name
    segments, info = model.transcribe(path, language=language, vad=True)
    text = "".join([s.text for s in segments])
    return {"text": text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7070)

Ejecutar
python .\stt_server.py