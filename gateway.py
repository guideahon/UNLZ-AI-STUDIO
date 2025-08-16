import os, sys, time, base64, tempfile, shutil, subprocess, signal
from pathlib import Path
from typing import Optional, Dict, Any
import psutil
import httpx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import uvicorn

# =========================
# CONFIG - AJUSTÁ ESTAS RUTAS
# =========================
#LLAMA_SERVER_EXE = r"C:\llama-cuda\llama-server.exe"  # Cuda
LLAMA_SERVER_EXE = "llama-server"  # Vulkan
LLAMA_MODEL      = r"C:\models\qwen3-coder-30b\Qwen3-Coder-30B-A3B-Instruct.Q5_K_M.gguf"
LLAMA_HOST       = "127.0.0.1"
LLAMA_PORT       = 8080
LLAMA_ARGS = [
    "-m", LLAMA_MODEL,
    "--host", LLAMA_HOST, "--port", str(LLAMA_PORT),
    "--ctx-size", "8192",
    "--n-gpu-layers", "35",
    "-t", "12" #,       # si usás Vulkan, quitá esta coma, si usas CUDA, dejala
    #"--flash-attn"     # si usás Vulkan, quitá este flag, si usas CUDA, dejalo
]

LMDEPLOY_CMD     = [sys.executable, "-m", "lmdeploy"]
VLM_MODEL_PATH = r"C:\models\qwen2.5-vl-7b-hf"
VLM_HOST         = "127.0.0.1"
VLM_PORT         = 9090
VLM_ARGS = ["serve", "api_server", "--model-path", VLM_MODEL_PATH, "--server-port", str(VLM_PORT)]

# STT (usar CUDA o CPU)
USE_CUDA_FOR_STT = False  # True usa GPU; False evita pelear VRAM con LLM
STT_MODEL_SIZE   = "medium"  # "small" si querés menos consumo
STT_DEVICE       = "cuda" if USE_CUDA_FOR_STT else "cpu"
STT_COMPUTE_TYPE = "float16" if USE_CUDA_FOR_STT else "int8"  # int8 en CPU va muy bien

# TTS - Piper
PIPER_EXE  = "piper"
PIPER_VOICE= r"C:\piper\voices\es_AR\daniela_high\es_AR-daniela-high.onnx"

# =========================
# MANAGER DE PROCESOS GPU
# =========================
class GpuProcessManager:
    def __init__(self):
        self.mode = None  # "llm" | "vlm" | None
        self.proc = None

    def _kill_tree(self, p: psutil.Process):
        for c in p.children(recursive=True):
            try: c.kill()
            except: pass
        try: p.kill()
        except: pass

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                p = psutil.Process(self.proc.pid)
                self._kill_tree(p)
            except: pass
        self.proc = None
        self.mode = None

    def ensure_mode(self, mode: str):
        assert mode in ("llm", "vlm")
        if self.mode == mode and self.proc and self.proc.poll() is None:
            return  # ya está
        # si hay algo distinto, matar
        self.stop()
        time.sleep(1.5)  # pequeña espera para liberar VRAM
        # lanzar
        if mode == "llm":
            print("Starting llama-server...")
            self.proc = subprocess.Popen([LLAMA_SERVER_EXE] + LLAMA_ARGS, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            # esperar a que levante
            self._wait_http_up(f"http://{LLAMA_HOST}:{LLAMA_PORT}/health", fallback=f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1/models")
        else:
            print("Starting lmdeploy api_server...")
            self.proc = subprocess.Popen(LMDEPLOY_CMD + VLM_ARGS, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            self._wait_http_up(f"http://{VLM_HOST}:{VLM_PORT}/v1/models")
        self.mode = mode

    def _wait_http_up(self, url: str, fallback: Optional[str]=None, timeout=60):
        deadline = time.time()+timeout
        with httpx.Client(timeout=2.0) as client:
            while time.time()<deadline:
                try:
                    r = client.get(url)
                    if r.status_code<500:
                        return
                except:
                    pass
                if fallback:
                    try:
                        r = client.get(fallback)
                        if r.status_code<500:
                            return
                    except:
                        pass
                time.sleep(1)
        # si no levantó igual seguimos (el backend podría tardar más)

manager = GpuProcessManager()
app = FastAPI(title="IA Gateway (LLM/VLM/ALM)")

# Pre-carga STT (en CPU por defecto para evitar conflictos)
print("Loading WhisperModel (STT)...")
stt_model = WhisperModel(STT_MODEL_SIZE, device=STT_DEVICE, compute_type=STT_COMPUTE_TYPE)

# =========================
# /llm  -> proxyea a llama.cpp
# =========================
@app.post("/llm")
async def llm_chat(payload: Dict[str, Any]):
    manager.ensure_mode("llm")
    async with httpx.AsyncClient(timeout=None) as client:
        url = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1/chat/completions"
        r = await client.post(url, json=payload)
        return JSONResponse(status_code=r.status_code, content=r.json())

# =========================
# /vlm -> proxyea a LMDeploy (Qwen-VL)
# payload estilo OpenAI chat.completions con image_url
# =========================
@app.post("/vlm")
async def vlm_chat(payload: Dict[str, Any]):
    manager.ensure_mode("vlm")
    async with httpx.AsyncClient(timeout=None) as client:
        url = f"http://{VLM_HOST}:{VLM_PORT}/v1/chat/completions"
        r = await client.post(url, json=payload)
        return JSONResponse(status_code=r.status_code, content=r.json())

# =========================
# /alm -> audio pipeline (STT -> LLM -> opcional TTS)
# Enviar multipart/form-data: file=@audio.wav, prompt_text (opcional), tts=true/false
# =========================
@app.post("/alm")
async def alm_pipeline(
    file: UploadFile = File(...),
    system_prompt: Optional[str] = Form("You are a helpful assistant."),
    tts: Optional[bool] = Form(True),
    target_lang: Optional[str] = Form("es")
):
    # 1) STT (por defecto CPU para no pelear VRAM; si USE_CUDA_FOR_STT=True, usará GPU)
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "audio").suffix or ".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        wav_path = tmp.name

    segments, info = stt_model.transcribe(wav_path, language=target_lang, vad=True)
    user_text = "".join([s.text for s in segments]).strip()
    os.unlink(wav_path)

    # 2) LLM (texto→texto)
    manager.ensure_mode("llm")
    llm_payload = {
        "model": "qwen3-coder-30b",
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
    }
    async with httpx.AsyncClient(timeout=None) as client:
        url = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1/chat/completions"
        r = await client.post(url, json=llm_payload)
        r.raise_for_status()
        llm_resp = r.json()
        answer_text = llm_resp["choices"][0]["message"]["content"]

    # 3) TTS (opcional, con Piper a WAV y se devuelve base64)
    audio_b64 = None
    if tts:
        try:
            proc = subprocess.Popen(
                [PIPER_EXE, "-m", PIPER_VOICE, "-f", "-"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            out, _ = proc.communicate(answer_text.encode("utf-8"))
            audio_b64 = "data:audio/wav;base64," + base64.b64encode(out).decode("ascii")
        except Exception as e:
            audio_b64 = None

    return {
        "stt_text": user_text,
        "llm_text": answer_text,
        "tts_audio": audio_b64
    }

@app.get("/health")
def health():
    return {"status":"ok","mode": manager.mode}

if __name__ == "__main__":
    uvicorn.run("gateway:app", host="0.0.0.0", port=8000, reload=False)
