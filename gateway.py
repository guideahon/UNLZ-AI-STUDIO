import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"ctranslate2(\.|$)")
import os, sys, time, base64, tempfile, shutil, subprocess, threading
from pathlib import Path
from typing import Optional, Dict, Any
import psutil
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import uvicorn

# --- Logs y TMP dirs (deben existir antes de usarlos) ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Temp dedicado para Piper
TTS_TMP_DIR = LOG_DIR / "piper_tmp"           # opción A: dentro de logs/
# TTS_TMP_DIR = Path(tempfile.gettempdir()) / "unlz_piper_tmp"  # opción B: en %TEMP%
TTS_TMP_DIR.mkdir(parents=True, exist_ok=True)


# --- VLM HuggingFace backend imports ---
import requests
from io import BytesIO
from PIL import Image
import traceback
from urllib.parse import urlparse


# Fallback image if payload URLs fail or are missing
DEFAULT_IMAGE_URL = os.environ.get(
    "DEFAULT_IMAGE_URL",
    "https://vectorft.com/images/favicon.ico?80172489139"
)

VLM_BACKEND = os.environ.get("VLM_BACKEND", "hf")  # "hf" | "lmdeploy"

# Lazy-load state for HF VLM
_hf_vlm = {"model": None, "processor": None, "device": None}
def _ensure_hf_vlm():
    if _hf_vlm["model"] is not None:
        return _hf_vlm
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import torch
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    ).eval()
    _hf_vlm.update(model=model, processor=processor, device=device)
    print(f"[HF VLM] Loaded {model_id} on {device}")
    return _hf_vlm


DEFAULT_UA = (
    "UNLZ-AI-STUDIO/0.1 (+https://ingenieria.unlz.edu.ar; contact: admin@example.com)"
)

def _download_image(url: str) -> Image.Image:
    # data: URL (inline base64)
    if url.startswith("data:"):
        header, b64 = url.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")

    # local paths or file:// URLs
    if url.startswith("file://"):
        return Image.open(url.replace("file://", "", 1)).convert("RGB")
    if os.path.exists(url):
        return Image.open(url).convert("RGB")

    # remote URL — send a friendly UA + referer (Wikimedia policy)
    headers = {
        "User-Agent": DEFAULT_UA,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": "https://www.wikipedia.org/",
    }
    r = requests.get(url, headers=headers, timeout=30, stream=True)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # surface a useful message if a host blocks us
        raise RuntimeError(
            f"Image fetch failed ({r.status_code}) from {url}. "
            f"Response head: {r.headers.get('Server','?')}. "
            f"Body preview: {r.text[:200]!r}"
        ) from e
    return Image.open(BytesIO(r.content)).convert("RGB")


def _hf_vlm_infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    bundle = _ensure_hf_vlm()
    model, processor, device = bundle["model"], bundle["processor"], bundle["device"]

    messages = payload.get("messages", [])
    images = []
    download_errors = []

    # Try to download any images provided in the payload
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url")
                    if not url:
                        continue
                    try:
                        images.append(_download_image(url))
                    except Exception as e:
                        download_errors.append((url, str(e)))

    # If none worked, fall back to the default image you requested
    if not images and DEFAULT_IMAGE_URL:
        try:
            images.append(_download_image(DEFAULT_IMAGE_URL))
            print(f"[HF VLM] No usable image from payload; fell back to {DEFAULT_IMAGE_URL}")
        except Exception as e:
            # Surface a helpful message
            first_err = download_errors[0][1] if download_errors else str(e)
            raise RuntimeError(f"No images could be loaded. Example error: {first_err}") from e

    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[chat_text], images=images, return_tensors="pt").to(model.device)

    import torch
    gen_kwargs = {
        "max_new_tokens": int(payload.get("max_tokens", 512)),
        "temperature": float(payload.get("temperature", 0.2)),
        "do_sample": float(payload.get("temperature", 0.2)) > 0.0,
        "top_p": float(payload.get("top_p", 0.9)),
    }
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    generated = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )[0]

    return {
        "id": "chatcmpl-local-hf",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "qwen2.5-vl-7b",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": generated},
            "finish_reason": "stop"
        }]
    }

# =====================
# Console log mirroring
# =====================
PRINT_CHILD_LOGS = True  # show llama/lmdeploy logs in this console
 # -------- console log mirroring helpers (restauradas) --------
def _ensure_triton_stub() -> str:
    stub_root = LOG_DIR / "stubs"
    pkg = stub_root / "triton"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(
        '# Triton stub; LMDeploy runs with --eager-mode on Windows.\n'
        '__version__ = "0.0.0-win-stub"\n'
        'def __getattr__(name):\n'
        '    raise ImportError("Triton is not available on Windows; stub active.")\n',
        encoding="utf-8"
    )
    (pkg / "language.py").write_text("", encoding="utf-8")
    return str(stub_root)

def _spawn_with_tee(cmd, log_path: Path, env=None, creationflags=0):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        creationflags=creationflags
    )
    def _pump():
        prefix = f"[{log_path.name}] "
        for line in proc.stdout:  # type: ignore
            try:
                f.write(line)
                if PRINT_CHILD_LOGS:
                    print(prefix + line.rstrip())
            except Exception:
                pass
        try: f.flush()
        except: pass
        try: f.close()
        except: pass
    threading.Thread(target=_pump, daemon=True).start()
    return proc


# === TTS - Piper ===
PIPER_EXE   = r"C:\piper\piper\piper.exe"
PIPER_VOICE = r"C:\piper\voices\es_AR\daniela_high\es\es_AR\daniela\high\es_AR-daniela-high.onnx"

# carpeta temporal aislada para Piper
TTS_TMP_DIR = LOG_DIR / "piper_tmp"
TTS_TMP_DIR.mkdir(parents=True, exist_ok=True)

def _effective_piper_voice() -> str:
    v = os.environ.get("PIPER_VOICE", PIPER_VOICE)
    if os.path.isdir(v):
        import glob, os as _os
        cand = sorted(glob.glob(_os.path.join(v, "*.onnx")))
        if not cand:
            raise RuntimeError(f"No se encontraron .onnx en el directorio: {v}")
        v = cand[0]
    if not os.path.isfile(v):
        raise RuntimeError(f"PIPER_VOICE no existe: {v}")

    # Aviso si falta el JSON al lado (muchas voces lo necesitan)
    sidecar_json = v + ".json" if not v.endswith(".json") else v
    if not os.path.isfile(sidecar_json):
        print(f"[TTS] AVISO: No se encontró el sidecar JSON junto al modelo: {sidecar_json}")

    return v

def _synth_with_piper(text: str) -> bytes:
    """
    Ejecuta Piper (pip wrapper) escribiendo a un WAV temporal en disco
    y devuelve los bytes del WAV. Evita el modo stdout (-f -) que
    en Windows da error.
    """
    import tempfile, os

    voice = _effective_piper_voice()
    print(f"[TTS] Piper exe: {PIPER_EXE}")
    print(f"[TTS] Piper voice: {voice}")
    print(f"[TTS] TMP dir: {TTS_TMP_DIR}")

    # archivo temporal .wav controlado por nosotros
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=str(TTS_TMP_DIR))
    tmp_wav_path = tmp.name
    tmp.close()

    env = os.environ.copy()
    env["TMP"] = env["TEMP"] = env["TMPDIR"] = str(TTS_TMP_DIR)

    # IMPORTANTE: usar -f <ruta>, no -f -
    cmd = [PIPER_EXE, "-m", voice, "-f", tmp_wav_path]

    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,   # no usamos stdout
            stderr=subprocess.PIPE,
            env=env,
            creationflags=flags
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"No se pudo ejecutar Piper ('{PIPER_EXE}'). ¿Está en PATH?") from e

    _, err = proc.communicate(text.encode("utf-8"))
    rc = proc.returncode
    err_txt = (err or b"").decode("utf-8", "ignore")

    if rc != 0:
        # limpia el temporal si falló
        try: os.unlink(tmp_wav_path)
        except: pass
        raise RuntimeError(f"Piper salió con código {rc}. STDERR: {err_txt[:800]}")

    # leer WAV generado
    try:
        with open(tmp_wav_path, "rb") as f:
            out = f.read()
    finally:
        try: os.unlink(tmp_wav_path)
        except: pass

    # validación mínima WAV
    if not out or len(out) < 44 or out[:4] != b"RIFF" or out[8:12] != b"WAVE":
        raise RuntimeError(f"Piper no produjo WAV válido (bytes={len(out)}). STDERR: {err_txt[:800]}")

    return out



# =========================
# CONFIG - AJUSTÁ ESTAS RUTAS
# =========================
#LLAMA_SERVER_EXE = r"C:\llama-cuda\llama-server.exe"  # CUDA
LLAMA_SERVER_EXE = "llama-server"  # Vulkan
LLAMA_MODEL = r"C:\models\qwen3-coder-30b\Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf"
LLAMA_HOST       = "127.0.0.1"
LLAMA_PORT       = 8080
LLAMA_ARGS = [
    "-m", LLAMA_MODEL,
    "--host", LLAMA_HOST, "--port", str(LLAMA_PORT),
    "--ctx-size", "4096",
    "--n-gpu-layers", "28",    # si tu build no acepta este flag, usar --ngl
    "-t", "12"
    # "--flash-attn"          # si usás CUDA/cuBLAS (no Vulkan)
]

# --- VLM (LMDeploy 0.9.2, Windows) ---
LMDEPLOY_CMD  = [sys.executable, "-m", "lmdeploy"]
VLM_MODEL_ID  = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_CACHE_DIR = r"C:\models\qwen2.5-vl-7b-hf"
VLM_HOST = "127.0.0.1"
VLM_PORT = 9090

# PyTorch backend + HF formato + cache local + EAGER (sin Triton)
VLM_ARGS = [
    "serve", "api_server", VLM_MODEL_ID,
    "--backend", "pytorch",
    "--model-format", "hf",
    "--download-dir", VLM_CACHE_DIR,
    "--server-port", str(VLM_PORT),
    "--model-name", "qwen2.5-vl-7b",
    "--eager-mode",
]

# STT (usar CUDA o CPU)
USE_CUDA_FOR_STT = False
STT_MODEL_SIZE   = "medium"
STT_DEVICE       = "cuda" if USE_CUDA_FOR_STT else "cpu"
STT_COMPUTE_TYPE = "float16" if USE_CUDA_FOR_STT else "int8"



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
            return

        self.stop()
        time.sleep(1.5)

        if mode == "llm":
            print("Starting llama-server...")
            log_path = LOG_DIR / "llama-server.log"
            try:
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                self.proc = _spawn_with_tee([LLAMA_SERVER_EXE] + LLAMA_ARGS, log_path, creationflags=flags)
                self._wait_llama_ready(timeout=600, proc=self.proc)
            except Exception as e:
                raise RuntimeError(f"llama-server no inició: {e}. Ver {log_path}")
        else:
            print("Starting lmdeploy api_server...")
            log_path = LOG_DIR / "lmdeploy.log"
            try:
                env = os.environ.copy()
                # Belt & suspenders: disable Triton / compilers in child
                env["LMDEPLOY_DISABLE_TRITON"] = "1"
                env["PYTORCH_TRITON_DISABLE"] = "1"
                env["TORCHINDUCTOR_FX_ENABLE"] = "0"
                env["TORCHINDUCTOR_DISABLE"] = "1"
                # Add Triton stub so 'import triton' succeeds
                stub_root = _ensure_triton_stub()
                env["PYTHONPATH"] = stub_root + (os.pathsep + env.get("PYTHONPATH",""))

                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                self.proc = _spawn_with_tee(LMDEPLOY_CMD + VLM_ARGS, log_path, env=env, creationflags=flags)
                # First boot can be slow -> 480s
                self._wait_http_ok(f"http://{VLM_HOST}:{VLM_PORT}/v1/models", timeout=480, proc=self.proc)
            except Exception as e:
                raise RuntimeError(f"lmdeploy no inició: {e}. Ver {log_path}")

        self.mode = mode

    def _wait_llama_ready(self, timeout=600, proc: Optional[subprocess.Popen]=None):
        """espera a que /health devuelva 200 (modelo cargado)"""
        url = f"http://{LLAMA_HOST}:{LLAMA_PORT}/health"
        deadline = time.time() + timeout
        with httpx.Client(timeout=3.0) as client:
            while time.time() < deadline:
                if proc is not None and proc.poll() is not None:
                    rc = proc.returncode
                    raise RuntimeError(f"llama-server terminó con código {rc}")
                try:
                    r = client.get(url)
                    if r.status_code == 200:
                        return
                except:
                    pass
                time.sleep(1.0)
        raise RuntimeError(f"/health no estuvo OK en {timeout}s (modelo grande: puede tardar).")

    def _wait_http_ok(self, url: str, timeout=120, proc: Optional[subprocess.Popen]=None):
        """espera a que un endpoint devuelva 200 (p.ej. /v1/models de LMDeploy)"""
        deadline = time.time() + timeout
        with httpx.Client(timeout=5.0) as client:
            while time.time() < deadline:
                if proc is not None and proc.poll() is not None:
                    rc = proc.returncode
                    raise RuntimeError(f"proceso terminó con código {rc}")
                try:
                    r = client.get(url)
                    if r.status_code == 200:
                        return
                except:
                    pass
                time.sleep(1.0)
        raise RuntimeError(f"{url} no respondió 200 en {timeout}s.")

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
    try:
        manager.ensure_mode("llm")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    async with httpx.AsyncClient(timeout=None) as client:
        url = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1/chat/completions"
        r = await client.post(url, json=payload)
        return JSONResponse(status_code=r.status_code, content=r.json())

# =========================
# /vlm -> VLM backend selector (HF Transformers or LMDeploy)
# =========================
@app.post("/vlm")
async def vlm_chat(payload: Dict[str, Any]):
    if VLM_BACKEND == "hf":
        try:
            resp = _hf_vlm_infer(payload)
            return JSONResponse(status_code=200, content=resp)
        except Exception as e:
            print("[HF VLM] Exception:\n" + traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"HF VLM error: {e}")
    else:
        try:
            manager.ensure_mode("vlm")
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        async with httpx.AsyncClient(timeout=None) as client:
            url = f"http://{VLM_HOST}:{VLM_PORT}/v1/chat/completions"
            r = await client.post(url, json=payload)
            return JSONResponse(status_code=r.status_code, content=r.json())

# =========================
# /alm -> audio pipeline (STT -> LLM -> opcional TTS)
# =========================
@app.post("/alm")
async def alm_pipeline(
    file: UploadFile = File(...),
    system_prompt: Optional[str] = Form("You are a helpful assistant."),
    tts: Optional[bool] = Form(True),
    target_lang: Optional[str] = Form("es")
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "audio").suffix or ".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        wav_path = tmp.name

    segments, info = stt_model.transcribe(wav_path, language=target_lang, vad_filter=True)
    user_text = "".join([s.text for s in segments]).strip()
    os.unlink(wav_path)

    try:
        manager.ensure_mode("llm")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

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

    # 3) TTS (opcional, Piper a WAV en base64)
    audio_b64 = None
    tts_error = None
    if tts:
        try:
            wav_bytes = _synth_with_piper(answer_text)
            audio_b64 = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")
        except Exception as e:
            tts_error = str(e)
            print("[TTS] ERROR:", tts_error)

    return {
        "stt_text": user_text,
        "llm_text": answer_text,
        "tts_audio": audio_b64,
        "tts_error": tts_error,
    }

@app.get("/health")
def health():
    return {"status": "ok", "mode": manager.mode}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
