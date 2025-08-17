import warnings
#warnings.filterwarnings("ignore", category=UserWarning, module=r"ctranslate2(\.|$)")
import os, sys, time, base64, tempfile, shutil, subprocess, threading
import atexit, signal, gc, asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import psutil
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi import Body
from fastapi import Depends
from fastapi import status
from fastapi import Response
from fastapi import BackgroundTasks
from fastapi import APIRouter
from fastapi import FastAPI
from pydantic import BaseModel, Field, ValidationError
from faster_whisper import WhisperModel
import uvicorn

# --- Logs y TMP dirs (deben existir antes de usarlos) ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Temp dedicado para Piper
TTS_TMP_DIR = LOG_DIR / "piper_tmp"           # opción A: dentro de logs/
# TTS_TMP_DIR = Path(tempfile.gettempdir()) / "unlz_piper_tmp"  # opción B: en %TEMP%
TTS_TMP_DIR.mkdir(parents=True, exist_ok=True)



# --- VLM (LMDeploy / HF ubicaciones) ---
LMDEPLOY_CMD  = [sys.executable, "-m", "lmdeploy"]
VLM_MODEL_ID  = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_CACHE_DIR = r"C:\models\qwen2.5-vl-7b-hf"   # <-- DEBE existir antes del warm
VLM_HOST = "127.0.0.1"
VLM_PORT = 9090
VLM_ARGS = [
    "serve", "api_server", VLM_MODEL_ID,
    "--backend", "pytorch",
    "--model-format", "hf",
    "--download-dir", VLM_CACHE_DIR,
    "--server-port", str(VLM_PORT),
    "--model-name", "qwen2.5-vl-7b",
    "--eager-mode",
]

# --- VLM HuggingFace backend imports ---
import requests
from io import BytesIO
from PIL import Image
import traceback
from urllib.parse import urlparse

# --- Warm helpers: leer archivos a RAM (page-cache) sin guardarlos en Python ---
from typing import Union
def _warm_file(path: Union[str, Path], chunk_mb: int = 64) -> None:
    p = Path(path)
    if not p.is_file():
        print(f"[warm] skip (no file): {p}")
        return
    try:
        sz = 0
        chunk = 1024 * 1024 * chunk_mb
        with open(p, "rb", buffering=0) as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                sz += len(b)
        print(f"[warm] {p} -> {sz / (1024**3):.2f} GiB cached")
    except Exception as e:
        print(f"[warm] error {p}: {e}")

def _warm_dir(root: Union[str, Path], patterns: tuple[str, ...]) -> None:
    root = Path(root)
    if not root.exists():
        print(f"[warm] skip dir: {root}")
        return
    for pat in patterns:
        for f in root.rglob(pat):
            _warm_file(f)


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
    import torch, os
    model_id_hub = "Qwen/Qwen2.5-VL-7B-Instruct"
    cache_dir = VLM_CACHE_DIR  # ya definido arriba
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_processor(local_only: bool):
        return AutoProcessor.from_pretrained(
            cache_dir if local_only else model_id_hub,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=local_only
        )

    def _load_model(local_only: bool):
        if device == "cuda":
            m = AutoModelForImageTextToText.from_pretrained(
                cache_dir if local_only else model_id_hub,
                torch_dtype=torch.float16,
                device_map={"": "cpu"},          # carga a RAM primero
                low_cpu_mem_usage=False,
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=local_only
            )
            return m.to("cuda", dtype=torch.float16)
        else:
            return AutoModelForImageTextToText.from_pretrained(
                cache_dir if local_only else model_id_hub,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=local_only
            )

    # 1) Intento 100% offline
    try:
        processor = _load_processor(local_only=True)
    except Exception as e:
        print(f"[HF VLM] Processor local no disponible ({e}). Descargando al cache_dir ...")
        processor = _load_processor(local_only=False)

    try:
        model = _load_model(local_only=True)
    except Exception as e:
        print(f"[HF VLM] Modelo local no disponible ({e}). Descargando al cache_dir ...")
        model = _load_model(local_only=False)

    model.eval()
    _hf_vlm.update(model=model, processor=processor, device=device)
    print(f"[HF VLM] Loaded {cache_dir} (fallback OK) on {device}")
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


import torch

def _hf_vlm_infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    bundle = _ensure_hf_vlm()
    model, processor, device = bundle["model"], bundle["processor"], bundle["device"]


    raw_messages = payload.get("messages", [])

    # 1) Coleccionar URLs (OpenAI style)
    url_list = []
    for m in raw_messages:
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if p.get("type") == "image_url":
                    u = p.get("image_url", {}).get("url")
                    if u:
                        url_list.append(u)

    # 2) Descargar imágenes con éxito
    images = []
    success_urls = []
    for u in url_list:
        try:
            images.append(_download_image(u))
            success_urls.append(u)
        except Exception as e:
            print(f"[HF VLM] image download failed: {u} :: {e}")

    # 3) Fallback si el usuario pidió imágenes y ninguna funcionó
    used_fallback = False
    if url_list and not images and DEFAULT_IMAGE_URL:
        try:
            images.append(_download_image(DEFAULT_IMAGE_URL))
            used_fallback = True
            print(f"[HF VLM] usando fallback de imagen: {DEFAULT_IMAGE_URL}")
        except Exception as e:
            print(f"[HF VLM] fallback de imagen falló: {e}")

    # 4) Construir mensajes alineados: solo crear placeholders por las imágenes disponibles
    images_needed = len(images)
    images_used = 0
    messages_aligned: list[dict] = []

    for m in raw_messages:
        new_m = {"role": m["role"]}
        c = m.get("content")
        if isinstance(c, list):
            new_c = []
            for p in c:
                if p.get("type") == "image_url":
                    # Ponemos placeholder solo si todavía hay una imagen disponible
                    if images_used < images_needed:
                        new_c.append({"type": "image"})
                        images_used += 1
                    else:
                        # No hay imagen para esta parte -> la omitimos
                        continue
                else:
                    # Mantener partes de texto u otras tal cual
                    new_c.append(p)
            new_m["content"] = new_c
        else:
            # Contenido plano (string): lo dejamos como está
            new_m["content"] = c
        messages_aligned.append(new_m)


    # --- Render chat template ---
    chat_text = processor.apply_chat_template(
        messages_aligned, tokenize=False, add_generation_prompt=True
    )

    # --- Alinear EXACTO: tantos <image> como imágenes reales, sin duplicar ---
    placeholder_token = getattr(processor, "image_token", "<image>")
    num_ph = chat_text.count(placeholder_token)
    num_imgs = len(images)

    def _cap_placeholders(s: str, token: str, keep: int) -> str:
        out, i, start = [], 0, 0
        while True:
            j = s.find(token, start)
            if j == -1:
                out.append(s[start:])
                break
            out.append(s[start:j])
            if i < keep:
                out.append(token)   # conservar solo los primeros `keep`
            # si no, se descarta el token extra
            i += 1
            start = j + len(token)
        return "".join(out)

    if num_imgs == 0 and num_ph > 0:
        # no tenemos imágenes: elimina todos los <image>
        chat_text = chat_text.replace(placeholder_token, "")
    elif num_ph > num_imgs:
        # hay más <image> en el prompt que imágenes reales -> recorta placeholders
        chat_text = _cap_placeholders(chat_text, placeholder_token, num_imgs)
    elif num_imgs > num_ph:
        # hay más imágenes que <image> en el prompt -> recorta imágenes
        images = images[:num_ph]

    print(f"[HF VLM] placeholders(final)={chat_text.count(placeholder_token)}, images(final)={len(images)}")

    # Tokenizar
    if images:
        inputs = processor(text=[chat_text], images=images, return_tensors="pt").to(model.device)
    else:
        inputs = processor(text=[chat_text], return_tensors="pt").to(model.device)

    gen_kwargs = {
        "max_new_tokens": int(payload.get("max_tokens", 512)),
        "temperature": float(payload.get("temperature", 0.2)),
        "do_sample": float(payload.get("temperature", 0.2)) > 0.0,
        "top_p": float(payload.get("top_p", 0.9)),
    }

    import torch
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    generated = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )[0]

    # cleanup
    try:
        for im in images:
            try: im.close()
            except: pass
        del inputs, output_ids
        gc.collect()
        if bundle["device"] == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

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

def _spawn_with_tee(cmd, log_path: Path, env=None, creationflags=0):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", encoding="utf-8", buffering=1)
    # IMPORTANT on Windows: own process group so we can send CTRL_BREAK
    if os.name == "nt":
        creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        creationflags=creationflags,
        close_fds=(os.name != "nt"),
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
    with tempfile.NamedTemporaryFile(dir=TTS_TMP_DIR, suffix=".wav", delete=False) as tmp:
        tmp_wav_path = tmp.name

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
    "-t", "12",
    "--no-mmap",           # fuerza copia completa a RAM (en vez de mmapped desde disco)
    # "--mlock",           # (opcional) intenta “fijar” en RAM; en Windows puede requerir privilegios
    # "--flash-attn"          # si usás CUDA/cuBLAS (no Vulkan)
]

# --- Preflight cleanup: kill orphans and free ports ---
def _kill_by_port(port: int):
    try:
        for c in psutil.net_connections(kind='tcp'):
            if c.laddr and c.laddr.port == port and c.pid:
                try: psutil.Process(c.pid).kill()
                except: pass
    except Exception:
        pass

def _kill_orphans():
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            name = (p.info['name'] or '').lower()
            cmd  = ' '.join(p.info['cmdline'] or []).lower()
            if 'llama-server' in name or 'llama-server' in cmd:
                p.kill()
            if 'lmdeploy' in cmd and 'api_server' in cmd:
                p.kill()
        except Exception:
            pass

print("[preflight] killing orphans / freeing ports …")
_kill_orphans()
_kill_by_port(LLAMA_PORT)
_kill_by_port(VLM_PORT)

# -------- Warm all weights into RAM (no GPU allocation yet) --------
print("[warm] Priming OS page cache (LLM/VLM/TTS) ...")
try:
    # LLM (GGUF)
    _warm_file(LLAMA_MODEL)

    # VLM (HF cache carpeta)
    if 'VLM_CACHE_DIR' in globals() and os.path.isdir(VLM_CACHE_DIR):
        _warm_dir(VLM_CACHE_DIR, patterns=(
            "*.safetensors", "tokenizer.json", "tokenizer.model", "merges.txt", "vocab.json", "*.json"
        ))
    else:
        print("[warm] skip VLM: VLM_CACHE_DIR no definido o carpeta inexistente")

    # Piper voice (onnx + json sidecar si existe)
    _warm_file(PIPER_VOICE)
    pv_json = PIPER_VOICE + ".json" if not PIPER_VOICE.endswith(".json") else PIPER_VOICE
    if os.path.isfile(pv_json):
        _warm_file(pv_json)
except Exception as e:
    print(f"[warm] Warming skipped with error: {e}")



# STT (usar CUDA o CPU)
USE_CUDA_FOR_STT = False
STT_MODEL_SIZE   = "medium"
STT_DEVICE       = "cuda" if USE_CUDA_FOR_STT else "cpu"
STT_COMPUTE_TYPE = "float16" if USE_CUDA_FOR_STT else "int8"



# =========================
# MANAGER DE PROCESOS GPU
# =========================


import threading

class GpuProcessManager:
    def __init__(self):
        self.mode = None
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()

    def _kill_tree(self, p: psutil.Process):
        for c in p.children(recursive=True):
            try: c.kill()
            except: pass
        try: p.kill()
        except: pass

    def stop(self, grace: float = 3.0):
        with self._lock:
            if not self.proc:
                self.mode = None
                return
            try:
                if self.proc.poll() is None:
                    p = psutil.Process(self.proc.pid)
                    if os.name == "nt":
                        # try graceful CTRL_BREAK to the process group
                        try:
                            os.kill(self.proc.pid, signal.CTRL_BREAK_EVENT)  # requires CREATE_NEW_PROCESS_GROUP
                            p.wait(timeout=grace)
                        except Exception:
                            pass
                        # if still alive -> taskkill whole tree
                        if self.proc.poll() is None:
                            try:
                                subprocess.run(["taskkill","/PID",str(self.proc.pid),"/T","/F"],
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                            except Exception:
                                pass
                    else:
                        try:
                            p.terminate()
                            p.wait(timeout=grace)
                        except Exception:
                            pass
                        if self.proc.poll() is None:
                            self._kill_tree(p)
            except Exception:
                pass
            finally:
                self.proc = None
                self.mode = None

    def ensure_mode(self, mode: str):
        assert mode in ("llm", "vlm")
        with self._lock:
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



# --- Global async httpx client ---
_httpx_async_client: httpx.AsyncClient = None

def get_httpx_client() -> httpx.AsyncClient:
    return _httpx_async_client

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _httpx_async_client
    _httpx_async_client = httpx.AsyncClient(timeout=None)
    yield
    await _httpx_async_client.aclose()

app = FastAPI(title="IA Gateway (LLM/VLM/ALM)", lifespan=lifespan)
manager = GpuProcessManager()

@atexit.register
def _at_exit():
    print("[atexit] stopping child processes …")
    try: manager.stop()
    except: pass
    # also clean any that escaped
    _kill_orphans()
    _kill_by_port(LLAMA_PORT)
    _kill_by_port(VLM_PORT)

@app.on_event("shutdown")
def _on_shutdown():
    print("[fastapi] shutdown -> stopping children …")
    manager.stop()
    _kill_orphans()
    _kill_by_port(LLAMA_PORT)
    _kill_by_port(VLM_PORT)

# Pre-carga STT (en CPU por defecto para evitar conflictos)
os.environ.setdefault("CT2_USE_EXPERIMENTAL_PACKED_GEMM", "1")  # acelera matmul en CPU si aplica
print("Loading WhisperModel (STT)...")
stt_model = WhisperModel(STT_MODEL_SIZE, device=STT_DEVICE, compute_type=STT_COMPUTE_TYPE)

# =========================
# /llm  -> proxyea a llama.cpp
# =========================

# --- Pydantic model for LLM payload ---
class LLMMessage(BaseModel):
    role: str
    content: str

class LLMChatPayload(BaseModel):
    model: str
    messages: list[LLMMessage]
    temperature: float = 0.2
    max_tokens: int = 512
    top_p: float = 0.9

from fastapi.concurrency import run_in_threadpool


@app.post("/llm")
async def llm_chat(payload: LLMChatPayload, client: httpx.AsyncClient = Depends(get_httpx_client)):
    try:
        await run_in_threadpool(manager.ensure_mode, "llm")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    url = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1/chat/completions"
    r = await client.post(url, json=payload.model_dump())
    return JSONResponse(status_code=r.status_code, content=r.json())

# =========================
# /clm -> igual a /llm pero usando Qwen/Qwen2.5-VL-7B-Instruct (HF in-proc)
# =========================
from fastapi.concurrency import run_in_threadpool

@app.post("/clm")
async def clm_chat(payload: LLMChatPayload):
    # Opcional: liberar GPU si tenés llama-server corriendo y querés evitar OOMs.
    # await run_in_threadpool(manager.stop)

    # Reutilizamos el mismo schema que /llm, pero llamamos al backend HF
    # (Qwen2.5-VL-7B-Instruct) que ya maneja _hf_vlm_infer.
    hf_payload = {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
        "top_p": payload.top_p,
        "messages": [{"role": m.role, "content": m.content} for m in payload.messages],
    }
    try:
        resp = await run_in_threadpool(_hf_vlm_infer, hf_payload)
        return JSONResponse(status_code=200, content=resp)
    except Exception as e:
        print("[CLM/HF] Exception:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"CLM error: {e}")

# =========================
# /vlm -> VLM backend selector (HF Transformers or LMDeploy)
# =========================

# --- Pydantic model for VLM payload ---
class VLMImageUrl(BaseModel):
    url: str

class VLMContentPart(BaseModel):
    type: str
    image_url: Optional[VLMImageUrl] = None

class VLMMessage(BaseModel):
    role: str
    content: list[VLMContentPart] | str

class VLMChatPayload(BaseModel):
    model: Optional[str] = None
    messages: list[VLMMessage]
    temperature: float = 0.2
    max_tokens: int = 512
    top_p: float = 0.9

@app.post("/vlm")
async def vlm_chat(payload: VLMChatPayload, client: httpx.AsyncClient = Depends(get_httpx_client)):
    if VLM_BACKEND == "hf":
        try:
            resp = await run_in_threadpool(_hf_vlm_infer, payload.model_dump())
            return JSONResponse(status_code=200, content=resp)
        except Exception as e:
            print("[HF VLM] Exception:\n" + traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"HF VLM error: {e}")
    else:
        try:
            await run_in_threadpool(manager.ensure_mode, "vlm")
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        url = f"http://{VLM_HOST}:{VLM_PORT}/v1/chat/completions"
        r = await client.post(url, json=payload.model_dump())
        return JSONResponse(status_code=r.status_code, content=r.json())

# =========================
# /alm -> audio pipeline (STT -> LLM -> opcional TTS)
# =========================

class ALMResponse(BaseModel):
    stt_text: str
    llm_text: str
    tts_audio: Optional[str] = None
    tts_error: Optional[str] = None

@app.post("/alm", response_model=ALMResponse)
async def alm_pipeline(
    file: UploadFile = File(...),
    system_prompt: Optional[str] = Form("You are a helpful assistant."),
    tts: Optional[bool] = Form(True),
    target_lang: Optional[str] = Form("es"),
    client: httpx.AsyncClient = Depends(get_httpx_client)
):
    def transcribe_and_llm():
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "audio").suffix or ".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            wav_path = tmp.name

        segments, info = stt_model.transcribe(wav_path, language=target_lang, vad_filter=True)
        user_text = "".join([s.text for s in segments]).strip()
        os.unlink(wav_path)
        return user_text

    user_text = await run_in_threadpool(transcribe_and_llm)

    try:
        await run_in_threadpool(manager.ensure_mode, "llm")
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
            wav_bytes = await run_in_threadpool(_synth_with_piper, answer_text)
            audio_b64 = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")
        except Exception as e:
            tts_error = str(e)
            print("[TTS] ERROR:", tts_error)

    return ALMResponse(
        stt_text=user_text,
        llm_text=answer_text,
        tts_audio=audio_b64,
        tts_error=tts_error,
    )

@app.get("/health")
def health():
    return {"status": "ok", "mode": manager.mode}

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    finally:
        print("[main] finally -> stopping children …")
        manager.stop()
        _kill_orphans()
        _kill_by_port(LLAMA_PORT)
        _kill_by_port(VLM_PORT)
