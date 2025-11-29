import logging
import requests

PRINT_CHILD_LOGS = False  # Avoid accidental spam from child processes

import warnings
#warnings.filterwarnings("ignore", category=UserWarning, module=r"ctranslate2(\.|$)")
import os, sys, time, base64, tempfile, shutil, subprocess, threading
import atexit, signal, gc, asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Iterable
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
from huggingface_hub import snapshot_download
import asyncio
from fastapi.responses import StreamingResponse
from runtime_profiles import detect_system_info, ProfileManager, run_dependency_checks
from studio_gui import start_gui_thread

# --- Logs y TMP dirs (deben existir antes de usarlos) ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Temp dedicado para Piper
TTS_TMP_DIR = LOG_DIR / "piper_tmp"           # opción A: dentro de logs/
# TTS_TMP_DIR = Path(tempfile.gettempdir()) / "unlz_piper_tmp"  # opción B: en %TEMP%
TTS_TMP_DIR.mkdir(parents=True, exist_ok=True)

GUI_LOGO_PATH = Path("assets") / "SOLO-LOGO-AZUL-HORIZONTAL-fondo-transparente.ico"
GUI_SPLASH_PATH = Path("assets") / "LOGO AZUL HORIZONTAL - fondo transparente.png"

SYSTEM_INFO = detect_system_info()
PROFILE_MANAGER = ProfileManager(SYSTEM_INFO, LOG_DIR)

env_profile = os.getenv("UNLZ_PROFILE")
if env_profile:
    ok, msg = PROFILE_MANAGER.set_active_profile(env_profile, force=True)
    print(f"[preflight] Perfil forzado ({env_profile}): {msg}")

PREFLIGHT_REPORT = run_dependency_checks(SYSTEM_INFO, LOG_DIR)

if PREFLIGHT_REPORT.get("warnings"):
    print("[preflight] Advertencias detectadas:")
    for warning in PREFLIGHT_REPORT["warnings"]:
        print("  -", warning)
else:
    print("[preflight] Dependencias OK.")

if PREFLIGHT_REPORT.get("suggestions"):
    print("[preflight] Sugerencias:")
    for suggestion in PREFLIGHT_REPORT["suggestions"]:
        print("  -", suggestion)

print("[preflight] Perfil activo:", PROFILE_MANAGER.describe_active_profile())


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
from io import BytesIO
from PIL import Image
import traceback
from urllib.parse import urlparse

# --- Warm helpers: leer archivos a RAM (page-cache) sin guardarlos en Python ---
from typing import Union
def _warm_file(path: Union[str, Path], chunk_mb: int = 256) -> None:
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
    "https://live.staticflickr.com/65535/54703830763_71e4af50f4_k.jpg"
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
    cache_dir = VLM_CACHE_DIR
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Resolver snapshot local (sin red). Si no existe, baja al cache_dir.
    local_repo = None
    try:
        local_repo = snapshot_download(
            repo_id=model_id_hub,
            cache_dir=cache_dir,
            local_files_only=True
        )
    except Exception as e:
        print(f"[HF VLM] snapshot local no encontrado ({e}). Bajando al cache_dir …")
        local_repo = snapshot_download(repo_id=model_id_hub, cache_dir=cache_dir, local_files_only=False)

    # 2) Cargar desde el snapshot resuelto
    if device == "cuda":
        model = AutoModelForImageTextToText.from_pretrained(
            local_repo, trust_remote_code=True, torch_dtype=torch.float16
        ).to("cuda", dtype=torch.float16)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            local_repo, trust_remote_code=True, torch_dtype=torch.float32
        )

    processor = AutoProcessor.from_pretrained(local_repo, trust_remote_code=True)

    model.eval()
    _hf_vlm.update(model=model, processor=processor, device=device)
    print(f"[HF VLM] Loaded {local_repo} on {device}")
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
    # Lower connect/read timeouts and avoid streaming to memory twice
    r = requests.get(url, headers=headers, timeout=(3.0, 10.0))
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
torch.set_grad_enabled(False)  # Always disable grad globally (eval only)

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
        # Removed gc.collect() and torch.cuda.empty_cache() to keep memory warm for speed
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





# --- TTS: rutas Piper (defínelas antes de PiperWorker) ---
PIPER_EXE   = r"C:\piper\piper\piper.exe"
PIPER_VOICE = r"C:\piper\voices\es_AR\daniela_high\es\es_AR\daniela\high\es_AR-daniela-high.onnx"
PIPER_CFG   = PIPER_VOICE + ".json"  # si existe, lo usamos en PiperWorker.start()

# === Piper (worker CLI persistente, sin HTTP) ===
import json

_piper_start_lock = threading.RLock()   # <- lock global para el arranque

class PiperWorker:
    def __init__(self, exe: str, voice: str, out_dir: Path, cfg: Optional[str] = None):
        self.exe = exe
        self.voice = voice
        self.cfg = cfg if (cfg and os.path.isfile(cfg)) else None
        self.out_dir = out_dir
        self.proc: Optional[subprocess.Popen] = None
        self.lock = threading.RLock()
        self._log_f = None  # para volcar stderr a archivo (evita bloqueo)

    def start(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [self.exe, "-m", self.voice, "--json-input", "--output_dir", str(self.out_dir)]
        if self.cfg:
            cmd += ["-c", self.cfg]

        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        # Abrí un log para stderr (no uses PIPE sin leerlo)
        piper_log = LOG_DIR / "piper-cli.log"
        self._log_f = open(piper_log, "a", encoding="utf-8", buffering=1)

        # Usá stdin binario (text=False) y escribí UTF-8 manualmente
        # bufsize=0 => sin buffer adicional
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._log_f,
            text=False,
            bufsize=0,
            creationflags=flags,
        )
        if not self.proc or self.proc.poll() is not None:
            raise RuntimeError("No se pudo iniciar Piper (CLI).")

        # Warmup rápido (si falla, no corta el arranque)
        try:
            self.synth("ok", timeout=10.0)
        except Exception:
            pass

    def stop(self):
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass
        self.proc = None
        try:
            if self._log_f:
                self._log_f.close()
        except Exception:
            pass
        self._log_f = None

    def synth(self, text: str, timeout: float = 30.0) -> bytes:
        if not self.proc or self.proc.poll() is not None:
            raise RuntimeError("Piper no está iniciado (worker).")

        # Usá ruta ABSOLUTA (y slashes) para evitar líos con --output_dir
        fname = f"p_{int(time.time()*1000)}_{os.getpid()}_{threading.get_ident()}.wav"
        out_path = (self.out_dir / fname).resolve()
        out_path_str = str(out_path).replace("\\", "/")

        payload = {"text": text, "output_file": out_path_str}
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

        with self.lock:
            try:
                self.proc.stdin.write(line)  # bytes
                self.proc.stdin.flush()
            except BrokenPipeError:
                raise RuntimeError("stdin de Piper se cerró (proceso muerto).")

        # Esperá el archivo
        deadline = time.time() + timeout
        while time.time() < deadline:
            if out_path.exists() and out_path.stat().st_size > 44:  # WAV mínimo
                try:
                    data = out_path.read_bytes()
                finally:
                    try: out_path.unlink()
                    except: pass
                return data

            # Si el proceso murió, devolvé el tail del log
            if self.proc.poll() is not None:
                try:
                    tail = (LOG_DIR / "piper-cli.log").read_text("utf-8", errors="ignore").splitlines()[-50:]
                except Exception:
                    tail = ["<sin log>"]
                raise RuntimeError("Piper murió durante la síntesis:\n" + "\n".join(tail))

            time.sleep(0.02)

        # Timeout: mostrale algo útil al caller
        try:
            tail = (LOG_DIR / "piper-cli.log").read_text("utf-8", errors="ignore").splitlines()[-50:]
        except Exception:
            tail = ["<sin log>"]
        raise RuntimeError("Timeout esperando el WAV de Piper.\n" + "\n".join(tail))

# instanciación (AHORA sí existen las constantes)
piper_worker = PiperWorker(PIPER_EXE, PIPER_VOICE, TTS_TMP_DIR, cfg=PIPER_CFG)

def ensure_piper_worker():
    with _piper_start_lock:
        if not piper_worker.proc or piper_worker.proc.poll() is not None:
            piper_worker.start()

def _synth_with_piper(text: str) -> bytes:
    ensure_piper_worker()
    return piper_worker.synth(text)



# =========================
# CONFIG - AJUSTÁ ESTAS RUTAS
# =========================
#LLAMA_SERVER_EXE = r"C:\llama-cuda\llama-server.exe"  # CUDA
LLAMA_SERVER_EXE = os.environ.get("LLAMA_SERVER_EXE", "llama-server")  # Vulkan por defecto
LLAMA_HOST       = os.environ.get("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT       = int(os.environ.get("LLAMA_PORT", "8080"))


def _active_llama_model() -> Path:
    preset = PROFILE_MANAGER.active_profile
    return PROFILE_MANAGER.resolve_model_path(preset.recommended_model_key)


def _llama_args() -> list[str]:
    return PROFILE_MANAGER.build_llama_args(LLAMA_HOST, LLAMA_PORT)

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
def _warm_all_assets(enabled_endpoints: Optional[Iterable[str]] = None):
    endpoints = set(enabled_endpoints or PROFILE_MANAGER.enabled_endpoints())
    if not endpoints:
        print("[warm] No endpoints seleccionados, warming omitido.")
        return
    print(f"[warm] Priming assets for endpoints: {', '.join(sorted(endpoints))}")
    try:
        if endpoints & {"llm", "alm", "slm"}:
            _warm_file(_active_llama_model())
        if endpoints & {"vlm", "clm"}:
            if 'VLM_CACHE_DIR' in globals() and os.path.isdir(VLM_CACHE_DIR):
                _warm_dir(VLM_CACHE_DIR, patterns=(
                    "snapshots/*/*.safetensors",
                    "snapshots/*/tokenizer.json",
                    "snapshots/*/preprocessor_config.json",
                    "snapshots/*/config.json",
                ))
            else:
                print("[warm] skip VLM: VLM_CACHE_DIR no definido o carpeta inexistente")
            if endpoints & {"clm"}:
                try:
                    _ensure_hf_vlm()
                except Exception as e:
                    print(f"[warm] HF VLM warm falló suavemente: {e}")
        if endpoints & {"alm", "slm"}:
            _warm_file(PIPER_VOICE)
            pv_json = PIPER_VOICE + ".json" if not PIPER_VOICE.endswith(".json") else PIPER_VOICE
            if os.path.isfile(pv_json):
                _warm_file(pv_json)
    except Exception as e:
        print(f"[warm] Warming skipped with error: {e}")


if os.getenv("PRELOAD_MODELS", "0") == "1":
    _warm_all_assets()


def _require_endpoint(name: str) -> None:
    if not PROFILE_MANAGER.is_endpoint_enabled(name):
        raise HTTPException(
            status_code=503,
            detail=f"Endpoint '/{name}' deshabilitado en la configuración.",
        )


def _apply_endpoint_defaults(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    settings = PROFILE_MANAGER.get_endpoint_settings(endpoint)
    for key, value in settings.items():
        if key not in payload or payload[key] is None:
            payload[key] = value
    return payload



# STT (usar CUDA o CPU)


# Use CUDA for STT if VRAM is available for faster inference
_primary_vram = SYSTEM_INFO.vram_gb_per_gpu[0] if SYSTEM_INFO.vram_gb_per_gpu else 0.0
USE_CUDA_FOR_STT = (
    SYSTEM_INFO.cuda_available
    and not PROFILE_MANAGER.active_profile.cpu_only
    and _primary_vram >= 8.0
)
STT_MODEL_SIZE   = "medium"
# Permitir override por variable de entorno
STT_DEVICE = os.getenv("STT_DEVICE", "cuda" if USE_CUDA_FOR_STT else "cpu")
STT_COMPUTE_TYPE = "float16" if STT_DEVICE == "cuda" else "int8"



# =========================
# MANAGER DE PROCESOS GPU
# =========================

import socket, json, time

def _wait_port_open(host: str, port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                s.connect((host, port))
                return True
            except Exception:
                time.sleep(0.25)
    return False

import threading

class GpuProcessManager:

    def __init__(self, profile_manager: ProfileManager):
        self.mode = None
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()
        self._profile_manager = profile_manager

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
        restart_required = self._profile_manager.consume_restart_flag(mode)
        with self._lock:
            if self.mode == mode and self.proc and self.proc.poll() is None and not restart_required:
                return

            if self.proc:
                self.stop()
                time.sleep(1.5)

            if mode == "llm":
                print("Starting llama-server...")
                profile_desc = self._profile_manager.describe_active_profile()
                print(
                    f"  -> Perfil {profile_desc['key']} ({profile_desc['title']}) "
                    f"modelo={profile_desc['model']} ctx={profile_desc['ctx_size']} ngl={profile_desc['n_gpu_layers']}"
                )
                log_path = LOG_DIR / "llama-server.log"
                try:
                    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                    llama_args = self._profile_manager.build_llama_args(LLAMA_HOST, LLAMA_PORT)
                    self.proc = _spawn_with_tee([LLAMA_SERVER_EXE] + llama_args, log_path, creationflags=flags)
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
                    # (triton stub omitted; see bugfix)

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

    def is_running(self, mode: str = "llm") -> bool:
        with self._lock:
            return self.mode == mode and self.proc and self.proc.poll() is None



# --- Global async httpx client ---
_httpx_async_client: httpx.AsyncClient = None

def get_httpx_client() -> httpx.AsyncClient:
    return _httpx_async_client

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):

    # ---- Banner DESPUÉS de los logs de Uvicorn ----
    async def _banner_after_uvicorn():
        # pequeño delay para quedar luego de:
        # "Application startup complete." y "Uvicorn running on ..."
        await asyncio.sleep(0.3)
        host_bind = os.getenv("HOST_BIND", "0.0.0.0")
        port_bind = int(os.getenv("PORT_BIND", "8000"))
        url_mostrable = (
            f"http://localhost:{port_bind}"
            if host_bind in ("0.0.0.0", "127.0.0.1")
            else f"http://{host_bind}:{port_bind}"
        )
        # usar el logger 'uvicorn.error' para que respete el formato "INFO:     ..."
        logging.getLogger("uvicorn.error").info(
            "Servidor levantado, ya puede usar los endpoints en %s  "
            "(/health, /llm, /clm, /vlm, /alm, /slm)",
            url_mostrable,
        )

    asyncio.create_task(_banner_after_uvicorn())
    global _httpx_async_client

    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=60)
    _httpx_async_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None),
        limits=limits,
        http2=False  # uvicorn doesn’t speak h2 by default; avoid negotiation overhead
    )


    # ⇩⇩ PRELOAD LLM ANTES DEL PRIMER REQUEST ⇩⇩
    if os.environ.get("PRELOAD_LLM", "0") == "1":
        print("[startup] Preloading Qwen3-coder-30b …")
        # Levanta el proceso y espera /health OK
        await asyncio.to_thread(manager.ensure_mode, "llm")

        # Warm-up mínimo para compilar kernels y llenar caches
        try:
            await _httpx_async_client.post(
                f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1/chat/completions",
                json={
                    "model": "qwen3-coder-30b",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "temperature": 0
                },
                timeout=10.0
            )
        except Exception as e:
            print("[startup] warm-up LLM fallo suave:", e)

    # Start Piper CLI worker at startup
    if os.environ.get("PRELOAD_PIPER", "0") == "1":
        print("[startup] Preloading Piper (CLI worker) …")
        await asyncio.to_thread(ensure_piper_worker)



    # (Opcional) si querés precargar también el VLM HF (ojo VRAM)
    if os.environ.get("PRELOAD_VLM", "0") == "1":
        print("[startup] Preloading Qwen2.5-VL-7B-Instruct …")
        await asyncio.to_thread(_ensure_hf_vlm)
        def _warmup_vlm():
            import torch
            from PIL import Image
            b = _ensure_hf_vlm()
            model, processor = b["model"], b["processor"]
            img = Image.new("RGB", (8, 8), (0, 0, 0))  # avoid ambiguous [3,1,1] warning
            inputs = processor(text=["Describe."], images=[img], return_tensors="pt").to(model.device)
            with torch.inference_mode():
                _ = model.generate(**inputs, max_new_tokens=1)  # no temperature=0
        try:
            await asyncio.to_thread(_warmup_vlm)
        except Exception as e:
            print("[startup] VLM warm-up skipped:", e)

    yield
    await _httpx_async_client.aclose()


app = FastAPI(title="IA Gateway (LLM/VLM/ALM)", lifespan=lifespan)
manager = GpuProcessManager(PROFILE_MANAGER)

if os.getenv("ENABLE_GUI", "1") != "0":
    gui_thread = start_gui_thread(
        PROFILE_MANAGER,
        PREFLIGHT_REPORT,
        LOG_DIR,
        GUI_LOGO_PATH,
        manager=manager,
        warm_callback=_warm_all_assets,
        splash_path=GUI_SPLASH_PATH,
    )
    if gui_thread:
        print("[gui] Interfaz Tkinter iniciada (puede minimizarse mientras se usa la API).")

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
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repeat_penalty: Optional[float] = None
    stop: Optional[list[str]] = None


from fastapi.concurrency import run_in_threadpool
from fastapi import Request, Query



@app.post("/llm")
async def llm_chat(
    payload: LLMChatPayload,
    request: Request,
    stream: bool = Query(False),
    client: httpx.AsyncClient = Depends(get_httpx_client),
):
    _require_endpoint("llm")
    payload_dict = payload.model_dump(exclude_none=True)
    payload_dict = _apply_endpoint_defaults("llm", payload_dict)
    try:
        await run_in_threadpool(manager.ensure_mode, "llm")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    url = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1/chat/completions"

    # Robust streaming detection: explicit query param or ?stream=true
    if stream or request.query_params.get("stream") == "true":
        def iter_llama():
            with httpx.stream("POST", url, json=payload_dict, timeout=None) as r:
                r.raise_for_status()
                for chunk in r.iter_bytes():
                    if chunk:
                        yield chunk
        return StreamingResponse(iter_llama(), media_type="text/event-stream; charset=utf-8")
    else:
        r = await client.post(url, json=payload_dict)
        import json
        # Asegurar UTF-8 en la respuesta
        try:
            data = r.json()
            content = json.dumps(data, ensure_ascii=False).encode('utf-8')
        except:
            content = r.content
        return Response(content=content, status_code=r.status_code, media_type="application/json; charset=utf-8")

# =========================
# /clm -> igual a /llm pero usando Qwen/Qwen2.5-VL-7B-Instruct (HF in-proc)
# =========================
from fastapi.concurrency import run_in_threadpool

@app.post("/clm")
async def clm_chat(payload: LLMChatPayload):
    _require_endpoint("clm")
    payload_dict = payload.model_dump(exclude_none=True)
    payload_dict = _apply_endpoint_defaults("clm", payload_dict)
    messages = payload_dict.get("messages") or [
        {"role": m.role, "content": m.content} for m in payload.messages
    ]
    hf_payload = {
        "model": payload_dict.get("model") or "Qwen/Qwen2.5-VL-7B-Instruct",
        "temperature": payload_dict.get("temperature"),
        "max_tokens": payload_dict.get("max_tokens"),
        "top_p": payload_dict.get("top_p"),
        "messages": messages,
    }
    try:
        resp = await run_in_threadpool(_hf_vlm_infer, hf_payload)
        import json
        return Response(content=json.dumps(resp), status_code=200, media_type="application/json")
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
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

@app.post("/vlm")
async def vlm_chat(payload: VLMChatPayload, client: httpx.AsyncClient = Depends(get_httpx_client)):
    _require_endpoint("vlm")
    payload_dict = payload.model_dump(exclude_none=True)
    payload_dict = _apply_endpoint_defaults("vlm", payload_dict)
    if VLM_BACKEND == "hf":
        try:
            resp = await run_in_threadpool(_hf_vlm_infer, payload_dict)
            import json
            return Response(content=json.dumps(resp), status_code=200, media_type="application/json")
        except Exception as e:
            print("[HF VLM] Exception:\n" + traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"HF VLM error: {e}")
    else:
        try:
            await run_in_threadpool(manager.ensure_mode, "vlm")
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        url = f"http://{VLM_HOST}:{VLM_PORT}/v1/chat/completions"
        r = await client.post(url, json=payload_dict)
        return Response(content=r.content, status_code=r.status_code, media_type="application/json")

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


    _require_endpoint("alm")

    def transcribe_audio():
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "audio").suffix or ".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            wav_path = tmp.name

        # Force greedy decoding for speed (beam_size=1)
        segments, info = stt_model.transcribe(
            wav_path,
            language=target_lang,
            vad_filter=True,
            beam_size=1,           # fastest
            without_timestamps=True
        )
        user_text = "".join([s.text for s in segments]).strip()
        os.unlink(wav_path)
        return user_text

    user_text = await run_in_threadpool(transcribe_audio)

    try:
        await run_in_threadpool(manager.ensure_mode, "llm")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    llm_payload = {
        "model": "qwen3-coder-30b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
    }
    llm_payload = _apply_endpoint_defaults("llm", llm_payload)
    alm_defaults = PROFILE_MANAGER.get_endpoint_settings("alm")
    for key, value in alm_defaults.items():
        llm_payload[key] = value
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


@app.post("/slm")
async def slm_pipeline(
    file: UploadFile = File(...),
    system_prompt: Optional[str] = Form("You are a helpful assistant."),
    tts: Optional[bool] = Form(True),
    target_lang: Optional[str] = Form("es"),
    client: httpx.AsyncClient = Depends(get_httpx_client)
):
    _require_endpoint("slm")
    import json
    from datetime import datetime

    # --- Transcripción (STT) ---
    def transcribe_audio() -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "audio").suffix or ".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            wav_path = tmp.name
        segments, info = stt_model.transcribe(
            wav_path,
            language=target_lang,
            vad_filter=True,
            beam_size=1,
            without_timestamps=True
        )
        try:
            os.unlink(wav_path)
        except: 
            pass
        return "".join(s.text for s in segments).strip()

    user_text = await run_in_threadpool(transcribe_audio)

    # --- Asegurar LLM arriba ---
    try:
        await run_in_threadpool(manager.ensure_mode, "llm")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # --- LLM (obtenemos texto primero) ---
    llm_payload = {
        "model": "qwen3-coder-30b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
    }
    llm_payload = _apply_endpoint_defaults("llm", llm_payload)
    slm_defaults = PROFILE_MANAGER.get_endpoint_settings("slm")
    for key, value in slm_defaults.items():
        if key != "chunk_size":
            llm_payload[key] = value
    url = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1/chat/completions"
        # ---- Banner DESPUÉS de los logs de Uvicorn ----
    async def _banner_after_uvicorn():
        # pequeño delay para quedar luego de:
        # "Application startup complete." y "Uvicorn running on ..."
        await asyncio.sleep(0.5)
        host_bind = os.getenv("HOST_BIND", "0.0.0.0")
        port_bind = int(os.getenv("PORT_BIND", "8000"))
        url_mostrable = (
            f"http://localhost:{port_bind}"
            if host_bind in ("0.0.0.0", "127.0.0.1")
            else f"http://{host_bind}:{port_bind}"
        )
        logging.getLogger("uvicorn.error").info(
            "Servidor levantado, ya puede usar los endpoints en %s  "
            "(/health, /llm, /clm, /vlm, /alm, /slm)",
            url_mostrable,
        )
    asyncio.create_task(_banner_after_uvicorn())
    r = await client.post(url, json=llm_payload)
    r.raise_for_status()
    llm_resp = r.json()
    answer_text = llm_resp["choices"][0]["message"]["content"]

    # --- Generador SSE (texto + audio chunked) ---
    async def sse_stream():
        # helpers SSE
        def sse_event(event: str, data_obj: dict | str):
            if isinstance(data_obj, (dict, list)):
                data = json.dumps(data_obj, ensure_ascii=False)
            else:
                data = str(data_obj)
            # SSE lines must end with double newline
            return f"event: {event}\ndata: {data}\n\n"



        try:
            # enviar texto primero
            yield sse_event("text", {
                "stt_text": user_text,
                "llm_text": answer_text
            })


            if tts:
                # Generar WAV y streammearlo como base64 incremental
                import tempfile
                import base64
                # Synthesize and write to a temp file for chunked reading
                def synth_to_file(text):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        wav_bytes = _synth_with_piper(text)
                        tmp.write(wav_bytes)
                        return tmp.name

                wav_path = await run_in_threadpool(synth_to_file, answer_text)
                chunk_kib = int(slm_defaults.get("chunk_size", 32) or 32)
                CHUNK = max(1, chunk_kib) * 1024
                seq = 0
                try:
                    with open(wav_path, "rb") as f:
                        encoder = base64.b64encode
                        while True:
                            chunk = f.read(CHUNK)
                            if not chunk:
                                break
                            b64 = encoder(chunk).decode("ascii")
                            last = f.tell() == os.fstat(f.fileno()).st_size
                            yield sse_event("audio", {
                                "seq": seq,
                                "last": last,
                                "mime": "audio/wav",
                                "data": b64
                            })
                            seq += 1
                            await asyncio.sleep(0)
                finally:
                    try:
                        os.unlink(wav_path)
                    except Exception:
                        pass

            # fin
            yield sse_event("done", {"ok": True, "ts": int(time.time())})

        except Exception as e:
            # error: notificar y cerrar
            yield sse_event("error", {"message": str(e)})

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",  # Nginx
        "Connection": "keep-alive",
    }
    return StreamingResponse(sse_stream(), media_type="text/event-stream", headers=headers)


@app.get("/health")
def health():
    return {"status": "ok", "mode": manager.mode}

if __name__ == "__main__":
    try:
        host = os.getenv("HOST_BIND", "0.0.0.0")
        port = int(os.getenv("PORT_BIND", "8000"))
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,
            log_level="info",  # less logging = less overhead
            backlog=2048,         # smoother under bursts
            # workers=1           # windows: keep 1 process; multiple workers add load without GPU sharing
        )
    finally:
        print("[main] finally -> stopping children …")
        manager.stop()
        _kill_orphans()
        _kill_by_port(LLAMA_PORT)
        _kill_by_port(VLM_PORT)
