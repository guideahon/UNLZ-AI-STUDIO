import os
import sys
import time
import signal
import socket
import logging
import threading
import subprocess
import psutil
import httpx
from typing import Optional, Dict, Any
from pathlib import Path

# Constants needed by Manager
LLAMA_SERVER_EXE = os.environ.get("LLAMA_SERVER_EXE", "llama-server")
LLAMA_HOST = os.environ.get("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))

# VLM Constants (mirrored from gateway for proper process launching)
LMDEPLOY_CMD = [sys.executable, "-m", "lmdeploy"]
VLM_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_CACHE_DIR = r"C:\models\qwen2.5-vl-7b-hf"
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

PRINT_CHILD_LOGS = False

def _spawn_with_tee(cmd, log_path: str, env=None, creationflags=0):
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
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
        prefix = f"[{os.path.basename(log_path)}] "
        for line in proc.stdout:
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

class GpuProcessManager:
    def __init__(self, profile_manager, log_dir="logs"):
        self.active_processes = {} # key -> {proc, config}
        self._lock = threading.RLock()
        self._profile_manager = profile_manager
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def start_process(self, key: str, config: Dict[str, Any]):
        """
        Starts a process for the given key (e.g., 'llm_chat', 'llm_service').
        config must contain: 'model_path', 'port', 'host'
        """
        proc_info = None
        wait_target = None
        
        # 1. PREPARE COMMAND (No Lock Needed yet)
        mode_type = "llm"
        if "vlm" in key: mode_type = "vlm"
        elif "alm" in key: mode_type = "alm"
        elif "slm" in key: mode_type = "slm"
        
        cmd = []
        env = None
        cwd = None
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        log_file = f"{key}.log"
        
        if mode_type == "llm":
            logging.info(f"Configuring {key} (llama-server) on port {config['port']}...")
            cmd = [
                LLAMA_SERVER_EXE,
                "-m", str(config["model_path"]),
                "--host", config.get("host", "127.0.0.1"),
                "--port", str(config["port"]),
                "--ctx-size", str(config.get("ctx_size", 2048)),
                "--n-gpu-layers", str(config.get("n_gpu_layers", 99)),
                "--threads", str(config.get("threads", 4)),
                "--no-mmap"
            ]
            if config.get("flash_attn"): cmd.append("--flash-attn")
            wait_target = ("health", config.get("host", "127.0.0.1"), config["port"])
            
        elif mode_type == "vlm":
            logging.info(f"Configuring {key} (llama-server VLM) on port {config['port']}...")
            # VLM now uses llama-server with mmproj
            # Path: C:\models\qwen2.5-vl-gguf\Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
            # mmproj: C:\models\qwen2.5-vl-gguf\mmproj-F16.gguf (Standard Unsloth pattern)
            
            model_dir = Path(r"C:\models\qwen2.5-vl-gguf")
            model_path = model_dir / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
            
            # Find existing mmproj (Order matters: check F16 first, then BF16, then legacy)
            mmproj_candidates = ["mmproj-F16.gguf", "mmproj-BF16.gguf", "mmproj-model-f16.gguf", "Qwen_Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"]
            mmproj_path = None
            for cand in mmproj_candidates:
                p = model_dir / cand
                if p.exists():
                    mmproj_path = p
                    break
            
            if not mmproj_path:
                 # Fallback to default if somehow nothing found but we want to try starting
                 mmproj_path = model_dir / "mmproj-F16.gguf"

            cmd = [
                LLAMA_SERVER_EXE,
                "-m", str(model_path),
                "--mmproj", str(mmproj_path), 
                "--host", config.get("host", "127.0.0.1"),
                "--port", str(config["port"]),
                "--n-gpu-layers", str(config.get("n_gpu_layers", 99)),
                "--ctx-size", "4096", # Vision needs context
                "--threads", "4"
            ]
            wait_target = ("health", config.get("host", "127.0.0.1"), config["port"])
            
        elif mode_type == "alm":
            logging.info(f"Starting ALM (Whisper) on port {config['port']}...")
            script = os.path.join(os.path.dirname(__file__), "modules", "alm_server.py")
            model_path = r"C:\models\whisper-large-v3" # Hardcoded default for now
            
            cmd = [sys.executable, script, "--port", str(config['port']), "--model-path", model_path]
            if sys.platform != "win32": cmd.append("--device"); cmd.append("cpu") # Default CPU if not forced
            
            wait_target = ("health", config.get("host", "127.0.0.1"), config["port"])
            
        elif mode_type == "slm":
            logging.info(f"Starting SLM (TTS) on port {config['port']}...")
            script = os.path.join(os.path.dirname(__file__), "modules", "slm_server.py")
            model_path = r"C:\models\xtts-v2"
            
            cmd = [sys.executable, script, "--port", str(config['port']), "--model-path", model_path]
            # XTTS might need C++ build tools or specific env, but let's try direct launch
            
            wait_target = ("health", config.get("host", "127.0.0.1"), config["port"])

        # 2. START PROCESS (Holds Lock briefly)
        with self._lock:
            if self.is_running(key):
                logging.info(f"Process {key} already running.")
                return

            self.stop(key) # Kill existing
            
            try:
                log_path = os.path.join(self.log_dir, log_file)
                config["proc"] = _spawn_with_tee(cmd, log_path, env=env, creationflags=creationflags)
                self.active_processes[key] = config
                proc_info = config["proc"]
            except Exception as e:
                logging.error(f"Failed to spawn {key}: {e}")
                raise e

        # 3. WAIT HEALTH (Outside Lock)
        if wait_target and proc_info:
            logging.info(f"Waiting for {key} to be ready...")
            try:
                if wait_target[0] == "health":
                     self._wait_health(wait_target[1], wait_target[2], timeout=60, proc=proc_info)
                elif wait_target[0] == "http":
                     self._wait_http_ok(wait_target[1], timeout=480, proc=proc_info)
                logging.info(f"{key} READY.")
            except Exception as e:
                logging.error(f"Health check failed for {key}: {e}")
                self.stop(key) # Cleanup on failure
                raise e

    def stop(self, key: str = None):
        """Stops specific key, or ALL if key is None."""
        with self._lock:
            if key is None:
                keys = list(self.active_processes.keys())
                for k in keys: self.stop(k)
                return

            if key in self.active_processes:
                proc = self.active_processes[key].get("proc")
                if proc:
                    logging.info(f"Stopping {key}...")
                    self._kill_proc(proc)
                del self.active_processes[key]

    def is_running(self, key: str) -> bool:
        with self._lock:
            if key in self.active_processes:
                proc = self.active_processes[key].get("proc")
                if proc and proc.poll() is None:
                    return True
            return False

    def _kill_proc(self, proc):
        # ... logic from _kill_tree / stop ...
        try:
            if os.name == "nt":
                subprocess.run(["taskkill","/PID",str(proc.pid),"/T","/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            else:
                proc.terminate()
        except: pass

    def _wait_health(self, host, port, timeout=60, proc=None):
        url = f"http://{host}:{port}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc and proc.poll() is not None:
                raise RuntimeError("Process exited prematurely")
            try:
                if httpx.get(url, timeout=1).status_code == 200: return
            except: pass
            time.sleep(1)
        raise RuntimeError("Health check timed out")

    def _wait_http_ok(self, url: str, timeout=120, proc: Optional[subprocess.Popen]=None):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc is not None and proc.poll() is not None:
                rc = proc.returncode
                raise RuntimeError(f"process exited with code {rc}")
            try:
                r = httpx.get(url, timeout=5.0)
                if r.status_code == 200:
                    return
            except:
                pass
            time.sleep(1.0)
        raise RuntimeError(f"{url} timed out after {timeout}s.")

    # --- Installation Management ---
    def get_service_status(self, service: str) -> dict:
        """Returns status dict: {'installed': bool, 'running': bool}"""
        installed = self.check_installed(service)
        running = self.is_running(service)
        return {"installed": installed, "running": running}

    def check_installed(self, service: str) -> bool:
        """Checks if the required assets for the service exist."""
        # LLM / CLM (Logic)
        # LLM / CLM (Shared Logic but specific markers)
        if "llm" in service or "clm" in service:
            base_dir = Path(r"C:\models\qwen2.5-coder-7b")
            # For llm_chat or manual, return True (assumed user managed)
            if service == "llm_chat": return True
            
            # Check specific marker
            marker_name = "llm.marker" if "llm" in service else "clm.marker"
            marker = base_dir / marker_name
            
            # Also check file
            model_file = base_dir / "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
            
            return marker.exists() and model_file.exists()

        elif "vlm" in service:
            model_dir = Path(r"C:\models\qwen2.5-vl-gguf")
            # Unsloth names
            model_exists = (model_dir / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf").exists()
            # Check any valid projector
            mmproj_exists = any((model_dir / f).exists() for f in ["mmproj-F16.gguf", "mmproj-BF16.gguf", "mmproj-model-f16.gguf"])
            return model_exists and mmproj_exists
            
        elif "alm" in service: # Audio (Whisper)
            return Path(r"C:\models\whisper-large-v3").exists()
            
        elif "slm" in service: # Speech (TTS)
            return Path(r"C:\models\xtts-v2").exists()
            
        return False

    def install_service(self, service: str, progress_callback=None):
        """Downloads assets for the service."""
        # Monkeypatch stdout/stderr for tqdm compatibility in pythonw
        class DummyWriter:
            def write(self, *args, **kwargs): pass
            def flush(self, *args, **kwargs): pass
            def isatty(self): return False
            
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            # Force dummy writers if None (pythonw case)
            if sys.stdout is None: sys.stdout = DummyWriter()
            if sys.stderr is None: sys.stderr = DummyWriter()
            
            os.environ["HF_HUB_DISABLE_PROGRESS_BAR"] = "0" # Try enabling but catching output? Or keep disabled.
            # actually keep disabled to be safe, but the patch handles 'write' calls anyway.
            os.environ["HF_HUB_DISABLE_PROGRESS_BAR"] = "1" 
            
            from huggingface_hub import hf_hub_download, snapshot_download
            
            if "llm" in service or "clm" in service:
                # Both use same default coding model for now for simplicity, or user can choose
                repo = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
                filename = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
                dest_dir = Path(r"C:\models\qwen2.5-coder-7b")
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                if progress_callback: progress_callback(f"Downloading {filename}...")
                hf_hub_download(repo_id=repo, filename=filename, local_dir=dest_dir, local_dir_use_symlinks=False)
                
                # Create Marker
                marker_name = "llm.marker" if "llm" in service else "clm.marker"
                (dest_dir / marker_name).touch()
                
            elif "vlm" in service:
                 # Switch to Unsloth repo which is reliable for GGUFs
                 repo = "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"
                 
                 # Filenames 
                 filename_model = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
                 # Unsloth uses simplified names
                 filename_mmproj = "mmproj-F16.gguf"
                 
                 files_to_download = [filename_model]
                 # Add primary mmproj to default list, but we have logic below for candidates.
                 # Actually, let's just use the robust candidate logic.
                 
                 dest_dir = Path(r"C:\models\qwen2.5-vl-gguf")
                 dest_dir.mkdir(parents=True, exist_ok=True)
                 
                 # Download Model
                 if progress_callback: progress_callback(f"Downloading {filename_model}...")
                 try:
                    hf_hub_download(repo_id=repo, filename=filename_model, local_dir=dest_dir, local_dir_use_symlinks=False)
                 except Exception as e:
                     logging.error(f"Failed to download model: {e}")
                     raise e

                 # Download MMProj
                 mmproj_candidates = ["mmproj-F16.gguf", "mmproj-BF16.gguf", "mmproj-model-f16.gguf"]
                 mmproj_success = False
                 for mmproj_file in mmproj_candidates:
                     if (dest_dir / mmproj_file).exists():
                         mmproj_success = True
                         break
                     
                     try:
                         if progress_callback: progress_callback(f"Downloading {mmproj_file}...")
                         hf_hub_download(repo_id=repo, filename=mmproj_file, local_dir=dest_dir, local_dir_use_symlinks=False)
                         mmproj_success = True
                         break # Stop after success
                     except Exception as e:
                         logging.warning(f"Candidate {mmproj_file} not found: {e}")
                 
                 if not mmproj_success:
                     logging.warning("Could not download any mmproj file. VLM might fail or run text-only.")

            elif "alm" in service: # Whisper
                 if progress_callback: progress_callback("Downloading generic Whisper assets...")
                 dest_dir = Path(r"C:\models\whisper-large-v3")
                 dest_dir.mkdir(parents=True, exist_ok=True)
                 # Placeholder: create a marker file to simulate install since we don't have binary
                 (dest_dir / "installed.marker").touch()
                 
            elif "slm" in service: # TTS
                 if progress_callback: progress_callback("Downloading generic TTS assets...")
                 dest_dir = Path(r"C:\models\xtts-v2")
                 dest_dir.mkdir(parents=True, exist_ok=True)
                 (dest_dir / "installed.marker").touch()

            if progress_callback: progress_callback("Done.")
            
        except Exception as e:
            logging.error(f"Install failed for {service}: {e}")
            raise e
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def uninstall_service(self, service: str):
        """Deletes assets for the service, respecting shared dependencies."""
        import shutil
        
        # Shared LLM/CLM
        if "llm" in service or "clm" in service:
            path = Path(r"C:\models\qwen2.5-coder-7b")
            marker_name = "llm.marker" if "llm" in service else "clm.marker"
            
            if not path.exists(): return
            
            # Remove specific marker
            marker = path / marker_name
            if marker.exists():
                marker.unlink()
            
            # Check if any other markers exist
            others = list(path.glob("*.marker"))
            if others:
                logging.info(f"Skipping deletion of {path} because other services use it: {[m.name for m in others]}")
                return
            
            # No other markers? Safe to delete
            shutil.rmtree(path)
            return

        # Exclusive services
        path = None
        if "vlm" in service:
             path = Path(r"C:\models\qwen2.5-vl-gguf")
        elif "alm" in service:
             path = Path(r"C:\models\whisper-large-v3")
        elif "slm" in service:
             path = Path(r"C:\models\xtts-v2")
             
        if path and path.exists():
            shutil.rmtree(path)
