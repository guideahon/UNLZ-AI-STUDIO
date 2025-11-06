"""
Runtime profile detection and recommendation helpers for UNLZ AI Studio.

This module centralises the heuristics we use to adapt llama.cpp launch
parameters to the hardware that is available on the host machine.
"""

from __future__ import annotations

import json
import os
import platform
import threading
import time
from dataclasses import asdict, dataclass
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import psutil

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dependency in practice
    torch = None


@dataclass
class SystemInfo:
    python_version: str
    platform: str
    cpu_name: str
    cpu_threads: int
    ram_gb: float
    gpu_count: int
    gpu_names: List[str]
    vram_gb_per_gpu: List[float]
    cuda_available: bool

    def to_display_dict(self) -> Dict[str, str]:
        primary_gpu = self.gpu_names[0] if self.gpu_names else "N/A"
        primary_vram = f"{self.vram_gb_per_gpu[0]:.1f} GB" if self.vram_gb_per_gpu else "N/A"
        return {
            "Python": self.python_version,
            "SO": self.platform,
            "CPU": f"{self.cpu_name} ({self.cpu_threads} hilos)",
            "RAM": f"{self.ram_gb:.1f} GB",
            "GPU": primary_gpu,
            "VRAM": primary_vram,
            "CUDA": "si" if self.cuda_available else "no",
        }


@dataclass
class ProfilePreset:
    key: str
    title: str
    description: str
    min_vram_gb: float
    min_ram_gb: float
    recommended_model_key: str
    n_gpu_layers: int
    ctx_size: int
    threads: int
    batch_size: int
    priority: int
    cpu_only: bool = False
    user_configurable: bool = False

    def fits(self, info: SystemInfo) -> bool:
        if self.user_configurable:
            return True
        enough_ram = info.ram_gb >= self.min_ram_gb
        enough_vram = self.cpu_only or (info.vram_gb_per_gpu and info.vram_gb_per_gpu[0] >= self.min_vram_gb)
        has_gpu = not self.cpu_only and info.gpu_count > 0
        if self.cpu_only:
            return enough_ram and (info.gpu_count == 0 or info.vram_gb_per_gpu[0] <= self.min_vram_gb + 0.5)
        return enough_ram and enough_vram and has_gpu


def _default_model_registry() -> Dict[str, Path]:
    base_dir = Path(os.environ.get("LLAMA_MODEL_DIR", r"C:\models"))
    registry = {
        "qwen3-coder-30b-q5": Path(
            os.environ.get(
                "LLAMA_MODEL_30B",
                base_dir / "qwen3-coder-30b" / "Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf",
            )
        ),
        "qwen3-coder-14b-q4": Path(
            os.environ.get(
                "LLAMA_MODEL_14B",
                base_dir / "qwen3-coder-14b" / "Qwen3-Coder-14B-Instruct-Q4_K_M.gguf",
            )
        ),
        "qwen3-coder-7b-q4": Path(
            os.environ.get(
                "LLAMA_MODEL_7B",
                base_dir / "qwen3-coder-7b" / "Qwen3-Coder-7B-Instruct-Q4_K_M.gguf",
            )
        ),
    }
    return registry


DEFAULT_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "qwen3-coder-30b-q5": {"min_vram_gb": 20.0, "prompt_latency_ms": 1850.0},
    "qwen3-coder-14b-q4": {"min_vram_gb": 12.0, "prompt_latency_ms": 980.0},
    "qwen3-coder-7b-q4": {"min_vram_gb": 6.0, "prompt_latency_ms": 640.0},
}

DEFAULT_ENDPOINT_FLAGS: Dict[str, bool] = {
    "llm": True,
    "clm": True,
    "vlm": False,
    "alm": True,
    "slm": True,
}

ENDPOINT_FIELD_SCHEMA: Dict[str, Dict[str, type]] = {
    "llm": {
        "temperature": float,
        "top_p": float,
        "top_k": int,
        "max_tokens": int,
        "repeat_penalty": float,
    },
    "clm": {
        "temperature": float,
        "top_p": float,
        "max_tokens": int,
    },
    "vlm": {
        "temperature": float,
        "top_p": float,
        "max_new_tokens": int,
    },
    "alm": {
        "temperature": float,
        "top_p": float,
        "max_tokens": int,
    },
    "slm": {
        "temperature": float,
        "top_p": float,
        "max_tokens": int,
        "chunk_size": int,
    },
}

DEFAULT_ENDPOINT_SETTINGS: Dict[str, Dict[str, object]] = {
    "llm": {"temperature": 0.2, "top_p": 0.9, "top_k": 40, "max_tokens": 512, "repeat_penalty": 1.1},
    "clm": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 512},
    "vlm": {"temperature": 0.2, "top_p": 0.9, "max_new_tokens": 512},
    "alm": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 256},
    "slm": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 256, "chunk_size": 32},
}

DEFAULT_PRESETS: List[ProfilePreset] = [
    ProfilePreset(
        key="ultra",
        title="Ultra (workstation)",
        description="Pensado para GPUs >= 24 GB y 96 GB de RAM. Usa el modelo 30B completo.",
        min_vram_gb=22.0,
        min_ram_gb=96.0,
        recommended_model_key="qwen3-coder-30b-q5",
        n_gpu_layers=60,
        ctx_size=8192,
        threads=24,
        batch_size=24,
        priority=100,
    ),
    ProfilePreset(
        key="alto",
        title="Alto rendimiento",
        description="Equipos con 16-24 GB de VRAM y 64 GB de RAM. Mantiene el modelo 30B con menos capas en GPU.",
        min_vram_gb=16.0,
        min_ram_gb=64.0,
        recommended_model_key="qwen3-coder-30b-q5",
        n_gpu_layers=42,
        ctx_size=6144,
        threads=18,
        batch_size=16,
        priority=90,
    ),
    ProfilePreset(
        key="balanceado",
        title="Balanceado",
        description="Buena opción para GPUs de 12 GB y 48 GB de RAM. Baja capas en GPU para evitar OOM.",
        min_vram_gb=12.0,
        min_ram_gb=48.0,
        recommended_model_key="qwen3-coder-14b-q4",
        n_gpu_layers=28,
        ctx_size=4096,
        threads=16,
        batch_size=12,
        priority=70,
    ),
    ProfilePreset(
        key="medio",
        title="Media GPU",
        description="Pensado para GPUs entre 8 y 10 GB con 32 GB de RAM.",
        min_vram_gb=8.0,
        min_ram_gb=32.0,
        recommended_model_key="qwen3-coder-14b-q4",
        n_gpu_layers=18,
        ctx_size=3072,
        threads=12,
        batch_size=8,
        priority=60,
    ),
    ProfilePreset(
        key="baja",
        title="Baja RAM / GPU chica",
        description="Ideal para GPUs tipo RTX 2060 / 3050 con 6 GB y 32 GB de RAM. Reduce más el contexto.",
        min_vram_gb=6.0,
        min_ram_gb=24.0,
        recommended_model_key="qwen3-coder-7b-q4",
        n_gpu_layers=8,
        ctx_size=2048,
        threads=10,
        batch_size=6,
        priority=50,
    ),
    ProfilePreset(
        key="cpu",
        title="CPU / sin GPU",
        description="Fallback sin GPU: usa el modelo 7B en CPU y contexto corto para no congelar el equipo.",
        min_vram_gb=0.0,
        min_ram_gb=16.0,
        recommended_model_key="qwen3-coder-7b-q4",
        n_gpu_layers=0,
        ctx_size=1536,
        threads=8,
        batch_size=4,
        priority=10,
        cpu_only=True,
    ),
    ProfilePreset(
        key="personalizado",
        title="Personalizado",
        description="Configura manualmente el modelo GGUF y parámetros de ejecución.",
        min_vram_gb=0.0,
        min_ram_gb=0.0,
        recommended_model_key="__custom__",
        n_gpu_layers=0,
        ctx_size=2048,
        threads=8,
        batch_size=4,
        priority=5,
        user_configurable=True,
    ),
]


def detect_system_info() -> SystemInfo:
    python_version = platform.python_version()
    platform_label = f"{platform.system()} {platform.release()}"
    cpu_name = platform.processor() or platform.machine()
    cpu_threads = psutil.cpu_count(logical=True) or 1
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    gpu_names: List[str] = []
    vram_list: List[float] = []
    cuda_available = False
    gpu_count = 0

    if torch is not None and torch.cuda.is_available():
        cuda_available = True
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpu_names.append(props.name)
            vram_list.append(round(props.total_memory / (1024**3), 2))

    return SystemInfo(
        python_version=python_version,
        platform=platform_label,
        cpu_name=cpu_name,
        cpu_threads=cpu_threads,
        ram_gb=ram_gb,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        vram_gb_per_gpu=vram_list,
        cuda_available=cuda_available,
    )


class ProfileManager:
    def __init__(
        self,
        system_info: SystemInfo,
        storage_dir: Path,
        presets: Optional[List[ProfilePreset]] = None,
        model_registry: Optional[Dict[str, Path]] = None,
        benchmarks: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.system_info = system_info
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._presets = {preset.key: preset for preset in (presets or DEFAULT_PRESETS)}
        self._model_registry = model_registry or _default_model_registry()
        self._benchmarks = benchmarks or DEFAULT_BENCHMARKS

        self._state_path = self.storage_dir / "runtime_profile.json"
        self._feedback_path = self.storage_dir / "tester_feedback.jsonl"
        self._lock = threading.RLock()
        self._pending_restarts: set[str] = set()
        state = self._load_state()
        if isinstance(state, dict):
            self._custom_config = state.get("custom_overrides", {}) or {}
            self._endpoint_config = state.get("endpoint_config", {}) or {}
            self._endpoint_settings = state.get("endpoint_settings", {}) or {}
            self._active_key = state.get("profile_key")
        else:
            self._custom_config = {}
            self._endpoint_config = {}
            self._endpoint_settings = {}
            self._active_key = None

        if self._active_key not in self._presets or not self._preset_supported(self._active_key):
            recommended = self.recommend_profile()
            self._active_key = recommended.key
            self._persist_state(recommendation=True)

    # --- Persistence -------------------------------------------------
    def _load_state(self) -> Dict[str, object]:
        if not self._state_path.exists():
            return {}
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _persist_state(self, recommendation: bool = False) -> None:
        payload = {
            "profile_key": self._active_key,
            "system": asdict(self.system_info),
            "ts": time.time(),
            "auto_selected": recommendation,
            "custom_overrides": self._custom_config,
            "endpoint_config": self._endpoint_config,
            "endpoint_settings": self._endpoint_settings,
        }
        try:
            self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    # --- Preset helpers ----------------------------------------------
    def _default_endpoint_config(self) -> Dict[str, bool]:
        return dict(DEFAULT_ENDPOINT_FLAGS)

    def _default_endpoint_settings(self) -> Dict[str, Dict[str, object]]:
        return deepcopy(DEFAULT_ENDPOINT_SETTINGS)

    def get_endpoint_config(self) -> Dict[str, bool]:
        cfg = self._default_endpoint_config()
        for key, value in self._endpoint_config.items():
            if key in cfg:
                cfg[key] = bool(value)
        if (cfg.get("alm") or cfg.get("slm")) and not cfg.get("llm"):
            cfg["llm"] = True
        return cfg

    def enabled_endpoints(self) -> List[str]:
        return [name for name, enabled in self.get_endpoint_config().items() if enabled]

    def is_endpoint_enabled(self, name: str) -> bool:
        cfg = self.get_endpoint_config()
        return bool(cfg.get(name, False))

    def update_endpoint_config(self, updates: Dict[str, object]) -> None:
        changed = False
        for key, value in updates.items():
            if key not in DEFAULT_ENDPOINT_FLAGS:
                continue
            new_val = bool(value)
            if new_val != bool(self.get_endpoint_config().get(key, False)):
                self._endpoint_config[key] = new_val
                changed = True
        if changed:
            cfg = self.get_endpoint_config()
            if (cfg.get("alm") or cfg.get("slm")) and not cfg.get("llm"):
                if not self._endpoint_config.get("llm"):
                    self._endpoint_config["llm"] = True
                changed = True
            self._persist_state()

    def get_endpoint_settings(self, endpoint: str) -> Dict[str, object]:
        base = dict(self._default_endpoint_settings().get(endpoint, {}))
        overrides = self._endpoint_settings.get(endpoint, {})
        base.update(overrides)
        return base

    def update_endpoint_settings(self, endpoint: str, updates: Dict[str, object]) -> Tuple[bool, str]:
        if endpoint not in ENDPOINT_FIELD_SCHEMA:
            return False, f"Endpoint desconocido: {endpoint}"
        schema = ENDPOINT_FIELD_SCHEMA[endpoint]
        cleaned: Dict[str, object] = {}
        for field, caster in schema.items():
            value = updates.get(field)
            if value is None:
                continue
            if value == "":
                continue
            try:
                if caster is int:
                    cleaned[field] = int(float(value))
                elif caster is float:
                    cleaned[field] = float(value)
                else:
                    cleaned[field] = value
            except (TypeError, ValueError):
                return False, f"Valor inválido para {endpoint}.{field}: {value}"
        if endpoint not in self._endpoint_settings:
            self._endpoint_settings[endpoint] = {}
        self._endpoint_settings[endpoint].update(cleaned)
        if endpoint == "llm":
            self._pending_restarts.add("llm")
        self._persist_state()
        return True, "Parámetros de endpoint guardados."

    def _preset_supported(self, key: str) -> bool:
        preset = self._presets.get(key)
        if not preset:
            return False
        if preset.user_configurable:
            return True
        if not preset.fits(self.system_info):
            return False
        model_path = self.resolve_model_path(preset.recommended_model_key)
        return model_path.exists()

    def list_presets(self) -> List[Dict[str, str]]:
        items = []
        for preset in sorted(self._presets.values(), key=lambda p: p.priority, reverse=True):
            status = "OK"
            if preset.user_configurable:
                cfg = self.get_custom_settings()
                model_path_str = (cfg.get("model_path") or "").strip()
                model_exists = bool(model_path_str) and Path(model_path_str).exists()
                status = "OK" if model_exists else "Configurar modelo"
            else:
                if not preset.fits(self.system_info):
                    status = "Fuera de rango"
                elif not self.resolve_model_path(preset.recommended_model_key).exists():
                    status = "Modelo faltante"
            items.append(
                {
                    "key": preset.key,
                    "title": preset.title,
                    "description": preset.description,
                    "status": status,
                }
            )
        return items

    def resolve_model_path(self, model_key: str) -> Path:
        if model_key == "__custom__":
            cfg = self.get_custom_settings()
            return Path(cfg.get("model_path", ""))
        path = self._model_registry.get(model_key)
        if path is None:
            return Path(model_key)
        return Path(path)

    @property
    def active_profile(self) -> ProfilePreset:
        return self._presets[self._active_key]

    def set_active_profile(self, key: str, force: bool = False) -> Tuple[bool, str]:
        with self._lock:
            if key not in self._presets:
                return False, f"Perfil desconocido: {key}"
            preset = self._presets[key]
            if preset.user_configurable:
                if not isinstance(self._custom_config, dict) or not self._custom_config:
                    self._custom_config = self._default_custom_config()
                    self._persist_state()
                self._active_key = key
                self._persist_state()
                self._pending_restarts.add("llm")
                cfg = self.get_custom_settings()
                model_path = Path(cfg.get("model_path", ""))
                msg = "Perfil personalizado activo. Ajustá los parámetros antes de usar el LLM."
                if not model_path.exists():
                    msg += f" Archivo pendiente: {model_path or '<sin definir>'}."
                return True, msg
            model_path = self.resolve_model_path(preset.recommended_model_key)
            if not model_path.exists():
                return False, f"El modelo '{model_path}' no existe en disco."
            if not preset.fits(self.system_info) and not force:
                return False, "El hardware actual no cumple los requisitos del perfil."
            self._active_key = key
            self._persist_state()
            self._pending_restarts.add("llm")
            note = "Perfil actualizado. Reiniciaremos el servidor LLM en el próximo uso."
            if not preset.fits(self.system_info) and force:
                note += " (forzado)"
            return True, note

    def _derive_launch_settings(self, preset: ProfilePreset) -> Dict[str, object]:
        vram = self.system_info.vram_gb_per_gpu[0] if self.system_info.vram_gb_per_gpu else 0.0
        threads = max(1, min(self.system_info.cpu_threads, preset.threads))
        ctx_size = preset.ctx_size
        if self.system_info.ram_gb < preset.min_ram_gb:
            ctx_size = max(1536, int(preset.ctx_size * 0.75))
        n_gpu_layers = preset.n_gpu_layers or 0
        if preset.cpu_only or vram <= 0.5:
            n_gpu_layers = 0
        else:
            n_gpu_layers = max(0, min(n_gpu_layers, self._suggested_gpu_layers(vram)))
        batch_size = preset.batch_size
        if preset.cpu_only or vram <= 0.5:
            batch_size = max(1, min(batch_size, 4))
        else:
            batch_size = max(2, min(batch_size, self._suggested_batch_size(vram)))
        model_path = self.resolve_model_path(preset.recommended_model_key)
        return {
            "model_path": model_path,
            "threads": threads,
            "ctx_size": ctx_size,
            "n_gpu_layers": int(n_gpu_layers),
            "batch_size": int(batch_size),
        }

    def _suggested_gpu_layers(self, vram_gb: float) -> int:
        table = [
            (24.0, 60),
            (20.0, 52),
            (16.0, 40),
            (12.0, 28),
            (10.0, 22),
            (8.0, 16),
            (6.0, 10),
            (4.0, 6),
        ]
        for limit, layers in table:
            if vram_gb >= limit:
                return layers
        return 0

    def _suggested_batch_size(self, vram_gb: float) -> int:
        table = [
            (24.0, 32),
            (16.0, 24),
            (12.0, 16),
            (10.0, 12),
            (8.0, 8),
            (6.0, 6),
            (4.0, 4),
        ]
        for limit, batch in table:
            if vram_gb >= limit:
                return batch
        return 2

    def build_llama_args(self, host: str, port: int) -> List[str]:
        preset = self.active_profile
        if preset.user_configurable:
            settings = self.get_custom_settings()
        else:
            settings = self._derive_launch_settings(preset)
        model_path = str(settings["model_path"])
        if not model_path:
            raise RuntimeError("Configuración personalizada sin modelo definido.")
        args = [
            "-m",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
            "--ctx-size",
            str(settings["ctx_size"]),
            "--threads",
            str(settings["threads"]),
            "--batch-size",
            str(settings["batch_size"]),
            "--no-mmap",
        ]
        args += ["--n-gpu-layers", str(settings["n_gpu_layers"])]
        tensor_split = str(settings.get("tensor_split", "") or "").strip()
        if tensor_split:
            args += ["--tensor-split", tensor_split]
        rope_freq_base = float(settings.get("rope_freq_base", 0.0) or 0.0)
        if rope_freq_base > 0:
            args += ["--rope-freq-base", str(rope_freq_base)]
        rope_freq_scale = float(settings.get("rope_freq_scale", 1.0) or 1.0)
        if abs(rope_freq_scale - 1.0) > 1e-6:
            args += ["--rope-freq-scale", str(rope_freq_scale)]
        if bool(settings.get("flash_attn")):
            args.append("--flash-attn")
        return args

    def recommend_profile(self) -> ProfilePreset:
        candidates = [
            preset
            for preset in sorted(self._presets.values(), key=lambda p: p.priority, reverse=True)
            if (
                not preset.user_configurable
                and preset.fits(self.system_info)
                and self.resolve_model_path(preset.recommended_model_key).exists()
            )
        ]
        if not candidates:
            return self._fallback_preset()

        feedback = self._load_feedback()
        preferred = self._apply_feedback_heuristics(candidates, feedback)
        return preferred

    def _fallback_preset(self) -> ProfilePreset:
        available = [
            preset
            for preset in sorted(self._presets.values(), key=lambda p: p.priority)
            if not preset.user_configurable and self.resolve_model_path(preset.recommended_model_key).exists()
        ]
        if available:
            return available[0]
        for preset in sorted(self._presets.values(), key=lambda p: p.priority):
            if not preset.user_configurable:
                return preset
        return self._presets["personalizado"]

    def _best_non_custom_preset(self) -> ProfilePreset:
        ordered = sorted(self._presets.values(), key=lambda p: p.priority, reverse=True)
        for preset in ordered:
            if preset.user_configurable:
                continue
            path = self.resolve_model_path(preset.recommended_model_key)
            if path.exists() and preset.fits(self.system_info):
                return preset
        for preset in ordered:
            if preset.user_configurable:
                continue
            path = self.resolve_model_path(preset.recommended_model_key)
            if path.exists():
                return preset
        for preset in ordered:
            if not preset.user_configurable:
                return preset
        return self._presets["personalizado"]

    def _default_custom_config(self) -> Dict[str, object]:
        base_preset = self._best_non_custom_preset()
        settings = self._derive_launch_settings(base_preset) if not base_preset.user_configurable else {
            "model_path": Path(self._model_registry.get("qwen3-coder-14b-q4", "")),
            "threads": max(1, min(self.system_info.cpu_threads, 12)),
            "ctx_size": 4096,
            "n_gpu_layers": 16,
            "batch_size": 8,
        }
        model_str = str(settings["model_path"])
        if model_str == ".":
            model_str = ""
        return {
            "model_path": model_str,
            "threads": int(settings["threads"]),
            "ctx_size": int(settings["ctx_size"]),
            "n_gpu_layers": int(settings["n_gpu_layers"]),
            "batch_size": int(settings["batch_size"]),
            "tensor_split": "",
            "rope_freq_base": 0.0,
            "rope_freq_scale": 1.0,
            "flash_attn": False,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
        }

    def get_custom_settings(self) -> Dict[str, object]:
        base = self._default_custom_config()
        overrides = {}
        if isinstance(self._custom_config, dict):
            overrides.update(self._custom_config)
        merged: Dict[str, Any] = {**base, **overrides}

        model_path = Path(str(merged.get("model_path", base["model_path"]))).expanduser()
        merged["model_path"] = str(model_path)

        def _sanitize(name: str, minimum: int, allow_zero: bool = False) -> int:
            value = merged.get(name, base[name])
            try:
                ivalue = int(value)
            except (TypeError, ValueError):
                ivalue = base[name]
            floor = 0 if allow_zero else minimum
            ivalue = max(floor, ivalue)
            return ivalue

        merged["ctx_size"] = _sanitize("ctx_size", 256)
        merged["n_gpu_layers"] = _sanitize("n_gpu_layers", 0, allow_zero=True)
        merged["threads"] = min(self.system_info.cpu_threads or 1, _sanitize("threads", 1))
        merged["batch_size"] = _sanitize("batch_size", 1)
        merged["top_k"] = _sanitize("top_k", 1)

        def _sanitize_float(name: str, minimum: float | None = None) -> float:
            value = merged.get(name, base.get(name))
            try:
                fvalue = float(value)
            except (TypeError, ValueError):
                fvalue = float(base.get(name, 0.0))
            if minimum is not None:
                fvalue = max(minimum, fvalue)
            return fvalue

        def _sanitize_bool(name: str) -> bool:
            value = merged.get(name, base.get(name, False))
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)

        merged["rope_freq_base"] = _sanitize_float("rope_freq_base", 0.0)
        merged["rope_freq_scale"] = _sanitize_float("rope_freq_scale", 0.01)
        merged["temperature"] = _sanitize_float("temperature", 0.0)
        merged["top_p"] = _sanitize_float("top_p", 0.0)
        merged["repeat_penalty"] = _sanitize_float("repeat_penalty", 0.1)
        merged["tensor_split"] = str(merged.get("tensor_split", "") or "").strip()
        merged["flash_attn"] = _sanitize_bool("flash_attn")
        return merged

    def update_custom_settings(self, updates: Dict[str, Any]) -> Tuple[bool, str]:
        settings = self.get_custom_settings()
        allowed = {
            "model_path",
            "ctx_size",
            "n_gpu_layers",
            "threads",
            "batch_size",
            "top_k",
            "tensor_split",
            "rope_freq_base",
            "rope_freq_scale",
            "flash_attn",
            "temperature",
            "top_p",
            "repeat_penalty",
        }
        for key, value in updates.items():
            if key not in allowed:
                continue
            if key == "model_path":
                path = Path(str(value)).expanduser()
                settings[key] = str(path)
            elif key in {"tensor_split"}:
                settings[key] = str(value).strip()
            elif key in {"rope_freq_base", "rope_freq_scale", "temperature", "top_p", "repeat_penalty"}:
                try:
                    settings[key] = float(value)
                except (TypeError, ValueError):
                    return False, f"Valor inválido para {key}: {value}"
            elif key == "flash_attn":
                settings[key] = str(value).strip().lower() in ("1", "true", "yes", "on")
            elif key in {"top_k", "ctx_size", "n_gpu_layers", "threads", "batch_size"}:
                try:
                    ivalue = int(value)
                except (TypeError, ValueError):
                    return False, f"Valor inválido para {key}: {value}"
                if key == "n_gpu_layers":
                    settings[key] = max(0, ivalue)
                else:
                    settings[key] = max(1, ivalue)
            else:
                try:
                    settings[key] = value
                except Exception:
                    pass

        settings["threads"] = min(self.system_info.cpu_threads or 1, int(settings["threads"]))
        self._custom_config = {
            "model_path": settings["model_path"],
            "ctx_size": int(settings["ctx_size"]),
            "n_gpu_layers": int(settings["n_gpu_layers"]),
            "threads": int(settings["threads"]),
            "batch_size": int(settings["batch_size"]),
            "top_k": int(settings["top_k"]),
            "tensor_split": settings["tensor_split"],
            "rope_freq_base": float(settings["rope_freq_base"]),
            "rope_freq_scale": float(settings["rope_freq_scale"]),
            "flash_attn": bool(settings["flash_attn"]),
            "temperature": float(settings["temperature"]),
            "top_p": float(settings["top_p"]),
            "repeat_penalty": float(settings["repeat_penalty"]),
        }
        self._persist_state()
        self._pending_restarts.add("llm")
        model_exists = Path(self._custom_config["model_path"]).exists()
        msg = "Configuración personalizada guardada."
        if not model_exists:
            msg += " Atención: el archivo del modelo no existe."
        return True, msg

    def autoconfigure_custom(self) -> Dict[str, object]:
        settings = self._default_custom_config()
        self._custom_config = settings
        self._persist_state()
        self._pending_restarts.add("llm")
        return settings

    def _apply_feedback_heuristics(
        self, candidates: List[ProfilePreset], feedback: List[Dict[str, str]]
    ) -> ProfilePreset:
        """Very light heuristic: if testers reported freezes for the top candidate + GPU, pick next."""
        if not feedback or len(candidates) == 1:
            return candidates[0]

        gpu_name = self.system_info.gpu_names[0] if self.system_info.gpu_names else "N/A"
        bad_presets = {
            entry.get("preset")
            for entry in feedback
            if entry.get("gpu") == gpu_name and entry.get("issue") in {"freeze", "oom"}
        }
        for preset in candidates:
            if preset.key not in bad_presets:
                return preset
        return candidates[-1]

    # --- Feedback management -----------------------------------------
    def _load_feedback(self) -> List[Dict[str, str]]:
        if not self._feedback_path.exists():
            return []
        rows: List[Dict[str, str]] = []
        try:
            with self._feedback_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            return []
        return rows

    def append_feedback(self, preset_key: str, notes: str, issue: str) -> None:
        record = {
            "preset": preset_key,
            "notes": notes,
            "issue": issue,
            "ts": time.time(),
            "gpu": self.system_info.gpu_names[0] if self.system_info.gpu_names else "N/A",
        }
        try:
            with self._feedback_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def get_feedback_summary(self) -> Dict[str, int]:
        rows = self._load_feedback()
        stats: Dict[str, int] = {}
        for entry in rows:
            key = entry.get("preset", "unknown")
            stats[key] = stats.get(key, 0) + 1
        return stats

    # --- Restart coordination ---------------------------------------
    def consume_restart_flag(self, mode: str) -> bool:
        with self._lock:
            if mode in self._pending_restarts:
                self._pending_restarts.remove(mode)
                return True
            return False

    # --- Debug helpers -----------------------------------------------
    def describe_active_profile(self) -> Dict[str, str]:
        preset = self.active_profile
        if preset.user_configurable:
            settings = self.get_custom_settings()
        else:
            settings = self._derive_launch_settings(preset)
        return {
            "key": preset.key,
            "title": preset.title,
            "model": str(settings["model_path"]),
            "ctx_size": str(settings["ctx_size"]),
            "threads": str(settings["threads"]),
            "n_gpu_layers": str(settings["n_gpu_layers"]),
            "batch_size": str(settings["batch_size"]),
        }


def run_dependency_checks(system_info: SystemInfo, log_dir: Path) -> Dict[str, Iterable[str]]:
    warnings: List[str] = []
    suggestions: List[str] = []

    major_minor = tuple(int(x) for x in system_info.python_version.split(".")[:2])
    if major_minor >= (3, 13):
        warnings.append(
            "Python 3.13 no está soportado por lmdeploy ni por torch estable. Recomendamos 3.10 / 3.11."
        )

    if torch is None:
        warnings.append("PyTorch no está instalado. Las funciones de GPU quedarán deshabilitadas.")
    else:
        version = getattr(torch, "__version__", "desconocido")
        cuda_version = getattr(torch.version, "cuda", "sin CUDA")
        if torch.cuda.is_available() and cuda_version and not cuda_version.startswith("12"):
            warnings.append(
                f"torch {version} usa CUDA {cuda_version}. Para GPUs RTX 20/30 sugerimos CUDA 12.1."
            )

    try:
        import importlib

        lmdeploy_spec = importlib.util.find_spec("lmdeploy")
        if lmdeploy_spec is None:
            warnings.append(
                "lmdeploy no está instalado en el entorno actual. El modo VLM usará únicamente el backend HF."
            )
        else:
            lmdeploy = importlib.import_module("lmdeploy")
            version = getattr(lmdeploy, "__version__", "desconocido")
            if major_minor >= (3, 12):
                warnings.append(
                    f"lmdeploy {version} puede presentar incompatibilidades con Python >= 3.12. "
                    "Considere un entorno 3.10."
                )
    except Exception as exc:  # pragma: no cover - defensivo
        warnings.append(f"No se pudo comprobar lmdeploy ({exc}).")

    suggestions.append(
        "Verifique que las rutas de modelos en C:\\models contengan las variantes 30B, 14B y 7B en formato GGUF."
    )

    report = {
        "warnings": warnings,
        "suggestions": suggestions,
        "system": asdict(system_info),
        "ts": time.time(),
    }
    try:
        (log_dir / "preflight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception:
        pass
    return report
