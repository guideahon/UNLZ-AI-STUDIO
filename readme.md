![Logo Institucional](https://github.com/JonatanBogadoUNLZ/PPS-Jonatan-Bogado/blob/9952aac097aca83a1aadfc26679fc7ec57369d82/LOGO%20AZUL%20HORIZONTAL%20-%20fondo%20transparente.png)

# Universidad Nacional de Lomas de Zamora – Facultad de Ingeniería  
## UNLZ-AI-STUDIO

Guía práctica de *self-hosting* de LLMs/VLMs por API para universidades y hobbistas.

---

## 🚀 ¿Qué expone este proyecto?

- **/llm** – texto↔texto con **llama.cpp** (Qwen3-Coder-30B, GGUF)  
- **/vlm** – imagen+prompt con **Qwen2.5-VL-7B** vía **LMDeploy**  
- **/alm** – audio→texto (**STT**) → LLM → texto→voz (**TTS**)

➡️ **Auto-switch**: el gateway levanta/para `llama-server` o `lmdeploy` para no pelear VRAM en una sola GPU.

---

## 🧩 Requisitos

- Windows 10/11 con PowerShell  
- Python 3.10+ (64-bit)  
- NVIDIA driver actualizado (CUDA opcional)  
- Conexión a internet para descargar modelos

---

## 📦 Instalación

> **PowerShell:** el continuador de línea es el **backtick** `` ` `` y debe ir como **último carácter** (sin espacios después).

### 1) Dependencias

```powershell
# llama.cpp (Vulkan por winget; si luego querés CUDA, bajá el zip cublas desde Releases)
winget install llama.cpp

# Backend y gateway
pip install -U fastapi uvicorn httpx psutil python-multipart faster-whisper

# VLM (LMDeploy)
pip install -U lmdeploy

# Hugging Face (incluye el comando 'hf')
pip install -U huggingface_hub

# TTS (instala el binario 'piper' en tu PATH de Python)
pip install -U piper-tts
```

### 2) Modelos

```powershell
# LLM (GGUF) – Qwen3-Coder-30B-A3B-Instruct (Q5_K_M)
hf download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF `
  --include "*Q5_K_M*.gguf" `
  --local-dir "C:\models\qwen3-coder-30b"

# VLM (safetensors oficial para LMDeploy) – Qwen2.5-VL-7B-Instruct
hf download Qwen/Qwen2.5-VL-7B-Instruct `
  --local-dir "C:\models\qwen2.5-vl-7b-hf"

# Piper: modelo + config (ambos necesarios) – voz es_AR/daniela/high
hf download rhasspy/piper-voices `
  --include "es/es_AR/daniela/high/es_AR-daniela-high.onnx*" `
  --local-dir "C:\piper\voices\es_AR\daniela_high"

```

**Rutas esperadas por el gateway:**
- **GGUF:** `C:\models\qwen3-coder-30b\Qwen3-Coder-30B-A3B-Instruct.Q5_K_M.gguf`  
- **VLM:**  `C:\models\qwen2.5-vl-7b-hf`  
- **Piper:** `C:\piper\voices\es_AR\daniela_high\...`

> Tip: si preferís sin backticks, ejecutá cada `hf download` en **una sola línea**.

---

## ▶️ Ejecutar el gateway

```powershell
python .\gateway.py
# (o)
uvicorn gateway:app --host 0.0.0.0 --port 8000
```

- **/llm** → proxyea a **llama-server** (si no está, lo levanta y apaga `lmdeploy`)  
- **/vlm** → proxyea a **LMDeploy** (si no está, lo levanta y apaga `llama-server`)  
- **/alm** → corre **STT** (CPU por defecto), luego **/llm**, y **TTS** con Piper (devuelve WAV en **data:base64**)

---

## 🧪 Ejemplos (PowerShell)

### LLM (texto)
```powershell
$body = @{
  model = "qwen3-coder-30b"
  messages = @(
    @{ role="system"; content="You are a helpful coding assistant." },
    @{ role="user";   content="Explicá PID con pseudocódigo." }
  )
} | ConvertTo-Json -Depth 3

Invoke-RestMethod -Uri "http://localhost:8000/llm" -Method Post -ContentType "application/json" -Body $body
```

### VLM (visión, estilo OpenAI)
```powershell
$body = @{
  model = "qwen2.5-vl-7b"
  messages = @(
    @{ role="user"; content=@(
      @{ type="text"; text="¿Qué dice esta placa y qué falla ves?" },
      @{ type="image_url"; image_url=@{ url="https://TU_SERVIDOR/imagen.jpg" } }
    )}
  )
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Uri "http://localhost:8000/vlm" -Method Post -ContentType "application/json" -Body $body
```

### ALM (audio completo)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/alm" -Method Post -Form @{
  file = Get-Item "C:\audios\pregunta.wav"
  system_prompt = "Sos un asistente de mecatrónica que responde de forma clara."
  tts = "true"
  target_lang = "es"
}
```

---

## ⚙️ Notas técnicas para `gateway.py`

- Para **LMDeploy** usá **pesos HF (safetensors)**, **no GGUF**.  
- `VLM_MODEL_PATH = r"C:\models\qwen2.5-vl-7b-hf"`  
- Para mayor robustez en Windows:
  ```python
  LMDEPLOY_CMD = [sys.executable, "-m", "lmdeploy"]
  # y al lanzar:
  subprocess.Popen(LMDEPLOY_CMD + VLM_ARGS, ...)
  ```
- `LLAMA_ARGS = [
    "-m", LLAMA_MODEL,
    "--host", LLAMA_HOST, "--port", str(LLAMA_PORT),
    "--ctx-size", "8192",
    "--n-gpu-layers", "35",
    "-t", "12"
    # "--flash-attn"  # solo si usás CUDA/cuBLAS (no Vulkan)
  ]`
- Si tu versión de llama da error con `--n-gpu-layers`, probá `--ngl`.

---

## 🛡️ Consejos rápidos

- Si usás **llama Vulkan** (winget), quitá `--flash-attn` en `LLAMA_ARGS`.  
- ¿OOM? Bajá `--n-gpu-layers` o `--ctx-size`, o cuantizá a **Q4_K_M**.  
- **STT en GPU**: `USE_CUDA_FOR_STT=True` (puede ir más lento si coincide con el LLM).  
- Para exponer en LAN, abrí el **puerto 8000** en el firewall de Windows:
  ```powershell
  netsh advfirewall firewall add rule name="UNLZ-AI-STUDIO 8000" dir=in action=allow protocol=TCP localport=8000
  ```

---

## 📂 Estructura del repositorio

- Código fuente del **gateway** y utilitarios  
- Documentación técnica y guías de uso  
- Ejemplos de clientes (Python/JS) para consumir la API

---

## 📜 Licencia y uso

Los proyectos aquí incluidos pueden tener diferentes licencias, según lo definido por sus autores.  
Antes de usar o modificar, revisá el archivo **LICENSE** correspondiente.

---

## 🌐 Sitios

- **Página principal:** https://unlzfi.github.io/  
- **Repositorios:** https://github.com/orgs/UNLZFI/repositories  
- **Facultad de Ingeniería – UNLZ:** https://ingenieria.unlz.edu.ar/

---

## 📬 Contacto

**Facultad de Ingeniería – UNLZ**  
Carrera de Ingeniería en Mecatrónica
