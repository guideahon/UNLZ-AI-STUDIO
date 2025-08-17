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

## 🧩 Requisitos PREVIOS

- Windows 10/11 con PowerShell  
- Python 3.10+ (64-bit) https://www.python.org/downloads/
- Conexión a internet para descargar modelos
- CUDA Toolkit https://developer.nvidia.com/cuda-toolkit

---

## 📦 Instalación

> **PowerShell:** el continuador de línea es el **backtick** `` ` `` y debe ir como **último carácter** (sin espacios después).

### 1) Modelos

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

> Tip: si preferís sin backticks, ejecutá cada `hf download` en **una sola línea**.

🔊 Instalar Piper (TTS) desde GitHub — recomendado en Windows

El wrapper de pip install piper-tts puede fallar en Windows. Usá el binario nativo.

Abrí Releases: https://github.com/rhasspy/piper/releases

Descargá el asset piper_windows_amd64.zip de la versión más reciente.

Creá la carpeta y descomprimí directo ahí:

New-Item -ItemType Directory -Force -Path "C:\piper" | Out-Null
$zip = "$env:TEMP\piper_windows_amd64.zip"
Invoke-WebRequest "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip" -OutFile $zip
Expand-Archive $zip -DestinationPath "C:\piper" -Force
Remove-Item $zip

```

### 2) Dependencias

```powershell
# llama.cpp (Vulkan por winget; si luego querés CUDA, bajá el zip cublas desde Releases)
winget install llama.cpp

# Backend y gateway
pip install -U fastapi uvicorn httpx psutil python-multipart faster-whisper

# VLM (LMDeploy)
pip install -U lmdeploy

# PyTorch CUDA (CUDA 12.1 wheels; ajustá si usás otra versión)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Hugging Face (incluye el comando 'hf')
pip install -U huggingface_hub

# Necesario para llamar a qwen_vl
pip install -U transformers accelerate pillow requests qwen_vl_utils

#TurboMind offline mode
lmdeploy convert hf Qwen/Qwen2.5-VL-7B-Instruct `
  --dst-path "C:\models\qwen2.5-vl-7b-tm"
```

**Rutas esperadas por el gateway:**
- **GGUF:** `C:\models\qwen3-coder-30b\Qwen3-Coder-30B-A3B-Instruct.Q5_K_M.gguf`  
- **VLM:**  `C:\models\qwen2.5-vl-7b-hf`  
- **Piper:** `C:\piper\voices\es_AR\daniela_high\...`

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
# 1) Health (opcional)
Invoke-RestMethod "http://localhost:8000/health"

# 2) Body JSON
$body = @{
  model = "qwen3-coder-30b"
  messages = @(
    @{ role="system"; content="You are a helpful coding assistant." },
    @{ role="user";   content="Explicá PID con pseudocódigo." }
  )
} | ConvertTo-Json -Depth 5

# 3) Enviar (UTF-8) y ver la respuesta completa (raw JSON)
$raw = Invoke-WebRequest -Uri "http://localhost:8000/llm" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes($body)) `
| Select-Object -ExpandProperty Content
$resp = $raw | ConvertFrom-Json
$msg  = $resp.choices[0].message
if ($msg.content -is [array]) {
  ($msg.content | ForEach-Object { $_.text }) -join "`n"
} else {
  $msg.content
}


```

### VLM (visión, estilo OpenAI)
```powershell
# IMPORTANTE: la imagen debe ser accesible por el servidor (URL pública)
$imageUrl = "https://upload.wikimedia.org/wikipedia/commons/3/3a/PCB_SMD.jpg"

$body = @{
  model = "qwen2.5-vl-7b"
  messages = @(
    @{
      role    = "user"
      content = @(
        @{ type = "text";      text = "¿Qué dice esta placa y qué falla ves?" },
        @{ type = "image_url"; image_url = @{ url = $imageUrl } }
      )
    }
  )
} | ConvertTo-Json -Depth 8

# Enviar (UTF-8) y ver raw JSON
$raw = Invoke-WebRequest -Uri "http://localhost:8000/vlm" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes($body)) `
| Select-Object -ExpandProperty Content
$raw

# Extraer texto (string o array de partes)
$resp = $raw | ConvertFrom-Json
$msg  = $resp.choices[0].message
if ($msg.content -is [array]) {
  ($msg.content | ForEach-Object { $_.text }) -join "`n"
} else {
  $msg.content
}

```

### ALM (audio completo) IMPORTANTE, CORRER EL COMANDO CON LA TERMINAL ABIERTA EN LA MISMA RUTA DE gateway.py

```powershell 5

$ErrorActionPreference = "Stop"

# Rutas basadas en la carpeta actual
$here   = (Get-Location).Path
$in     = Join-Path $here 'test.wav'
$outDir = Join-Path $here 'audio-out'
$out    = Join-Path $outDir 'respuesta.wav'

if (-not (Test-Path $in)) { throw "No se encontró el archivo de entrada: $in" }
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

# ----- Enviar multipart/form-data con .NET HttpClient (compatible con PS 5.1) -----
Add-Type -AssemblyName System.Net.Http

$client = [System.Net.Http.HttpClient]::new()
$mp     = [System.Net.Http.MultipartFormDataContent]::new()

# Parte de archivo
$fs = [System.IO.FileStream]::new($in, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read)
$sc = [System.Net.Http.StreamContent]::new($fs)
$sc.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse("audio/wav")
$mp.Add($sc, "file", [System.IO.Path]::GetFileName($in))

# Partes de texto
$mp.Add([System.Net.Http.StringContent]::new("Sos un asistente de mecatrónica que responde de forma clara."), "system_prompt")
$mp.Add([System.Net.Http.StringContent]::new("true"), "tts")
$mp.Add([System.Net.Http.StringContent]::new("es"), "target_lang")

try {
  $resp = $client.PostAsync("http://localhost:8000/alm", $mp).Result
  if (-not $resp.IsSuccessStatusCode) {
    $body = $resp.Content.ReadAsStringAsync().Result
    throw "HTTP $($resp.StatusCode) $($resp.ReasonPhrase) :: $body"
  }
  $raw = $resp.Content.ReadAsStringAsync().Result
}
finally {
  $fs.Dispose()
  $client.Dispose()
}

# ----- Procesar respuesta -----
$jr = $raw | ConvertFrom-Json
$jr.stt_text
$jr.llm_text

# Mostrar error de TTS si vino
if ($jr.tts_error) {
  "TTS error: $($jr.tts_error)"
}

if ($jr.tts_audio) {
  $b64   = ($jr.tts_audio -split ",",2)[-1]
  $bytes = [Convert]::FromBase64String($b64)

  # Evitar escribir WAV vacío: 44 bytes es el mínimo del header RIFF/WAVE
  if ($bytes.Length -ge 44) {
    [IO.File]::WriteAllBytes($out, $bytes)
    "Audio guardado en: $out (bytes: $($bytes.Length))"
  } else {
    "TTS vino vacío (bytes=$($bytes.Length)). No se guardó archivo."
    if ($jr.tts_error) { "Detalle: $($jr.tts_error)" }
  }
} else {
  "No vino audio (tts_audio = null)." + ($(if ($jr.tts_error) { " Detalle: $($jr.tts_error)" } else { "" }))
}



```

```powershell 7
# Rutas (basadas en la carpeta actual)
$here   = (Get-Location).Path
$in     = Join-Path $here 'test.wav'
$outDir = Join-Path $here 'audio-out'
$out    = Join-Path $outDir 'respuesta.wav'

if (-not (Test-Path $in)) {
  Write-Error "No se encontró el archivo de entrada: $in"
  return
}

# Crear carpeta de salida si no existe
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

# Enviar WAV/MP3/M4A, etc. como multipart/form-data
$resp = Invoke-RestMethod -Uri "http://localhost:8000/alm" -Method Post -Form @{
  file          = Get-Item $in
  system_prompt = "Sos un asistente de mecatrónica que responde de forma clara."
  tts           = "true"
  target_lang   = "es"
}

# Ver STT y respuesta de LLM
$resp.stt_text
$resp.llm_text

# Guardar el WAV devuelto (es data:audio/wav;base64,XXXXX)
if ($resp.tts_audio) {
  $b64   = ($resp.tts_audio -split ",",2)[-1]
  $bytes = [Convert]::FromBase64String($b64)
  [IO.File]::WriteAllBytes($out, $bytes)
  "Audio guardado en: $out"
} else {
  "No vino audio (tts_audio = null)."
}


```

---


## ⚙️ Notas técnicas para `gateway.py`

- Para **LMDeploy 0.9.2**:
  - Usá repo HF posicional + --download-dir con tu carpeta local:
    ```python
    LMDEPLOY_CMD  = [sys.executable, "-m", "lmdeploy"]
    VLM_MODEL_ID  = "Qwen/Qwen2.5-VL-7B-Instruct"
    VLM_CACHE_DIR = r"C:\models\qwen2.5-vl-7b-hf"
    VLM_ARGS = [
      "serve", "api_server", VLM_MODEL_ID,
      "--backend", "pytorch", "--model-format", "hf",
      "--download-dir", VLM_CACHE_DIR,
      "--server-port", "9090",
    ]
    ```
  - Alternativa 100% offline: convertir a TurboMind y servir la carpeta convertida.

- Para **llama.cpp**:
  ```python
  LLAMA_ARGS = [
      "-m", LLAMA_MODEL,
      "--host", LLAMA_HOST, "--port", str(LLAMA_PORT),
      "--ctx-size", "8192",
      "--n-gpu-layers", "35",
      "-t", "12"
      # "--flash-attn"  # solo si usás CUDA/cuBLAS (no Vulkan)
    ]
  ```
  Si tu versión de llama da error con `--n-gpu-layers`, probá `--ngl`.

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
