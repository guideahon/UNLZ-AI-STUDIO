![Logo Institucional](https://github.com/JonatanBogadoUNLZ/PPS-Jonatan-Bogado/blob/9952aac097aca83a1aadfc26679fc7ec57369d82/LOGO%20AZUL%20HORIZONTAL%20-%20fondo%20transparente.png)

# Universidad Nacional de Lomas de Zamora – Facultad de Ingeniería
## UNLZ-AI-STUDIO
---
## Objetivo del repositorio

Uso educativo y de laboratorio
Que una universidad (como la UNLZ) pueda tener un servidor local donde docentes y estudiantes:

Hagan TPs y prototipos con LLMs/VLMs.

Prueben chatbots, asistentes de programación, análisis de imágenes, STT y TTS.

Programen contra una sola API sin importar qué modelo o backend hay por detrás.

🛠️ Simplificar el self-hosting en PCs reales

Detecta CPU/RAM/GPU y elige automáticamente un perfil (alto, medio, baja, cpu).

Configura parámetros de llama.cpp (ctx, n-gpu-layers, batch, etc.) según el equipo.

Evita que llama-server y lmdeploy peleen por la VRAM (auto-switch).

🧩 Ofrecer una puerta de entrada clara para hobbistas

Guía paso a paso para instalar modelos, dependencias y TTS en Windows.

Endpoints listos para consumir desde Python, PowerShell, ESP32, etc.

Ejemplos concretos de uso: texto→texto, imagen+texto, audio→texto→voz.

🖥️ Dar una interfaz “humana” para operar el servidor

GUI en Tkinter (estilo Ingeniería) y **Web UI (Next.js)** para:

Ver hardware y perfil activo.

Levantar/apagar servidores.

Activar/desactivar endpoints.

Ajustar presets y modelos personalizados.

---

## 🚀 Arquitectura Modular

UNLZ-AI-STUDIO cuenta con un sistema de **Módulos Pluggables** que permite extender la funcionalidad de la plataforma. Los módulos pueden instalarse desde la **GUI** (o Web UI) y se encuentran en la carpeta `modules/`.

### Módulos Incluidos

#### 1. Gaussian Splatting (Visualización 3D)
- **Ruta**: `modules/gaussian/`
- **Funcionalidad**: Permite crear escenas 3D a partir de imágenes utilizando **SharpSplat**.
- **Interfaz**: Visor 3D interactivo integrado en la aplicación.

#### 2. LLM Frontend (Chat & Manager)
- **Ruta**: `modules/llm_frontend/`
- **Funcionalidad**:
    - **Chat**: Interfaz gráfica para conversar con los modelos locales servidos por `gateway.py`.
    - **Gestor de Modelos**: Escanea tu carpeta de modelos (`C:\models`) y permite cambiar el modelo activo con un clic.
    - **Descargas**: Descarga modelos GGUF directamente desde Hugging Face usando RepoID.

#### 3. Inclu-IA (Subtitulado en Tiempo Real)
- **Ruta**: `modules/inclu_ia/`
- **Descripción**: Sistema de accesibilidad para aulas.
- **Funcionamiento**: Convierte tu PC en un servidor de subtítulos. Captura audio del micrófono, lo transcribe con IA (Faster-Whisper) y lo distribuye vía Web (Wi-Fi local) a los dispositivos de los alumnos.
- **Origen**: Adaptación del proyecto homónimo para Raspberry Pi.

---

## 🚀 API Gateway

El `gateway.py` sigue siendo el núcleo que gestiona los procesos pesados:
- **/llm** – texto↔texto con **llama.cpp**
- **/clm** – texto↔texto con **HF Transformers**
- **/vlm** – imagen+prompt con **LMDeploy**
- **/alm** – audio→texto→LLM→voz
- **/slm** – streaming audio/texto

---

## Ajustes automaticos de hardware

Desde `gateway.py` ahora se detectan CPU, RAM y GPU al iniciar, y se elige el preset mas apropiado para `llama-server`. Los perfiles principales son:
- `ultra` y `alto`: equipos con >=16 GB de VRAM y 64+ GB de RAM, usan Qwen3-Coder-30B con mas capas en GPU y contexto amplio.
- `balanceado` y `medio`: GPUs de 8 a 12 GB con 32-48 GB de RAM, priorizan Qwen3-Coder-14B y ajustan `--n-gpu-layers` para evitar OOM.
- `baja`: pensado para RTX 2060/3050 (6 GB) con 24-32 GB de RAM.
- `cpu`: modo de emergencia cuando no hay GPU o no hay espacio de VRAM; usa **Qwen2.5-Coder-7B**.

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

# LLM (Perfil Bajo/CPU) - Qwen2.5-Coder-7B-Instruct
hf download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF `
  --include "qwen2.5-coder-7b-instruct-q4_k_m.gguf" `
  --local-dir "C:\models\qwen2.5-coder-7b"

# Piper: modelo + config (ambos necesarios) – voz es_AR/daniela/high
hf download rhasspy/piper-voices `
  --include "es/es_AR/daniela/high/es_AR-daniela-high.onnx*" `
  --local-dir "C:\piper\voices\es_AR\daniela_high"
```

### 2) Dependencias

```powershell
# llama.cpp
winget install llama.cpp

# Python requirements
pip install -U fastapi uvicorn httpx psutil python-multipart faster-whisper
pip install -U lmdeploy
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -U huggingface_hub transformers accelerate pillow requests

# GUI & Modules
pip install customtkinter flask flask-socketio SpeechRecognition pyaudio
```

**Rutas esperadas por el gateway:**
- **GGUF 30B:** `C:\models\qwen3-coder-30b\Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf`
- **GGUF 7B:** `C:\models\qwen2.5-coder-7b\qwen2.5-coder-7b-instruct-q4_k_m.gguf`
- **VLM:**  `C:\models\qwen2.5-vl-7b-hf`
- **Piper:** `C:\piper\voices\es_AR\daniela_high\...`

---

## 🧪 Ejemplos de uso

### LLM (Perfil Bajo - Qwen2.5 7B)
```powershell
$body = @{
  model = "qwen2.5-coder-7b"
  messages = @(
    @{ role="system"; content="You are an expert in Finite element analysis." },
    @{ role="user";   content="Explica la diferencia entre analisis lineal y no lineal" }
  )
} | ConvertTo-Json -Depth 5

$web = Invoke-WebRequest -Uri "http://localhost:8000/llm" -Method Post -Body ([Text.Encoding]::UTF8.GetBytes($body)) -ContentType "application/json; charset=utf-8" 
$response = $web.Content | ConvertFrom-Json 
Write-Output $response.choices[0].message.content
```

### Web UI (Nueva Migración)
Para acceder a la nueva interfaz web:
```bash
cd web_ui
npm install
npm run dev
# Abrir http://localhost:3000
```
Si queres, puedo sumar telemetria basica de estado (CPU/RAM/GPU en tiempo real) o un boton de "Abrir docs" en la Home.
