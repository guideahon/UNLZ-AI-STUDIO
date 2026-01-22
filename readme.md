![Logo Institucional](https://github.com/JonatanBogadoUNLZ/PPS-Jonatan-Bogado/blob/9952aac097aca83a1aadfc26679fc7ec57369d82/LOGO%20AZUL%20HORIZONTAL%20-%20fondo%20transparente.png)

# Universidad Nacional de Lomas de Zamora - Facultad de Ingeniería
## UNLZ AI Studio

Plataforma local para investigación, docencia y prototipado con modelos de IA. Permite operar LLM/VLM/STT/TTS en un solo servidor, con GUI de escritorio y Web UI, y perfiles automáticos según hardware.

---

## Objetivo
Que la universidad pueda tener un servidor local donde docentes y estudiantes:
- Armen TPs y prototipos con LLMs/VLMs.
- Prueben chatbots, asistentes de programación, análisis de imágenes, STT y TTS.
- Programen contra una sola API sin importar el backend.
- Tengan una interfaz clara para operar servicios sin tocar consola.

---

## Características principales
- Detección de CPU/RAM/GPU y presets automáticos de ejecución.
- Control de servicios desde GUI (Tkinter) y Web UI (Next.js).
- Arquitectura modular con instalación desde la interfaz.
- Web Bridge para exponer la API del runtime local.
- Compatibilidad con modelos GGUF y backends de visión/audio.

---

## Arquitectura (alto nivel)
- `system/gateway.py`: orquestador principal para procesos de IA.
- `system/process_manager.py`: arranque/parada de servicios y control de recursos.
- `system/studio_gui.py`: GUI de escritorio (Tkinter).
- `system/web_bridge.py`: puente API para la Web UI.
- `system/web_ui/`: interfaz Next.js.
- `system/modules/`: módulos plug-and-play.

---

## Estructura del repositorio
- `system/`: runtime, gateway, GUI, assets, módulos.
- `system/web_ui/`: aplicación web (Next.js).
- `system/assets/`: logos, tema, idiomas.
- `system/data/`: configuraciones y datos de usuario.
- `docs/`: documentación adicional.

---

## Requisitos
- Windows 10/11.
- Python 3.10+.
- Node.js 18+ (para la Web UI).
- GPU NVIDIA recomendada (CUDA) para rendimiento.
- `llama.cpp` instalado si vas a usar backend GGUF con `llama-server`.

---

## Instalación rápida

### 1) Modelos
```powershell
# LLM (GGUF) - Qwen3-Coder-30B-A3B-Instruct (Q5_K_M)
hf download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF `
  --include "*Q5_K_M*.gguf" `
  --local-dir "C:\models\qwen3-coder-30b"

# VLM (safetensors para LMDeploy) - Qwen2.5-VL-7B-Instruct
hf download Qwen/Qwen2.5-VL-7B-Instruct `
  --local-dir "C:\models\qwen2.5-vl-7b-hf"

# LLM (perfil bajo/CPU) - Qwen2.5-Coder-7B-Instruct
hf download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF `
  --include "qwen2.5-coder-7b-instruct-q4_k_m.gguf" `
  --local-dir "C:\models\qwen2.5-coder-7b"

# Piper (TTS) - voz es_AR/daniela/high
hf download rhasspy/piper-voices `
  --include "es/es_AR/daniela/high/es_AR-daniela-high.onnx*" `
  --local-dir "C:\piper\voices\es_AR\daniela_high"
```

### 2) Dependencias
```powershell
# llama.cpp
winget install llama.cpp

# Python runtime
pip install -U fastapi uvicorn httpx psutil python-multipart faster-whisper
pip install -U lmdeploy
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -U huggingface_hub transformers accelerate pillow requests

# GUI y módulos
pip install customtkinter flask flask-socketio SpeechRecognition pyaudio
```

Rutas esperadas por el runtime:
- GGUF 30B: `C:\models\qwen3-coder-30b\Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf`
- GGUF 7B: `C:\models\qwen2.5-coder-7b\qwen2.5-coder-7b-instruct-q4_k_m.gguf`
- VLM: `C:\models\qwen2.5-vl-7b-hf`
- Piper: `C:\piper\voices\es_AR\daniela_high\...`

---

## Ejecución

### GUI de escritorio (Tkinter)
```powershell
python system\studio_gui.py
```
También podés usar los accesos directos:
- `UNLZ AI Studio.lnk`

### Web UI (Next.js)
```powershell
cd system\web_ui
npm install
npm run dev
```
Luego abrir: `http://localhost:3000`

El bridge API se ejecuta con:
```powershell
python system\web_bridge.py
```
Acceso rápido:
- `system\run_web_ui.bat` (Web UI + Web Bridge)
- `UNLZ AI Studio Web.lnk`

---

## Endpoints (gateway)
Principales rutas expuestas por el runtime:
- `/llm`: texto -> texto (llama.cpp)
- `/clm`: texto -> texto (HF Transformers)
- `/vlm`: imagen + prompt
- `/alm`: audio -> texto -> LLM -> voz
- `/slm`: streaming audio/texto

---

## Módulos (detalle y autoría)
Los módulos se instalan desde la GUI o la Web UI. Código en `system/modules/`.

### Monitor (Endpoints de IA)
- Autor original: UNLZ AI Studio.
- Descripción: panel para ver hardware y gestionar servicios LLM/VLM/Audio.
- Funcionamiento: consulta el estado vía Web Bridge y dispara acciones `/services/*`.
- Uso en la app: Web UI > Monitor (iniciar/detener, instalar/desinstalar, seleccionar modelo).

### ML-SHARP
- Autor original: Apple (`https://github.com/apple/ml-sharp`).
- Descripción: síntesis monocular rápida con splats gaussianos.
- Funcionamiento: ejecuta `sharp predict` con selección de dispositivo y render opcional, y genera escenas 3D con visor web.
- Uso en la app: instalar dependencias, elegir input/salida, correr inferencia, gestionar escenas (library) y abrir visor.
- Extra (SharpSplat): descarga automática del modelo `sharp_2572gikvuh.pt` y setup del viewer (`gaussians/index.html`).

### LLM Frontend (Chat & Manager)
- Autor original: UNLZ AI Studio.
- Descripción: chat local con modelos GGUF + gestor de modelos.
- Funcionamiento: inicia el servidor local `llm_chat` y consulta `/v1/chat/completions`.
- Uso en la app: elegir modelo, iniciar servidor, chatear y descargar modelos.

### Fine-tune GLM-4.7
- Autor original: Unsloth (pipeline) + modelo base `unsloth/GLM-4.7-Flash`.
- Descripción: ajuste fino local con LoRA y export a GGUF.
- Funcionamiento: script `system/data/finetune_glm/finetune_glm_4_7_flash.py`.
- Uso en la app: cargar dataset, ejecutar entrenamiento y abrir carpeta de salida.

### Inclu-IA
- Autor original: UNLZ AI Studio (adaptación local del proyecto Inclu-IA).
- Descripción: subtitulado y accesibilidad en tiempo real para aulas.
- Funcionamiento: servidor local que captura micrófono, transcribe y publica vía web.
- Uso en la app: iniciar servidor y abrir la interfaz web para alumnos.

### Asistente de Investigación
- Autor original: UNLZ AI Studio.
- Descripción: gestor de bibliografía local con resúmenes y citas.
- Funcionamiento: indexa PDFs, genera resúmenes extractivos y búsqueda TF-IDF.
- Uso en la app: importar PDFs, construir índice, consultar y copiar citas.

### SpotEdit
- Autor original: Biangbiang0321 (`https://github.com/Biangbiang0321/SpotEdit`).
- Descripción: edición local por regiones con Diffusion Transformers.
- Funcionamiento: descarga backend, prepara entorno y ejecuta edición por máscara.
- Uso en la app: seleccionar imagen, marcar región y ejecutar edición.

### Flux 2 Klein
- Autor original: Black Forest Labs (`https://huggingface.co/black-forest-labs/FLUX.2-klein-4B`).
- Descripción: generación de imágenes texto-a-imagen.
- Funcionamiento: descarga modelo, prepara entorno y ejecuta prompt.
- Uso en la app: definir prompt, parámetros y generar.

### HunyuanWorld-Mirror
- Autor original: Tencent Hunyuan (`https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror`).
- Descripción: reconstrucción 3D con el stack HunyuanWorld.
- Funcionamiento: clona backend y ejecuta demo/inferencia.
- Uso en la app: instalar backend, descargar pesos y correr generación.

### CyberScraper 2077
- Autor original: itsOwen (`https://github.com/itsOwen/CyberScraper-2077`).
- Descripción: scraping con Streamlit y LLMs.
- Funcionamiento: instala backend, dependencias y levanta la UI web.
- Uso en la app: instalar backend, iniciar servidor y abrir navegador.

### HY-Motion 1.0
- Autor original: Tencent Hunyuan (`https://github.com/Tencent-Hunyuan/HY-Motion-1.0`).
- Descripción: generación y edición de movimiento.
- Funcionamiento: clona repo, instala deps y ejecuta pipeline.
- Uso en la app: instalar, descargar weights y correr generación.

### NeuTTS
- Autor original: Neuphonic (`https://github.com/neuphonic/neutts`).
- Descripción: texto a voz con variantes Air/Nano.
- Funcionamiento: instala backend y dependencias, ejecuta generación local.
- Uso en la app: seleccionar variante, generar audio y abrir salida.

### ProEdit
- Autor original: iSEE-Laboratory (`https://github.com/iSEE-Laboratory/ProEdit`).
- Descripción: edición avanzada basada en inversión para imagen y video.
- Funcionamiento: backend con modelos y scripts de edición guiada.
- Uso en la app: instalar backend, preparar input y ejecutar edición.

### Generación de modelo 3D
- Autor original: Stepfun AI (`https://github.com/stepfun-ai/Step1X-3D`), Tencent (`https://github.com/Tencent/Hunyuan3D-2`), Meta/FAIR (`https://github.com/facebookresearch/sam-3d-objects`).
- Descripción: crea modelos 3D desde una o más imágenes o video.
- Funcionamiento: permite instalar backends, bajar pesos y ejecutar por backend.
- Uso en la app: elegir backend, input y ejecutar generación.

---

## Configuración
- Carpeta de modelos GGUF: variable `LLAMA_MODEL_DIR` o desde la UI.
- Idioma y tema: `system/data/app_settings.json`.
- Logs: `system/logs/` y `system/web_ui/.next/`.

---

## Troubleshooting rápido
- No aparecen modelos: verificá rutas y extensiones `.gguf`.
- Fallos de GPU: reduce modelo o perfil, revisá VRAM disponible.
- Web UI no responde: asegurate de tener `web_bridge.py` corriendo.
- Errores de arranque: revisar `system/logs/startup.log`.

---

## Notas
Este proyecto está orientado a uso educativo y de laboratorio. Si querés sumar módulos o integraciones, revisá `system/modules/` y los ejemplos en `system/Ejemplos/`.
