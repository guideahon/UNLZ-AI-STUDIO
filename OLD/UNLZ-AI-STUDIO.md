UNLZ-AI-STUDIO

Expone:
- /llm  (texto↔texto con llama.cpp)
- /vlm  (imagen+prompt con Qwen2.5-VL vía LMDeploy)
- /alm  (audio→texto→LLM→texto→TTS)

Auto-switch: levanta/para llama-server o lmdeploy para no pelear VRAM.

Tutorial

1) Dependencias
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

# Modelos
hf download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF `
  --include "*Q5_K_M*.gguf" `
  --local-dir "C:\models\qwen3-coder-30b"

hf download Qwen/Qwen2.5-VL-7B-Instruct `
  --local-dir "C:\models\qwen2.5-vl-7b-hf"

# Piper: modelo + config (ambos necesarios)
hf download rhasspy/piper-voices `
  --include "es/es_AR/daniela/high/es_AR-daniela-high.onnx" `
  --include "es/es_AR/daniela/high/es_AR-daniela-high.onnx.json" `
  --local-dir "C:\piper\voices\es_AR\daniela_high"

Modelos a utilizar
- GGUF: C:\models\qwen3-coder-30b\Qwen3-Coder-30B-A3B-Instruct.Q5_K_M.gguf
- VLM:  C:\models\qwen2.5-vl-7b-hf
- Piper: C:\piper\voices

2) Ejecutar el gateway
python .\gateway.py
# (o) uvicorn gateway:app --host 0.0.0.0 --port 8000

/llm → proxyea a llama-server (si no está, lo levanta y apaga lmdeploy)
/vlm → proxyea a lmdeploy (si no está, lo levanta y apaga llama-server)
/alm → STT (CPU por defecto), luego /llm, y TTS con Piper (WAV base64 en la respuesta)

3) Ejemplos (PowerShell)

# LLM
$body = @{
  model = "qwen3-coder-30b"
  messages = @(
    @{ role="system"; content="You are a helpful coding assistant." },
    @{ role="user";   content="Explicá PID con pseudocódigo." }
  )
} | ConvertTo-Json -Depth 3
Invoke-RestMethod -Uri "http://localhost:8000/llm" -Method Post -ContentType "application/json" -Body $body

# VLM
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

# ALM
Invoke-RestMethod -Uri "http://localhost:8000/alm" -Method Post -Form @{
  file = Get-Item "C:\audios\pregunta.wav"
  system_prompt = "Sos un asistente de mecatrónica que responde de forma clara."
  tts = "true"
  target_lang = "es"
}

4) Consejos rápidos
- Si usás llama Vulkan (winget), quitá --flash-attn en LLAMA_ARGS.
- Si ves OOM, bajá --n-gpu-layers o --ctx-size o cuantizá a Q4_K_M.
- Para STT en GPU: USE_CUDA_FOR_STT=True (puede ir más lento si coincide con LLM).
- Podés abrir el puerto 8000 en el firewall para uso desde otras PCs.

Notas técnicas para gateway.py
- Para LMDeploy usá pesos HF (safetensors), no GGUF.
- VLM_MODEL_PATH = r"C:\models\qwen2.5-vl-7b-hf"
- LMDEPLOY_CMD = [sys.executable, "-m", "lmdeploy"]
- LLAMA_ARGS = [
    "-m", LLAMA_MODEL,
    "--host", LLAMA_HOST, "--port", str(LLAMA_PORT),
    "--ctx-size", "8192",
    "--n-gpu-layers", "35",
    "-t", "12"
    # "--flash-attn"  # solo si usás CUDA
  ]
- Si alguna versión de llama te tira error con --n-gpu-layers, probá --ngl.
