1) Installar llama.cpp con winget

winget install llama.cpp

2) Descargar el GGUF buscado, en este caso "Qwen3-Coder-30B-A3B-Instruct" a Q4_K_M

pip install --upgrade huggingface_hub

hf download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF --include "*Q5_K_M*.gguf" --local-dir "C:\models\qwen3-coder-30b"

3) Abrir un server OpenAI-API compatible

$MODEL = "C:\models\qwen3-coder-30b\Qwen3-Coder-30B-A3B-Instruct.Q5_K_M.gguf"

# Vulkan build: NO uses --flash-attn (es para CUDA). Si te tira error con --ngl, cambiá a --n-gpu-layers
llama-server `
  -m $MODEL `
  --host 0.0.0.0 --port 8080 `
  --ctx-size 8192 `
  --n-gpu-layers 35 `
  -t 12

4) Testear

$body = @{
  model = "qwen3-coder-30b"
  temperature = 0.2
  messages = @(
    @{ role="system"; content="You are a helpful coding assistant." },
    @{ role="user"; content="Write a Python function that computes the FFT of a signal and plots the magnitude spectrum." }
  )
} | ConvertTo-Json -Depth 3

Invoke-RestMethod -Uri "http://localhost:8080/v1/chat/completions" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
