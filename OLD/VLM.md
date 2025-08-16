VLM: Imagen → Texto / Q&A sobre imagen

Modelo sugerido: Qwen/Qwen2.5-VL-7B-Instruct (entra en 24 GB con cuantización/optimización moderada y anda bien para clase).

Instalar LMDeploy (una vez)
pip install -U lmdeploy

Descargar el modelo (Hugging Face)
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "C:\models\qwen2.5-vl-7b"

Levantar servidor VLM
$VLM="C:\models\qwen2.5-vl-7b"
lmdeploy serve api_server `
  --model-path $VLM `
  --server-port 9090


Este server expone un API estilo OpenAI para chat con imágenes (pueden mandarte una image_url o subir binario según el cliente).
Si preferís unificar, más abajo te dejo un “gateway” FastAPI que te deja todo bajo rutas tipo OpenAI.