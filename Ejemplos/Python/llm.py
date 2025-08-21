# pip install requests sseclient-py
import requests

URL = "http://localhost:8000/llm"

payload = {
    "model": "qwen3-coder-30b",
    "temperature": 0.2,
    "max_tokens": 256,
    "top_p": 0.9,
    "messages": [
        {"role": "system", "content": "Sos un asistente útil y conciso."},
        {"role": "user", "content": "Explicame qué es un debounce en electrónica."}
    ],
    # "stream": True  # si tu backend lo soporta y querés proxyear SSE del llama-server
}

r = requests.post(URL, json=payload, timeout=60)
r.raise_for_status()
print(r.json())
