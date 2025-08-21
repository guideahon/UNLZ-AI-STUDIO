import requests

URL = "http://localhost:8000/clm"

payload = {
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "temperature": 0.2,
    "max_tokens": 256,
    "top_p": 0.9,
    "messages": [
        {"role": "system", "content": "Sos un experto en visión y razonamiento."},
        {"role": "user", "content": "Dame 3 ideas para un proyecto con ESP32 y cámara."}
    ]
}

r = requests.post(URL, json=payload, timeout=60)
r.raise_for_status()
print(r.json())
