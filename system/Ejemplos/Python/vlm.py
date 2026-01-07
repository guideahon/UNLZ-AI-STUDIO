# pip install requests pillow
import base64, requests
from PIL import Image
from io import BytesIO

URL = "http://localhost:8000/vlm"

# Cargar una imagen desde disco y convertir a data URL
img = Image.open("foto.jpg").convert("RGB")
buf = BytesIO(); img.save(buf, format="JPEG", quality=90)
b64 = base64.b64encode(buf.getvalue()).decode("ascii")
data_url = f"data:image/jpeg;base64,{b64}"

payload = {
    "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "¿Qué ves en esta imagen?"},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]}
    ],
    "temperature": 0.2,
    "max_tokens": 256,
    "top_p": 0.9
}

r = requests.post(URL, json=payload, timeout=120)
r.raise_for_status()
print(r.json())
