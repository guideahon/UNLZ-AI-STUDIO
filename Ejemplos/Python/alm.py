# pip install requests
import requests

URL = "http://localhost:8000/alm"

with open("input.wav", "rb") as f:
    files = {"file": ("input.wav", f, "audio/wav")}
    data = {
        "system_prompt": "You are a helpful assistant.",
        "tts": "true",
        "target_lang": "es"
    }
    r = requests.post(URL, files=files, data=data, timeout=120)

r.raise_for_status()
resp = r.json()
print("STT:", resp["stt_text"])
print("LLM:", resp["llm_text"])
if resp.get("tts_audio"):
    print("Audio base64 (WAV):", resp["tts_audio"][:60], "...")
