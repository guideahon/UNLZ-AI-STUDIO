# pip install httpx
import httpx
import json
import asyncio

async def main():
    URL = "http://localhost:8000/slm"
    data = {
        "system_prompt": "You are a helpful assistant.",
        "tts": "true",
        "target_lang": "es",
    }
    files = {"file": ("input.wav", open("input.wav", "rb"), "audio/wav")}

    async with httpx.AsyncClient(timeout=None) as client:
        # NOTA: httpx no tiene parser SSE; leemos el stream y separamos eventos manualmente.
        async with client.stream("POST", URL, data=data, files=files) as r:
            r.raise_for_status()
            audio_chunks = []
            async for b in r.aiter_bytes():
                for block in b.decode("utf-8", errors="ignore").split("\n\n"):
                    if not block.strip(): 
                        continue
                    if block.startswith(":"):  # heartbeat
                        continue
                    lines = block.split("\n")
                    event, dat = "message", ""
                    for ln in lines:
                        if ln.startswith("event:"):
                            event = ln[6:].strip()
                        elif ln.startswith("data:"):
                            dat += ln[5:].strip()
                    if not dat:
                        continue
                    try:
                        obj = json.loads(dat)
                    except:
                        continue

                    if event == "text":
                        print("STT:", obj.get("stt_text"))
                        print("LLM:", obj.get("llm_text"))
                    elif event == "audio":
                        audio_chunks.append(obj.get("data",""))
                        if obj.get("last"):
                            b64 = "".join(audio_chunks)
                            print("WAV base64 len =", len(b64))
                            audio_chunks = []
                    elif event == "error":
                        print("ERROR:", obj.get("message"))
                    elif event == "done":
                        print("DONE:", obj)
                        return

asyncio.run(main())
