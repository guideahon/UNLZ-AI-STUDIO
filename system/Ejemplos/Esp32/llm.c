#include <WiFi.h>
#include <HTTPClient.h>

const char* SSID = "TU_WIFI";
const char* PASS = "TU_PASS";
const char* URL  = "http://<PC_IP>:8000/llm";

void setup() {
  Serial.begin(115200);
  WiFi.begin(SSID, PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nWiFi OK");

  HTTPClient http;
  http.begin(URL);
  http.addHeader("Content-Type", "application/json");

  String payload = R"({
    "model":"qwen3-coder-30b",
    "temperature":0.2,
    "max_tokens":128,
    "top_p":0.9,
    "messages":[
      {"role":"system","content":"Sos un asistente útil."},
      {"role":"user","content":"Dame 2 ideas de proyectos con ESP32."}
    ]
  })";

  int code = http.POST(payload);
  Serial.println(code);
  String resp = http.getString();
  Serial.println(resp);
  http.end();
}

void loop(){}
