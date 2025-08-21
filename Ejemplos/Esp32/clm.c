#include <WiFi.h>
#include <HTTPClient.h>

const char* SSID = "TU_WIFI";
const char* PASS = "TU_PASS";
const char* URL  = "http://<PC_IP>:8000/clm";

void setup() {
  Serial.begin(115200);
  WiFi.begin(SSID, PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); }
  HTTPClient http; http.begin(URL);
  http.addHeader("Content-Type", "application/json");

  String payload = R"({
    "model":"Qwen/Qwen2.5-VL-7B-Instruct",
    "temperature":0.2,
    "max_tokens":128,
    "top_p":0.9,
    "messages":[
      {"role":"system","content":"Sos experto en visión."},
      {"role":"user","content":"¿Qué puedo hacer con una ESP32-CAM?"}
    ]
  })";

  int code = http.POST(payload);
  Serial.println(code);
  Serial.println(http.getString());
  http.end();
}
void loop(){}
