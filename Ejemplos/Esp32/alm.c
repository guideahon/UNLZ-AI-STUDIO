#include <WiFi.h>
#include <HTTPClient.h>
#include "SPIFFS.h"

const char* SSID = "TU_WIFI";
const char* PASS = "TU_PASS";
const char* URL  = "http://<PC_IP>:8000/alm";

// Asumimos que ya subiste /input.wav a SPIFFS (Sketch Data Upload)
bool postMultipart(const char* url, const char* path) {
  File f = SPIFFS.open(path, "r");
  if (!f) { Serial.println("No file"); return false; }

  WiFiClient client;
  HTTPClient http;
  http.begin(client, url);

  String boundary = "----ESP32FormBoundary7MA4YWxkTrZu0gW";
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

  String head = "--" + boundary + "\r\n";
  head += "Content-Disposition: form-data; name=\"file\"; filename=\"input.wav\"\r\n";
  head += "Content-Type: audio/wav\r\n\r\n";

  String tail = "\r\n--" + boundary + "\r\n";
  tail += "Content-Disposition: form-data; name=\"system_prompt\"\r\n\r\nYou are a helpful assistant.\r\n";
  tail += "--" + boundary + "\r\n";
  tail += "Content-Disposition: form-data; name=\"tts\"\r\n\r\ntrue\r\n";
  tail += "--" + boundary + "\r\n";
  tail += "Content-Disposition: form-data; name=\"target_lang\"\r\n\r\nes\r\n";
  tail += "--" + boundary + "--\r\n";

  int len = head.length() + f.size() + tail.length();
  int code = http.sendRequest("POST", (uint8_t*)NULL, 0, &len); // prepara headers con Content-Length
  if (code != HTTP_SEND_HEADER_FINISHED) { Serial.println("Hdr fail"); http.end(); f.close(); return false; }

  // enviar head + archivo + tail
  WiFiClient* stream = http.getStreamPtr();
  stream->print(head);
  uint8_t buf[1024];
  while (f.available()) {
    size_t n = f.read(buf, sizeof(buf));
    stream->write(buf, n);
  }
  stream->print(tail);

  int httpCode = http.GET_SIZE(); // fuerza lectura respuesta
  httpCode = http.GET();          // o http.collectHeaders(...)
  Serial.printf("HTTP: %d\n", httpCode);
  String resp = http.getString();
  Serial.println(resp);
  http.end(); f.close();
  return httpCode > 0;
}

void setup(){
  Serial.begin(115200);
  WiFi.begin(SSID, PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); }
  SPIFFS.begin(true);
  postMultipart(URL, "/input.wav");
}
void loop(){}
