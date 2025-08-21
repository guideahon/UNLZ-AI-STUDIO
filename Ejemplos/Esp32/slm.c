#include <WiFi.h>
#include "SPIFFS.h"

const char* SSID = "TU_WIFI";
const char* PASS = "TU_PASS";
const char* HOST = "<PC_IP>";
const int   PORT = 8000;
const char* PATH = "/slm";

String boundary = "----ESP32FormBoundary7MA4YWxkTrZu0gW";

void sendMultipart(WiFiClient& client, File& f){
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

  size_t contentLength = head.length() + f.size() + tail.length();

  // Request
  client.printf("POST %s HTTP/1.1\r\n", PATH);
  client.printf("Host: %s:%d\r\n", HOST, PORT);
  client.println("Connection: close");
  client.println("Accept: text/event-stream");
  client.print("Content-Type: multipart/form-data; boundary=");
  client.println(boundary);
  client.print("Content-Length: "); client.println(contentLength);
  client.println(); // end headers

  // body
  client.print(head);
  uint8_t buf[1024];
  while (f.available()){
    int n = f.read(buf, sizeof(buf));
    client.write(buf, n);
  }
  client.print(tail);
}

void setup(){
  Serial.begin(115200);
  WiFi.begin(SSID, PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  SPIFFS.begin(true);

  File f = SPIFFS.open("/input.wav", "r");
  if(!f){ Serial.println("No /input.wav"); return; }

  WiFiClient client;
  if (!client.connect(HOST, PORT)) { Serial.println("Conn fail"); return; }

  sendMultipart(client, f);
  f.close();

  // Leer respuesta: cabezal HTTP -> luego stream SSE
  // Consumir headers HTTP
  while (client.connected()){
    String line = client.readStringUntil('\n');
    if (line == "\r") break; // fin headers
  }

  // Parse simple SSE
  String event = "", data = "";
  while (client.connected() || client.available()){
    String line = client.readStringUntil('\n');
    if (line.length() == 0) { delay(1); continue; }

    if (line.startsWith(":")) {
      // heartbeat
      continue;
    } else if (line.startsWith("event:")) {
      event = line.substring(6); event.trim();
    } else if (line.startsWith("data:")) {
      data += line.substring(5); data.trim();
    } else if (line == "\r") {
      // Fin de evento
      if (event.length()){
        Serial.print("[SSE] event="); Serial.println(event);
        Serial.print("[SSE] data=");  Serial.println(data.substring(0,80)); // preview
        // Si event=="audio", data es JSON con {seq,last,mime,data}
        // Podés acumular 'data' base64 hasta last==true para reconstruir el WAV completo.
      }
      event = ""; data = "";
    }
  }

  client.stop();
}

void loop(){}
