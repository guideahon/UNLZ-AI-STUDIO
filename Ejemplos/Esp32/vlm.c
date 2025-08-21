#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include "Base64.h"

// Configura pines de tu módulo (ej. AI-Thinker)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

const char* SSID = "TU_WIFI";
const char* PASS = "TU_PASS";
const char* URL  = "http://<PC_IP>:8000/vlm";

void setup() {
  Serial.begin(115200);
  WiFi.begin(SSID, PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("WiFi OK");

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_QVGA; // chico para no saturar
  config.jpeg_quality = 12;
  config.fb_count     = 1;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Cam init failed");
    return;
  }

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { Serial.println("fb null"); return; }

  // Base64 del JPEG
  size_t enc_len = base64_enc_len(fb->len);
  char* out = (char*)malloc(enc_len + 1);
  if (!out) { esp_camera_fb_return(fb); Serial.println("malloc fail"); return; }
  base64_encode(out, (char*)fb->buf, fb->len);
  esp_camera_fb_return(fb);

  // Construir JSON con data URL
  String dataUrl = "data:image/jpeg;base64,";
  dataUrl += out;
  free(out);

  String json = String("{\"messages\":[{\"role\":\"user\",\"content\":[") +
    "{\"type\":\"text\",\"text\":\"Describí la imagen brevemente.\"}," +
    "{\"type\":\"image_url\",\"image_url\":{\"url\":\"" + dataUrl + "\"}}" +
  "]}],\"temperature\":0.2,\"max_tokens\":128,\"top_p\":0.9}";

  HTTPClient http; http.begin(URL);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(json);
  Serial.println(code);
  Serial.println(http.getString());
  http.end();
}

void loop(){}
