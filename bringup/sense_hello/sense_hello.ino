// sense_hello: bring-up check for the XIAO ESP32S3 Sense (room unit).
// Prints the Wi-Fi MAC (the address ESP-NOW uses), PSRAM size and msg_t size.
//
// Tools menu: Board "XIAO_ESP32S3" | USB CDC On Boot "Enabled" | PSRAM "OPI PSRAM"
// Serial Monitor: any baud (native USB). Prints every 2 s, so open it whenever.

#include <esp_mac.h>
#include <protocol.h>

void setup() {
  Serial.begin(115200);
}

void loop() {
  uint8_t mac[6];
  if (esp_read_mac(mac, ESP_MAC_WIFI_STA) == ESP_OK) {
    Serial.printf("MAC:      %02X:%02X:%02X:%02X:%02X:%02X\n",
                  mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    Serial.printf("For code: {0x%02X, 0x%02X, 0x%02X, 0x%02X, 0x%02X, 0x%02X}\n",
                  mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  } else {
    Serial.println("MAC:      read failed");
  }

  Serial.printf("PSRAM:    %lu bytes (expect about 8 MB; 0 means PSRAM is off in Tools)\n",
                (unsigned long)ESP.getPsramSize());
  Serial.printf("msg_t:    %u bytes (must be 10 on every board)\n\n",
                (unsigned)sizeof(msg_t));
  delay(2000);
}
