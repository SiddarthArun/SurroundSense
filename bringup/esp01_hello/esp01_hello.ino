// esp01_hello: bring-up check for the ESP-01 (wristband), flashed through the DevKit V1.
// Prints the MAC (the address ESP-NOW uses), the supply voltage and msg_t size.
//
// Tools menu: Board "Generic ESP8266 Module" | Upload Speed 115200
//             Reset Method "no dtr (aka ck)" | Flash Size "1MB" (Flash Mode DOUT if uploads fail)
// Port: the DevKit V1's COM port, with the V1's EN jumpered to GND.
// Upload: no reset button on the band, so GPIO0 to GND *before* power arrives, then Upload.
//         Free GPIO0, power-cycle again, Serial Monitor at 115200.

#include <ESP8266WiFi.h>
#include <protocol.h>

ADC_MODE(ADC_VCC);  // lets ESP.getVcc() read the supply; reused later for the LOW BATT warning

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
}

void loop() {
  uint8_t mac[6];
  WiFi.macAddress(mac);
  Serial.printf("MAC:      %02X:%02X:%02X:%02X:%02X:%02X\n",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  Serial.printf("For code: {0x%02X, 0x%02X, 0x%02X, 0x%02X, 0x%02X, 0x%02X}\n",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

  Serial.printf("Supply:   %u mV (about 3300 on the programmer; replace AAs below 2900)\n",
                (unsigned)ESP.getVcc());
  Serial.printf("msg_t:    %u bytes (must be 10 on every board)\n\n",
                (unsigned)sizeof(msg_t));
  delay(2000);
}
