/**
 * @file listener_esp32s3.ino
 * @brief yamDetector ESP32-S3 Firmware
 */

#include <Arduino.h>
#include <driver/i2s.h>
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/experimental/microfrontend/lib/frontend.h>
#include <tensorflow/lite/experimental/microfrontend/lib/frontend_config.h>

#include " model.h\

#define PDM_CLK_PIN 1
#define PDM_DATA0_PIN 2
#define PDM_DATA1_PIN 4

#define SAMPLE_RATE 16000
#define WINDOW_SAMPLES 16000

const char* kClassLabels[kYamOutputClasses] = {
 \siren\, \car_horn\, \alarm\, \knock\, \background\
};

int16_t audioBuffer[WINDOW_SAMPLES];
int bufferCursor = 0;

tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
constexpr int kTensorArenaSize = 40 * 1024;
alignas(16) uint8_t tensorArena[kTensorArenaSize];

FrontendConfig feConfig;
FrontendState feState;

void setup_pdm_i2s() {
 i2s_config_t i2s_config = {
 .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM),
 .sample_rate = SAMPLE_RATE,
 .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
 .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
 .communication_format = I2S_COMM_FORMAT_STANDI_I2S,
 .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
 .dma_buf_count = 8,
 .dma_buf_len = 512,
 .use_apll = false,
 .tx_desc_auto_clear = false,
 .fixed_mclk = 0
 };
 i2s_pin_config_t pin_config = {
 .bck_io_num = I2S_PIN_NO_CHANGE,
 .ws_io_num = PDM_CLK_PIN,
 .data_out_num = I2S_PIN_NO_CHANGE,
 .data_in_num = PDM_DATA0_PIN
 };
 i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
 i2s_set_pin(I2S_NUM_0, &pin_config);
 i2s_set_sample_rates(I2S_NUM_0, SAMPLE_RATE);
}

void setup() {
 Serial.begin(115200);
 delay(1000);
 setup_pdm_i2s();
 feConfig.window.size_ms = 30; feConfig.window.step_ms = 20; 
 feConfig.filterbank.num_channels = kYamInputChannels; 
 feConfig.filterbank.lower_band_limit = 125.0f; 
 feConfig.filterbank.upper_band_limit = 7500.0f; 
 feConfig.noise_reduction.enable_pcan = 1;
 feConfig.log_scale.enable_log = 1;
 FrontendPopulateState(&feConfig, &feState, SAMPLE_RATE);
 tflModel = tflite::GetModel(g_yam_model_tflite);
 tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, kTensorArenaSize);
 tflInterpreter->AllocateTensors();
 Serial.println(\Ready!\);
}

void runInference(int16_t* windowSamples) {
 TfLiteTensor* inputTensor = tflInterpreter->input(0);
 auto candidate = FrontendProcessSamples(&feState, windowSamples, WINDOW_SAMPLES, NULL);
 int frameIndex = 0;
 while (candidate.values != NULL && frameIndex < kYamInputFrames) {
 for (int c = 0; c < kYamInputChannels; c++) {
 float val = (float)candidate.values[c] / 255.0f;
 int8_t quantized = (int8_t)round(val / kYamInputScale + kYamInputZeroPoint);
 inputTensor->data.int8[frameIndex * kYamInputChannels + c] = quantized;
 }
 frameIndex++;
 candidate = FrontendProcessSamples(&feState, NULL, 0, NULL);
 }
 tflInterpreter->Invoke();
 TfLiteTensor* outputTensor = tflInterpreter->output(0);
 int bestClass = 0; int8_t bestScore = -128;
 for (int i = 0; i < kYamOutputClasses; i++) {
 if (outputTensor->data.int8[i] > bestScore) { bestScore = outputTensor->data.int8[i]; bestClass = i; }
 }
 Serial.print(\Event: \); Serial.println(kClassLabels[bestClass]);
}

void loop() {
 int16_t dmaBuffer[256];
 size_t bytesRead = 0;
 i2s_read(I2S_NUM_0, &dmaBuffer, sizeof(dmaBuffer), &bytesRead, portMAX_DELAY);
 if (bytesRead > 0) {
 for (int i = 0; i < bytesRead/sizeof(int16_t); i++) {
 audioBuffer[bufferCursor++] = dmaBuffer[i];
 if (bufferCursor >= WINDOW_SAMPLES) { runInference(audioBuffer); bufferCursor = 0; }
 }
 }
}
