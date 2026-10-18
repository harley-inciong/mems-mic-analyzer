/**
 * Real-Time MEMS Microphone Analyzer - ESP32-C3 Firmware
 * 
 * Hardware:
 *  - ESP32-C3 Development Board
 *  - INMP441 MEMS Microphone
 * 
 * Connections:
 *  - INMP441 VDD   -> ESP32-C3 3.3V
 *  - INMP441 GND   -> ESP32-C3 GND
 *  - INMP441 L/R   -> ESP32-C3 GND (Left Channel)
 *  - INMP441 SCK   -> ESP32-C3 GPIO 2 (Serial Clock / BCLK)
 *  - INMP441 WS    -> ESP32-C3 GPIO 3 (Word Select / LRCK)
 *  - INMP441 SD    -> ESP32-C3 GPIO 4 (Serial Data Out)
 * 
 * This firmware reads audio samples using DMA, packages them into a framed
 * packet protocol to avoid synchronization issues on the host, and streams
 * raw PCM over USB serial. It also handles commands from the host.
 */

#include <Arduino.h>
#include <driver/i2s.h>

// Default I2S Pin Definitions for ESP32-C3
#define DEFAULT_I2S_SCK 2
#define DEFAULT_I2S_WS  3
#define DEFAULT_I2S_SD  4
#define I2S_PORT I2S_NUM_0

// Packet Framing Constants
#define SYNC_BYTE_1 0x5A
#define SYNC_BYTE_2 0xA5
#define SAMPLES_PER_PACKET 256
#define PAYLOAD_SIZE (SAMPLES_PER_PACKET * sizeof(int16_t)) // 512 bytes

// Global variables
int i2sSckPin = DEFAULT_I2S_SCK;
int i2sWsPin  = DEFAULT_I2S_WS;
int i2sSdPin  = DEFAULT_I2S_SD;
uint32_t currentSampleRate = 44100;
float softwareGain = 2.0f;
bool isStreaming = true;
uint8_t sequenceCounter = 0;

// DMA read buffers (stereo 32-bit slots: Left + Right per frame)
int32_t i2sRawBuffer[SAMPLES_PER_PACKET * 2];
int16_t audioPayload[SAMPLES_PER_PACKET];

// Struct representing the packet header
struct __attribute__((__packed__)) PacketHeader {
  uint8_t sync1;
  uint8_t sync2;
  uint8_t seq;
  uint16_t length;
  uint32_t timestamp;
  uint32_t sampleRate;
};

/**
 * Initializes or re-initializes the I2S peripheral with the specified sample rate.
 */
bool initI2S(uint32_t sampleRate) {
  // If driver is already installed, uninstall it first
  i2s_driver_uninstall(I2S_PORT);

  // Configure I2S peripheral settings
  // INMP441 outputs 24-bit data inside a 32-bit slot.
  // Reading RIGHT_LEFT stereo slots avoids ESP32-C3 mono slot alignment bugs
  // and automatically supports L/R wired to either GND (Left) or 3.3V (Right).
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = sampleRate,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = SAMPLES_PER_PACKET,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  // Configure pins
  i2s_pin_config_t pin_config = {
    .bck_io_num = i2sSckPin,
    .ws_io_num = i2sWsPin,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = i2sSdPin
  };

  // Install driver
  esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    return false;
  }

  // Assign pins
  err = i2s_set_pin(I2S_PORT, &pin_config);
  if (err != ESP_OK) {
    return false;
  }

  // Clear DMA buffer
  i2s_zero_dma_buffer(I2S_PORT);
  
  currentSampleRate = sampleRate;
  return true;
}

/**
 * Reads a line from the Serial port and executes commands.
 * Commands format:
 *   - "R:<rate>\n"         : Change sample rate (8000, 16000, 22050, 44100)
 *   - "G:<gain>\n"         : Change software digital gain multiplier (e.g. 1.0, 2.5)
 *   - "S:<1|0>\n"          : Start (1) or Stop (0) streaming
 *   - "P:<sck>,<ws>,<sd>\n": Reconfigure I2S GPIO pins dynamically
 */
void checkSerialCommands() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd.startsWith("R:")) {
      long rate = cmd.substring(2).toInt();
      if (rate == 8000 || rate == 16000 || rate == 22050 || rate == 44100) {
        if (initI2S(rate)) {
          Serial.printf("LOG:Sample rate changed to %ld\n", rate);
        } else {
          Serial.println("ERR:Failed to set sample rate");
        }
      } else {
        Serial.printf("ERR:Unsupported sample rate %ld\n", rate);
      }
    } 
    else if (cmd.startsWith("G:")) {
      float gain = cmd.substring(2).toFloat();
      if (gain >= 0.0f && gain <= 100.0f) {
        softwareGain = gain;
        Serial.printf("LOG:Gain changed to %.2f\n", gain);
      } else {
        Serial.println("ERR:Invalid gain value");
      }
    } 
    else if (cmd.startsWith("S:")) {
      int val = cmd.substring(2).toInt();
      isStreaming = (val == 1);
      Serial.printf("LOG:Streaming %s\n", isStreaming ? "ON" : "OFF");
    }
    else if (cmd.startsWith("P:")) {
      String pins = cmd.substring(2);
      int c1 = pins.indexOf(',');
      int c2 = pins.indexOf(',', c1 + 1);
      if (c1 > 0 && c2 > c1) {
        int newSck = pins.substring(0, c1).toInt();
        int newWs  = pins.substring(c1 + 1, c2).toInt();
        int newSd  = pins.substring(c2 + 1).toInt();
        i2sSckPin = newSck;
        i2sWsPin  = newWs;
        i2sSdPin  = newSd;
        initI2S(currentSampleRate);
      }
    }
  }
}

void setup() {
  // Set up high-speed hardware USB CDC / UART serial interface
  Serial.begin(921600);
  
  // Wait for serial to connect (helpful for debugging, but doesn't block forever)
  uint32_t startWait = millis();
  while (!Serial && (millis() - startWait < 2000)) {
    delay(10);
  }

  Serial.println("LOG:MEMS Microphone Analyzer initializing...");

  // Settle time for MEMS microphone powerup
  delay(500);

  // Initialize I2S at default 44.1 kHz
  if (initI2S(currentSampleRate)) {
    Serial.println("LOG:I2S initialized successfully");
  } else {
    Serial.println("ERR:I2S initialization failed!");
  }
}

void loop() {
  // Check for commands from the dashboard app
  checkSerialCommands();

  // If streaming is disabled, sleep a bit and return
  if (!isStreaming) {
    delay(50);
    return;
  }

  size_t bytesRead = 0;
  // Read raw 32-bit sample frames from the ESP32 I2S DMA
  esp_err_t result = i2s_read(I2S_PORT, 
                              &i2sRawBuffer, 
                              sizeof(i2sRawBuffer), 
                              &bytesRead, 
                              portMAX_DELAY);

  if (result == ESP_OK && bytesRead > 0) {
    int totalWords = bytesRead / sizeof(int32_t);
    int sampleCount = totalWords / 2;
    if (sampleCount > SAMPLES_PER_PACKET) {
      sampleCount = SAMPLES_PER_PACKET;
    }
    
    // Process and convert 32-bit stereo raw frames to mono 16-bit PCM
    for (int i = 0; i < sampleCount; i++) {
      // Pick whichever slot (Left or Right) contains the active INMP441 data
      int32_t ch0 = i2sRawBuffer[i * 2];
      int32_t ch1 = i2sRawBuffer[i * 2 + 1];
      int32_t rawSample = (ch0 != 0) ? ch0 : ch1;

      // The INMP441 outputs a 24-bit word left-justified in a 32-bit buffer.
      // Shifting right by 14 preserves extra voice dynamic range before clipping.
      int32_t shifted = rawSample >> 14;
      
      // Apply software digital gain with clipping safety
      float scaledVal = shifted * softwareGain;
      if (scaledVal > 32767.0f) {
        audioPayload[i] = 32767;
      } else if (scaledVal < -32768.0f) {
        audioPayload[i] = -32768;
      } else {
        audioPayload[i] = (int16_t)scaledVal;
      }
    }

    // Prepare packet header
    PacketHeader header;
    header.sync1 = SYNC_BYTE_1;
    header.sync2 = SYNC_BYTE_2;
    header.seq = sequenceCounter++;
    header.length = sampleCount * sizeof(int16_t);
    header.timestamp = micros();
    header.sampleRate = currentSampleRate;

    // Calculate CRC / XOR Checksum of header payload (excluding sync bytes)
    uint8_t checksum = 0;
    
    // Checksum header fields
    uint8_t* headerPtr = (uint8_t*)&header;
    for (size_t i = 2; i < sizeof(PacketHeader); i++) {
      checksum ^= headerPtr[i];
    }
    
    // Checksum audio payload
    uint8_t* payloadPtr = (uint8_t*)audioPayload;
    for (size_t i = 0; i < header.length; i++) {
      checksum ^= payloadPtr[i];
    }

    // Write complete frame over serial: [Header] [Audio Data] [Checksum]
    Serial.write((uint8_t*)&header, sizeof(PacketHeader));
    Serial.write(payloadPtr, header.length);
    Serial.write(checksum);
  }
}
