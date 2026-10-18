/**
 * Real-Time MEMS Microphone Voice Recorder & Analyzer - ESP32-C3 Firmware
 * 
 * Hardware:
 *  - ESP32-C3 Development Board
 *  - INMP441 MEMS Microphone
 * 
 * Connections (Revised to avoid RST / Pin 4):
 *  - INMP441 VDD   -> ESP32-C3 3.3V
 *  - INMP441 GND   -> ESP32-C3 GND
 *  - INMP441 L/R   -> ESP32-C3 GND (Left Channel)
 *  - INMP441 SCK   -> ESP32-C3 GPIO 2 (Serial Clock / BCLK)
 *  - INMP441 WS    -> ESP32-C3 GPIO 3 (Word Select / LRCK)
 *  - INMP441 SD    -> ESP32-C3 GPIO 1 (Serial Data Out / DOUT)
 * 
 * Uses the native ESP32 Arduino Core 3.x <ESP_I2S.h> (i2s_std) driver
 * configured for 32-bit Mono Left-slot Philips I2S capture with DC-offset
 * removal and framed binary packet streaming over USB Serial.
 */

#include <Arduino.h>
#include <ESP_I2S.h>

// Default I2S Pin Definitions for ESP32-C3 (GPIO 1, 2, 3 are adjacent and avoid RST/Pin 4)
#define DEFAULT_I2S_SCK 2
#define DEFAULT_I2S_WS  3
#define DEFAULT_I2S_SD  1

// Packet Framing Constants
#define SYNC_BYTE_1 0x5A
#define SYNC_BYTE_2 0xA5
#define SAMPLES_PER_PACKET 256
#define PAYLOAD_SIZE (SAMPLES_PER_PACKET * sizeof(int16_t)) // 512 bytes

// Global I2S instance and state
I2SClass i2s;
int i2sSckPin = DEFAULT_I2S_SCK;
int i2sWsPin  = DEFAULT_I2S_WS;
int i2sSdPin  = DEFAULT_I2S_SD;
uint32_t currentSampleRate = 44100;
float softwareGain = 2.5f;
bool isStreaming = true;
uint8_t sequenceCounter = 0;

// DC blocker filter state (removes INMP441 static DC offset)
float dcPrevX = 0.0f;
float dcPrevY = 0.0f;

#include <driver/gpio.h>

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
 * Initializes or re-initializes the ESP32-C3 I2S peripheral using the Core 3.x i2s_std driver.
 */
bool initI2S(uint32_t sampleRate) {
  i2s.end();

  // Set I2S pins: BCLK (SCK), WS (LRCK), DOUT (-1 unused), DIN (SD)
  i2s.setPins(i2sSckPin, i2sWsPin, -1, i2sSdPin);

  // Initialize standard Philips I2S in 32-bit Stereo (Both Left & Right slots)
  if (!i2s.begin(I2S_MODE_STD, sampleRate, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO, I2S_STD_SLOT_BOTH)) {
    return false;
  }

  // CRITICAL: Enable internal pull-down on SD *after* i2s.begin() because i2s_std_set_gpio()
  // resets pull resistors. Pull-down holds the INMP441's tri-stated half-frame at 0 (no static!).
  gpio_pullup_dis((gpio_num_t)i2sSdPin);
  gpio_pulldown_en((gpio_num_t)i2sSdPin);

  currentSampleRate = sampleRate;
  dcPrevX = 0.0f;
  dcPrevY = 0.0f;
  return true;
}

/**
 * Non-blocking serial command parser.
 * Commands format:
 *   - "R:<rate>\n"         : Change sample rate (8000, 16000, 22050, 44100)
 *   - "G:<gain>\n"         : Change software digital gain multiplier (e.g. 1.0, 2.5)
 *   - "S:<1|0>\n"          : Start (1) or Stop (0) streaming
 *   - "P:<sck>,<ws>,<sd>\n": Reconfigure I2S GPIO pins dynamically
 * Note: Does NOT print ASCII log strings into the serial stream while streaming binary audio.
 */
void checkSerialCommands() {
  while (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() == 0) continue;

    if (cmd.startsWith("R:")) {
      long rate = cmd.substring(2).toInt();
      if (rate == 8000 || rate == 16000 || rate == 22050 || rate == 44100) {
        initI2S((uint32_t)rate);
      }
    } 
    else if (cmd.startsWith("G:")) {
      float gain = cmd.substring(2).toFloat();
      if (gain >= 0.1f && gain <= 100.0f) {
        softwareGain = gain;
      }
    } 
    else if (cmd.startsWith("S:")) {
      int val = cmd.substring(2).toInt();
      isStreaming = (val == 1);
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
  Serial.begin(921600);
  Serial.setTimeout(5); // Fast 5ms timeout so command reads never stall DMA audio

  uint32_t startWait = millis();
  while (!Serial && (millis() - startWait < 1500)) {
    delay(10);
  }

  // Settle time for INMP441 MEMS microphone powerup
  delay(200);

  // Initialize I2S at default 44.1 kHz
  initI2S(currentSampleRate);
}

void loop() {
  checkSerialCommands();

  if (!isStreaming) {
    delay(20);
    return;
  }

  // Read 256 stereo 32-bit frames (2048 bytes) from the INMP441
  size_t bytesRead = i2s.readBytes((char*)i2sRawBuffer, sizeof(i2sRawBuffer));

  if (bytesRead > 0) {
    int totalWords = bytesRead / sizeof(int32_t);
    int sampleCount = totalWords / 2;
    if (sampleCount > SAMPLES_PER_PACKET) {
      sampleCount = SAMPLES_PER_PACKET;
    }

    // Determine which stereo slot (0 = Left, 1 = Right) has the active microphone signal
    // for the entire packet (never switch mid-packet on zero-crossings!)
    int64_t energy0 = 0;
    int64_t energy1 = 0;
    for (int i = 0; i < sampleCount; i++) {
      energy0 += llabs((long long)(i2sRawBuffer[i * 2] >> 14));
      energy1 += llabs((long long)(i2sRawBuffer[i * 2 + 1] >> 14));
    }
    int activeSlot = (energy0 >= energy1) ? 0 : 1;

    for (int i = 0; i < sampleCount; i++) {
      // INMP441 outputs 24-bit signed two's complement left-aligned in 32-bit word.
      // Shifting right by 14 converts 24-bit speech dynamic range into clean 16-bit PCM range.
      float x = (float)(i2sRawBuffer[i * 2 + activeSlot] >> 14);

      // 1st-order high-pass DC blocking filter (removes mic DC bias & sub-20Hz thump)
      float y = x - dcPrevX + 0.995f * dcPrevY;
      dcPrevX = x;
      dcPrevY = y;

      // Apply software digital gain with hard-clipping protection
      float scaledVal = y * softwareGain;
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

    // Calculate XOR Checksum of header (excluding sync bytes) + audio payload
    uint8_t checksum = 0;
    uint8_t* headerPtr = (uint8_t*)&header;
    for (size_t i = 2; i < sizeof(PacketHeader); i++) {
      checksum ^= headerPtr[i];
    }

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
