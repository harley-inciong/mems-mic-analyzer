# Real-Time MEMS Microphone Analyzer (ESP32-C3 + Streamlit)

A complete, high-performance, real-time audio analysis tool that streams high-fidelity digital audio from an **INMP441 MEMS Microphone** connected to an **ESP32-C3**, and visualizes/processes the stream in a professional, modular **Python Streamlit Dashboard** containing dynamic digital signal processing (DSP) filters, frequency metrics, spectral analysis, voice activity detection, and 3D waterfalls.

---

## 🏗️ Project Architecture Overview

```
mems-mic-analyzer/
│
├── firmware/
│   └── firmware.ino          # ESP32-C3 Arduino firmware (I2S DMA + framing)
│
├── dashboard/
│   ├── app.py                # Main Streamlit Dashboard UI & update loop
│   ├── serial_reader.py      # Threaded PySerial reader & Mock data generator
│   ├── dsp.py                # Core DSP blocks (IIR design, features, VAD)
│   ├── visualization.py      # High-speed Plotly themes & charts
│   ├── config.py             # YAML Config parser & defaults
│   └── utils.py              # Export tools (WAV, CSV, snapshots)
│
├── tests/
│   └── test_dsp.py           # Unit test suite for verifying DSP modules
│
├── .github/
│   └── workflows/
│       └── ci.yml            # CI validation workflow (linter + pytest)
│
├── config.yaml               # Application parameters & default values
├── requirements.txt          # Python dependencies
├── Dockerfile                # Streamlit container configuration
└── README.md                 # Setup & Reference Manual (this file)
```

---

## 🔌 Hardware Setup & Wiring

The **INMP441** is an omnidirectional, digital I2S microphone. Connect it to the **ESP32-C3** according to the table below:

| INMP441 Pin | ESP32-C3 Pin | Pin Function | Description |
| :--- | :--- | :--- | :--- |
| **VDD** | **3.3V** | Power | 1.6V to 3.6V supply voltage |
| **GND** | **GND** | Ground | Common system ground |
| **L/R** | **GND** | Channel select | Pull to GND for Left channel (or 3.3V for Right) |
| **SCK** | **GPIO 2** | Serial Clock | I2S Serial Data Clock (BCLK) |
| **WS** | **GPIO 3** | Word Select | I2S Word Select (LRCK) |
| **SD** | **GPIO 1** | Serial Data | I2S Serial Data Output (avoids RST/Pin 4; firmware also auto-scans GPIO 1, 5, 6, 7, 10, 0) |

> [!IMPORTANT]
> The **L/R** channel pin **MUST** be connected to **GND** so that the microphone outputs audio frames on the Left channel. Leaving it floating causes the ESP32 to receive empty buffers or continuous noise.

---

## ⚡ Firmware Architecture (ESP32-C3)

The ESP32-C3 firmware (`firmware/firmware.ino`) is written in C++ using the Arduino core. It initializes the I2S peripheral, configures the Direct Memory Access (DMA) buffers to capture sample blocks without CPU overhead, scales 32-bit slot width samples down to 16-bit PCM, wraps them in a framed structure, and pushes the packets over serial.

### 1. Packet Framing Protocol
To prevent data misalignment (e.g. byte loss causing MSB/LSB swaps that sound like loud white static), all streams are organized into binary packets:
- **Header Sync Word (2 bytes)**: `0x5A, 0xA5`
- **Sequence Counter (1 byte)**: Increments `0` to `255` (used to track dropped packets in the UI)
- **Payload Size (2 bytes)**: Size of the PCM audio data block (default `512` bytes for 256 samples)
- **Timestamp (4 bytes)**: ESP32 microsecond clock (`micros()`)
- **Sample Rate (4 bytes)**: Current sample rate configured in hardware (e.g. `44100` Hz)
- **PCM Data Payload (N bytes)**: Raw 16-bit signed integers
- **XOR Checksum (1 byte)**: Bitwise XOR of all header bytes (excluding sync) and payload bytes

### 2. Live Configuration API
The firmware continuously listens to commands sent from the Python dashboard:
* `R:<rate>\n` - Instructs the ESP32 to dynamically uninstall I2S, change the sample rate (`8000`, `16000`, `22050`, or `44100` Hz) and re-initialize.
* `G:<gain>\n` - Adjusts the software pre-gain multiplier on the board (e.g., `G:2.0\n`).
* `S:<1|0>\n` - Commands the board to start (`1`) or pause (`0`) audio streaming.

---

## 🚀 Dashboard Installation & Running

### Option A — Local Setup

1. **Clone/Move** into the project workspace directory:
   ```bash
   cd mems-mic-analyzer
   ```
2. **Install** Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Execute** the Streamlit application:
   ```bash
   streamlit run dashboard/app.py
   ```

### Option B — Run in Docker
You can run the dashboard in a Docker container (perfect for hosting or quick testing). Note that serial hardware connections (`COM` ports) aren't automatically mapped inside containers on Windows without specific USB IP forwarders, but the dashboard's **Mock Mode** will run perfectly in Docker.

1. **Build** the image:
   ```bash
   docker build -t mems-analyzer .
   ```
2. **Run** the container:
   ```bash
   docker run -p 8501:8501 mems-analyzer
   ```
3. Navigate to `http://localhost:8501` in your browser.

---

## 🧪 Quick Test: Hardware-Free Mock Mode

If you don't have the physical ESP32-C3 or INMP441 mic on hand, the dashboard includes a **Mock Mode**:
1. Open the application.
2. In the sidebar, set the **Serial Port** to **`MOCK`**.
3. The dashboard will automatically simulate realistic voice sweeps, synthetic click anomalies, and ambient background noise matching the correct physical sample rate (e.g. 44.1 kHz at exactly 5.8 ms packet timing) so you can test all DSP filters, graphs, VAD, and measurements immediately!

---

## 🎛️ DSP & Dashboard Features

The dashboard is structured into **7 interactive tabs**:

### 1. 📈 Live Audio
* **Time-Domain Waveform**: Real-time oscilloscope chart.
* **VU Gauges**: Smooth, analog-like circular dials representing RMS and Peak amplitude.
* **Clipping Alarm**: High-frequency indicator that flashes red when input approaches 16-bit limits (`>32000`).

### 2. ⚡ Frequency Analysis
* **FFT Amplitude Spectrum**: Identifies the dominant frequency (`F0`) and plots custom markers.
* **Logarithmic Spectrum**: Toggle to visualize power in decibels (dB).
* **2D Spectrogram**: High-contrast scrolling heatmap of frequency density.
* **3D Waterfall Plot**: Wireframe mesh rendering spectral evolution as a 3D landscape.

### 3. 🎛️ DSP Demonstrations
An active cascade filter chain. Toggle them on/off live and adjust properties to see original vs filtered waveforms:
* **IIR Butterworth/Chebyshev Filters**: Low-pass, High-pass, Band-pass, and Band-stop.
* **Nonlinear Smoothers**: Moving Average and Median filtering (excellent for removing spike anomalies).
* **Dynamic Range Compressor (DRC)**: Dynamic curve with configurable threshold, ratio, and attack/release envelope tracking.
* **Spectral Noise Gate**: Smoothly suppresses segments that fall below the noise floor.

### 4. 📏 Audio Measurements & Exports
* Complete statistics table: RMS, Peak, Crest Factor, Variance, Zero Crossing Rate (ZCR).
* **Signal-to-Noise (SNR)** and **Total Harmonic Distortion (THD%)** estimations:
  $$\text{THD} = \frac{\sqrt{V_2^2 + V_3^2 + V_4^2 + V_5^2}}{V_1} \times 100\%$$
* **Export Actions**: Record the buffer to a standard WAV file for click-free browser playback and download, export metrics to CSV, or save the current FFT spectrum to a JSON snapshot.

### 5. 🌌 Time-Frequency Analysis
* **STFT**: Raw Short-Time Fourier Transform.
* **Mel-Scale Spectrogram**: Computes energy across 40 triangular Mel-filterbanks.
* **MFCCs**: Extracts the first 13 Mel-Frequency Cepstral Coefficients.
* **Real Cepstrum**: Log-spectrum IFFT highlighting pitch periods.

### 6. 🗣️ Voice Activity Detection (VAD)
* **Speech/Silence State Card**: Changes colors depending on activity.
* **Voice Probability**: Continuous confidence gauge.
* **Noise Floor Tracking**: Recursive minimum tracking that adapts to stationary room noise.
* **Vocal Duration Tracker**: Keeps track of current speech lengths.

### 7. 🖥️ Device Status
* **Packet Log**: Monitors Streaming FPS, checksum validation error rates, packet counters, and relative system jitter.

---

## ⚙️ Customizing App Parameters
Modify defaults directly in the `config.yaml` file in the project root. You can adjust clipping limits, VAD thresholds, default themes, buffer sizes, and supported baud rates.

---

## 🧪 Running Automated Tests
Run unit tests to verify the integrity of the math libraries and DSP filters:
```bash
pytest tests/
```
