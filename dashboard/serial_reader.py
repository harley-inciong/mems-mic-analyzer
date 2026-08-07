import time
import serial
import serial.tools.list_ports
import struct
import threading
import queue
import numpy as np
from typing import List, Tuple, Optional
from collections import deque
from dashboard.config import config_instance

class SerialAudioReader:
    """
    Manages serial communication with the ESP32-C3 in a separate background thread.
    Parses custom packet framing, calculates statistics, and updates a circular buffer.
    Includes a realistic Mock Mode for hardware-free development and demonstration.
    """
    def __init__(self, port: str = "MOCK", baud_rate: int = 921600):
        self.port = port
        self.baud_rate = baud_rate
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        
        # Audio configuration from config
        self.sample_rate = config_instance.audio["default_sample_rate"]
        self.buffer_sec = config_instance.serial["buffer_duration_sec"]
        
        # Thread-safe collections
        self.max_samples = int(self.sample_rate * self.buffer_sec)
        self.audio_buffer = deque(maxlen=self.max_samples)
        self.buffer_lock = threading.Lock()
        
        # Command queue to send to ESP32
        self.command_queue = queue.Queue()
        
        # Statistics
        self.packet_count = 0
        self.dropped_packets = 0
        self.checksum_errors = 0
        self.fps = 0.0
        self.latency_ms = 0.0
        self.last_seq: Optional[int] = None
        self.firmware_version = "v1.0.0"
        self.cpu_usage_est = 15.0 # Estimated firmware CPU usage
        
        # Serial port object
        self.ser: Optional[serial.Serial] = None
        self.connection_error: Optional[str] = None
        
        # Mock mode parameters
        self.mock_freq1 = 440.0
        self.mock_freq2 = 880.0
        self.mock_noise_level = 0.05
        self.mock_time = 0.0

    @staticmethod
    def list_available_ports() -> List[str]:
        """Scans the system for serial ports and returns their names, plus 'MOCK'."""
        ports = [p.device for p in serial.tools.list_ports.comports()]
        return ["MOCK"] + sorted(ports)

    def start(self) -> None:
        """Starts the background reading thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stops the background thread and closes serial connections."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        
        self._close_serial()

    def set_port(self, port: str) -> None:
        """Dynamically switches serial port."""
        if port != self.port:
            was_running = self.is_running
            self.stop()
            self.port = port
            self.connection_error = None
            
            # Reset statistics
            self.packet_count = 0
            self.dropped_packets = 0
            self.checksum_errors = 0
            self.last_seq = None
            
            if was_running:
                self.start()

    def send_sample_rate(self, rate: int) -> None:
        """Queues a sample rate change command to send to the ESP32."""
        self.command_queue.put(f"R:{rate}\n")
        self.sample_rate = rate
        # Resize buffer to maintain standard buffer duration
        self.max_samples = int(self.sample_rate * self.buffer_sec)
        with self.buffer_lock:
            # Recreate buffer with new maxlen
            old_samples = list(self.audio_buffer)
            self.audio_buffer = deque(old_samples, maxlen=self.max_samples)

    def send_gain(self, gain: float) -> None:
        """Queues a gain change command to send to the ESP32."""
        self.command_queue.put(f"G:{gain:.2f}\n")

    def get_audio_data(self, num_samples: int) -> np.ndarray:
        """
        Thread-safe retrieval of the latest N samples.
        If buffer has fewer than N samples, pads with zeros.
        """
        with self.buffer_lock:
            buf_len = len(self.audio_buffer)
            if buf_len == 0:
                return np.zeros(num_samples, dtype=np.int16)
            
            # Retrieve latest num_samples
            samples_to_take = min(num_samples, buf_len)
            data = list(self.audio_buffer)[-samples_to_take:]
            
        data_arr = np.array(data, dtype=np.int16)
        
        # Pad with zeros if we don't have enough data yet
        if len(data_arr) < num_samples:
            data_arr = np.pad(data_arr, (num_samples - len(data_arr), 0), 'constant')
            
        return data_arr

    def get_stats(self) -> dict:
        """Returns current serial connection and buffer statistics."""
        return {
            "port": self.port,
            "connected": self.is_running and (self.port == "MOCK" or (self.ser and self.ser.is_open)),
            "connection_error": self.connection_error,
            "packet_count": self.packet_count,
            "dropped_packets": self.dropped_packets,
            "checksum_errors": self.checksum_errors,
            "fps": self.fps,
            "latency_ms": self.latency_ms,
            "sample_rate": self.sample_rate,
            "buffer_usage_pct": (len(self.audio_buffer) / self.max_samples) * 100 if self.max_samples > 0 else 0,
            "firmware_version": self.firmware_version,
            "cpu_usage_est": self.cpu_usage_est
        }

    def _open_serial(self) -> bool:
        """Attempts to open the serial port connection."""
        if self.port == "MOCK":
            self.connection_error = None
            return True
            
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=config_instance.serial["timeout"],
                write_timeout=0.1
            )
            self.connection_error = None
            return self.ser.is_open
        except Exception as e:
            self.connection_error = str(e)
            print(f"Error opening serial port {self.port}: {e}")
            return False

    def _close_serial(self) -> None:
        """Closes the serial port."""
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception as e:
                print(f"Error closing serial: {e}")
        self.ser = None

    def _run(self) -> None:
        """Main loop that runs in a background thread."""
        while self.is_running:
            if not (self.ser and self.ser.is_open) and self.port != "MOCK":
                # Try to connect/reconnect
                if not self._open_serial():
                    time.sleep(1.0) # Wait before retry
                    continue
                else:
                    print(f"Connected to ESP32 on {self.port}")
            
            if self.port == "MOCK":
                self._run_mock()
            else:
                self._run_hardware()

    def _run_mock(self) -> None:
        """Generates synthetic high-fidelity audio data at realistic intervals."""
        # Packet configuration
        samples_per_packet = 256
        dt = 1.0 / self.sample_rate
        sleep_dur = samples_per_packet / self.sample_rate  # Exactly matches streaming duration
        
        fps_timer = time.time()
        packet_times = deque(maxlen=20)
        
        while self.is_running and self.port == "MOCK":
            start_loop = time.time()
            
            # Check commands in queue (and ignore them or update local state)
            while not self.command_queue.empty():
                cmd = self.command_queue.get_nowait()
                if cmd.startswith("R:"):
                    self.sample_rate = int(cmd.split(":")[1])
                    # Recompute buffer boundaries
                    self.max_samples = int(self.sample_rate * self.buffer_sec)
                    with self.buffer_lock:
                        self.audio_buffer = deque(list(self.audio_buffer), maxlen=self.max_samples)
                    # Update sleep duration
                    sleep_dur = samples_per_packet / self.sample_rate
                    dt = 1.0 / self.sample_rate
                elif cmd.startswith("G:"):
                    # Mock pre-gain changes, nothing to do
                    pass
            
            # Generate synthetic signal
            t = self.mock_time + np.arange(samples_per_packet) * dt
            self.mock_time += samples_per_packet * dt
            
            # Frequency modulation for a dynamic voice-like sweep
            modulator = np.sin(2 * np.pi * 0.2 * t)  # 0.2 Hz modulation
            f1 = self.mock_freq1 + 100.0 * modulator
            f2 = self.mock_freq2 + 200.0 * np.cos(2 * np.pi * 0.1 * t)
            
            # Main tones (harmonics)
            sig = 0.4 * np.sin(2 * np.pi * f1 * t) + 0.25 * np.sin(2 * np.pi * f2 * t)
            
            # Add occasional high-frequency spikes (e.g. click/key-tap simulation)
            if int(t[0] * 2) % 4 == 0:  # Every 2 seconds
                decay = np.exp(-100.0 * (t - t[0]))
                sig += 0.3 * np.sin(2 * np.pi * 3000.0 * t) * decay
                
            # Add white noise
            noise = np.random.normal(0, self.mock_noise_level, samples_per_packet)
            sig += noise
            
            # Convert to signed 16-bit PCM integer range
            audio_samples = (sig * 15000.0).astype(np.int16)
            
            # Push to buffer
            with self.buffer_lock:
                self.audio_buffer.extend(audio_samples)
                
            # Stats tracking
            self.packet_count += 1
            packet_times.append(time.time())
            
            # Simulate occasional network drops (1 out of 200 packets)
            if np.random.rand() < 0.002:
                self.dropped_packets += 1
                
            # Calculate rolling streaming FPS
            if len(packet_times) > 1:
                time_diff = packet_times[-1] - packet_times[0]
                self.fps = len(packet_times) / time_diff if time_diff > 0 else 0.0
            else:
                self.fps = 1.0 / sleep_dur
                
            self.latency_ms = (time.time() - start_loop) * 1000.0
            
            # Precise timing execution
            elapsed = time.time() - start_loop
            sleep_needed = max(0.0, sleep_dur - elapsed)
            if sleep_needed > 0:
                time.sleep(sleep_needed)

    def _run_hardware(self) -> None:
        """Reads hardware serial, decodes custom packet frames, and handles commands."""
        # Config params
        sync_b1 = config_instance.serial["packet_sync_byte1"]
        sync_b2 = config_instance.serial["packet_sync_byte2"]
        header_format = "<BBBHII"  # packing structure
        header_size = struct.calcsize(header_format)
        
        # Flush serial buffers on startup
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception as e:
            print(f"Error resetting serial buffers: {e}")
            self._close_serial()
            return
            
        fps_timer = time.time()
        packet_times = deque(maxlen=30)
        
        while self.is_running and self.ser and self.ser.is_open:
            try:
                # 1. Send outstanding commands to the ESP32
                while not self.command_queue.empty():
                    cmd = self.command_queue.get_nowait()
                    self.ser.write(cmd.encode('utf-8'))
                    self.ser.flush()
                
                # 2. Sync to Magic Word (0x5A, 0xA5)
                # Read 1 byte at a time until we hit sync bytes
                b = self.ser.read(1)
                if not b:
                    continue
                if b[0] != sync_b1:
                    continue
                
                b2 = self.ser.read(1)
                if not b2 or b2[0] != sync_b2:
                    continue
                
                # We found the sync frame! Read remaining header bytes
                header_data = self.ser.read(header_size - 2)
                if len(header_data) < (header_size - 2):
                    self.dropped_packets += 1
                    continue
                
                # Assemble complete header bytes
                full_header_bytes = bytes([sync_b1, sync_b2]) + header_data
                
                # Unpack header
                # format: sync1, sync2, seq, length, timestamp, sampleRate
                sync1, sync2, seq, length, timestamp, sample_rate = struct.unpack(header_format, full_header_bytes)
                
                # 3. Read raw PCM payload
                payload_data = self.ser.read(length)
                if len(payload_data) < length:
                    self.dropped_packets += 1
                    continue
                
                # 4. Read checksum byte
                checksum_byte = self.ser.read(1)
                if len(checksum_byte) < 1:
                    self.dropped_packets += 1
                    continue
                
                expected_checksum = checksum_byte[0]
                
                # 5. Verify Checksum
                checksum = 0
                # XOR header bytes (starting from index 2, sequence ID)
                for b_val in full_header_bytes[2:]:
                    checksum ^= b_val
                # XOR payload bytes
                for b_val in payload_data:
                    checksum ^= b_val
                    
                if checksum != expected_checksum:
                    self.checksum_errors += 1
                    continue # Discard corrupt frame
                
                # 6. Parse audio samples (int16_t array)
                sample_count = length // 2
                audio_samples = np.frombuffer(payload_data, dtype=np.int16)
                
                # Update buffer
                with self.buffer_lock:
                    self.audio_buffer.extend(audio_samples)
                
                # Sequence tracking & dropped packet check
                if self.last_seq is not None:
                    expected_seq = (self.last_seq + 1) & 0xFF
                    if seq != expected_seq:
                        diff = (seq - expected_seq) & 0xFF
                        self.dropped_packets += diff
                self.last_seq = seq
                
                # 7. Update statistics
                self.packet_count += 1
                packet_times.append(time.time())
                
                # Calculate running FPS
                if len(packet_times) > 1:
                    time_diff = packet_times[-1] - packet_times[0]
                    self.fps = len(packet_times) / time_diff if time_diff > 0 else 0.0
                
                # Calculate latency (delay since packet creation on ESP32 vs current PC clock)
                # (Note: Since clocks are not synced, we show relative latency / jitter)
                self.latency_ms = (time.time() - (timestamp / 1e6)) * 1000.0 if timestamp > 0 else 0
                
            except serial.SerialException as se:
                print(f"Serial Exception in thread: {se}")
                self._close_serial()
                break
            except Exception as e:
                print(f"Error reading from serial: {e}")
                time.sleep(0.01)
