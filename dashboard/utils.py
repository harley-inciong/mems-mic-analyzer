import wave
import io
import json
import pandas as pd
from typing import Dict, Any, List
import numpy as np

class AudioUtils:
    """Helper utilities for exporting data (WAV, CSV, snapshots) from the application."""
    
    @staticmethod
    def save_wav(samples: np.ndarray, sample_rate: int) -> bytes:
        """
        Converts 16-bit PCM samples into a standard WAV byte buffer.
        Returns bytes suitable for downloading in Streamlit.
        """
        # Create an in-memory byte buffer
        wav_buffer = io.BytesIO()
        
        # Write WAV file headers
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)      # Mono
            wav_file.setsampwidth(2)      # 16-bit is 2 bytes
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())
            
        return wav_buffer.getvalue()

    @staticmethod
    def parse_wav_bytes(wav_bytes: bytes):
        """
        Parses WAV bytes (e.g. from st.audio_input) into (mono_int16_samples, sample_rate).
        """
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw_frames = wf.readframes(wf.getnframes())
            if sampwidth == 2:
                samples = np.frombuffer(raw_frames, dtype=np.int16)
            elif sampwidth == 4:
                samples = (np.frombuffer(raw_frames, dtype=np.int32) >> 16).astype(np.int16)
            else:
                samples = ((np.frombuffer(raw_frames, dtype=np.uint8).astype(np.int16) - 128) * 256).astype(np.int16)
            if n_channels > 1:
                samples = samples[::n_channels]
            return samples, sr

    @staticmethod
    def export_measurements_csv(metrics: Dict[str, Any]) -> str:
        """
        Formats the current audio measurements into a CSV string.
        Strips complex nested items (like harmonics list) for clean parsing.
        """
        clean_metrics = {}
        for key, val in metrics.items():
            if key == "harmonics":
                # Convert list of tuples (freq, mag) to string
                clean_metrics[key] = "; ".join([f"{f:.1f}Hz:{m:.4f}" for f, m in val])
            elif isinstance(val, (int, float, str)):
                clean_metrics[key] = val
                
        df = pd.DataFrame([clean_metrics])
        return df.to_csv(index=False)

    @staticmethod
    def export_fft_snapshot_json(freqs: np.ndarray, mags: np.ndarray, sample_rate: int) -> str:
        """
        Packages a snapshot of the FFT magnitude spectrum into a JSON string.
        Includes sample rate and FFT metadata.
        """
        snapshot = {
            "metadata": {
                "sample_rate": sample_rate,
                "num_bins": len(freqs),
                "resolution_hz": freqs[1] - freqs[0] if len(freqs) > 1 else 0
            },
            "data": {
                "frequencies": freqs.tolist(),
                "magnitudes": mags.tolist()
            }
        }
        return json.dumps(snapshot, indent=2)

    @staticmethod
    def export_fft_snapshot_csv(freqs: np.ndarray, mags: np.ndarray) -> str:
        """Packages a snapshot of the FFT magnitude spectrum into a CSV string."""
        df = pd.DataFrame({
            "Frequency_Hz": freqs,
            "Magnitude": mags
        })
        return df.to_csv(index=False)

    @staticmethod
    def compute_waveform_preview(samples: np.ndarray, num_points: int = 120) -> List[float]:
        """
        Downsamples a recording into normalized envelope bars (0.0 to 1.0)
        for fast rendering in a Voice Memos style recording list.
        """
        if len(samples) == 0:
            return [0.0] * num_points
        abs_samples = np.abs(samples.astype(np.float32)) / 32768.0
        if len(abs_samples) < num_points:
            padded = np.pad(abs_samples, (0, num_points - len(abs_samples)))
            return np.clip(padded, 0.02, 1.0).tolist()
        chunks = np.array_split(abs_samples, num_points)
        peaks = [float(np.max(c)) if len(c) > 0 else 0.02 for c in chunks]
        max_peak = max(peaks) if peaks else 1.0
        if max_peak > 0:
            peaks = [max(0.04, min(1.0, p / max(max_peak, 0.15))) for p in peaks]
        return peaks

    @staticmethod
    def compute_audio_summary(samples: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Computes duration, RMS dBFS, and Peak dBFS for a recorded voice clip."""
        if len(samples) == 0 or sample_rate <= 0:
            return {"duration_sec": 0.0, "rms_dbfs": -96.0, "peak_dbfs": -96.0}
        norm = samples.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(norm ** 2)))
        peak = float(np.max(np.abs(norm)))
        rms_db = 20.0 * np.log10(max(rms, 1e-5))
        peak_db = 20.0 * np.log10(max(peak, 1e-5))
        return {
            "duration_sec": len(samples) / float(sample_rate),
            "rms_dbfs": round(float(rms_db), 1),
            "peak_dbfs": round(float(peak_db), 1),
        }

