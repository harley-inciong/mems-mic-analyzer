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
