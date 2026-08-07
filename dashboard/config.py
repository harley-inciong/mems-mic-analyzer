import os
import yaml
from typing import Dict, Any

# Default hardcoded configuration as a fallback
DEFAULT_CONFIG: Dict[str, Any] = {
    "serial": {
        "default_port": "COM3",
        "baud_rate": 921600,
        "timeout": 0.1,
        "packet_sync_byte1": 0x5A,
        "packet_sync_byte2": 0xA5,
        "payload_size_bytes": 512,
        "buffer_duration_sec": 2.0
    },
    "audio": {
        "default_sample_rate": 44100,
        "sample_rates": [8000, 16000, 22050, 44100],
        "bits_per_sample": 16,
        "channels": 1
    },
    "dsp": {
        "default_fft_size": 1024,
        "fft_sizes": [256, 512, 1024, 2048, 4096],
        "default_window": "hann",
        "windows": ["hamming", "hann", "blackman", "kaiser"],
        "clipping_threshold": 32000,
        "noise_gate_threshold": 0.005,
        "vad": {
            "energy_threshold": 0.01,
            "zcr_threshold": 0.15,
            "smoothing_factor": 0.95,
            "hangover_frames": 5
        }
    },
    "ui": {
        "default_theme": "Cyberpunk",
        "themes": ["Cyberpunk", "Ocean", "Emerald", "Dark", "Light"],
        "refresh_rate_ms": 100
    }
}

class AppConfig:
    """Manages application settings loaded from config.yaml with default fallbacks."""
    
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
        
        # Look for config.yaml in the project root relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.abspath(os.path.join(current_dir, "..", "config.yaml"))
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config:
                        # Deep merge or simple update
                        self._deep_update(self.config, loaded_config)
            except Exception as e:
                print(f"Error loading config.yaml: {e}. Using defaults.")
                
        # Expose properties for ease of access
        self.serial = self.config["serial"]
        self.audio = self.config["audio"]
        self.dsp = self.config["dsp"]
        self.ui = self.config["ui"]

    def _deep_update(self, base_dict: dict, update_dict: dict) -> None:
        """Recursively updates a nested dictionary."""
        for key, val in update_dict.items():
            if isinstance(val, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], val)
            else:
                base_dict[key] = val

# Global configuration instance
config_instance = AppConfig()
