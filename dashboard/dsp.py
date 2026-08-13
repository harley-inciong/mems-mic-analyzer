import numpy as np
import scipy.signal as signal
from scipy.fft import fft, ifft, dct
from typing import Tuple, Dict, Any, List, Optional

class AudioDSP:
    """
    Implements digital signal processing algorithms, audio measurements,
    time-frequency features, and voice activity detection using NumPy and SciPy.
    """
    
    @staticmethod
    def apply_gain(x: np.ndarray, gain_db: float) -> np.ndarray:
        """Applies digital gain in decibels to the signal."""
        factor = 10.0 ** (gain_db / 20.0)
        return x * factor

    @staticmethod
    def design_iir_filter(
        filter_type: str,
        cutoff_hz: float,
        fs: float,
        order: int = 4,
        rp: float = 1.0,
        stop_hz: Optional[float] = None
    ) -> np.ndarray:
        """
        Designs lowpass, highpass, bandpass, or bandstop Butterworth or Chebyshev I filter.
        Returns SOS (Second Order Sections) format for numerical stability.
        """
        nyq = 0.5 * fs
        
        # Check boundary cases
        if cutoff_hz <= 0 or cutoff_hz >= nyq:
            return np.array([])
            
        # Normalize cutoff
        w0 = cutoff_hz / nyq
        
        if filter_type.startswith("butter_"):
            btype = filter_type.split("_")[1] # lowpass, highpass, bandpass, bandstop
            if btype in ["bandpass", "bandstop"] and stop_hz is not None:
                w0 = [cutoff_hz / nyq, stop_hz / nyq]
                # Double check boundaries
                if w0[0] <= 0 or w0[1] >= 1.0 or w0[0] >= w0[1]:
                    return np.array([])
            return signal.butter(order, w0, btype=btype, output='sos')
            
        elif filter_type.startswith("cheby1_"):
            btype = filter_type.split("_")[1]
            if btype in ["bandpass", "bandstop"] and stop_hz is not None:
                w0 = [cutoff_hz / nyq, stop_hz / nyq]
                if w0[0] <= 0 or w0[1] >= 1.0 or w0[0] >= w0[1]:
                    return np.array([])
            return signal.cheby1(order, rp, w0, btype=btype, output='sos')
            
        return np.array([])

    @staticmethod
    def apply_iir_filter(x: np.ndarray, sos: np.ndarray, zero_phase: bool = False) -> np.ndarray:
        """Applies an IIR filter using second-order sections."""
        if sos.size == 0:
            return x.copy()
        if zero_phase:
            return signal.sosfiltfilt(sos, x)
        else:
            return signal.sosfilt(sos, x)

    @staticmethod
    def apply_moving_average(x: np.ndarray, window_size: int) -> np.ndarray:
        """Applies a moving average smoothing filter."""
        if window_size <= 1:
            return x.copy()
        window = np.ones(window_size) / window_size
        # mode='same' keeps dimensions aligned
        return np.convolve(x, window, mode='same')

    @staticmethod
    def apply_median_filter(x: np.ndarray, window_size: int) -> np.ndarray:
        """Applies a 1D median filter (excellent for spike removal)."""
        if window_size <= 1:
            return x.copy()
        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1
        return signal.medfilt(x, kernel_size=window_size)

    @staticmethod
    def apply_drc(
        x: np.ndarray,
        fs: float,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        attack_ms: float = 10.0,
        release_ms: float = 100.0
    ) -> np.ndarray:
        """
        Applies Dynamic Range Compression (DRC) to the input signal.
        Computes the signal envelope using a fast peak detector,
        calculates gain attenuation, and smooths the response.
        """
        # Convert thresholds
        threshold = 10.0 ** (threshold_db / 20.0)
        
        # Coefficients
        alpha_attack = np.exp(-1.0 / (fs * (attack_ms / 1000.0)))
        alpha_release = np.exp(-1.0 / (fs * (release_ms / 1000.0)))
        
        # Envelope follower
        envelope = np.zeros_like(x)
        curr_env = 0.0
        for i in range(len(x)):
            abs_x = abs(x[i])
            if abs_x > curr_env:
                curr_env = alpha_attack * curr_env + (1.0 - alpha_attack) * abs_x
            else:
                curr_env = alpha_release * curr_env + (1.0 - alpha_release) * abs_x
            envelope[i] = curr_env
            
        # Avoid division-by-zero
        envelope = np.clip(envelope, 1e-6, None)
        
        # Compression gain curve
        # Normalized amplitude: assuming maximum input is 1.0 (float32 range)
        # We scale raw signal to +/-1.0 first, process, then scale back if needed
        max_val = np.max(np.abs(x))
        if max_val == 0:
            return x.copy()
            
        x_norm = x / max_val
        env_norm = envelope / max_val
        
        gain = np.ones_like(x_norm)
        for i in range(len(x_norm)):
            if env_norm[i] > threshold:
                # Gain reduction formula (in log domain)
                excess_db = 20.0 * np.log10(env_norm[i] / threshold)
                gain_db = -excess_db * (1.0 - 1.0 / ratio)
                gain[i] = 10.0 ** (gain_db / 20.0)
                
        return (x_norm * gain) * max_val

    @staticmethod
    def extract_envelope(x: np.ndarray) -> np.ndarray:
        """Extracts the amplitude envelope of a signal using rectifying and low-pass filter."""
        # Simple, fast recursive envelope follower
        abs_x = np.abs(x)
        b, a = signal.butter(1, 0.01, btype='low')
        # filtfilt makes it smooth and time-aligned
        envelope = signal.filtfilt(b, a, abs_x)
        return np.maximum(0, envelope)

    @staticmethod
    def apply_noise_gate(x: np.ndarray, fs: float, threshold_norm: float = 0.005) -> np.ndarray:
        """
        Applies a basic noise gate / spectral subtraction.
        Mutes portions of the signal that fall below the threshold.
        """
        # Quick check: normalized signal range
        max_val = np.max(np.abs(x))
        if max_val == 0:
            return x.copy()
            
        # Envelope-based noise gating
        env = AudioDSP.extract_envelope(x)
        gate = (env > (threshold_norm * max_val)).astype(float)
        
        # Smooth gate transitions to avoid clicks
        b, a = signal.butter(1, 0.02, btype='low')
        gate_smooth = signal.filtfilt(b, a, gate)
        
        return x * gate_smooth

    @staticmethod
    def calculate_window(name: str, size: int) -> np.ndarray:
        """Returns the requested window function vector."""
        if name == "hamming":
            return np.hamming(size)
        elif name == "blackman":
            return np.blackman(size)
        elif name == "kaiser":
            return np.kaiser(size, beta=14)
        else: # Default/Hann
            return np.hanning(size)

    @staticmethod
    def compute_fft(x: np.ndarray, window_name: str = "hann", fft_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the magnitude spectrum of the signal.
        Returns:
            freqs: FFT frequency bins
            mag: FFT magnitude spectrum (normalized linear scale)
        """
        N = len(x)
        if N < fft_size:
            # Pad with zeros
            pad_len = fft_size - N
            x_padded = np.pad(x, (0, pad_len), 'constant')
        else:
            x_padded = x[-fft_size:]
            
        window = AudioDSP.calculate_window(window_name, fft_size)
        windowed_signal = x_padded * window
        
        # Compute FFT and get single-sided spectrum
        fft_res = fft(windowed_signal, n=fft_size)
        half_n = fft_size // 2
        
        # Frequencies axis
        freqs = np.fft.fftfreq(fft_size)[:half_n]
        
        # Magnitude (scaled to represent peak amplitudes)
        mag = (2.0 / fft_size) * np.abs(fft_res[:half_n])
        
        return freqs, mag

    @staticmethod
    def calculate_measurements(x: np.ndarray, fs: float, include_fft: bool = True) -> Dict[str, Any]:
        """
        Calculates time-domain and frequency-domain audio metrics.
        Includes THD and SNR estimations.
        """
        metrics = {}
        
        N = len(x)
        if N == 0:
            return {}
            
        x_float = x.astype(np.float64)
        
        # 1. Time-Domain Metrics
        mean_val = np.mean(x_float)
        metrics["mean"] = mean_val
        
        variance = np.var(x_float)
        metrics["variance"] = variance
        metrics["std"] = np.sqrt(variance)
        
        peak_val = np.max(np.abs(x_float))
        metrics["peak"] = peak_val
        metrics["peak_to_peak"] = np.max(x_float) - np.min(x_float)
        
        rms = np.sqrt(np.mean(x_float ** 2))
        metrics["rms"] = rms
        
        # Crest Factor (ratio of peak to RMS)
        metrics["crest_factor"] = peak_val / rms if rms > 0 else 0.0
        
        # Dynamic Range in dB (relative to maximum scale for int16)
        # Max peak is 32768
        metrics["dynamic_range"] = 20.0 * np.log10(peak_val / (rms + 1e-5)) if rms > 0 else 0.0
        
        # Zero Crossing Rate
        zero_crossings = np.nonzero(np.diff(x_float >= 0))[0]
        metrics["zcr"] = len(zero_crossings) / N
        
        # Signal Energy and Power
        energy = np.sum(x_float ** 2)
        metrics["energy"] = energy
        metrics["power"] = energy / N
        
        # 2. Frequency-Domain Metrics (from FFT)
        if include_fft:
            fft_size = int(2**np.ceil(np.log2(N)))
            # Restrict FFT size to reasonable range (256 - 4096)
            fft_size = np.clip(fft_size, 256, 4096)
            
            freqs, mags = AudioDSP.compute_fft(x_float, "hann", fft_size)
            fs_nyq = fs / 2.0
            freqs = freqs * fs  # Scale back to Hz
            
            # Dominant frequency (peak bin)
            max_idx = np.argmax(mags)
            dominant_freq = freqs[max_idx]
            metrics["dominant_freq"] = dominant_freq
            
            # SNR Estimation (Peak bin compared to average of other bins)
            if len(mags) > 5 and mags[max_idx] > 0:
                noise_bins = np.delete(mags, max_idx)
                avg_noise = np.mean(noise_bins)
                metrics["snr"] = 20.0 * np.log10(mags[max_idx] / (avg_noise + 1e-8))
            else:
                metrics["snr"] = 0.0
                
            # THD Estimation (Total Harmonic Distortion)
            # Search for integer multiples of fundamental (dominant) frequency
            thd = 0.0
            harmonics = []
            if dominant_freq > 20.0 and max_idx > 0:
                fundamental_mag = mags[max_idx]
                harmonic_squares_sum = 0.0
                
                # Find up to 5 harmonics
                for h in range(2, 6):
                    h_freq = dominant_freq * h
                    if h_freq >= fs_nyq:
                        break
                    
                    # Search in a small window around the target harmonic frequency
                    # Find the bin closest to h_freq
                    bin_idx = np.argmin(np.abs(freqs - h_freq))
                    # Search window of +/- 2 bins
                    search_slice = mags[max(0, bin_idx - 2): min(len(mags), bin_idx + 3)]
                    if len(search_slice) > 0:
                        h_mag = np.max(search_slice)
                        harmonic_squares_sum += h_mag ** 2
                        harmonics.append((h_freq, h_mag))
                
                if fundamental_mag > 0:
                    thd = np.sqrt(harmonic_squares_sum) / fundamental_mag
                    
            metrics["thd"] = thd * 100.0  # Percentage
            metrics["harmonics"] = harmonics
        else:
            # Quick default placeholders to prevent downstream errors
            metrics["dominant_freq"] = 0.0
            metrics["snr"] = 0.0
            metrics["thd"] = 0.0
            metrics["harmonics"] = []
        
        return metrics

    @staticmethod
    def make_mel_filterbank(fs: float, n_fft: int, n_mels: int = 40) -> np.ndarray:
        """
        Generates Mel-filterbank matrix.
        Maps linear frequency bins to Mel bins.
        """
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)
            
        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
            
        low_hz = 0.0
        high_hz = fs / 2.0
        
        low_mel = hz_to_mel(low_hz)
        high_mel = hz_to_mel(high_hz)
        
        # Grid of Mel center frequencies
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert Hz to FFT bin numbers
        bins = np.floor((n_fft + 1) * hz_points / fs).astype(int)
        
        # Create filterbank matrix
        fbank = np.zeros((n_mels, n_fft // 2))
        
        for m in range(1, n_mels + 1):
            f_m_minus = bins[m - 1]
            f_m = bins[m]
            f_m_plus = bins[m + 1]
            
            # Left slope
            for k in range(f_m_minus, f_m):
                if f_m - f_m_minus > 0:
                    fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            # Right slope
            for k in range(f_m, f_m_plus):
                if f_m_plus - f_m > 0:
                    fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
                    
        return fbank

    @staticmethod
    def compute_mel_spectrogram(x: np.ndarray, fs: float, n_fft: int = 1024, n_mels: int = 40) -> np.ndarray:
        """Computes Mel Spectrogram of the input buffer."""
        # Get standard FFT spectrum
        freqs, mags = AudioDSP.compute_fft(x, "hann", n_fft)
        
        # Construct filterbank
        fbank = AudioDSP.make_mel_filterbank(fs, n_fft, n_mels)
        
        # Apply filterbank to the FFT magnitudes
        # mags size: n_fft // 2, fbank size: (n_mels, n_fft // 2)
        mel_spectrum = np.dot(fbank, mags)
        
        # Convert to Decibels (log Mel spectrogram)
        # Avoid log(0)
        log_mel_spectrogram = 20.0 * np.log10(np.maximum(mel_spectrum, 1e-5))
        return log_mel_spectrogram

    @staticmethod
    def compute_mfcc(x: np.ndarray, fs: float, n_fft: int = 1024, n_mels: int = 40, n_mfcc: int = 13) -> np.ndarray:
        """Computes Mel-Frequency Cepstral Coefficients (MFCCs)."""
        log_mel_spec = AudioDSP.compute_mel_spectrogram(x, fs, n_fft, n_mels)
        # Apply Discrete Cosine Transform (DCT) type II
        mfcc = dct(log_mel_spec, type=2, norm='ortho')[:n_mfcc]
        return mfcc

    @staticmethod
    def compute_cepstrum(x: np.ndarray) -> np.ndarray:
        """
        Computes the real Cepstrum of the signal.
        cepstrum = IFFT(log(|FFT(x)|))
        """
        N = len(x)
        if N == 0:
            return np.zeros(1)
        # Take magnitude FFT
        fft_res = fft(x)
        # Avoid log(0)
        log_mag = np.log(np.abs(fft_res) + 1e-8)
        # Take IFFT
        cep = np.real(ifft(log_mag))
        return cep

    @staticmethod
    def run_vad(
        x: np.ndarray,
        prev_state: Dict[str, Any],
        energy_threshold: float = 0.01,
        zcr_threshold: float = 0.15,
        smoothing_factor: float = 0.95,
        hangover_frames: int = 5
    ) -> Dict[str, Any]:
        """
        Voice Activity Detection (VAD) algorithm.
        Tracks running noise floor energy and classifies frames as Speech/Silence.
        """
        if len(x) == 0:
            return prev_state
            
        x_float = x.astype(np.float64) / 32768.0  # Normalize to -1.0 .. 1.0
        
        # Frame parameters
        energy = np.mean(x_float ** 2)
        zero_crossings = np.nonzero(np.diff(x_float >= 0))[0]
        zcr = len(zero_crossings) / len(x_float)
        
        # Load state
        running_noise = prev_state.get("noise_floor", 0.001)
        speech_hangover = prev_state.get("hangover", 0)
        speech_duration = prev_state.get("speech_duration", 0.0)
        
        # Adaptive noise floor tracking
        # If energy is small, update noise floor quickly, else update slowly
        if energy < running_noise * 2:
            running_noise = smoothing_factor * running_noise + (1 - smoothing_factor) * energy
        else:
            # Slow adaptation to changes
            running_noise = 0.999 * running_noise + 0.001 * energy
            
        # Safety clamp
        running_noise = max(running_noise, 1e-6)
        
        # Features relative to noise floor
        energy_ratio = energy / running_noise
        
        # Decision logic
        is_speech_frame = (energy_ratio > 3.0 and energy > energy_threshold) or (zcr > zcr_threshold and energy_ratio > 1.5)
        
        # Voice Probability estimation
        voice_prob = 0.0
        if energy_ratio > 1.0:
            # S-curve/sigmoid-like mapping
            voice_prob = min(1.0, (np.log10(energy_ratio) / 3.0))
            if zcr > zcr_threshold:
                voice_prob = min(1.0, voice_prob + 0.15)
                
        voice_prob = max(0.0, voice_prob)
        
        # Apply speech hangover to avoid chopping speech ends
        if is_speech_frame:
            speech_hangover = hangover_frames
            is_voiced = True
        else:
            if speech_hangover > 0:
                speech_hangover -= 1
                is_voiced = True
            else:
                is_voiced = False
                
        # Track speech duration (in seconds, assuming this frame is ~256 samples at 44.1kHz -> ~5.8ms)
        frame_duration = len(x) / 44100.0  # fallback approximate
        if is_voiced:
            speech_duration += frame_duration
        else:
            speech_duration = 0.0  # Reset on silence
            
        return {
            "is_voiced": is_voiced,
            "voice_probability": voice_prob,
            "noise_floor": running_noise,
            "energy_ratio": energy_ratio,
            "hangover": speech_hangover,
            "speech_duration": speech_duration,
            "zcr": zcr,
            "energy": energy
        }
