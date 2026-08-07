import pytest
import numpy as np
import scipy.signal as signal
from dashboard.dsp import AudioDSP

def test_apply_gain():
    x = np.ones(100, dtype=np.float32)
    # Apply +6 dB gain (should double the signal approximately)
    y = AudioDSP.apply_gain(x, 6.02)
    assert np.allclose(y, x * 2.0, rtol=1e-2)

def test_iir_filter_lowpass():
    # Generate signal with low (100 Hz) and high (4000 Hz) frequencies
    fs = 16000.0
    t = np.arange(16000) / fs
    low_freq = np.sin(2 * np.pi * 100.0 * t)
    high_freq = np.sin(2 * np.pi * 4000.0 * t)
    sig = low_freq + high_freq
    
    # Design lowpass filter with cutoff 500 Hz
    sos = AudioDSP.design_iir_filter("butter_lowpass", 500.0, fs, order=4)
    filtered = AudioDSP.apply_iir_filter(sig, sos)
    
    # Calculate power of the high frequency components in original vs filtered
    # High frequency components should be heavily attenuated
    original_high_power = np.mean(np.sin(2 * np.pi * 4000.0 * t[-1000:]) ** 2)
    
    # Let's take the FFT of the filtered signal and verify 4000 Hz component is gone
    freqs, mags = AudioDSP.compute_fft(filtered, fft_size=2048)
    freqs_hz = freqs * fs
    
    bin_4k = np.argmin(np.abs(freqs_hz - 4000.0))
    bin_100 = np.argmin(np.abs(freqs_hz - 100.0))
    
    # Magnitude at 4 kHz should be close to 0, magnitude at 100 Hz should remain strong
    assert mags[bin_4k] < 0.05
    assert mags[bin_100] > 0.4

def test_apply_moving_average():
    x = np.zeros(50)
    x[20:30] = 1.0 # Step input
    y = AudioDSP.apply_moving_average(x, window_size=10)
    
    # Convolution output of a step of length 10 with MA size 10 has a ramp shape
    assert len(y) == len(x)
    assert np.max(y) <= 1.0
    assert y[20] > 0.0

def test_apply_median_filter():
    x = np.zeros(50)
    x[25] = 100.0 # Single huge spike (impulse noise)
    y = AudioDSP.apply_median_filter(x, window_size=5)
    
    # Median filter should completely remove the single-sample spike
    assert y[25] == 0.0
    assert np.all(y == 0.0)

def test_apply_drc():
    fs = 8000.0
    t = np.arange(4000) / fs
    sig = np.sin(2 * np.pi * 200.0 * t)
    # Scale signal so it has dynamic segments (small amplitude and large amplitude)
    sig[:2000] = sig[:2000] * 0.1
    sig[2000:] = sig[2000:] * 1.0
    
    compressed = AudioDSP.apply_drc(sig, fs, threshold_db=-20.0, ratio=10.0, attack_ms=2.0, release_ms=10.0)
    
    # The ratio of the max peak of the large part to the small part should be compressed
    orig_ratio = np.max(np.abs(sig[2000:])) / (np.max(np.abs(sig[:2000])) + 1e-6)
    comp_ratio = np.max(np.abs(compressed[2000:])) / (np.max(np.abs(compressed[:2000])) + 1e-6)
    
    assert comp_ratio < orig_ratio

def test_calculate_measurements():
    fs = 10000.0
    t = np.arange(10000) / fs
    # Pure sine wave at 2500 Hz (fs/4), amplitude 10000 (RMS = 10000 / sqrt(2) = 7071)
    # At fs/4, samples land exactly at n*pi/2, meaning the peak amplitude of 10000.0 is captured.
    sig = 10000.0 * np.sin(2 * np.pi * 2500.0 * t)
    
    metrics = AudioDSP.calculate_measurements(sig, fs)
    
    assert abs(metrics["rms"] - 7071.0) < 50.0
    assert abs(metrics["peak"] - 10000.0) < 50.0
    assert abs(metrics["crest_factor"] - np.sqrt(2)) < 0.1
    assert abs(metrics["dominant_freq"] - 2500.0) < 10.0
    assert metrics["thd"] < 1.0  # Pure sine should have very low harmonic distortion
    assert metrics["zcr"] > 0.1 and metrics["zcr"] < 0.6

def test_time_frequency_features():
    fs = 8000.0
    x = np.random.normal(0, 1.0, 1024)
    
    # 1. Mel Spectrogram
    log_mel = AudioDSP.compute_mel_spectrogram(x, fs, n_fft=512, n_mels=40)
    assert log_mel.shape == (40,)
    
    # 2. MFCCs
    mfccs = AudioDSP.compute_mfcc(x, fs, n_fft=512, n_mels=40, n_mfcc=13)
    assert mfccs.shape == (13,)
    
    # 3. Cepstrum
    cep = AudioDSP.compute_cepstrum(x)
    assert len(cep) == len(x)

def test_vad():
    fs = 8000.0
    state = {
        "noise_floor": 0.001,
        "hangover": 0,
        "speech_duration": 0.0,
        "is_voiced": False,
        "voice_probability": 0.0
    }
    
    # Silence frame (white noise at very small level)
    silence = np.random.normal(0, 10.0, 256)
    state_silence = AudioDSP.run_vad(silence, state, energy_threshold=0.01, zcr_threshold=0.15)
    
    # Speech frame (large amplitude sine wave)
    t = np.arange(256) / fs
    speech = 15000.0 * np.sin(2 * np.pi * 500.0 * t)
    state_speech = AudioDSP.run_vad(speech, state_silence, energy_threshold=0.01, zcr_threshold=0.15)
    
    assert state_silence["is_voiced"] is False
    assert state_speech["is_voiced"] is True
    assert state_speech["voice_probability"] > 0.5
    assert state_speech["speech_duration"] > 0.0
