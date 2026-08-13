import sys
import os

# Ensure the project root is in sys.path so 'from dashboard.xxx' works
# regardless of where Streamlit launches the script from (local vs Cloud)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import time
import pandas as pd
import scipy.signal as signal
import plotly.graph_objects as go
from typing import Dict, Any, List
from dashboard.config import config_instance
from dashboard.serial_reader import SerialAudioReader
from dashboard.dsp import AudioDSP
from dashboard.visualization import AudioVisualizer, get_theme_colors
from dashboard.utils import AudioUtils

# Set up Streamlit Page Configuration
st.set_page_config(
    page_title="Real-Time MEMS Microphone Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# 1. CSS & Theme Injection for Premium Aesthetics
# ----------------------------------------------------
def inject_custom_css(theme_name: str):
    colors = get_theme_colors(theme_name)
    css = f"""
    <style>
    /* Main app styling */
    .stApp {{
        background-color: {colors["bg"]};
        color: {colors["text"]};
        font-family: 'Courier New', Courier, monospace;
    }}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background-color: {colors["paper_bg"]};
        border-right: 1px solid {colors["grid"]};
    }}
    
    /* Card/Block container styling (Glassmorphism look) */
    div.stMetric, div[data-testid="stCard"] {{
        background-color: {colors["paper_bg"]};
        border: 1px solid {colors["grid"]};
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }}
    
    /* Metrics fonts */
    div[data-testid="stMetricValue"] {{
        font-size: 24px !important;
        font-weight: bold;
        color: {colors["signal"]} !important;
    }}
    
    /* Alerts and custom text boxes */
    .clipping-alert {{
        background-color: rgba(211, 47, 47, 0.2);
        border: 1px solid #D32F2F;
        color: #FFCDD2;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
        animation: blinker 1s linear infinite;
    }}
    
    .vad-alert-voice {{
        background-color: rgba(0, 230, 118, 0.2);
        border: 1px solid #00E676;
        color: #B9F6CA;
        padding: 12px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }}
    
    .vad-alert-silence {{
        background-color: rgba(128, 128, 128, 0.1);
        border: 1px solid {colors["grid"]};
        color: {colors["text"]};
        padding: 12px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }}
    
    @keyframes blinker {{
        50% {{ opacity: 0.5; }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Session State Initialization
# ----------------------------------------------------
if "reader" not in st.session_state:
    st.session_state.reader = SerialAudioReader(port="MOCK")
    st.session_state.reader.start()

if "vad_state" not in st.session_state:
    st.session_state.vad_state = {
        "noise_floor": 0.001,
        "hangover": 0,
        "speech_duration": 0.0,
        "is_voiced": False,
        "voice_probability": 0.0
    }

if "history_mags" not in st.session_state:
    st.session_state.history_mags = np.zeros((50, 512))

if "is_paused" not in st.session_state:
    st.session_state.is_paused = False

if "stat_history_timestamps" not in st.session_state:
    st.session_state.stat_history_timestamps = []
    st.session_state.stat_history_fps = []
    st.session_state.stat_history_dropped = []
    st.session_state.loop_start_time = time.time()

# ----------------------------------------------------
# 3. Sidebar UI & Navigation Dispatcher
# ----------------------------------------------------
st.sidebar.markdown(f"## 🎙️ Control Center")

# Dynamic Theme Selector
theme_name = st.sidebar.selectbox("Color Theme", config_instance.ui["themes"], index=0)
inject_custom_css(theme_name)
colors = get_theme_colors(theme_name)

# Sidebar Tab Navigation (Server-Side Dispatcher)
st.sidebar.markdown("### 🗂️ Active Dashboard View")
selected_tab = st.sidebar.radio(
    "Select View",
    ["📈 Live Audio", "⚡ Frequency Analysis", "🎛️ DSP Demonstrations", "📏 Audio Measurements", "🌌 Time-Frequency", "🗣️ Voice Activity", "🖥️ Device Status"],
    index=0
)

# Device Connection
st.sidebar.markdown("### 🔌 Device Connection")
available_ports = SerialAudioReader.list_available_ports()
default_index = 1 if len(available_ports) > 1 else 0

if st.sidebar.button("🔄 Refresh Port List"):
    st.rerun()

selected_port = st.sidebar.selectbox("Serial Port", available_ports, index=default_index)

# Apply port update
if selected_port != st.session_state.reader.port:
    st.session_state.reader.set_port(selected_port)
    fft_size = config_instance.dsp["default_fft_size"]
    st.session_state.history_mags = np.zeros((50, fft_size // 2))

# Live Control Checks
live_stream = st.sidebar.checkbox("Start Live Stream", value=True)

if st.sidebar.button("▶️ Resume Rendering" if st.session_state.is_paused else "⏹️ Pause Rendering"):
    st.session_state.is_paused = not st.session_state.is_paused
    st.rerun()

# Sample Rate Control
sample_rate = st.sidebar.selectbox(
    "Sample Rate (Hz)", 
    config_instance.audio["sample_rates"], 
    index=config_instance.audio["sample_rates"].index(config_instance.audio["default_sample_rate"])
)

# Apply sample rate update
if sample_rate != st.session_state.reader.sample_rate:
    st.session_state.reader.send_sample_rate(sample_rate)
    fft_size = config_instance.dsp["default_fft_size"]
    st.session_state.history_mags = np.zeros((50, fft_size // 2))

# Digital Gain
hardware_gain = st.sidebar.slider("Digital Pre-Gain (ESP32)", 0.5, 10.0, 1.0, step=0.1)
st.session_state.reader.send_gain(hardware_gain)

# Selective Sidebar FFT Config (Only shows when relevant to active view)
uses_fft = selected_tab in ["⚡ Frequency Analysis", "🎛️ DSP Demonstrations", "📏 Audio Measurements", "🌌 Time-Frequency"]
if uses_fft:
    st.sidebar.markdown("### 📊 FFT Settings")
    fft_size = st.sidebar.selectbox("FFT Window Size", config_instance.dsp["fft_sizes"], index=2) # 1024
    window_type = st.sidebar.selectbox("Window Function", config_instance.dsp["windows"], index=1) # Hann
    
    # Resize rolling history if FFT size changed
    num_bins = fft_size // 2
    if st.session_state.history_mags.shape[1] != num_bins:
        st.session_state.history_mags = np.zeros((50, num_bins))
else:
    # Use config defaults in background
    fft_size = config_instance.dsp["default_fft_size"]
    window_type = config_instance.dsp["default_window"]

# Refresh rate control
refresh_rate_ms = st.sidebar.slider("Dashboard Refresh Rate (ms)", 50, 500, 100, step=10)

# Display connection error in sidebar if present
stats = st.session_state.reader.get_stats()
if stats.get("connection_error"):
    st.sidebar.error(f"❌ Connection Error:\n{stats['connection_error']}")

# Title Header
st.markdown(f"<h1 style='text-align: center; color: {colors['signal']}; margin-top: 10px; margin-bottom: 20px;'>🎙️ MEMS Microphone Analyzer</h1>", unsafe_allow_html=True)

# ----------------------------------------------------
# 4. Optimized Selective Buffer Pulling
# ----------------------------------------------------
# Adjust pulled audio size depending on what is being displayed
if selected_tab == "📈 Live Audio":
    display_length = 1024
elif selected_tab == "🖥️ Device Status":
    display_length = 256
else:
    display_length = max(2048, fft_size)

raw_samples = st.session_state.reader.get_audio_data(display_length)
norm_samples = raw_samples.astype(np.float32) / 32768.0

# ----------------------------------------------------
# 5. Active Tab Dispatcher (Saves massive CPU)
# ----------------------------------------------------

if selected_tab == "📈 Live Audio":
    # Skip heavy FFT and only compute time-domain measurements
    metrics = AudioDSP.calculate_measurements(raw_samples, sample_rate, include_fft=False)
    
    st.subheader("Time-Domain Live Analysis")
    t1_col1, t1_col2 = st.columns([3, 1])
    
    with t1_col1:
        fig_wave = AudioVisualizer.plot_waveform(raw_samples[-1024:], sample_rate, theme_name)
        st.plotly_chart(fig_wave, width="stretch", config={'displayModeBar': False})
        
    with t1_col2:
        peak_val = metrics.get("peak", 0.0)
        if peak_val >= config_instance.dsp["clipping_threshold"]:
            st.markdown('<div class="clipping-alert">⚠️ SIGNAL CLIPPING DETECTED!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; color: #888; font-weight: bold; padding: 10px; border: 1px solid #333; border-radius: 5px;">Signal Level OK</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        fig_rms = AudioVisualizer.plot_gauge(metrics.get("rms", 0.0), "RMS Amplitude", 0, 32768, "PCM", theme_name)
        st.plotly_chart(fig_rms, width="stretch", config={'displayModeBar': False})
        
        fig_peak = AudioVisualizer.plot_gauge(metrics.get("peak", 0.0), "Peak Amplitude", 0, 32768, "PCM", theme_name)
        st.plotly_chart(fig_peak, width="stretch", config={'displayModeBar': False})

elif selected_tab == "⚡ Frequency Analysis":
    # Compute spectral components
    freqs, mags = AudioDSP.compute_fft(raw_samples, window_name=window_type, fft_size=fft_size)
    freqs_hz = freqs * sample_rate
    
    # Roll spectrogram history
    st.session_state.history_mags = np.roll(st.session_state.history_mags, -1, axis=0)
    st.session_state.history_mags[-1, :] = mags
    
    # Compute peak metrics for dominant/harmonic indicators
    metrics = AudioDSP.calculate_measurements(raw_samples, sample_rate, include_fft=True)
    
    st.subheader("Frequency Spectral Analysis")
    t2_col1, t2_col2 = st.columns([1, 1])
    
    with t2_col1:
        fft_log_toggle = st.checkbox("Logarithmic dB Scale", value=True)
        fig_fft = AudioVisualizer.plot_fft(
            freqs_hz, 
            mags, 
            metrics.get("dominant_freq", 0.0), 
            metrics.get("harmonics", []), 
            fft_log_toggle, 
            theme_name
        )
        st.plotly_chart(fig_fft, width="stretch", config={'displayModeBar': False})
        
    with t2_col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        fig_spec = AudioVisualizer.plot_spectrogram(st.session_state.history_mags, freqs_hz, theme_name)
        st.plotly_chart(fig_spec, width="stretch", config={'displayModeBar': False})
        
    # Checkbox to let user toggle the heavy 3D Waterfall to completely remove lag
    st.markdown("---")
    enable_waterfall = st.checkbox("💻 Enable Heavy 3D Waterfall (Disable if lagging)", value=False)
    if enable_waterfall:
        fig_water = AudioVisualizer.plot_waterfall(st.session_state.history_mags, freqs_hz, theme_name)
        st.plotly_chart(fig_water, width="stretch", config={'displayModeBar': False})
    else:
        st.info("3D Waterfall mesh rendering is currently disabled. Toggle the checkbox above to enable it.")

elif selected_tab == "🎛️ DSP Demonstrations":
    st.subheader("Cascade Digital Signal Processing Blocks")
    
    # Dynamic parameter layout
    dsp_ctrl1, dsp_ctrl2, dsp_ctrl3, dsp_ctrl4 = st.columns(4)
    with dsp_ctrl1:
        dsp_gain = st.slider("Post-Gain (dB)", -20.0, 20.0, 0.0, step=0.5)
        enable_lpf = st.checkbox("Low-Pass Filter (LPF)")
        lpf_cutoff = st.slider("LPF Cutoff (Hz)", 100, int(sample_rate/2 - 100), 2000)
    with dsp_ctrl2:
        enable_hpf = st.checkbox("High-Pass Filter (HPF)")
        hpf_cutoff = st.slider("HPF Cutoff (Hz)", 20, 5000, 150)
        enable_bpf = st.checkbox("Band-Pass Filter (BPF)")
        bpf_center = st.slider("BPF Center (Hz)", 100, int(sample_rate/2 - 200), 1000)
        bpf_width = st.slider("BPF Width (Hz)", 50, 2000, 500)
    with dsp_ctrl3:
        enable_bsf = st.checkbox("Band-Stop Filter (BSF)")
        bsf_center = st.slider("BSF Center (Hz)", 50, 5000, 60)
        bsf_width = st.slider("BSF Width (Hz)", 10, 500, 20)
        enable_mov_avg = st.checkbox("Moving Average")
        mov_avg_len = st.slider("MA Window Size", 2, 50, 8)
    with dsp_ctrl4:
        enable_median = st.checkbox("Median Filter (De-spiking)")
        median_len = st.slider("Median Window Size", 3, 31, 5, step=2)
        enable_drc = st.checkbox("Dynamic Range Compressor")
        drc_threshold = st.slider("DRC Threshold (dB)", -40, -5, -20)
        enable_noise_gate = st.checkbox("Spectral Noise Gate")

    # Apply calculations
    filtered_samples = raw_samples.copy()
    if dsp_gain != 0.0:
        filtered_samples = AudioDSP.apply_gain(filtered_samples, dsp_gain)
    if enable_lpf:
        sos = AudioDSP.design_iir_filter("butter_lowpass", lpf_cutoff, sample_rate)
        filtered_samples = AudioDSP.apply_iir_filter(filtered_samples, sos)
    if enable_hpf:
        sos = AudioDSP.design_iir_filter("butter_highpass", hpf_cutoff, sample_rate)
        filtered_samples = AudioDSP.apply_iir_filter(filtered_samples, sos)
    if enable_bpf:
        sos = AudioDSP.design_iir_filter("butter_bandpass", bpf_center, sample_rate, stop_hz=bpf_center+bpf_width)
        filtered_samples = AudioDSP.apply_iir_filter(filtered_samples, sos)
    if enable_bsf:
        sos = AudioDSP.design_iir_filter("butter_bandstop", bsf_center, sample_rate, stop_hz=bsf_center+bsf_width)
        filtered_samples = AudioDSP.apply_iir_filter(filtered_samples, sos)
    if enable_mov_avg:
        filtered_samples = AudioDSP.apply_moving_average(filtered_samples, mov_avg_len)
    if enable_median:
        filtered_samples = AudioDSP.apply_median_filter(filtered_samples, median_len)
    if enable_drc:
        filtered_samples = AudioDSP.apply_drc(filtered_samples, sample_rate, threshold_db=drc_threshold)
    if enable_noise_gate:
        filtered_samples = AudioDSP.apply_noise_gate(filtered_samples, sample_rate)

    t3_col1, t3_col2 = st.columns(2)
    with t3_col1:
        fig_orig = AudioVisualizer.plot_waveform(raw_samples[-512:], sample_rate, theme_name)
        fig_orig.update_layout(title="Original Signal (Latest 512 samples)", height=220)
        st.plotly_chart(fig_orig, width="stretch", config={'displayModeBar': False})
    with t3_col2:
        fig_filt = AudioVisualizer.plot_waveform(filtered_samples[-512:], sample_rate, theme_name)
        fig_filt.update_layout(title="Filtered Signal (Latest 512 samples)", height=220)
        fig_filt.update_traces(line=dict(color=colors["secondary"]))
        st.plotly_chart(fig_filt, width="stretch", config={'displayModeBar': False})
        
    freqs, orig_mags = AudioDSP.compute_fft(raw_samples, window_name=window_type, fft_size=fft_size)
    _, filt_mags = AudioDSP.compute_fft(filtered_samples, window_name=window_type, fft_size=fft_size)
    freqs_hz = freqs * sample_rate
    
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=freqs_hz, y=20*np.log10(np.maximum(orig_mags, 1e-5)), name="Original", line=dict(color=colors["signal"], width=1.5)))
    fig_comp.add_trace(go.Scatter(x=freqs_hz, y=20*np.log10(np.maximum(filt_mags, 1e-5)), name="Filtered", line=dict(color=colors["secondary"], width=1.5)))
    AudioVisualizer._apply_layout(fig_comp, colors, "FFT Filter Response Comparison", "Frequency (Hz)", "Power (dB)")
    fig_comp.update_layout(showlegend=True, height=280)
    st.plotly_chart(fig_comp, width="stretch", config={'displayModeBar': False})

elif selected_tab == "📏 Audio Measurements":
    st.subheader("Quantitative Audio Metrology")
    
    # Run full FFT metrics
    metrics = AudioDSP.calculate_measurements(raw_samples, sample_rate, include_fft=True)
    freqs, mags = AudioDSP.compute_fft(raw_samples, window_name=window_type, fft_size=fft_size)
    freqs_hz = freqs * sample_rate
    
    t4_col1, t4_col2 = st.columns([1, 1])
    with t4_col1:
        st.markdown("### Audio Analysis Metrics")
        measurement_data = {
            "Metric": [
                "RMS Level (PCM)", "Peak Level (PCM)", "Peak-to-Peak", 
                "Mean Offset", "Standard Deviation", "Crest Factor", 
                "Dynamic Range (dB)", "Zero Crossing Rate (ZCR)", "Signal Energy",
                "Estimated SNR (dB)", "Estimated THD (%)", "Dominant Frequency (Hz)"
            ],
            "Current Value": [
                f"{metrics.get('rms', 0.0):.2f}", 
                f"{metrics.get('peak', 0.0):.2f}", 
                f"{metrics.get('peak_to_peak', 0.0):.2f}",
                f"{metrics.get('mean', 0.0):.2f}", 
                f"{metrics.get('std', 0.0):.2f}", 
                f"{metrics.get('crest_factor', 0.0):.2f}",
                f"{metrics.get('dynamic_range', 0.0):.1f} dB", 
                f"{metrics.get('zcr', 0.0):.4f}", 
                f"{metrics.get('energy', 0.0):.1e}",
                f"{metrics.get('snr', 0.0):.1f} dB", 
                f"{metrics.get('thd', 0.0):.2f}%", 
                f"{metrics.get('dominant_freq', 0.0):.1f} Hz"
            ]
        }
        st.table(pd.DataFrame(measurement_data))
        
    with t4_col2:
        st.markdown("### Captures & Export Utilities")
        rec_duration = st.slider("Capture Duration (seconds)", 1.0, 10.0, 3.0, step=0.5)
        capture_btn = st.button("🔴 Start Capture Recording")
        
        if capture_btn:
            st.info("Recording buffer capture...")
            rec_samples_count = int(rec_duration * sample_rate)
            captured_samples = st.session_state.reader.get_audio_data(rec_samples_count)
            st.success(f"Captured {rec_duration} seconds of audio!")
            
            wav_bytes = AudioUtils.save_wav(captured_samples, sample_rate)
            st.audio(wav_bytes, format="audio/wav")
            
            d_col1, d_col2, d_col3 = st.columns(3)
            with d_col1:
                st.download_button(
                    label="💾 Download Audio as WAV",
                    data=wav_bytes,
                    file_name="audio_capture.wav",
                    mime="audio/wav"
                )
            with d_col2:
                csv_metrics = AudioUtils.export_measurements_csv(metrics)
                st.download_button(
                    label="📊 Download Measurements CSV",
                    data=csv_metrics,
                    file_name="measurements.csv",
                    mime="text/csv"
                )
            with d_col3:
                fft_snap_json = AudioUtils.export_fft_snapshot_json(freqs_hz, mags, sample_rate)
                st.download_button(
                    label="🌌 Download FFT Snapshot JSON",
                    data=fft_snap_json,
                    file_name="fft_snapshot.json",
                    mime="application/json"
                )

elif selected_tab == "🌌 Time-Frequency":
    st.subheader("Time-Frequency & Cepstral Features")
    
    t5_col1, t5_col2 = st.columns(2)
    with t5_col1:
        stft_nperseg = min(256, fft_size)
        stft_noverlap = stft_nperseg // 2
        f_stft, t_stft, Zxx = signal.stft(norm_samples, fs=sample_rate, nperseg=stft_nperseg, noverlap=stft_noverlap)
        
        fig_stft = go.Figure(data=go.Heatmap(
            x=t_stft,
            y=f_stft,
            z=20.0 * np.log10(np.maximum(np.abs(Zxx), 1e-5)),
            colorscale=colors["colorscale"],
            zmin=-80,
            zmax=-10,
            showscale=False
        ))
        AudioVisualizer._apply_layout(fig_stft, colors, "Short-Time Fourier Transform (STFT)", "Time (sec)", "Frequency (Hz)")
        fig_stft.update_layout(height=260)
        st.plotly_chart(fig_stft, width="stretch", config={'displayModeBar': False})
        
        cep = AudioDSP.compute_cepstrum(raw_samples)
        quefrency_axis = np.arange(len(cep)) / sample_rate
        fig_cep = go.Figure()
        show_bins = min(150, len(cep))
        fig_cep.add_trace(go.Scatter(
            x=quefrency_axis[:show_bins] * 1000.0,
            y=cep[:show_bins],
            mode='lines',
            line=dict(color=colors["signal"], width=1.5)
        ))
        AudioVisualizer._apply_layout(fig_cep, colors, "Real Cepstrum", "Quefrency (ms)", "Cepstral Magnitude")
        fig_cep.update_layout(height=260)
        st.plotly_chart(fig_cep, width="stretch", config={'displayModeBar': False})
        
    with t5_col2:
        log_mel = AudioDSP.compute_mel_spectrogram(raw_samples, sample_rate, n_fft=fft_size, n_mels=40)
        mel_bins = np.arange(40)
        fig_mel = go.Figure(data=go.Bar(
            x=mel_bins,
            y=log_mel,
            marker_color=colors["secondary"]
        ))
        AudioVisualizer._apply_layout(fig_mel, colors, "Mel-Scale Spectrogram (40 filterbanks)", "Mel Bin Index", "Magnitude (dB)")
        fig_mel.update_layout(height=260)
        st.plotly_chart(fig_mel, width="stretch", config={'displayModeBar': False})
        
        mfcc = AudioDSP.compute_mfcc(raw_samples, sample_rate, n_fft=fft_size, n_mels=40, n_mfcc=13)
        mfcc_bins = np.arange(13)
        fig_mfcc = go.Figure(data=go.Bar(
            x=mfcc_bins,
            y=mfcc,
            marker_color=colors["accent"]
        ))
        AudioVisualizer._apply_layout(fig_mfcc, colors, "MFCCs (First 13 coefficients)", "Coefficient Index", "Amplitude")
        fig_mfcc.update_layout(height=260)
        st.plotly_chart(fig_mfcc, width="stretch", config={'displayModeBar': False})

elif selected_tab == "🗣️ Voice Activity":
    st.subheader("Speech & Voice Activity Detection (VAD)")
    
    # Run VAD logic
    st.session_state.vad_state = AudioDSP.run_vad(
        raw_samples[-256:], 
        st.session_state.vad_state,
        energy_threshold=config_instance.dsp["vad"]["energy_threshold"],
        zcr_threshold=config_instance.dsp["vad"]["zcr_threshold"],
        smoothing_factor=config_instance.dsp["vad"]["smoothing_factor"],
        hangover_frames=config_instance.dsp["vad"]["hangover_frames"]
    )
    vad = st.session_state.vad_state
    
    t6_col1, t6_col2 = st.columns([1, 2])
    with t6_col1:
        st.markdown("### Detection States")
        if vad.get("is_voiced", False):
            st.markdown('<div class="vad-alert-voice">🗣️ SPEECH DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="vad-alert-silence">🤫 SILENCE / NOISE</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        v_prob = vad.get("voice_probability", 0.0)
        st.markdown(f"**Voice Probability:** {v_prob * 100:.1f}%")
        st.progress(float(v_prob))
        
        st.markdown(f"""
        - **Running Speech Duration**: {vad.get('speech_duration', 0.0):.2f} seconds
        - **Estimated Noise Floor**: {vad.get('noise_floor', 0.0):.1e} RMS²
        - **Current Frame ZCR**: {vad.get('zcr', 0.0):.4f}
        """)
        
    with t6_col2:
        st.markdown("### Energy & Zero-Crossing Rates")
        fig_vad = go.Figure()
        fig_vad.add_trace(go.Scatter(y=[vad.get('energy', 0.0)], mode='markers', name="Energy", marker=dict(size=15, color=colors["signal"])))
        fig_vad.add_trace(go.Scatter(y=[vad.get('zcr', 0.0)], mode='markers', name="ZCR", marker=dict(size=15, color=colors["secondary"])))
        AudioVisualizer._apply_layout(fig_vad, colors, "VAD Multi-feature Space", "Feature", "Value")
        fig_vad.update_layout(height=220)
        st.plotly_chart(fig_vad, width="stretch", config={'displayModeBar': False})

elif selected_tab == "🖥️ Device Status":
    st.subheader("ESP32 Serial & Buffer Status Logger")
    
    t7_col1, t7_col2 = st.columns(2)
    with t7_col1:
        st.markdown("### Diagnostic Information")
        conn_status = "Connected" if stats["connected"] else "Disconnected"
        conn_color = "#00E676" if stats["connected"] else "#D32F2F"
        st.markdown(f"""
        - **Active Port**: `{stats['port']}`
        - **Connection Status**: <span style="color: {conn_color}; font-weight: bold;">{conn_status}</span>
        - **Running Frame Rate**: `{stats['fps']:.1f} FPS`
        - **Total Received Packets**: `{stats['packet_count']}`
        - **Dropped Packet Count**: `{stats['dropped_packets']}`
        - **XOR Checksum Errors**: `{stats['checksum_errors']}`
        - **Relative Latency/Jitter**: `{stats['latency_ms']:.1f} ms`
        - **Firmware Version**: `{stats['firmware_version']}`
        - **Fm CPU Estimate**: `{stats['cpu_usage_est']}%`
        """, unsafe_allow_html=True)
        
    with t7_col2:
        st.markdown("### Streaming Performance Diagnostic Log")
        fig_diag = go.Figure()
        fig_diag.add_trace(go.Scatter(x=list(st.session_state.stat_history_timestamps), y=list(st.session_state.stat_history_fps), name="Streaming FPS", line=dict(color=colors["signal"])))
        fig_diag.add_trace(go.Scatter(x=list(st.session_state.stat_history_timestamps), y=list(st.session_state.stat_history_dropped), name="Dropped Packets", line=dict(color=colors["secondary"])))
        AudioVisualizer._apply_layout(fig_diag, colors, "Stream Performance over Time", "Time (sec)", "Value")
        fig_diag.update_layout(showlegend=True, height=220)
        st.plotly_chart(fig_diag, width="stretch", config={'displayModeBar': False})

# ----------------------------------------------------
# 6. Stream Rate Limiting and Dynamic Rerun Trigger
# ----------------------------------------------------
if live_stream and not st.session_state.is_paused:
    time.sleep(refresh_rate_ms / 1000.0)
    st.rerun()
else:
    st.info("⏸️ Rendering is paused. Check 'Start Live Stream' or click 'Resume Rendering' in the sidebar to start the analyzer stream.")
