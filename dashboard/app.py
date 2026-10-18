import sys
import os

# Ensure the project root is in sys.path so 'from dashboard.xxx' works
# regardless of where Streamlit launches the script from (local vs Cloud)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import numpy as np
import time
import json
import wave
import io
from datetime import datetime
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
    page_title="MEMS Voice Recorder & Studio Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "recordings")
METADATA_FILE = os.path.join(RECORDINGS_DIR, "recordings_index.json")


# ----------------------------------------------------
# 1. Persistent Recordings Library Helpers
# ----------------------------------------------------
def ensure_recordings_dir() -> bool:
    try:
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        return True
    except Exception:
        return False


def load_saved_recordings() -> List[Dict[str, Any]]:
    """Loads saved recordings from the local recordings/ folder if available."""
    if not ensure_recordings_dir() or not os.path.exists(METADATA_FILE):
        return []
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        loaded = []
        for item in index_data:
            wav_path = os.path.join(RECORDINGS_DIR, item.get("filename", ""))
            if os.path.exists(wav_path):
                with open(wav_path, "rb") as wf:
                    item["wav_bytes"] = wf.read()
                loaded.append(item)
        return loaded
    except Exception:
        return []


def persist_recordings_index(recordings: List[Dict[str, Any]]) -> None:
    """Saves recording metadata (excluding raw bytes) to disk."""
    if not ensure_recordings_dir():
        return
    try:
        serializable = []
        for r in recordings:
            copy_r = {k: v for k, v in r.items() if k != "wav_bytes"}
            serializable.append(copy_r)
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
    except Exception:
        pass


def save_new_recording(
    samples: np.ndarray,
    sample_rate: int,
    title: str,
    source_port: str,
) -> Dict[str, Any]:
    """Packages PCM samples into a WAV recording entry, saves to disk & session state."""
    wav_bytes = AudioUtils.save_wav(samples, sample_rate)
    summary = AudioUtils.compute_audio_summary(samples, sample_rate)
    preview_bars = AudioUtils.compute_waveform_preview(samples, num_points=90)

    now = datetime.now()
    rec_id = now.strftime("%Y%m%d_%H%M%S_%f")[:19]
    safe_filename = f"rec_{rec_id}.wav"

    if ensure_recordings_dir():
        try:
            with open(os.path.join(RECORDINGS_DIR, safe_filename), "wb") as wf:
                wf.write(wav_bytes)
        except Exception:
            pass

    entry = {
        "id": rec_id,
        "title": title.strip() if title.strip() else f"Voice Memo {len(st.session_state.recordings) + 1}",
        "created_at": now.strftime("%b %d, %Y • %I:%M:%S %p"),
        "duration_sec": round(summary["duration_sec"], 2),
        "sample_rate": int(sample_rate),
        "num_samples": int(len(samples)),
        "rms_dbfs": summary["rms_dbfs"],
        "peak_dbfs": summary["peak_dbfs"],
        "source": source_port,
        "filename": safe_filename,
        "preview_bars": preview_bars,
        "wav_bytes": wav_bytes,
    }

    st.session_state.recordings.insert(0, entry)
    persist_recordings_index(st.session_state.recordings)
    return entry


def delete_recording(rec_id: str) -> None:
    """Deletes a recording from session state and disk."""
    remaining = []
    for r in st.session_state.recordings:
        if r["id"] == rec_id:
            if ensure_recordings_dir() and r.get("filename"):
                wav_path = os.path.join(RECORDINGS_DIR, r["filename"])
                if os.path.exists(wav_path):
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass
        else:
            remaining.append(r)
    st.session_state.recordings = remaining
    persist_recordings_index(st.session_state.recordings)


def format_timer(seconds: float) -> str:
    """Formats seconds as MM:SS.d like standard voice recorder apps."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    tenths = int((seconds - int(seconds)) * 10)
    return f"{mins:02d}:{secs:02d}.{tenths}"


# ----------------------------------------------------
# 2. Recognizable Voice Recorder CSS Styling
# ----------------------------------------------------
def inject_custom_css(theme_name: str):
    colors = get_theme_colors(theme_name)
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

    .stApp {{
        background-color: {colors["bg"]};
        color: {colors["text"]};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {colors["paper_bg"]};
        border-right: 1px solid {colors["grid"]};
    }}

    /* Recorder Studio Deck Card */
    .recorder-deck {{
        background: linear-gradient(145deg, {colors["paper_bg"]}, {colors["bg"]});
        border: 1px solid {colors["grid"]};
        border-radius: 20px;
        padding: 28px 24px 18px 24px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        margin-bottom: 16px;
    }}

    .timer-display {{
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 56px;
        font-weight: 700;
        letter-spacing: 2px;
        color: {colors["text"]};
        margin: 8px 0 4px 0;
        line-height: 1.1;
    }}

    .status-pill-rec {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 59, 48, 0.16);
        border: 1px solid #FF3B30;
        color: #FF6961;
        padding: 5px 16px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }}

    .rec-dot {{
        width: 10px;
        height: 10px;
        background-color: #FF3B30;
        border-radius: 50%;
        display: inline-block;
        animation: pulse-red 1s infinite;
    }}

    .status-pill-paused {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 159, 10, 0.16);
        border: 1px solid #FF9F0A;
        color: #FFB340;
        padding: 5px 16px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }}

    .status-pill-ready {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(142, 142, 147, 0.14);
        border: 1px solid {colors["grid"]};
        color: {colors["text"]};
        padding: 5px 16px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 0.5px;
    }}

    /* Recording List Item Card */
    .memo-card {{
        background-color: {colors["paper_bg"]};
        border: 1px solid {colors["grid"]};
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }}

    .memo-title {{
        font-size: 18px;
        font-weight: 600;
        color: {colors["text"]};
        margin-bottom: 2px;
    }}

    .memo-meta {{
        font-size: 13px;
        color: #8E8E93;
        margin-bottom: 8px;
    }}

    .memo-badge {{
        display: inline-block;
        padding: 2px 9px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        background: rgba(0, 240, 255, 0.12);
        color: {colors["signal"]};
        border: 1px solid {colors["grid"]};
        margin-left: 6px;
    }}

    @keyframes pulse-red {{
        0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 59, 48, 0.7); }}
        70% {{ transform: scale(1.05); box-shadow: 0 0 0 8px rgba(255, 59, 48, 0); }}
        100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 59, 48, 0); }}
    }}

    div.stMetric, div[data-testid="stCard"] {{
        background-color: {colors["paper_bg"]};
        border: 1px solid {colors["grid"]};
        border-radius: 12px;
        padding: 14px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ----------------------------------------------------
# 3. Session State Initialization
# ----------------------------------------------------
if "reader" not in st.session_state:
    ports = SerialAudioReader.list_available_ports()
    # Automatically prefer a real hardware port (e.g. COM11) if detected and free
    hw_ports = [p for p in ports if p != "MOCK" and "COM6" not in p and "COM7" not in p]
    initial_port = hw_ports[0] if hw_ports else "MOCK"
    st.session_state.reader = SerialAudioReader(port=initial_port)
    st.session_state.reader.start()

if "recordings" not in st.session_state:
    st.session_state.recordings = load_saved_recordings()

if "next_memo_name" not in st.session_state:
    st.session_state.next_memo_name = f"Voice Memo {len(st.session_state.recordings) + 1}"

if "selected_recording_id" not in st.session_state:
    st.session_state.selected_recording_id = None

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
# 4. Sidebar Controls & Pinout Reference
# ----------------------------------------------------
st.sidebar.markdown("## 🎙️ Voice Recorder Studio")

app_mode = st.sidebar.radio(
    "Application Mode",
    ["🎙️ Voice Recorder", "🔬 Studio Signal Analyzer"],
    index=0,
    help="Switch between the clean Voice Recorder interface and the full engineering DSP analyzer."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔌 Microphone Source")
available_ports = SerialAudioReader.list_available_ports()

if st.sidebar.button("🔄 Scan Serial Ports", width="stretch"):
    st.rerun()

current_port_idx = (
    available_ports.index(st.session_state.reader.port)
    if st.session_state.reader.port in available_ports
    else 0
)
selected_port = st.sidebar.selectbox("Input Device / Port", available_ports, index=current_port_idx)

if selected_port != st.session_state.reader.port:
    st.session_state.reader.set_port(selected_port)
    fft_size = config_instance.dsp["default_fft_size"]
    st.session_state.history_mags = np.zeros((50, fft_size // 2))
    st.rerun()

# Quality / Sample Rate & Gain
sample_rate = st.sidebar.selectbox(
    "Recording Sample Rate (Hz)",
    config_instance.audio["sample_rates"],
    index=config_instance.audio["sample_rates"].index(st.session_state.reader.sample_rate)
    if st.session_state.reader.sample_rate in config_instance.audio["sample_rates"]
    else 3
)

if sample_rate != st.session_state.reader.sample_rate:
    st.session_state.reader.send_sample_rate(sample_rate)

hardware_gain = st.sidebar.slider("Microphone Gain Boost", 0.5, 15.0, 2.5, step=0.5)
st.session_state.reader.send_gain(hardware_gain)

# Dynamic I2S Pinout Configuration & Quick Reference
with st.sidebar.expander("📌 INMP441 ↔ ESP32-C3 Pinout & Pins", expanded=False):
    st.markdown(
        """
        | INMP441 Pin | ESP32-C3 Pin | Function |
        | :--- | :--- | :--- |
        | **VDD** | **3.3V** | Power (Do NOT use 5V) |
        | **GND** | **GND** | Common Ground |
        | **L/R** | **GND** (or 3.3V) | Channel Select |
        | **SCK** | **GPIO 2** | Serial Clock (BCLK) |
        | **WS** | **GPIO 3** | Word Select (LRCK) |
        | **SD** | **GPIO 4** | Serial Data (DOUT) |
        """
    )
    st.caption("Wired to different GPIOs? Update them live below:")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        pin_sck = st.number_input("SCK", min_value=0, max_value=21, value=2, step=1)
    with p_col2:
        pin_ws = st.number_input("WS", min_value=0, max_value=21, value=3, step=1)
    with p_col3:
        pin_sd = st.number_input("SD", min_value=0, max_value=21, value=4, step=1)
    if st.button("Apply GPIO Pins to ESP32", width="stretch"):
        st.session_state.reader.send_pins(int(pin_sck), int(pin_ws), int(pin_sd))
        st.toast(f"Updated ESP32 I2S pins: SCK={pin_sck}, WS={pin_ws}, SD={pin_sd}")

st.sidebar.markdown("---")
theme_name = st.sidebar.selectbox("Theme", config_instance.ui["themes"], index=3)  # Default 'Dark'
inject_custom_css(theme_name)
colors = get_theme_colors(theme_name)

live_stream = st.sidebar.checkbox("Live Waveform Monitor", value=True)
refresh_rate_ms = st.sidebar.slider("UI Refresh Interval (ms)", 80, 500, 150, step=10)

stats = st.session_state.reader.get_stats()
if stats.get("connection_error"):
    st.sidebar.error(f"❌ Port Error:\n{stats['connection_error']}")


# ----------------------------------------------------
# 5. Helper Plotters for Voice Recorder UI
# ----------------------------------------------------
def plot_voice_recorder_bars(samples: np.ndarray, is_recording: bool, theme_colors: Dict[str, str]) -> go.Figure:
    """Renders a sleek scrolling bar waveform like Apple Voice Memos / Google Recorder."""
    bars = AudioUtils.compute_waveform_preview(samples, num_points=72)
    # Mirror bars symmetrically (+ and -) for classic voice recorder look
    x_idx = np.arange(len(bars))
    bar_color = "#FF3B30" if is_recording else theme_colors["signal"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_idx,
            y=bars,
            marker_color=bar_color,
            marker_line_width=0,
            opacity=0.92,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Bar(
            x=x_idx,
            y=[-b for b in bars],
            marker_color=bar_color,
            marker_line_width=0,
            opacity=0.45,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        barmode="overlay",
        bargap=0.35,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=6, b=6),
        height=175,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[-1.08, 1.08]),
        showlegend=False,
    )
    return fig


def plot_saved_memo_sparkline(preview_bars: List[float], theme_colors: Dict[str, str]) -> go.Figure:
    """Renders a compact waveform overview for a saved recording card."""
    bars = preview_bars if preview_bars else [0.05] * 60
    x_idx = np.arange(len(bars))
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_idx,
            y=bars,
            marker_color=theme_colors["signal"],
            marker_line_width=0,
            opacity=0.85,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Bar(
            x=x_idx,
            y=[-b * 0.7 for b in bars],
            marker_color=theme_colors["secondary"],
            marker_line_width=0,
            opacity=0.45,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        barmode="overlay",
        bargap=0.3,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=2, b=2),
        height=68,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[-1.05, 1.05]),
        showlegend=False,
    )
    return fig


# ====================================================
# MODE A: VOICE RECORDER APP (DEFAULT)
# ====================================================
if app_mode == "🎙️ Voice Recorder":
    rec_status = st.session_state.reader.get_recording_status()
    is_rec = rec_status["is_recording"]
    is_rec_paused = rec_status["is_paused"]
    rec_duration = rec_status["duration_sec"]

    # Pull latest 1.5 seconds of audio for live waveform display
    live_window_samples = st.session_state.reader.get_audio_data(int(sample_rate * 1.5))
    live_summary = AudioUtils.compute_audio_summary(live_window_samples[-2048:], sample_rate)
    live_peak_abs = int(np.max(np.abs(live_window_samples[-2048:]))) if len(live_window_samples) > 0 else 0

    # Header Row with Source Status Badge
    hdr_col1, hdr_col2 = st.columns([2.2, 1])
    with hdr_col1:
        st.markdown("## 🎙️ Voice Memos & Recorder")
    with hdr_col2:
        conn_badge = (
            f"🟢 Hardware Mic ({stats['port']}) • {sample_rate // 1000:.1f} kHz"
            if stats["connected"] and stats["port"] != "MOCK"
            else f"🔵 Demo Synth ({stats['port']}) • {sample_rate // 1000:.1f} kHz"
        )
        st.markdown(
            f"<div style='text-align:right; padding-top:12px; font-size:13px; opacity:0.85;'>{conn_badge} | {stats['fps']:.0f} pkt/s</div>",
            unsafe_allow_html=True,
        )

    # Hardware zero-signal hint if connected to COM port but receiving digital silence
    if stats["connected"] and stats["port"] != "MOCK" and stats["packet_count"] > 20 and live_peak_abs == 0:
        st.warning(
            "⚠️ **ESP32 is connected and streaming packets at 44.1 kHz, but the microphone data line (SD) is reading 0.** "
            "Please verify your INMP441 jumper wires: **VDD → 3.3V**, **GND → GND**, **L/R → GND**, **SCK → GPIO 2**, **WS → GPIO 3**, **SD → GPIO 4** "
            "(or adjust the GPIO numbers in the sidebar under *INMP441 ↔ ESP32-C3 Pinout*)."
        )

    # ----------------------------------------------------
    # TOP SECTION: RECOGNIZABLE VOICE RECORDER STUDIO DECK
    # ----------------------------------------------------
    if is_rec and not is_rec_paused:
        pill_html = '<span class="status-pill-rec"><span class="rec-dot"></span> RECORDING LIVE</span>'
    elif is_rec and is_rec_paused:
        pill_html = '<span class="status-pill-paused">⏸ RECORDING PAUSED</span>'
    else:
        pill_html = '<span class="status-pill-ready">● READY TO RECORD</span>'

    timer_str = format_timer(rec_duration if is_rec else 0.0)

    st.markdown(
        f"""
        <div class="recorder-deck">
            <div>{pill_html}</div>
            <div class="timer-display">{timer_str}</div>
            <div style="font-size: 13px; color: #8E8E93; margin-bottom: 4px;">
                Input Level: <b>{live_summary['rms_dbfs']:.1f} dBFS</b> &nbsp;•&nbsp;
                Peak: <b>{live_summary['peak_dbfs']:.1f} dBFS</b> &nbsp;•&nbsp;
                Format: <b>16-bit PCM WAV ({sample_rate} Hz)</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Live Scrolling Voice Memos Waveform Bar Chart
    fig_live_bars = plot_voice_recorder_bars(live_window_samples, is_rec and not is_rec_paused, colors)
    st.plotly_chart(fig_live_bars, width="stretch", config={"displayModeBar": False})

    # Transport Control Bar (Record / Pause / Stop & Save / Quick Record)
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.6, 1.1, 1.1, 1.2])

    with ctrl_col1:
        memo_title_input = st.text_input(
            "Recording Name",
            value=st.session_state.next_memo_name,
            label_visibility="collapsed",
            placeholder="Enter recording name (e.g. Voice Memo 1)...",
        )

    if not is_rec:
        with ctrl_col2:
            if st.button("🔴 Start Recording", type="primary", width="stretch"):
                st.session_state.reader.start_recording()
                st.rerun()
        with ctrl_col3:
            if st.button("⚡ Quick 3s Clip", width="stretch", help="Instantly saves the last 3 seconds from buffer"):
                quick_samples = st.session_state.reader.get_audio_data(int(sample_rate * 3))
                entry = save_new_recording(
                    quick_samples,
                    sample_rate,
                    memo_title_input or f"Voice Memo {len(st.session_state.recordings) + 1}",
                    stats["port"],
                )
                st.session_state.selected_recording_id = entry["id"]
                st.session_state.next_memo_name = f"Voice Memo {len(st.session_state.recordings) + 1}"
                st.toast(f"Saved '{entry['title']}' ({entry['duration_sec']:.1f}s)!")
                st.rerun()
        with ctrl_col4:
            if st.button(
                "▶️ Resume Live Monitor" if st.session_state.is_paused else "⏸️ Pause Monitor",
                width="stretch",
            ):
                st.session_state.is_paused = not st.session_state.is_paused
                st.rerun()
    else:
        with ctrl_col2:
            if st.button("⏹️ Stop & Save", type="primary", width="stretch"):
                samples, sr, dur = st.session_state.reader.stop_recording()
                if len(samples) == 0:
                    # Fallback to latest buffer if stopped immediately
                    samples = st.session_state.reader.get_audio_data(int(sample_rate * 1.0))
                entry = save_new_recording(
                    samples,
                    sr,
                    memo_title_input or f"Voice Memo {len(st.session_state.recordings) + 1}",
                    stats["port"],
                )
                st.session_state.selected_recording_id = entry["id"]
                st.session_state.next_memo_name = f"Voice Memo {len(st.session_state.recordings) + 1}"
                st.toast(f"✅ Saved '{entry['title']}' ({entry['duration_sec']:.1f}s)")
                st.rerun()
        with ctrl_col3:
            if is_rec_paused:
                if st.button("▶️ Resume Rec", width="stretch"):
                    st.session_state.reader.resume_recording()
                    st.rerun()
            else:
                if st.button("⏸️ Pause Rec", width="stretch"):
                    st.session_state.reader.pause_recording()
                    st.rerun()
        with ctrl_col4:
            if st.button("🗑️ Discard", width="stretch"):
                st.session_state.reader.cancel_recording()
                st.toast("Recording discarded.")
                st.rerun()

    st.markdown("---")

    # ----------------------------------------------------
    # BOTTOM SECTION: RECORDED AUDIO LIBRARY LIST & REPLAY
    # ----------------------------------------------------
    lib_hdr1, lib_hdr2 = st.columns([3, 1])
    with lib_hdr1:
        st.markdown(f"### 📂 All Recordings ({len(st.session_state.recordings)})")
    with lib_hdr2:
        search_query = st.text_input(
            "Search recordings",
            value="",
            label_visibility="collapsed",
            placeholder="🔍 Search recordings...",
        )

    if not st.session_state.recordings:
        st.info(
            "No voice recordings yet. Click **🔴 Start Recording** (then **⏹️ Stop & Save**) or **⚡ Quick 3s Clip** above to record your first voice memo!"
        )
    else:
        filtered_recs = [
            r for r in st.session_state.recordings
            if search_query.lower() in r.get("title", "").lower()
            or search_query.lower() in r.get("created_at", "").lower()
        ]

        for idx, rec in enumerate(filtered_recs):
            with st.container():
                st.markdown(
                    f"""
                    <div class="memo-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                            <div>
                                <span class="memo-title">🎵 {rec['title']}</span>
                                <span class="memo-badge">{format_timer(rec['duration_sec'])}</span>
                                <span class="memo-badge">{rec['sample_rate'] // 1000:.1f} kHz</span>
                                <span class="memo-badge">{rec.get('source', 'MIC')}</span>
                            </div>
                            <div class="memo-meta">
                                {rec['created_at']} &nbsp;•&nbsp; Peak: {rec.get('peak_dbfs', 0.0):.1f} dBFS &nbsp;•&nbsp; RMS: {rec.get('rms_dbfs', 0.0):.1f} dBFS
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                row_col1, row_col2, row_col3 = st.columns([2.2, 2.2, 1.0])
                with row_col1:
                    # Replay recorded audio right in the list
                    if rec.get("wav_bytes"):
                        st.audio(rec["wav_bytes"], format="audio/wav")
                with row_col2:
                    fig_spark = plot_saved_memo_sparkline(rec.get("preview_bars", []), colors)
                    st.plotly_chart(
                        fig_spark,
                        width="stretch",
                        config={"displayModeBar": False},
                        key=f"spark_{rec['id']}_{idx}",
                    )
                with row_col3:
                    btn_c1, btn_c2 = st.columns(2)
                    with btn_c1:
                        if rec.get("wav_bytes"):
                            st.download_button(
                                label="💾 WAV",
                                data=rec["wav_bytes"],
                                file_name=f"{rec['title'].replace(' ', '_')}.wav",
                                mime="audio/wav",
                                key=f"dl_{rec['id']}_{idx}",
                                width="stretch",
                            )
                    with btn_c2:
                        if st.button("🗑️", key=f"del_{rec['id']}_{idx}", help="Delete recording", width="stretch"):
                            delete_recording(rec["id"])
                            st.rerun()

# ====================================================
# MODE B: FULL STUDIO SIGNAL ANALYZER (ORIGINAL TABS)
# ====================================================
else:
    st.sidebar.markdown("### 🗂️ Analyzer View")
    selected_tab = st.sidebar.radio(
        "Select View",
        [
            "📈 Live Audio",
            "⚡ Frequency Analysis",
            "🎛️ DSP Demonstrations",
            "📏 Audio Measurements",
            "🌌 Time-Frequency",
            "🗣️ Voice Activity",
            "🖥️ Device Status",
        ],
        index=0,
    )

    uses_fft = selected_tab in [
        "⚡ Frequency Analysis",
        "🎛️ DSP Demonstrations",
        "📏 Audio Measurements",
        "🌌 Time-Frequency",
    ]
    if uses_fft:
        st.sidebar.markdown("### 📊 FFT Settings")
        fft_size = st.sidebar.selectbox("FFT Window Size", config_instance.dsp["fft_sizes"], index=2)
        window_type = st.sidebar.selectbox("Window Function", config_instance.dsp["windows"], index=1)
        num_bins = fft_size // 2
        if st.session_state.history_mags.shape[1] != num_bins:
            st.session_state.history_mags = np.zeros((50, num_bins))
    else:
        fft_size = config_instance.dsp["default_fft_size"]
        window_type = config_instance.dsp["default_window"]

    if selected_tab == "📈 Live Audio":
        display_length = 1024
    elif selected_tab == "🖥️ Device Status":
        display_length = 256
    else:
        display_length = max(2048, fft_size)

    raw_samples = st.session_state.reader.get_audio_data(display_length)
    norm_samples = raw_samples.astype(np.float32) / 32768.0

    st.markdown(
        f"<h2 style='text-align: center; color: {colors['signal']};'>🔬 MEMS Microphone Studio Analyzer</h2>",
        unsafe_allow_html=True,
    )

    if selected_tab == "📈 Live Audio":
        metrics = AudioDSP.calculate_measurements(raw_samples, sample_rate, include_fft=False)
        st.subheader("Time-Domain Live Analysis")
        t1_col1, t1_col2 = st.columns([3, 1])
        with t1_col1:
            fig_wave = AudioVisualizer.plot_waveform(raw_samples[-1024:], sample_rate, theme_name)
            st.plotly_chart(fig_wave, width="stretch", config={"displayModeBar": False})
        with t1_col2:
            fig_rms = AudioVisualizer.plot_gauge(metrics.get("rms", 0.0), "RMS Amplitude", 0, 32768, "PCM", theme_name)
            st.plotly_chart(fig_rms, width="stretch", config={"displayModeBar": False})
            fig_peak = AudioVisualizer.plot_gauge(metrics.get("peak", 0.0), "Peak Amplitude", 0, 32768, "PCM", theme_name)
            st.plotly_chart(fig_peak, width="stretch", config={"displayModeBar": False})

    elif selected_tab == "⚡ Frequency Analysis":
        freqs, mags = AudioDSP.compute_fft(raw_samples, window_name=window_type, fft_size=fft_size)
        freqs_hz = freqs * sample_rate
        st.session_state.history_mags = np.roll(st.session_state.history_mags, -1, axis=0)
        st.session_state.history_mags[-1, :] = mags
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
                theme_name,
            )
            st.plotly_chart(fig_fft, width="stretch", config={"displayModeBar": False})
        with t2_col2:
            fig_spec = AudioVisualizer.plot_spectrogram(st.session_state.history_mags, freqs_hz, theme_name)
            st.plotly_chart(fig_spec, width="stretch", config={"displayModeBar": False})

        st.markdown("---")
        enable_waterfall = st.checkbox("💻 Enable Heavy 3D Waterfall", value=False)
        if enable_waterfall:
            fig_water = AudioVisualizer.plot_waterfall(st.session_state.history_mags, freqs_hz, theme_name)
            st.plotly_chart(fig_water, width="stretch", config={"displayModeBar": False})

    elif selected_tab == "🎛️ DSP Demonstrations":
        st.subheader("Cascade Digital Signal Processing Blocks")
        dsp_ctrl1, dsp_ctrl2, dsp_ctrl3, dsp_ctrl4 = st.columns(4)
        with dsp_ctrl1:
            dsp_gain = st.slider("Post-Gain (dB)", -20.0, 20.0, 0.0, step=0.5)
            enable_lpf = st.checkbox("Low-Pass Filter (LPF)")
            lpf_cutoff = st.slider("LPF Cutoff (Hz)", 100, int(sample_rate / 2 - 100), 2000)
        with dsp_ctrl2:
            enable_hpf = st.checkbox("High-Pass Filter (HPF)")
            hpf_cutoff = st.slider("HPF Cutoff (Hz)", 20, 5000, 150)
            enable_bpf = st.checkbox("Band-Pass Filter (BPF)")
            bpf_center = st.slider("BPF Center (Hz)", 100, int(sample_rate / 2 - 200), 1000)
            bpf_width = st.slider("BPF Width (Hz)", 50, 2000, 500)
        with dsp_ctrl3:
            enable_bsf = st.checkbox("Band-Stop Filter (BSF)")
            bsf_center = st.slider("BSF Center (Hz)", 50, 5000, 60)
            bsf_width = st.slider("BSF Width (Hz)", 10, 500, 20)
            enable_mov_avg = st.checkbox("Moving Average")
            mov_avg_len = st.slider("MA Window Size", 2, 50, 8)
        with dsp_ctrl4:
            enable_median = st.checkbox("Median Filter")
            median_len = st.slider("Median Window Size", 3, 31, 5, step=2)
            enable_drc = st.checkbox("Dynamic Range Compressor")
            drc_threshold = st.slider("DRC Threshold (dB)", -40, -5, -20)
            enable_noise_gate = st.checkbox("Spectral Noise Gate")

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
            sos = AudioDSP.design_iir_filter("butter_bandpass", bpf_center, sample_rate, stop_hz=bpf_center + bpf_width)
            filtered_samples = AudioDSP.apply_iir_filter(filtered_samples, sos)
        if enable_bsf:
            sos = AudioDSP.design_iir_filter("butter_bandstop", bsf_center, sample_rate, stop_hz=bsf_center + bsf_width)
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
            fig_orig.update_layout(title="Original Signal", height=220)
            st.plotly_chart(fig_orig, width="stretch", config={"displayModeBar": False})
        with t3_col2:
            fig_filt = AudioVisualizer.plot_waveform(filtered_samples[-512:], sample_rate, theme_name)
            fig_filt.update_layout(title="Filtered Signal", height=220)
            fig_filt.update_traces(line=dict(color=colors["secondary"]))
            st.plotly_chart(fig_filt, width="stretch", config={"displayModeBar": False})

    elif selected_tab == "📏 Audio Measurements":
        st.subheader("Quantitative Audio Metrology")
        metrics = AudioDSP.calculate_measurements(raw_samples, sample_rate, include_fft=True)
        measurement_data = {
            "Metric": [
                "RMS Level (PCM)", "Peak Level (PCM)", "Peak-to-Peak",
                "Mean Offset", "Standard Deviation", "Crest Factor",
                "Dynamic Range (dB)", "Zero Crossing Rate (ZCR)", "Signal Energy",
                "Estimated SNR (dB)", "Estimated THD (%)", "Dominant Frequency (Hz)",
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
                f"{metrics.get('dominant_freq', 0.0):.1f} Hz",
            ],
        }
        st.table(pd.DataFrame(measurement_data))

    elif selected_tab == "🌌 Time-Frequency":
        st.subheader("Time-Frequency & Cepstral Features")
        t5_col1, t5_col2 = st.columns(2)
        with t5_col1:
            stft_nperseg = min(256, fft_size)
            f_stft, t_stft, Zxx = signal.stft(
                norm_samples, fs=sample_rate, nperseg=stft_nperseg, noverlap=stft_nperseg // 2
            )
            fig_stft = go.Figure(
                data=go.Heatmap(
                    x=t_stft,
                    y=f_stft,
                    z=20.0 * np.log10(np.maximum(np.abs(Zxx), 1e-5)),
                    colorscale=colors["colorscale"],
                    zmin=-80,
                    zmax=-10,
                    showscale=False,
                )
            )
            AudioVisualizer._apply_layout(fig_stft, colors, "STFT Spectrogram", "Time (sec)", "Frequency (Hz)")
            fig_stft.update_layout(height=260)
            st.plotly_chart(fig_stft, width="stretch", config={"displayModeBar": False})
        with t5_col2:
            mfcc = AudioDSP.compute_mfcc(raw_samples, sample_rate, n_fft=fft_size, n_mels=40, n_mfcc=13)
            fig_mfcc = go.Figure(data=go.Bar(x=np.arange(13), y=mfcc, marker_color=colors["accent"]))
            AudioVisualizer._apply_layout(fig_mfcc, colors, "MFCCs (13 Coefficients)", "Index", "Amplitude")
            fig_mfcc.update_layout(height=260)
            st.plotly_chart(fig_mfcc, width="stretch", config={"displayModeBar": False})

    elif selected_tab == "🗣️ Voice Activity":
        st.subheader("Speech & Voice Activity Detection (VAD)")
        st.session_state.vad_state = AudioDSP.run_vad(
            raw_samples[-256:],
            st.session_state.vad_state,
            energy_threshold=config_instance.dsp["vad"]["energy_threshold"],
            zcr_threshold=config_instance.dsp["vad"]["zcr_threshold"],
            smoothing_factor=config_instance.dsp["vad"]["smoothing_factor"],
            hangover_frames=config_instance.dsp["vad"]["hangover_frames"],
        )
        vad = st.session_state.vad_state
        st.metric("Voice Probability", f"{vad.get('voice_probability', 0.0) * 100:.1f}%")
        st.progress(float(vad.get("voice_probability", 0.0)))

    elif selected_tab == "🖥️ Device Status":
        st.subheader("ESP32 Serial & Buffer Diagnostics")
        st.json(stats)


# ----------------------------------------------------
# 6. Auto-Refresh Loop for Live Monitor / Active Recording
# ----------------------------------------------------
is_actively_recording = st.session_state.reader.get_recording_status()["is_recording"]
if (live_stream and not st.session_state.is_paused) or is_actively_recording:
    time.sleep(refresh_rate_ms / 1000.0)
    st.rerun()
