import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any, List, Tuple
from dashboard.config import config_instance

def get_theme_colors(theme_name: str) -> Dict[str, str]:
    """Returns a dictionary of theme hex codes based on the selected theme."""
    themes = {
        "Cyberpunk": {
            "bg": "#0B0C10",
            "paper_bg": "#12131C",
            "grid": "#1F2833",
            "text": "#C5C6C7",
            "signal": "#00F0FF",      # Neon Cyan
            "secondary": "#FF007F",   # Neon Pink
            "accent": "#FFFF00",      # Neon Yellow
            "colorscale": "Electric"
        },
        "Ocean": {
            "bg": "#0A192F",
            "paper_bg": "#172A45",
            "grid": "#233554",
            "text": "#8892B0",
            "signal": "#64FFDA",      # Turquoise
            "secondary": "#30A2FF",   # Ocean Blue
            "accent": "#F2F597",      # Sand Yellow
            "colorscale": "YlGnBu"
        },
        "Emerald": {
            "bg": "#0F1A15",
            "paper_bg": "#1C2E24",
            "grid": "#2A4535",
            "text": "#A3B899",
            "signal": "#00E676",      # Emerald Green
            "secondary": "#AEEA00",   # Lime Green
            "accent": "#FFAB00",      # Amber
            "colorscale": "Algae"
        },
        "Dark": {
            "bg": "#121212",
            "paper_bg": "#1E1E1E",
            "grid": "#2D2D2D",
            "text": "#E0E0E0",
            "signal": "#BB86FC",      # Pastel Purple
            "secondary": "#03DAC6",   # Teal
            "accent": "#CF6679",      # Coral
            "colorscale": "Inferno"
        },
        "Light": {
            "bg": "#F5F5F5",
            "paper_bg": "#FFFFFF",
            "grid": "#E0E0E0",
            "text": "#333333",
            "signal": "#1976D2",      # Blue
            "secondary": "#388E3C",   # Green
            "accent": "#D32F2F",      # Red
            "colorscale": "Viridis"
        }
    }
    return themes.get(theme_name, themes["Cyberpunk"])

class AudioVisualizer:
    """Helper class containing custom Plotly graphs for real-time visualization."""
    
    @staticmethod
    def _apply_layout(fig: go.Figure, colors: Dict[str, str], title: str, x_title: str, y_title: str) -> None:
        """Applies consistent styling, fonts, backgrounds, and borders to a Plotly figure."""
        fig.update_layout(
            title={
                "text": title,
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": {"size": 16, "color": colors["text"]}
            },
            xaxis={
                "title": x_title,
                "gridcolor": colors["grid"],
                "color": colors["text"],
                "zerolinecolor": colors["grid"]
            },
            yaxis={
                "title": y_title,
                "gridcolor": colors["grid"],
                "color": colors["text"],
                "zerolinecolor": colors["grid"]
            },
            plot_bgcolor=colors["bg"],
            paper_bgcolor=colors["paper_bg"],
            font={"family": "Courier New, monospace", "color": colors["text"]},
            margin={"l": 50, "r": 30, "t": 60, "b": 50},
            showlegend=False,
            height=320
        )

    @classmethod
    def plot_waveform(cls, x: np.ndarray, fs: float, theme_name: str) -> go.Figure:
        """Plots the real-time time-domain waveform."""
        colors = get_theme_colors(theme_name)
        time_axis = np.arange(len(x)) / fs
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=x,
            mode='lines',
            line=dict(color=colors["signal"], width=1.5),
            name="Waveform"
        ))
        
        cls._apply_layout(fig, colors, "Live Time-Domain Waveform", "Time (seconds)", "Amplitude (16-bit PCM)")
        # Auto scale limits
        fig.update_yaxes(range=[-32768, 32767])
        return fig

    @classmethod
    def plot_fft(
        cls, 
        freqs: np.ndarray, 
        mags: np.ndarray, 
        dominant_freq: float,
        harmonics: List[Tuple[float, float]],
        log_scale: bool,
        theme_name: str
    ) -> go.Figure:
        """Plots the frequency spectrum with optional log scaling and peak harmonic markers."""
        colors = get_theme_colors(theme_name)
        
        display_mags = mags.copy()
        y_label = "Amplitude (Linear)"
        
        if log_scale:
            # Convert to decibels relative to max 16-bit PCM
            display_mags = 20.0 * np.log10(np.maximum(display_mags, 1e-5))
            y_label = "Amplitude (dB)"
            
        fig = go.Figure()
        
        # Plot spectrum line
        fig.add_trace(go.Scatter(
            x=freqs,
            y=display_mags,
            mode='lines',
            line=dict(color=colors["signal"], width=1.5),
            name="Spectrum"
        ))
        
        # Plot dominant frequency marker
        if len(freqs) > 0 and dominant_freq > 0:
            # Find the bin closest to dominant frequency
            dom_idx = np.argmin(np.abs(freqs - dominant_freq))
            dom_y = display_mags[dom_idx]
            
            fig.add_trace(go.Scatter(
                x=[dominant_freq],
                y=[dom_y],
                mode='markers+text',
                marker=dict(color=colors["accent"], size=10, symbol='diamond'),
                text=[f"F0: {dominant_freq:.1f} Hz"],
                textposition="top right",
                name="Dominant"
            ))
            
            # Plot harmonics markers
            harmonic_x = []
            harmonic_y = []
            harmonic_text = []
            for i, (h_freq, h_mag) in enumerate(harmonics):
                h_y = 20.0 * np.log10(np.maximum(h_mag, 1e-5)) if log_scale else h_mag
                harmonic_x.append(h_freq)
                harmonic_y.append(h_y)
                harmonic_text.append(f"H{i+2}")
                
            if harmonic_x:
                fig.add_trace(go.Scatter(
                    x=harmonic_x,
                    y=harmonic_y,
                    mode='markers+text',
                    marker=dict(color=colors["secondary"], size=7, symbol='x'),
                    text=harmonic_text,
                    textposition="bottom center",
                    name="Harmonics"
                ))

        cls._apply_layout(fig, colors, "FFT Magnitude Spectrum", "Frequency (Hz)", y_label)
        if log_scale:
            fig.update_yaxes(range=[-100, 10]) # Standard decibel range for relative PCM
        else:
            fig.update_yaxes(range=[0, np.max(display_mags) * 1.1 + 10])
            
        return fig

    @classmethod
    def plot_spectrogram(
        cls, 
        history_mags: np.ndarray, 
        freqs: np.ndarray, 
        theme_name: str
    ) -> go.Figure:
        """
        Plots the running 2D spectrogram history.
        history_mags shape: (num_history_steps, num_freq_bins)
        """
        colors = get_theme_colors(theme_name)
        
        # Set up dimensions
        steps = np.arange(history_mags.shape[0])
        
        # Convert magnitude to dB
        db_mags = 20.0 * np.log10(np.maximum(history_mags, 1e-5))
        
        fig = go.Figure(data=go.Heatmap(
            z=db_mags.T,
            x=steps,
            y=freqs,
            colorscale=colors["colorscale"],
            zmin=-90,
            zmax=0,
            colorbar=dict(title="dB", thickness=15, len=0.8)
        ))
        
        cls._apply_layout(fig, colors, "Real-Time Spectrogram", "Time Frames (Rolling)", "Frequency (Hz)")
        fig.update_layout(height=360)
        return fig

    @classmethod
    def plot_waterfall(
        cls, 
        history_mags: np.ndarray, 
        freqs: np.ndarray, 
        theme_name: str
    ) -> go.Figure:
        """
        Plots a 3D Waterfall surface representation of the spectral history.
        history_mags shape: (num_history_steps, num_freq_bins)
        """
        colors = get_theme_colors(theme_name)
        
        # Limit dimensions for 3D layout performance
        # We downsample the data so that Plotly doesn't freeze the browser
        downsample_freq_factor = max(1, len(freqs) // 64)
        downsample_time_factor = max(1, history_mags.shape[0] // 30)
        
        z = 20.0 * np.log10(np.maximum(history_mags[::downsample_time_factor, ::downsample_freq_factor], 1e-5))
        y_freqs = freqs[::downsample_freq_factor]
        x_frames = np.arange(z.shape[0])
        
        # Meshgrid creation for surface plot
        fig = go.Figure(data=[go.Surface(
            z=z,
            x=x_frames,
            y=y_freqs,
            colorscale=colors["colorscale"],
            cmin=-90,
            cmax=0,
            showscale=False
        )])
        
        fig.update_layout(
            title={
                "text": "3D Waterfall Plot (Frequency Evolution)",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": {"size": 16, "color": colors["text"]}
            },
            scene={
                "xaxis": {"title": "Frame History", "gridcolor": colors["grid"], "color": colors["text"], "backgroundcolor": colors["bg"]},
                "yaxis": {"title": "Freq (Hz)", "gridcolor": colors["grid"], "color": colors["text"], "backgroundcolor": colors["bg"]},
                "zaxis": {"title": "dB Power", "gridcolor": colors["grid"], "color": colors["text"], "backgroundcolor": colors["bg"], "range": [-90, 0]},
                "camera": {"eye": {"x": 1.5, "y": -1.5, "z": 1.2}}
            },
            plot_bgcolor=colors["bg"],
            paper_bgcolor=colors["paper_bg"],
            font={"family": "Courier New, monospace", "color": colors["text"]},
            margin={"l": 10, "r": 10, "t": 50, "b": 10},
            height=380
        )
        return fig

    @classmethod
    def plot_gauge(
        cls, 
        value: float, 
        name: str, 
        min_val: float, 
        max_val: float, 
        unit: str, 
        theme_name: str
    ) -> go.Figure:
        """Plots a circular gauge (useful for RMS or Peak levels)."""
        colors = get_theme_colors(theme_name)
        
        # Calculate thresholds for green/orange/red warnings
        span = max_val - min_val
        warn_th = min_val + span * 0.7
        crit_th = min_val + span * 0.9
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': f"{name} ({unit})", 'font': {'size': 14, 'color': colors["text"]}},
            gauge={
                'axis': {'range': [min_val, max_val], 'tickcolor': colors["text"]},
                'bar': {'color': colors["signal"]},
                'bgcolor': colors["bg"],
                'borderwidth': 1,
                'bordercolor': colors["grid"],
                'steps': [
                    {'range': [min_val, warn_th], 'color': 'rgba(0,0,0,0)'},
                    {'range': [warn_th, crit_th], 'color': 'rgba(255, 171, 0, 0.4)'},
                    {'range': [crit_th, max_val], 'color': 'rgba(211, 47, 47, 0.4)'}
                ],
            }
        ))
        
        fig.update_layout(
            paper_bgcolor=colors["paper_bg"],
            font={"family": "Courier New, monospace", "color": colors["text"]},
            margin={"l": 25, "r": 25, "t": 40, "b": 20},
            height=180
        )
        return fig
