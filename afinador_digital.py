import streamlit as st
import numpy as np
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
import queue
import io
from scipy.io.wavfile import write as wav_write, read as wav_read

# Intentar importar pyaudio para grabar
try:
    import pyaudio
    pyaudio_available = True
except Exception:
    pyaudio_available = False

# Configuración
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
AMPLITUDE_THRESHOLD = 20  # Umbral para filtrar ruido
FREQUENCY_RANGE = (0, 2000)

audio_queue = queue.Queue()

# Estado de sesión
if 'stream' not in st.session_state:
    st.session_state.stream = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'mic_active' not in st.session_state:
    st.session_state.mic_active = False
if 'last_freq' not in st.session_state:
    st.session_state.last_freq = 0
if 'last_yf' not in st.session_state:
    st.session_state.last_yf = np.array([])
if 'last_xf' not in st.session_state:
    st.session_state.last_xf = np.array([])
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = "Esperando sonido..."
if 'last_sound_type' not in st.session_state:
    st.session_state.last_sound_type = "N/A"
if 'last_voice_type' not in st.session_state:
    st.session_state.last_voice_type = None
if 'last_age_group' not in st.session_state:
    st.session_state.last_age_group = None
if 'recorded_chunks' not in st.session_state:
    st.session_state.recorded_chunks = []
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None

def audio_callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    audio_queue.put(audio_data.copy())
    st.session_state.recorded_chunks.append(audio_data.copy())
    return (None, pyaudio.paContinue)

def get_spectrum(data, sample_rate=SAMPLE_RATE):
    N = len(data)
    window = np.hamming(N)
    data = data * window
    yf = fft(data)
    xf = fftfreq(N, 1 / sample_rate)
    yf = np.abs(yf[:N // 2])
    xf = xf[:N // 2]
    return yf, xf

def get_dominant_freq(xf, yf, min_freq=50, max_freq=FREQUENCY_RANGE[1]):
    valid_idx = np.where((xf >= min_freq) & (xf <= max_freq))[0]
    if len(valid_idx) == 0:
        return 0
    yf_valid = yf[valid_idx]
    max_idx = np.argmax(yf_valid)
    return xf[valid_idx][max_idx]

def classify_sound(freq, yf, xf, max_amplitude):
    if max_amplitude < AMPLITUDE_THRESHOLD:
        return "Ruido", None, None

    if 85 <= freq <= 255:
        if 85 <= freq <= 180:
            voice_type = "Hombre"
        elif 165 <= freq <= 255:
            voice_type = "Mujer"
        else:
            voice_type = "Niño"

        variance = np.var(yf[xf > 50])
        age_group = "Niño" if variance < 500 and freq > 250 else "Joven" if variance < 2000 else "Adulto"

        harmonic_band = (xf > freq * 1.5) & (xf < freq * 4)
        harmonic_peaks = yf[harmonic_band]

        harmonic_energy = np.sum(harmonic_peaks)
        fundamental_energy = np.sum(yf[(xf >= freq - 5) & (xf <= freq + 5)])

        harmonic_ratio = harmonic_energy / fundamental_energy if fundamental_energy > 0 else 0

        if harmonic_ratio > 0.7:
            return "Música o instrumento", None, None

        return "Voz humana (probablemente hablando)", voice_type, age_group

    harmonic_count = np.sum(yf > (max_amplitude * 0.25))
    if harmonic_count > 5 and freq > 100:
        return "Música o instrumento", None, None

    return "Ruido ambiental o sonido no identificado", None, None

def analyze_sound(freq, yf, xf):
    max_amplitude = np.max(yf)
    if max_amplitude < AMPLITUDE_THRESHOLD:
        return "Esperando sonido...", "Esperando sonido...", None, None

    valid_idx = (xf >= FREQUENCY_RANGE[0]) & (xf <= FREQUENCY_RANGE[1])
    if not np.any(valid_idx):
        return "Sin análisis disponible", "Sin análisis disponible", None, None

    if freq > 0:
        sound_type, _, _ = classify_sound(freq, yf, xf, max_amplitude)
        description = f"Frecuencias detectadas: {sound_type}."
        voice_type = None
        age_group = None
        if "Voz humana" in sound_type:
            _, voice_type, age_group = classify_sound(freq, yf, xf, max_amplitude)
        return sound_type, description, voice_type, age_group

    return "Esperando sonido...", "Esperando sonido...", None, None

def plot_spectrum(yf, xf, dominant_freq):
    fig = go.Figure()
    max_y = max(yf) * 1.1 if len(yf) > 0 else 1

    fig.add_shape(type="rect", x0=FREQUENCY_RANGE[0], x1=85, y0=0, y1=max_y,
                  fillcolor="lightgray", opacity=0.3, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=85, x1=255, y0=0, y1=max_y,
                  fillcolor="lightgreen", opacity=0.3, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=255, x1=800, y0=0, y1=max_y,
                  fillcolor="lightblue", opacity=0.3, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=800, x1=FREQUENCY_RANGE[1], y0=0, y1=max_y,
                  fillcolor="lightyellow", opacity=0.3, layer="below", line_width=0)

    fig.add_trace(go.Scatter(x=xf, y=yf, mode='lines', name='Espectro', line=dict(color='#1e3a8a')))
    if dominant_freq > 0:
        fig.add_vline(x=dominant_freq, line=dict(color='#dc2626', dash='dash'), name='Frecuencia Dominante')

    fig.add_annotation(x=70, y=max_y*0.95, text="Ruido/Ambiente", showarrow=False, font=dict(color="gray"))
    fig.add_annotation(x=170, y=max_y*0.95, text="Voz Humana", showarrow=False, font=dict(color="green"))
    fig.add_annotation(x=530, y=max_y*0.95, text="Música/Instrumentos", showarrow=False, font=dict(color="blue"))
    fig.add_annotation(x=1200, y=max_y*0.95, text="Alta Frecuencia", showarrow=False, font=dict(color="orange"))

    fig.update_layout(
        template="plotly_white",
        title="Espectro de Frecuencias",
        xaxis_title="Frecuencia (Hz)",
        yaxis_title="Amplitud",
        xaxis=dict(range=FREQUENCY_RANGE),
        showlegend=True,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# Configurar página y CSS para texto negro en métricos
st.set_page_config(page_title="Analizador de Sonido", layout="wide")

st.markdown("""
<style>
div[data-testid="metric-container"] div[class*="value"] {
    color: black !important;
}
div[data-testid="metric-container"] div[class*="delta"] {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .main {background-color: #f0f4f8;}
    h1 {color: #1e3a8a; font-family: 'Arial', sans-serif;}
    .stMarkdown {font-family: 'Arial', sans-serif; color: #333;}
    </style>
""", unsafe_allow_html=True)

st.title("🎙️ Analizador de Frecuencias con Grabación y Carga de WAV")
st.markdown("Habla, reproduce un sonido o carga un archivo WAV para analizar.")

# Sección: Grabación y Micrófono
st.subheader("Grabar Audio desde el Micrófono")
if pyaudio_available:
    mic_button_label = "Detener Grabación y Micrófono" if st.session_state.mic_active else "Activar Micrófono y Comenzar Grabación"
    if st.button(mic_button_label):
        if st.session_state.mic_active:
            if st.session_state.stream:
                st.session_state.stream.stop_stream()
                st.session_state.stream.close()
                st.session_state.audio_interface.terminate()
            st.session_state.mic_active = False
            st.session_state.is_running = False

            if len(st.session_state.recorded_chunks) > 0:
                audio_data = np.concatenate(st.session_state.recorded_chunks, axis=0)
                audio_int16 = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
                wav_io = io.BytesIO()
                wav_write(wav_io, SAMPLE_RATE, audio_int16)
                wav_io.seek(0)
                st.session_state.audio_bytes = wav_io.read()
                st.success("Grabación finalizada y guardada")
            else:
                st.warning("No se grabó audio")

            st.session_state.recorded_chunks = []

        else:
            try:
                st.session_state.recorded_chunks = []
                st.session_state.audio_interface = pyaudio.PyAudio()
                st.session_state.stream = st.session_state.audio_interface.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=BUFFER_SIZE,
                    stream_callback=audio_callback
                )
                st.session_state.stream.start_stream()
                st.session_state.mic_active = True
                st.session_state.is_running = True
                st.session_state.audio_bytes = None
            except Exception as e:
                st.error(f"Error al iniciar captura de audio: {e}")
else:
    st.info("Grabación en vivo no disponible en Streamlit Cloud. Por favor, carga un archivo WAV para analizar.")

# Sección: Análisis en Vivo
if pyaudio_available and st.session_state.mic_active and st.session_state.is_running:
    live_container = st.container()
    with live_container:
        st.markdown("### Análisis del Sonido en Vivo")
        try:
            audio_data = audio_queue.get_nowait()
            yf, xf = get_spectrum(audio_data)
            max_amplitude = np.max(yf)

            freq = get_dominant_freq(xf, yf)

            if freq > 0 and max_amplitude > AMPLITUDE_THRESHOLD:
                sound_type, description, voice_type, age_group = analyze_sound(freq, yf, xf)
                st.session_state.last_freq = freq
                st.session_state.last_yf = yf
                st.session_state.last_xf = xf
                st.session_state.last_analysis = description
                st.session_state.last_sound_type = sound_type
                st.session_state.last_voice_type = voice_type
                st.session_state.last_age_group = age_group
            else:
                sound_type = st.session_state.last_sound_type
                description = st.session_state.last_analysis
                voice_type = st.session_state.last_voice_type
                age_group = st.session_state.last_age_group
                yf = st.session_state.last_yf
                xf = st.session_state.last_xf
                freq = st.session_state.last_freq

            fig = plot_spectrum(yf, xf, freq)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tipo de Sonido", sound_type, delta_color="off")
            with col2:
                st.metric("Frecuencia Dominante", f"{freq:.2f} Hz" if freq > 0 else "N/A")
            with col3:
                st.metric("Amplitud Máxima", f"{max_amplitude:.2f}" if freq > 0 else "N/A")

            if sound_type == "Voz humana (probablemente hablando)":
                col4, col5 = st.columns(2)
                with col4:
                    st.metric("Tipo de Voz", voice_type or "N/A", delta_color="off")
                with col5:
                    st.metric("Grupo de Edad", age_group or "N/A", delta_color="off")

            if freq == 0:
                st.markdown("**Esperando sonido...**")

        except queue.Empty:
            st.markdown("**Cola de audio vacía - No se recibieron datos.**")
        except Exception as e:
            st.error(f"Error: {e}")

# Sección: Cargar Archivo WAV
st.subheader("Cargar Archivo WAV para Análisis")
uploaded_file = st.file_uploader("Carga un archivo WAV", type=["wav"])

audio_loaded = False
audio_data_loaded = None
sample_rate_loaded = None

if uploaded_file is not None:
    try:
        sample_rate_loaded, audio_data_loaded = wav_read(uploaded_file)
        if len(audio_data_loaded.shape) > 1 and audio_data_loaded.shape[1] == 2:
            audio_data_loaded = audio_data_loaded.mean(axis=1)
        audio_loaded = True
        st.success(f"Archivo cargado: tasa de muestreo {sample_rate_loaded} Hz, duración {len(audio_data_loaded)/sample_rate_loaded:.2f} s")
    except Exception as e:
        st.error(f"Error leyendo el archivo: {e}")

if audio_loaded:
    audio_float = audio_data_loaded.astype(np.float32)
    audio_float /= np.max(np.abs(audio_float))

    max_samples = SAMPLE_RATE * 5
    if len(audio_float) > max_samples:
        audio_float = audio_float[:max_samples]

    yf, xf = get_spectrum(audio_float, sample_rate=sample_rate_loaded)
    max_ampl = np.max(yf)
    freq = get_dominant_freq(xf, yf)

    sound_type, description, voice_type, age_group = analyze_sound(freq, yf, xf)

    fig = plot_spectrum(yf, xf, freq)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Análisis del Audio Cargado")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tipo de Sonido", sound_type, delta_color="off")
    with col2:
        st.metric("Frecuencia Dominante", f"{freq:.2f} Hz" if freq > 0 else "N/A")
    with col3:
        st.metric("Amplitud Máxima", f"{max_ampl:.2f}")

    if sound_type == "Voz humana (probablemente hablando)":
        col4, col5 = st.columns(2)
        with col4:
            st.metric("Tipo de Voz", voice_type or "N/A", delta_color="off")
        with col5:
            st.metric("Grupo de Edad", age_group or "N/A", delta_color="off")

# Sección: Generar Tono para Pruebas
st.subheader("Generar Tono para Pruebas")
st.info("La generación y reproducción de tonos no está disponible en Streamlit Cloud. Por favor, carga un archivo WAV para analizar.")

# Sección: Reproducción y Descarga de Grabación
if st.session_state.audio_bytes:
    st.subheader("Reproducción de Audio Grabado")
    st.audio(st.session_state.audio_bytes, format="audio/wav")
    st.download_button("Descargar Audio Grabado", data=st.session_state.audio_bytes, file_name="grabacion.wav", mime="audio/wav")

# Limpieza al cerrar
if 'stream' in st.session_state and st.session_state.stream and not st.session_state.mic_active:
    st.session_state.stream.stop_stream()
    st.session_state.stream.close()
    st.session_state.audio_interface.terminate()
    st.session_state.stream = None
