# Import library yang dibutuhin untuk aplikasi ini
import streamlit as st  # Library untuk bikin web app interaktif
import numpy as np  # Library untuk operasi numerik (array, matrix, dll.)
import pandas as pd  # Library untuk manipulasi data (DataFrame, CSV, dll.)
from tensorflow.keras.models import load_model  # Fungsi untuk load model machine learning yang udah dilatih
import plotly.graph_objects as go  # Library untuk bikin visualisasi interaktif (grafik, heatmap, dll.)
import os  # Library untuk operasi sistem (cek file, path, dll.)
from scipy.signal import butter, lfilter  # Fungsi untuk filtering sinyal (low-pass filter)
import logging  # Library untuk logging (nulis pesan debug/error)
import base64  # Library untuk encode gambar ke base64 (buat background)
import requests  # Library untuk bikin HTTP request (buat Gemini API)
import io  # Library untuk handle input/output (buat konversi gambar ke bytes)

# Setup logging biar bisa monitor apa yang terjadi di aplikasi
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# level=logging.INFO: Log semua pesan INFO ke atas (INFO, WARNING, ERROR)
# format: Format pesan log (waktu - level - pesan)

# ---- Gemini API Key ----
# Ambil API Key Gemini dari environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Cek apakah API Key ada, kalau nggak ada, stop aplikasi
if not GEMINI_API_KEY:
    st.error("🚨 Gemini API Key not found! Please set the environment variable `GEMINI_API_KEY`.")
    st.stop()

# ---- Gemini AI Analysis Function ----
# Fungsi untuk analisis velocity map pake Gemini API
def analyze_velocity_map_with_gemini(predictions):
    # URL endpoint Gemini API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    # Header untuk request (kasih tau server kita kirim data JSON)
    headers = {'Content-Type': 'application/json'}

    # Import matplotlib di dalam fungsi (buat bikin heatmap dari velocity map)
    import matplotlib.pyplot as plt
    # Bikin figure dan axes untuk plot
    fig, ax = plt.subplots()
    # Bikin heatmap dari velocity map pake cmap='jet' (warna biru-merah)
    cax = ax.imshow(predictions[0], cmap='jet')
    # Tambah colorbar biar tau skala warnanya
    plt.colorbar(cax)
    # Kasih judul plot
    plt.title("Velocity Map")
    # Hilangin sumbu biar fokus ke heatmap
    plt.axis('off')

    # Simpan plot ke bytes (buat dikirim ke Gemini)
    buffered = io.BytesIO()
    plt.savefig(buffered, format="PNG", bbox_inches='tight')
    plt.close()
    # Encode gambar ke base64
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Payload (data yang dikirim ke Gemini)
    payload = {
        "contents": [
            {
                "parts": [
                    # Instruksi ke Gemini: analisis velocity map
                    {"text": "Analyze this velocity map from a seismic waveform inversion and provide an accurate geophysical explanation regarding the subsurface structure."},
                    # Kirim gambar heatmap dalam format base64
                    {"inline_data": {"mime_type": "image/png", "data": image_base64}}
                ]
            }
        ]
    }

    # Kirim request ke Gemini API
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()  # Kalau ada error (misalnya 404), bakal raise exception
        # Ambil teks hasil analisis dari response
        return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Analysis unavailable.")
    except requests.exceptions.RequestException as e:
        # Kalau gagal (misalnya timeout), kasih pesan error
        return f"⚠️ Error retrieving analysis: {e}"

# ---- Set Page Config ----
# Atur konfigurasi halaman Streamlit
st.set_page_config(page_title="Waveform Inversion Predictor", layout="wide")
# page_title: Judul tab browser
# layout="wide": Bikin layout lebar (full width)

# ---- Encode Gambar ke Base64 ----
# Fungsi untuk encode gambar ke base64 (buat background)
def get_base64_image(file_path):
    try:
        # Buka file gambar dalam mode binary
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        # Kalau gagal (misalnya file nggak ada), log error
        logging.error(f"Error encoding image {file_path}: {e}")
        return ""

# Encode gambar background dan sidebar
sidebar_img = get_base64_image("/home/ghost00/Videos/Project-Kaggle/sidebar.jpg")
bg_img = get_base64_image("/home/ghost00/Videos/Project-Kaggle/bg.jpg")

# Fallback ke gambar online kalau file lokal gagal
main_bg_url = (f'data:image/jpeg;base64,{bg_img}' if bg_img else "https://via.placeholder.com/1920x1080?text=Waveform+Inversion+BG")
sidebar_bg_url = (f'data:image/jpeg;base64,{sidebar_img}' if sidebar_img else "https://via.placeholder.com/300x1080?text=Sidebar+BG")

# ---- Custom CSS untuk Desain yang Lebih Keren ----
# Tambah CSS custom untuk styling tampilan
st.markdown(f"""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@700&family=Montserrat:wght@400;700&display=swap');

/* General Styling */
[data-testid="stAppViewContainer"] {{
    background: url("{main_bg_url}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-blend-mode: overlay;
    background-color: rgba(20, 20, 20, 0.9);
    color: #ffffff;
    min-height: 100vh;
    position: relative;
    overflow: hidden;
}}

/* Particle Effect di Background */
[data-testid="stAppViewContainer"]::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: url('https://www.transparenttextures.com/patterns/stardust.png');
    opacity: 0.1;
    pointer-events: none;
    animation: particleMove 20s linear infinite;
}}
@keyframes particleMove {{
    0% {{ background-position: 0 0; }}
    100% {{ background-position: 1000px 1000px; }}
}}

/* Sidebar Styling */
[data-testid="stSidebar"] {{
    background: url("{sidebar_bg_url}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-blend-mode: overlay;
    background-color: rgba(20, 20, 20, 0.75);
    color: #ffffff;
    border-right: 3px solid #00ffcc;
    box-shadow: 0 0 20px rgba(0, 255, 204, 0.5);
    animation: neonBorder 2s infinite alternate;
    overflow-y: auto; /* Biar sidebar bisa discroll */
}}
@keyframes neonBorder {{
    0% {{ border-right-color: #00ffcc; box-shadow: 0 0 10px rgba(0, 255, 204, 0.5); }}
    100% {{ border-right-color: #34d399; box-shadow: 0 0 20px rgba(52, 211, 153, 0.8); }}
}}

/* Title Styling */
.title {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.5em;
    text-align: center;
    color: #00ffcc;
    text-shadow: 2px 2px 10px rgba(0, 255, 204, 0.8), 0 0 20px rgba(0, 255, 204, 0.5);
    animation: glow 2s ease-in-out infinite alternate, fadeIn 1.5s ease-in-out;
}}
@keyframes glow {{
    from {{ text-shadow: 2px 2px 10px rgba(0, 255, 204, 0.8), 0 0 20px rgba(0, 255, 204, 0.5); }}
    to {{ text-shadow: 2px 2px 15px rgba(52, 211, 153, 1), 0 0 30px rgba(52, 211, 153, 0.8); }}
}}
@keyframes fadeIn {{
    0% {{ opacity: 0; transform: translateY(20px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}

/* Subtitle Styling */
.subtitle {{
    font-family: 'Montserrat', sans-serif;
    font-size: 1em;
    text-align: center;
    color: #ffffff;
    font-weight: bold;
    text-shadow: 1px 1px 5px rgba(0, 0, 0, 0.5);
    animation: fadeIn 2s ease-in-out;
}}

/* Button Styling */
.stButton>button {{
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 10px;
    font-size: 16px;
    font-family: 'Montserrat', sans-serif;
    padding: 10px 20px;
    box-shadow: 0 0 15px rgba(255, 65, 108, 0.5), 0 0 30px rgba(255, 75, 43, 0.3);
    transition: all 0.3s ease;
    animation: fadeIn 2.5s ease-in-out;
}}
.stButton>button:hover {{
    transform: scale(1.05);
    background: linear-gradient(90deg, #ff4b2b, #ff416c);
    box-shadow: 0 0 25px rgba(255, 65, 108, 0.8), 0 0 50px rgba(255, 75, 43, 0.5);
}}

/* Disclaimer Styling */
.disclaimer {{
    text-align: center;
    background: rgba(255, 165, 0, 0.3);
    padding: 15px;
    border-radius: 10px;
    border: 2px solid #ffcc00;
    color: #ffffff;
    font-family: 'Montserrat', sans-serif;
    font-size: 0.85em;
    font-weight: bold;
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    animation: fadeIn 2s ease-in-out;
}}

/* Sidebar Links Styling */
.sidebar-link {{
    display: inline-block;
    background: linear-gradient(90deg, #00ffcc, #34d399);
    color: #1a1a1a;
    padding: 8px 15px;
    border-radius: 20px;
    text-decoration: none;
    font-family: 'Montserrat', sans-serif;
    font-size: 0.85em;
    font-weight: bold;
    margin: 5px 0;
    transition: all 0.3s ease;
    animation: fadeIn 2.5s ease-in-out;
}}
.sidebar-link:hover {{
    transform: scale(1.05);
    background: linear-gradient(90deg, #34d399, #00ffcc);
    box-shadow: 0 0 15px rgba(52, 211, 153, 0.5);
}}

/* Sidebar Text Styling */
.sidebar-text {{
    font-family: 'Montserrat', sans-serif;
    font-size: 0.85em;
    color: #00ffcc;
    text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
    animation: fadeIn 2s ease-in-out;
}}

/* Result Card Styling */
.result-card {{
    background: rgba(20, 20, 20, 0.9);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
    border: 2px solid #00ffcc;
    box-shadow: 0 0 20px rgba(0, 255, 204, 0.5);
    color: #ffffff;
    text-align: center;
    animation: fadeIn 1s ease-in-out;
}}
.result-card h3 {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.5em;
    color: #00ffcc;
    text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
    margin-bottom: 10px;
}}
.result-card p {{
    font-family: 'Montserrat', sans-serif;
    font-size: 0.95em;
    color: #ffffff;
    margin: 5px 0;
}}

/* Plot Container Styling */
.plot-container {{
    background: rgba(20, 20, 20, 0.9);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 15px;
    margin: 20px 0;
    border: 2px solid #00ffcc;
    box-shadow: 0 0 20px rgba(0, 255, 204, 0.5);
    transition: all 0.3s ease;
    animation: fadeIn 1.5s ease-in-out;
}}
.plot-container:hover {{
    transform: scale(1.02);
    box-shadow: 0 0 30px rgba(52, 211, 153, 0.8);
}}
</style>
""", unsafe_allow_html=True)

# ---- Title ----
# Tampilin judul utama
st.markdown('<p class="title">🌍 Waveform Inversion Predictor</p>', unsafe_allow_html=True)
# Tampilin subtitle
st.markdown('<p class="subtitle">🔍 Predict Subsurface Structures with AI</p>', unsafe_allow_html=True)

# ---- Load Model ----
# Path ke file model machine learning
model_path = "/home/ghost00/Videos/Project-Kaggle/waveform_inversion_model.h5"
# Cek apakah file model ada
if os.path.exists(model_path):
    # Load model kalau ada
    model = load_model(model_path)
    # Tampilin pesan sukses di sidebar
    st.sidebar.success("✅ Model Loaded Successfully!")
else:
    # Tampilin pesan error di sidebar kalau model nggak ada
    st.sidebar.error("❌ Model Not Found! Ensure the .h5 File Exists.")
    # Stop aplikasi
    st.stop()

# ---- Sidebar Disclaimer ----
# Tampilin disclaimer di sidebar
st.sidebar.markdown("""
<div class="disclaimer">
    ⚠️ <b>Disclaimer:</b><br>
    These predictions are not official geophysical data.<br>
    Use for research purposes and consult an expert.
</div>
""", unsafe_allow_html=True)

# ---- Sidebar Developer Info ----
# Tampilin info developer di sidebar
st.sidebar.markdown("""
---
<p class="sidebar-text">👨‍💻 <b>Developed by Farrel0xx</b></p>
<div style="text-align: center;">
    <a href="https://github.com/Farrel0xx" target="_blank" class="sidebar-link">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" style="vertical-align: middle; margin-right: 5px;"> GitHub
    </a><br>
    <a href="https://youtube.com/@Farrel0xx" target="_blank" class="sidebar-link">
        <img src="https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg" width="30" style="vertical-align: middle; margin-right: 5px;"> YouTube
    </a>
</div>
""", unsafe_allow_html=True)

# ---- Upload Seismic File ----
# Tambah file uploader di sidebar untuk upload file seismic (.npy)
uploaded_file = st.sidebar.file_uploader("📤 Upload Seismic File (.npy)", type=["npy"])

# ---- Process and Predict ----
# Cek apakah user udah upload file
if uploaded_file is not None:
    # Load data seismic dari file .npy
    seismic_data = np.load(uploaded_file)
    # Tampilin pesan sukses
    st.success("✅ Seismic File Uploaded Successfully!")
    # Log shape data seismic
    logging.info(f"Seismic data shape: {seismic_data.shape}")

    # Visualize seismic data
    # Bikin container untuk plot
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    # Tambah judul untuk visualisasi
    st.markdown("### 📈 Seismic Data Visualization")
    # Bikin figure Plotly
    fig = go.Figure()

    # Cek dimensi data seismic
    if len(seismic_data.shape) == 4:
        # Shape: (batch_size, num_sources, time_steps, num_receivers)
        num_sources = seismic_data.shape[1]
        # Plot 3 sumber pertama
        for i in range(min(3, num_sources)):
            fig.add_trace(go.Scatter(
                y=seismic_data[0, i, :, 0],
                mode='lines',
                name=f'Source {i+1}',
                line=dict(color=f'rgb({50 + i*50}, {150 - i*30}, 255)')
            ))
    elif len(seismic_data.shape) == 3:
        # Shape: (num_sources, time_steps, num_receivers) atau (batch_size, time_steps, num_receivers)
        # Asumsi (num_sources, time_steps, num_receivers)
        num_sources = seismic_data.shape[0]
        for i in range(min(3, num_sources)):
            fig.add_trace(go.Scatter(
                y=seismic_data[i, :, 0],
                mode='lines',
                name=f'Source {i+1}',
                line=dict(color=f'rgb({50 + i*50}, {150 - i*30}, 255)')
            ))
    else:
        # Kalau format data nggak didukung, tampilin error
        st.error("❌ Unsupported Seismic Data Format! Must be 3D or 4D.")
        st.stop()

    # Atur layout plot
    fig.update_layout(
        title="Sample Seismic Waveform",
        xaxis_title="Time Steps",
        yaxis_title="Amplitude",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Montserrat", size=12, color="#ffffff")
    )
    # Tampilin plot
    st.plotly_chart(fig, use_container_width=True)
    # Tutup container plot
    st.markdown('</div>', unsafe_allow_html=True)

    # Tambah tombol untuk prediksi
    if st.button("🔎 Predict Velocity Map"):
        # Tampilin spinner selama proses
        with st.spinner("⏳ Processing Seismic Data..."):
            # Preprocessing: Low-pass filter (simple)
            # Fungsi untuk filter sinyal (hilangin noise frekuensi tinggi)
            def lowpass_filter(data, cutoff=10, fs=100, order=5):
                nyquist = 0.5 * fs
                normal_cutoff = cutoff / nyquist
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                filtered_data = lfilter(b, a, data)
                return filtered_data

            # Bikin array kosong untuk data yang udah difilter
            filtered_seismic = np.zeros_like(seismic_data)
            # Filter data berdasarkan dimensi
            if len(seismic_data.shape) == 4:
                for i in range(seismic_data.shape[0]):
                    for j in range(seismic_data.shape[1]):
                        for k in range(seismic_data.shape[3]):
                            filtered_seismic[i, j, :, k] = lowpass_filter(seismic_data[i, j, :, k])
            elif len(seismic_data.shape) == 3:
                for i in range(seismic_data.shape[0]):
                    for j in range(seismic_data.shape[2]):
                        filtered_seismic[i, :, j] = lowpass_filter(seismic_data[i, :, j])

            # Prepare input untuk model
            if len(seismic_data.shape) == 3:
                # Tambah dimensi batch_size kalau data 3D
                filtered_seismic = filtered_seismic[np.newaxis, ...]
            # Tambah dimensi channel (buat model)
            X = filtered_seismic[..., np.newaxis]  # Shape: (batch_size, num_sources, time_steps, num_receivers, 1)

            # Prediksi pake model
            predictions = model.predict(X)  # Shape: (batch_size, height, width)

            # Display predicted velocity map
            # Bikin container untuk velocity map
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            # Tambah judul
            st.markdown("### 🗺️ Predicted Velocity Map")
            # Bikin heatmap dari hasil prediksi
            fig = go.Figure(data=go.Heatmap(
                z=predictions[0],
                colorscale="Jet",
                showscale=True
            ))
            # Atur layout heatmap
            fig.update_layout(
                title="Predicted Velocity Map",
                xaxis_title="Width",
                yaxis_title="Height",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Montserrat", size=12, color="#ffffff")
            )
            # Tampilin heatmap
            st.plotly_chart(fig, use_container_width=True)
            # Tutup container
            st.markdown('</div>', unsafe_allow_html=True)

            # Gemini AI Explanation
            # Tampilin penjelasan dari Gemini API
            st.markdown('<div class="result-card"><h3>📝 AI Explanation</h3>', unsafe_allow_html=True)
            desc = analyze_velocity_map_with_gemini(predictions)
            st.markdown(f"<p>{desc}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Prepare submission
            # Flatten hasil prediksi
            predictions_flattened = predictions.reshape(predictions.shape[0], -1)
            # Ambil kolom ganjil
            odd_columns = predictions_flattened[:, 1::2]
            # Reshape ke 1D
            submission_data = odd_columns.reshape(-1)

            # Pad data kalau kurang dari jumlah baris yang dibutuhin
            total_rows = 460726
            submission_data = np.pad(submission_data, (0, total_rows - len(submission_data)), 'constant', constant_values=3000.0)

            # Create submission file
            # Baca file sample submission
            sample_submission = pd.read_csv('/home/ghost00/Videos/Project-Kaggle/sample_submission.csv')
            # Bikin DataFrame untuk submission
            submission = pd.DataFrame({'velocity': submission_data})
            # Simpan ke file CSV
            submission_file = "submission_waveform.csv"
            submission.to_csv(submission_file, index=False)

            # Display results
            # Tampilin hasil prediksi
            st.markdown(f"""
            <div class="result-card">
                <h3>📊 Prediction Results</h3>
                <p>Number of samples processed: {predictions.shape[0]}</p>
                <p>Submission file created: {submission_file}</p>
            </div>
            """, unsafe_allow_html=True)

            # Download button
            # Tambah tombol untuk download file submission
            with open(submission_file, "rb") as f:
                st.download_button(
                    label="📥 Download Submission File",
                    data=f,
                    file_name=submission_file,
                    mime="text/csv"
                )
