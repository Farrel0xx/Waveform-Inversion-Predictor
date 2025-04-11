import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import os
from scipy.signal import butter, lfilter
import logging
import base64

# Setup logging biar bisa monitor
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---- Set Page Config ----
st.set_page_config(page_title="Waveform Inversion Predictor", layout="wide")

# ---- Encode Gambar ke Base64 ----
def get_base64_image(file_path):
    try:
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        logging.error(f"Error encoding image {file_path}: {e}")
        return ""

sidebar_img = get_base64_image("/home/ghost00/Videos/Project-Kaggle/sidebar.jpg")
bg_img = get_base64_image("/home/ghost00/Videos/Project-Kaggle/bg.jpg")

# Fallback to online images if local files fail
main_bg_url = (f'data:image/jpeg;base64,{bg_img}' if bg_img else "https://via.placeholder.com/1920x1080?text=Waveform+Inversion+BG")
sidebar_bg_url = (f'data:image/jpeg;base64,{sidebar_img}' if sidebar_img else "https://via.placeholder.com/300x1080?text=Sidebar+BG")

# ---- Custom CSS untuk Desain Mirip AI Brain Tumor Detector ----
st.markdown(f"""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Poppins:wght@400;700&display=swap');

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
}}
@keyframes neonBorder {{
    0% {{ border-right-color: #00ffcc; box-shadow: 0 0 10px rgba(0, 255, 204, 0.5); }}
    100% {{ border-right-color: #00ccff; box-shadow: 0 0 20px rgba(0, 204, 255, 0.8); }}
}}

/* Title Styling */
.title {{
    font-family: 'Orbitron', sans-serif;
    font-size: 2.8em;
    text-align: center;
    color: #00ffcc;
    text-shadow: 2px 2px 10px rgba(0, 255, 204, 0.8), 0 0 20px rgba(0, 255, 204, 0.5);
    animation: glow 2s ease-in-out infinite alternate, fadeIn 1.5s ease-in-out;
}}
@keyframes glow {{
    from {{ text-shadow: 2px 2px 10px rgba(0, 255, 204, 0.8), 0 0 20px rgba(0, 255, 204, 0.5); }}
    to {{ text-shadow: 2px 2px 20px rgba(0, 255, 204, 1), 0 0 40px rgba(0, 255, 204, 0.8); }}
}}
@keyframes fadeIn {{
    0% {{ opacity: 0; transform: translateY(20px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}

/* Subtitle Styling */
.subtitle {{
    font-family: 'Poppins', sans-serif;
    font-size: 1.2em;
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
    font-size: 18px;
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
    font-family: 'Poppins', sans-serif;
    font-weight: bold;
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    animation: fadeIn 2s ease-in-out;
}}

/* Sidebar Links Styling */
.sidebar-link {{
    display: inline-block;
    background: linear-gradient(90deg, #00ffcc, #00ccff);
    color: #1a1a1a;
    padding: 8px 15px;
    border-radius: 20px;
    text-decoration: none;
    font-family: 'Poppins', sans-serif;
    font-weight: bold;
    margin: 5px 0;
    transition: all 0.3s ease;
    animation: fadeIn 2.5s ease-in-out;
}}
.sidebar-link:hover {{
    transform: scale(1.05);
    background: linear-gradient(90deg, #00ccff, #00ffcc);
    box-shadow: 0 0 15px rgba(0, 255, 204, 0.5);
}}

/* Sidebar Text Styling */
.sidebar-text {{
    font-family: 'Poppins', sans-serif;
    color: #00ffcc;
    text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
    animation: fadeIn 2s ease-in-out;
}}

/* Sidebar Image Styling */
.sidebar-image {{
    display: block;
    margin: 10px auto;
    width: 80%;
    border-radius: 10px;
    border: 1px solid #00ffcc;
    box-shadow: 0 0 10px rgba(0, 255, 204, 0.3);
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
    font-family: 'Orbitron', sans-serif;
    color: #00ffcc;
    text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
    margin-bottom: 10px;
}}
.result-card p {{
    font-family: 'Poppins', sans-serif;
    font-size: 1.1em;
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
    animation: fadeIn 1.5s ease-in-out;
}}
</style>
""", unsafe_allow_html=True)

# ---- Title ----
st.markdown('<p class="title">🌍 Waveform Inversion Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🔍 Predict Subsurface Structures with AI</p>', unsafe_allow_html=True)

# ---- Load Model ----
model_path = "/home/ghost00/Videos/Project-Kaggle/waveform_inversion_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
    st.sidebar.success("✅ Model Loaded Successfully!")
else:
    st.sidebar.error("❌ Model Not Found! Ensure the .h5 File Exists.")
    st.stop()

# ---- Sidebar Disclaimer ----
st.sidebar.markdown("""
<div class="disclaimer">
    ⚠️ <b>Disclaimer:</b><br>
    These predictions are not official geophysical data.<br>
    Use for research purposes and consult an expert.
</div>
""", unsafe_allow_html=True)

# ---- Sidebar Developer Info ----
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

# ---- Sidebar Image ----
st.sidebar.markdown(f"""
<img src="data:image/jpeg;base64,{sidebar_img}" class="sidebar-image">
""", unsafe_allow_html=True)

# ---- Upload Seismic File ----
uploaded_file = st.sidebar.file_uploader("📤 Upload Seismic File (.npy)", type=["npy"])

# ---- Process and Predict ----
if uploaded_file is not None:
    # Load seismic data
    seismic_data = np.load(uploaded_file)
    st.success("✅ Seismic File Uploaded Successfully!")
    logging.info(f"Seismic data shape: {seismic_data.shape}")

    # Visualize seismic data
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.markdown("### 📈 Seismic Data Visualization")
    fig = go.Figure()

    # Check seismic data dimensions
    if len(seismic_data.shape) == 4:
        # Shape: (batch_size, num_sources, time_steps, num_receivers)
        num_sources = seismic_data.shape[1]
        for i in range(min(3, num_sources)):  # Plot first 3 sources
            fig.add_trace(go.Scatter(
                y=seismic_data[0, i, :, 0],
                mode='lines',
                name=f'Source {i+1}',
                line=dict(color=f'rgb({50 + i*50}, {150 - i*30}, 255)')
            ))
    elif len(seismic_data.shape) == 3:
        # Shape: (num_sources, time_steps, num_receivers) or (batch_size, time_steps, num_receivers)
        # Assume (num_sources, time_steps, num_receivers)
        num_sources = seismic_data.shape[0]
        for i in range(min(3, num_sources)):
            fig.add_trace(go.Scatter(
                y=seismic_data[i, :, 0],
                mode='lines',
                name=f'Source {i+1}',
                line=dict(color=f'rgb({50 + i*50}, {150 - i*30}, 255)')
            ))
    else:
        st.error("❌ Unsupported Seismic Data Format! Must be 3D or 4D.")
        st.stop()

    fig.update_layout(
        title="Sample Seismic Waveform",
        xaxis_title="Time Steps",
        yaxis_title="Amplitude",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Poppins", size=12, color="#ffffff")
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔎 Predict Velocity Map"):
        with st.spinner("⏳ Processing Seismic Data..."):
            # Preprocessing: Low-pass filter (simple)
            def lowpass_filter(data, cutoff=10, fs=100, order=5):
                nyquist = 0.5 * fs
                normal_cutoff = cutoff / nyquist
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                filtered_data = lfilter(b, a, data)
                return filtered_data

            filtered_seismic = np.zeros_like(seismic_data)
            if len(seismic_data.shape) == 4:
                for i in range(seismic_data.shape[0]):
                    for j in range(seismic_data.shape[1]):
                        for k in range(seismic_data.shape[3]):
                            filtered_seismic[i, j, :, k] = lowpass_filter(seismic_data[i, j, :, k])
            elif len(seismic_data.shape) == 3:
                for i in range(seismic_data.shape[0]):
                    for j in range(seismic_data.shape[2]):
                        filtered_seismic[i, :, j] = lowpass_filter(seismic_data[i, :, j])

            # Prepare input for model
            if len(seismic_data.shape) == 3:
                # Add batch_size dimension if 3D
                filtered_seismic = filtered_seismic[np.newaxis, ...]
            X = filtered_seismic[..., np.newaxis]  # Shape: (batch_size, num_sources, time_steps, num_receivers, 1)

            # Predict
            predictions = model.predict(X)  # Shape: (batch_size, height, width)

            # Display predicted velocity map
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.markdown("### 🗺️ Predicted Velocity Map")
            fig = go.Figure(data=go.Heatmap(
                z=predictions[0],
                colorscale="Jet",
                showscale=True
            ))
            fig.update_layout(
                title="Predicted Velocity Map",
                xaxis_title="Width",
                yaxis_title="Height",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Poppins", size=12, color="#ffffff")
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Prepare submission
            predictions_flattened = predictions.reshape(predictions.shape[0], -1)
            odd_columns = predictions_flattened[:, 1::2]  # Take odd columns
            submission_data = odd_columns.reshape(-1)

            # Pad with placeholder
            total_rows = 460726
            submission_data = np.pad(submission_data, (0, total_rows - len(submission_data)), 'constant', constant_values=3000.0)

            # Create submission file
            sample_submission = pd.read_csv('/home/ghost00/Videos/Project-Kaggle/sample_submission.csv')
            submission = pd.DataFrame({'velocity': submission_data})
            submission_file = "submission_waveform.csv"
            submission.to_csv(submission_file, index=False)

            # Display results
            st.markdown(f"""
            <div class="result-card">
                <h3>📊 Prediction Results</h3>
                <p>Number of samples processed: {predictions.shape[0]}</p>
                <p>Submission file created: {submission_file}</p>
            </div>
            """, unsafe_allow_html=True)

            # Download button
            with open(submission_file, "rb") as f:
                st.download_button(
                    label="📥 Download Submission File",
                    data=f,
                    file_name=submission_file,
                    mime="text/csv"
                )
