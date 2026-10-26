import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time

# --- 1. CONFIGURATION DAN LOAD MODEL ---
MODEL_PATH = "saved/best_finetuned.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal memuat model dari {MODEL_PATH}. Error: {e}")
    st.stop()

# Label Kelas (dengan nama lebih profesional)
label_map = {
    0: {"name": "Hazardous", "description": "Bahan berbahaya dan beracun", "color": "#E53935"},
    1: {"name": "Non-Recyclable", "description": "Sampah tidak dapat didaur ulang", "color": "#757575"},
    2: {"name": "Organic", "description": "Sampah organik/biodegradable", "color": "#43A047"},
    3: {"name": "Recyclable", "description": "Bahan dapat didaur ulang", "color": "#1E88E5"}
}

IMG_SIZE = (224, 224)

# --- 2. CUSTOM CSS STYLING ---
st.markdown("""
c
""", unsafe_allow_html=True)

# --- 3. PAGE SETTINGS ---
st.set_page_config(
    page_title="Smart Waste Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🗑️"
)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style='padding: 1rem;'>
            <h2 style='color: white;'>Model Analytics</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Model Stats
    st.markdown("""
        <div class='stat-card'>
            <div class='stat-value'>76%</div>
            <div class='stat-label'>Accuracy</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Model Details
    st.markdown("### Model Specifications")
    st.markdown("""
    - **Architecture**: MobileNetV2 Fine-tuned
    - **Input Size**: 224×224 pixels
    - **Classes**: 4 Categories
    - **Training Dataset**: Custom Waste Collection
    """)
    
    st.divider()
    
    # Category Guide
    st.markdown("### Waste Categories")
    for i, (key, value) in enumerate(label_map.items()):
        st.markdown(f"""
            <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                <div style='width: 12px; height: 12px; background-color: {value["color"]}; 
                     border-radius: 50%; margin-right: 10px;'></div>
                <span style='color: white;'>{value["name"]}</span>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Improvement Roadmap
    st.markdown("### Improvement Roadmap")
    st.info("""
    - Enhanced dataset collection
    - Data augmentation techniques
    - Real-time performance metrics
    - Multi-model ensemble approach
    """)

# --- 5. MAIN CONTENT ---
# Header Section
st.markdown("""
    <div class='header-container'>
        <h1 class='header-title'>Smart Waste Classifier</h1>
        <p class='header-subtitle'>
            Advanced AI-powered waste classification system using deep learning
            for efficient waste management and recycling optimization
        </p>
    </div>
""", unsafe_allow_html=True)

# Main Content Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Image Input")
    st.markdown("Upload an image of waste material for classification analysis")
    
    # Upload Section
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "png", "jpeg"],
        help="Supported formats: JPG, PNG, JPEG",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, use_column_width=True, caption="Uploaded Image")
        
        # Predict Button
        if st.button("Analyze Waste Category", type="primary"):
            with st.spinner('Processing image with AI model...'):
                time.sleep(1)
                
                # Preprocess
                img_resized = img.resize(IMG_SIZE)
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                pred = model.predict(img_array)[0]
                class_id = np.argmax(pred)
                confidence = float(pred[class_id] * 100)
                
                # Get class details
                class_detail = label_map[class_id]
                class_name = class_detail["name"]
                class_color = class_detail["color"]
                class_desc = class_detail["description"]
                
                # Store results in session state
                st.session_state.prediction_result = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'class_color': class_color,
                    'confidence': confidence,
                    'description': class_desc
                }
                
                st.rerun()
    else:
        st.markdown("""
            <div class='upload-area'>
                <div style='font-size: 4rem; margin-bottom: 1rem;'>📷</div>
                <h3>Drag & Drop or Click to Upload</h3>
                <p style='color: #666;'>Upload an image of waste material for analysis</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Analysis Results")
    
    if 'prediction_result' in st.session_state:
        result = st.session_state.prediction_result
        
        st.markdown(f"""
            <div class='prediction-card' style='border-left-color: {result["class_color"]};'>
                <h2 style='color: {result["class_color"]}; margin-bottom: 0.5rem;'>
                    {result["class_name"]}
                </h2>
                <p style='color: #666; margin-bottom: 1.5rem;'>{result["description"]}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Confidence Visualization
        st.markdown("#### Confidence Level")
        confidence = result["confidence"]
        
        # Custom confidence meter
        st.markdown(f"""
            <div class='confidence-meter'>
                <div class='confidence-fill' style='width: {confidence}%; background: {result["class_color"]};'></div>
            </div>
            <div style='display: flex; justify-content: space-between; margin-top: 0.5rem;'>
                <span style='color: #666;'>0%</span>
                <span style='font-weight: 600; color: {result["class_color"]};'>{confidence:.1f}%</span>
                <span style='color: #666;'>100%</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Status Indicator
        if confidence >= 80:
            status = "High Confidence"
            status_color = "#4CAF50"
        elif confidence >= 60:
            status = "Medium Confidence"
            status_color = "#FF9800"
        else:
            status = "Low Confidence"
            status_color = "#F44336"
        
        st.markdown(f"""
            <div style='display: flex; align-items: center; margin: 1.5rem 0;'>
                <div style='width: 10px; height: 10px; background-color: {status_color}; 
                     border-radius: 50%; margin-right: 10px;'></div>
                <span>Classification Status: <strong>{status}</strong></span>
            </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("#### Recommended Action")
        recommendations = {
            "Hazardous": "Handle with protective equipment. Dispose at designated hazardous waste facility.",
            "Non-Recyclable": "Dispose in general waste. Consider waste-to-energy options if available.",
            "Organic": "Compost or use for biogas production. Avoid landfill disposal.",
            "Recyclable": "Clean and separate material. Send to appropriate recycling facility."
        }
        
        st.info(recommendations.get(result["class_name"], "Consult local waste management guidelines."))
        
        # Reset Button
        if st.button("Analyze Another Image"):
            del st.session_state.prediction_result
            st.rerun()
    else:
        st.markdown("""
            <div style='text-align: center; padding: 3rem; color: #999;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>🔍</div>
                <h3>Awaiting Analysis</h3>
                <p>Upload an image and click "Analyze Waste Category" to see results</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer Section
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Smart Waste Classifier v1.0 | Deep Learning Model | For Environmental Sustainability</p>
        <p style='font-size: 0.9rem;'>This system assists in waste classification and should be used as guidance only.</p>
    </div>
""", unsafe_allow_html=True)