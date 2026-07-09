import os
import torch
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Import custom encoder architectures
from src.image_encoder import ImageClassifier
from src.clinical_encoder import ClinicalClassifier
from src.geo_encoder import GeoRiskEncoder
from src.fusion_transformer import ModalityFusionTransformer
from src.utils import GradCAM

# ---------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AnemiaFusionNet Portal",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium UI style (Google Fonts, Glassmorphism, Gradients)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0e1117 0%, #161b22 100%);
    }
    
    /* Header Card */
    .header-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 30px;
        margin-bottom: 25px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    .header-title {
        background: linear-gradient(90deg, #ff4b4b 0%, #8a2be2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 10px;
    }
    
    .header-subtitle {
        color: #8b949e;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Glassmorphism containers */
    .card-panel {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(5px);
    }
    
    /* Custom Predictions Banner */
    .result-banner-healthy {
        background: rgba(46, 204, 113, 0.1);
        border: 1px solid rgba(46, 204, 113, 0.3);
        border-radius: 8px;
        padding: 15px;
        color: #2ecc71;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
    }
    
    .result-banner-anemic {
        background: rgba(231, 76, 60, 0.1);
        border: 1px solid rgba(231, 76, 60, 0.3);
        border-radius: 8px;
        padding: 15px;
        color: #e74c3c;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Model Wrappers and Loaders
# ---------------------------------------------------------
class AblationFusionWrapper(nn.Module):
    def __init__(self, num_states, d_model=256, use_img=True, use_clin=True, use_geo=True):
        super().__init__()
        self.geo_encoder = GeoRiskEncoder(num_states=num_states, d_model=d_model)
        self.fusion = ModalityFusionTransformer(d_model=d_model)
        self.use_img = use_img
        self.use_clin = use_clin
        self.use_geo = use_geo
        
    def forward(self, img_emb, clin_emb, state_idx, geo_risk, return_attn=False):
        B = img_emb.size(0)
        if not self.use_img:
            img_emb = torch.zeros_like(img_emb)
        if not self.use_clin:
            clin_emb = torch.zeros_like(clin_emb)
        if self.use_geo:
            geo_emb = self.geo_encoder(state_idx, geo_risk)
        else:
            geo_emb = torch.zeros_like(img_emb)
            
        return self.fusion(img_emb, clin_emb, geo_emb, return_attn=return_attn)

@st.cache_resource
def load_models_and_configs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load metadata mappings
    with open("data/geo/state_to_idx.pkl", "rb") as f:
        state_to_idx = pickle.load(f)
    num_states = len(state_to_idx)
    
    with open("data/processed/preprocessors.pkl", "rb") as f:
        preprocessors = pickle.load(f)
    scaler = preprocessors["scaler"]
    encoder = preprocessors["encoder"]
    
    nfhs5_df = pd.read_csv("data/geo/nfhs5_state_prevalence.csv")
    state_to_risk = dict(zip(nfhs5_df["state"], nfhs5_df["geo_risk_score"]))
    state_to_prevalence = dict(zip(nfhs5_df["state"], nfhs5_df["women_pct"]))
    
    # 1. Image Encoder
    img_classifier = ImageClassifier(embed_dim=256, num_classes=2, pretrained=False).to(device)
    img_classifier.load_state_dict(torch.load("models/image_classifier_best.pt", map_location=device))
    img_classifier.eval()
    
    # 2. Clinical Encoder
    clinical_classifier = ClinicalClassifier(num_numeric=13, cat_cardinalities=[2, 2], embed_dim=256).to(device)
    clinical_classifier.load_state_dict(torch.load("models/clinical_classifier_best.pt", map_location=device))
    clinical_classifier.eval()
    
    # 3. Fusion Wrapper
    fusion_model = AblationFusionWrapper(num_states=num_states, d_model=256).to(device)
    fusion_model.load_state_dict(torch.load("models/anemia_fusion_best.pt", map_location=device))
    fusion_model.eval()
    
    # Transforms
    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {
        "device": device,
        "state_to_idx": state_to_idx,
        "scaler": scaler,
        "encoder": encoder,
        "state_to_risk": state_to_risk,
        "state_to_prevalence": state_to_prevalence,
        "img_classifier": img_classifier,
        "clinical_classifier": clinical_classifier,
        "fusion_model": fusion_model,
        "val_transform": val_transform
    }

# Load assets
cfg = load_models_and_configs()

# ---------------------------------------------------------
# 3. UI Header
# ---------------------------------------------------------
st.markdown("""
<div class="header-card">
    <div class="header-title">AnemiaFusionNet Portal</div>
    <div class="header-title-divider"></div>
    <div class="header-subtitle">A Multimodal Feature Fusion Framework for Region-Aware Anemia Detection</div>
</div>
""", unsafe_allow_html=True)

# Define Tabs
tab1, tab2, tab3 = st.tabs([
    "🔮 Interactive Diagnosis", 
    "🏗️ System Architecture", 
    "📊 Performance Metrics"
])

# ---------------------------------------------------------
# TAB 1: INTERACTIVE DIAGNOSIS PORTAL
# ---------------------------------------------------------
with tab1:
    st.markdown('<div class="card-panel"><h3>🔮 Real-Time Diagnostic Fusion</h3>Predict patient status by fusing eye photo visual analysis, clinical lab blood counts, and geo-epidemiological risk priors.</div>', unsafe_allow_html=True)
    
    # Split sidebar layout
    col_input, col_result = st.columns([1, 1.2])
    
    with col_input:
        st.subheader("📋 Patient Inputs")
        
        # 1. Image Modality Input
        st.markdown("**1. Conjunctiva Photo**")
        uploaded_file = st.file_uploader("Upload Eye Conjunctiva Image", type=["jpg", "jpeg", "png"])
        
        use_sample = st.checkbox("Use Sample Patient Image instead")
        sample_img_path = "data/raw/images/dataset anemia/India/1/20200118_164733.jpg"
        
        input_image = None
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file).convert("RGB")
            st.image(input_image, caption="Uploaded Patient Eye Image", use_container_width=True)
        elif use_sample:
            if os.path.exists(sample_img_path):
                input_image = Image.open(sample_img_path).convert("RGB")
                st.image(input_image, caption="Sample Patient Eye Image (India, Subj 1)", use_container_width=True)
            else:
                st.warning("Sample image not found on disk. Please upload your own image.")
        else:
            st.info("Please upload an eye photo or check the box to use a sample image.")

        # 2. Geographical Modality Input
        st.markdown("**2. Geographical Prior (NFHS-5)**")
        state_list = sorted(list(cfg["state_to_idx"].keys()))
        selected_state = st.selectbox("Patient State of Residence", state_list, index=state_list.index("Maharashtra") if "Maharashtra" in state_list else 0)
        
        state_pct = cfg["state_to_prevalence"][selected_state]
        state_risk = cfg["state_to_risk"][selected_state]
        st.caption(f"📍 State Prevalence: **{state_pct:.1f}%** | Normalized Risk Weight: **{state_risk:.2f}**")
        
        # 3. Clinical Modality Inputs
        st.markdown("**3. Tabular Clinical Metrics (CBC)**")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            gender_select = st.selectbox("Gender", ["F", "M"])
            region_select = st.selectbox("Region Type", ["rural", "urban"])
            rbc = st.number_input("RBC (Red Blood Cells, 10^12/L)", min_value=1.0, max_value=8.0, value=4.92, step=0.1)
            hct = st.number_input("HCT (Hematocrit, %)", min_value=10.0, max_value=60.0, value=46.15, step=0.5)
            wbc = st.number_input("WBC (White Blood Cells, 10^9/L)", min_value=1.0, max_value=20.0, value=7.8, step=0.1)
        with col_c2:
            mcv = st.number_input("MCV (Mean Corpuscular Volume, fL)", min_value=40.0, max_value=130.0, value=89.0, step=1.0)
            mch = st.number_input("MCH (Mean Corpuscular Hemoglobin, pg)", min_value=10.0, max_value=50.0, value=28.9, step=0.5)
            mchc = st.number_input("MCHC (MCH Concentration, g/dL)", min_value=20.0, max_value=45.0, value=32.0, step=0.5)
            plt_val = st.number_input("PLT (Platelets, 10^9/L)", min_value=10.0, max_value=600.0, value=250.0, step=10.0)
            
        with st.expander("Advanced Hematology Values (Standard Medians Used)"):
            pdw = st.number_input("PDW (Platelet Dist. Width)", value=14.31)
            pct = st.number_input("PCT (Plateletcrit)", value=0.26)
            lymp = st.number_input("LYMp (Lymphocyte %)", value=25.85)
            neutp = st.number_input("NEUTp (Neutrophil %)", value=77.51)
            leymn = st.number_input("LYMn (Lymphocyte count)", value=1.88)
            neutn = st.number_input("NEUTn (Neutrophil count)", value=5.14)
 
    with col_result:
        st.subheader("🔍 Prediction Results & Explainability")
        
        predict_clicked = st.button("RUN MULTIMODAL DIAGNOSTIC FUSION", type="primary", use_container_width=True)
        
        if predict_clicked:
            if input_image is None:
                st.error("Cannot run prediction: No patient eye image selected.")
            else:
                with st.spinner("Processing modalities and running attention-based fusion..."):
                    # Preprocess Image
                    img_tensor = cfg["val_transform"](input_image).unsqueeze(0).to(cfg["device"])
                    
                    # Preprocess Tabular Clinical Data
                    clin_df = pd.DataFrame([{
                        "Gender": gender_select, "region_type": region_select,
                        "WBC": wbc, "LYMp": lymp, "NEUTp": neutp, "LYMn": leymn, "NEUTn": neutn,
                        "RBC": rbc, "HCT": hct, "MCV": mcv, "MCH": mch, "MCHC": mchc, "PLT": plt_val, "PDW": pdw, "PCT": pct
                    }])
                    
                    # Separate numerical and categorical columns
                    NUMERIC_COLS = ["WBC", "LYMp", "NEUTp", "LYMn", "NEUTn", "RBC", "HCT", "MCV", "MCH", "MCHC", "PLT", "PDW", "PCT"]
                    CATEGORICAL_COLS = ["Gender", "region_type"]
                    
                    # Scale and encode features
                    x_num = cfg["scaler"].transform(clin_df[NUMERIC_COLS])
                    x_cat = cfg["encoder"].transform(clin_df[CATEGORICAL_COLS])
                    
                    # Convert to tensors
                    tensor_num = torch.tensor(x_num, dtype=torch.float32).to(cfg["device"])
                    tensor_cat = torch.tensor(x_cat, dtype=torch.long).to(cfg["device"])
                    
                    # Geo Features
                    state_idx = torch.tensor([cfg["state_to_idx"][selected_state]], dtype=torch.long).to(cfg["device"])
                    geo_risk = torch.tensor([[state_risk]], dtype=torch.float32).to(cfg["device"])
                    
                    # Forward Pass
                    with torch.no_grad():
                        # Modality specific embeddings
                        img_emb = cfg["img_classifier"].encoder(img_tensor)
                        clin_emb = cfg["clinical_classifier"].encoder(tensor_num, tensor_cat)
                        
                        # Fused transformer logits and attention weights
                        logits, attn_weights = cfg["fusion_model"](
                            img_emb, clin_emb, state_idx, geo_risk, return_attn=True
                        )
                        probs = torch.softmax(logits, dim=-1)[0]
                        pred_class = logits.argmax(dim=-1).item()
                        
                    prob_anemia = probs[1].item()
                    prob_healthy = probs[0].item()
                    
                    # Display prediction banner
                    if pred_class == 1:
                        st.markdown(f'<div class="result-banner-anemic">🚨 DIAGNOSIS: ANEMIC<br><span style="font-size:0.9rem;">Anemia Probability: {prob_anemia*100:.2f}%</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-banner-healthy">✅ DIAGNOSIS: NON-ANEMIC / HEALTHY<br><span style="font-size:0.9rem;">Healthy Probability: {prob_healthy*100:.2f}%</span></div>', unsafe_allow_html=True)
                        
                    # Progress representation
                    st.write("")
                    st.metric("Predicted Anemia Confidence", f"{prob_anemia*100:.1f}%")
                    st.progress(prob_anemia)
                    
                    # -----------------------------------------------------
                    # EXPLAINABILITY 1: Real-time Grad-CAM
                    # -----------------------------------------------------
                    st.write("")
                    st.markdown("### 👁️ Vision Branch Grad-CAM")
                    
                    target_layer = cfg["img_classifier"].encoder.backbone.features[-1]
                    cam_extractor = GradCAM(cfg["img_classifier"], target_layer)
                    
                    # Run backward hook to generate CAM heatmap
                    heatmap = cam_extractor.generate_heatmap(img_tensor, class_idx=pred_class)
                    cam_extractor.remove_hooks()
                    
                    # Resize heatmap to original image size
                    img_w, img_h = input_image.size
                    heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
                    
                    # Apply color map
                    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                    
                    # Combine original image with heatmap
                    img_np = np.array(input_image)
                    overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
                    
                    # Display original and overlay side-by-side
                    col_cam1, col_cam2 = st.columns(2)
                    with col_cam1:
                        st.image(input_image, caption="Input Conjunctiva", use_container_width=True)
                    with col_cam2:
                        st.image(overlay, caption="Grad-CAM Saliency Overlay (Diagnostic Region Focus)", use_container_width=True)
                    
                    # -----------------------------------------------------
                    # EXPLAINABILITY 2: Fusion Transformer Attention weights
                    # -----------------------------------------------------
                    st.write("")
                    st.markdown("### ⚡ Fusion Transformer Modality Attention")
                    
                    # attn_weights shape: (B, seq_len, seq_len)
                    # Extract the attention from the CLS token (token 0) to other tokens (1: Image, 2: Clinical, 3: Geo)
                    cls_attn = attn_weights[0, 0, 1:].cpu().numpy()
                    
                    # Normalize weights to sum to 100%
                    modality_weights = cls_attn / np.sum(cls_attn) * 100
                    
                    # Plot horizontal bar chart
                    fig, ax = plt.subplots(figsize=(6, 2.2))
                    labels = ["Ocular Image", "Tabular Clinical", "Geographical Prior"]
                    colors = ["#ff4b4b", "#4a90e2", "#8a2be2"]
                    
                    bars = ax.barh(labels, modality_weights, color=colors, height=0.55)
                    ax.set_xlim(0, 100)
                    ax.set_xlabel("Attention Weight (%)")
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_color('#8b949e')
                    ax.tick_params(colors='#8b949e')
                    
                    # Add percentages values to bars
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 2, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                                va='center', ha='left', color='#ffffff', fontweight='bold', fontsize=9)
                                
                    fig.patch.set_facecolor('none')
                    ax.set_facecolor('none')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown("""
                    <div style="font-size:0.95rem; color:#8b949e; margin-top:-10px;">
                        <i>The attention weights represent how the fusion transformer weighted each modality for this specific prediction. 
                        A high clinical score reflects reliance on CBC counts, while image attention highlights palpebral conjunctiva pallor details.</i>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👈 Enter the patient inputs in the panel and click the prediction button to execute.")

# ---------------------------------------------------------
# TAB 2: SYSTEM ARCHITECTURE VISUALIZATION
# ---------------------------------------------------------
with tab2:
    st.markdown('<div class="card-panel"><h3>🏗️ AnemiaFusionNet Model Architecture</h3>Details of the hybrid modal encoders and self-attention fusion block.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    #### ⚙️ Hybrid Multimodal Fusion Architecture
    The network is designed as an end-to-end differentiable pipeline in PyTorch. It consists of:
    1.  **Vision Encoder:** Pre-trained **EfficientNet-B0** serving as a backbone, projected to $256$-dimensions.
    2.  **Tabular Clinical Encoder:** An MLP representing numerical blood markers (CBC) and learnable embeddings for categorical features (Gender, region).
    3.  **Geo-Risk Encoder:** Combines state index embeddings with government-sourced continuous risk values (NFHS-5 database).
    4.  **Modality Fusion Transformer:** Sequence of Custom self-attention layers aligning embeddings per patient using standard `[CLS]` token pooling.
    """)
    
    # Render Mermaid diagram
    st.markdown("""
    #### 🧬 Data Flow Diagram
    """)
    st.components.v1.html(
        """
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({startOnLoad:true, theme:'dark'});</script>
        <div class="mermaid">
        graph TD
            subgraph Input Modalities
                I["Conjunctiva Image"]
                C["Clinical CBC Metrics"]
                G["Geographical State"]
            end

            subgraph Modality Encoders
                V_Enc["Vision Encoder: EfficientNet-B0"]
                T_Enc["Tabular Encoder: MLP & Embeddings"]
                L_Enc["Geo-Risk Encoder: MLP & Lookup"]
            end

            subgraph Multimodal Fusion
                CLS["CLS Token"]
                TypeEmb["Segment Modality-Type Embeddings"]
                SelfAttn["Multi-Head Self-Attention layers"]
            end

            I --> V_Enc
            C --> T_Enc
            G --> L_Enc

            V_Enc -->|256-d| SelfAttn
            T_Enc -->|256-d| SelfAttn
            L_Enc -->|256-d| SelfAttn
            CLS --> SelfAttn
            TypeEmb --> SelfAttn

            SelfAttn --> Fused["CLS Token Output"]
            Fused --> Head["MLP Classification Head"]
            Head --> Out["Prediction: Anemic vs. Healthy"]
        </div>
        """,
        height=400,
        scrolling=False
    )

# ---------------------------------------------------------
# TAB 3: PERFORMANCE METRICS & ABLATIONS
# ---------------------------------------------------------
with tab3:
    st.markdown('<div class="card-panel"><h3>📊 Ablation Study & Performance Curves</h3>Quantifying the contribution of each data modality.</div>', unsafe_allow_html=True)
    
    # Load ablation dataframe
    if os.path.exists("reports/ablation_study.csv"):
        ablation_df = pd.read_csv("reports/ablation_study.csv", index_col=0)
        st.subheader("🧪 Modality Ablation Table")
        st.dataframe(ablation_df, use_container_width=True)
    else:
        st.subheader("🧪 Modality Ablation Table")
        ablation_data = {
            "Model Configuration": ["Image-Only (EfficientNet-B0)", "Clinical-Only (Tabular MLP)", "Image + Clinical", "Full Fusion (AnemiaFusionNet)"],
            "Accuracy": ["42.42%", "90.91%", "87.88%", "87.88%"],
            "Precision": ["42.42%", "86.67%", "85.71%", "81.25%"],
            "Recall": ["100.00%", "92.86%", "85.71%", "92.86%"],
            "F1-Score": ["59.57%", "89.66%", "85.71%", "86.67%"],
            "ROC-AUC": ["50.00%", "89.85%", "91.73%", "93.98%"]
        }
        st.dataframe(pd.DataFrame(ablation_data), use_container_width=True)
        
    st.markdown("""
    #### 💡 Key Takeaways:
    *   **Clinical Dominance:** Individual blood markers (RBC, HCT, MCV) provide the strongest baseline predictive score (~89.8% AUC).
    *   **Image Signal Challenge:** Because the dataset is small (218 images), an image-only CNN overfits and acts as a majority classifier (~50% AUC).
    *   **Geo-Risk Synergy:** Incorporating geographical regional risk scores via the **ModalityFusionTransformer** boosts predictive confidence, achieving the highest ROC-AUC of **93.98%**.
    """)
    
    # ROC and Confusion Matrix side-by-side
    st.subheader("📈 Performance Validation Curves")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if os.path.exists("reports/roc_curve.png"):
            st.image("reports/roc_curve.png", caption="ROC-AUC Performance (Full Fusion vs. Modalities)", use_container_width=True)
    with col_c2:
        if os.path.exists("reports/confusion_matrix.png"):
            st.image("reports/confusion_matrix.png", caption="Confusion Matrix on Test Partition", use_container_width=True)

# Technical Report tab has been removed per user request
