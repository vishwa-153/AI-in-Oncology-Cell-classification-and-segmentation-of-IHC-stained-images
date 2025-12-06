# app.py
import os
import io
import tempfile
import traceback

import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import timm
import numpy as np
from fpdf import FPDF
from cellpose import models
import cv2

# Local imports (your custom modules)
from LLM import LLM_Analysis  # must read GEMINI_API_KEY from env inside LLM.py
from histogpt.models.ctranspath import swin_tiny_patch4_window7_224, ConvStem

# Hugging Face helper
from huggingface_hub import hf_hub_download

# --------------------------
# Config / constants
# --------------------------
MODEL_REPO = "vishwa-153/Cell-classification-and-segmentation-of-IHC-stained-images"
MODEL_FILENAME = "best_swin_tiny_model.pth"
CTRANS_FILENAME = "ctranspath.pth"
MODEL_CACHE_DIR = "models"

CLASS_NAMES = ['Immune cells', 'Necrosis', 'Other', 'Stroma', 'Tumor', 'alveoli', 'background']
COLORS = np.array([
    [255, 0, 0], [0, 255, 0], [0, 0, 255],
    [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 128, 128]
])

HISTO_PARAGRAPH = """This report presents an AI-assisted histopathological analysis of immunohistochemically (IHC) stained tissue, specifically from lung or colon samples. The integration of SWIN TRANSFORMER MODEL enables automated classification of cellular and tissue components, including immune cells, tumor regions, necrotic areas, stroma, alveolar structures, and background. Such classification supports detailed assessment of the tumor microenvironment (TME), immune infiltration, and pathological progression. This approach aims to enhance diagnostic precision and accelerate interpretation workflows in clinical oncology and digital pathology."""

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Model download helper
# --------------------------
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def ensure_models():
    """
    Download model files from HF Hub into MODEL_CACHE_DIR if missing.
    If your model repo is private, set HF_TOKEN in Space secrets (env var).
    """
    hf_token = os.environ.get("HF_TOKEN", None)
    global MODEL_PATH, CTRANS_PATH

    try:
        # model
        local_model_path = os.path.join(MODEL_CACHE_DIR, MODEL_FILENAME)
        if not os.path.exists(local_model_path):
            st.write("Downloading model weights from HF Hub...")
            MODEL_PATH = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILENAME,
                cache_dir=MODEL_CACHE_DIR,
                repo_type="model",
                token=hf_token
            )
        else:
            MODEL_PATH = local_model_path

        # ctrans
        local_ctrans_path = os.path.join(MODEL_CACHE_DIR, CTRANS_FILENAME)
        if not os.path.exists(local_ctrans_path):
            st.write("Downloading ctrans weights from HF Hub...")
            CTRANS_PATH = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=CTRANS_FILENAME,
                cache_dir=MODEL_CACHE_DIR,
                repo_type="model",
                token=hf_token
            )
        else:
            CTRANS_PATH = local_ctrans_path

        st.write("MODEL_PATH:", MODEL_PATH)
        st.write("CTRANS_PATH:", CTRANS_PATH)

    except Exception as e:
        st.error("Failed to download models from HF Hub. Check HF_TOKEN (if private) and repo name.")
        st.exception(e)
        raise

# run early so load_models can use paths
ensure_models()

# --------------------------
# Model loading
# --------------------------
@st.cache_resource
def load_models():
    """
    Load swin classification model, cellpose model, and feature_extractor.
    Raises RuntimeError if files are missing so logs are clear in Spaces.
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(CTRANS_PATH)):
        raise RuntimeError(f"Missing model files. MODEL_PATH exists? {os.path.exists(MODEL_PATH)}, CTRANS_PATH exists? {os.path.exists(CTRANS_PATH)}")

    # classification model
    swin_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=len(CLASS_NAMES), drop_rate=0.1)
    state = torch.load(MODEL_PATH, map_location=device)
    # allow state dict being wrapped in 'model' key
    if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
        state = state['model']
    swin_model.load_state_dict(state)
    swin_model.to(device).eval()

    # cellpose model (may be heavy on CPU)
    try:
        cellpose_model = models.Cellpose(gpu=torch.cuda.is_available(), model_type='cyto')
    except Exception as e:
        st.warning("Cellpose failed to initialize. Segmentation may not work on this environment.")
        st.exception(e)
        cellpose_model = None

    # feature extractor (ctrans)
    feature_extractor = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
    feature_extractor.head = torch.nn.Identity()

    state_dict = torch.load(CTRANS_PATH, map_location=device)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    feature_extractor.load_state_dict(state_dict)
    feature_extractor.to(device).eval()

    return swin_model, cellpose_model

# --------------------------
# Transforms
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# --------------------------
# Segmentation + Classification
# --------------------------
def run_segmentation_model(image: Image.Image):
    """
    Perform segmentation (cellpose) and classification (swin).
    Returns segmented PIL image, predicted class label, and confidence score.
    """
    try:
        swin_model, cellpose_model = load_models()
    except Exception as e:
        st.error("Model loading failed.")
        st.exception(e)
        raise

    image_np = np.array(image)

    # Segment (if cellpose loaded)
    masks = None
    if cellpose_model is not None:
        try:
            masks, _, _, _ = cellpose_model.eval(image_np, diameter=40, channels=[0, 0], do_3D=False)
        except Exception as e:
            st.warning("Cellpose segmentation failed for this image.")
            st.exception(e)
            masks = None
    else:
        st.info("Segmentation skipped (Cellpose unavailable).")

    # Classify
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = swin_model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    # Draw contours if masks available
    segmented_img = image_np.copy()
    if masks is not None:
        for mask_id in np.unique(masks)[1:]:
            mask = masks == mask_id
            if np.sum(mask) > 10:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(segmented_img, contours, -1, COLORS[pred].tolist(), 2)

    # fallback: if no masks, return original image as segmented image
    return Image.fromarray(segmented_img), CLASS_NAMES[pred], conf

# --------------------------
# PDF generator
# --------------------------
def generate_pdf_report(original_img, segmented_img, eval_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Histopathology Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, HISTO_PARAGRAPH)
    pdf.ln(5)

    # Save images temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as orig_tmp:
        original_img.save(orig_tmp.name)
        orig_path = orig_tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as seg_tmp:
        segmented_img.save(seg_tmp.name)
        seg_path = seg_tmp.name

    # Insert original image
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Original Image:", ln=True)
    pdf.image(orig_path, x=10, w=90)
    pdf.ln(10)

    # Insert segmented image
    pdf.cell(0, 10, "Segmented Image:", ln=True)
    pdf.image(seg_path, x=10, w=90)
    pdf.ln(10)

    # Insert evaluation and LLM explanation
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Evaluation Results:", ln=True)
    pdf.set_font("Arial", size=10)
    safe_text = eval_text.encode('latin-1', errors='replace').decode('latin-1')
    pdf.multi_cell(0, 8, safe_text)

    # Clean up temp files
    try:
        os.remove(orig_path)
        os.remove(seg_path)
    except Exception:
        pass

    return pdf.output(dest='S').encode('latin1')

# --------------------------
# Streamlit UI
# --------------------------
def main():
    st.set_page_config(layout="wide")
    st.title("Classification and segmentation of cells in IHC stained tissue images")

    col1, col2 = st.columns([2, 2])

    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            try:
                original_img = Image.open(uploaded_file).convert("RGB")
            except Exception as e:
                st.error("Failed to open the uploaded image.")
                st.exception(e)
                return

            st.image(original_img, caption="Uploaded Image", width=200)

            if st.button("Run Model"):
                with st.spinner("Processing..."):
                    try:
                        segmented_img, pred_class, conf = run_segmentation_model(original_img)
                        segmented_img.save("output.png")
                        st.subheader("Segmented Image")
                        st.image(segmented_img, caption="Segmented Image", width=200)

                        # Save state
                        st.session_state["segmented_img"] = segmented_img
                        st.session_state["original_img"] = original_img
                        st.session_state["pred_class"] = pred_class
                        st.session_state["conf"] = conf

                        # Call LLM for explanation; wrap in try/except so errors don't break UI
                        try:
                            llm_result = LLM_Analysis("output.png", pred_class, f"{conf:.4f}")
                        except Exception as e:
                            llm_result = f"LLM call failed: {e}"
                            st.warning("LLM call failed; continuing without LLM explanation.")
                            st.exception(e)

                        st.session_state["llm_result"] = llm_result

                    except Exception as e:
                        st.error("Model run failed.")
                        st.exception(e)

    with col2:
        if "pred_class" in st.session_state:
            st.header("Prediction Summary")
            st.write(f"**Predicted class:** {st.session_state['pred_class']}")
            st.write(f"**Model confidence:** {st.session_state['conf']:.2f}")
            st.markdown("**Diagnosis**")
            st.write(st.session_state.get("llm_result", "No LLM result."))

            # Compose formatted text
            eval_text = (
                f"Model Name: Swin Tiny\n"
                f"Prediction: {st.session_state['pred_class']}\n"
                f"Confidence: {st.session_state['conf']:.2f}\n\n"
                f"{st.session_state.get('llm_result', '')}"
            )

            # Generate and show download button
            try:
                st.session_state["pdf_bytes"] = generate_pdf_report(
                    st.session_state["original_img"],
                    st.session_state["segmented_img"],
                    eval_text
                )

                st.download_button(
                    label="Download Report PDF",
                    data=st.session_state["pdf_bytes"],
                    file_name="histopathology_report.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error("Failed to generate PDF.")
                st.exception(e)


if __name__ == "__main__":
    main()
