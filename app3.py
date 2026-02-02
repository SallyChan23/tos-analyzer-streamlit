import streamlit as st
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from torch.nn.functional import softmax
import joblib
import json
from docx import Document
import io
import re

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="ToS Analyzer",
    page_icon="📄",
    layout="wide"
)

# =========================
# Device setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Paths & settings
# =========================
MODELS_DIR = Path("models")
THRESHOLD = 0.40  # Fixed threshold

# =========================
# Low-info filtering helpers
# =========================
STOPWORDS = set()

GENERIC_SHORT = set([
    "privacy policy",
    "platform",
    "support for users",
    "press & media center"
])

def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

def is_low_info_hit(text: str, min_words: int = 10, min_chars: int = 40) -> bool:
    """
    Returns True if text is likely a heading/fragment that is not informative,
    even if the model is confident.
    """
    t = normalize_text(text)
    if not t:
        return True

    tl = t.lower()

    # 1) Generic heading exact match
    if tl in GENERIC_SHORT:
        return True

    # 2) Too short in chars
    if len(t) < min_chars:
        return True

    # 3) Too few word tokens (letters only)
    words = re.findall(r"[A-Za-z]+", t)
    if len(words) < min_words:
        return True

    # 4) Mostly stopwords -> not informative
    content_words = [w.lower() for w in words if w.lower() not in STOPWORDS]
    if len(content_words) <= 2:
        return True

    # 5) ALL CAPS short heading
    letters = re.findall(r"[A-Za-z]", t)
    if letters:
        upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
        if upper_ratio > 0.85 and len(words) <= 12:
            return True

    return False

# =========================
# Cache model loading
# =========================
@st.cache_resource
def load_models():
    """Load all three models and ensemble metadata"""

    # Load ensemble metadata
    with open(MODELS_DIR / "ensemble_meta_v2.json", "r") as f:
        meta = json.load(f)

    classes = meta["classes"]
    weights = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)

    models = {}

    # Load LegalBERT
    legal_path = MODELS_DIR / "LegalBERT_full_v2"
    models["legal_tok"] = AutoTokenizer.from_pretrained(str(legal_path), local_files_only=True)
    models["legal_model"] = AutoModelForSequenceClassification.from_pretrained(
        str(legal_path), local_files_only=True
    ).to(device).eval()

    # Load DistilBERT
    distil_path = MODELS_DIR / "DistilBERT_full_v2"
    models["distil_tok"] = AutoTokenizer.from_pretrained(str(distil_path), local_files_only=True)
    models["distil_model"] = AutoModelForSequenceClassification.from_pretrained(
        str(distil_path), local_files_only=True
    ).to(device).eval()

    # Load SBERT
    sbert_path = MODELS_DIR / "SBERT_full_v2"
    models["sbert"] = SentenceTransformer(str(sbert_path / "sbert_encoder"), device=str(device))
    models["sbert_clf"] = joblib.load(str(sbert_path / "sbert_lr_v2.joblib"))

    return models, classes, weights

# =========================
# Document parsing
# =========================
def extract_paragraphs_from_docx(file):
    """Extract paragraphs from uploaded .docx file"""
    doc = Document(io.BytesIO(file.read()))
    paragraphs = []
    for p in doc.paragraphs:
        text = normalize_text(p.text)
        if text:
            paragraphs.append(text)
    return paragraphs

# =========================
# Model inference helpers
# =========================
def predict_hf(texts, tok, model, batch_size=32, max_length=256):
    """Predict using HuggingFace models (LegalBERT, DistilBERT)"""
    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            all_logits.append(logits.cpu())

    logits = torch.cat(all_logits, dim=0)
    probs = softmax(logits, dim=-1).numpy()
    return probs

def predict_sbert(texts, sbert, clf, batch_size=256):
    """Predict using SBERT + Logistic Regression"""
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = sbert.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False
        )
        all_vecs.append(emb)

    X = np.vstack(all_vecs)
    probs = clf.predict_proba(X)
    return probs

def get_ensemble_predictions(texts, models, classes, weights):
    """Get ensemble predictions from all three models"""

    legal_probs = predict_hf(texts, models["legal_tok"], models["legal_model"])
    distil_probs = predict_hf(texts, models["distil_tok"], models["distil_model"])
    sbert_probs = predict_sbert(texts, models["sbert"], models["sbert_clf"])

    # (3, N, C)
    member_probs = np.stack([legal_probs, distil_probs, sbert_probs], axis=0)

    # Weighted ensemble => (N, C)
    ens_probs = np.tensordot(weights.numpy(), member_probs, axes=([0], [0]))
    return ens_probs

def format_violation_type(label):
    """Format violation label for display"""
    label_map = {
        "arbitration": "⚖️ Arbitration Clause",
        "content_license": "📝 Content License Issue",
        "limitation_liability": "🛡️ Limitation of Liability",
        "privacy": "🔒 Privacy Concern",
        "prohibited_use": "🚫 Prohibited Use",
        "termination": "❌ Termination Clause"
    }
    return label_map.get(label, label)

# =========================
# Main app
# =========================
def main():
    st.title("📄 Terms of Service Analyzer")
    st.markdown("Upload your Terms of Service or Terms & Conditions document to check for potentially problematic clauses.")

    # Load models
    with st.spinner("Loading AI models... (this may take a minute)"):
        models, classes, weights = load_models()

    st.success("✅ Ready to analyze documents!")

    # Info box
    with st.expander("ℹ️ What does this tool do?"):
        st.markdown("""
        This tool analyzes Terms of Service documents and identifies potentially problematic clauses in 6 categories:
        - **Arbitration**: Forced arbitration clauses
        - **Content License**: Broad content rights granted to the company
        - **Limitation of Liability**: Company limiting their responsibility
        - **Privacy**: Concerning privacy practices
        - **Prohibited Use**: Restrictive usage terms
        - **Termination**: Unfair termination conditions
        """)

    # Upload
    uploaded_file = st.file_uploader(
        "Choose a .docx file",
        type=["docx"],
        help="Upload a Terms of Service or Terms & Conditions document in .docx format"
    )

    if uploaded_file is None:
        return

    # Extract paragraphs
    with st.spinner("Reading document..."):
        paragraphs = extract_paragraphs_from_docx(uploaded_file)

    st.info(f"📄 Document contains {len(paragraphs)} paragraphs")

    if len(paragraphs) == 0:
        st.warning("No readable paragraphs found in the uploaded file.")
        return

    # Low-info filtering controls (podcast style: simple but adjustable)
    st.markdown("### 🧹 Noise control (optional)")
    colA, colB, colC = st.columns(3)
    with colA:
        show_low_info = st.checkbox("Show low-info hits", value=False)
    with colB:
        min_words = st.slider("Min words", 3, 25, 10)
    with colC:
        min_chars = st.slider("Min chars", 10, 200, 40)

    # Predictions
    with st.spinner("Analyzing document with AI models..."):
        ens_probs = get_ensemble_predictions(paragraphs, models, classes, weights)

    max_probs = ens_probs.max(axis=1)
    pred_ids = ens_probs.argmax(axis=1)

    # Filter violations + hide low-info by default
    violations = []
    low_info_hits = []

    for idx, (text, max_prob, pred_id) in enumerate(zip(paragraphs, max_probs, pred_ids)):
        if max_prob < THRESHOLD:
            continue

        violation_type = classes[pred_id]
        hit = {
            "paragraph": idx + 1,
            "type": violation_type,
            "text": text,
            "confidence": float(max_prob)
        }

        low_info = is_low_info_hit(text, min_words=min_words, min_chars=min_chars)
        if low_info and not show_low_info:
            low_info_hits.append(hit)
            continue

        violations.append(hit)

    # Display results
    st.header("📊 Analysis Results")

    total = len(paragraphs)
    violation_count = len(violations)
    safe_count = total - violation_count

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Paragraphs", total)
    col2.metric("✅ Safe", safe_count, delta=f"{safe_count/total*100:.1f}%")
    col3.metric("⚠️ Issues Found", violation_count, delta=f"{violation_count/total*100:.1f}%", delta_color="inverse")

    if len(low_info_hits) > 0 and not show_low_info:
        st.caption(f"🧹 Hid {len(low_info_hits)} low-info hit(s). Enable **Show low-info hits** to review them.")

    # Show violations
    if violation_count > 0:
        st.subheader("⚠️ Problematic Clauses Found")
        st.markdown(f"Found **{violation_count}** potentially problematic clause(s) in this document:")

        # Group by type
        violation_types = {}
        for v in violations:
            v_type = v["type"]
            violation_types.setdefault(v_type, []).append(v)

        # Display per category
        for v_type, items in violation_types.items():
            with st.expander(f"{format_violation_type(v_type)} ({len(items)} found)", expanded=True):
                for item in items:
                    st.markdown(f"**Paragraph {item['paragraph']}** (Confidence: {item['confidence']:.1%})")
                    st.info(item["text"])
                    st.markdown("---")

        # Summary table
        st.subheader("📋 Summary")
        summary_df = pd.DataFrame(
            [{"Violation Type": format_violation_type(k), "Count": len(v)} for k, v in violation_types.items()]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Download option
        csv_df = pd.DataFrame([{
            "Paragraph": v["paragraph"],
            "Violation Type": v["type"],
            "Confidence": f"{v['confidence']:.2%}",
            "Text": v["text"]
        } for v in violations])

        csv = csv_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Detailed Report (CSV)",
            data=csv,
            file_name=f"{uploaded_file.name}_violations.csv",
            mime="text/csv"
        )

    else:
        st.success("🎉 Great news! No problematic clauses detected in this document.")
        st.balloons()

        # Optional: if everything got filtered as low-info but threshold triggered,
        # you can still hint user to enable view.
        if len(low_info_hits) > 0 and not show_low_info:
            st.info("Some hits were filtered as low-info. Enable **Show low-info hits** if you want to review them.")

if __name__ == "__main__":
    main()
