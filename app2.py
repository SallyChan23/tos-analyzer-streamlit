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

# Set page config
st.set_page_config(
    page_title="ToS Analyzer",
    page_icon="📄",
    layout="wide"
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODELS_DIR = Path("models")
THRESHOLD = 0.40  # Fixed threshold

# Cache model loading
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
    models['legal_tok'] = AutoTokenizer.from_pretrained(str(legal_path), local_files_only=True)
    models['legal_model'] = AutoModelForSequenceClassification.from_pretrained(
        str(legal_path), local_files_only=True
    ).to(device).eval()
    
    # Load DistilBERT
    distil_path = MODELS_DIR / "DistilBERT_full_v2"
    models['distil_tok'] = AutoTokenizer.from_pretrained(str(distil_path), local_files_only=True)
    models['distil_model'] = AutoModelForSequenceClassification.from_pretrained(
        str(distil_path), local_files_only=True
    ).to(device).eval()
    
    # Load SBERT
    sbert_path = MODELS_DIR / "SBERT_full_v2"
    models['sbert'] = SentenceTransformer(str(sbert_path / "sbert_encoder"), device=str(device))
    models['sbert_clf'] = joblib.load(str(sbert_path / "sbert_lr_v2.joblib"))
    
    return models, classes, weights

def extract_paragraphs_from_docx(file):
    """Extract paragraphs from uploaded .docx file"""
    doc = Document(io.BytesIO(file.read()))
    paragraphs = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            paragraphs.append(text)
    return paragraphs

def predict_hf(texts, tok, model, batch_size=32, max_length=256):
    """Predict using HuggingFace models (LegalBERT, DistilBERT)"""
    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
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
        batch = texts[i:i+batch_size]
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
    
    # Predict with LegalBERT
    legal_probs = predict_hf(texts, models['legal_tok'], models['legal_model'])
    
    # Predict with DistilBERT
    distil_probs = predict_hf(texts, models['distil_tok'], models['distil_model'])
    
    # Predict with SBERT
    sbert_probs = predict_sbert(texts, models['sbert'], models['sbert_clf'])
    
    # Stack predictions (3 models x N paragraphs x 6 classes)
    member_probs = np.stack([legal_probs, distil_probs, sbert_probs], axis=0)
    
    # Weighted ensemble - combine all 3 models
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

# Main app
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
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a .docx file",
        type=['docx'],
        help="Upload a Terms of Service or Terms & Conditions document in .docx format"
    )
    
    if uploaded_file is not None:
        # Extract paragraphs
        with st.spinner("Reading document..."):
            paragraphs = extract_paragraphs_from_docx(uploaded_file)
        
        st.info(f"📄 Document contains {len(paragraphs)} paragraphs")
        
        if len(paragraphs) > 0:
            # Get predictions
            with st.spinner("Analyzing document with AI models..."):
                ens_probs = get_ensemble_predictions(paragraphs, models, classes, weights)
            
            # Process results
            max_probs = ens_probs.max(axis=1)
            pred_ids = ens_probs.argmax(axis=1)
            
            # Filter violations only
            violations = []
            for idx, (text, max_prob, pred_id) in enumerate(zip(paragraphs, max_probs, pred_ids)):
                if max_prob >= THRESHOLD:
                    violation_type = classes[pred_id]
                    violations.append({
                        "paragraph": idx + 1,
                        "type": violation_type,
                        "text": text,
                        "confidence": max_prob
                    })
            
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
            
            # Show violations
            if violation_count > 0:
                st.subheader("⚠️ Problematic Clauses Found")
                st.markdown(f"Found **{violation_count}** potentially problematic clause(s) in this document:")
                
                # Group by violation type
                violation_types = {}
                for v in violations:
                    v_type = v['type']
                    if v_type not in violation_types:
                        violation_types[v_type] = []
                    violation_types[v_type].append(v)
                
                # Display by category
                for v_type, items in violation_types.items():
                    with st.expander(f"{format_violation_type(v_type)} ({len(items)} found)", expanded=True):
                        for item in items:
                            st.markdown(f"**Paragraph {item['paragraph']}** (Confidence: {item['confidence']:.1%})")
                            st.info(item['text'])
                            st.markdown("---")
                
                # Summary table
                st.subheader("📋 Summary")
                summary_data = []
                for v_type, items in violation_types.items():
                    summary_data.append({
                        "Violation Type": format_violation_type(v_type),
                        "Count": len(items)
                    })
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
            else:
                st.success("🎉 Great news! No problematic clauses detected in this document.")
                st.balloons()
            
            # Download option (detailed CSV for reference)
            if violation_count > 0:
                csv_data = []
                for v in violations:
                    csv_data.append({
                        "Paragraph": v['paragraph'],
                        "Violation Type": v['type'],
                        "Confidence": f"{v['confidence']:.2%}",
                        "Text": v['text']
                    })
                csv_df = pd.DataFrame(csv_data)
                csv = csv_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Detailed Report (CSV)",
                    data=csv,
                    file_name=f"{uploaded_file.name}_violations.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
