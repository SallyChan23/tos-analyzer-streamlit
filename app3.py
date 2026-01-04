Skip to content
Chat history




You said:
saya mau lanjut ke permasalahan streamlit
ChatGPT said:
Siap 👍
Kita lanjut ke permasalahan Streamlit.

Langsung aja, biar efisien dan kontekstual:

➡️ Masalah Streamlit apa yang sedang kamu hadapi sekarang?
Pilih salah satu (atau jelaskan langsung kalau beda):

App tidak jalan / error

ModuleNotFoundError

StreamlitAPIException

error pas deploy (local jalan, cloud tidak)

State & logic

st.session_state bingung / reset terus

Button ditekan tapi state tidak nyimpan

Multi step (wizard) tapi lompat lompat

Performance

Lama banget load model (U-Net, SegFormer, dsb)

GPU tidak kepakai

Tiap interaction malah retrain / reload model

Layout & UX

Sidebar vs main layout

Image + metrics tidak rapi

Tabel, plot, atau metric card berantakan

Integrasi ML pipeline

Load .pth / .pt model

Inference per image

Thresholding, mask, metrics (Dice, F1, AUC)

Risk Decision stage (patch → image-wise)

Deployment

Streamlit Cloud / HuggingFace / local server

Environment mismatch

File path Kaggle vs Streamlit

Cara kita bahas (biar nggak overload)
Kita pakai gaya podcast teknis pelan seperti sebelumnya:

Kamu kirim potongan kode / error / screenshot

Aku jelaskan:

apa yang seharusnya terjadi

apa yang sekarang terjadi

kenapa Streamlit sering “terlihat benar tapi salah”

Baru kita betulkan satu titik saja, bukan semuanya

👉 Kirimkan:

error message atau

potongan app.py / cell Streamlit atau

ceritakan flow: user klik apa → harusnya apa → yang terjadi apa

Aku siap lanjut dari situ.

You said:
Uploaded image
Uploaded image
Uploaded image
Uploaded image
disini masalahnya adalah pas sy di run , ada seperti beberapa kata2 yang sebenarnya ga terlalu penting seperti yang kurang dari 10 kata dan tidak ada hubungan karena tidak menjelaskana apa2 ini bagusnya diapakan? 

apakah kasih naik threshold aja (sekarang pakai 0.4 karena dari normal distributionnya)
atau kita buat batasan di app.py di github bahwa jika jumlaha kata kurang tidak usah di kasih masuk

atau ada cara lain?
ChatGPT said:
Ngerti. Itu “hasil kecil kecil” (misalnya cuma “PRIVACY POLICY”, “Platform.”, “Support for users;”) biasanya bukan bug model, tapi post processing / UI yang terlalu “literal”: model nge-flag apa pun yang mirip pola, walau informasi hukumnya nol.

Kalau pilihannya: naikkan threshold vs filter di app.py, aku jauh lebih condong ke:

Pilihan terbaik: tetap threshold 0.4, tapi tambahkan “informativeness filter” + UI toggle
Kenapa?

Naik threshold = kamu bisa kehilangan temuan penting yang memang pendek tapi relevan (mis. “terminate immediately”, “without notice”, “liability cap” kadang pendek).

Filter rule = kamu buang noise tanpa mengorbankan recall model.

Strategi yang aku sarankan (praktis, paling stabil)
1) Filter berbasis panjang (minimum words / chars)
Contoh aturan:

buang kalau word_count < 10 atau char_count < 40

Tapi ini sendirian kadang buang yang penting.

2) Filter berbasis “informasi”
Buang kalau teks:

mayoritas non-huruf (punctuation) / terlalu pendek

hanya ALL CAPS pendek (“PRIVACY POLICY”)

hanya 1–2 kata generik (“Platform.”)

hampir semua stopwords (“and / or / the / to …”)

3) UX yang enak: jangan dihapus total, tapi di-hide
Jadi:

default: hide low-info hits

ada checkbox: Show low-info hits (kalau user mau audit)

Implementasi simpel di app.py (langsung kepake)
import re

STOPWORDS = set("""
a an the and or but if then else for to of in on at by with without from as
is are was were be been being this that these those it its their his her
""".split())

GENERIC_SHORT = set([
    "privacy policy", "platform", "support for users", "press & media center"
])

def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip())

def is_low_info_hit(text: str, min_words=10, min_chars=40) -> bool:
    t = normalize_text(text)
    tl = t.lower()

    # 1) sangat pendek
    if len(t) < min_chars:
        return True

    # 2) kata terlalu sedikit
    words = re.findall(r"[A-Za-z]+", t)
    if len(words) < min_words:
        return True

    # 3) generic phrase match (exact-ish)
    if tl in GENERIC_SHORT:
        return True

    # 4) mayoritas stopwords (tidak informatif)
    content_words = [w.lower() for w in words if w.lower() not in STOPWORDS]
    if len(content_words) <= 2:   # kebanyakan stopwords / kosong
        return True

    # 5) ALL CAPS pendek (biasanya heading)
    letters = re.findall(r"[A-Za-z]", t)
    if letters:
        upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
        if upper_ratio > 0.85 and len(words) <= 12:
            return True

    return False
Lalu saat kamu mau render hasil highlight:

show_low_info = st.checkbox("Show low-info hits", value=False)

filtered = []
for hit in hits:  # hit: {text, confidence, paragraph_id, category, ...}
    if (not show_low_info) and is_low_info_hit(hit["text"], min_words=10, min_chars=40):
        continue
    filtered.append(hit)
Kalau kamu tetap mau pakai “threshold naik”
Boleh, tapi aku sarankan jangan global.

Lebih aman:

threshold tetap 0.4 untuk semua

tapi untuk kategori yang sering noise (mis. “privacy concern”), pakai threshold lebih tinggi (mis. 0.55)

Contoh pola:

THRESH_BY_CLASS = {
  "privacy": 0.55,
  "termination": 0.45,
  "arbitration": 0.45,
  "limitation_liability": 0.45,
}

thr = THRESH_BY_CLASS.get(hit["category"], 0.4)
if hit["confidence"] < thr:
    continue
Saran final (yang paling “engineering correct”)
✅ Tetap threshold 0.4 (sesuai normal distribution kamu)
✅ Tambahkan low-info filter di app.py
✅ Jangan delete total: hide by default + toggle
✅ Opsional: class-specific threshold untuk kategori yang noisy

Kalau kamu kirim potongan struktur hits / output JSON yang dipakai Streamlit (1 contoh saja), aku bisa pasangin filter ini persis ke format datamu (biar tinggal copy paste).

You said:
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

ini code streamlit ku apa yang harus diubah?
ChatGPT said:
Sip, jelas. Masalah “PRIVACY POLICY / Platform.” itu terjadi karena kamu filter hanya pakai confidence threshold (max_prob >= 0.40). Jadi paragraf pendek/heading tetap lolos kalau model yakin.

Yang perlu diubah: bukan modelnya, tapi post processing sebelum masuk violations.append().

Di bawah ini aku kasih perubahan paling aman + enak UX:

1) Tambahkan “low-info filter” + toggle (hide by default)
A. Tambahkan ini di atas (setelah format_violation_type boleh)
import re

STOPWORDS = set("""
a an the and or but if then else for to of in on at by with without from as
is are was were be been being this that these those it its their his her
""".split())

GENERIC_SHORT = set([
    "privacy policy", "platform", "support for users", "press & media center"
])

def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip())

def is_low_info_hit(text: str, min_words=10, min_chars=40) -> bool:
    t = normalize_text(text)
    tl = t.lower()

    # generic headings
    if tl in GENERIC_SHORT:
        return True

    # too short
    if len(t) < min_chars:
        return True

    # word count (letters only)
    words = re.findall(r"[A-Za-z]+", t)
    if len(words) < min_words:
        return True

    # mostly stopwords -> not informative
    content_words = [w.lower() for w in words if w.lower() not in STOPWORDS]
    if len(content_words) <= 2:
        return True

    # ALL CAPS short heading
    letters = re.findall(r"[A-Za-z]", t)
    if letters:
        upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
        if upper_ratio > 0.85 and len(words) <= 12:
            return True

    return False
B. Tambahkan kontrol di UI (setelah metrics / sebelum “Show violations” enak)
Taruh setelah bagian st.header("📊 Analysis Results") atau sebelum group-by:

colA, colB, colC = st.columns(3)
with colA:
    show_low_info = st.checkbox("Show low-info hits", value=False)
with colB:
    min_words = st.slider("Min words", 3, 25, 10)
with colC:
    min_chars = st.slider("Min chars", 10, 200, 40)
C. Ubah loop filter violations (bagian ini yang utama)
Ganti blok ini:

for idx, (text, max_prob, pred_id) in enumerate(zip(paragraphs, max_probs, pred_ids)):
    if max_prob >= THRESHOLD:
        violation_type = classes[pred_id]
        violations.append({...})
jadi:

violations = []
low_info_hits = []  # optional list, for auditing

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

    # low-info filtering
    low_info = is_low_info_hit(text, min_words=min_words, min_chars=min_chars)
    if low_info and not show_low_info:
        low_info_hits.append(hit)
        continue

    violations.append(hit)
D. Optional: tampilkan berapa yang di-hide (bagus buat transparansi)
Setelah metrics:

if 'low_info_hits' in locals() and len(low_info_hits) > 0 and not show_low_info:
    st.caption(f"🧹 Hid {len(low_info_hits)} low-info hit(s) (enable 'Show low-info hits' to review).")
2) Alternatif yang lebih “legal”: class-specific threshold (opsional)
Kalau kamu merasa noise paling banyak dari privacy, kamu bisa kombinasikan:

Tambahkan di atas:

THRESH_BY_CLASS = {
    "privacy": 0.55,
    "termination": 0.45,
    "arbitration": 0.45,
    "limitation_liability": 0.45,
    "content_license": 0.45,
    "prohibited_use": 0.45
}
Lalu di loop:

violation_type = classes[pred_id]
thr = THRESH_BY_CLASS.get(violation_type, THRESHOLD)
if max_prob < thr:
    continue
Tapi ini aku anggap opsional; filter low-info biasanya sudah cukup.

Jawaban ke pertanyaan kamu: “naik threshold atau batasan kata?”
✅ Batasan kata + low-info filter lebih aman daripada menaikkan threshold global.
Karena threshold global bisa “membunuh” clause penting yang pendek.

Kalau kamu mau aku “rapihin” lebih jauh: kita bisa juga hapus paragraf yang cuma heading sejak awal di extract_paragraphs_from_docx() (pre-filter), tapi aku sengaja taruh di post-processing dulu supaya kamu masih bisa audit dan tidak kehilangan data.

Kalau kamu kirim 5 contoh paragraf pendek yang sering muncul (copy paste teksnya), aku bisa bikin rule “heading detector” yang lebih tajam untuk dataset ToS kamu.

You said:
kita pakai yang cara pertama saja
tolong ubah semua codeku sesuai agar siap pakai saja
ChatGPT said:
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
STOPWORDS = set("""
a an the and or but if then else for to of in on at by with without from as
is are was were be been being this that these those it its their his her
""".split())

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

