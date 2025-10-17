import streamlit as st
import pandas as pd
import numpy as np
import io, time, os, re
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from langdetect import detect
import fitz  # PyMuPDF
import docx

st.set_page_config(
    page_title="SAP Test Case Generator",
    page_icon="https://www.inovace-solutions.com/wp-content/uploads/2023/04/cropped-favicon-32x32.png",
    layout="wide"
)

# --- Header ---
st.markdown("""
<style>
.header {display:flex; align-items:center; background-color:#003366; padding:10px; border-radius:10px;}
.header img {height:50px; margin-right:15px;}
.header h1 {color:white; font-size:26px; margin:0;}
</style>
<div class="header">
    <img src="https://www.inovace-solutions.com/wp-content/uploads/2023/04/cropped-favicon-32x32.png"/>
    <h1>SAP Test Case Generator</h1>
</div>
""", unsafe_allow_html=True)

# --- Modelle ---
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    generator = pipeline("text-generation", model="google/gemma-2b-it", device_map="auto", max_new_tokens=500)
    return embedder, generator

embedder, generator = load_models()

# --- Dateiinhalt extrahieren ---
def extract_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            pdf = fitz.open(stream=file.read(), filetype="pdf")
            for page in pdf: text += page.get_text()
        elif ext in [".docx"]:
            doc = docx.Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext in [".txt", ".csv", ".json", ".xml"]:
            text = file.read().decode("utf-8", errors="ignore")
        elif ext in [".xlsx"]:
            df = pd.read_excel(file)
            text = "\n".join(df.astype(str).fillna("").values.flatten())
        else:
            text = ""
    except Exception as e:
        st.warning(f"⚠️ Fehler beim Lesen {file.name}: {e}")
    return text

# --- Eingabe ---
uploaded_files = st.file_uploader("📂 Beliebige Dateien hochladen", type=["pdf","docx","txt","csv","xlsx","json","xml"], accept_multiple_files=True)
user_text = st.text_area("📝 Szenario Beschreibung (optional)")

# --- Verarbeitung ---
if st.button("🚀 Testfälle generieren"):
    all_text = ""

    if uploaded_files:
        for f in uploaded_files:
            all_text += extract_text(f) + "\n"
    if user_text.strip():
        all_text += "\n" + user_text.strip()

    if not all_text.strip():
        st.error("Bitte eine Datei hochladen oder Text eingeben.")
    else:
        st.info("🔍 Verarbeite Daten...")
        start = time.time()

        # Kleine Chunks erstellen
        chunks = re.split(r'(?<=[.!?]) +', all_text)
        vectors = embedder.encode(chunks, convert_to_tensor=True)

        # Ähnliche SAP Themen simulieren
        sap_refs = ["SAP Testautomatisierung", "SAP Fiori App Testing", "SAP Best Practices", "SAP Module Integration"]
        sap_vecs = embedder.encode(sap_refs, convert_to_tensor=True)
        sims = util.cos_sim(vectors.mean(0), sap_vecs)
        best_topic = sap_refs[int(np.argmax(sims))]

        # Generierung
        prompt = f"""
Du bist SAP QA Experte. Erstelle basierend auf folgendem Kontext strukturierte Testfälle im Tabellenformat:

Kontext:
{all_text[:3000]}

SAP Thema: {best_topic}

Format:
Testcase Name | Step Number | Step Description | Step Instruction | Expected Result
"""
        output = generator(prompt, temperature=0.3, do_sample=False)[0]["generated_text"]

        # Ergebnisse speichern
        df = pd.DataFrame([{"Beschreibung": user_text or "Dateiupload", "Testfall": output, "SAP Thema": best_topic}])
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        buf.seek(0)

        st.success("✅ Testfälle erfolgreich generiert!")
        st.download_button("⬇️ Excel herunterladen", buf, file_name="SAP_Testfaelle.xlsx")

        st.markdown("### 📋 Vorschau")
        st.text_area("Ergebnis", output[:3000])

        st.caption(f"⏱️ Dauer: {time.time()-start:.2f}s")
