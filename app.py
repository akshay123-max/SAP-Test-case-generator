import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from PyPDF2 import PdfReader
from docx import Document

# --- Page Setup ---
st.set_page_config(page_title="SAP Test Case Generator", layout="wide")

st.markdown("""
# SAP Test Case Generator
Upload files or type a scenario description, then generate test cases.
""")

# --- Load Models ---
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline(
        "text-generation",
        model="sshleifer/tiny-gpt2",
        max_new_tokens=200
    )
    return embedder, generator

embedder, generator = load_models()

# --- File Handling Functions ---
def extract_text_from_file(file):
    name = file.name.lower()
    text = ""
    if name.endswith(".txt"):
        text = file.read().decode("utf-8")
    elif name.endswith(".csv"):
        df = pd.read_csv(file)
        text = " ".join(df.astype(str).fillna("").values.flatten())
    elif name.endswith(".xlsx"):
        df = pd.read_excel(file)
        text = " ".join(df.astype(str).fillna("").values.flatten())
    elif name.endswith(".pdf"):
        reader = PdfReader(file)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    elif name.endswith(".docx"):
        doc = Document(file)
        text = "\n".join([p.text for p in doc.paragraphs])
    else:
        text = ""
    return text

# --- User Input ---
uploaded_files = st.file_uploader(
    "📄 Upload any file type", 
    type=["txt","csv","xlsx","pdf","docx"], 
    accept_multiple_files=True
)
single_desc = st.text_area("📝 Scenario Description (optional)")

# --- Process Input ---
scenarios = []

if uploaded_files:
    for f in uploaded_files:
        txt = extract_text_from_file(f)
        if txt.strip():
            scenarios.append(txt.strip())

if single_desc.strip():
    scenarios.append(single_desc.strip())

if not scenarios:
    st.info("Upload a file or enter text to generate test cases.")

# --- Generate Test Cases ---
if st.button("🚀 Generate Test Cases") and scenarios:
    results = []
    all_texts = scenarios.copy()
    
    # --- Compute embeddings for all input texts ---
    embeddings = embedder.encode(all_texts, convert_to_tensor=True)

    for idx, desc in enumerate(scenarios):
        st.info(f"Generating test case {idx+1}/{len(scenarios)}...")
        # --- Retrieve similar context if needed ---
        context = ""
        if len(all_texts) > 1:
            # find top 1 similar
            cosine_scores = util.cos_sim(embedder.encode(desc), embeddings)[0]
            top_idx = int(np.argmax(cosine_scores))
            if top_idx != idx:
                context = all_texts[top_idx]
        
        prompt = f"Generate a step-by-step SAP test case.\nDescription: {desc}\nContext: {context}\nFormat: Testcase Name | Step Number | Step Description | Step Instruction | Expected Result"
        
        output = generator(prompt, max_new_tokens=200, do_sample=True)[0]['generated_text']
        results.append({"Description": desc, "Test Case": output})
    
    df_out = pd.DataFrame(results)
    st.dataframe(df_out)
    
    # --- Excel Download ---
    towrite = io.BytesIO()
    df_out.to_excel(towrite, index=False)
    towrite.seek(0)
    st.download_button("⬇️ Download Excel", towrite, file_name="SAP_Test_Cases.xlsx")
