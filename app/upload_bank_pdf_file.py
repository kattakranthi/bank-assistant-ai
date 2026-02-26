import streamlit as st
import json
import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

PDF_JSON_FOLDER = "documents"
os.makedirs(PDF_JSON_FOLDER, exist_ok=True)

def extract_text_from_pdf(uploaded_pdf):
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    print(full_text)
    return full_text

def extract_structured_data(pdf_text):
    prompt = f"""
    Extract the following fields from this bank statement.

    Return valid JSON only with this schema:
    {{
        "account_number": "",
        "statement_date": "",
        "total_balance": ""
        
    }}

    Document:
    {pdf_text}
    """
    print("pdf data ",pdf_text)
    print("prompt ", prompt)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract structured banking data."},
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "bank_statement",
                "schema": {
                    "type": "object",
                    "properties": {
                        "account_number": {"type": "string"},
                        "statement_date": {"type": "string"},
                        "total_balance": {"type": "string"}
                    },
                    "required": ["account_number", "statement_date", "total_balance"]
                }
            }},
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except:
        st.error("Failed to parse JSON from LLM.")
        return None

def save_json(data):
    account = data.get("account_number", "unknown")
    date = data.get("statement_date", "unknown")
    filename = f"{PDF_JSON_FOLDER}/{account}_{date}.json"

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    return filename

def chunk_text(text, chunk_size=800):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def get_embeddings_batch(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embeddings = [e.embedding for e in response.data]
    return np.array(embeddings).astype("float32")


def build_faiss_index(text_chunks):
    embeddings = get_embeddings_batch(text_chunks)

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index

st.subheader("Upload Bank PDF")

uploaded_pdf = st.file_uploader("Upload Bank Statement PDF", type=["pdf"])

if uploaded_pdf:

    st.info("Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(uploaded_pdf)

    st.success("Text extracted.")

    st.info("Extracting structured data using AI...")
    structured_data = extract_structured_data(pdf_text)

    if structured_data:
        st.json(structured_data)

        json_file = save_json(structured_data)
        st.success(f"Saved structured JSON to {json_file}")

        st.info("Creating embeddings for RAG...")

        chunks = chunk_text(pdf_text)
        index = build_faiss_index(chunks)

        st.success("FAISS index built for this document.")