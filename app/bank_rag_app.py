import app
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import pickle
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------------
# Setup
# -------------------------------
load_dotenv()
client = OpenAI()

st.set_page_config(page_title="Bank Assistant AI", layout="wide")
st.title("Bank Assistant AI")

INDEX_PATH = "embeddings/faiss.index"
METADATA_PATH = "embeddings/metadata.pkl"


# -------------------------------
# Utility: Batch Embeddings
# -------------------------------
def get_embeddings_batch(texts, batch_size=50):
    embeddings = []
    print(texts)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)

    return np.array(embeddings).astype("float32")


# -------------------------------
# Build FAISS Index
# -------------------------------
def build_faiss_index(texts):
    embeddings = get_embeddings_batch(texts)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save to disk
    faiss.write_index(index, INDEX_PATH)

    # Save metadata
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(texts, f)

    return index


# -------------------------------
# Load FAISS Index
# -------------------------------
def load_faiss_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            texts = pickle.load(f)
        return index, texts
    return None, None


# -------------------------------
# Upload Data
# -------------------------------
uploaded_file = st.file_uploader("Upload bank statements CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Convert rows to text
    texts = [
        f"""
    Transaction ID: {row['transaction_id']}
    Customer Name: {row['customer_name']}
    Account ID: {row['account_id']}
    Date: {row['date']}
    Transaction Type: {row['transaction_type']}
    Merchant: {row['merchant']}
    Category: {row['category']}
    Amount: {row['amount']}
    Balance: {row['balance']}
    City: {row['city']}
    """
        for _, row in df.iterrows()
    ]

    # Check if index exists
    index, stored_texts = load_faiss_index()

    if index is None:
        st.info("Building FAISS index (first time only)...")
        index = build_faiss_index(texts)
        stored_texts = texts
        st.success("Index built and saved.")
    else:
        st.success("Loaded existing FAISS index.")

    # -------------------------------
    # Chat Section
    # -------------------------------
    st.subheader("Ask a Question")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Enter your question")

    if user_query:
        # Embed query
        query_embedding = get_embeddings_batch([user_query])
        faiss.normalize_L2(query_embedding)

        # Search
        k = 5
        distances, indices = index.search(query_embedding, k)

        retrieved_texts = [stored_texts[i] for i in indices[0]]

        context = "\n".join(retrieved_texts)

        st.write("User Query:", retrieved_texts)


        prompt = f"""
You are a bank assistant. Answer the question using ONLY the context below.

Context:
{context}

Question:
{user_query}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful bank assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        answer = response.choices[0].message.content

        st.session_state.chat_history.append(("User", user_query))
        st.session_state.chat_history.append(("Assistant", answer))

    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == "User":
            st.markdown(f"**🧑 {message}**")
        else:
            st.markdown(f"**🤖 {message}**")

else:
    st.info("Upload a CSV file to start.")