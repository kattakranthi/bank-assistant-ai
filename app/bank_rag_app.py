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
# Hybrid Search Function (Day 2)
# -------------------------------
def hybrid_search(user_query, index, stored_texts, k=5):
    user_query_lower = user_query.lower()

    # 1️⃣ Exact Match Search (Transaction ID, Account ID, Merchant)
    exact_matches = [
        text for text in stored_texts
        if user_query_lower in text.lower()
    ]

    if exact_matches:
        return exact_matches[:k], "exact"

    # 2️⃣ Semantic Search (FAISS)
    query_embedding = get_embeddings_batch([user_query])
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)

    retrieved = [stored_texts[i] for i in indices[0]]

    return retrieved, "semantic"

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
    # -------------------------------
    # Chat Section (Enhanced)
    # -------------------------------
    st.subheader("Ask a Question")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Enter your question")

    if user_query:

        # 🔎 Hybrid Retrieval
        retrieved_texts, search_type = hybrid_search(user_query, index, stored_texts)

        context = "\n".join(retrieved_texts)

        st.write(f"🔍 Search Type Used: {search_type}")
        st.write("📄 Retrieved Context:")
        st.write(retrieved_texts)

        # 📊 Simple Analytics Feature
        if "total spending" in user_query.lower():
            total_spending = df[df["amount"] < 0]["amount"].sum()
            st.write(f"💰 Total Spending: {total_spending}")

        if "spending by category" in user_query.lower():
            category_summary = df.groupby("category")["amount"].sum()
            st.write("📊 Spending by Category:")
            st.write(category_summary)

        # 🧠 LLM Prompt
        prompt = f"""
    You are a bank assistant. Answer ONLY using the context below.

    Context:
    {context}

    Question:
    {user_query}
    """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        answer = response.choices[0].message.content

        st.session_state.chat_history.append(("User", user_query))
        st.session_state.chat_history.append(("Assistant", answer))

    # Display chat
    for role, message in st.session_state.chat_history:
        if role == "User":
            st.markdown(f"**🧑 {message}**")
        else:
            st.markdown(f"**🤖 {message}**")


else:
    st.info("Upload a CSV file to start.")