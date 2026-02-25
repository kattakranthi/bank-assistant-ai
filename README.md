# Bank Assistant AI

**Bank Assistant AI** is a Retrieval-Augmented Generation (RAG) chatbot for bank statements.  
It allows users to ask questions about transactions, balances, and account activity using AI powered by **FAISS vector search** and **OpenAI LLMs**, 
all within a **Streamlit dashboard**.

---

## Features

## Day 2 Enhancements

- Hybrid retrieval (Exact match + FAISS semantic search)
- Transaction ID and Account ID direct lookup
- Spending summary analytics
- Improved retrieval debugging

---

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API Key 
- Libraries in `requirements.txt`

### Create .env file in root folder 
OPENAI_API_KEY=give the open ai api key
get the open ai api key from the Open AI

### Install Dependencies
pip install -r requirements.txt

#Run the app
streamlit run app/bank_rag_app.py

#Debug the app

streamlit run app/bank_rag_app.py --logger.level=debug
