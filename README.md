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

## Day 2 Enhancements

- Create a StreamLit Dashboard to upload the pdf file
- PDF data is extracted 
- PDF Extraction data is given to LLM as prompt
- LLM Returns the Structured Data
- The Structured Data is saved into Json.
---
## Day 3 Enhancements

- Refactor OpenAI integration to use AsyncOpenAI with proper await handling

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

#Run the upload app
streamlit run app/upload_bank_pdf_file.py

#Debug the app

streamlit run app/bank_rag_app.py --logger.level=debug


