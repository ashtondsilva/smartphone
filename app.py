import os
import pandas as pd
import faiss
import numpy as np
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# === CONFIG ===
CSV_PATH = "data/smartphone.csv.csv"
INDEX_PATH = "embeddings/faiss_index.index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # local SBERT for embedding
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"  # replace this!

# === STEP 1: Load and Preprocess Data ===
@st.cache_data
def load_data():
    use_cols = ['Company Name','Model Name','Mobile Weight','RAM','Front Camera','Back Camera','Processor','Battery Capacity','Screen Size','Launched Price (India)','Launched Year']
    df = pd.read_csv(CSV_PATH, usecols=use_cols,encoding='ISO-8859-1')
    df.fillna("Unknown", inplace=True)
    return df


def row_to_text(row):
    return (
        f"Company: {row['Company Name']}. "
        f"Model: {row['Model Name']}. "
        f"Weight: {row['Mobile Weight']}g. "
        f"RAM: {row['RAM']}. "
        f"Front Camera: {row['Front Camera']}. "
        f"Back Camera: {row['Back Camera']}. "
        f"Processor: {row['Processor']}. "
        f"Battery: {row['Battery Capacity']} mAh. "
        f"Screen Size: {row['Screen Size']} inches. "
        f"Launched Price in India: ₹{row['Launched Price (India)']}. "
        f"Launched Year: {row['Launched Year']}."
    )


@st.cache_data
def convert_to_texts(df):
    return df.apply(row_to_text, axis=1).tolist()

# === STEP 2: Embeddings + Vector Index ===
@st.cache_resource
def build_faiss_index(docs):
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = embedder.encode(docs, show_progress_bar=True)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_PATH)
    return index, embedder

@st.cache_resource
def load_faiss_index(docs):
    if os.path.exists(INDEX_PATH):
        embedder = SentenceTransformer(EMBED_MODEL_NAME)
        index = faiss.read_index(INDEX_PATH)
        return index, embedder
    else:
        return build_faiss_index(docs)

# === STEP 3: Gemini Setup ===
def init_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")

def query_rag(user_query, docs, index, embedder, k=5):
    query_embedding = embedder.encode([user_query])
    D, I = index.search(np.array(query_embedding), k)
    retrieved_docs = [docs[i] for i in I[0]]
    context = "\n".join(retrieved_docs)

    prompt = (
        f"You are a smartphone recommender system. Based on this smartphone data:\n"
        f"{context}\n\n"
        f"Answer the following question:\n{user_query}"
    )

    gemini = init_gemini()
    response = gemini.generate_content(prompt)
    return response.text

# === STEP 4: UI ===
def main():
    st.set_page_config(page_title="Smartphone Recommender with Gemini")
    st.title("📱 Smartphone Recommender (RAG + Gemini)")

    df = load_data()
    docs = convert_to_texts(df)
    index, embedder = load_faiss_index(docs)

    user_query = st.text_input("Ask something like: 'Phone under ₹10,000 with low weight and basic features'")
    if user_query:
        with st.spinner("Thinking..."):
            answer = query_rag(user_query, docs, index, embedder)
            st.success(answer)

if __name__ == "__main__":
    main()
