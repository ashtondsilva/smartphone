# 📱 Smartphone Recommendation RAG App (Gemini 1.5 Flash)

This project is a Retrieval-Augmented Generation (RAG) system that helps users find the best smartphones based on natural language queries, such as:

> "I want a phone under ₹10,000 with decent battery and not too many features."

It uses vector search with FAISS and Google Gemini 1.5 Flash to retrieve the most relevant smartphones from a dataset and generate natural language responses.

---

## 🚀 Features

- 🔍 Semantic search over smartphone specs using FAISS
- 🧠 Smart responses generated via Gemini 1.5 Flash
- 🗃️ Works with 900+ rows of real smartphone data
- 🧾 Customizable column selection and embedding format
- ⚡ Streamlit UI for interactive querying

---

## 🧠 Tech Stack

- Python 3.10+
- Pandas
- FAISS (for vector search)
- SentenceTransformers (`all-MiniLM-L6-v2`)
- Google Gemini 1.5 Flash (via API)
- Streamlit

---

## 🗂️ Directory Structure

```
smartphone-rag/
│
├── app.py                      # Main Streamlit app
├── utils.py                    # Helper functions
├── requirements.txt            # Python dependencies
├── .env                        # Gemini API key (DO NOT commit this)
│
├── data/
│   └── smartphone.csv          # Your dataset (900 rows, 11 columns)
│
└── embeddings/
    └── faiss_index.index       # FAISS index (auto-generated)
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/smartphone-rag.git
   cd smartphone-rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Gemini API key**

   Create a `.env` file:
   ```bash
   echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## ✍️ Sample Queries

- "Suggest a phone under 15000 rupees with good battery and camera"
- "I need a lightweight phone with less than 6.2 inch screen"
- "Find me a phone launched after 2022 with 8GB RAM"

---

## 🧪 Example Dataset Columns

- Company Name  
- Model Name  
- Mobile Weight  
- RAM  
- Front Camera  
- Back Camera  
- Processor  
- Battery Capacity  
- Screen Size  
- Launched Price (India)  
- Launched Year  

---

## 🙏 Acknowledgments

- [Google Gemini API](https://ai.google.dev/)
- [FAISS - Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [SentenceTransformers](https://www.sbert.net/)
- Streamlit for the UI

---

## 📜 License

MIT License
