# 🧠 Nexus RAG: Production-Ready Document Intelligence System  

Nexus RAG is a production-oriented Retrieval-Augmented Generation (RAG) system designed to extract, process, and interact with complex PDF documents. It combines efficient local vector retrieval with ultra-fast cloud-based LLM inference to deliver accurate, context-aware responses with low latency.

---

## 🚀 Live Demo  
https://huggingface.co/spaces/Subhamm2342/Nexus-RAG-Engine  

---

## 🖼️ Application Preview  

<!-- INSERT UI SCREENSHOT HERE -->
<!-- Example: assets/ui.png -->

> Add a screenshot of your Streamlit interface showing a question + answer

---

## 🏗️ System Architecture  

<!-- INSERT ARCHITECTURE DIAGRAM HERE -->
<!-- Example: assets/architecture.png -->

The system follows a modular RAG pipeline:

PDF → Text Chunking → Embeddings → Vector Store → Retriever → LLM → Response  

### Components:

- **Document Ingestion**  
  Parses PDFs using PyPDFLoader  

- **Chunking Strategy**  
  RecursiveCharacterTextSplitter (~1000 chars) to preserve semantic meaning  

- **Embedding Model**  
  all-MiniLM-L6-v2 (HuggingFace) for efficient vectorization  

- **Vector Store**  
  ChromaDB for similarity search  

- **Retriever**  
  Top-k similarity-based retrieval (k configurable)  

- **LLM Layer**  
  LLaMA-3.3-70B via Groq API for fast inference  

---

## 🔄 System Workflow  

<!-- INSERT FLOW DIAGRAM HERE -->
<!-- Example: assets/workflow.png -->

1. User uploads or queries a document  
2. Document is split into semantic chunks  
3. Chunks are converted into embeddings  
4. Stored in ChromaDB  
5. User query is embedded  
6. Top-k relevant chunks are retrieved  
7. Context is injected into LLM prompt  
8. LLM generates final response  

---

## ⚙️ Core Implementation Details  

### 🔍 Retrieval Strategy  
- Uses cosine similarity search  
- Retrieves top-k relevant chunks  
- Ensures context grounding  

### 🧠 Prompt Engineering  
- Combines:
  - User query  
  - Retrieved context  
- Structured prompt to minimize hallucination  

### ⚡ Performance Optimization  
- Offloads LLM inference to Groq Cloud  
- Uses `@st.cache_resource` to avoid recomputation  
- Reduces repeated embedding generation  

---

## ✨ Key Features  

- 🚀 Low-Latency Responses (Groq LPU acceleration)  
- 🧩 Semantic Search (context-aware retrieval)  
- 💬 Conversational Memory (session-based chat history)  
- 🧠 Efficient Resource Usage (runs on low-RAM systems)  
- ⚡ Smart Caching for faster repeated queries  

---

## 📊 Performance & Evaluation  

| Metric                  | Observation |
|------------------------|------------|
| Response Latency       | ~1–2 seconds |
| Latency Improvement    | ~85% (vs local LLM) |
| Memory Usage           | < 512MB (cloud-compatible) |

> Note: Metrics may vary based on document size and API latency  

---

## ⚠️ Limitations  

- Single-document processing (no multi-doc retrieval yet)  
- Limited handling of tables and complex PDF layouts  
- No source citation highlighting  
- Depends on external API (Groq)  

---

## 📊 Use Cases  

- 📑 Policy & CSR Analysis  
- ⚖️ Legal Document Review  
- 🎓 Academic Research Summarization  

---

## 🧰 Tech Stack  

- Frontend: Streamlit  
- Orchestration: LangChain  
- Embeddings: HuggingFace Transformers  
- Vector Database: ChromaDB  
- LLM Provider: Groq Cloud  

---

## 🚀 Deployment  

- Hosted on HuggingFace Spaces  
- Designed for low-memory environments (≤512MB RAM)  
- Uses cloud inference to bypass local hardware limits  

---

## ⚙️ Local Setup  

### 1. Clone Repository  
git clone https://github.com/Subham093/Nexus_Rag_Assistant.git  
cd Nexus_Rag_Assistant  

### 2. Install Dependencies  
pip install -r requirements.txt  

### 3. Setup Environment Variables  
Create `.env` file:

GROQ_API_KEY=your_groq_api_key_here  

### 4. Run Application  
streamlit run streamlit_app.py  

---

## 📌 Future Improvements  

- Multi-document querying  
- Source attribution / citations  
- Hybrid search (keyword + semantic)  
- Fine-tuned domain-specific embeddings  

---

## 🤝 Contributing  

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## ⭐ Support  

If you found this project useful, consider giving it a star ⭐
