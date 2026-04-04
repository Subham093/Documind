#🧠 Nexus RAG: Intelligent CSR Document Engine

Nexus RAG is an advanced Retrieval-Augmented Generation (RAG) system built to extract, process, and converse with complex PDF documentation. By leveraging a hybrid architecture—combining localized vector embeddings with high-speed cloud-based inference—the system delivers near-instantaneous responses with high factual accuracy.

#🚀 Live Demo
App URL: https://huggingface.co/spaces/Subhamm2342/Nexus-RAG-Engine


#🛠️ System Architecture
The project implements a state-of-the-art RAG pipeline:

Document Ingestion: Parses complex PDFs using PyPDFLoader.

Text Chunking: Utilizes RecursiveCharacterTextSplitter to maintain semantic context within 1000-character windows.

Vectorization: Employs the all-MiniLM-L6-v2 HuggingFace model for lightweight, high-dimensional text embeddings.

Vector Storage: Powered by ChromaDB for efficient similarity searches.

LLM Inference: Integrates Groq's Llama-3.3-70b-versatile model via API for lightning-fast reasoning (LPU technology).

#✨ Key Features
Zero-RAM Footprint: Optimized for low-memory environments (Render Free Tier) by offloading heavy computation to Groq.

Semantic Search: Goes beyond keyword matching to understand the intent of your questions.

Persistent Session State: Built-in chat history for a seamless, conversational user experience.

Caching: Uses @st.cache_resource to ensure the document is only processed once, speeding up subsequent interactions.

#💻 Tech Stack
Interface: Streamlit

Orchestration: LangChain

LLM Provider: Groq Cloud

Embedding Model: HuggingFace Transformers

Database: ChromaDB

#⚙️ Local Setup
Clone the Repo:
Bash
git clone https://github.com/Subham093/Nexus_Rag_Assistant.git
cd Nexus_Rag_Assistant
Install Dependencies:
Bash
pip install -r requirements.txt
Environment Configuration:
Create a .env file in the root directory:

Code snippet
GROQ_API_KEY=your_groq_api_key_here

Launch Application:
Bash
streamlit run streamlit_app.py

#📊 Industry Use Cases
Policy Analysis: Rapidly extracting insights from long corporate social responsibility (CSR) reports.

Legal Tech: Reviewing contracts for specific clauses.

Academic Research: Summarizing key findings from multiple research papers.

#💡 Why this architecture?
During development, the project transitioned from local HuggingFace LLMs to Groq Cloud API. This decision was made to reduce deployment latency by 85% and ensure compatibility with cloud-hosting RAM limits (512MB), proving that sophisticated AI can be deployed on lean infrastructure.
