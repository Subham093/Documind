import os
import warnings
import shutil

# 1. Silence Python deprecation alerts
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 2. Silence Hugging Face Hub log messages and token warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_LOG_LEVEL"] = "ERROR"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ─────────────────────────────────────────
# 1. SETUP
# ─────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="Nexus RAG Engine", page_icon="🧠", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(BASE_DIR, "temp_data")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

# ─────────────────────────────────────────
# 2. MODELS
# ─────────────────────────────────────────
@st.cache_resource
def load_resources():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return llm, embeddings

llm, embeddings = load_resources()

# ─────────────────────────────────────────
# 3. STATE MANAGEMENT
# ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ─────────────────────────────────────────
# 4. SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.title("📂 Nexus Control")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        if st.button("🚀 Process & Index"):
            try:
                with st.spinner("Nexus is reading the document..."):

                    # Step 1 — Save uploaded PDF to temp_data folder
                    file_path = os.path.join(TMP_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Step 2 — Windows safe fix: clear old vectorstore
                    if st.session_state.vector_store is not None:
                        st.session_state.vector_store.delete_collection()
                        st.session_state.vector_store = None

                    # Step 3 — Load PDF and split into chunks
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=80
                    )
                    chunks = text_splitter.split_documents(docs)

                    # Step 4 — Create fresh vector store
                    st.session_state.vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=PERSIST_DIR
                    )
                    st.success(f"Successfully indexed: {uploaded_file.name}")

            except Exception as e:
                st.error(f"Processing Error: {str(e)}")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ─────────────────────────────────────────
# 5. RAG LOGIC
# ─────────────────────────────────────────
def get_response(user_query, vector_store):

    # Part A — Rewrite question to be standalone (include history context)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, vector_store.as_retriever(), contextualize_q_prompt
    )

    # Part B — Answer question using retrieved context
   # STRONGER — forces LLM to stay inside the PDF only
    system_prompt = (
        "You are Nexus AI. You ONLY answer questions based on the context provided below. "
        "If the answer is NOT found in the context, you MUST say exactly: "
        "'I cannot find this information in the uploaded document.' "
        "Do NOT use your own knowledge. Do NOT make up answers. "
        "Use three sentences maximum. "
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain.invoke({
        "input": user_query,
        "chat_history": st.session_state.chat_history
    })

# ─────────────────────────────────────────
# 6. CHAT UI
# ─────────────────────────────────────────
st.title("🧠 Nexus RAG Engine")

# Display existing chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Handle new user input
if query := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vector_store:
        response = get_response(query, st.session_state.vector_store)
        assistant_answer = response["answer"]

        with st.chat_message("assistant"):
            st.markdown(assistant_answer)

        # Save to history
        st.session_state.chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=assistant_answer),
        ])
    else:
        st.warning("Please upload a file first.")
