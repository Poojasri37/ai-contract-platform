import streamlit as st
import numpy as np
from typing import List, Tuple
import os
import fitz # PyMuPDF
import shutil
import hashlib
import urllib.parse
import re 
from datetime import datetime, timedelta
import pandas as pd # NEW: Added for dashboard data processing
from dashboard import load_css, create_professional_header, show_executive_dashboard, show_profit_optimizer, prepare_dashboard_data # Import dashboard logic

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# --- App Constants ---
DB_SAVE_PATH = "multi_document_db"
DOCS_FOLDER = "source_documents"
SUMMARY_FOLDER = "summary"

# --------------------------------------------------------------------------
# VectorDB Class (Core Logic Unchanged)
# --------------------------------------------------------------------------
class VectorDB:
    def __init__(self, dim: int, embedding_model):
        self.dim = dim
        self.embedding_model = embedding_model
        self.vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.texts: List[str] = []
        self.ids: List[str] = []
        self.doc_hashes: dict = {}

    def add_document(self, doc_id: str, text_content: str, content_hash: str, chunk_size=1000, chunk_overlap=100):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(text_content)
        if not chunks: return False
        embeddings = self.embedding_model.embed_documents(chunks)
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            self.vectors = np.vstack([self.vectors, np.array(embedding, dtype=np.float32)])
            self.texts.append(chunk); self.ids.append(chunk_id)
        self.doc_hashes[doc_id] = content_hash
        return True

    def get_content_and_check_duplicate(self, file_path: str):
        try:
            doc = fitz.open(file_path); text_content = "".join(page.get_text() for page in doc); doc.close()
            if not text_content.strip(): return None, None, None
        except Exception: return None, None, None
        content_hash = hashlib.sha256(text_content.encode('utf-8')).hexdigest()
        existing_doc_id = next((doc_id for doc_id, hash_val in self.doc_hashes.items() if hash_val == content_hash), None)
        return text_content, content_hash, existing_doc_id

    def search(self, query_text: str, k: int = 3, doc_id: str = None):
        if len(self.vectors) == 0: return []
        query_vector = self.embedding_model.embed_query(query_text)
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        search_vectors, original_indices_map = (self.vectors, list(range(len(self.ids))))
        if doc_id:
            filtered_indices = [i for i, id_str in enumerate(self.ids) if id_str.startswith(doc_id)]
            if not filtered_indices: return []
            search_vectors, original_indices_map = (self.vectors[filtered_indices], filtered_indices)
        norms = np.linalg.norm(search_vectors, axis=1, keepdims=True)
        vec_normed = search_vectors / (norms + 1e-8)
        q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        sims = vec_normed @ q_norm.T
        top_idx_filtered = np.argsort(-sims.squeeze())[:k]
        final_indices = [original_indices_map[i] for i in top_idx_filtered]
        return [(self.ids[i], float(sims[top_idx_filtered[j]][0]), self.texts[i]) for j, i in enumerate(final_indices)]

    def save(self, folder_path: str):
        if not os.path.exists(folder_path): os.makedirs(folder_path)
        np.save(os.path.join(folder_path, "vectors.npy"), self.vectors)
        with open(os.path.join(folder_path, "metadata.txt"), "w", encoding="utf-8") as f:
            for id_, text in zip(self.ids, self.texts): f.write(f"{id_}\t{text.replace('\n', ' ')}\n")
        with open(os.path.join(folder_path, "hashes.txt"), "w", encoding="utf-8") as f:
            for doc_id, hash_val in self.doc_hashes.items(): f.write(f"{doc_id}\t{hash_val}\n")

    def load(self, folder_path: str):
        paths = [os.path.join(folder_path, name) for name in ["vectors.npy", "metadata.txt", "hashes.txt"]]
        if not all(os.path.exists(p) for p in paths): return False
        self.vectors = np.load(paths[0])
        with open(paths[1], "r", encoding="utf-8") as f:
            for line in f: parts = line.strip().split("\t", 1); self.ids.append(parts[0]); self.texts.append(parts[1])
        with open(paths[2], "r", encoding="utf-8") as f:
            for line in f: parts = line.strip().split("\t", 1); self.doc_hashes[parts[0]] = parts[1]
        return True

# --------------------------------------------------------------------------
# LLM/Summary Helper Functions
# --------------------------------------------------------------------------

@st.cache_resource
def load_models():
    """Load and cache the embedding and chat models, using the enhanced prompt."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # ENHANCED PROMPT: Now includes directives for Alerts and Flags
    SUMMARY_SYSTEM_PROMPT = (
        "You are an expert at extracting and summarizing key contractual information. "
        "Answer the question based ONLY on the provided context, focusing on being concise. "
        "In addition to the standard summary data, you MUST specifically look for and extract two key items: "
        "1. **Alerts**: Highlight any specific dates or timeframes that signify contract expiration or renewal. "
        "2. **Flags**: Note any clauses that are highly unusual, restrictive, or punitive compared to standard agreements. "
        "Provide the concise summary, followed by the specific Alert and Flag details."
    )
    
    summary_prompt_template = ChatPromptTemplate.from_template(
        SUMMARY_SYSTEM_PROMPT +
        "\n\nContext:\n{context}\n\nQuestion:\n{question}"
    )
    summary_chain = summary_prompt_template | llm | StrOutputParser()
    
    return embeddings, llm, summary_chain

def generate_and_save_summary(doc_id, db, summary_chain):
    """Generates a structured summary for a single document and saves it."""
    
    summary_topics = {
        "Margins": "Extract key numerical values for distributor margins, commissions, or discount rates. Be very brief.",
        "Service Timelines": "Extract key details about service timelines, support levels (SLAs), or uptime guarantees. Be concise.",
        # Crucial for Alerts: LLM is prompted to give a clean date
        "Expiration/Renewal Date": "Extract the single most important contract expiration or renewal date. Respond ONLY with the date in YYYY-MM-DD format, or 'N/A' if none is found. Example: 2025-12-31",
        "Unusual Clauses (Flags)": "Identify any highly restrictive covenants, termination penalties, or non-standard legal terms. Be concise.", # NEW FLAG TOPIC
        "Selling Rights": "Extract key details about territory rights, specifying the region and if they are exclusive."
    }
    
    summary_content = f"Concise Summary for Document: {doc_id}\n" + "=" * (30 + len(doc_id)) + "\n\n"
    
    for topic, question in summary_topics.items():
        search_results = db.search(question, k=4, doc_id=doc_id)
        context_text = "\n\n---\n\n".join([doc[2] for doc in search_results])
        
        if not context_text.strip():
            answer = "N/A" if "Date" in topic else "No specific details found."
        else:
            answer = summary_chain.invoke({"context": context_text, "question": question})
            
        summary_content += f"## {topic}\n{answer}\n\n"
            
    if not os.path.exists(SUMMARY_FOLDER): os.makedirs(SUMMARY_FOLDER)
    file_path = os.path.join(SUMMARY_FOLDER, f"summary_of_{doc_id}.txt")
    with open(file_path, "w", encoding="utf-8") as f: f.write(summary_content)
    
    return summary_content

@st.cache_data
def get_summary(doc_id):
    """Load a pre-generated summary from the summary folder."""
    summary_path = os.path.join(SUMMARY_FOLDER, f"summary_of_{doc_id}.txt")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return "Error reading summary file."
    return "No summary could be found for this document. Please click 'Process New Documents (Rebuild)'."

# --------------------------------------------------------------------------
# Alerts Feature Functions (Uses Summary Data)
# --------------------------------------------------------------------------

def extract_dates_from_summary(summary_content):
    """
    Scans the summary for the 'Expiration/Renewal Date' section and extracts a YYYY-MM-DD date.
    Returns a list of dictionaries: [{'description': ..., 'date': ...}]
    """
    alerts = []
    # Regex to find the date within the designated section
    match = re.search(r"## Expiration/Renewal Date\n(.*?)\n\n", summary_content, re.DOTALL)
    
    if match:
        date_text = match.group(1).strip()
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", date_text)
        
        if date_match:
            date_str = date_match.group(1)
            alerts.append({
                "description": "Contract Expiration/Renewal Date",
                "date": date_str
            })
            
    return alerts

def calculate_time_left(target_date_str):
    """Calculates time difference and formats the output."""
    try:
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
        now = datetime.now()
        delta = target_date - now
        
        if delta.total_seconds() < 0:
            days_ago = abs(delta.days)
            return f"🚨 Past Due by {days_ago} days" if days_ago > 0 else "🚨 DUE TODAY"
        else:
            days_left = delta.days
            if days_left == 0: return "⚠️ DUE TODAY"
            elif days_left <= 30: return f"🔥 {days_left} days left"
            else: return f"✅ {days_left} days left"
            
    except ValueError:
        return "❌ Invalid Date Format"

def display_alert_page(available_docs):
    """Renders the Alerts tab content using data from the summaries."""
    st.header("Document Timeline Alerts")
    
    st.markdown("Comparing document deadlines with the current system time (**" + datetime.now().strftime('%Y-%m-%d') + "**):")
    
    found_alerts = False
    
    for doc_id in available_docs:
        summary_content = get_summary(doc_id)
        timelines = extract_dates_from_summary(summary_content)
        
        if timelines:
            found_alerts = True
            with st.expander(f"**📄 {doc_id}** (Timeline Found)", expanded=True):
                st.markdown("---")
                for timeline in timelines:
                    target_date_str = timeline['date']
                    description = timeline['description']
                    time_status = calculate_time_left(target_date_str)
                    
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1: st.markdown(f"**Date:** `{target_date_str}`")
                    with col2: st.markdown(f"**Status:** {time_status}")
                    with col3: st.markdown(f"**Details:** {description}")
                st.markdown("---")
    
    if not found_alerts:
        st.info("No actionable timelines (in YYYY-MM-DD format) were found in the document summaries.")

# --------------------------------------------------------------------------
# Main Application Logic
# --------------------------------------------------------------------------

def main():
    # --- Load CSS and set page config ---
    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="📄", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    load_css() # Load the professional styling

    # --- URL Parameter Check (Handles separate summary page) ---
    params = st.query_params
    if "page" in params and params["page"] == "summary":
        doc_id_encoded = params.get("doc_id", [None])[0]
        if doc_id_encoded:
            doc_id = urllib.parse.unquote(doc_id_encoded)
            st.set_page_config(page_title=f"Summary: {doc_id}", layout="wide")
            st.title(f"📄 Summary for: {doc_id}")
            st.markdown(get_summary(doc_id))
        else:
            st.error("Document ID not provided in the URL.")
        return

    # --- Main App Initialization ---
    embeddings, llm, summary_chain = load_models()
    chat_chain = ChatPromptTemplate.from_template("Answer based on Context:\n\n{context}\n\nQuestion:\n{question}") | llm | StrOutputParser()
    
    if "db" not in st.session_state:
        db = VectorDB(dim=384, embedding_model=embeddings)
        if db.load(DB_SAVE_PATH): st.session_state.db = db; st.toast("Database loaded from disk.")
        else: st.session_state.db = db

    db = st.session_state.db
    available_docs = sorted(list(set([id.split('_chunk_')[0] for id in db.ids])))
    
    # --- Professional Header ---
    create_professional_header(title="📄 AI Contract Assistant", subtitle="RAG-Powered Document Intelligence")

    # --- Sidebar for Controls and Document List ---
    with st.sidebar:
        st.header("Controls")
        
        # --- File Upload ---
        uploaded_files = st.file_uploader("Upload new PDF documents", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            if not os.path.exists(DOCS_FOLDER): os.makedirs(DOCS_FOLDER)
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DOCS_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
            st.success(f"{len(uploaded_files)} file(s) uploaded to `{DOCS_FOLDER}`! Click 'Process' to add them.")

        # --- Document Processing ---
        if st.button("🔄 Process New Documents (Rebuild)"):
            st.session_state.messages = [] 
            processed_count = 0
            
            with st.spinner("Rebuilding database and generating summaries..."):
                if os.path.exists(DB_SAVE_PATH): shutil.rmtree(DB_SAVE_PATH)
                if os.path.exists(SUMMARY_FOLDER): shutil.rmtree(SUMMARY_FOLDER)
                
                new_db = VectorDB(dim=384, embedding_model=embeddings)
                pdf_files = [f for f in os.listdir(DOCS_FOLDER) if f.lower().endswith(".pdf")]
                
                for pdf_file in pdf_files:
                    file_path = os.path.join(DOCS_FOLDER, pdf_file)
                    doc_id = os.path.splitext(pdf_file)[0]
                    text_content, content_hash, existing_doc_id = new_db.get_content_and_check_duplicate(file_path)
                    
                    if text_content is None: continue
                    
                    if existing_doc_id:
                        st.warning(f"Content of '{pdf_file}' is identical to '{existing_doc_id}'.")
                        duplicate_decision = st.radio(
                            f"Duplicate of '{existing_doc_id}'. Process '{doc_id}'?",
                            ('Skip', f'Process as new ({doc_id})'),
                            key=f'dup_radio_{doc_id}',
                            index=0
                        )
                        if duplicate_decision == 'Skip': continue
                                
                    if new_db.add_document(doc_id, text_content, content_hash):
                        generate_and_save_summary(doc_id, new_db, summary_chain)
                        processed_count += 1
                                
                if new_db.ids: new_db.save(DB_SAVE_PATH)
                
                st.session_state.db = new_db
                st.toast(f"Database rebuilt successfully! {processed_count} documents processed.")
                st.rerun()

        st.header("Available Documents")
        if not available_docs:
            st.info("No documents have been processed yet.")
        else:
            for doc in available_docs:
                safe_doc_id = urllib.parse.quote(doc)
                link_html = f'<a href="/?page=summary&doc_id={safe_doc_id}" target="_blank" style="text-decoration: none; color: #80a3ff;">📄 {doc}</a>'
                st.markdown(link_html, unsafe_allow_html=True)


    # --- Dashboard Data Preparation (for visualization modules) ---
    df_contracts = prepare_dashboard_data(available_docs, get_summary)

    # --- Main Content Tabs/Navigation ---
    # ADDED Dashboard and Profit Optimizer modules
    module = st.selectbox(
        "Select Module:",
        [
            "📊 Executive Dashboard",
            "💬 Chat Assistant",
            "📰 Summarizer", 
            "🆚 Compare Documents",
            "🚨 Alerts",
            "💰 Profit Optimizer"
        ],
        label_visibility="collapsed"
    )

    # --- Module Routing ---
    if module == "📊 Executive Dashboard":
        show_executive_dashboard(df_contracts)

    elif module == "💰 Profit Optimizer":
        show_profit_optimizer(df_contracts)
    
    elif module == "💬 Chat Assistant":
        st.header("💬 Chat with Your Documents")
        if "messages" not in st.session_state: st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    query_doc_id, user_question = None, prompt
                    if " in " in prompt.lower():
                        parts = prompt.split(" in "); potential_doc_name = parts[-1].strip()
                        available_docs_lower = {doc.lower(): doc for doc in available_docs}
                        if potential_doc_name.lower() in available_docs_lower:
                            query_doc_id = available_docs_lower[potential_doc_name.lower()]; user_question = " in ".join(parts[:-1]) 
                    search_results = db.search(user_question, k=5, doc_id=query_doc_id)
                    context_text = "\n\n---\n\n".join([f"From '{id.split('_chunk_')[0]}':\n{text}" for id, score, text in search_results])
                    if not context_text: context_text = "No relevant context was found."
                    answer = chat_chain.invoke({"context": context_text, "question": user_question})
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
    elif module == "📰 Summarizer":
        st.header("📰 Document Summaries")
        if not available_docs:
            st.info("Please process documents first to view summaries here.")
        else:
            selected_doc = st.selectbox("Select a Document to View its Summary:", options=available_docs, key="summary_selector")
            if selected_doc:
                st.markdown(get_summary(selected_doc))

    elif module == "🆚 Compare Documents":
        st.header("🆚 Compare Two Documents")
        if len(available_docs) < 2:
            st.warning("You need at least two processed documents to use the compare feature.")
        else:
            col1, col2 = st.columns(2)
            with col1: doc1 = st.selectbox("Select Document 1", options=available_docs, key="doc1")
            with col2: doc2 = st.selectbox("Select Document 2", options=available_docs, key="doc2")
            if st.button("Generate Side-by-Side Summary Comparison"):
                summary1 = get_summary(doc1); summary2 = get_summary(doc2)
                st.subheader(f"Side-by-Side Summary: {doc1} vs. {doc2}")
                colA, colB = st.columns(2)
                with colA: st.markdown(f"**{doc1}**"); st.markdown(summary1)
                with colB: st.markdown(f"**{doc2}**"); st.markdown(summary2)
                
    elif module == "🚨 Alerts":
        if not available_docs:
            st.info("No documents have been processed yet.")
        else:
            display_alert_page(available_docs)


if __name__ == "__main__":
    main()