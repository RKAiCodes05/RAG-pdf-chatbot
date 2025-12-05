import streamlit as st
import os
import json
import csv
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PAGE CONFIGURATION & CUSTOM STYLING
# ============================================================================

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium UI
st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* General Settings */
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
        /* Header Styling */
        .main-header {
            background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .sub-header {
            color: #6B7280;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }

        /* Chat Message Styling */
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        
        /* Dark mode adjustments for sidebar if needed */
        @media (prefers-color-scheme: dark) {
            section[data-testid="stSidebar"] {
                background-color: #1e293b;
                border-right: 1px solid #334155;
            }
        }

        /* Metric Cards */
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: 600;
            color: #4F46E5;
        }

        /* Custom Buttons */
        .stButton button {
            border-radius: 0.5rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        /* Source Cards */
        .source-card {
            background-color: #f3f4f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            border-left: 4px solid #4F46E5;
        }
        
        @media (prefers-color-scheme: dark) {
            .source-card {
                background-color: #1f2937;
                border-left: 4px solid #818cf8;
            }
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

# ============================================================================
# RAG SYSTEM LOADING
# ============================================================================

@st.cache_resource
def get_rag_system():
    """Load and cache the RAG system"""
    try:
        from src.rag_system import RAGSystem
        from src.vector_store import VectorStore
        from src.embeddings import EmbeddingManager
        from src.llm_client import GroqClient
        
        vector_store = VectorStore()
        embedding_manager = EmbeddingManager()
        llm_client = GroqClient()
        
        return RAGSystem(
            vector_store=vector_store,
            embedding_manager=embedding_manager,
            llm_client=llm_client
        )
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def prepare_export_data():
    """Convert message history to exportable format"""
    export_data = []
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            # Find the corresponding user question (the message before this one)
            # This assumes a strict user-assistant turn-taking
            idx = st.session_state.messages.index(msg)
            question = st.session_state.messages[idx-1]["content"] if idx > 0 else "Unknown"
            
            export_data.append({
                "timestamp": msg.get("timestamp", datetime.now().isoformat()),
                "question": question,
                "answer": msg["content"],
                "sources": msg.get("sources", []),
                "metrics": msg.get("metrics", {})
            })
    return export_data

def export_json(data):
    return json.dumps({"exported_at": datetime.now().isoformat(), "conversations": data}, indent=2)

def export_csv(data):
    if not data: return ""
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp", "Question", "Answer", "Num Sources", "Avg Confidence"])
    for item in data:
        writer.writerow([
            item["timestamp"],
            item["question"],
            item["answer"],
            len(item["sources"]),
            f"{item['metrics'].get('confidence', 0):.2%}"
        ])
    return output.getvalue()

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.subheader("Retrieval Settings")
    top_k = st.slider("Sources to Retrieve", 1, 10, 3, help="Number of document chunks to retrieve for context")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, help="Minimum similarity score for retrieved chunks")
    
    st.divider()
    
    st.subheader("📊 Session Stats")
    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    st.metric("Total Queries", len(user_msgs))
    
    if st.session_state.messages:
        st.divider()
        st.subheader("💾 Export History")
        
        export_data = prepare_export_data()
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "JSON",
                data=export_json(export_data),
                file_name=f"rag_history_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )
        with col2:
            st.download_button(
                "CSV",
                data=export_csv(export_data),
                file_name=f"rag_history_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.markdown("""
        <div style='font-size: 0.8rem; color: #666;'>
            <b>Powered by:</b><br>
            • Groq (Llama 3)<br>
            • ChromaDB<br>
            • LangChain
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN INTERFACE
# ============================================================================

# Header
st.markdown('<h1 class="main-header">RAG PDF Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about your documents with AI-powered precision.</p>', unsafe_allow_html=True)

# Initialize RAG System
rag_system = get_rag_system()

if not rag_system:
    st.warning("⚠️ RAG System is initializing or encountered an error. Please check the logs.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If it's an assistant message, show sources and metrics in an expander
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources & Metrics"):
                # Metrics Row
                m_col1, m_col2, m_col3 = st.columns(3)
                metrics = message.get("metrics", {})
                m_col1.metric("Confidence", f"{metrics.get('confidence', 0):.1%}")
                m_col2.metric("Sources Used", len(message.get("sources", [])))
                m_col3.metric("Time", f"{metrics.get('time', 0):.2f}s")
                
                st.divider()
                
                # Sources List
                for idx, src in enumerate(message["sources"], 1):
                    st.markdown(f"""
                        <div class="source-card">
                            <div style="font-weight: 600; margin-bottom: 0.25rem;">
                                📄 {src.get('file', 'Unknown File')} (Page {src.get('page', '?')})
                            </div>
                            <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">
                                {src.get('content', '')[:200]}...
                            </div>
                            <div style="font-size: 0.8rem; color: #666;">
                                Relevance: {src.get('similarity', 0):.1%}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.now().isoformat()})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    if rag_system:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                
                # Query the RAG system
                response = rag_system.query(
                    question=prompt,
                    top_k=top_k,
                    score_threshold=confidence_threshold
                )
                
                elapsed_time = time.time() - start_time
                
                answer = response.get('answer', 'I could not generate an answer.')
                sources = response.get('sources', [])
                avg_similarity = response.get('avg_similarity', 0)
                
                # Display answer
                st.markdown(answer)
                
                # Display sources preview immediately
                if sources:
                    with st.expander("📚 Sources & Details", expanded=False):
                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("Confidence", f"{avg_similarity:.1%}")
                        m_col2.metric("Sources", len(sources))
                        m_col3.metric("Time", f"{elapsed_time:.2f}s")
                        
                        st.divider()
                        
                        for src in sources:
                            st.markdown(f"- **{src.get('file', 'Doc')}** (Page {src.get('page', '?')}): {src.get('similarity', 0):.1%}")

            # Add assistant message to state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources,
                "metrics": {
                    "confidence": avg_similarity,
                    "time": elapsed_time
                },
                "timestamp": datetime.now().isoformat()
            })
    else:
        st.error("RAG System not ready.")