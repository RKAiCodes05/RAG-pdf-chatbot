import streamlit as st
import os
import json
import csv
from datetime import datetime
from dotenv import load_dotenv
import time

load_dotenv()

groq_api_key = os.getenv("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")
# ============================================================================
# CONFIGURE PAGE
# ============================================================================

st.set_page_config(
    page_title="RAG PDF Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG-Powered PDF Question Answering")
st.markdown("""
    Ask questions about your PDF documents using AI-powered retrieval 
    and question answering.
""")

# ============================================================================
# INITIALIZE SESSION STATE (IMPORTANT!)
# ============================================================================

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("📋 Information")
    st.markdown("""
        **How it works:**
        1. Upload or use existing PDFs
        2. Ask a question
        3. System retrieves relevant sections
        4. AI generates answer with citations
        
        **Tech Stack:**
        - Embeddings: Sentence-BERT
        - Vector DB: ChromaDB
        - LLM: Groq (Llama 3.3)
        - Framework: LangChain
    """)
    
    st.markdown("---")
    st.subheader("📊 Session Stats")
    st.metric("Total Queries", len(st.session_state.chat_history))
    
    # View chat history in sidebar
    if len(st.session_state.chat_history) > 0:
        st.markdown("**Recent Queries:**")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
            st.caption(f"{i}. {chat['question'][:40]}...")
        
        if st.button("🗑️ Clear All History"):
            st.session_state.chat_history = []
            st.rerun()

# ============================================================================
# LOAD RAG SYSTEM (CACHED)
# ============================================================================

@st.cache_resource
def load_rag_system():
    """Load and cache RAG system"""
    try:
        from src.rag_system import RAGSystem
        from src.vector_store import VectorStore
        from src.embeddings import EmbeddingManager
        from src.llm_client import GroqClient
        
        vector_store = VectorStore()
        embedding_manager = EmbeddingManager()
        llm_client = GroqClient()
        
        rag_system = RAGSystem(
            vector_store=vector_store,
            embedding_manager=embedding_manager,
            llm_client=llm_client
        )
        return rag_system
    except Exception as e:
        st.error(f"Error loading RAG system: {e}")
        return None

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_to_json():
    """Export chat history as JSON"""
    if not st.session_state.chat_history:
        st.warning("No chat history to export")
        return None
    
    export_data = {
        "exported_at": datetime.now().isoformat(),
        "total_queries": len(st.session_state.chat_history),
        "conversations": st.session_state.chat_history
    }
    
    json_str = json.dumps(export_data, indent=2)
    return json_str

def export_to_csv():
    """Export chat history as CSV"""
    if not st.session_state.chat_history:
        st.warning("No chat history to export")
        return None
    
    csv_data = []
    for chat in st.session_state.chat_history:
        csv_data.append({
            "Question": chat.get('question', ''),
            "Answer": chat.get('answer', '')[:500],  # Limit answer length for CSV
            "Sources Used": chat.get('num_sources', 0),
            "Timestamp": chat.get('timestamp', '')
        })
    
    # Convert to CSV format
    import io
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["Question", "Answer", "Sources Used", "Timestamp"])
    writer.writeheader()
    writer.writerows(csv_data)
    
    return output.getvalue()

def export_to_markdown():
    """Export chat history as Markdown"""
    if not st.session_state.chat_history:
        st.warning("No chat history to export")
        return None
    
    md_content = "# RAG Chat History\n\n"
    md_content += f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"**Total Queries:** {len(st.session_state.chat_history)}\n\n"
    md_content += "---\n\n"
    
    for i, chat in enumerate(st.session_state.chat_history, 1):
        md_content += f"## Query {i}: {chat.get('question', '')}\n\n"
        md_content += f"**Answer:**\n\n{chat.get('answer', '')}\n\n"
        md_content += f"**Sources:** {chat.get('num_sources', 0)}\n\n"
        md_content += f"**Timestamp:** {chat.get('timestamp', '')}\n\n"
        md_content += "---\n\n"
    
    return md_content

# ============================================================================
# MAIN CONTENT
# ============================================================================

col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_input("🔍 Ask your question:", placeholder="What is RAG?")
    top_k = st.slider("Number of sources:", 1, 10, 3)
    confidence_threshold = st.slider("Confidence threshold:", 0.0, 1.0, 0.5)

with col2:
    st.metric("Total Documents", "272 pages")
    st.metric("Models", "Groq LLM")

# ============================================================================
# PROCESS QUERY
# ============================================================================

if query:
    rag_system = load_rag_system()
    
    if rag_system:
        with st.spinner("Retrieving and generating response..."):
            start_time = time.time()
            
            response = rag_system.query(
                question=query,
                top_k=top_k,
                score_threshold=confidence_threshold
            )
            
            elapsed_time = time.time() - start_time
            
            # Store in chat history
            chat_entry = {
                "question": query,
                "answer": response.get('answer', 'No answer generated'),
                "num_sources": response.get('num_contexts_used', 0),
                "sources": response.get('sources', []),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "response_time": f"{elapsed_time:.2f}s",
                "confidence": response.get('avg_similarity', 0)
            }
            
            st.session_state.chat_history.append(chat_entry)
            st.session_state.query_count += 1
        
        # Display answer
        st.success("✅ Response generated")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Answer")
            st.write(response['answer'])
        
        with col2:
            st.subheader("Metrics")
            st.metric("Confidence", f"{response.get('avg_similarity', 0):.1%}")
            st.metric("Contexts Used", response.get('num_contexts_used', 0))
            st.metric("Response Time", f"{elapsed_time:.2f}s")
        
        # Display sources
        st.subheader("📄 Sources")
        sources = response.get('sources', [])
        
        if sources:
            for idx, src in enumerate(sources, 1):
                with st.expander(f"{idx}. {src.get('file', 'Unknown')} - Page {src.get('page', 'N/A')}"):
                    st.markdown(f"**Relevance:** {src.get('similarity', 0):.1%}")
                    st.write(src)
        else:
            st.info("No sources found for this query")

# ============================================================================
# CHAT HISTORY DISPLAY
# ============================================================================

st.divider()
st.subheader("💬 Chat History")

if st.session_state.chat_history:
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Full History", "Summary", "Statistics"])
    
    # Tab 1: Full History
    with tab1:
        st.markdown(f"**Total Conversations:** {len(st.session_state.chat_history)}")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"#{len(st.session_state.chat_history) - i + 1} - {chat['question'][:60]}..."):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sources", chat.get('num_sources', 0))
                with col2:
                    st.metric("Time", chat.get('response_time', 'N/A'))
                with col3:
                    st.metric("Confidence", f"{chat.get('confidence', 0):.1%}")
                with col4:
                    st.metric("Timestamp", chat.get('timestamp', 'N/A'))
    
    # Tab 2: Summary
    with tab2:
        st.markdown("### Conversation Summary")
        
        # Calculate statistics
        total_sources = sum(chat.get('num_sources', 0) for chat in st.session_state.chat_history)
        avg_sources = total_sources / len(st.session_state.chat_history) if st.session_state.chat_history else 0
        avg_confidence = sum(chat.get('confidence', 0) for chat in st.session_state.chat_history) / len(st.session_state.chat_history) if st.session_state.chat_history else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", len(st.session_state.chat_history))
        with col2:
            st.metric("Avg Sources/Query", f"{avg_sources:.1f}")
        with col3:
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Questions list
        st.markdown("**All Questions:**")
        questions_text = "\n".join([f"- {chat['question']}" for chat in st.session_state.chat_history])
        st.text(questions_text)
    
    # Tab 3: Statistics
    with tab3:
        st.markdown("### Session Statistics")
        
        times = [float(chat.get('response_time', '0').replace('s', '')) for chat in st.session_state.chat_history]
        confidences = [chat.get('confidence', 0) for chat in st.session_state.chat_history]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Response Time", f"{min(times):.2f}s" if times else "N/A")
        with col2:
            st.metric("Avg Response Time", f"{sum(times)/len(times):.2f}s" if times else "N/A")
        with col3:
            st.metric("Max Response Time", f"{max(times):.2f}s" if times else "N/A")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Confidence", f"{min(confidences):.1%}" if confidences else "N/A")
        with col2:
            st.metric("Avg Confidence", f"{sum(confidences)/len(confidences):.1%}" if confidences else "N/A")
        with col3:
            st.metric("Max Confidence", f"{max(confidences):.1%}" if confidences else "N/A")

else:
    st.info("💡 Start by asking a question to build chat history")

# ============================================================================
# EXPORT FUNCTIONALITY (WORKING!)
# ============================================================================

st.divider()
st.subheader("💾 Export Options")

if st.session_state.chat_history:
    col1, col2, col3 = st.columns(3)
    
    # Export as JSON
    with col1:
        st.markdown("**Export as JSON**")
        json_data = export_to_json()
        if json_data:
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Export as CSV
    with col2:
        st.markdown("**Export as CSV**")
        csv_data = export_to_csv()
        if csv_data:
            st.download_button(
                label="📊 Download CSV",
                data=csv_data,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Export as Markdown
    with col3:
        st.markdown("**Export as Markdown**")
        md_data = export_to_markdown()
        if md_data:
            st.download_button(
                label="📝 Download Markdown",
                data=md_data,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

else:
    st.info("💡 No chat history to export yet. Ask a question first!")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.85rem;'>
        <p>🚀 RAG PDF Q&A System • Powered by ChromaDB, Groq & Streamlit</p>
    </div>
""", unsafe_allow_html=True)