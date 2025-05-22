import os
import streamlit as st
from document_processor import process_documents
from vector_store import VectorStore
from groq_processor import GroqProcessor
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'groq_processor' not in st.session_state:
    st.session_state.groq_processor = None

# Configure app
st.set_page_config(page_title="Document Research Chatbot", layout="wide")
st.title("Document Research & Theme Identification Chatbot")
st.markdown("""
This chatbot processes your documents and identifies common themes across them with proper citations.
""")

# Sidebar for document upload and settings
with st.sidebar:
    st.header("Configuration")

    # Environment setup
    if st.button("Load Environment Variables"):
        try:
            load_dotenv()
            groq_api_key = os.getenv("GROQ_API_KEY")
            if groq_api_key:
                st.success("Environment variables loaded successfully")
                st.session_state.groq_api_key = groq_api_key
            else:
                st.error("GROQ_API_KEY not found in .env file")
        except Exception as e:
            st.error(f"Error loading environment: {str(e)}")

    # Manual API key input as fallback
    groq_api_key_input = st.text_input("Or enter Groq API Key manually:", type="password")
    if groq_api_key_input:
        st.session_state.groq_api_key = groq_api_key_input

    st.header("Document Processing")
    uploaded_files = st.file_uploader(
        "Upload PDF or image files",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if not st.session_state.get('groq_api_key'):
            st.error("Please provide a Groq API key")
        elif not uploaded_files:
            st.error("Please upload at least one document")
        else:
            # Create documents directory if not exists
            os.makedirs("documents", exist_ok=True)

            # Clear previous documents
            for filename in os.listdir("documents"):
                file_path = os.path.join("documents", filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    st.error(f"Error deleting {file_path}: {e}")

            # Save uploaded files
            for uploaded_file in uploaded_files:
                with open(os.path.join("documents", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # Process documents
            with st.spinner("Processing documents..."):
                try:
                    documents = process_documents("documents")
                    if documents:
                        st.session_state.vector_store = VectorStore()
                        st.session_state.vector_store.add_documents(documents)
                        st.session_state.groq_processor = GroqProcessor(api_key=st.session_state.groq_api_key)

                        st.session_state.processed = True
                        st.success(f"Processed {len(documents)} document chunks from {len(uploaded_files)} files")
                    else:
                        st.error("No valid text could be extracted from the documents")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")

# Main chat interface
if st.session_state.processed:
    st.header("Document Query Interface")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Enter your query about the documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("Searching documents and identifying themes..."):
                try:
                    # Get relevant documents
                    results = st.session_state.vector_store.query(prompt)
                    docs = [{
                        'text': doc,
                        'metadata': meta
                    } for doc, meta in zip(results['documents'][0], results['metadatas'][0])]

                    # Process query via GroqProcessor
                    response_data = st.session_state.groq_processor.process_query(prompt, docs)

                    # 1. Synthesized answer
                    st.markdown("### 🧾 Synthesized Answer")
                    st.markdown(response_data.get("answer", "No answer generated."))

                    # 2. Tabular document-level answers
                    st.markdown("### 📄 Document-Level Responses")
                    st.table([{
                        "Document ID": d["doc_id"],
                        "Extracted Answer": d["answer"],
                        "Citation": d["citation"]
                    } for d in response_data.get("doc_responses", [])])

                    # 3. Thematic summary
                    st.markdown("### 🧩 Identified Themes")
                    for t in response_data.get("themes", []):
                        st.markdown(f"#### {t['name']}")
                        st.markdown(f"{t['description']}")
                        st.markdown("**Supported by:**")
                        for doc in t.get("supporting_docs", []):
                            st.markdown(f"- {doc.get('doc_id')} (page {doc.get('page')})")

                    full_response = "✅ Query processed successfully."

                except Exception as e:
                    full_response = f"Error processing query: {str(e)}"

            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("Please upload documents and configure your API key in the sidebar to begin.")

# Document viewer section
if st.session_state.processed and os.path.exists("documents"):
    with st.expander("Uploaded Documents"):
        st.write("List of processed documents:")
        cols = st.columns(3)
        for i, filename in enumerate(os.listdir("documents")):
            cols[i % 3].write(f"📄 {filename}")
