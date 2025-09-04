import streamlit as st
import pandas as pd
import os
import tempfile
import hashlib
import asyncio
import nest_asyncio
from dotenv import load_dotenv

# Apply nest_asyncio at the very beginning
nest_asyncio.apply()

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Check for Google API key
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("❌ GOOGLE_API_KEY not found! Please add it to your .env file")
    st.info("Create a .env file with: GOOGLE_API_KEY=your_api_key_here")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key

# Initialize models function - will be called only when needed
def get_models():
    """Get or create LLM and embeddings models"""
    if "llm" not in st.session_state or "embeddings" not in st.session_state:
        try:
            with st.spinner("🔧 Initializing AI models..."):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

                st.session_state.llm = llm
                st.session_state.embeddings = embeddings

                return llm, embeddings
        except Exception as e:
            st.error(f"❌ Error initializing AI models: {str(e)}")
            st.error("Please check your GOOGLE_API_KEY in the .env file")
            st.error("Make sure you have a valid Google API key for Gemini")
            st.stop()
    else:
        return st.session_state.llm, st.session_state.embeddings

# State management functions
def initialize_session_state():
    """Initialize session state variables with default values"""
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    if "data" not in st.session_state:
        st.session_state.data = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "retrieval_chain" not in st.session_state:
        st.session_state.retrieval_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False

def reset_rag_state():
    """Clear all RAG-related states when new file is uploaded"""
    st.session_state.vector_store = None
    st.session_state.retrieval_chain = None
    st.session_state.chat_history = []
    st.session_state.file_processed = False

def get_file_hash(uploaded_file):
    """Generate hash for uploaded file to detect changes"""
    if uploaded_file is not None:
        return hashlib.md5(uploaded_file.getvalue()).hexdigest()
    return None

def is_new_file_uploaded(uploaded_file):
    """Check if a new file has been uploaded"""
    if uploaded_file is None:
        return False

    current_file_name = uploaded_file.name
    if st.session_state.uploaded_file_name != current_file_name:
        return True
    return False

# Initialize session state
initialize_session_state()

# Streamlit UI
st.title("🤖 CSV RAG Chatbot")
st.markdown("Upload a CSV file and ask questions about your data!")

# File upload section
st.header("📁 Upload Your Data")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload a CSV file to analyze and ask questions about"
)

# Handle file upload and processing
if uploaded_file is not None:
    # Check if this is a new file
    if is_new_file_uploaded(uploaded_file):
        reset_rag_state()
        st.session_state.uploaded_file_name = uploaded_file.name

    try:
        # Read and display the CSV data
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data

        # Display file information
        st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

        # Show data preview
        st.header("📊 Data Preview")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Rows", len(data))
        with col2:
            st.metric("Total Columns", len(data.columns))
        with col3:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

        # Show first few rows
        st.subheader("First 5 rows:")
        st.dataframe(data.head())

        # Show column information
        with st.expander("📋 Column Details"):
            col_info = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes,
                'Non-Null Count': data.count(),
                'Null Count': data.isnull().sum()
            })
            st.dataframe(col_info)

    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        st.stop()

    # Process data for RAG if not already processed
    if not st.session_state.file_processed:
        with st.spinner("🔄 Processing data for AI analysis..."):
            try:
                # Get models
                llm, embeddings = get_models()

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # Load documents using CSVLoader
                loader = CSVLoader(file_path=tmp_file_path)
                docs = loader.load_and_split()

                # Create vector store
                vector_store = FAISS.from_documents(docs, embeddings)
                st.session_state.vector_store = vector_store

                # Clean up temporary file
                os.unlink(tmp_file_path)

                st.session_state.file_processed = True
                st.success("✅ Data processed successfully! You can now ask questions.")

            except Exception as e:
                st.error(f"❌ Error processing data: {str(e)}")
                st.stop()

    # Create retrieval chain if data is processed
    if st.session_state.file_processed and st.session_state.retrieval_chain is None:
        with st.spinner("🔧 Setting up AI chat system..."):
            try:
                # Get models
                llm, embeddings = get_models()

                # Create retriever
                retriever = st.session_state.vector_store.as_retriever()

                # Create prompt template
                system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Use three sentences maximum and keep the "
                    "answer concise."
                    "\n\n"
                    "{context}"
                )

                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                # Create question-answering chain
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
                st.session_state.retrieval_chain = retrieval_chain

            except Exception as e:
                st.error(f"❌ Error setting up chat system: {str(e)}")

    # Chat interface
    if st.session_state.file_processed and st.session_state.retrieval_chain is not None:
        st.header("💬 Ask Questions About Your Data")

        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History:")
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.expander(f"Q{i+1}: {question[:50]}..."):
                    st.write(f"**Question:** {question}")
                    st.write(f"**Answer:** {answer}")

        # User input
        user_input = st.text_input(
            "Ask a question about your data:",
            placeholder="e.g., What are the most common issues in the data?"
        )

        # Process user question
        if user_input:
            with st.spinner("🤔 Thinking..."):
                try:
                    answer = st.session_state.retrieval_chain.invoke({"input": user_input})
                    response = answer["answer"]

                    # Display answer
                    st.subheader("Answer:")
                    st.write(response)

                    # Add to chat history
                    st.session_state.chat_history.append((user_input, response))

                    # Show source documents
                    if "context" in answer:
                        with st.expander("📄 Source Information"):
                            for i, doc in enumerate(answer["context"]):
                                st.write(f"**Source {i+1}:**")
                                st.write(doc.page_content[:200] + "...")

                except Exception as e:
                    st.error(f"❌ Error processing question: {str(e)}")

        # Clear chat history button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

else:
    # Show instructions when no file is uploaded
    st.info("👆 Please upload a CSV file to get started!")

    # Show example questions
    st.subheader("💡 Example Questions You Can Ask:")
    st.write("- What are the most common issues in the data?")
    st.write("- Which customers have the highest satisfaction scores?")
    st.write("- What is the average resolution time?")
    st.write("- Show me trends in the data")
    st.write("- Which department handles the most cases?")