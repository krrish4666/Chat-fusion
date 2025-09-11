# app.py --- Chat Fusion (RAG Complete & Retry Logic)
import os
import streamlit as st
from google import genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tempfile 
import time # Import the time module for delays

# --- RAG Imports ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Page Configuration ---
st.set_page_config(
    page_title="Chat Fusion: Gemini & Llama",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- API Key and Model Loading ---
api_key_gemini = os.getenv("GEMINI_API_KEY")

@st.cache_resource
def get_model_and_tokenizer(model_name):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load local model '{model_name}'. Error: {e}", icon="🔥")
        return None, None

@st.cache_resource
def get_tokenizer(model_name):
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        st.error(f"Failed to load tokenizer '{model_name}'. Error: {e}")
        return None

# --- RAG Backend Function ---
@st.cache_data
def process_documents(uploaded_files):
    """Loads, splits, embeds, and stores documents in a FAISS vector store."""
    all_docs = []
    temp_dir = tempfile.gettempdir()

    for uploaded_file in uploaded_files:
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_filepath)
        elif uploaded_file.name.endswith(".md"):
            loader = UnstructuredMarkdownLoader(temp_filepath)
        else:
            loader = TextLoader(temp_filepath)
            
        all_docs.extend(loader.load())
        os.remove(temp_filepath)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(all_docs)
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store

@st.cache_resource
def get_embedding_model():
    """Loads the embedding model from HuggingFace."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# --- Response Generation ---
def generate_llama_response(prompt, messages_history, model, tokenizer, max_history):
    try:
        conversation = messages_history[-max_history:]
        conversation[-1] = {"role": "user", "content": prompt}

        prompt_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        input_ids_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_ids_length:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)

    except Exception as e:
        st.error(f"Error during generation: {e}", icon="🔥")
        return "An error occurred while generating the response."

# --- Sidebar UI ---
with st.sidebar:
    st.title("🤖 Chat Fusion")
    st.markdown("An application that combines the power of Google's Gemini API with a locally-run Llama model.")
    st.markdown("---")

    if api_key_gemini:
        st.success("Gemini API key loaded.")
    else:
        st.warning("Gemini API key not found.")

    model_provider = st.selectbox(
        "Choose a model provider:",
        ["Gemini (Google)", "Local (TinyLlama)"],
        key="model_provider"
    )

    selected_model = None
    if model_provider == "Gemini (Google)":
        selected_model = st.selectbox(
            "Choose a Gemini model:",
            ["gemini-2.5-pro", "gemini-2.5-flash"],
            key="gemini_model"
        )
    else:
        selected_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        st.info(f"Selected model: **{selected_model}**")
        st.warning("The local model uses your CPU/GPU. Performance may vary.", icon="⚠️")
        max_history = st.slider(
            "Conversation history length:", 1, 10, 4,
            help="Number of past messages to include as context."
        )
    st.markdown("---")

    # --- RAG UI Section ---
    st.header("📄 Chat with Your Documents")
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Processing documents... This may take a moment."):
            vector_store = process_documents(uploaded_files)
            st.session_state.vector_store = vector_store
            st.success("Documents processed and ready for chat!")
            
    if "vector_store" in st.session_state and st.session_state.vector_store:
        st.info("Vector store is ready. You can now ask questions about your documents.")
    
    st.markdown("---")

# --- Tokenizer Explorer UI ---
with st.expander("🔍 Tokenizer Explorer"):
    # ... (rest of the unchanged UI code)
    tokenizer_text = st.text_area("Enter text...", key="tokenizer_input")
    # ...

# --- Main Chat UI ---
st.header(f"💬 Chat with: {selected_model or 'Select a model'}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # --- RAG LOGIC BLOCK ---
        final_prompt = prompt
        if "vector_store" in st.session_state and st.session_state.vector_store:
            with st.spinner("Searching documents..."):
                vector_store = st.session_state.vector_store
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                final_prompt = f"""
Based ONLY on the following context, please provide a detailed and comprehensive answer. If the context does not contain the answer, say 'I cannot answer this based on the provided documents.'

CONTEXT:
---
{context}
---

QUESTION:
{prompt}
"""
                st.info("**Info:** Your question is being answered using the uploaded documents as context.")
        
        # --- Main Generation Logic ---
        try:
            if model_provider == "Gemini (Google)":
                if not api_key_gemini:
                    st.error("Cannot proceed without a Gemini API key.", icon="🔑")
                else:
                    # --- NEW RETRY LOGIC FOR GEMINI ---
                    max_retries = 3
                    retry_delay = 2 # seconds
                    for attempt in range(max_retries):
                        try:
                            client = genai.Client(api_key=api_key_gemini)
                            api_history = []
                            for msg in st.session_state.messages[:-1]:
                                role = "model" if msg["role"] == "assistant" else "user"
                                api_history.append({"role": role, "parts": [{"text": msg["content"]}]})

                            stream = client.models.generate_content_stream(
                                model=selected_model,
                                contents=api_history + [
                                    {"role": "user", "parts": [{"text": final_prompt}]}
                                ]
                            )

                            for chunk in stream:
                                if chunk.text:
                                    full_response += chunk.text
                                    message_placeholder.markdown(full_response + "▌")
                            message_placeholder.markdown(full_response)
                            break # If successful, exit the loop

                        except Exception as e:
                            if "503" in str(e) and attempt < max_retries - 1:
                                st.warning(f"Model is busy. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                                time.sleep(retry_delay)
                                retry_delay *= 2 # Exponential backoff
                            else:
                                raise e # Re-raise the exception if it's not a 503 or if retries are exhausted
                    
            elif model_provider == "Local (TinyLlama)":
                model, tokenizer = get_model_and_tokenizer(selected_model)
                if model and tokenizer:
                    with st.spinner("Generating response..."):
                        full_response = generate_llama_response(
                            final_prompt, st.session_state.messages, model, tokenizer, max_history
                        )
                    message_placeholder.markdown(full_response)
                else:
                    full_response = "Error: Local model could not be loaded."
                    message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            full_response = "An error occurred."
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

if st.session_state.messages:
    if st.sidebar.button("Clear Conversation History"):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.rerun()

