# app.py --- Chat Fusion (Final Corrected Version, Gemini 2.5 Updated)
import os
import streamlit as st
from google import genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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

# --- Response Generation ---
def generate_llama_response(prompt, messages_history, model, tokenizer, max_history):
    try:
        conversation = messages_history[-max_history:]
        conversation.append({"role": "user", "content": prompt})

        prompt_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
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

# --- Tokenizer Explorer UI ---
st.header("🔍 Tokenizer Explorer")
tokenizer_text = st.text_area(
    "Enter text to compare tokenization:",
    "Tokenization is the first step in language processing.",
    key="tokenizer_input"
)

if tokenizer_text:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gemini")
        if api_key_gemini:
            try:
                client = genai.Client(api_key=api_key_gemini)
                result = client.models.count_tokens(model="gemini-2.5-pro", contents=tokenizer_text)
                st.info(f"**Token Count:** {result.total_tokens}")
            except Exception as e:
                st.error(f"Gemini tokenization failed: {e}")
        else:
            st.warning("Gemini API key required.")
    with col2:
        st.subheader("TinyLlama")
        llama_tokenizer = get_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        if llama_tokenizer:
            tokens = llama_tokenizer.encode(tokenizer_text)
            st.info(f"**Token Count:** {len(tokens)}")
            st.text_area("Token IDs", str(tokens), height=100, disabled=True)
st.markdown("---")

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
        try:
            # --- UPDATED GEMINI CHAT LOGIC ---
            if model_provider == "Gemini (Google)":
                if not api_key_gemini:
                    st.error("Cannot proceed without a Gemini API key.", icon="🔑")
                else:
                    client = genai.Client(api_key=api_key_gemini)

                    # Prepare history for Gemini API
                    api_history = []
                    for msg in st.session_state.messages[:-1]:  # Exclude current user input
                        role = "model" if msg["role"] == "assistant" else "user"
                        api_history.append({"role": role, "parts": [{"text": msg["content"]}]})

                    # ✅ Stream response using latest SDK
                    stream = client.models.generate_content_stream(
                        model=selected_model,
                        contents=api_history + [
                            {"role": "user", "parts": [{"text": prompt}]}
                        ]
                    )

                    for chunk in stream:
                        if chunk.text:
                            full_response += chunk.text
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

            elif model_provider == "Local (TinyLlama)":
                model, tokenizer = get_model_and_tokenizer(selected_model)
                if model and tokenizer:
                    with st.spinner("Generating response..."):
                        full_response = generate_llama_response(
                            prompt, st.session_state.messages, model, tokenizer, max_history
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
        st.rerun()
