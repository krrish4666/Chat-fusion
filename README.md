# 🤖 Chat Fusion: Gemini + Llama Chatbot

Chat Fusion is a **hybrid AI chatbot** that combines the power of **Google Gemini (2.5 Pro & 2.5 Flash)** with a **locally hosted Llama model (TinyLlama-1.1B-Chat)**.  
It is built using **Streamlit** for a clean UI, and integrates both **cloud-based AI** and **offline inference**.

---

## ✨ Features
- 🔹 **Choose AI provider**:  
  - Gemini (2.5 Pro / 2.5 Flash via Google GenAI API)  
  - Local (TinyLlama, runs on your CPU/GPU)  

- 🔹 **Streaming responses** for Gemini (like real-time typing).  
- 🔹 **Tokenizer Explorer** – compare tokenization between Gemini and Llama.  
- 🔹 **Conversation memory** with adjustable history for the local model.  
- 🔹 **Clear history button** for fresh sessions.  

---

## ⚡ Quick Start

### 1. Clone this repository
git clone https://github.com/krrish4666/Chat-fusion.git
cd Chat-fusion

### 2. Install dependencies

pip install -r requirements.txt

### 3. Add your Gemini API key

Set your API key as an environment variable:

export GEMINI_API_KEY="your_api_key_here"   # Linux / macOS
set GEMINI_API_KEY="your_api_key_here"      # Windows PowerShell

### 4. Run the app
streamlit run app.py

### 🛠️ Tech Stack

Streamlit
 – frontend UI

Google GenAI SDK
 – Gemini 2.5 integration

Transformers (Hugging Face)
 – Llama tokenizer & model

PyTorch
 – inference backend for local models
