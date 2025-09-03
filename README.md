# 🤖 Chat Fusion: Gemini + Llama Chatbot

Chat Fusion is a **hybrid AI chatbot** that combines the power of **Google Gemini (2.5 Pro & 2.5 Flash)** with a **locally hosted Llama model (TinyLlama-1.1B-Chat)**.  
It is built with **Streamlit** and demonstrates how cloud-based AI and local inference can work together in one unified interface.

---

## 🔗 Repository
GitHub Repo: [Chat Fusion](https://github.com/krrish4666/Chat-fusion)

---

## ✨ Features
- 🌐 **Gemini Integration** (2.5 Pro & Flash via Google GenAI API)  
- 💻 **Local Model Support** with TinyLlama (runs on CPU/GPU)  
- ⚡ **Streaming responses** for Gemini (real-time typing effect)  
- 🔍 **Tokenizer Explorer** – compare Gemini vs Llama tokenization  
- 🧠 **Conversation memory** with adjustable history length  
- 🧹 **Clear chat history** option for fresh starts  

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) – frontend UI framework  
- [Google GenAI SDK](https://pypi.org/project/google-genai/) – Gemini integration  
- [Transformers (Hugging Face)](https://huggingface.co/docs/transformers) – local Llama model  
- [PyTorch](https://pytorch.org/) – deep learning backend  

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/krrish4666/Chat-fusion.git
cd Chat-fusion
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Set up API Key

Export your Gemini API key as an environment variable:

Linux / macOS:
```bash
export GEMINI_API_KEY="your_api_key_here"
```
Windows PowerShell:
```bash
set GEMINI_API_KEY="your_api_key_here"
```

Alternatively, create a .env file and load it in app.py.

4. Run the App
```bash
streamlit run app.py
```

👤 Author

Built with ❤️ by Krishna Yadav
