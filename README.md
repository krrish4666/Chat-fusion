# 🤖 Chat Fusion: Gemini + Llama with RAG

Chat Fusion is a **hybrid AI chatbot** that not only combines the power of **Google Gemini** and a **local Llama model** but also allows you to **chat with your own documents**.  
It is built with **Streamlit** and demonstrates how to build a Retrieval-Augmented Generation (RAG) pipeline to ground LLM responses in your specific data.

---

## 🔗 Repository
GitHub Repo: [Chat Fusion](https://github.com/krrish4666/Chat-fusion)

---

## ✨ Features
- 📄 **Chat with Your Documents (RAG)** - Upload your own PDF, TXT, or Markdown files.
  - Answers questions based *only* on the content of your documents.
  - Uses an in-memory FAISS vector store for fast and efficient retrieval.
- 🌐 **Gemini Integration** (2.5 Pro & Flash via Google GenAI API)  
- 💻 **Local Model Support** with TinyLlama (runs on CPU/GPU)  
- ⚡ **Streaming responses** for Gemini (real-time typing effect)  
- 🔍 **Tokenizer Explorer** – compare Gemini vs Llama tokenization  
- 🧠 **Conversation memory** with adjustable history length  
- 🧹 **Clear chat history** option for fresh starts  

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) – frontend UI framework  
- [LangChain](https://www.langchain.com/) – RAG pipeline orchestration  
- [Sentence-Transformers](https://www.sbert.net/) – text embedding creation  
- [FAISS (Facebook AI)](https://faiss.ai/) – efficient similarity search  
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

### 4. Run the App
```bash
streamlit run app.py
```

### 👤 Author

Built with ❤️ by Krishna Yadav
