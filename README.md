# RAG Chatbot — Azure OpenAI + Azure AI Search

A production-style Retrieval-Augmented Generation (RAG) chatbot built with Azure OpenAI and Azure AI Search. Ask questions from your own documents and get grounded, accurate answers.

---

## 🏗️ Architecture

Documents → Chunk → Embed → Store in Azure AI Search
User Question → Embed → Hybrid Search → Retrieve Chunks → GPT-4o → Answer

---

## 🛠️ Tech Stack

- **Azure OpenAI** — GPT-4o (chat) + text-embedding-3-small (embeddings)
- **Azure AI Search** — Vector + hybrid search index
- **Python** — Core pipeline
- **Gradio** — Chat UI
- **Managed Identity** — Secure authentication (no API keys in production)

---

## 📁 Project Structure

rag-chatbotrix/
├── docs/ ← Put your documents here
├── indexer.ipynb ← Indexing pipeline
├── retriever.ipynb ← Retrieval + generation pipeline
├── app.py ← Gradio chat UI
├── requirements.txt ← Python dependencies
├── .env ← Credentials (never commit this)
└── .gitignore

---

## 🚀 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/rag-chatbotrix.git
cd rag-chatbotrix
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up credentials

Create a `.env` file with:
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_SEARCH_ENDPOINT=your-search-endpoint
AZURE_SEARCH_API_KEY=your-search-key

### 5. Index your documents

- Add your documents to the `docs/` folder
- Run all cells in `indexer.ipynb`

### 6. Run the chatbot

```bash
python app.py
```

---

## 🔐 Security

- Uses **Azure Managed Identity** in production — no API keys in code
- API keys used for local development only via `.env`
- `.env` is excluded from version control via `.gitignore`

---

## 👩‍💻 Author

Anum — AI Engineer in training, specializing in GenAI + Azure
