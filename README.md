# ChromaDB-LangChain Notebook Collection

This repository contains a curated collection of Python scripts showcasing how to build powerful document-based Q\&A systems by integrating **LangChain** with **ChromaDB**, as well as using **ChromaDB natively**.

Each script is self-contained, progressively increases in complexity, and demonstrates unique use cases — from basic in-memory search to production-ready RAG (Retrieval-Augmented Generation) pipelines.

---

## 🚀 Why Use This Repository?

LangChain simplifies interactions with LLMs, while ChromaDB offers fast and efficient vector storage. This combination empowers you to:

* Build conversational AI systems.
* Store and query large documents semantically.
* Implement RAG pipelines with OpenAI models.
* Evaluate local vs. cloud-based embeddings.

Whether you're a beginner or deploying enterprise-grade solutions, this repo has something for you.

---

## 📁 Repository Structure

```
1_LangChain_vectorstore_query_basic.py       # Basic Q&A using in-memory vector store
2_LangChain_chromaDB_basic.py                # LangChain with persistent ChromaDB
3_LangChain_ChromaDB_advanced.py             # Advanced ChromaDB with native client
4_LangChain_chromaDB_Application.py          # Conversational memory with ChromaDB
5_chromaDB_Basic.py                          # Pure ChromaDB (no LangChain)
6_ChromaDB_Persist_Native.py                 # Native persistent ChromaDB with local embeddings
7_ChromaDB_Multiple_Emb.py                   # Compare OpenAI & local embeddings
8_VectorDB_Llm_Advanced.py                   # Full modular RAG pipeline
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

2. **Create virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
   Each script lists its required `pip install` commands at the top. Example:

```bash
pip install langchain langchain-openai langchain-community chromadb python-dotenv
```

4. **Set OpenAI API Key**

* Create a `.env` file:

```
OPENAI_API_KEY="your_openai_api_key_here"
```

* Or export manually:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

5. **Prepare your data**
   Ensure `data.txt` exists in the root directory with meaningful text content to query.

---

## 📜 Script Breakdown

### 1. `1_LangChain_vectorstore_query_basic.py`

* **Goal:** Demonstrate a minimal LangChain setup for direct Q\&A over local text without persistent storage.
* **Libraries:** `langchain`, `langchain-openai`, `TextLoader`, `OpenAIEmbeddings`, `VectorstoreIndexCreator`, `OpenAI`.
* **Key Functions:** `VectorstoreIndexCreator.from_loaders()`, `query_with_sources()`
* **Features:**

  * Builds an in-memory vector index from the input file using LangChain.
  * Uses OpenAI's GPT (e.g., `gpt-4o-mini`) to respond to a single query.
  * Retrieval and generation happen in one step with no storage overhead.
* ✅ Ideal for fast prototyping, small-scale tools, or demonstration purposes.

### 2. `2_LangChain_chromaDB_basic.py`

* **Goal:** Persist document embeddings using ChromaDB and enable retrieval-based QA via LangChain.
* **Libraries:** `langchain`, `langchain-community`, `langchain-openai`, `chromadb`, `TextLoader`, `CharacterTextSplitter`, `OpenAIEmbeddings`, `RetrievalQA`, `OpenAI`, `Chroma`.
* **Key Functions:** `Chroma.from_documents()`, `RetrievalQA.from_chain_type()`
* **Features:**

  * Uses `CharacterTextSplitter` to chunk text efficiently (chunk\_size=200, overlap=50).
  * Stores vector embeddings in a persistent Chroma database directory (`chroma_DB`).
  * Wraps retrieval and answer generation using `RetrievalQA` and `gpt-4o-mini`.
* ✅ Provides persistent, reloadable Q\&A system suitable for production setups.

### 3. `3_LangChain_ChromaDB_advanced.py`

* **Goal:** Combine LangChain with native Chroma client for full control over embedding storage and retrieval.
* **Libraries:** `chromadb`, `langchain-chroma`, `langchain-openai`, `TextLoader`, `CharacterTextSplitter`, `OpenAIEmbeddingFunction`, `ChatOpenAI`, `RetrievalQA`.
* **Key Functions:** `PersistentClient.get_or_create_collection()`, `collection.add()`, `Chroma()`
* **Features:**

  * Manually defines IDs and documents, then upserts to Chroma using `OpenAIEmbeddingFunction()`.
  * Uses LangChain to wrap the Chroma collection into a retriever.
  * Performs question-answering using `RetrievalQA` + `ChatOpenAI`.
* ✅ Provides flexibility in embedding logic and database structure for custom pipelines.

### 4. `4_LangChain_chromaDB_Application.py`

* **Goal:** Build a conversational assistant with context memory and persistent vector storage.
* **Libraries:** `langchain`, `langchain-openai`, `langchain-chroma`, `TextLoader`, `CharacterTextSplitter`, `ConversationBufferMemory`, `ConversationalRetrievalChain`, `ChatOpenAI`.
* **Key Functions:** `ConversationalRetrievalChain.from_llm()`, `.invoke()` loop.
* **Features:**

  * Splits and embeds text using LangChain, stores to persistent ChromaDB (`vector_db`).
  * Maintains multi-turn conversation using memory buffer.
  * Retrieves top-k chunks and uses them for contextual LLM responses.
* ✅ Ideal for chatbots, digital assistants, or memory-aware agents.

### 5. `5_chromaDB_Basic.py`

* **Goal:** Demonstrate the minimal setup and querying workflow using ChromaDB without external APIs.
* **Libraries:** `chromadb`
* **Key Functions:** `Client.get_or_create_collection()`, `collection.upsert()`, `collection.query()`
* **Features:**

  * Inserts documents into an in-memory ChromaDB collection.
  * Queries using default embedding function for semantic similarity.
  * Displays similar documents with distance metrics.
* ✅ Great for grasping Chroma’s internal document handling and search logic.

### 6. `6_ChromaDB_Persist_Native.py`

* **Goal:** Use ChromaDB in persistent mode with local embedding logic for privacy-first applications.
* **Libraries:** `chromadb`, `embedding_functions`
* **Key Functions:** `PersistentClient()`, `DefaultEmbeddingFunction()`, `collection.upsert()`, `collection.query()`
* **Features:**

  * Uses local `DefaultEmbeddingFunction()` without external API calls.
  * Stores vectors to disk in `./db/chroma_persist_default`.
  * Retrieves semantically similar documents from persistent store.
* ✅ Excellent for environments requiring no cloud dependencies or secure local inference.

### 7. `7_ChromaDB_Multiple_Emb.py`

* **Goal:** Enable experimentation with multiple embedding strategies (OpenAI vs local) for document similarity.
* **Libraries:** `chromadb`, `dotenv`, `OpenAIEmbeddingFunction`, `DefaultEmbeddingFunction`
* **Key Functions:** `PersistentClient()`, `collection.upsert()`, `collection.query()`
* **Features:**

  * Creates persistent Chroma collection with OpenAI embeddings.
  * Loads a rich dataset focused on AI/ML topics.
  * Executes similarity search and prints top-k matched chunks.
* ✅ Useful for comparing semantic retrieval accuracy across embedding types.

### 8. `8_VectorDB_Llm_Advanced.py`

* **Goal:** Deliver a full Retrieval-Augmented Generation (RAG) pipeline using modular components and OpenAI.
* **Libraries:** `chromadb`, `dotenv`, `OpenAI`, `OpenAIEmbeddingFunction`
* **Key Functions:** `load_documents_from_directory()`, `split_text()`, `get_openai_embedding()`, `collection.query()`, `chat.completions.create()`
* **Features:**

  * Optional modular pipeline for loading → chunking → embedding → upserting documents.
  * Retrieves top-matching chunks and feeds them into OpenAI's chat model.
  * Produces concise answers using context-driven prompts.
* ✅ Designed for advanced production pipelines and research-based use cases.

---

## 📌 Highlights

* ✅ Supports OpenAI embeddings + local embeddings
* ✅ Demonstrates both LangChain & native ChromaDB usage
* ✅ Clean, modular, reusable code blocks
* ✅ Scalable for both prototypes and production

---

## 🤝 Contributing

Have suggestions, fixes, or ideas? Feel free to fork, open an issue, or submit a pull request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📧 Contact

For questions or collaborations, reach out via \[[your-email@example.com](mailto:your-email@example.com)] or open an issue on GitHub.

---

Enjoy building powerful retrieval-based applications with LangChain & ChromaDB! 🚀
