# pip install langchain
# pip install -U langchain-community
# pip install -U langchain-openai
# pip install chromadb

# Import necessary libraries
import os
import chromadb
from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os

# Load and process text file
data = "data.txt"
loader = TextLoader(data, encoding="utf-8")
docs = loader.load()

# Split text into larger chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings()
embedding_fn = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

# Initialize Chroma client and collection
client = chromadb.PersistentClient(path="chroma_native_db")
collection = client.get_or_create_collection(name="example_collection", embedding_function=embedding_fn)

# Add documents to collection
texts = [doc.page_content for doc in chunks]
ids = [f"doc_{i}" for i in range(len(texts))]
collection.add(documents=texts, ids=ids)

# Create Chroma vector store
vectorstore = Chroma(collection_name="example_collection", embedding_function=embeddings, persist_directory="chroma_native_db")

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff")

# Query and print results
query = "summary into 8 words"
response = qa_chain.invoke({"query": query})
print("Response:", response["result"])


