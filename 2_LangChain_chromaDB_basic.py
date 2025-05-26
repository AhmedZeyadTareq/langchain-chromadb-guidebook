# Install required packages
# pip install langchain
# pip install -U langchain-community
# pip install -U langchain-openai
# pip install chromadb

import shutil
# Delete the persistent ChromaDB directory if it exists (clean setup)
shutil.rmtree("chroma_DB", ignore_errors=True)

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# Load text file
data = "data.txt"
text = TextLoader(data, encoding="utf-8")
docs = text.load()

# Split the document into chunks with overlap to preserve context
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(docs)

print(f"Total chunks: {len(chunks)}")

# Create a persistent ChromaDB vector store from the chunks
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="chroma_DB")

# Initialize LLM (OpenAI)
llm = OpenAI(temperature=0, model="gpt-4o-mini")

# Convert vector store to a retriever
retrieve = vectorstore.as_retriever()

# Create RetrievalQA chain (Retriever + LLM)
query = "what is the best choice for dry hair??"
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retrieve, chain_type="stuff", verbose=True)

# Invoke the QA chain with the query
response = chain.invoke({"query": query})

# Print the answer
print(f"The Query:: {query}. \nResponse:: {response['result']}")