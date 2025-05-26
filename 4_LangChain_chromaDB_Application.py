# pip install langchain
# pip install -U langchain-community
# pip install -U langchain-openai
# pip install chromadb
# pip install -U langchain-core

# db_name = "vector_db"
# import shutil
# shutil.rmtree(db_name, ignore_errors=True)

# Import necessary components
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma

# Initialize paths and parameters
data = "data.txt"
db_name = "vector_db"

# Load and process text file
text = TextLoader(data, encoding="utf-8")
docs = text.load()

# Split text into chunks
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# Clean up existing database if exists
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

#This is Chroma.from_documents from langchain for more simple auto create (database & collection & embedding)
# Create persistent Chroma vector store
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retrieve = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create conversational chain
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retrieve, memory=memory)

# we need this for keeping memory-history
# Interactive query loop
while True:
    query = input("Ask: ")
    if query.lower() == "exit":
        break

    response = chain.invoke({"question": query})
    print(f"The Query:: {query}. \nResponse:: {response['answer']}")

