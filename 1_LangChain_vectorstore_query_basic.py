# Install required packages
# pip install langchain
# pip install -U langchain-community
# pip install -U langchain-openai

# Load text data from a local file
data = "data.txt"

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAI

# Load the document using LangChain's TextLoader
text = TextLoader(data, encoding="utf-8")

# Initialize OpenAI embedding model
embeddings = OpenAIEmbeddings()

# Create a vector index (non-persistent) from the document using LangChain
index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([text])

# Define the LLM (OpenAI) for answering queries
llm = OpenAI(temperature=0.1, model="gpt-4o-mini")

# Sample query
query = "what is best shampoo for post protein?"
prompt_query = "Answer directly and concise and specific:\n" + query

# Perform a retrieval-based query using the index and LLM
response = index.query_with_sources(prompt_query, llm=llm)

# Display the final result
print(f"The Query:: {query}. \nResponse :: {response['answer']}")