import chromadb
import os
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize embedding functions
default_ef = embedding_functions.DefaultEmbeddingFunction()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-3-small")

# Initialize persistent Chroma client
chroma_client = chromadb.PersistentClient(path="./db/chroma_persist_3")

# Create collection with OpenAI embeddings
collection = chroma_client.get_or_create_collection("my_story", embedding_function=openai_ef)

# AI-related documents
documents = [
    {"id": "doc1", "text": "Welcome to the AI universe!"},
    {"id": "doc2", "text": "What's your favorite programming language?"},
    {"id": "doc3", "text": "Until next time, happy coding!"},
    {
        "id": "doc4",
        "text": "Apple is a leading tech company known for its innovative hardware and software. It was co-founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
    },
    {
        "id": "doc5",
        "text": "Blockchain is a decentralized digital ledger that records transactions across many computers in a way that ensures security and transparency.",
    },
    {
        "id": "doc6",
        "text": "Cloud Computing refers to the delivery of computing services—servers, storage, databases, networking, and software—over the internet.",
    },
    {
        "id": "doc7",
        "text": "Quantum Computing leverages quantum-mechanical phenomena like superposition and entanglement to perform calculations far faster than classical computers.",
    },
    {
        "id": "doc8",
        "text": "Cybersecurity involves protecting systems, networks, and programs from digital attacks aimed at accessing or destroying sensitive data.",
    },
    {
        "id": "doc9",
        "text": "Robotics combines engineering and computer science to design, build, and operate robots that can assist or replace human efforts.",
    },
    {
        "id": "doc10",
        "text": "The Internet of Things (IoT) refers to a network of physical devices embedded with sensors and software to connect and exchange data.",
    },
    {
        "id": "doc11",
        "text": "Big Data refers to extremely large datasets that can be analyzed computationally to reveal patterns, trends, and associations.",
    },
    {
        "id": "doc12",
        "text": "The concept of Moore's Law observes that the number of transistors on a microchip doubles approximately every two years, though this trend is slowing.",
    },
]

# Add documents to collection
for doc in documents:
    collection.upsert(ids=doc["id"], documents=[doc["text"]])

# Query the collection
query_text = "find document related to computer vision"
results = collection.query(query_texts=[query_text], n_results=3)

# Print formatted results
for idx, document in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][idx]
    distance = results["distances"][0][idx]

    print(f" For the query: {query_text}, \n Found similar document: {document} (ID: {doc_id}, Distance: {distance})")

