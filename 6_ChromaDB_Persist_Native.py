import chromadb

from chromadb.utils import embedding_functions

# Initialize embedding function
default_ef = embedding_functions.DefaultEmbeddingFunction()

# Initialize persistent Chroma client
chroma_client = chromadb.PersistentClient(path="./db/chroma_persist_default")

# Create collection with default embedding function
collection = chroma_client.get_or_create_collection("my_story", embedding_function=default_ef)

# Sample documents
documents = [
    {"id": "doc1", "text": "Greetings, adventurer!"},
    {"id": "doc2", "text": "How goes your quest today?"},
    {"id": "doc3", "text": "Farewell, until we meet again in the realm!"},
    {
        "id": "doc4",
        "text": "Dwarven Forge Inc. is a legendary tech guild specializing in enchanted hardware and rune-based software. \
        It was founded by the twin brothers Thorin and Balin Ironfoot in the Third Age.",
    },
]

# Add documents to collection
for doc in documents:
    collection.upsert(ids=doc["id"], documents=[doc["text"]])

# Query the collection
query_text = "find document related to technology company"
results = collection.query(query_texts=[query_text], n_results=2)

# Print formatted results
for idx, document in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][idx]
    distance = results["distances"][0][idx]

    print(f" For the query: {query_text}, \n Found similar document: {document} (ID: {doc_id}, Distance: {distance})")
