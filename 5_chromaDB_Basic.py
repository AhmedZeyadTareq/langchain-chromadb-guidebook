# Basic ChromaDB example
import chromadb

# Initialize Chroma client
chroma_client = chromadb.Client()
# Create or get collection with default embedding function
collection = chroma_client.get_or_create_collection(name="test_collection")
# here with this simple app Chroma DB uses: DefaultEmbeddingFunction

# Sample documents
documents = [
    {"id": "doc1", "text": "Hello, World"},
    {"id": "doc2", "text": "what do you do Today?"},
    {"id": "doc3", "text": "Goodbye, have a nice day"}
]

# Add documents to collection
for doc in documents:
    collection.upsert(ids=doc['id'], documents=doc['text'])

# Query the collection
query = "greetings"
results = collection.query(query_texts=query, n_results=2)

# Print raw results
print(results)
print("===================")

# Print formatted results
for idx, document in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][idx]
    distance = results["distances"][0][idx]
    print(f"For The Query: {query}\n Found similar document: {document}, (ID: {doc_id}, Distance: {distance})")