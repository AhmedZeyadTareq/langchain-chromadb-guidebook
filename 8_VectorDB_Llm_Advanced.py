import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_key, model_name="text-embedding-3-small")

# Initialize persistent Chroma client
chroma_client = chromadb.PersistentClient(path="./db/chroma_persistent_storage")
collection = chroma_client.get_or_create_collection(name="document_QA_collection", embedding_function=openai_ef)

# Initialize OpenAI client
client = OpenAI(api_key=openai_key)

# =================================
# === For initial setup -- Uncomment (below) all for the first run, and then comment it all out ===
# =================================
# Function to load documents from a directory
# def load_documents_from_directory(directory_path):
#     print("==== Loading documents from directory ====")
#     documents = []
#     for filename in os.listdir(directory_path):
#         if filename.endswith(".txt"):
#             with open(
#                 os.path.join(directory_path, filename), "r", encoding="utf-8"
#             ) as file:
#                 documents.append({"id": filename, "text": file.read()})
#     return documents
#
#
#
# # Function to split text into chunks
# def split_text(text, chunk_size=1000, chunk_overlap=20):
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start = end - chunk_overlap
#     return chunks
#
#
# # Load documents from the directory
# directory_path = "./data/new_articles"
# documents = load_documents_from_directory(directory_path)
#
# # Split the documents into chunks
# chunked_documents = []
# for doc in documents:
#     chunks = split_text(doc["text"])
#     print("==== Splitting docs into chunks ====")
#     for i, chunk in enumerate(chunks):
#         chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})
#
#
# # Function to generate embeddings using OpenAI API
# def get_openai_embedding(text):
#     response = client.embeddings.create(input=text, model="text-embedding-3-small")
#     embedding = response.data[0].embedding
#     print("==== Generating embeddings... ====")
#     return embedding
#
#
# # Generate embeddings for the document chunks
# for doc in chunked_documents:
#     print("==== Generating embeddings... ====")
#     doc["embedding"] = get_openai_embedding(doc["text"])
#
#
# # Upsert documents with embeddings into Chroma
# for doc in chunked_documents:
#     print("==== Inserting chunks into db;;; ====")
#     collection.upsert(
#         ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
#     )



# === End of the initial setup -- Uncomment all for the first run, and then comment it all out ===
# =================================


# Sample documents (in a real scenario, use the commented setup code)
documents = [
    {"id": "doc1", "text": "Apple designs consumer electronics and software."},
    {"id": "doc2", "text": "Microsoft develops operating systems and cloud services."},
    {"id": "doc3", "text": "Google specializes in search technology and advertising."}
]

# Add documents to collection
for doc in documents:
    collection.add(ids=[doc["id"]], documents=[doc["text"]])

# Query function
def query_documents(question, n_results=2):
    # query_embedding = get_openai_embedding(question)
    results = collection.query(query_texts=question, n_results=n_results)

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks
    # for idx, document in enumerate(results["documents"][0]):
    #     doc_id = results["ids"][0][idx]
    #     distance = results["distances"][0][idx]
    #     print(f"Found document chunk: {document} (ID: {doc_id}, Distance: {distance})")


# Response generation function
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        temperature=0.1
    )
    return response.choices[0].message

# Example usage
question = "tell me about space x ships."
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

# Print results
print("==== Answer ====")
print(relevant_chunks)
print(answer.content)

