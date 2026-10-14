
import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="rag_documents"
)


def store_chunks(chunks, embeddings, filename):
    ids = [
        f"{filename}_chunk_{i}"
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "filename": filename,
                "chunk_number": i
            }
            for i in range(len(chunks))
        ]
    )

    return len(chunks)

def search_chunks(query_embedding, top_k=3):
    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total_chunks)
    )

    documents = results.get("documents", [[]])[0]

    return documents
    