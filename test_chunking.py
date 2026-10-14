from chunking import create_chunks

sample_text = " ".join(
    [f"word{i}" for i in range(1, 1201)]
)

chunks = create_chunks(
    sample_text,
    chunk_size=500,
    overlap=50
)

print("Total chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"Chunk {index} word count:", len(chunk.split()))