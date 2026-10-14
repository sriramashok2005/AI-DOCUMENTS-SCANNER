def create_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        if chunk_text:
            chunks.append(chunk_text)

        start += chunk_size - overlap

    return chunks