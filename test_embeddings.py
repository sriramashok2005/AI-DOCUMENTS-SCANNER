from embeddings import create_embedding

text = "How to troubleshoot a computer problem?"

vector = create_embedding(text)

print("Embedding generated successfully!")
print("Vector dimensions:", len(vector))
print("First 5 values:", vector[:5])