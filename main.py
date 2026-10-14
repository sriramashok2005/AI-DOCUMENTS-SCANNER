import ollama
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from read_pdf import extract_text_from_pdf, extract_text_with_ocr
from chunking import create_chunks
from doucument_classifier import classify_document
from embeddings import create_embedding, create_embeddings
from vector_store import store_chunks, search_chunks
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)



@app.get("/")
def home():
    return {"message": "welcome to my RAG PROJECT!"}



@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    filename = Path(file.filename).name
    file_path = UPLOAD_DIR / filename

    content = await file.read()

    if not content.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file."
        )

    file_path.write_bytes(content)

    return {
        "filename": filename,
        "message": "PDF uploaded successfully."
    }


@app.get("/documents/{filename}/ocr")
def extract_pdf_text(filename: str):
    file_path = UPLOAD_DIR / Path(filename).name

    if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=404,
            detail="File not found."
        )

    text = extract_text_from_pdf(file_path)

    return {
        "filename": file_path.name,
        "extracted_text": text
    }


@app.get("/documents/{filename}/ocr-scanned")
def read_scanned_pdf(filename: str):
    file_path = UPLOAD_DIR / Path(filename).name

    if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=404,
            detail="PDF file not found."
        )

    text = extract_text_with_ocr(file_path)

    return {
        "filename": file_path.name,
        "ocr_text": text
    }


@app.post("/classify/{filename}")
def classify_uploaded_document(filename: str):
    file_path = UPLOAD_DIR / Path(filename).name

    if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=404,
            detail="PDF file not found."
        )

    text = extract_text_from_pdf(file_path)

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text found in PDF. Try the OCR endpoint for scanned PDFs."
        )

    document_type = classify_document(text)

    return {
        "filename": file_path.name,
        "document_type": document_type
    }

@app.post("/documents/{filename}/chunks")
def chunk_uploaded_document(filename: str):
    file_path = UPLOAD_DIR / Path(filename).name

    if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=404,
            detail="PDF file not found."
        )

    text = extract_text_from_pdf(file_path)

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text found in PDF."
        )

    chunks = create_chunks(
        text,
        chunk_size=500,
        overlap=50
    )

    return {
        "filename": file_path.name,
        "total_chunks": len(chunks),
        "chunks": chunks
    }



@app.post("/index/{filename}")
def index_document(filename: str):
    file_path = UPLOAD_DIR / Path(filename).name

    if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=404,
            detail="PDF file not found."
        )

    text = extract_text_from_pdf(file_path)

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text found in PDF. If this is a scanned PDF, use OCR."
        )

    chunks = create_chunks(
        text,
        chunk_size=500,
        overlap=50
    )

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No chunks created from PDF."
        )

    embeddings = create_embeddings(chunks)

    if len(embeddings) != len(chunks):
        raise HTTPException(
            status_code=500,
            detail="Embedding count does not match chunk count."
        )

    total_chunks = store_chunks(
        chunks,
        embeddings,
        file_path.name
    )

    return {
        "message": "PDF indexed successfully",
        "filename": file_path.name,
        "total_chunks": total_chunks
    }


class SearchRequest(BaseModel):
    query: str



@app.post("/search")
def search_document(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty."
        )

    query_embedding = create_embedding(request.query)

    results = search_chunks(
        query_embedding,
        top_k=3
    )

    return {
        "query": request.query,
        "results": results
    }

@app.post("/ask")
def ask_question(request: SearchRequest):
    
    query_embedding = create_embedding(request.query)

    
    context_chunks = search_chunks(
        query_embedding,
        top_k=3
    )

    
    context = "\n\n".join(context_chunks)


    response = ollama.chat(
        model="llama3.2:latest",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using the provided "
                    "PDF context. If the answer is not available "
                    "in the context, clearly say you could not find "
                    "the answer in the document."
                )
            },
            {
                "role": "user",
                "content": (
                    f"PDF Context:\n{context}\n\n"
                    f"Question: {request.query}"
                )
            }
        ]
    )


    return {
        "question": request.query,
        "answer": response["message"]["content"],
        "context": context_chunks
    }
