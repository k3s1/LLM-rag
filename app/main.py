# app/main.py
import os
import shutil
import uuid
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy_utils import database_exists, create_database

from app.config import settings
from app.vector_db_utils import (
    load_document, chunk_documents, get_vector_store,
    save_faiss_index, load_faiss_index, retrieve_documents
)
from app.llm_integration import generate_response_with_llm
from app.database import engine, Base, get_db
from app.crud import (
    create_document_metadata, get_all_documents,
    get_document_by_filename, DocumentCreate, DocumentResponse, delete_document_metadata
)

app = FastAPI(
    title="RAG Document QA System [cite: 1]",
    description="Upload documents and ask questions based on their content[cite: 1].",
    version="1.0.0"
)

# Global FAISS index for simplicity in this example
# In a production system, this might be managed by a dedicated service or a persistent vector DB.
global_faiss_index = None

@app.on_event("startup")
async def startup_event():
    """
    On application startup, create database tables and load FAISS index if it exists.
    """
    print("Starting up application...")
    # Create database tables if they don't exist
    if not database_exists(engine.url):
        create_database(engine.url)
    Base.metadata.create_all(bind=engine)
    print("Database tables checked/created.")

    # Load FAISS index if available
    global global_faiss_index
    global_faiss_index = load_faiss_index(settings.FAISS_INDEX_PATH)
    if global_faiss_index:
        print("FAISS index loaded successfully during startup.")
    else:
        print("No existing FAISS index found. It will be created upon first document upload.")

@app.post("/upload-document/", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Uploads a PDF document, processes it, and stores its chunks in the vector database.
    [cite: 9]
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Check document count limit [cite: 4]
    current_docs = db.query(DBDocument).count()
    if current_docs >= settings.MAX_DOCUMENTS:
        raise HTTPException(status_code=400, detail=f"Maximum {settings.MAX_DOCUMENTS} documents already uploaded. Please delete some before uploading new ones.")

    # Save the file temporarily
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_location = os.path.join(settings.DOCUMENTS_DIR, unique_filename)
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

    # Process the document
    try:
        # [cite: 5]
        documents = load_document(file_location)
        if not documents:
            raise HTTPException(status_code=400, detail="Could not load document content. Is it a valid PDF?")

        # Check page count [cite: 4]
        if len(documents) > settings.MAX_PAGES_PER_DOCUMENT:
            # We already truncate in load_document, but a warning here is good.
            print(f"Warning: Document '{file.filename}' exceeds {settings.MAX_PAGES_PER_DOCUMENT} pages. Only the first {settings.MAX_PAGES_PER_DOCUMENT} pages will be processed.")

        chunks = chunk_documents(documents) # [cite: 5]
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not chunk document content. Is there enough text?")

        # Store in vector DB [cite: 6]
        global global_faiss_index
        global_faiss_index = get_vector_store(chunks, faiss_index=global_faiss_index)
        save_faiss_index(global_faiss_index, settings.FAISS_INDEX_PATH)

        # Store metadata in relational DB [cite: 10]
        doc_data = DocumentCreate(
            filename=unique_filename,
            original_filename=file.filename,
            file_path=file_location,
            page_count=len(documents),
            chunk_count=len(chunks)
        )
        db_document = create_document_metadata(db, doc_data)

        return DocumentResponse.model_validate(db_document)

    except Exception as e:
        # Clean up the uploaded file if processing fails
        if os.path.exists(file_location):
            os.remove(file_location)
        raise HTTPException(status_code=500, detail=f"Document processing failed: {e}")

@app.post("/query/")
async def query_system(
    query: str = Form(...),
):
    """
    Accepts a user query and returns a contextual response based on uploaded documents.
    [cite: 7]
    """
    if not global_faiss_index:
        raise HTTPException(status_code=400, detail="No documents uploaded yet. Please upload documents first.")

    try:
        # Retrieve relevant chunks [cite: 7]
        retrieved_chunks = retrieve_documents(query, global_faiss_index)
        if not retrieved_chunks:
            return JSONResponse(content={"answer": "No relevant information found in the uploaded documents."}, status_code=200)

        # Generate response with LLM [cite: 8]
        answer = generate_response_with_llm(query, retrieved_chunks)
        return JSONResponse(content={"answer": answer}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")

@app.get("/documents/", response_model=List[DocumentResponse])
async def get_documents_metadata(db: Session = Depends(get_db)):
    """
    Retrieves metadata for all processed documents.
    [cite: 10]
    """
    documents = get_all_documents(db)
    return [DocumentResponse.model_validate(doc) for doc in documents]

@app.delete("/documents/{doc_id}/", response_model=DocumentResponse)
async def delete_document(doc_id: int, db: Session = Depends(get_db)):
    """
    Deletes a document and its metadata.
    Note: This currently only deletes the metadata and the original uploaded file.
    It does NOT remove the chunks from the FAISS index. For a robust solution,
    you'd need to re-index or manage deletions within the vector store.
    """
    db_document = delete_document_metadata(db, doc_id)
    if db_document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    # IMPORTANT: Deleting from FAISS is more complex for in-memory FAISS.
    # For this beginner project, we're not rebuilding the FAISS index on delete.
    # A proper solution with persistent vector DBs (e.g., Pinecone, Weaviate, ChromaDB)
    # would handle this better. For now, it will only remove metadata.
    # If you delete all documents and upload new ones, the FAISS index will effectively reset.

    return DocumentResponse.model_validate(db_document)