# app/crud.py
from sqlalchemy.orm import Session
from typing import List, Optional

from app.models import Document as DBDocument
from pydantic import BaseModel
import os

# Pydantic models for request/response bodies
class DocumentCreate(BaseModel):
    filename: str
    original_filename: str
    file_path: str
    page_count: int
    chunk_count: int

class DocumentResponse(BaseModel):
    id: int
    filename: str
    original_filename: str
    upload_time: str
    page_count: int
    chunk_count: int
    status: str

    class Config:
        from_attributes = True # Allow Pydantic to read ORM models

def create_document_metadata(db: Session, doc_data: DocumentCreate):
    """Creates a new document metadata entry in the database."""
    db_document = DBDocument(
        filename=doc_data.filename,
        original_filename=doc_data.original_filename,
        file_path=doc_data.file_path,
        page_count=doc_data.page_count,
        chunk_count=doc_data.chunk_count,
        status="processed"
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def get_document_by_filename(db: Session, filename: str) -> Optional[DBDocument]:
    """Retrieves a document by its internal filename."""
    return db.query(DBDocument).filter(DBDocument.filename == filename).first()

def get_all_documents(db: Session, skip: int = 0, limit: int = 100) -> List[DBDocument]:
    """Retrieves all document metadata entries."""
    return db.query(DBDocument).offset(skip).limit(limit).all()

def delete_document_metadata(db: Session, doc_id: int) -> Optional[DBDocument]:
    """Deletes a document metadata entry and its associated file."""
    db_document = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if db_document:
        # Optionally delete the actual file
        if os.path.exists(db_document.file_path):
            os.remove(db_document.file_path)
        db.delete(db_document)
        db.commit()
        return db_document
    return None