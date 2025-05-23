# app/models.py
from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.sql import func
from app.database import Base

class Document(Base):
    """SQLAlchemy model for storing document metadata."""
    # [cite: 10]
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True, unique=True)
    original_filename = Column(String) # To store the name user uploaded
    file_path = Column(String)
    upload_time = Column(DateTime(timezone=True), server_default=func.now())
    page_count = Column(Integer)
    chunk_count = Column(Integer)
    status = Column(String, default="processed") # e.g., "processing", "processed", "failed"