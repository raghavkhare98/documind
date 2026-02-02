from app.document_loader import DocumentLoader
from app.document_processor import DocumentProcessor

import os
import uuid

class IngestionPipeline:

    def ingest_document(self, file_path: str) -> dict:
        
        loader = DocumentLoader(file_path)
        raw_text = loader.load()
        
        processor = DocumentProcessor(raw_text)
        processor.clean_whitespace().remove_special_characters().preserve_structure()
        cleaned_text = processor.text
        
        doc_id = str(uuid.uuid4())
        doc_name = os.path.basename(file_path)
        metadata = processor.extract_metadata(
            doc_id=doc_id, 
            doc_name=doc_name, 
            doc_path = file_path)
        
        return {
            "doc_id": doc_id,
            "text": cleaned_text,
            "metadata": metadata,
            "file_path": file_path
        }