from app.document_loader import DocumentLoader
from app.document_processor import DocumentProcessor

import os
import uuid

class IngestionPipeline:

    def ingest_document(self, file_path: str) -> dict:
        
        loader = DocumentLoader(file_path)
        raw_text = loader.load()
        loader_metadata = loader.get_metadata()
        
        processor = DocumentProcessor(
            raw_text,
            doc_type=loader_metadata["doc_type"],
            source=loader_metadata["source"]
        )
        processor.clean_whitespace().remove_special_characters().preserve_structure()
        cleaned_text = processor.text
        
        doc_id = str(uuid.uuid4())
        doc_name = loader_metadata["file_name"]
        metadata = processor.extract_metadata(
            doc_id=doc_id, 
            doc_name=doc_name, 
            doc_path=loader_metadata["file_path"]
        )
        
        return {
            "text": cleaned_text,
            "metadata": metadata
        }