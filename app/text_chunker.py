from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import hashlib

from app.utils import count_words

"""
Previous chunker was storing state, i.e, self.text, self.doc_id, self.metadata which
leads to creation of new chunker for each document

New design makes chunker stateless and reusable across multiple documents
"""

class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        separators: List[str] = None
        ):
        """
        Initialize the chunker with text and size params

        Args:
            chunk_size: Target chunk size in words
            overlap: Number of overlapping words between chunks
            separators: List of separators for splitting (optional)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

        if separators is None:
            self.separators = [
                "\n\n",
                "\n", 
                ". ", 
                "! ",
                "? ",
                "; ",
                ", ",
                " ",
                ""
            ]
        else:
            self.separators = separators
        
        self.splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            length_function=count_words
        )


    
    def _generate_chunk_id(self, doc_id: str, chunk_idx: int, content: str) -> str:
        """
        Generate a unique chunk ID
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"{doc_id}_chunk_{chunk_idx}_{content_hash}"
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split the text into chunks

        Args:
            text: The preprocessed doc text
            metadata: Doc metadata from DocumentProcessor.extract_metadata()
                Must contain: doc_id, doc_name, doc_path, doc_type, source
        
        Returns:
            List of chunk dicts ready for embedding
        """
        if not text:
            return []
        
        doc_id = metadata.get("doc_id", "unknown")
        
        chunks = self.splitter.split_text(text)
        
        chunk_objects = []
        
        for idx, chunk_content in enumerate(chunks):
            chunk_id = self._generate_chunk_id(doc_id, idx, chunk_content)
            
            merged_metadata = {
                "doc_name": metadata["doc_name"],
                "doc_type": metadata["doc_type"],
                "source": metadata["source"],
                "doc_path": metadata["doc_path"],
                "word_count_at_source": metadata["word_count"],
                "chunk_index": idx,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "total_chunks": len(chunks)
            }
            
            chunk_objects.append({
                "chunk_id": chunk_id,
                "doc_id": self.doc_id,
                "content": chunk_content,
                "char_count": len(chunk_content),
                "word_count": count_words(chunk_content),
                "metadata": merged_metadata
            })
        
        return chunk_objects