from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import hashlib

from app.utils import count_words

class DocumentChunker:
    def __init__(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        separators: List[str] = None,
        **kwargs
        ):
        """
        Initialize the chunker with text and size params

        Args:
            text: The preprocessed doc text
            target_chunk_size: Ideal chunk size in words
            max_chunk_size: Maximum allowed chunk size in words
            min_chunk_size: Minimum chunk size (smaller chunks will get merged)
            overlap_size: Number of overlapping words b/w chunks
        """
        self.text = text
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.doc_id = kwargs.get('doc_id', 'unknown')
        self.metadata = kwargs.get('metadata', {})

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
    
    def chunk(self) -> List[Dict[str, Any]]:
        """
        Split the text into chunks
        """
        if not self.text:
            return []
        
        chunks = self.splitter.split_text(self.text)
        
        chunk_objects = []
        
        for idx, chunk_content in enumerate(chunks):
            chunk_id = self._generate_chunk_id(self.doc_id, idx, chunk_content)
            
            merged_metadata = {
                "doc_name": self.metadata["doc_name"],
                "doc_path": self.metadata["doc_path"],
                "word_count_at_source": self.metadata["word_count"],
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
        
        return chunk_objects[0]['metadata']