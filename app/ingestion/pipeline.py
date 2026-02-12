from app.document_loader import DocumentLoader
from app.document_processor import DocumentProcessor
from app.embeddings import EmbeddingGenerator
from app.vector_store import MilvusVectorStore
from app.text_chunker import DocumentChunker

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
            "doc_id": doc_id,
            "text": cleaned_text,
            "metadata": metadata,
            "file_path": file_path
        }

class IndexingPipeline:
    """Handles chunking, embedding generation, and vector storage"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        milvus_host: str = None,
        collection_name: str = "technical_docs",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the indexing pipeline
        
        Args:
            chunk_size: Size of text chunks in words
            chunk_overlap: Overlap between chunks in words
            milvus_host: Milvus server host (defaults to MILVUS_HOST env var)
            collection_name: Name of Milvus collection
            embedding_model: OpenAI embedding model to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.chunker = DocumentChunker(
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap
        )
        
        self.embedder = EmbeddingGenerator(model=embedding_model)
        
        # Initialize vector store
        self.vector_store = MilvusVectorStore(
            collection_name=collection_name,
            host=milvus_host,
            embedding_dim=self.embedder.get_embedding_dimension()
        )
    
    def index_document(
        self,
        doc_data: dict,
        store_in_milvus: bool = True
    ) -> dict:
        """
        Index a processed document: chunk → embed → store
        
        Args:
            doc_data: Output from IngestionPipeline.ingest_document()
                Must contain:
                    - doc_id: Unique document ID
                    - text: Preprocessed text
            store_in_milvus: Whether to store in Milvus
            
        Returns:
            Dictionary with indexing results
        """
        doc_id = doc_data["doc_id"]
        cleaned_text = doc_data["text"]
        metadata = doc_data["metadata"]
        doc_name = metadata.get("doc_name", "unknown")
        
        print(f"\nIndexing: {doc_name}")
        
        print("1/3 Chunking text...")
        chunks = self.chunker.chunk(cleaned_text, metadata)
        print(f"Created {len(chunks)} chunks")

        if not chunks:
            print(f"No chunks created for {doc_name}")
            return{
                "doc_id": doc_id,
                "doc_name": doc_name,
                "file_path": doc_data.get("file_path"),
                "num_chunks": 0,
                "chunk_ids": [],
                "stored_in_milvus": False,
                "status": "No chunks"
            }
        
        print("2/3 Generating embeddings...")
        chunks_with_embeddings = self.embedder.embed_chunks(chunks)
        print(f"Generated {len(chunks_with_embeddings)} embeddings")

        chunk_ids = []
        if store_in_milvus:
            print("3/3 Storing in Milvus...")
            chunk_ids = self.vector_store.insert_chunks(chunks_with_embeddings)
            print(f"Stored in {len(chunk_ids)} chunks")
        
        print(f"Successfully indexed: {doc_name}")
        
        return {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "file_path": doc_data.get("file_path"),
            "num_chunks": len(chunks),
            "chunk_ids": chunk_ids,
            "stored_in_milvus": store_in_milvus,
            "status": "success"
        }
    
    def index_directory(
        self,
        directory_path: str,
        store_in_milvus: bool = True
    ) -> dict:
        """
        Index all documents in a directory
        
        Args:
            directory_path: Path to directory
            store_in_milvus: Whether to store in Milvus
            
        Returns:
            Dictionary with summary statistics
        """
        ingestion_pipeline = IngestionPipeline()
        results = []
        supported_exts = {'.pdf', '.txt', '.docx', '.md'}
        
        print(f"\n{'='*60}")
        print(f"\nScanning directory: {directory_path}")
        print(f"{'='*60}")
        
        all_files = []
        for root, dirs, files in os.walk(directory_path):
            for filename in files:
                if filename.startswith('.'):
                    continue
                
                ext = os.path.splitext(filename)[1].lower()
                if ext in supported_exts:
                    file_path = os.path.join(root, filename)
                    all_files.append(file_path)
        
        print(f"Found {len(all_files)} supported documents\n")
        for idx, file_path in enumerate(all_files, 1):
            filename = os.path.basename(file_path)
            print(f"[{idx}/{len(all_files)}] Processing: {filename}")

            try:
                # Ingest (load + process)
                doc_data = ingestion_pipeline.ingest_document(file_path)
                
                # Index (chunk + embed + store)
                result = self.index_document(doc_data, store_in_milvus)
                results.append(result)
                
            except Exception as e:
                print(f"Error indexing {filename}: {str(e)}")
                results.append({
                    "file_path": file_path,
                    "doc_name": filename,
                    "error": str(e),
                    "status": "failed"
                })
        
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "failed"]
        total_chunks = sum(r.get("num_chunks", 0) for r in successful)

        print(f"\n{'='*60}")
        print(f"INDEXING SUMMARY")
        print(f"{'='*60}")
        print(f"Total documents: {len(all_files)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Total chunks created: {total_chunks}")
        
        if failed:
            print(f"\n Failed documents:")
            for f in failed:
                print(f" - {f['doc_name']}: {f.get('error', 'Unknown error')}")
        
        print(f"{'='*60}")

        return{
            "total_documents": len(all_files),
            "successful": len(successful),
            "failed": len(failed),
            "total_chunks": total_chunks,
            "results": results
        }
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store"""
        return self.vector_store.get_collection_stats()
