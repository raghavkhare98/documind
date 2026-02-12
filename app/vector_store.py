import os
from typing import Any, Optional, Literal
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from dotenv import load_dotenv
load_dotenv()

DocType = Literal["documentation", "rfc", "research", "manual"]

class MilvusVectorStore:
    """Milvus vector database client for RAG system"""
    
    DOC_TYPES = {
        "documentation": "Software documentation (Docker, FastAPI, Go Fiber, etc.)",
        "rfc": "Protocol RFCs (HTTPS, HTTP/2, WebSocket, etc.)",
        "research": "Research papers and academic publications",
        "manual": "Software manuals and guides"
    }

    def __init__(
        self,
        collection_name: str = "technical_documents",
        host: str = None,
        port: int = 19530,
        embedding_dim: int = 1536
    ):
        """
        Initialize Milvus vector store
        
        Args:
            collection_name: Name of the Milvus collection
            host: Milvus server host (defaults to MILVUS_HOST env var)
            port: Milvus server port
            embedding_dim: Dimension of embedding vectors
        """
        self.collection_name = collection_name
        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port
        self.embedding_dim = embedding_dim
        self.collection = None
        self._connect()
        self._init_collection()
    
    def _connect(self):
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                timeout=10
            )
            print(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {str(e)}")
    
    def _create_schema(self) -> CollectionSchema:
        """
        Create collection schema for document chunks
        
        Returns:
            CollectionSchema with defined fields
        """
        fields = [
            FieldSchema(
                name="chunk_id",
                dtype=DataType.VARCHAR,
                max_length=256,
                is_primary=True,
                auto_id=False,
                description="Unique chunk identifier"
            ),
            FieldSchema(
                name="doc_id",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="Parent document ID"
            ),
            FieldSchema(
                name="doc_name",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="Document filename"
            ),
            FieldSchema(
                name="doc_type",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="Document type"
            ),
            FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="Document source"
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="Chunk text content"
            ),
            FieldSchema(
                name="chunk_index",
                dtype=DataType.INT64,
                description="Chunk position in document"
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dim,
                description="Embedding vector"
            ),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Document chunks with embeddings for RAG"
        )
        
        return schema
    
    def _init_collection(self):
        """Initialize or load the collection"""
        if utility.has_collection(self.collection_name):
            print(f"Loading existing collection: {self.collection_name}")
            self.collection = Collection(self.collection_name)
        else:
            print(f"Creating new collection: {self.collection_name}")
            schema = self._create_schema()
            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            self._create_index()
        
        self.collection.load()
    
    def _create_index(self):
        """Create IVF_FLAT index for vector similarity search"""
        index_params = {
            "metric_type": "COSINE",  # L2 distance (can also use IP or cosine sim)
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        print("Created vector index")
    
    def insert_chunks(self, chunks: list[dict[str, Any]]) -> list[str]:
        """
        Insert document chunks with embeddings into Milvus
        
        Args:
            chunks: List of chunk dictionaries with required fields
            
        Returns:
            List of inserted chunk IDs
        """
        if not chunks:
            return []
        
        if chunks and "metadata" in chunks[0]:
            doc_type = chunks[0]["metadata"].get("doc_type")
            if doc_type and doc_type not in self.DOC_TYPES:
                raise ValueError(f"Invalid document type: {doc_type}. Must be one of {list(self.DOC_TYPES.keys())}")

        chunk_ids = []
        doc_ids = []
        doc_names = []
        doc_types = []
        sources = []
        contents = []
        chunk_indices = []
        embeddings = []
        
        for chunk in chunks:
            chunk_ids.append(chunk["chunk_id"])
            doc_ids.append(chunk["doc_id"])
            doc_names.append(chunk["metadata"]["doc_name"])
            doc_types.append(chunk["metadata"]["doc_type"])
            sources.append(chunk["metadata"]["source"])
            contents.append(chunk["content"])
            chunk_indices.append(chunk["metadata"]["chunk_index"])
            embeddings.append(chunk["embedding"])
        
        data = [
            chunk_ids,
            doc_ids,
            doc_names,
            doc_types,
            sources,
            contents,
            chunk_indices,
            embeddings
        ]
        
        try:
            insert_result = self.collection.insert(data)
            self.collection.flush()
            print(f"Inserted {len(chunk_ids)} chunks into Milvus")
            return chunk_ids
        
        except Exception as e:
            raise RuntimeError(f"Failed to insert chunks: {str(e)}")
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        doc_type: Optional[str] = None,
        source: Optional[str] = None,
        filter_expr: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            doc_type: Filter by document type(documentation/rfc/research paper/manual)
            source: Filter by specific source(eg "docker", "fastapi" etc.)
            filter_expr: Custom filter expression (overrides doc_type and source)
            
        Returns:
            List of search results with metadata
        """
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        if filter_expr is None and (doc_type or source):
            filters = []
            if doc_type:
                if doc_type not in self.DOC_TYPES:
                    raise ValueError(f"Invalid doc_type. Must be one of: {list(self.DOC_TYPES.keys())}")
                filters.append(f'doc_type == "{doc_type}')
            if source:
                filters.append(f'source == "{source}"')
            filter_expr = " && ".join(filters) if filters else None

        try:
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["chunk_id", "doc_id", "doc_name", "doc_type", "source", "content", "chunk_index"]
            )
            
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "chunk_id": hit.entity.get("chunk_id"),
                        "doc_id": hit.entity.get("doc_id"),
                        "doc_name": hit.entity.get("doc_name"),
                        "doc_type": hit.entity.get("doc_type"),
                        "source": hit.entity.get("source"),
                        "content": hit.entity.get("content"),
                        "chunk_index": hit.entity.get("chunk_index"),
                        "distance": hit.distance,
                        "score": 1 / (1 + hit.distance)  # Convert distance to similarity score
                    })
            
            return formatted_results
        
        except Exception as e:
            raise RuntimeError(f"Search failed: {str(e)}")
    
    def search_by_type(
        self,
        query_embedding: list[float],
        doc_type: str,
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Convenience method: Search within a specific document type

        Args:
            query_embedding: Query vector
            doc_type: Document type to search in
            top_k: Number of results
        """
        return self.search(query_embedding, top_k=top_k, doc_type=doc_type)
    
    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all chunks belonging to a document
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            Number of deleted chunks
        """
        try:
            expr = f'doc_id == "{doc_id}"'
            result = self.collection.delete(expr)
            self.collection.flush()
            print(f"Deleted chunks for doc_id: {doc_id}")
            return result.delete_count
        
        except Exception as e:
            raise RuntimeError(f"Failed to delete chunks: {str(e)}")
    
    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks belonging to a source
        """
        try:
            expr = f'source == "{source}"'
            result = self.collection.delete(expr)
            self.collection.flush()
            print(f"Deleted chunks for source: {source}")
            return result.delete_count
        
        except Exception as e:
            raise RuntimeError(f"Failed to delete chunks: {str(e)}")

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the collection"""
        self.collection.flush()
        stats = {
            "name": self.collection_name,
            "total_chunks": self.collection.num_entities,
            "embedding_dim": self.embedding_dim,
            "doc_types": {}
        }
        
        for doc_type in self.DOC_TYPES.keys():
            try:
                results = self.collection.query(
                    expr=f'doc_type == "{doc_type}"',
                    output_fields=["chunk_id"],
                    limit=16384
                )
                stats["doc_types"][doc_type] = len(results)
            except Exception as e:
                stats["doc_types"][doc_type] = "N/A"
        
        return stats
    
    def list_sources(self, doc_type: Optional[str] = None) -> list[str]:
        """
        List all unique sources in the collection

        Args:
            doc_type: Optional filter by document type
        """

        expr = f'doc_type == "{doc_type}"' if doc_type else None

        try:
            results = self.collection.query(
                expr=expr,
                output_fields=["source"],
                limit=16384
            )
            sources = list(set([r["source"] for r in results]))
            return sorted(sources)
        except Exception as e:
            raise RuntimeError(f"Failed to list sources: {str(e)}")

    def disconnect(self):
        """Disconnect from Milvus"""
        connections.disconnect("default")
        print("Disconnected from Milvus")