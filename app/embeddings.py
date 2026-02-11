import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class EmbeddingGenerator:
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = None
    ):
        """
        Initialize the embedding generator
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (defaults to OPENAI_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        
        self.model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        self.embedding_dim = self.model_dimensions.get(model, 1536)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process per API call
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                raise RuntimeError(f"Failed to generate batch embeddings: {str(e)}")
        
        return all_embeddings
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks
        
        Args:
            chunks: List of chunk dictionaries with 'content' field
            
        Returns:
            List of chunks with added 'embedding' field
        """
        if not chunks:
            return []
        
        texts = [chunk.get("content", "") for chunk in chunks]
        
        embeddings = self.generate_embeddings_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model"""
        return self.embedding_dim
