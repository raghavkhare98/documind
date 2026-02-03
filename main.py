from app.text_chunker import DocumentChunker
from app.ingestion.pipeline import IngestionPipeline

def main():

    pipeline = IngestionPipeline()
    doc = pipeline.ingest_document("/Users/raghavkhare/repos/documind/data/raw/research_papers/blended_rag.pdf")
    
    chunker = DocumentChunker(text=doc["text"], doc_id=doc["doc_id"], metadata=doc["metadata"])
    chunks = chunker.chunk()
    print(chunks)

if __name__ == "__main__":
    main()
