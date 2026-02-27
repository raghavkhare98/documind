from pymilvus import connections, Collection, utility
import os

def count_words(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def count_characters(text: str, exclude_whitespace: bool = True) -> int:
    """
    Count the number of characters in a text string
    
    Args:
        text: Input text string
        exclude_whitespace: If True, excludes spaces, newlines, and tabs from count
        
    Returns:
        Number of characters in the text
    """
    if not text:
        return 0
    
    if exclude_whitespace:
        return len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
    return len(text)

def check_collection(collection_name: str="technical_docs", port: int=19530) -> None:
    connections.connect(
        host=os.getenv("MILVUS_HOST"),
        port=port
    )

    print(f"Collection exists: {utility.has_collection(collection_name)}")

    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        
        print(f"Collection schema: {collection.schema}")
        print(f"Number of entities (before load): {collection.num_entities}")
        
        print("Attempting to load collection...")
        collection.load()
        
        # Flush to ensure data is persisted
        print("Flushing collection...")
        collection.flush()
        
        print(f"Number of entities (after load): {collection.num_entities}")
        
        print("\nTrying to query for any data...")
        results = collection.query(
            expr="chunk_index >= 0",
            output_fields=["chunk_id", "doc_name", "doc_type"],
            limit=10
        )
        
        print(f"Found {len(results)} chunks")
        if results:
            print("Sample data:")
            for r in results[:3]:
                print(f"  - {r}")

    connections.disconnect("default")