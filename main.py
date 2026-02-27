from app.ingestion.pipeline import IngestionPipeline, IndexingPipeline

def main():

    indexing_pipeline = IndexingPipeline()
    summary = indexing_pipeline.index_directory("./data/raw/documentations", skip_existing=True, store_in_milvus=True)
    print(summary)

if __name__ == "__main__":
    main()

