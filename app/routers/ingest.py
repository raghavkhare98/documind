from fastapi import APIRouter
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.pipeline import IndexingPipeline

router = APIRouter(prefix="/ingest")

