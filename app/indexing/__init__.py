"""Indexing and corpus ingestion modules."""

from app.indexing.crawler import (
    CrawlResult,
    crawl_corpus,
    load_persisted_index,
    persist_index,
)
from app.indexing.dispatcher import dispatch_file, is_supported
from app.indexing.models import DocumentChunk, ParsedDocument

__all__ = [
    "crawl_corpus",
    "persist_index",
    "load_persisted_index",
    "CrawlResult",
    "dispatch_file",
    "is_supported",
    "DocumentChunk",
    "ParsedDocument",
]
