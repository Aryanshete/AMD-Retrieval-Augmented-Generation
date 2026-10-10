"""Defensive corpus crawler and indexer.

Recursively walks the corpus directory tree, skips unknown file types,
safely handles permission errors, encrypted files, and corrupt documents,
and produces persisted index artifacts without crashing.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import INDEX_DIR
from app.indexing.dispatcher import dispatch_file, is_supported
from app.indexing.models import DocumentChunk, ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    documents: List[ParsedDocument] = field(default_factory=list)
    failed_files: Dict[str, str] = field(default_factory=dict)
    skipped_files: List[str] = field(default_factory=list)
    all_chunks: List[DocumentChunk] = field(default_factory=list)
    manifest: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "valid_documents": len([d for d in self.documents if d.is_valid]),
            "failed_files": len(self.failed_files),
            "skipped_files": len(self.skipped_files),
            "total_chunks": len(self.all_chunks),
            "files_by_type": {
                ft: len([d for d in self.documents if d.file_type == ft and d.is_valid])
                for ft in set(d.file_type for d in self.documents)
            },
        }


def crawl_corpus(corpus_dir: Path) -> CrawlResult:
    """Safely traverse the corpus directory tree and parse all supported files.
    
    Guarantees:
    - Never raises on permission errors or corrupt/encrypted files.
    - Handles empty directories gracefully.
    - Normalizes paths to POSIX-style relative paths.
    """
    corpus_dir = Path(corpus_dir).resolve()
    result = CrawlResult()

    if not corpus_dir.exists():
        logger.error(f"Corpus directory does not exist: {corpus_dir}")
        result.manifest = {"error": f"Directory not found: {corpus_dir}"}
        return result

    if not corpus_dir.is_dir():
        logger.error(f"Corpus path is not a directory: {corpus_dir}")
        result.manifest = {"error": f"Path is not a directory: {corpus_dir}"}
        return result

    # Traverse directory tree defensively
    for root, dirs, files in os.walk(corpus_dir):
        # Empty directories simply produce no files - fully safe
        for filename in files:
            file_path = Path(root) / filename
            
            # Compute POSIX relative path
            try:
                rel_path = file_path.relative_to(corpus_dir).as_posix()
            except ValueError:
                rel_path = filename

            # 1. Check supported extensions
            if not is_supported(file_path):
                result.skipped_files.append(rel_path)
                continue

            # 2. Defensive parse per file
            try:
                # Check for permission by testing read access if possible
                if not os.access(file_path, os.R_OK):
                    result.failed_files[rel_path] = "permission_denied"
                    logger.warning(f"Unreadable file (permission denied): {rel_path}")
                    continue

                parsed_doc = dispatch_file(file_path, rel_path)

                if parsed_doc is None:
                    result.skipped_files.append(rel_path)
                    continue

                if parsed_doc.error:
                    result.failed_files[rel_path] = parsed_doc.error
                    logger.warning(f"Failed to parse {rel_path}: {parsed_doc.error}")
                    result.documents.append(parsed_doc)
                    continue

                result.documents.append(parsed_doc)
                result.all_chunks.extend(parsed_doc.chunks)

            except PermissionError as perm_err:
                result.failed_files[rel_path] = "permission_denied"
                logger.warning(f"PermissionError on {rel_path}: {perm_err}")
                continue
            except Exception as exc:
                err_name = type(exc).__name__
                result.failed_files[rel_path] = f"{err_name}: {exc}"
                logger.warning(f"Exception parsing {rel_path}: {err_name} - {exc}")
                continue

    result.manifest = {
        "corpus_dir": str(corpus_dir),
        "summary": result.summary(),
        "failed_files": result.failed_files,
        "skipped_files": result.skipped_files,
    }
    return result


def persist_index(crawl_result: CrawlResult, index_dir: Optional[Path] = None) -> Path:
    """Save parsed documents, chunks, and manifest to the index directory."""
    target_dir = Path(index_dir) if index_dir else INDEX_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = target_dir / "manifest.json"
    docs_path = target_dir / "documents.jsonl"
    chunks_path = target_dir / "chunks.jsonl"

    # Write manifest.json
    manifest_path.write_text(
        json.dumps(crawl_result.manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Write documents.jsonl
    with open(docs_path, "w", encoding="utf-8") as f:
        for doc in crawl_result.documents:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

    # Write chunks.jsonl
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in crawl_result.all_chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    # Build and persist HybridRetriever (lexical BM25 + dense vector indices)
    try:
        from app.retrieval.hybrid import HybridRetriever
        retriever = HybridRetriever()
        retriever.build(crawl_result.all_chunks)
        retriever.save(target_dir)
        logger.info(f"Persisted hybrid indices (lexical + vector) to {target_dir}")
    except Exception as exc:
        logger.warning(f"Failed to persist hybrid indices: {exc}")

    logger.info(f"Persisted index to {target_dir} ({len(crawl_result.all_chunks)} chunks)")
    return target_dir


def load_persisted_index(index_dir: Optional[Path] = None) -> Optional[CrawlResult]:
    """Load previously indexed chunks and manifest from disk."""
    target_dir = Path(index_dir) if index_dir else INDEX_DIR
    chunks_path = target_dir / "chunks.jsonl"
    manifest_path = target_dir / "manifest.json"
    docs_path = target_dir / "documents.jsonl"

    if not chunks_path.exists() or not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        
        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(DocumentChunk.from_dict(json.loads(line)))

        docs = []
        if docs_path.exists():
            with open(docs_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        docs.append(ParsedDocument.from_dict(json.loads(line)))

        return CrawlResult(
            documents=docs,
            failed_files=manifest.get("failed_files", {}),
            skipped_files=manifest.get("skipped_files", []),
            all_chunks=chunks,
            manifest=manifest,
        )
    except Exception as exc:
        logger.error(f"Error loading persisted index from {target_dir}: {exc}")
        return None
