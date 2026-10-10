"""Python source code parser for structural extraction.

Extracts constants, assignments, comments, functions, classes, and default values.
Preserves syntactic context and verbatim definitions.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List

from app.indexing.models import DocumentChunk, ParsedDocument


def parse_source(file_path: Path, rel_path: str) -> ParsedDocument:
    """Parse Python source files, extracting constants, comments, and structure."""
    try:
        content_bytes = file_path.read_bytes()
    except Exception as exc:
        return ParsedDocument(
            source_path=rel_path,
            file_type="source",
            error=f"Unreadable file: {type(exc).__name__}: {exc}",
        )

    # Decode defensively
    code_text = None
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            code_text = content_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if code_text is None:
        return ParsedDocument(
            source_path=rel_path,
            file_type="source",
            error="EncodingDecodeError: unable to decode source code",
        )

    chunks: List[DocumentChunk] = []
    lines = code_text.splitlines()

    # 1. AST Parsing for constants, assignments, functions
    try:
        tree = ast.parse(code_text, filename=rel_path)
    except SyntaxError as e:
        tree = None

    constants: Dict[str, Any] = {}
    functions: List[Dict[str, Any]] = []

    if tree:
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        val_str = ""
                        if isinstance(node.value, ast.Constant):
                            val_str = str(node.value.value)
                            constants[target.id] = node.value.value
                        elif isinstance(node.value, ast.Name):
                            val_str = target.id
                        
                        start_line = node.lineno
                        # Collect preceding comments
                        comment_lines = []
                        idx = start_line - 2
                        while idx >= 0 and lines[idx].strip().startswith("#"):
                            comment_lines.insert(0, lines[idx].strip())
                            idx -= 1
                        
                        chunk_text = f"Assignment: {target.id} = {ast.unparse(node.value)}"
                        if comment_lines:
                            chunk_text += f"\nComments:\n" + "\n".join(comment_lines)
                        
                        chunks.append(
                            DocumentChunk(
                                chunk_id=f"{rel_path}#const_{target.id}",
                                source_path=rel_path,
                                file_type="source",
                                row=start_line,
                                text=chunk_text,
                                metadata={
                                    "variable": target.id,
                                    "value": val_str,
                                    "comments": comment_lines,
                                    "is_constant": target.id.isupper(),
                                },
                                content_type="code",
                            )
                        )

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                docstring = ast.get_docstring(node) or ""
                # Default argument values
                defaults = [ast.unparse(d) for d in node.args.defaults]
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{rel_path}#def_{func_name}",
                        source_path=rel_path,
                        file_type="source",
                        row=node.lineno,
                        section=func_name,
                        text=f"def {func_name}(...):\nDocstring: {docstring}\nDefaults: {', '.join(defaults)}",
                        metadata={"function": func_name, "docstring": docstring},
                        content_type="code",
                    )
                )

    # 2. Add complete source file chunk so general regex/exact matches work
    chunks.append(
        DocumentChunk(
            chunk_id=f"{rel_path}#full",
            source_path=rel_path,
            file_type="source",
            text=code_text,
            metadata={"constants": constants, "total_lines": len(lines)},
            content_type="code",
        )
    )

    return ParsedDocument(
        source_path=rel_path,
        file_type="source",
        chunks=chunks,
        metadata={"constants": list(constants.keys()), "total_chunks": len(chunks)},
    )
