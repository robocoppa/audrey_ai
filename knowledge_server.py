"""
Knowledge server — local document RAG for Audrey.

A standalone FastAPI service that ingests files from a knowledge directory,
chunks and indexes them with SQLite FTS5, and exposes search endpoints
that Audrey discovers via OpenAPI.

Run:  uvicorn knowledge_server:app --host 0.0.0.0 --port 8002
"""

import hashlib
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("knowledge-server")

# ── Config ───────────────────────────────────────────────────────────────────

KNOWLEDGE_ROOT = os.getenv("KNOWLEDGE_ROOT", "/knowledge")
KNOWLEDGE_DB = os.getenv("KNOWLEDGE_DB", "/app/data/knowledge.db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "3500"))        # chars (~900 tokens)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "500"))   # chars (~130 tokens)
MAX_RESULT_CHUNKS = int(os.getenv("MAX_RESULT_CHUNKS", "10"))

# ── Text extraction ──────────────────────────────────────────────────────────

PLAIN_EXTENSIONS = {
    ".txt", ".md", ".log", ".csv", ".json", ".yaml", ".yml",
    ".xml", ".ini", ".conf", ".py", ".js", ".ts", ".sh", ".toml",
    ".rst", ".cfg", ".env", ".sql", ".r", ".go", ".java", ".c",
    ".cpp", ".h", ".hpp", ".rb", ".php", ".swift", ".kt",
}


def extract_text(path: Path) -> str | None:
    """Extract text content from a file. Returns None on failure."""
    suffix = path.suffix.lower()
    try:
        if suffix in PLAIN_EXTENSIONS:
            return path.read_text(errors="replace")

        if suffix == ".pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(str(path))
                pages = [pg.extract_text() or "" for pg in reader.pages[:200]]
                return "\n\n".join(pages)
            except ImportError:
                logger.warning("PyPDF2 not installed, skipping %s", path)
                return None

        if suffix in {".html", ".htm"}:
            html = path.read_text(errors="replace")
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style"]):
                    tag.decompose()
                return soup.get_text(separator="\n", strip=True)
            except ImportError:
                import re
                return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()

        if suffix == ".docx":
            try:
                from docx import Document
                doc = Document(str(path))
                return "\n\n".join(
                    para.text for para in doc.paragraphs if para.text.strip()
                )
            except ImportError:
                logger.warning("python-docx not installed, skipping %s", path)
                return None

        return None
    except Exception as e:
        logger.error("Failed to extract text from %s: %s", path, e)
        return None


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


# ── Database ─────────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(KNOWLEDGE_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db():
    os.makedirs(os.path.dirname(KNOWLEDGE_DB) or ".", exist_ok=True)
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            path        TEXT UNIQUE NOT NULL,
            filename    TEXT NOT NULL,
            collection  TEXT NOT NULL DEFAULT '',
            tags        TEXT NOT NULL DEFAULT '',
            checksum    TEXT NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            file_size   INTEGER NOT NULL DEFAULT 0,
            indexed_at  REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id   INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content     TEXT NOT NULL,
            UNIQUE(source_id, chunk_index)
        );

        -- FTS5 virtual table for keyword search
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content,
            content='chunks',
            content_rowid='id',
            tokenize='porter unicode61'
        );

        -- Triggers to keep FTS in sync
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
        END;
    """)
    conn.close()
    logger.info("Knowledge DB initialized at %s", KNOWLEDGE_DB)


def _file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _infer_collection(rel_path: str) -> str:
    """Infer collection from the first directory component of the relative path."""
    parts = Path(rel_path).parts
    if len(parts) > 1:
        return parts[0]
    return "general"


# ── Ingestion logic ──────────────────────────────────────────────────────────

def _ingest_single_file(conn: sqlite3.Connection, file_path: Path,
                        collection: str | None, tags: str) -> dict[str, Any]:
    """Ingest one file. Returns status dict."""
    if not file_path.exists():
        return {"path": str(file_path), "status": "not_found"}
    if not file_path.is_file():
        return {"path": str(file_path), "status": "not_a_file"}

    checksum = _file_checksum(file_path)

    # Check if already indexed with same checksum
    row = conn.execute(
        "SELECT id, checksum FROM sources WHERE path = ?",
        (str(file_path),)
    ).fetchone()
    if row and row["checksum"] == checksum:
        return {"path": str(file_path), "status": "unchanged"}

    text = extract_text(file_path)
    if text is None or not text.strip():
        return {"path": str(file_path), "status": "unsupported_or_empty"}

    chunks = chunk_text(text)

    # Resolve collection
    try:
        rel = file_path.relative_to(KNOWLEDGE_ROOT)
        rel_str = str(rel)
    except ValueError:
        rel_str = file_path.name
    coll = collection or _infer_collection(rel_str)

    # Remove old data if re-indexing
    if row:
        conn.execute("DELETE FROM chunks WHERE source_id = ?", (row["id"],))
        conn.execute("DELETE FROM sources WHERE id = ?", (row["id"],))

    cur = conn.execute(
        """INSERT INTO sources (path, filename, collection, tags, checksum,
                                chunk_count, file_size, indexed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (str(file_path), file_path.name, coll, tags, checksum,
         len(chunks), file_path.stat().st_size, time.time())
    )
    source_id = cur.lastrowid

    for i, chunk in enumerate(chunks):
        conn.execute(
            "INSERT INTO chunks (source_id, chunk_index, content) VALUES (?, ?, ?)",
            (source_id, i, chunk)
        )

    return {
        "path": str(file_path),
        "status": "indexed",
        "collection": coll,
        "chunks": len(chunks),
    }


# ── Pydantic models ─────────────────────────────────────────────────────────

class IngestPathRequest(BaseModel):
    path: str = Field(..., description="File or folder path to ingest, relative to knowledge root or absolute")
    collection: str | None = Field(None, description="Collection name (auto-detected from folder if omitted)")
    tags: str = Field("", description="Comma-separated tags for filtering")
    recursive: bool = Field(True, description="Recurse into subdirectories when path is a folder")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query — keywords or natural language")
    collection: str | None = Field(None, description="Filter to a specific collection")
    top_k: int = Field(5, description="Number of results to return", ge=1, le=50)
    tags: str | None = Field(None, description="Comma-separated tags to filter by")


class DeleteSourceRequest(BaseModel):
    path: str = Field(..., description="Path of the source to delete from the index")


class GetChunkRequest(BaseModel):
    chunk_id: int = Field(..., description="ID of the chunk to retrieve")


# ── App ──────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    logger.info("Knowledge server ready  root=%s  db=%s", KNOWLEDGE_ROOT, KNOWLEDGE_DB)
    yield


app = FastAPI(
    title="Knowledge Server",
    version="1.0.0",
    description="Local document knowledge base: ingest files, search by keyword (FTS5). "
                "Designed as a tool server for Audrey.",
    lifespan=lifespan,
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/ingest_path",
          summary="Ingest a file or folder into the knowledge base",
          operation_id="ingest_path")
async def ep_ingest_path(req: IngestPathRequest):
    """Ingest one file or an entire folder into the knowledge index.
    Files are chunked and stored for keyword search.
    If a file was previously indexed and hasn't changed (by checksum), it is skipped."""

    # Resolve path: allow relative (to KNOWLEDGE_ROOT) or absolute
    raw = Path(req.path)
    if raw.is_absolute():
        target = raw
    else:
        target = Path(KNOWLEDGE_ROOT) / raw

    target = target.resolve()

    # Safety: must be under KNOWLEDGE_ROOT
    kr = Path(KNOWLEDGE_ROOT).resolve()
    if not str(target).startswith(str(kr)):
        return {"error": f"Path must be under {KNOWLEDGE_ROOT}"}

    if not target.exists():
        return {"error": f"Path not found: {req.path}"}

    conn = _get_db()
    results = []
    try:
        if target.is_file():
            results.append(_ingest_single_file(conn, target, req.collection, req.tags))
        else:
            glob_pattern = "**/*" if req.recursive else "*"
            for f in sorted(target.glob(glob_pattern)):
                if f.is_file() and not f.name.startswith("."):
                    results.append(
                        _ingest_single_file(conn, f, req.collection, req.tags)
                    )
        conn.commit()
    finally:
        conn.close()

    summary = {}
    for r in results:
        s = r.get("status", "unknown")
        summary[s] = summary.get(s, 0) + 1

    return {"results": results, "summary": summary}


@app.post("/search_knowledge",
          summary="Search the knowledge base by keyword or phrase",
          operation_id="search_knowledge")
async def ep_search_knowledge(req: SearchRequest):
    """Search indexed documents using full-text keyword search (BM25 ranking).
    Returns the most relevant chunks with source metadata."""

    conn = _get_db()
    try:
        # Build FTS5 query: wrap each word in quotes for safety
        words = req.query.strip().split()
        if not words:
            return {"results": [], "total": 0}
        fts_query = " OR ".join(f'"{w}"' for w in words)

        # Base query with optional filters
        params: list[Any] = [fts_query]
        where_extra = ""

        if req.collection:
            where_extra += " AND s.collection = ?"
            params.append(req.collection)

        if req.tags:
            for tag in req.tags.split(","):
                tag = tag.strip()
                if tag:
                    where_extra += " AND s.tags LIKE ?"
                    params.append(f"%{tag}%")

        params.append(req.top_k)

        rows = conn.execute(f"""
            SELECT c.id, c.chunk_index, c.content,
                   s.path, s.filename, s.collection, s.tags, s.chunk_count,
                   rank
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            JOIN sources s ON s.id = c.source_id
            WHERE chunks_fts MATCH ?
            {where_extra}
            ORDER BY rank
            LIMIT ?
        """, params).fetchall()

        results = [
            {
                "chunk_id": row["id"],
                "chunk_index": row["chunk_index"],
                "total_chunks": row["chunk_count"],
                "content": row["content"],
                "source_path": row["path"],
                "filename": row["filename"],
                "collection": row["collection"],
                "tags": row["tags"],
                "score": round(row["rank"], 4),
            }
            for row in rows
        ]
        return {"results": results, "total": len(results), "query": req.query}
    finally:
        conn.close()


@app.post("/get_chunk",
          summary="Retrieve a specific chunk by ID",
          operation_id="get_chunk")
async def ep_get_chunk(req: GetChunkRequest):
    """Fetch one exact chunk by its ID, with source metadata."""
    conn = _get_db()
    try:
        row = conn.execute("""
            SELECT c.id, c.chunk_index, c.content,
                   s.path, s.filename, s.collection, s.chunk_count
            FROM chunks c
            JOIN sources s ON s.id = c.source_id
            WHERE c.id = ?
        """, (req.chunk_id,)).fetchone()
        if not row:
            return {"error": f"Chunk {req.chunk_id} not found"}
        return {
            "chunk_id": row["id"],
            "chunk_index": row["chunk_index"],
            "total_chunks": row["chunk_count"],
            "content": row["content"],
            "source_path": row["path"],
            "filename": row["filename"],
            "collection": row["collection"],
        }
    finally:
        conn.close()


@app.get("/list_sources",
         summary="List all indexed sources in the knowledge base",
         operation_id="list_sources")
async def ep_list_sources(collection: str | None = None):
    """List all files that have been indexed, with metadata.
    Optionally filter by collection."""
    conn = _get_db()
    try:
        if collection:
            rows = conn.execute(
                "SELECT * FROM sources WHERE collection = ? ORDER BY indexed_at DESC",
                (collection,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sources ORDER BY indexed_at DESC"
            ).fetchall()

        sources = [
            {
                "id": row["id"],
                "path": row["path"],
                "filename": row["filename"],
                "collection": row["collection"],
                "tags": row["tags"],
                "chunk_count": row["chunk_count"],
                "file_size": row["file_size"],
                "indexed_at": row["indexed_at"],
            }
            for row in rows
        ]

        # Collection summary
        coll_rows = conn.execute(
            "SELECT collection, COUNT(*) as count, SUM(chunk_count) as chunks "
            "FROM sources GROUP BY collection ORDER BY collection"
        ).fetchall()
        collections = {r["collection"]: {"sources": r["count"], "chunks": r["chunks"]}
                       for r in coll_rows}

        return {"sources": sources, "total": len(sources), "collections": collections}
    finally:
        conn.close()


@app.post("/delete_source",
          summary="Remove a source and its chunks from the knowledge base",
          operation_id="delete_source")
async def ep_delete_source(req: DeleteSourceRequest):
    """Delete one indexed source and all its chunks from the knowledge base."""
    conn = _get_db()
    try:
        row = conn.execute("SELECT id FROM sources WHERE path = ?",
                           (req.path,)).fetchone()
        if not row:
            return {"error": f"Source not found: {req.path}"}
        conn.execute("DELETE FROM chunks WHERE source_id = ?", (row["id"],))
        conn.execute("DELETE FROM sources WHERE id = ?", (row["id"],))
        conn.commit()
        return {"deleted": req.path}
    finally:
        conn.close()


@app.get("/health")
async def health():
    """Health check — verifies DB is accessible and returns basic stats."""
    try:
        conn = _get_db()
        src = conn.execute("SELECT COUNT(*) as c FROM sources").fetchone()["c"]
        chk = conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
        conn.close()
        return {
            "ok": True,
            "knowledge_root": KNOWLEDGE_ROOT,
            "indexed_sources": src,
            "indexed_chunks": chk,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
