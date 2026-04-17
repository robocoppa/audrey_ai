"""
Knowledge server — local document RAG for Audrey.

A standalone FastAPI service that ingests files from a knowledge directory,
chunks and indexes them with SQLite FTS5 + vector embeddings, and exposes
hybrid search endpoints that Audrey discovers via OpenAPI.

Run:  uvicorn knowledge_server:app --host 0.0.0.0 --port 8002
"""

import asyncio
import base64
import hashlib
import logging
import math
import os
import re
import sqlite3
import struct
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiohttp
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

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "4"))

VISION_MODEL = os.getenv("VISION_MODEL", "llava:34b")
VISION_ENABLED = os.getenv("VISION_ENABLED", "true").lower() in ("true", "1", "yes")
VISION_PROMPT = os.getenv(
    "VISION_PROMPT",
    "Describe this image in detail for a knowledge base. Identify the subject "
    "and its key characteristics. Note visual details (color, texture, shape, "
    "structure, patterns), any labels or text visible, and what domain or "
    "category the image belongs to. Be specific enough that someone could find "
    "this image by searching for related terms.",
)

AUTO_SCAN = os.getenv("AUTO_SCAN", "true").lower() in ("true", "1", "yes")
RESCAN_INTERVAL = int(os.getenv("RESCAN_INTERVAL", "1800"))  # seconds, 0 = startup only
INGEST_QUIET_START = int(os.getenv("INGEST_QUIET_START", "20"))  # hour (0-23), no new scans after this
INGEST_QUIET_END = int(os.getenv("INGEST_QUIET_END", "7"))      # hour (0-23), scans resume after this

# ── Globals ──────────────────────────────────────────────────────────────────

_http_session: aiohttp.ClientSession | None = None
_embedding_dim: int | None = None  # detected on first embed call
_bg_task: asyncio.Task | None = None
_ingest_status: dict[str, Any] = {
    "running": False,
    "phase": "idle",
    "total_files": 0,
    "processed_files": 0,
    "current_file": None,
    "errors": [],
    "last_scan_at": None,
    "last_scan_duration": None,
}

# ── Text extraction ──────────────────────────────────────────────────────────

PLAIN_EXTENSIONS = {
    ".txt", ".md", ".log", ".csv", ".json", ".yaml", ".yml",
    ".xml", ".ini", ".conf", ".py", ".js", ".ts", ".sh", ".toml",
    ".rst", ".cfg", ".env", ".sql", ".r", ".go", ".java", ".c",
    ".cpp", ".h", ".hpp", ".rb", ".php", ".swift", ".kt",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

INGESTIBLE_EXTENSIONS = PLAIN_EXTENSIONS | {".pdf", ".html", ".htm", ".docx"} | IMAGE_EXTENSIONS


def _is_ingestible(path: Path) -> bool:
    """Check if a file type is supported for ingestion."""
    return path.suffix.lower() in INGESTIBLE_EXTENSIONS


async def caption_image(path: Path) -> str | None:
    """Generate a text caption for an image using a vision LLM via Ollama."""
    if not VISION_ENABLED or _http_session is None:
        return None
    try:
        image_bytes = path.read_bytes()
        b64 = base64.b64encode(image_bytes).decode("ascii")
        async with _http_session.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": VISION_MODEL,
                "prompt": VISION_PROMPT,
                "images": [b64],
                "stream": False,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                logger.warning("Vision caption failed for %s: HTTP %d", path.name, resp.status)
                return None
            data = await resp.json()
            caption = data.get("response", "").strip()
            if caption:
                # Prepend filename for searchability
                return f"[Image: {path.name}]\n\n{caption}"
            return None
    except Exception as e:
        logger.warning("Vision caption error for %s: %s", path.name, e)
        return None


async def extract_text(path: Path) -> str | None:
    """Extract text content from a file. Returns None on failure."""
    suffix = path.suffix.lower()
    try:
        if suffix in IMAGE_EXTENSIONS:
            return await caption_image(path)

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

def _split_into_blocks(text: str) -> list[str]:
    """Split text into structural blocks: headings, paragraphs, code fences."""
    lines = text.split("\n")
    blocks: list[str] = []
    current_block: list[str] = []
    in_code_fence = False

    for line in lines:
        if line.strip().startswith("```"):
            if in_code_fence:
                current_block.append(line)
                blocks.append("\n".join(current_block))
                current_block = []
                in_code_fence = False
            else:
                if current_block:
                    blocks.append("\n".join(current_block))
                    current_block = []
                current_block.append(line)
                in_code_fence = True
        elif in_code_fence:
            current_block.append(line)
        elif re.match(r"^#{1,6}\s", line):
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
            current_block.append(line)
        elif line.strip() == "" and current_block:
            blocks.append("\n".join(current_block))
            current_block = []
        else:
            current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block))

    return [b for b in blocks if b.strip()]


def _char_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fallback: split text into overlapping chunks by character count."""
    chunks: list[str] = []
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


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, respecting structural boundaries."""
    if len(text) <= chunk_size:
        return [text]

    blocks = _split_into_blocks(text)
    chunks: list[str] = []
    current = ""
    overlap_prefix = ""

    for block in blocks:
        candidate = (current + "\n\n" + block).strip() if current else block
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current.strip())
                overlap_prefix = current[-overlap:] if len(current) > overlap else current
            if len(block) > chunk_size:
                sub_chunks = _char_split(block, chunk_size, overlap)
                chunks.extend(sub_chunks)
                overlap_prefix = sub_chunks[-1][-overlap:] if sub_chunks else ""
                current = ""
            else:
                current = (overlap_prefix + "\n\n" + block).strip() if overlap_prefix else block

    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text]


# ── Embeddings ───────────────────────────────────────────────────────────────

def _pack_embedding(vec: list[float]) -> bytes:
    """Pack a float list into a compact binary blob (float32)."""
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_embedding(blob: bytes) -> list[float]:
    """Unpack a binary blob back into a float list."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors. Pure Python, no numpy."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def embed_text(text: str) -> list[float] | None:
    """Get embedding vector from Ollama. Returns None on failure."""
    global _embedding_dim
    if _http_session is None:
        return None
    try:
        async with _http_session.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                logger.warning("Embedding request failed: %d", resp.status)
                return None
            data = await resp.json()
            embeddings = data.get("embeddings")
            if embeddings and len(embeddings) > 0:
                vec = embeddings[0]
                if _embedding_dim is None:
                    _embedding_dim = len(vec)
                    logger.info("Embedding dimension detected: %d", _embedding_dim)
                return vec
            return None
    except Exception as e:
        logger.warning("Embedding failed: %s", e)
        return None


async def embed_batch(texts: list[str]) -> list[list[float] | None]:
    """Embed multiple texts concurrently with bounded parallelism."""
    semaphore = asyncio.Semaphore(EMBED_CONCURRENCY)

    async def _embed_with_limit(text: str) -> list[float] | None:
        async with semaphore:
            return await embed_text(text)

    return list(await asyncio.gather(*[_embed_with_limit(t) for t in texts]))


# ── Database ─────────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(KNOWLEDGE_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
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
            embedding   BLOB,
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
    # Migrate: add embedding column if upgrading from Phase 1
    try:
        conn.execute("SELECT embedding FROM chunks LIMIT 0")
    except sqlite3.OperationalError:
        logger.info("Migrating DB: adding embedding column to chunks")
        conn.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB")
    # Migrate: add embed_model column if upgrading from Phase 2
    try:
        conn.execute("SELECT embed_model FROM chunks LIMIT 0")
    except sqlite3.OperationalError:
        logger.info("Migrating DB: adding embed_model column to chunks")
        conn.execute("ALTER TABLE chunks ADD COLUMN embed_model TEXT")
    conn.close()
    logger.info("Knowledge DB initialized at %s", KNOWLEDGE_DB)


def _mark_stale_embeddings():
    """Null out embeddings created by a different model so they get re-embedded."""
    conn = _get_db()
    stale = conn.execute(
        "SELECT COUNT(*) as c FROM chunks "
        "WHERE embedding IS NOT NULL AND (embed_model IS NULL OR embed_model != ?)",
        (EMBEDDING_MODEL,),
    ).fetchone()["c"]
    if stale > 0:
        logger.info("Marking %d chunks as stale (model changed to %s)", stale, EMBEDDING_MODEL)
        conn.execute(
            "UPDATE chunks SET embedding = NULL, embed_model = NULL "
            "WHERE embedding IS NOT NULL AND (embed_model IS NULL OR embed_model != ?)",
            (EMBEDDING_MODEL,),
        )
        conn.commit()
    conn.close()


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

async def _ingest_single_file(conn: sqlite3.Connection, file_path: Path,
                              collection: str | None, tags: str) -> dict[str, Any]:
    """Ingest one file. Returns status dict."""
    if not file_path.exists():
        return {"path": str(file_path), "status": "not_found"}
    if not file_path.is_file():
        return {"path": str(file_path), "status": "not_a_file"}
    if not _is_ingestible(file_path):
        return {"path": str(file_path), "status": "skipped_unsupported"}

    checksum = _file_checksum(file_path)

    # Check if already indexed with same checksum
    row = conn.execute(
        "SELECT id, checksum FROM sources WHERE path = ?",
        (str(file_path),)
    ).fetchone()
    if row and row["checksum"] == checksum:
        return {"path": str(file_path), "status": "unchanged"}

    text = await extract_text(file_path)
    if text is None or not text.strip():
        return {"path": str(file_path), "status": "unsupported_or_empty"}

    chunks = chunk_text(text)

    # Embed all chunks
    embeddings = await embed_batch(chunks)
    embedded_count = sum(1 for e in embeddings if e is not None)

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

    conn.executemany(
        "INSERT INTO chunks (source_id, chunk_index, content, embedding, embed_model) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (source_id, i, chunk,
             _pack_embedding(emb) if emb else None,
             EMBEDDING_MODEL if emb else None)
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
    )

    return {
        "path": str(file_path),
        "status": "indexed",
        "collection": coll,
        "chunks": len(chunks),
        "embedded": embedded_count,
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
    semantic_weight: float = Field(0.5, description="Weight for semantic/embedding search (0-1)", ge=0, le=1)
    keyword_weight: float = Field(0.5, description="Weight for keyword/BM25 search (0-1)", ge=0, le=1)


class DeleteSourceRequest(BaseModel):
    path: str = Field(..., description="Path of the source to delete from the index")


class GetChunkRequest(BaseModel):
    chunk_id: int = Field(..., description="ID of the chunk to retrieve")


# ── Background scan ─────────────────────────────────────────────────────────

async def _background_scan():
    """Scan KNOWLEDGE_ROOT and ingest new/changed files."""
    global _ingest_status
    _ingest_status["running"] = True
    _ingest_status["phase"] = "scanning"
    _ingest_status["errors"] = []
    t0 = time.monotonic()

    try:
        kr = Path(KNOWLEDGE_ROOT).resolve()
        all_files = [
            f for f in sorted(kr.rglob("*"))
            if f.is_file() and not f.name.startswith(".") and _is_ingestible(f)
        ]
        _ingest_status["total_files"] = len(all_files)
        _ingest_status["processed_files"] = 0
        _ingest_status["phase"] = "ingesting"
        logger.info("Background scan: found %d ingestible files", len(all_files))

        conn = _get_db()
        try:
            for f in all_files:
                _ingest_status["current_file"] = str(f)
                try:
                    result = await _ingest_single_file(conn, f, None, "")
                    if result.get("status") == "indexed":
                        conn.commit()
                        logger.info("Background indexed: %s (%d chunks)",
                                    f.name, result.get("chunks", 0))
                except Exception as e:
                    _ingest_status["errors"].append({"file": str(f), "error": str(e)})
                    logger.error("Background ingest error for %s: %s", f, e)
                _ingest_status["processed_files"] += 1
                await asyncio.sleep(0)

            # Re-embed chunks with stale/missing embeddings
            batch_size = 100
            offset = 0
            _ingest_status["phase"] = "re_embedding"
            while True:
                stale_chunks = conn.execute(
                    "SELECT c.id, c.content FROM chunks c "
                    "WHERE c.embedding IS NULL LIMIT ? OFFSET ?",
                    (batch_size, offset),
                ).fetchall()
                if not stale_chunks:
                    break
                logger.info("Re-embedding batch of %d stale chunks", len(stale_chunks))
                for chunk_row in stale_chunks:
                    vec = await embed_text(chunk_row["content"])
                    if vec:
                        conn.execute(
                            "UPDATE chunks SET embedding = ?, embed_model = ? WHERE id = ?",
                            (_pack_embedding(vec), EMBEDDING_MODEL, chunk_row["id"]),
                        )
                    await asyncio.sleep(0)
                conn.commit()
                offset += batch_size
        finally:
            conn.close()

    except Exception as e:
        logger.error("Background scan failed: %s", e)
        _ingest_status["errors"].append({"file": None, "error": str(e)})
    finally:
        elapsed = round(time.monotonic() - t0, 2)
        _ingest_status["running"] = False
        _ingest_status["phase"] = "idle"
        _ingest_status["current_file"] = None
        _ingest_status["last_scan_at"] = time.time()
        _ingest_status["last_scan_duration"] = elapsed
        logger.info("Background scan complete in %.2fs", elapsed)


def _in_quiet_hours() -> bool:
    """Check if current local time is within the ingestion quiet window."""
    hour = time.localtime().tm_hour
    if INGEST_QUIET_START <= INGEST_QUIET_END:
        return INGEST_QUIET_START <= hour < INGEST_QUIET_END
    # Wraps midnight (e.g. 20:00 → 07:00)
    return hour >= INGEST_QUIET_START or hour < INGEST_QUIET_END


async def _periodic_scan():
    """Run background scan on startup, then repeat at RESCAN_INTERVAL."""
    if _in_quiet_hours():
        logger.info("Skipping startup scan — quiet hours (%02d:00–%02d:00)",
                     INGEST_QUIET_START, INGEST_QUIET_END)
    else:
        await _background_scan()
    while RESCAN_INTERVAL > 0:
        await asyncio.sleep(RESCAN_INTERVAL)
        if _in_quiet_hours():
            logger.debug("Skipping periodic scan — quiet hours")
            continue
        await _background_scan()


# ── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_session, _bg_task
    _init_db()
    _mark_stale_embeddings()
    _http_session = aiohttp.ClientSession()
    logger.info("Knowledge server ready  root=%s  db=%s  embedding_model=%s",
                KNOWLEDGE_ROOT, KNOWLEDGE_DB, EMBEDDING_MODEL)
    if AUTO_SCAN:
        _bg_task = asyncio.create_task(_periodic_scan())
    yield
    if _bg_task and not _bg_task.done():
        _bg_task.cancel()
        try:
            await _bg_task
        except asyncio.CancelledError:
            pass
    await _http_session.close()


app = FastAPI(
    title="Knowledge Server",
    version="3.0.0",
    description="Local document knowledge base with hybrid keyword + semantic search. "
                "Designed as a tool server for Audrey.",
    lifespan=lifespan,
)


# ── Search helpers ───────────────────────────────────────────────────────────

def _keyword_search(conn: sqlite3.Connection, query: str,
                    collection: str | None, tags: str | None,
                    limit: int) -> list[dict[str, Any]]:
    """Run FTS5 keyword search. Returns ranked results."""
    words = query.strip().split()
    if not words:
        return []
    fts_query = " OR ".join(f'"{w}"' for w in words)

    params: list[Any] = [fts_query]
    where_extra = ""

    if collection:
        where_extra += " AND s.collection = ?"
        params.append(collection)
    if tags:
        for tag in tags.split(","):
            tag = tag.strip()
            if tag:
                where_extra += " AND s.tags LIKE ?"
                params.append(f"%{tag}%")

    params.append(limit)

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

    return [
        {
            "chunk_id": row["id"],
            "chunk_index": row["chunk_index"],
            "total_chunks": row["chunk_count"],
            "content": row["content"],
            "source_path": row["path"],
            "filename": row["filename"],
            "collection": row["collection"],
            "tags": row["tags"],
            "keyword_score": -row["rank"],  # FTS5 rank is negative (lower=better)
        }
        for row in rows
    ]


def _semantic_search(conn: sqlite3.Connection, query_embedding: list[float],
                     collection: str | None, tags: str | None,
                     limit: int) -> list[dict[str, Any]]:
    """Brute-force cosine similarity search over stored embeddings."""
    where_clauses = ["c.embedding IS NOT NULL"]
    params: list[Any] = []

    if collection:
        where_clauses.append("s.collection = ?")
        params.append(collection)
    if tags:
        for tag in tags.split(","):
            tag = tag.strip()
            if tag:
                where_clauses.append("s.tags LIKE ?")
                params.append(f"%{tag}%")

    where_sql = " AND ".join(where_clauses)

    rows = conn.execute(f"""
        SELECT c.id, c.chunk_index, c.content, c.embedding,
               s.path, s.filename, s.collection, s.tags, s.chunk_count
        FROM chunks c
        JOIN sources s ON s.id = c.source_id
        WHERE {where_sql}
    """, params).fetchall()

    scored = []
    for row in rows:
        emb = _unpack_embedding(row["embedding"])
        sim = _cosine_similarity(query_embedding, emb)
        scored.append({
            "chunk_id": row["id"],
            "chunk_index": row["chunk_index"],
            "total_chunks": row["chunk_count"],
            "content": row["content"],
            "source_path": row["path"],
            "filename": row["filename"],
            "collection": row["collection"],
            "tags": row["tags"],
            "semantic_score": sim,
        })

    scored.sort(key=lambda x: x["semantic_score"], reverse=True)
    return scored[:limit]


def _merge_results(keyword_results: list[dict], semantic_results: list[dict],
                   keyword_weight: float, semantic_weight: float,
                   top_k: int) -> list[dict]:
    """Merge keyword and semantic results with weighted scoring."""
    # Normalize scores to 0-1 range within each result set
    if keyword_results:
        max_kw = max(r["keyword_score"] for r in keyword_results)
        min_kw = min(r["keyword_score"] for r in keyword_results)
        rng = max_kw - min_kw if max_kw != min_kw else 1.0
        for r in keyword_results:
            r["keyword_norm"] = (r["keyword_score"] - min_kw) / rng

    if semantic_results:
        max_sem = max(r["semantic_score"] for r in semantic_results)
        min_sem = min(r["semantic_score"] for r in semantic_results)
        rng = max_sem - min_sem if max_sem != min_sem else 1.0
        for r in semantic_results:
            r["semantic_norm"] = (r["semantic_score"] - min_sem) / rng

    # Merge into a single dict keyed by chunk_id
    merged: dict[int, dict] = {}

    for r in keyword_results:
        cid = r["chunk_id"]
        merged[cid] = {
            **r,
            "keyword_norm": r.get("keyword_norm", 0),
            "semantic_norm": 0,
        }

    for r in semantic_results:
        cid = r["chunk_id"]
        if cid in merged:
            merged[cid]["semantic_norm"] = r.get("semantic_norm", 0)
            merged[cid]["semantic_score"] = r.get("semantic_score", 0)
        else:
            merged[cid] = {
                **r,
                "keyword_norm": 0,
                "semantic_norm": r.get("semantic_norm", 0),
            }

    # Compute combined score
    total_weight = keyword_weight + semantic_weight
    if total_weight == 0:
        total_weight = 1.0

    for item in merged.values():
        item["score"] = round(
            (keyword_weight * item["keyword_norm"]
             + semantic_weight * item["semantic_norm"]) / total_weight,
            4
        )

    # Sort by combined score descending, take top_k
    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    # Clean up internal fields
    for r in ranked:
        r.pop("keyword_norm", None)
        r.pop("semantic_norm", None)
        r.pop("keyword_score", None)
        r.pop("semantic_score", None)

    return ranked


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/ingest_path",
          summary="Ingest a file or folder into the knowledge base",
          operation_id="ingest_path")
async def ep_ingest_path(req: IngestPathRequest):
    """Ingest one file or an entire folder into the knowledge index.
    Files are chunked, embedded, and stored for hybrid search.
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
    t0 = time.monotonic()
    try:
        if target.is_file():
            results.append(await _ingest_single_file(conn, target, req.collection, req.tags))
        else:
            glob_pattern = "**/*" if req.recursive else "*"
            for f in sorted(target.glob(glob_pattern)):
                if f.is_file() and not f.name.startswith(".") and _is_ingestible(f):
                    results.append(
                        await _ingest_single_file(conn, f, req.collection, req.tags)
                    )
        conn.commit()
    finally:
        conn.close()

    elapsed = round(time.monotonic() - t0, 2)
    summary = {}
    for r in results:
        s = r.get("status", "unknown")
        summary[s] = summary.get(s, 0) + 1

    logger.info("Ingestion complete: %s in %.2fs", summary, elapsed)
    return {"results": results, "summary": summary, "elapsed_seconds": elapsed}


@app.post("/search_knowledge",
          summary="Search the knowledge base by keyword, meaning, or both",
          operation_id="search_knowledge")
async def ep_search_knowledge(req: SearchRequest):
    """Search indexed documents using hybrid keyword + semantic search.
    Keyword search uses FTS5 BM25 ranking. Semantic search uses embedding
    cosine similarity. Results are merged with configurable weights.
    Falls back to keyword-only if embeddings are unavailable."""

    conn = _get_db()
    try:
        keyword_results = []
        semantic_results = []

        # Keyword search (always available)
        if req.keyword_weight > 0:
            keyword_results = _keyword_search(
                conn, req.query, req.collection, req.tags,
                limit=req.top_k * 2  # fetch more to improve merge quality
            )

        # Semantic search (if embeddings available)
        if req.semantic_weight > 0:
            query_embedding = await embed_text(req.query)
            if query_embedding:
                semantic_results = _semantic_search(
                    conn, query_embedding, req.collection, req.tags,
                    limit=req.top_k * 2
                )
            elif req.keyword_weight == 0:
                # Semantic-only requested but embeddings unavailable
                return {"results": [], "total": 0, "query": req.query,
                        "note": "Semantic search unavailable — embedding model not reachable"}

        # Merge results
        if keyword_results and semantic_results:
            results = _merge_results(
                keyword_results, semantic_results,
                req.keyword_weight, req.semantic_weight, req.top_k
            )
            search_mode = "hybrid"
        elif keyword_results:
            # Keyword only — normalize scores
            for r in keyword_results:
                r["score"] = round(r.pop("keyword_score", 0), 4)
            results = keyword_results[:req.top_k]
            search_mode = "keyword"
        elif semantic_results:
            # Semantic only — use cosine similarity as score
            for r in semantic_results:
                r["score"] = round(r.pop("semantic_score", 0), 4)
            results = semantic_results[:req.top_k]
            search_mode = "semantic"
        else:
            results = []
            search_mode = "none"

        return {
            "results": results,
            "total": len(results),
            "query": req.query,
            "search_mode": search_mode,
        }
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
    """Health check — verifies DB is accessible, embedding model reachable, and returns stats."""
    result: dict[str, Any] = {"ok": True, "knowledge_root": KNOWLEDGE_ROOT}
    try:
        conn = _get_db()
        src = conn.execute("SELECT COUNT(*) as c FROM sources").fetchone()["c"]
        chk = conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
        emb = conn.execute(
            "SELECT COUNT(*) as c FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()["c"]
        conn.close()
        result.update({
            "indexed_sources": src,
            "indexed_chunks": chk,
            "embedded_chunks": emb,
            "embedding_model": EMBEDDING_MODEL,
        })
    except Exception as e:
        result.update({"ok": False, "error": str(e)})

    # Check if embedding model is reachable
    test_vec = await embed_text("test")
    result["embedding_available"] = test_vec is not None
    if test_vec:
        result["embedding_dimension"] = len(test_vec)

    return result


@app.get("/ingest_status",
         summary="Check background ingestion status",
         operation_id="ingest_status")
async def ep_ingest_status():
    """Returns the current state of background ingestion."""
    return {
        "running": _ingest_status["running"],
        "phase": _ingest_status["phase"],
        "progress": {
            "processed": _ingest_status["processed_files"],
            "total": _ingest_status["total_files"],
        },
        "current_file": _ingest_status["current_file"],
        "errors": _ingest_status["errors"][-20:],
        "last_scan_at": _ingest_status["last_scan_at"],
        "last_scan_duration": _ingest_status["last_scan_duration"],
    }
