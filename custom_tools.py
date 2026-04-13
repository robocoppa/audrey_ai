"""
Custom local tool server.

A standalone FastAPI app exposing tools that the orchestrator discovers
via its /openapi.json spec — no hardcoding needed.

Run:  uvicorn custom_tools:app --host 0.0.0.0 --port 8001
"""

import json
import logging
import os
import platform
import shutil
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("custom-tools")

# ── Config ───────────────────────────────────────────────────────────────────

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")
SEARXNG_MAX_RESULTS = int(os.getenv("SEARXNG_MAX_RESULTS", "5"))
SANDBOX_DIR = os.getenv("TOOLS_SANDBOX_DIR", "/app/sandbox")
MEMORY_DB = os.getenv("TOOLS_MEMORY_DB", "/app/data/memory.db")
PYTHON_TIMEOUT = int(os.getenv("TOOLS_PYTHON_TIMEOUT", "30"))
PYTHON_MAX_OUTPUT = int(os.getenv("TOOLS_PYTHON_MAX_OUTPUT", "10000"))
DOC_MAX_CHARS = int(os.getenv("TOOLS_DOC_MAX_CHARS", "50000"))

_http_session: Optional[aiohttp.ClientSession] = None

# ── App ──────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_session
    _http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
    os.makedirs(SANDBOX_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MEMORY_DB) or ".", exist_ok=True)
    _init_memory_db()
    logger.info("Custom tools server ready  sandbox=%s  searxng=%s", SANDBOX_DIR, SEARXNG_URL)
    yield
    await _http_session.close()


app = FastAPI(title="Custom Tools", version="1.0.0",
              description="Local tools: search, memory, python sandbox, system monitor, document reader",
              lifespan=lifespan)


# ── Request / response models ────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    max_results: int = SEARXNG_MAX_RESULTS

class MemoryStoreRequest(BaseModel):
    key: str
    value: str

class MemoryRecallRequest(BaseModel):
    key: str

class MemorySearchRequest(BaseModel):
    query: str

class FileReadRequest(BaseModel):
    path: str

class FileWriteRequest(BaseModel):
    path: str
    content: str

class FileListRequest(BaseModel):
    path: str = "."

class SqlQueryRequest(BaseModel):
    query: str

class PythonRunRequest(BaseModel):
    code: str

class DocumentReadRequest(BaseModel):
    path: str


# ══════════════════════════════════════════════════════════════════════════════
#  1 — Web Search  (SearXNG)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/web_search", summary="Search the web for current information",
          operation_id="web_search")
async def ep_web_search(req: SearchRequest):
    """Search the internet via SearXNG. Returns titles, snippets, and URLs."""
    try:
        async with _http_session.get(
            f"{SEARXNG_URL}/search",
            params={"q": req.query, "format": "json", "safesearch": "0"},
        ) as resp:
            if resp.status != 200:
                return {"results": [], "error": f"SearXNG {resp.status}"}
            data = await resp.json()
        results = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in data.get("results", [])[:req.max_results]
        ]
        return {"results": results}
    except Exception as e:
        return {"results": [], "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  2 — Memory  (SQLite key-value)
# ══════════════════════════════════════════════════════════════════════════════

def _init_memory_db():
    conn = sqlite3.connect(MEMORY_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS memories (
        key TEXT PRIMARY KEY, value TEXT NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    conn.commit()
    conn.close()


@app.post("/memory_store", summary="Store information for later recall",
          operation_id="memory_store")
async def ep_memory_store(req: MemoryStoreRequest):
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(MEMORY_DB)
    conn.execute(
        "INSERT OR REPLACE INTO memories (key,value,created_at,updated_at) "
        "VALUES (?,?,COALESCE((SELECT created_at FROM memories WHERE key=?),?),?)",
        (req.key, req.value, req.key, now, now))
    conn.commit()
    conn.close()
    return {"stored": req.key}


@app.post("/memory_recall", summary="Recall a stored memory by key",
          operation_id="memory_recall")
async def ep_memory_recall(req: MemoryRecallRequest):
    conn = sqlite3.connect(MEMORY_DB)
    row = conn.execute("SELECT value,updated_at FROM memories WHERE key=?", (req.key,)).fetchone()
    conn.close()
    if row:
        return {"key": req.key, "value": row[0], "updated_at": row[1]}
    return {"error": f"No memory for: {req.key}"}


@app.post("/memory_search", summary="Search memories by keyword",
          operation_id="memory_search")
async def ep_memory_search(req: MemorySearchRequest):
    conn = sqlite3.connect(MEMORY_DB)
    rows = conn.execute(
        "SELECT key,value,updated_at FROM memories WHERE key LIKE ? OR value LIKE ? "
        "ORDER BY updated_at DESC LIMIT 10",
        (f"%{req.query}%", f"%{req.query}%")).fetchall()
    conn.close()
    return {"results": [{"key": r[0], "value": r[1], "updated_at": r[2]} for r in rows]}


@app.get("/memory_list", summary="List all stored memory keys",
         operation_id="memory_list")
async def ep_memory_list():
    conn = sqlite3.connect(MEMORY_DB)
    rows = conn.execute("SELECT key,updated_at FROM memories ORDER BY updated_at DESC").fetchall()
    conn.close()
    return {"keys": [{"key": r[0], "updated_at": r[1]} for r in rows]}


# ══════════════════════════════════════════════════════════════════════════════
#  3 — Python Sandbox
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/run_python", summary="Execute Python code and return output",
          operation_id="run_python")
async def ep_run_python(req: PythonRunRequest):
    """Run Python in a subprocess with timeout. For calculations, data processing, verification."""
    try:
        r = subprocess.run(
            ["python3", "-c", req.code],
            capture_output=True, text=True, timeout=PYTHON_TIMEOUT,
            cwd=SANDBOX_DIR,
            env={"PATH": "/usr/bin:/usr/local/bin", "HOME": SANDBOX_DIR,
                 "PYTHONDONTWRITEBYTECODE": "1"},
        )
        out = ""
        if r.stdout:
            out += r.stdout
        if r.stderr:
            out += f"\nSTDERR:\n{r.stderr}"
        if r.returncode != 0:
            out += f"\nExit code: {r.returncode}"
        if len(out) > PYTHON_MAX_OUTPUT:
            out = out[:PYTHON_MAX_OUTPUT] + "\n[truncated]"
        return {"output": out or "(no output)"}
    except subprocess.TimeoutExpired:
        return {"output": f"Timed out after {PYTHON_TIMEOUT}s"}
    except Exception as e:
        return {"output": f"Error: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
#  4 — System Monitor
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/system_stats", summary="Get CPU, memory, disk, and GPU usage",
         operation_id="system_stats")
async def ep_system_stats():
    stats: Dict[str, Any] = {"timestamp": datetime.utcnow().isoformat(),
                              "platform": platform.platform(),
                              "cpu": {}, "memory": {}, "disk": {}, "gpu": []}
    try:
        load = os.getloadavg()
        stats["cpu"] = {"load_1m": load[0], "load_5m": load[1], "load_15m": load[2],
                        "cores": os.cpu_count()}
    except OSError:
        stats["cpu"] = {"cores": os.cpu_count()}

    try:
        with open("/proc/meminfo") as f:
            mi = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    mi[parts[0].strip()] = int(parts[1].strip().split()[0])
            total, avail = mi.get("MemTotal", 0), mi.get("MemAvailable", 0)
            stats["memory"] = {"total_mb": total // 1024, "available_mb": avail // 1024,
                               "used_pct": round((1 - avail / max(total, 1)) * 100, 1)}
    except Exception:
        pass

    try:
        u = shutil.disk_usage("/")
        stats["disk"] = {"total_gb": round(u.total / 1073741824, 1),
                         "free_gb": round(u.free / 1073741824, 1),
                         "used_pct": round(u.used / max(u.total, 1) * 100, 1)}
    except Exception:
        pass

    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,"
             "utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) >= 6:
                    stats["gpu"].append({"name": p[0], "vram_total_mb": int(p[1]),
                                         "vram_used_mb": int(p[2]), "vram_free_mb": int(p[3]),
                                         "util_pct": int(p[4]), "temp_c": int(p[5])})
    except Exception:
        pass

    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  5 — Document Reader
# ══════════════════════════════════════════════════════════════════════════════

def _safe_path(user_path: str) -> Path:
    base = Path(SANDBOX_DIR).resolve()
    target = (base / user_path).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError(f"Path escapes sandbox: {user_path}")
    return target


@app.post("/read_document", summary="Extract text from PDF, HTML, DOCX, or plain text files",
          operation_id="read_document")
async def ep_read_document(req: DocumentReadRequest):
    try:
        p = _safe_path(req.path)
    except ValueError as e:
        return {"error": str(e)}
    if not p.exists():
        return {"error": f"Not found: {req.path}"}

    suffix = p.suffix.lower()
    try:
        if suffix in {".txt", ".md", ".log", ".csv", ".json", ".yaml", ".yml",
                      ".xml", ".ini", ".conf", ".py", ".js", ".ts", ".sh", ".toml"}:
            text = p.read_text(errors="replace")
            return {"content": text[:DOC_MAX_CHARS]}

        if suffix == ".pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(str(p))
                pages = [f"--- Page {i+1} ---\n{pg.extract_text() or ''}"
                         for i, pg in enumerate(reader.pages[:50])]
                return {"content": "\n\n".join(pages)[:DOC_MAX_CHARS]}
            except ImportError:
                return {"error": "PyPDF2 not installed"}

        if suffix in {".html", ".htm"}:
            import re as _re
            html = p.read_text(errors="replace")
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style"]):
                    tag.decompose()
                return {"content": soup.get_text(separator="\n", strip=True)[:DOC_MAX_CHARS]}
            except ImportError:
                return {"content": _re.sub(r"\s+", " ", _re.sub(r"<[^>]+>", " ", html)).strip()[:DOC_MAX_CHARS]}

        if suffix == ".docx":
            try:
                from docx import Document
                doc = Document(str(p))
                return {"content": "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())[:DOC_MAX_CHARS]}
            except ImportError:
                return {"error": "python-docx not installed"}

        return {"error": f"Unsupported: {suffix}"}
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  6 — Filesystem  (sandboxed)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/read_file", summary="Read a file from the workspace",
          operation_id="read_file")
async def ep_read_file(req: FileReadRequest):
    try:
        p = _safe_path(req.path)
        if not p.exists():
            return {"error": f"Not found: {req.path}"}
        if p.stat().st_size > 1_000_000:
            return {"error": "File too large (>1MB)"}
        return {"content": p.read_text(errors="replace")}
    except ValueError as e:
        return {"error": str(e)}


@app.post("/write_file", summary="Write content to a file in the workspace",
          operation_id="write_file")
async def ep_write_file(req: FileWriteRequest):
    try:
        p = _safe_path(req.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(req.content)
        return {"written": req.path, "bytes": len(req.content)}
    except ValueError as e:
        return {"error": str(e)}


@app.post("/list_files", summary="List files and directories in the workspace",
          operation_id="list_files")
async def ep_list_files(req: FileListRequest):
    try:
        p = _safe_path(req.path)
        if not p.is_dir():
            return {"error": f"Not a directory: {req.path}"}
        entries = [{"name": item.name,
                    "path": str(item.relative_to(Path(SANDBOX_DIR).resolve())),
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None}
                   for item in sorted(p.iterdir())]
        return {"entries": entries}
    except ValueError as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  7 — SQL Query  (local SQLite)
# ══════════════════════════════════════════════════════════════════════════════

SQL_DB = os.getenv("TOOLS_SQL_DB", "/app/data/local.db")
_SQL_ALLOWED = {"SELECT", "PRAGMA", "CREATE", "INSERT", "UPDATE", "DELETE"}

@app.post("/sql_query", summary="Execute a SQL query against the local database",
          operation_id="sql_query")
async def ep_sql_query(req: SqlQueryRequest):
    first = req.query.strip().split()[0].upper() if req.query.strip() else ""
    if first not in _SQL_ALLOWED:
        return {"error": f"Not allowed: {first}"}
    try:
        conn = sqlite3.connect(SQL_DB)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(req.query)
        if first in {"SELECT", "PRAGMA"}:
            rows = [dict(r) for r in cur.fetchmany(100)]
            conn.close()
            return {"rows": rows, "count": len(rows)}
        conn.commit()
        affected = cur.rowcount
        conn.close()
        return {"ok": True, "rows_affected": affected}
    except Exception as e:
        return {"error": str(e)}


@app.get("/sql_schema", summary="Get database schema",
         operation_id="sql_schema")
async def ep_sql_schema():
    try:
        conn = sqlite3.connect(SQL_DB)
        tables = [r[0] for r in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table'").fetchall() if r[0]]
        conn.close()
        return {"tables": tables}
    except Exception as e:
        return {"error": str(e)}


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"ok": True}
