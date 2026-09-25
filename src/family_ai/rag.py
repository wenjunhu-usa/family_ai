import io
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import psycopg


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    title: str
    content: str
    score: float
    chunk_index: int


def chunk_text(text: str, max_chars: int = 850, overlap: int = 120) -> list[str]:
    clean = re.sub(r"\r\n?", "\n", text)
    clean = re.sub(r"[ \t]+", " ", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    if not clean:
        return []
    chunks = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + max_chars)
        if end < len(clean):
            boundary = max(clean.rfind(mark, start + max_chars // 2, end) for mark in ("\n\n", "。", ". ", "\n"))
            if boundary > start:
                end = boundary + 1
        piece = clean[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap)
    return chunks


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    if suffix not in {".txt", ".md", ".markdown", ".csv", ".json", ".html", ".htm"}:
        raise ValueError("Supported files: PDF, TXT, Markdown, CSV, JSON, HTML")
    text = data.decode("utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
    return text


class RagStore:
    def __init__(self, database_url: str, ollama_base_url: str, embedding_model: str):
        self.database_url = database_url
        self.embed_url = ollama_base_url.rstrip("/") + "/api/embed"
        self.embedding_model = embedding_model
        self._setup()

    def connect(self):
        return psycopg.connect(self.database_url, row_factory=psycopg.rows.dict_row)

    def _setup(self) -> None:
        with self.connect() as db:
            db.execute("CREATE EXTENSION IF NOT EXISTS vector")
            db.execute("""CREATE TABLE IF NOT EXISTS family_rag_documents (
                id TEXT PRIMARY KEY, member_id TEXT NOT NULL, scope TEXT NOT NULL CHECK(scope IN ('private','family')),
                title TEXT NOT NULL, filename TEXT NOT NULL, content_type TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())""")
            db.execute("""CREATE TABLE IF NOT EXISTS family_rag_chunks (
                id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES family_rag_documents(id) ON DELETE CASCADE,
                member_id TEXT NOT NULL, scope TEXT NOT NULL CHECK(scope IN ('private','family')),
                chunk_index INTEGER NOT NULL, content TEXT NOT NULL, embedding vector(1024) NOT NULL,
                UNIQUE(document_id, chunk_index))""")
            db.execute("CREATE INDEX IF NOT EXISTS family_rag_chunks_access ON family_rag_chunks(member_id, scope)")
            db.execute("CREATE INDEX IF NOT EXISTS family_rag_chunks_embedding_hnsw ON family_rag_chunks USING hnsw (embedding vector_cosine_ops)")

    @staticmethod
    def vector_literal(vector: list[float]) -> str:
        return "[" + ",".join(f"{value:.9g}" for value in vector) + "]"

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.embedding_model, "input": texts}).encode()
        request = urllib.request.Request(self.embed_url, data=payload, headers={"content-type": "application/json"})
        with urllib.request.urlopen(request, timeout=180) as response:
            vectors = json.loads(response.read())["embeddings"]
        return [[float(value) for value in vector] for vector in vectors]

    def ingest(self, member_id: str, scope: str, filename: str, content_type: str, data: bytes) -> dict:
        text = extract_text(filename, data)
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No readable text was found in this file")
        if len(chunks) > 500:
            raise ValueError("Document is too large; maximum 500 chunks")
        vectors = []
        for start in range(0, len(chunks), 12):
            vectors.extend(self.embed(chunks[start:start + 12]))
        document_id = str(uuid4())
        with self.connect() as db:
            db.execute("INSERT INTO family_rag_documents(id,member_id,scope,title,filename,content_type) VALUES(%s,%s,%s,%s,%s,%s)",
                       (document_id, member_id, scope, Path(filename).stem, filename, content_type))
            for index, (content, vector) in enumerate(zip(chunks, vectors)):
                db.execute("INSERT INTO family_rag_chunks(id,document_id,member_id,scope,chunk_index,content,embedding) VALUES(%s,%s,%s,%s,%s,%s,%s::vector)",
                           (str(uuid4()), document_id, member_id, scope, index, content, self.vector_literal(vector)))
        return {"id": document_id, "filename": filename, "chunks": len(chunks), "scope": scope}

    def search(self, member_id: str, query: str, limit: int = 5) -> list[RetrievedChunk]:
        query_vector = self.vector_literal(self.embed([query])[0])
        with self.connect() as db:
            rows = db.execute("""SELECT c.document_id,d.title,c.content,c.chunk_index,
                    1 - (c.embedding <=> %s::vector) AS score
                FROM family_rag_chunks c JOIN family_rag_documents d ON d.id=c.document_id
                WHERE c.member_id=%s OR c.scope='family'
                ORDER BY c.embedding <=> %s::vector LIMIT %s""", (query_vector, member_id, query_vector, limit)).fetchall()
        return [RetrievedChunk(row["document_id"], row["title"], row["content"], float(row["score"]), row["chunk_index"])
                for row in rows if float(row["score"]) >= 0.28]

    def list_documents(self, member_id: str) -> list[dict]:
        with self.connect() as db:
            return list(db.execute("""SELECT d.id,d.title,d.filename,d.scope,d.created_at,COUNT(c.id) AS chunks
                FROM family_rag_documents d LEFT JOIN family_rag_chunks c ON c.document_id=d.id
                WHERE d.member_id=%s OR d.scope='family' GROUP BY d.id ORDER BY d.created_at DESC""", (member_id,)).fetchall())

    def delete_document(self, member_id: str, document_id: str) -> bool:
        with self.connect() as db:
            result = db.execute("DELETE FROM family_rag_documents WHERE id=%s AND member_id=%s", (document_id, member_id))
        return result.rowcount == 1
