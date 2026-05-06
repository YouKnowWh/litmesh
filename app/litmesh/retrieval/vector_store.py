"""
Vector store wrapper using LanceDB for embedding-based retrieval.

Design: LanceDB is the recall index. SQLite is the source of truth.
The vector store can be rebuilt from SQLite at any time.
"""

from pathlib import Path
from typing import Optional


class VectorStore:
    """LanceDB-backed vector index for claims, evidence, and sections."""

    def __init__(self, db_path: str = "./data/vector_store"):
        self.db_path = Path(db_path)
        self._table = None
        self._db = None

    def _get_db(self):
        if self._db is None:
            import lancedb
            self.db_path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.db_path))
        return self._db

    def _get_table(self, table_name: str = "claims"):
        db = self._get_db()
        try:
            return db.open_table(table_name)
        except Exception:
            return None

    def index_claims(
        self,
        items: list[dict],
        embedding_fn,
        table_name: str = "claims",
    ):
        """Index claims with embeddings.

        Args:
            items: list of dicts with at least: id, text, paper_id, graph_id
            embedding_fn: callable(text) -> list[float]
        """
        import pyarrow as pa

        records = []
        for item in items:
            vec = embedding_fn(item["text"])
            records.append({
                "id": item["id"],
                "text": item["text"],
                "paper_id": item.get("paper_id", ""),
                "graph_id": item.get("graph_id", ""),
                "claim_type": item.get("claim_type", ""),
                "vector": vec,
            })

        db = self._get_db()
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("paper_id", pa.string()),
            pa.field("graph_id", pa.string()),
            pa.field("claim_type", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), len(records[0]["vector"]) if records else 768)),
        ])

        db.drop_table(table_name, ignore_missing=True)
        self._table = db.create_table(table_name, records, schema=schema)

    def search(self, query_vec: list[float], top_k: int = 20,
               graph_ids: Optional[list[str]] = None,
               table_name: str = "claims") -> list[dict]:
        """Vector similarity search with optional graph scope filter."""
        table = self._get_table(table_name)
        if table is None:
            return []

        results = table.search(query_vec).limit(top_k).to_list()

        # Filter by graph scope if specified
        if graph_ids:
            results = [r for r in results if r.get("graph_id") in graph_ids]

        return results

    def delete_by_graph(self, graph_id: str, table_name: str = "claims"):
        """Delete all vectors for a graph (for rebuild)."""
        table = self._get_table(table_name)
        if table:
            table.delete(f"graph_id = '{graph_id}'")


class DummyEmbedder:
    """Dummy embedder for testing without a real embedding model."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        """Return a deterministic pseudo-embedding based on text hash."""
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        # Expand hash bytes to dim dimensions
        result = []
        for i in range(self.dim):
            byte_val = h[i % len(h)]
            result.append((byte_val / 255.0) * 2 - 1)  # Scale to [-1, 1]
        return result
