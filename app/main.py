"""
LitMesh ASGI entry point.
Creates the FastAPI app with database, multi-model LLM clients, and embedding provider.
"""
from pathlib import Path

from app.litmesh.storage.sqlite import LitMeshDB
from app.litmesh.extraction.llm_config import load_all_endpoints, MultiLLMClient
from app.litmesh.retrieval.embedding_providers import load_embedding_endpoint
from app.litmesh.api.routes import create_app

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

db = LitMeshDB(str(DATA_DIR / "litmesh.db"))
db.connect()
db.init_schema()

# LLM: per-role model configs
endpoints = load_all_endpoints()
llm_clients = MultiLLMClient(endpoints)

# Embedding: separate config for vector search
embed_endpoint = load_embedding_endpoint("DEFAULT")
embed_provider = embed_endpoint.create_provider()

app = create_app(db, llm_clients=llm_clients, embed_provider=embed_provider)
