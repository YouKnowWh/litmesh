"""
LitMesh ASGI entry point.
Creates the FastAPI app with database, multi-model LLM clients, and embedding provider.
"""
import logging
import sys
from pathlib import Path

# ---- Logging Setup ----
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "litmesh.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("litmesh")
logger.info("LitMesh starting up")

from app.litmesh.storage.sqlite import LitMeshDB
from app.litmesh.extraction.llm_config import load_all_endpoints, MultiLLMClient
from app.litmesh.retrieval.embedding_providers import load_embedding_endpoint
from app.litmesh.config.settings import SettingsManager
from app.litmesh.api.routes import create_app

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

db = LitMeshDB(str(DATA_DIR / "litmesh.db"))
db.connect()
db.init_schema()
logger.info("Database initialized: %s", DATA_DIR / "litmesh.db")

# Settings: env vars + data/settings.json overlay
settings_mgr = SettingsManager()

# LLM: per-role model configs (may be overridden by settings.json)
endpoints = load_all_endpoints()
llm_clients = MultiLLMClient(endpoints)
logger.info("LLM configs: %s", list(endpoints.keys()))

# Embedding: separate config for vector search
embed_endpoint = load_embedding_endpoint("DEFAULT")
embed_provider = embed_endpoint.create_provider()
logger.info("Embedding: %s/%s (dim=%s)", embed_endpoint.provider, embed_endpoint.model, embed_endpoint.dimension)

app = create_app(db, llm_clients=llm_clients, embed_provider=embed_provider, settings_manager=settings_mgr)
logger.info("LitMesh ready")
