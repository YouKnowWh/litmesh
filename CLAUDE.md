# CLAUDE.md

## Project: LitMesh — Concept-Centered Literature Knowledge Skill

AcompaLLM 的知识库模块。核心目标：将 PDF/论文/书籍转化为结构化、可审查、可回溯、可修正的认知上下文。

## Architecture Principles

1. **Series-first, bridge-later** — 每个 PDF 独立成图，用 SeriesGroup 索引层分组，不合并底层图
2. **SQLite as source of truth** — 向量库只是召回索引，可以重建
3. **JSON as protocol, not database** — JSON/JSONL 用于中间产物和 PromptPacket
4. **LLM produces candidates, program controls writes** — LLM 产出候选，程序校验入库
5. **No source span, no active claim** — 没有 source_span 的 ClaimBlock 不能进入正式知识库
6. **Review before high-impact use** — 新概念、跨图桥接、冲突关系必须审核
7. **Pointer-directed traversal** — 用户问题 → TraversalPlan → 类型化指针遍历 → PromptPacket

## Tech Stack

- Python 3.11+, Pydantic v2, SQLite, FastAPI, LanceDB, datasketch
- LLM: Anthropic/OpenAI-compatible provider abstraction
- Tests: pytest (in-memory SQLite, no external deps needed for tests)

## Run Tests

```
. .venv/bin/activate
python -m pytest app/litmesh/tests/ -q
```

## Start API Server

```
. .venv/bin/activate
uvicorn app.litmesh.api.routes:app --host 127.0.0.1 --port 8000
```

UI at http://127.0.0.1:8000/ui

## Key Files

- `models/` — 15 Pydantic models (Claim, Evidence, Limitation, Concept, Relation, etc.)
- `storage/sqlite.py` — SQLite storage layer with FTS5 + graph traversal queries
- `ingestion/pipeline.py` — PDF import + extraction pipeline
- `extraction/` — LLM-based claim/evidence/limitation/concept extractors
- `registry/` — ConceptNormalizer, SeriesDetector (MinHash+LSH), BridgeDetector
- `traversal/` — 7-mode Typed Pointer Traversal (TraversalExecutor + presets)
- `retrieval/` — Hybrid retrieval (VectorStore + FTS5 + graph expansion)
- `compiler/` — KnowledgeQueryEngine + PromptPacketCompiler
- `api/` — FastAPI routes + admin UI
