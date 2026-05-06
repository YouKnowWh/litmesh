# LitMesh — Concept-Centered Literature Knowledge Skill

将 PDF、论文、书籍转化为结构化认知上下文的编译器。不是普通 RAG。

## 核心区别

| | 普通 RAG | 知识图谱 | LitMesh |
|---|---|---|---|
| 核心对象 | chunk | entity-relation triple | concept-claim-evidence-limitation |
| 检索方式 | 相似度召回 | SPARQL / 图查询 | 类型化指针遍历 |
| 输出 | chunk 列表 | 三元组 | PromptPacket (结构化上下文) |

## 架构

```
PDF → PaperCard → SectionBlock → SourceSpan              (v0.1)
  → ClaimBlock → EvidenceBlock → LimitationBlock         (v0.2)
  → ConceptKey → ConceptNormalizer → GraphRelation       (v0.3)
  → TraversalPlan → TraversalExecutor → PromptPacket     (v0.4)
  → RetrievalGate → VectorStore + FTS5 + GraphExpand     (v0.5)
  → KnowledgeQueryEngine (query → packet → text)         (v0.6)
  → BridgeDetector (cross-graph same_as/analogous_to)    (v0.7)
  → Admin UI (/ui)                                       (v0.8)
```

## 核心概念

- **SeriesGraph**: 每个 PDF 独立成图，从不合并
- **SeriesGroup**: 索引层，记录哪些图属于同一系列
- **ConceptKey**: 带 namespace 的概念索引单位，不是普通标签
- **Claim → Evidence → Limitation**: 论证三角，不是扁平的 chunk
- **SourceSpan**: 每个 claim 必须锚定原文位置
- **Typed Pointer**: 节点间用 supports/constrains/contradicts/refines 连接，不是 related_to
- **PromptPacket**: 给 LLM 的结构化上下文，区分 claims/evidence/limitations/low-confidence

## 安装

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install datasketch
```

## 运行测试

```bash
python -m pytest app/litmesh/tests/ -q   # 127 tests
```

## 启动 API

```bash
uvicorn app.litmesh.api.routes:create_app --host 127.0.0.1 --port 8000
```

管理界面: http://127.0.0.1:8000/ui

## 项目结构

```
app/litmesh/
├── models/          # 15 Pydantic 模型
├── storage/         # SQLite + FTS5 + 图遍历查询
├── ingestion/       # PDF 导入 → PaperCard → SectionBlock
├── extraction/      # LLM 抽取 (Claim/Evidence/Limitation/Concept)
├── registry/        # 概念归一化 + 系列检测 + 桥接检测
├── traversal/       # 7 种类型化指针遍历模式
├── retrieval/       # 混合检索 (向量 + 全文 + 图扩展)
├── compiler/        # PromptPacket 编译器 + 查询引擎
├── review/          # 审核队列管理
├── api/             # FastAPI 路由 + 管理 UI
└── tests/           # 127 项集成测试
```

## License

MIT
