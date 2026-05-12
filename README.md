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
PDF → outsourced Markdown parser / pdfplumber fallback → SectionBlock (v0.1)
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
docker compose build
```

主 Docker 依赖不安装 Docling/MinerU/Marker；它们只作为手动 benchmark，避免默认拉取 Torch/CUDA 等大型依赖。

## PDF 读取流程

默认流程：

```text
PDF → MinerU API / Markdown sidecar / external command（优先）
→ Markdown heading + paragraph normalize
→ 若外包不可用则 fallback 到 pdfplumber 完整抽页
→ TOCExtractor 先解析目录并定位正文页
→ LLMPageSegmenter 以 3 页窗口/1 页重叠恢复自然段
→ 程序做 source alignment、overlap dedup、质量审计
→ ParsedDocument → SectionBlock
```

`parser=auto` 会先尝试外包结构解析：`mineru_api → external_markdown/sidecar → pdfplumber`。PyMuPDF block 不再进入 auto fallback，只保留给显式 `parser=pymupdf_blocks` 的诊断场景。章节标题优先来自外部 Markdown heading 或目录/outline；只有结构解析不可用时，才降级到 pdfplumber + LLM/rule 分段。`segmenter=auto` 在配置了 segment LLM key 时使用 LLM 分段，否则使用 rule fallback。可显式传 `parser=mineru_api|markdown|external_markdown|mineru_markdown|marker_markdown|pdfplumber`。

远端 MinerU API 建议约定为：

```text
POST $LITMESH_MINERU_API_URL
multipart/form-data: files=<pdf>
backend=pipeline
return_md=true
返回 JSON: {"results":{"document.pdf":{"md_content":"..."}}}
也兼容直接返回 text/markdown，或 JSON: {"markdown": "..."}
```

当前台式机 MinerU 部署可直接这样配置：

```bash
LITMESH_MINERU_API_URL=https://api.82736541.xyz/file_parse
LITMESH_MINERU_API_BACKEND=pipeline
LITMESH_MINERU_API_RETURN_MD=true
```

也可以把外部工具输出的 Markdown 放到 PDF 同目录同名 `.md`，或 `data/parsed_markdown/<pdf-stem>.md`，LitMesh 会优先使用它。

当文本中确实存在目录、前言或编写说明时，它们会作为保留结构块入库，标题固定为 `目录`、`前言` 或 `编写说明`；它们不参与正文章节链和 v0.2 Claim/Evidence/Limitation 抽取。

DeepSeek 统一走 OpenAI-compatible 协议：

```bash
DEEPSEEK_API_KEY=...
LITMESH_LLM_PROVIDER=openai_compatible
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat

LITMESH_SEGMENT_PROVIDER=openai_compatible
LITMESH_SEGMENT_BASE_URL=https://api.deepseek.com/v1
LITMESH_SEGMENT_MODEL=deepseek-chat
```

## 运行测试

```bash
docker compose run --rm -e PYTHONPATH=/app litmesh pytest app/litmesh/tests/ -q   # 157 tests
```

## 启动 API

```bash
docker compose up litmesh
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
└── tests/           # 157 项集成测试
```

## License

MIT
