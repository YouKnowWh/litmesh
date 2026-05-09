# Changelog

## 2026-05-09

### LitMesh 结构上下文与概念候选修正
- **切分策略改为段落优先** — `section_splitter` 现在用章节标题维护 `heading_path`，实际入库为 `paragraph_group` 粒度的 `SectionBlock`，更适合作为抽取和上下文串联单位。
- **补全上下文 typed pointers** — v0.1 导入时将章节顺序写入 `section_next` 关系，将章节层级写入 `section_parent` 关系，避免只有字段、没有可遍历边。
- **新增传统 RAG fallback** — `HybridRetriever` 增加 `context_blocks` / `chunk_walk`，当 typed traversal 或结构化召回结果不足时，按段落命中并扩展前后文窗口，让 LLM 可顺序阅读语块作为保底上下文。
- **补全论证块溯源关系** — v0.2 抽取 Claim/Evidence/Limitation 后自动创建 `belongs_to` 指向原 `SectionBlock`，让后续 trace/audit traversal 能回到原上下文。
- **收紧概念候选抽取** — Claim/Evidence/Limitation prompt 改为输出 `concept_terms`，不再要求 LLM 直接生成正式 `ConceptKey`。
- **过滤照抄式关键词** — ConceptExtractor 增加短术语清洗、通用词过滤、长句拒绝和稳定 slug 生成，避免把原文句子或章节标题伪装成概念 key。
- **更新回归测试** — 原 v0.1 结构测试改为段落级 fixture，新增段落切分、`section_next` typed relation、论证块 `belongs_to` 原章节、段落 `chunk_walk` fallback、长句式概念候选拒绝测试，并同步中文 UI 文案断言。
- **修复测试 span id 碰撞** — v0.6-v0.7 E2E 测试的手工 `span_id` 现在带序号，避免不同 paper 随机生成相同前缀导致 SQLite 唯一约束失败。

### Bug 修复 (12:00-18:00)
- **修复 source_span FK 约束** — claim/evidence/limitation extractor 设置 `source_span_id=None`，避免指向未插入的 span
- **修复 section FK 约束** — `next_section_id` 在章节插入后再设置
- **修复 CJK PDF 文本提取** — 中文字符间空格合并、拆分行合并、`x_tolerance/y_tolerance` 调优
- **修复 min_section_chars 过滤** — 过短章节被过滤，全部被过滤时回退到整篇作为单章节
- **修复 prompt .format() 冲突** — 7 个 extractor 的 JSON 模板大括号改用 `.replace()`
- **导入改为异步** — v0.1 同步返回，v0.2 后台运行，避免 HTTP 超时
- **新增** `GET /papers/{id}/extraction-status` — 轮询抽取进度

### 多模型 LLM 配置
- **新增** `app/litmesh/extraction/llm_config.py` — 按角色独立配置 LLM 端点
  - 四个角色：`extraction`（抽取）、`review`（审核）、`compilation`（编译）、`default`（兜底）
  - 每个角色有独立环境变量 `LITMESH_{ROLE}_PROVIDER/MODEL/BASE_URL/API_KEY`
  - 未设置时自动回退到全局变量
- **新增** `MultiLLMClient` — 懒加载按角色获取 LLMClient
- **修改** `app/main.py` — 启动时加载多模型配置
- **修改** `app/litmesh/api/routes.py` — `create_app()` 接受 `llm_clients` 和 `embed_provider`
- **修改** `docker-compose.yml` — 添加所有角色 LLM 和 Embedding 环境变量

### 向量模型配置
- **新增** `app/litmesh/retrieval/embedding_providers.py` — Embedding 提供者抽象
  - 三种 provider：`openai_compatible`、`sentence_transformers`、`dummy`
  - 环境变量：`LITMESH_EMBED_PROVIDER/MODEL/BASE_URL/API_KEY/DIMENSION`
  - 默认使用 SiliconFlow `BAAI/bge-large-zh-v1.5` (1024维)
- **修改** `docker-compose.yml` — 配置 SiliconFlow 向量模型

### Prompt 模板修复
- **修复** `app/litmesh/ingestion/metadata_extractor.py` — `.format()` → `.replace()` 避免 JSON 大括号冲突
- **修复** `app/litmesh/extraction/claim_extractor.py` — 同上
- **修复** `app/litmesh/extraction/evidence_extractor.py` — 同上
- **修复** `app/litmesh/extraction/limitation_extractor.py` — 同上
- **修复** `app/litmesh/extraction/concept_extractor.py` — 同上
- **修复** `app/litmesh/extraction/relation_linker.py` — 同上
- **修复** `app/litmesh/registry/series_detector.py` — 同上

### PDF 解析改进
- **修改** `app/litmesh/ingestion/pdf_parser.py` — CJK 文本清理
  - 移除 CJK 字符间的多余空格
  - 合并被拆散的中文行
  - pdfplumber 容错参数调优（`x_tolerance=2, y_tolerance=2`）
  - pdfplumber 失败时 fallback 到 PyPDF2
- **修改** `app/litmesh/ingestion/section_splitter.py` — 过滤过短章节
  - 实施 `min_section_chars` 最小章节字符数过滤
  - 全部被过滤时回退到整篇作为单章节

### 导入流程改进
- **修改** `app/litmesh/api/routes.py` — `/papers/upload` v0.1 同步返回 + v0.2 后台线程跑（`run_in_executor`），避免 HTTP 超时
- **新增** `GET /graph-full` — 返回完整图数据（claims/evidence/limitations/concepts/relations/sections）
- **新增** `GET /graph-relations` — 返回类型化关系
- **新增** `GET /papers/{id}/extraction-status` — 轮询抽取进度

### UI 重设计
- **重写** `app/litmesh/api/ui.html` — 全中文界面
  - **标签页精简到 5 个**：论文（含导入）/ 系列图 / 桥接图 / 审核 / 统计
  - 导入合并到论文页顶部，可折叠
  - D3.js 力导向图展示系列图和桥接图
  - **方向箭头** — 每条边带 SVG marker 箭头，颜色对应关系类型
  - **上下文边 (context)** — 青色粗虚线箭头连接相邻章节的论点，保持原文阅读顺序
  - **触控板缩放** — `wheelDelta` 适配触控板双指捏合
  - 全局进度条（切换 tab 不丢失）
  - 审核页「批量通过」按钮
  - 暗色主题 + 渐变效果 + 毛玻璃 header
  - 图谱交互：滚轮/触控板缩放、拖拽平移、悬停详情
- **修改** `app/litmesh/api/routes.py` — `graph-full` 返回 sections 排序数据用于上下文边

### 基础设施
- **新增** `app/main.py` — ASGI 入口点
- **新增** `Dockerfile` — Python 3.11-slim 镜像
- **新增** `docker-compose.yml` — 端口 8001、数据卷、环境变量
- **新增** `.dockerignore`
- **新增** `CHANGELOG.md`
- **修改** `requirements.txt` — 添加 `lancedb>=0.17.0`、`python-multipart>=0.0.9`

### 其他 AI 修改 (parallel session)
- **修改** `app/litmesh/retrieval/hybrid_retriever.py` — 新增 `retrieve_context_blocks()` 段落块回退检索
  - `include_context_blocks` / `context_window` 参数
  - 当结构遍历命中太少时，回退到段落级分块检索
- **修改** `app/litmesh/tests/test_integration.py` — 段落级 section 测试 (`PARAGRAPH_GROUP`)
- **修改** `app/litmesh/tests/test_v05_retrieval.py` — 新增 `test_context_block_fallback_walks_paragraph_neighbors`
- **修改** `app/litmesh/tests/test_v06_v07_e2e.py` — 小调整
- **修改** `app/litmesh/tests/test_v08_ui.py` — 小调整

### 最新修复 (18:00+)
- **修复触控板缩放** — 重写 wheel handler，手动计算 `zoomTransform` 而非依赖 D3 默认 delta 转换
- **扩展上下文边** — 从仅连接相邻章节改为窗口 2 内的章节都连接，边数从 18 增至约 40+
- **UI 合并** — 导入区合并到论文页顶部 (`<details>` 折叠)
