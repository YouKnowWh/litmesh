# Changelog

## 2026-05-10

### 实时进度与 SSE
- **新增** `app/litmesh/ingestion/progress.py` — 线程安全 `ProgressTracker`，pipeline 线程写入，SSE 端点读取
- **新增** `GET /tasks/{task_id}/events` — SSE 端点，推送实时 pipeline 进度（解析/元数据/分段/抽取各阶段 + 百分比）
- **前端进度卡片** — 上传后论文卡片立即出现（脉冲边框+进度条），刷新不丢失（占位 paper 已落库）
- **Pipeline 全阶段进度** — `run_v0_1` 解析/元数据/建图/章节/系列、`run_v0_2` 合并提取 + 并行窗口均发射进度事件
- **PDF 解析页级进度** — PyMuPDF/pdfplumber 适配器每 3-5 页回传一次 `page N/M`

### 管线加速
- **三合一提取** (`combined_extractor.py`) — claims + evidence + limitations 合并为每个 SectionBlock 一次 LLM 调用，从 ~3N 降到 ~N
- **并行提取** — `ThreadPoolExecutor(max_workers=3)` 并发处理章节提取，总耗时约 1/3
- **短 prompt + 低 max_tokens** — 输入 2000 字符 / 输出 1024 tokens，单次调用 5-10s
- **LLM 分段并发** — `LLMPageSegmenter` 用 `ThreadPoolExecutor(max_workers=4)` 并行处理页面窗口
- **Boundary-first 分段** — LLM 只返回 start_quote/end_quote 定位，程序从原文切出 text，减少输出 token
- **Segment max_tokens 降低** — 从 4096 降到 1200
- **LLM 调用日志** — provider 层记录 model、prompt_len、resp_len、time、tokens_in/out

### 分段质量提升
- **PDF 结构解析外包优先** — 新增 Markdown parser 适配层，`parser=auto` 优先尝试 `mineru_api → external_markdown/sidecar → pdfplumber`
- **远端 MinerU API 接入** — 新增 `parser=mineru_api`，通过 `LITMESH_MINERU_API_URL` 上传 PDF，接收 `text/markdown` 或 `{"markdown": "..."}`，避免本机安装/运行重型解析器
- **台式机 MinerU API 对接** — `RemoteMarkdownAdapter` 现兼容 `POST /file_parse` 的 `files=@pdf`、`backend=pipeline`、`return_md=true`，并优先读取返回 JSON 中的 `results.*.md_content`
- **远端优先回归修正** — `parser=auto` 的测试现明确区分“已配置远端 MinerU API 时优先走 `mineru_api`”与“未配置时回退到 `external_markdown`”，避免旧测试继续误报 `pdfplumber`/sidecar 优先
- **Markdown sidecar 接入** — 支持 `parser=markdown|external_markdown|mineru_markdown|marker_markdown`，可读取 PDF 同名 `.md` 或 `data/parsed_markdown/<stem>.md`
- **MinerU sidecar 实测** — 本机旧环境位于 `/Users/alex/Projects/PDF拆分/.venvbrew311/bin/magic-pdf`，需设置 `MINERU_TOOLS_CONFIG_JSON=/Users/alex/Projects/PDF拆分/magic-pdf.json`；89 页小书 CPU 解析约 3 分钟，可产出 Markdown sidecar
- **Markdown heading 顺序修复** — MinerU/Markdown 无真实页码时按元素顺序推进 heading，不再用 page=1 的 outline 区间覆盖所有段落
- **LLM 断章继续收紧** — `LLMPageSegmenter` 默认窗口扩大到 `4` 页、重叠提升到 `2` 页，`segment_max_tokens` 提高到 `2000`，并移除只命中 `start_quote` 时的 `2000` 字硬截断
- **目录优先结构识别** — 新增 `TOCExtractor`，pdfplumber 抽页后先解析目录并建立 `ParsedDocument.outline`
- **TOC 驱动 heading_path** — `split_parsed_document` 在存在可用目录时忽略 LLM/ParsedElement heading，用目录区间分配章节路径
- **目录/前言结构块保留** — 只有文本确实呈现目录/前言/编写说明特征时才保留为对应 SectionBlock；v0.2 抽取跳过这些保留块
- **目录对齐审计** — 记录 toc_entry_count、toc_page_count、toc_alignment_confidence、toc_printed_page_offset、toc_unaligned_entries
- **显示标题清理** — `display_title` 改为段落首句摘要，不再混入 `P001/C01-Sxx` 结构标签
- **年份前缀保护** — `section_splitter` 的前置页码清洗不再误删 `1949 年`、`2007 年 1 月`、`21 世纪` 这类正文开头数字
- **Parser 优先级修正** — auto 顺序收敛为外包 Markdown 优先，中文教材优先走成熟解析器输出，再 fallback 到完整页级文本 + LLM/规则分段
- **Parser source-of-truth 收敛** — `parser=auto` 不再让 PyMuPDF 自动兜底；PyMuPDF 只保留显式 `parser=pymupdf_blocks` 诊断入口
- **PyMuPDF 严格禁用门禁** — PyMuPDF 现在也读取 TOC 指标；若元素数明显低于页数/目录规模，标记 `pymupdf_too_sparse_vs_*`
- **Reranker 依赖入镜像** — Docker 现安装 `transformers`，并从 PyTorch CPU wheel 源安装 `torch==2.7.0+cpu`，repair 层不再因为缺依赖而退化成“只记候选、不做打分”，同时避免误拉整套 CUDA 依赖
- **dry_run 不再拉模型** — `RepairExecutor` 在默认 `dry_run` 观察模式下只记录候选，不再触发 reranker 初始化/下载，避免首次导入时卡住前端
- **chapter_context 图接口兼容旧库** — 修复 `/graph-view?mode=chapter_context` 中将 `sqlite3.Row` 误当 dict 使用导致的 500；同时在数据库启动兼容迁移里补齐 `section_blocks.toc_anchor_id / toc_anchor_title`，旧库无需手工改表即可进入章节图模式
- **Markdown 目录恢复加强** — `markdown_adapter` 现在能识别“目录 heading 被装饰性 heading 打断后才出现真实目录段”的教材场景；同时在无 TOC block 的 fallback 下，只把 `structural/toc` 级 heading 放入 outline，不再把 `问题探讨/讨论/本节聚焦` 一股脑当目录结构
- **无页码目录项与脏 TOC 降噪** — `toc_extractor` 现支持解析 `第1章 ...` / `第1节 ...` 这类无页码目录行；但若正文里找不到对应 heading，则不会把这类残缺目录项写进 outline，避免它们错误抢占前置材料和章节锚点
- **chapter_context 单书图自动走 TOC 骨架** — 仅传 `graph_id` 且图内只有一本书时，`/graph-view?mode=chapter_context` 会自动推断 `paper_id` 并优先使用 `outline_nodes`；同时 `chapter_index<=0` 的前言/目录/出版信息不会再被强行提升成伪章节
- **Boundary-first bug 修复** — `_extract_text` 在 `end_quote` 缺失时不再触发 `orig_start` 未绑定错误
- **Containment dedup** — 检测长段包含子段，保留长的拒收短的，计数入 `QualityReport.containment_duplicate_count`
- **Role 分流** — LLM prompt 扩展类型枚举：body/heading/sidebar/activity/exercise/caption/table_text/toc/front_matter/header_footer/page_number/uncertain
- **SectionBlock 不混入非正文** — `split_parsed_document` 过滤 sidebar/activity/exercise
- **页码清除** — `_strip_page_number()` 正则去掉段首的 P144、第144页、独立数字等
- **ElementType 扩展** — 新增 SIDEBAR/ACTIVITY/EXERCISE
- **ParsedElement 新增 role 字段**

### 审计与调试
- **Parse Audit JSONL** — `logs/parse_audit/{paper_id}.jsonl`，记录 window_result/segment_rejected/segment_repaired/window_fallback/quality_gate_done
- **新 API** — `GET /papers/{id}/parse-audit`、`GET /papers/{id}/segments`
- **Pipeline 阶段日志** — v0.1/v0.2 每阶段耗时、PDF 解析耗时、LLM 窗口耗时
- **QualityReport 扩展** — containment_duplicate_count、sidebar/activity/exercise_segment_count、cross_window_merge_count
- **质量门禁收紧** — alignment_fail_ratio>0.15、reject_ratio>0.2、containment_duplicates、paragraph_count<page_count*2 均触发 needs_structure_review

### 论文删除功能
- **新增** `DELETE /papers/{paper_id}` — 级联删除论文及所有关联数据（claims/evidence/limitations/sections/extraction_runs/graph 等）
- **前端删除按钮** — 每篇论文卡片右侧红色"删除"按钮，带确认对话框

### 修复
- **DeepSeek API key 空值回退** — `_env()` 和 `_getenv()` 将空字符串当缺失处理，`DEEPSEEK_API_KEY` 能正确回退
- **upload_paper Pydantic 校验** — year 字段修复为 Optional[int]，corpus 自动创建
- **claim_run 变量名** — 合并提取后引用修正为 ext_run
- **Prompt 大括号转义** — 新分段 prompt 的 JSON 示例用 `{{}}` 转义

### 已知问题（待修复）

1. **PyMuPDF 对中文课本段落提取极少**（必修一 146 页仅 8 段） — 已加严格门禁
   - 原因：PyMuPDF block mode 对中文排版不敏感
   - 当前处理：`parser=auto` 不再调用 PyMuPDF；PyMuPDF 只作为显式诊断 parser，且若明显低于页数/TOC 规模，会记录 `pymupdf_too_sparse_vs_pages` / `pymupdf_too_sparse_vs_toc`

2. **`_extract_text` 未绑定变量 bug**
   - 文件：`llm_page_segmenter.py:345`
   - `orig_start` 在 `si >= 0` 分支引用但未被赋值（当 `si >= 0 and ei < 0` 时触发）
   - 此 bug 导致 pdfplumber 解析器回退失败（`UnboundLocalError`），进而导致 PyMuPDF 被当作唯一可用解析器

3. **目录内容混入正文**
   - 原因：(a) PyMuPDF 适配器不识别目录；(b) LLM prompt 对目录特征描述不够具体
   - 修复方向：修复 bug #2 让 pdfplumber+LLM 正常工作后，LLM 可按点线/页码配对识别目录

4. **章节标题含 "P001" 式页码**
   - 原因：`_structure_label()` 生成的是 `P001` 格式标签（不是真实页码），当正文前几段是目录时标签无意义
   - 修复方向：`display_title` 用 `_first_sentence` 已 strip 页码，但 `structural_label` 仍显示段落序号；可改为仅显示章节号格式 "1.2.3"

5. **Parser 优先级对中文不友好**
   - 当前：`docling → pymupdf_blocks → pdfplumber`
   - 问题：docling 未安装 → PyMuPDF 产生极少段落 → pdfplumber 回退被 bug #2 阻断
   - 修复方向：修复 bug #2 后调整优先级为 `pdfplumber → pymupdf_blocks`（中文优先），或加入快速质量判断（元素数 < 页数 → 自动降级）

## 2026-05-09

### PDF 读取与 LLM 分段主路径落地
- **修正 DeepSeek 协议路径** — 默认 LLM provider 改为 `openai_compatible`，默认 base URL 改为 `https://api.deepseek.com/v1`；Anthropic provider 只保留给真正 Claude/Anthropic endpoint 使用。
- **新增 segment LLM role** — `llm_config.py` 增加 `SEGMENT` 角色，`MultiLLMClient.segment` 可独立配置 PDF 分段模型，Docker 增加 `LITMESH_SEGMENT_*` 环境变量。
- **新增** `app/litmesh/ingestion/llm_page_segmenter.py` — 以 3 页窗口/1 页重叠让 LLM 主导段落边界、跨页合并、toc/front_matter/heading/paragraph 分类。
- **来源校验入库门禁** — LLM segment 必须通过窗口原文 substring/source quote/fuzzy alignment；无法对齐的 segment 被拒绝并进入质量报告，不生成 `SectionBlock`。
- **窗口失败可降级** — LLM 窗口失败时只对该窗口回退 `PageTextSegmenter`，整本导入不中断。
- **重叠窗口去重** — 按 normalized text hash、页码范围和文本相似度移除重复段落，避免 overlap 导致重复 `SectionBlock`。
- **扩展质量审计** — `QualityReport` 增加 `segmenter_name`、LLM window/fail/alignment/rejected/duplicate/cross-page/front-matter/toc/low-confidence 指标，并继续完整 JSON 落库。
- **API 增加 segmenter 参数** — `/papers/import` 与 `/papers/upload` 支持 `segmenter=auto|rule|llm`，`parse-quality` 返回 segmenter 指标。
- **Docling 维持可选 benchmark** — 主 `requirements.txt` 不引入 Docling/MinerU/Marker，避免 Docker 默认拉取 Torch/CUDA 大依赖。
- **文档更新** — README/CLAUDE 记录 Docker 优先、DeepSeek OpenAI-compatible 配置、PDF 分段主流程与旧污染 graph 重新导入约定。
- **测试通过** — Docker 内 `157 passed`。

### PDF 分段重构：外部解析器优先 (深夜，进行中)
- **新增** `app/litmesh/ingestion/parsed_document.py` — 统一中间格式
  - `ParsedDocument` / `ParsedElement` / `OutlineItem` / `QualityReport`
  - Element 类型：`title | heading | paragraph | list_item | table | figure | caption | footer | header | page_number | toc | unknown`
- **新增** `app/litmesh/ingestion/parsers/` — parser 适配器层
  - `__init__.py` — `create_parser("auto")` 工厂，按优先级选最优
  - `pymupdf_adapter.py` — PyMuPDF block 级提取，按布局和文本特征分类 paragraph/heading，提取 PDF 内嵌 TOC
  - `pdfplumber_adapter.py` — pdfplumber + PageTextSegmenter 二次分段
  - `docling_adapter.py` — Docling 可选适配器；不进入主依赖，避免默认安装拉取 Torch/CUDA 巨型依赖
- **设计决策**：LitMesh 不再自己判目录、标题、段落边界，由外部解析器产出结构后 LitMesh 只做知识编译
- **pipeline 已切换 parser adapter** — v0.1 通过 `parse_document()` 解析 PDF，`auto` 会按质量从 Docling/PyMuPDF/pdfplumber 逐级回退
- **SectionBlock 从 ParsedElement 生成** — heading 只更新上下文，paragraph/list/table/caption 才生成段落块；段落记录 `parser_name/parser_element_id/parser_confidence`
- **新增质量审计** — `parse_quality_reports` 落库，`GET /papers/{id}/parse-quality` 查看最新解析质量，旧库启动时自动补兼容列
- **新增 outline API** — `GET /papers/{id}/outline` 从 `heading_path` 重建轻量目录
- **新增** `app/litmesh/ingestion/page_segmenter.py` — pdfplumber 二次结构化
  - 清洗封面碎字、页眉页脚、页码、目录点线（前导+尾随）
  - 识别 front matter（目录到正文第一章前的页面）
  - 按自然段重建：空行分割 + CJK 断行合并
- **必修二实测**：pdfplumber+segmenter → 489 段落 (原 138 页级块)、0 点线污染、正文从 p5 开始、`needs_structure_review=False`
- **必修一实测**：525 段落
- **测试通过** — 150 passed

### LitMesh 结构上下文与概念候选修正
- **修复 UI 脚本卡死** — 移除 `ui.html` 中残留的半截 `function viewPaperGraph` 声明，避免浏览器因 JavaScript 语法错误停止执行整个管理后台脚本。
- **修复图谱入口参数传递** — `loadPanel()` 现在会把 `graphId/paperId` 传给系列图面板，论文列表中的“查看图谱”会打开对应论文图谱。
- **加固 D3 图谱渲染** — 上下文边只连接当前已渲染节点，避免 claim 被裁剪后 D3 force link 引用不存在节点导致图谱页报错或假死。
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

### v0.9 遍历索引层 (2026-05-09 晚间)
- **新增 schema** — `node_index`, `block_concepts`, `claim_evidence_links`, `claim_limitation_links` 四个索引表
- **新增复合索引** — `idx_relations_graph_source_type`, `idx_relations_graph_target_type`
- **写入路径同步** — `insert_claim/evidence/limitation/section/concept` 自动写 `node_index`
- **批量节点解析** — `batch_resolve_nodes()` 一次 SQL 查多个节点类型
- **TraversalExecutor 重写**
  - `_resolve_node_fast()` 用 `node_index` 做 O(1) 类型查找，代替原来的 5 表顺序探测
  - BFS 改为 priority queue（`heapq`），按 depth + traversal_cost + importance + confidence 排序
  - 修复 `confidence < require_source_span` 的 bool/float 比较 bug
  - 强制 `graph_scope` 过滤，越界节点直接跳过
- **段落 fallback 强化** — `fallback_reason` 字段 + `max_context_blocks` 参数
- **PromptPacket 扩展** — 新增 `FallbackContextBlock` 模型 + `context_block_policy` 字段 (`IGNORE/APPEND_AS_RAW/INLINE_CITED`)
- **11 项新测试** — schema 表存在、node_index 写入、batch_resolve、block_concepts 链接、graph_scope 过滤、source_span 门控、fallback reason

### PDF 结构识别与段落索引 (2026-05-09 深夜)
- **PDF 清洗重写** — 保留换行边界，只合并 CJK 字符间的空格/tab，不再合并换行
- **噪声行过滤** — 独立页码、比例残片（`1 2`、`11::11`）、单位残片（`100 nm`）、字符重复行
- **标题识别 format-first** — 以独占行/行长/字号/加粗/上下空白为主信号，显式编号为强信号，关键词降级
- **SectionBlock 新增稳定索引**
  - `chapter_index`, `section_index`, `block_index`, `global_order_index` — 1-based 稳定排序
  - `display_title` — 不可靠标题时用段落首句 fallback
  - `heading_confidence` — 标题置信度
  - `structure_status` — clean / needs_structure_review / reconstructed
- **段落链完整** — `prev_section_id` / `next_section_id` / `section_next` relation 全部写入
- **Schema 迁移** — `section_blocks` 表新增 7 列；旧数据库需要重建后重新导入
- **248 passed** — 所有测试通过

### 图生成与连通性修复 (2026-05-09 晚间)
- **新增 `GET /graph-view`** — 轻量子图接口，支持 `paragraph/argument/mixed` 三种模式，默认 limit=300
- **新增 `POST /graphs/{id}/repair-connectivity`** — 幂等修复：自动补全 section_next, belongs_to, mentions, evidence→claim supports, limitation→claim constrains 边，缺失概念自动创建为 candidate
- **新增 `GET /graphs/{id}/connectivity-report`** — 图健康度诊断（缺失概念数、无归属块数）
- **前端图改为段落链布局** — 段落节点按文档顺序纵向排列，claim 挂载到所属 paragraph 旁，section_next 用虚线主链连接。不再使用全局 force simulation
- **/graph-full 保留但前端不再使用** — 保留给调试/导出

### 最新修复 (18:00+)
- **修复触控板缩放** — 重写 wheel handler，手动计算 `zoomTransform` 而非依赖 D3 默认 delta 转换
- **扩展上下文边** — 按章节页码排序所有论点，相邻全部连上（382 条 context 边）
- **UI 合并** — 导入区合并到论文页顶部 (`<details>` 折叠)
- **测试通过** — 148 passed, 0 failed
