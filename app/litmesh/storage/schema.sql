-- LitMesh SQLite Schema v0.1-v0.2
-- Principle: SQLite is the source of truth. Vector DB is a recall index.

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ============================================================
-- v0.1: Structure import (corpus, graph, paper, section, source_span)
-- ============================================================

CREATE TABLE IF NOT EXISTS corpora (
    corpus_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    corpus_type TEXT NOT NULL DEFAULT 'paper_collection',
    domain TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    source_items TEXT NOT NULL DEFAULT '[]',          -- JSON array
    default_graph_id TEXT,
    integration_policy TEXT NOT NULL DEFAULT 'bridge_review',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS series_graphs (
    graph_id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL,
    name TEXT NOT NULL,
    graph_type TEXT NOT NULL DEFAULT 'paper_collection',
    domain TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    concept_namespace TEXT NOT NULL DEFAULT '',
    merge_policy TEXT NOT NULL DEFAULT 'review_required',
    cross_graph_policy TEXT NOT NULL DEFAULT 'strict',
    status TEXT NOT NULL DEFAULT 'building',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (corpus_id) REFERENCES corpora(corpus_id)
);

-- Index-layer: groups isolated graphs into series (never merges underlying graphs)
CREATE TABLE IF NOT EXISTS series_groups (
    group_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    graph_ids TEXT NOT NULL DEFAULT '[]',              -- JSON array of graph_ids
    domain TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_series_groups_domain ON series_groups(domain);

CREATE TABLE IF NOT EXISTS paper_cards (
    paper_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    title TEXT NOT NULL,
    authors TEXT NOT NULL DEFAULT '[]',               -- JSON array
    year INTEGER,
    source_file TEXT NOT NULL,
    abstract TEXT NOT NULL DEFAULT '',
    abstract_summary TEXT NOT NULL DEFAULT '',
    keywords TEXT NOT NULL DEFAULT '[]',              -- JSON array
    research_type TEXT NOT NULL DEFAULT 'other',
    main_framework TEXT NOT NULL DEFAULT '',
    domain_keys TEXT NOT NULL DEFAULT '[]',           -- JSON array
    raw_text_hash TEXT NOT NULL DEFAULT '',
    page_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (graph_id) REFERENCES series_graphs(graph_id)
);

CREATE TABLE IF NOT EXISTS section_blocks (
    section_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    heading TEXT NOT NULL DEFAULT '',
    heading_path TEXT NOT NULL DEFAULT '[]',          -- JSON array
    heading_level TEXT NOT NULL DEFAULT 'section',
    raw_text TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    page_start INTEGER,
    page_end INTEGER,
    concept_keys TEXT NOT NULL DEFAULT '[]',          -- JSON array
    parent_section_id TEXT,
    prev_section_id TEXT,
    next_section_id TEXT,
    content_hash TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (paper_id) REFERENCES paper_cards(paper_id),
    FOREIGN KEY (graph_id) REFERENCES series_graphs(graph_id),
    FOREIGN KEY (parent_section_id) REFERENCES section_blocks(section_id),
    FOREIGN KEY (prev_section_id) REFERENCES section_blocks(section_id),
    FOREIGN KEY (next_section_id) REFERENCES section_blocks(section_id)
);

CREATE TABLE IF NOT EXISTS source_spans (
    span_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    section_id TEXT,
    span_type TEXT NOT NULL DEFAULT 'paragraph',
    source_text TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    line_start INTEGER,
    line_end INTEGER,
    pdf_bbox TEXT,
    content_hash TEXT NOT NULL DEFAULT '',
    normalized_text TEXT,
    verified INTEGER NOT NULL DEFAULT 0,
    verified_by TEXT,
    verified_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (paper_id) REFERENCES paper_cards(paper_id),
    FOREIGN KEY (section_id) REFERENCES section_blocks(section_id)
);

-- ============================================================
-- v0.2: Argument extraction (claim, evidence, limitation)
-- ============================================================

CREATE TABLE IF NOT EXISTS claim_blocks (
    claim_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    section_id TEXT,
    claim_text TEXT NOT NULL,
    normalized_claim TEXT NOT NULL DEFAULT '',
    claim_type TEXT NOT NULL DEFAULT 'theoretical',
    concept_keys TEXT NOT NULL DEFAULT '[]',          -- JSON array
    evidence_refs TEXT NOT NULL DEFAULT '[]',         -- JSON array of evidence_ids
    limitation_refs TEXT NOT NULL DEFAULT '[]',       -- JSON array of limitation_ids
    extraction_confidence REAL NOT NULL DEFAULT 0.0,
    claim_confidence REAL NOT NULL DEFAULT 0.0,
    importance TEXT NOT NULL DEFAULT 'supporting',
    status TEXT NOT NULL DEFAULT 'candidate',
    source_span_id TEXT,
    extraction_run_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (paper_id) REFERENCES paper_cards(paper_id),
    FOREIGN KEY (graph_id) REFERENCES series_graphs(graph_id),
    FOREIGN KEY (section_id) REFERENCES section_blocks(section_id),
    FOREIGN KEY (source_span_id) REFERENCES source_spans(span_id),
    FOREIGN KEY (extraction_run_id) REFERENCES extraction_runs(run_id)
);

CREATE TABLE IF NOT EXISTS evidence_blocks (
    evidence_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    section_id TEXT,
    supports_claim_ids TEXT NOT NULL DEFAULT '[]',    -- JSON array
    evidence_text TEXT NOT NULL,
    evidence_type TEXT NOT NULL DEFAULT 'other',
    strength TEXT NOT NULL DEFAULT 'unassessed',
    concept_keys TEXT NOT NULL DEFAULT '[]',          -- JSON array
    source_span_id TEXT,
    extraction_run_id TEXT,
    status TEXT NOT NULL DEFAULT 'candidate',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (paper_id) REFERENCES paper_cards(paper_id),
    FOREIGN KEY (graph_id) REFERENCES series_graphs(graph_id),
    FOREIGN KEY (section_id) REFERENCES section_blocks(section_id),
    FOREIGN KEY (source_span_id) REFERENCES source_spans(span_id),
    FOREIGN KEY (extraction_run_id) REFERENCES extraction_runs(run_id)
);

CREATE TABLE IF NOT EXISTS limitation_blocks (
    limitation_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    section_id TEXT,
    limitation_text TEXT NOT NULL,
    affected_claim_ids TEXT NOT NULL DEFAULT '[]',    -- JSON array
    risk_type TEXT NOT NULL DEFAULT 'other',
    severity TEXT NOT NULL DEFAULT 'unassessed',
    concept_keys TEXT NOT NULL DEFAULT '[]',          -- JSON array
    source_span_id TEXT,
    extraction_run_id TEXT,
    status TEXT NOT NULL DEFAULT 'candidate',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (paper_id) REFERENCES paper_cards(paper_id),
    FOREIGN KEY (graph_id) REFERENCES series_graphs(graph_id),
    FOREIGN KEY (section_id) REFERENCES section_blocks(section_id),
    FOREIGN KEY (source_span_id) REFERENCES source_spans(span_id),
    FOREIGN KEY (extraction_run_id) REFERENCES extraction_runs(run_id)
);

-- ============================================================
-- Concepts and relations
-- ============================================================

CREATE TABLE IF NOT EXISTS concept_keys (
    concept_key TEXT NOT NULL,                        -- e.g., 'framework:CPE_3DF'
    graph_id TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'concept',
    label_zh TEXT NOT NULL DEFAULT '',
    label_en TEXT NOT NULL DEFAULT '',
    definition TEXT NOT NULL DEFAULT '',
    aliases TEXT NOT NULL DEFAULT '[]',               -- JSON array
    parent_keys TEXT NOT NULL DEFAULT '[]',           -- JSON array
    child_keys TEXT NOT NULL DEFAULT '[]',            -- JSON array
    related_keys TEXT NOT NULL DEFAULT '[]',          -- JSON array
    do_not_merge_with TEXT NOT NULL DEFAULT '[]',     -- JSON array
    status TEXT NOT NULL DEFAULT 'candidate',
    review_status TEXT NOT NULL DEFAULT 'pending',
    merge_policy TEXT NOT NULL DEFAULT 'strict',
    extraction_run_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (concept_key, graph_id),
    FOREIGN KEY (graph_id) REFERENCES series_graphs(graph_id)
);

CREATE TABLE IF NOT EXISTS graph_relations (
    relation_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    target_type TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.8,
    importance REAL NOT NULL DEFAULT 0.5,
    traversal_cost REAL NOT NULL DEFAULT 1.0,
    extraction_run_id TEXT,
    evidence_json TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (graph_id) REFERENCES series_graphs(graph_id)
);

CREATE TABLE IF NOT EXISTS bridge_relations (
    bridge_id TEXT PRIMARY KEY,
    source_graph_id TEXT NOT NULL,
    target_graph_id TEXT NOT NULL,
    source_key TEXT NOT NULL,
    target_key TEXT NOT NULL,
    bridge_type TEXT NOT NULL,
    bridge_confidence REAL NOT NULL DEFAULT 0.5,
    evidence_json TEXT NOT NULL DEFAULT '',
    warning TEXT NOT NULL DEFAULT '',
    review_status TEXT NOT NULL DEFAULT 'candidate',
    extraction_run_id TEXT,
    traversal_cost REAL NOT NULL DEFAULT 5.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (source_graph_id) REFERENCES series_graphs(graph_id),
    FOREIGN KEY (target_graph_id) REFERENCES series_graphs(graph_id)
);

-- ============================================================
-- Audit: extraction runs and review inbox
-- ============================================================

CREATE TABLE IF NOT EXISTS extraction_runs (
    run_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    graph_id TEXT NOT NULL,
    target TEXT NOT NULL DEFAULT 'all',
    status TEXT NOT NULL DEFAULT 'running',
    model TEXT NOT NULL DEFAULT 'deepseek-v4-pro[1m]',
    prompt_version TEXT NOT NULL DEFAULT 'v0.2',
    prompt_template TEXT NOT NULL DEFAULT '',
    section_ids TEXT NOT NULL DEFAULT '[]',           -- JSON array
    input_token_count INTEGER NOT NULL DEFAULT 0,
    items_produced INTEGER NOT NULL DEFAULT 0,
    items_accepted INTEGER NOT NULL DEFAULT 0,
    items_rejected INTEGER NOT NULL DEFAULT 0,
    output_token_count INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    completed_at TEXT,
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    rolled_back_at TEXT,
    rolled_back_by TEXT,
    error_message TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (paper_id) REFERENCES paper_cards(paper_id),
    FOREIGN KEY (graph_id) REFERENCES series_graphs(graph_id)
);

CREATE TABLE IF NOT EXISTS extraction_run_items (
    item_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    raw_llm_output TEXT NOT NULL DEFAULT '',
    extraction_confidence REAL NOT NULL DEFAULT 0.0,
    accepted INTEGER NOT NULL DEFAULT 0,
    reviewer_notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (run_id) REFERENCES extraction_runs(run_id)
);

CREATE TABLE IF NOT EXISTS review_inbox (
    inbox_id TEXT PRIMARY KEY,
    inbox_type TEXT NOT NULL,                          -- extraction, concept, bridge, conflict
    item_id TEXT NOT NULL,                             -- FK to the candidate object
    item_type TEXT NOT NULL,                           -- claim, evidence, limitation, concept, bridge_relation
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    source_text TEXT NOT NULL DEFAULT '',
    extraction_confidence REAL NOT NULL DEFAULT 0.0,
    priority TEXT NOT NULL DEFAULT 'medium',
    decision TEXT NOT NULL DEFAULT 'pending',
    decided_by TEXT,
    decided_at TEXT,
    decision_notes TEXT NOT NULL DEFAULT '',
    merge_target_id TEXT,
    extraction_run_id TEXT,
    graph_id TEXT,
    paper_id TEXT,
    suggested_actions TEXT NOT NULL DEFAULT '[]',      -- JSON array
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ============================================================
-- Traversal audit
-- ============================================================

CREATE TABLE IF NOT EXISTS traversal_traces (
    trace_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    plan_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ============================================================
-- FTS5: Full-text search for sections, claims, and papers
-- ============================================================

CREATE VIRTUAL TABLE IF NOT EXISTS section_fts USING fts5(
    heading,
    raw_text,
    summary,
    content='section_blocks',
    content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS claim_fts USING fts5(
    claim_text,
    normalized_claim,
    content='claim_blocks',
    content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS paper_fts USING fts5(
    title,
    abstract,
    keywords,
    content='paper_cards',
    content_rowid='rowid'
);

-- ============================================================
-- Indexes for traversal performance
-- ============================================================

-- Source spans by paper
CREATE INDEX IF NOT EXISTS idx_spans_paper ON source_spans(paper_id);
CREATE INDEX IF NOT EXISTS idx_spans_section ON source_spans(section_id);

-- Sections by paper and navigation
CREATE INDEX IF NOT EXISTS idx_sections_paper ON section_blocks(paper_id);
CREATE INDEX IF NOT EXISTS idx_sections_parent ON section_blocks(parent_section_id);

-- Claims by paper, status, type
CREATE INDEX IF NOT EXISTS idx_claims_paper ON claim_blocks(paper_id);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claim_blocks(status);
CREATE INDEX IF NOT EXISTS idx_claims_type ON claim_blocks(claim_type);
CREATE INDEX IF NOT EXISTS idx_claims_graph ON claim_blocks(graph_id);

-- Evidence by paper, status
CREATE INDEX IF NOT EXISTS idx_evidence_paper ON evidence_blocks(paper_id);
CREATE INDEX IF NOT EXISTS idx_evidence_status ON evidence_blocks(status);

-- Limitations by paper, status
CREATE INDEX IF NOT EXISTS idx_limitations_paper ON limitation_blocks(paper_id);
CREATE INDEX IF NOT EXISTS idx_limitations_status ON limitation_blocks(status);

-- Concepts by graph and status
CREATE INDEX IF NOT EXISTS idx_concepts_graph ON concept_keys(graph_id);
CREATE INDEX IF NOT EXISTS idx_concepts_status ON concept_keys(status);

-- Graph relations for traversal (composite)
CREATE INDEX IF NOT EXISTS idx_relations_source ON graph_relations(graph_id, source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON graph_relations(graph_id, target_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON graph_relations(graph_id, relation_type);

-- Bridge relations for cross-graph traversal
CREATE INDEX IF NOT EXISTS idx_bridge_source ON bridge_relations(source_graph_id);
CREATE INDEX IF NOT EXISTS idx_bridge_target ON bridge_relations(target_graph_id);
CREATE INDEX IF NOT EXISTS idx_bridge_type ON bridge_relations(bridge_type);

-- Extraction runs by paper
CREATE INDEX IF NOT EXISTS idx_extraction_runs_paper ON extraction_runs(paper_id);

-- Review inbox
CREATE INDEX IF NOT EXISTS idx_inbox_type ON review_inbox(inbox_type);
CREATE INDEX IF NOT EXISTS idx_inbox_decision ON review_inbox(decision);
CREATE INDEX IF NOT EXISTS idx_inbox_paper ON review_inbox(paper_id);

-- ============================================================
-- Triggers for auto-updating updated_at
-- ============================================================

CREATE TRIGGER IF NOT EXISTS trg_corpora_updated AFTER UPDATE ON corpora
BEGIN UPDATE corpora SET updated_at = datetime('now') WHERE corpus_id = NEW.corpus_id; END;

CREATE TRIGGER IF NOT EXISTS trg_graphs_updated AFTER UPDATE ON series_graphs
BEGIN UPDATE series_graphs SET updated_at = datetime('now') WHERE graph_id = NEW.graph_id; END;

CREATE TRIGGER IF NOT EXISTS trg_papers_updated AFTER UPDATE ON paper_cards
BEGIN UPDATE paper_cards SET updated_at = datetime('now') WHERE paper_id = NEW.paper_id; END;

CREATE TRIGGER IF NOT EXISTS trg_sections_updated AFTER UPDATE ON section_blocks
BEGIN UPDATE section_blocks SET updated_at = datetime('now') WHERE section_id = NEW.section_id; END;

CREATE TRIGGER IF NOT EXISTS trg_spans_updated AFTER UPDATE ON source_spans
BEGIN UPDATE source_spans SET updated_at = datetime('now') WHERE span_id = NEW.span_id; END;

CREATE TRIGGER IF NOT EXISTS trg_claims_updated AFTER UPDATE ON claim_blocks
BEGIN UPDATE claim_blocks SET updated_at = datetime('now') WHERE claim_id = NEW.claim_id; END;

CREATE TRIGGER IF NOT EXISTS trg_evidence_updated AFTER UPDATE ON evidence_blocks
BEGIN UPDATE evidence_blocks SET updated_at = datetime('now') WHERE evidence_id = NEW.evidence_id; END;

CREATE TRIGGER IF NOT EXISTS trg_limitations_updated AFTER UPDATE ON limitation_blocks
BEGIN UPDATE limitation_blocks SET updated_at = datetime('now') WHERE limitation_id = NEW.limitation_id; END;

CREATE TRIGGER IF NOT EXISTS trg_concepts_updated AFTER UPDATE ON concept_keys
BEGIN UPDATE concept_keys SET updated_at = datetime('now') WHERE concept_key = NEW.concept_key; END;

CREATE TRIGGER IF NOT EXISTS trg_relations_updated AFTER UPDATE ON graph_relations
BEGIN UPDATE graph_relations SET updated_at = datetime('now') WHERE relation_id = NEW.relation_id; END;

CREATE TRIGGER IF NOT EXISTS trg_bridge_updated AFTER UPDATE ON bridge_relations
BEGIN UPDATE bridge_relations SET updated_at = datetime('now') WHERE bridge_id = NEW.bridge_id; END;

CREATE TRIGGER IF NOT EXISTS trg_inbox_updated AFTER UPDATE ON review_inbox
BEGIN UPDATE review_inbox SET updated_at = datetime('now') WHERE inbox_id = NEW.inbox_id; END;
