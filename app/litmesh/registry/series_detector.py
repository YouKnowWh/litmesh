"""
SeriesDetector: determines if a new paper belongs to an existing series.

Uses datasketch (MinHash + LSH) for fast rule-based title matching,
with LLM fallback for semantic matching when surface text differs.

Design:
- Each paper title is shingled into character n-grams (supports Chinese + English)
- MinHash signatures are stored per SeriesGroup
- LSH index queries candidate groups in O(1)
- Jaccard similarity threshold determines "same series"
- LLM handles differently-worded titles (e.g. "OSTEP Ch3" vs "Operating Systems: Three Easy Pieces — CPU Scheduling")
"""

import json
import re

from ..models.series_group import SeriesGroup


# --- Shingling ---

def _shingle(text: str, k: int = 3) -> set[str]:
    """Character n-gram shingling. Works for Chinese and English."""
    text = text.lower().strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    if len(text) < k:
        return {text}
    return {text[i:i + k] for i in range(len(text) - k + 1)}


# --- LLM prompt ---

SERIES_DETECTION_PROMPT = """You are a literature series classifier. Determine if a new paper belongs to the same series as an existing group.

A "series" means: same book, same textbook, same research project, same thematic collection.

New paper: {title} ({year})
Framework: {framework}
Keywords: {keywords}

Existing group: {group_name}
Papers in this group: {paper_titles}

Rate likelihood 0-100 that the new paper belongs to this group.
Return JSON only: {"score": 0-100, "reasoning": "one sentence"}
"""


class SeriesDetector:
    """Detects series membership using MinHash + LSH + LLM fallback."""

    def __init__(self, db, llm_client):
        self.db = db
        self.llm = llm_client

    def detect(self, paper_card, graph_id: str, domain: str = "") -> dict:
        """Detect if a paper belongs to an existing series.

        Pipeline:
        1. Build MinHash signature for the new paper title
        2. Query LSH index for candidate groups
        3. Jaccard verify candidates
        4. If no match, try LLM semantic matching
        5. If still no match, create new group
        """
        from datasketch import MinHash, MinHashLSH

        title = paper_card.title
        shingles = _shingle(title, k=3)
        if not shingles:
            return self._new_group(paper_card, graph_id, domain, "Empty title")

        # 1. Build MinHash for new paper
        m_new = MinHash(num_perm=128)
        for s in shingles:
            m_new.update(s.encode("utf-8"))

        # 2. Retrieve existing groups and build LSH index
        existing = self.db.list_series_groups(domain)
        if not existing:
            existing = self.db.list_series_groups()

        if not existing:
            return self._new_group(paper_card, graph_id, domain, "First paper in system")

        # Build LSH index + MinHashes for all existing groups
        lsh = MinHashLSH(threshold=0.3, num_perm=128)
        group_minhashes: dict[str, MinHash] = {}
        group_info: dict[str, dict] = {}

        for g in existing:
            # Build a representative MinHash from all paper titles in the group
            graph_ids = json.loads(g["graph_ids"]) if isinstance(g["graph_ids"], str) else g["graph_ids"]
            m_group = MinHash(num_perm=128)

            titles_in_group = []
            for gid in graph_ids:
                papers = self.db.list_papers(gid)
                for p in papers:
                    p_title = p.get("title", "")
                    titles_in_group.append(p_title)
                    for s in _shingle(p_title, k=3):
                        m_group.update(s.encode("utf-8"))

            group_minhashes[g["group_id"]] = m_group
            group_info[g["group_id"]] = {**g, "_titles": titles_in_group}
            lsh.insert(g["group_id"], m_group)

        # 3. Query LSH for candidates
        candidates = lsh.query(m_new)

        # 4. Jaccard verification
        best_match = None
        best_jaccard = 0.0

        for group_id in candidates:
            m_group = group_minhashes[group_id]
            jaccard = m_new.jaccard(m_group)
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_match = group_id

        if best_match and best_jaccard >= 0.25:
            info = group_info[best_match]
            return {
                "action": "add_to_existing",
                "group_id": best_match,
                "group_name": info["name"],
                "confidence": round(best_jaccard + 0.2, 2),  # Boost slightly
                "reasoning": f"MinHash Jaccard similarity {best_jaccard:.2%} with group '{info['name']}'",
            }

        # 5. LLM fallback: try semantic matching against top candidates
        if self.llm and candidates:
            for group_id in candidates[:3]:
                info = group_info[group_id]
                result = self._llm_match(paper_card, info)
                if result and result.get("score", 0) >= 60:
                    return {
                        "action": "add_to_existing",
                        "group_id": group_id,
                        "group_name": info["name"],
                        "confidence": result["score"] / 100,
                        "reasoning": result.get("reasoning", ""),
                    }

        # 6. No match — create new group
        return self._new_group(
            paper_card, graph_id, domain,
            f"No matching series found (best Jaccard: {best_jaccard:.2%})"
        )

    def _llm_match(self, paper_card, group_info: dict) -> dict | None:
        """LLM-based semantic matching for differently-worded titles."""
        prompt = SERIES_DETECTION_PROMPT.format(
            title=paper_card.title,
            year=paper_card.year or "",
            framework=paper_card.main_framework,
            keywords=", ".join(paper_card.keywords[:5]),
            group_name=group_info["name"],
            paper_titles=", ".join(group_info.get("_titles", [])[:5]),
        )
        try:
            raw = self.llm.complete(prompt, system="Output only JSON.", temperature=0.1)
            return _parse_json(raw)
        except Exception:
            return None

    def _new_group(self, paper_card, graph_id: str, domain: str, reasoning: str) -> dict:
        group_name = paper_card.main_framework or paper_card.title[:50]
        group = SeriesGroup(
            name=group_name,
            graph_ids=[graph_id],
            domain=domain or paper_card.main_framework,
            description=reasoning,
            confidence=1.0,
        )
        self.db.insert_series_group(group)
        return {
            "action": "new_group",
            "group_id": group.group_id,
            "group_name": group.name,
            "confidence": 1.0,
            "reasoning": reasoning,
        }


def _parse_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}
