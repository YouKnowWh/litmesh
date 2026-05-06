"""
ConceptRouter: routes a user query to relevant ConceptKeys.

Given a user question, finds which concepts in the knowledge base
are most relevant. Uses FTS5 on concept labels/definitions first,
then graph expansion to find related concepts.

This determines the start_nodes for TraversalPlanner.
"""

import json


class ConceptRouter:
    """Routes user queries to relevant concepts for traversal start points."""

    def __init__(self, db):
        self.db = db

    def route(self, query: str, graph_scope: list[str], top_k: int = 5) -> list[dict]:
        """Find concepts relevant to the query.

        Returns list of concept dicts with relevance signals.
        """
        results = []
        # Strip punctuation and whitespace
        stripped = query.replace("？", "").replace("?", "").replace("，", "").replace(",", "").replace(" ", "")

        for graph_id in graph_scope:
            # Get all active concepts in this graph
            rows = self.db.conn.execute(
                "SELECT * FROM concept_keys WHERE graph_id = ? AND status = 'active'",
                (graph_id,)
            ).fetchall()

            for r in rows:
                d = dict(r)
                label = d.get("label_zh", "")
                aliases_raw = d.get("aliases", "[]")

                # Bidirectional match: label in query OR query in label
                if label and (label in stripped or stripped in label):
                    if d["concept_key"] not in [x["concept_key"] for x in results]:
                        d["relevance"] = 0.9 if label in stripped else 0.5
                        d["match_type"] = "label_match"
                        results.append(d)
                        continue

                # Check aliases
                aliases = json.loads(aliases_raw) if isinstance(aliases_raw, str) else aliases_raw
                for alias in aliases:
                    if alias and (alias in stripped or stripped in alias):
                        if d["concept_key"] not in [x["concept_key"] for x in results]:
                            d["relevance"] = 0.8
                            d["match_type"] = "alias_match"
                            results.append(d)
                            break

        # Sort by relevance and take top_k
        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        return results[:top_k]
