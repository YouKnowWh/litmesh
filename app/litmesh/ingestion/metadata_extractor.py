"""
LLM-based metadata extractor for PaperCard fields.

Extracts: title, authors, year, abstract, keywords, research_type, main_framework.

Principle 4: LLM produces candidates, program controls writes.
The extracted metadata goes through the PaperCard model validation before storage.
"""

import json
from typing import Optional

from ..models.paper import PaperCard, ResearchType


METADATA_EXTRACTION_PROMPT = """你是一个学术文献元数据提取器。请从以下论文文本中提取元数据。

请以 JSON 格式返回，不要输出任何其他内容：

{
  "title": "论文标题",
  "authors": ["作者1", "作者2"],
  "year": 2024,
  "abstract": "摘要原文（如果存在）",
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "research_type": "theoretical|empirical|case_study|review|practice_report|policy|textbook_chapter|other",
  "main_framework": "论文中主要使用的理论框架名称（如 PACADI、CPE-3DF 等），如果没有则留空"
}

判断 research_type 的标准：
- theoretical: 主要是理论分析或框架构建
- empirical: 包含实验、数据收集、统计分析
- case_study: 以一个或多个具体案例为主
- review: 文献综述
- practice_report: 教学实践报告/经验总结
- policy: 政策分析或建议
- textbook_chapter: 教材章节
- other: 无法判断

只返回 JSON，不要输出解释或其他文字。

论文文本（前 3000 字）：
{text}
"""


class MetadataExtractor:
    """Extracts PaperCard metadata using an LLM.

    Usage:
        extractor = MetadataExtractor(llm_client)
        paper_card = extractor.extract(full_text, source_file, graph_id)
    """

    def __init__(self, llm_client):
        """
        Args:
            llm_client: An object with a `complete(prompt: str) -> str` method.
                        Expected to be the litmesh.extraction.llm_client.LLMClient.
        """
        self.llm = llm_client

    def extract(self, full_text: str, source_file: str, graph_id: str) -> PaperCard:
        """Extract metadata from paper text.

        Only the first 3000 characters are sent to the LLM (title/abstract/keywords
        are always in the beginning of a paper).
        """
        text_sample = full_text[:3000]
        prompt = METADATA_EXTRACTION_PROMPT.format(text=text_sample)

        raw_response = self.llm.complete(prompt)
        metadata = self._parse_response(raw_response)

        return PaperCard(
            graph_id=graph_id,
            title=metadata.get("title", source_file),
            authors=metadata.get("authors", []),
            year=metadata.get("year"),
            source_file=source_file,
            abstract=metadata.get("abstract", ""),
            keywords=metadata.get("keywords", []),
            research_type=ResearchType(metadata.get("research_type", "other")),
            main_framework=metadata.get("main_framework", ""),
        )

    def _parse_response(self, raw: str) -> dict:
        """Parse LLM JSON response, with error tolerance."""
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            import re
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {}
