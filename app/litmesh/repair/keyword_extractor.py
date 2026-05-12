"""
Lightweight rule-based keyword extractor for Chinese textbook content.

Extracts 6-18 character keyword phrases from section raw_text
without external NLP dependencies (no jieba, no LLM).

Used to augment display titles for context headings like "问题探讨",
turning them into "问题探讨｜细胞证据与观察".
"""

from __future__ import annotations

import re
from collections import Counter


# ---- Stop words: grammatical particles ----
_STOP_WORDS: set[str] = {
    "的", "了", "是", "在", "和", "与", "等", "这", "那",
    "一个", "一种", "可以", "能够", "通过", "以及",
    "因此", "所以", "但是", "然而", "如果", "虽然",
    "因为", "或者", "并且", "而且", "不仅",
    "已经", "正在", "将", "会", "被", "把",
    "从", "对", "向", "以", "用", "为", "也", "就",
    "都", "还", "要", "能", "可", "所", "其",
    "更", "最", "很", "较", "不", "没", "有",
    "上", "下", "中", "里", "外", "内", "前", "后",
    "我们", "它们", "他们", "自己", "什么", "怎么",
}

# ---- Function words: textbook activity labels ----
_FUNCTION_WORDS: set[str] = {
    "问题", "讨论", "本节", "聚焦", "探究", "实践",
    "相关", "信息", "资料", "阅读", "分析",
    "思考", "练习", "应用", "检测", "拓展", "小结", "复习",
    "目的", "要求", "材料", "用具", "方法", "步骤", "结果",
    "结论", "注意", "提示", "说明", "简介", "背景",
}

# ---- Over-generic terms to filter out ----
_GENERIC_TERMS: set[str] = {
    "细胞", "分子", "结构", "过程", "作用", "功能",
    "组成", "变化", "物质", "生物", "实验", "研究",
    "方法", "结果", "发生", "形成", "存在", "进行",
    "具有", "主要", "不同", "重要", "基本", "特点",
    "单位", "生命", "活动", "系统", "环境", "条件",
    "内容", "部分", "方面", "情况", "关系", "问题",
}

# Split text into sentences
_SENTENCE_SPLIT = re.compile(r"[。！？!?\n；;]")

# Delimiters for phrase splitting
_PHRASE_DELIM = re.compile(r"[,，、；;：:。！？!?\s的在了和与等对向以用为是]+")

# Common verbs/words to filter from keyword phrases (also used as splitters)
_VERB_WORDS = [
    "看到", "可以", "能够", "通过", "进行", "具有",
    "揭示", "提出", "发现", "显示", "表明", "说明",
    "包括", "含有", "存在", "形成", "产生", "组成",
    "构成", "获得", "使得", "引起", "导致", "影响",
    "利用", "采用", "根据", "按照", "关于", "对于",
    "主要", "需要", "可能", "应该", "必须", "一定",
    "然后", "首先", "其次", "最后", "接着", "之后",
    "其中", "这个", "那个", "这些", "那些", "什么",
    "一种", "各种", "不同", "相同", "类似",
]
_VERB_FILTER: set[str] = set(_VERB_WORDS)

# Build delimiter regex including verb words
_VERB_PATTERN = "|".join(re.escape(w) for w in _VERB_WORDS)
_PHRASE_DELIM = re.compile(r"[,，、；;：:。！？!?\s的在了和与等对向以用为是]|" + _VERB_PATTERN)

# Filter: keep only terms with >= 2 Chinese characters
_HAS_CHINESE = re.compile(r"[一-鿿]{2,}")


def _split_phrases(text: str) -> list[str]:
    """Split Chinese text into meaningful phrases by common delimiters."""
    parts = _PHRASE_DELIM.split(text)
    phrases = []
    for p in parts:
        p = p.strip()
        # Keep only chunks with enough Chinese characters
        chinese_chars = "".join(re.findall(r"[一-鿿]", p))
        if len(chinese_chars) >= 3:
            phrases.append(p)
    return phrases


class KeywordExtractor:
    """Extract keyword phrases from Chinese textbook section text."""

    def __init__(self, max_len: int = 18, min_len: int = 6):
        self.max_len = max_len
        self.min_len = min_len

    def extract(
        self, raw_text: str, heading: str = "", role: str = ""
    ) -> str:
        """Extract a 6-18 character keyword summary.

        Args:
            raw_text: Full section text.
            heading: The section's heading (words in heading are excluded).
            role: HeadingRole value — only CONTEXT_HEADING gets forced augmentation.

        Returns:
            Keyword phrase, or empty string if extraction fails.
        """
        if not raw_text:
            return ""

        # 1. Take first few sentences
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(raw_text)[:5] if s.strip()]
        if not sentences:
            return ""

        target = "".join(sentences[:3])

        # 3. Split into phrases by delimiters (，、；：的 等)
        phrases = _split_phrases(target)
        if not phrases:
            return ""

        # 4. Filter and count
        heading_tokens = self._heading_tokens(heading)
        filtered = []
        for p in phrases:
            p = p.strip()
            if len(p) < 3 or len(p) > 10:
                continue
            if p in _STOP_WORDS or p in _FUNCTION_WORDS:
                continue
            if p in _GENERIC_TERMS:
                continue
            if p in _VERB_FILTER:
                continue
            if p in heading_tokens:
                continue
            # Skip if entirely contained in another phrase
            filtered.append(p)

        if not filtered:
            filtered = [p for p in phrases if len(p) >= 3 and p not in _STOP_WORDS and p not in _FUNCTION_WORDS]

        if not filtered:
            return ""

        # 5. Score: prefer longer, unique phrases
        counter = Counter(filtered)
        # Prefer phrases that appear once (more specific)
        unique = [p for p, c in counter.items() if c == 1]
        if len(unique) >= 2:
            filtered = unique
        else:
            filtered = [p for p, _ in counter.most_common()]

        # 6. Take top phrases, prefer longer ones
        filtered.sort(key=lambda p: -len(p))
        result = ""
        for p in filtered:
            if len(result) + len(p) <= self.max_len:
                if p not in result:
                    result += p
            if len(result) >= self.min_len:
                break

        if len(result) < self.min_len and filtered:
            return filtered[0][:self.max_len]

        return result[:self.max_len]

    @staticmethod
    def _heading_tokens(heading: str) -> set[str]:
        """Extract phrases from heading to exclude from keywords."""
        tokens: set[str] = set()
        for p in _split_phrases(heading):
            tokens.add(p)
        return tokens
