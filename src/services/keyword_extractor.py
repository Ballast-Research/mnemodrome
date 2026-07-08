"""Simple keyword extraction for context queries."""

import re


def extract_keywords(prompt: str | None, max_keywords: int = 3) -> str | None:
    """
    Extract meaningful keywords from prompt for context filtering.

    Simple implementation using basic NLP-free approach:
    - Filter out common stop words
    - Extract capitalized words (likely entities/names)
    - Return most significant keyword

    Future enhancement: Use spaCy or similar for better extraction.

    Args:
        prompt: User prompt text
        max_keywords: Maximum keywords to consider (uses first for now)

    Returns:
        Single keyword or None if no meaningful keywords found
    """
    if not prompt or not prompt.strip():
        return None

    # Common stop words to filter out
    STOP_WORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "just",
        "about",
        "up",
        "out",
        "into",
        "tell",
        "me",
    }

    # Clean and tokenize
    words = re.findall(r"\b\w+\b", prompt.lower())

    # Filter stop words and short words
    keywords = [word for word in words if word not in STOP_WORDS and len(word) > 3]

    if not keywords:
        return None

    # For now, return first meaningful keyword
    # Future: rank by TF-IDF, capitalization, or domain relevance
    return keywords[0]
