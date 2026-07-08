"""Unit tests for keyword extraction module (src/services/keyword_extractor.py).

Tests the keyword extraction logic used for context filtering.
"""

from src.services.keyword_extractor import extract_keywords


class TestExtractKeywords:
    """Tests for the extract_keywords function."""

    def test_extract_keywords_none_prompt(self):
        """Test that None prompt returns None."""
        result = extract_keywords(None)
        assert result is None

    def test_extract_keywords_empty_string(self):
        """Test that empty string prompt returns None."""
        result = extract_keywords("")
        assert result is None

    def test_extract_keywords_whitespace_only(self):
        """Test that whitespace-only prompt returns None."""
        result = extract_keywords("   \n\t  ")
        assert result is None

    def test_extract_keywords_all_stop_words(self):
        """Test that prompt with only stop words returns None."""
        result = extract_keywords("the and or but with from at")
        assert result is None

    def test_extract_keywords_single_short_word(self):
        """Test that words shorter than 4 chars are filtered out."""
        result = extract_keywords("cat dog fox the")
        assert result is None

    def test_extract_keywords_simple_valid_keyword(self):
        """Test extraction of simple valid keyword."""
        result = extract_keywords("Mnemodrome project")
        assert result == "mnemodrome"  # lowercase

    def test_extract_keywords_capitalized_word(self):
        """Test that capitalized words are recognized as keywords."""
        result = extract_keywords("Tell me about Carlos")
        assert result == "carlos"

    def test_extract_keywords_mixed_case(self):
        """Test keyword extraction from mixed case text."""
        result = extract_keywords("How is Neo4j performance?")
        assert result == "neo4j"

    def test_extract_keywords_multiple_valid_words(self):
        """Test that first valid keyword is returned from multiple valid options."""
        result = extract_keywords("Design pattern implementation for software")
        # First valid keyword that's not a stop word and > 3 chars
        assert result in ["design", "pattern", "implementation", "software"]
        # Should be first one found
        assert result == "design"

    def test_extract_keywords_with_numbers(self):
        """Test keyword extraction with alphanumeric words."""
        result = extract_keywords("Phase4a development started")
        assert result == "phase4a"

    def test_extract_keywords_special_characters(self):
        """Test that special characters are handled correctly."""
        result = extract_keywords("Test-driven development approach")
        # Hyphen is non-word char, so "test" and "driven" are separate
        # "test" is only 4 chars (borderline), "driven" has 6
        assert result in ["test", "driven"]

    def test_extract_keywords_long_prompt(self):
        """Test extraction from a long, realistic prompt."""
        prompt = """I'm working on Mnemodrome, a sophisticated AI infrastructure project.
        I need help with Neo4j queries for entity deduplication. Can you explain the MERGE pattern?"""
        result = extract_keywords(prompt)
        # Should find one of the capitalized proper nouns/significant words
        assert result is not None
        assert len(result) > 3

    def test_extract_keywords_max_keywords_parameter(self):
        """Test that max_keywords parameter is accepted (future use)."""
        # Currently the function uses first keyword, but parameter is accepted
        result = extract_keywords("Mnemodrome Neo4j Python", max_keywords=5)
        assert result == "mnemodrome"

    def test_extract_keywords_single_letter_words(self):
        """Test extraction with single letter words and valid keywords."""
        result = extract_keywords("I am working on it")
        # "working" is > 3 chars and not a stop word
        assert result == "working"

    def test_extract_keywords_case_insensitive_stop_words(self):
        """Test that stop word filtering is case-insensitive."""
        result = extract_keywords("THE SOLUTION AND APPROACH")
        # "SOLUTION" and "APPROACH" are > 3 chars and not stop words
        assert result in ["solution", "approach"]

    def test_extract_keywords_numbers_only(self):
        """Test prompt with numbers as keywords."""
        result = extract_keywords("12345 6789")
        # Numbers are extracted as words (both > 3 chars)
        assert result == "12345"

    def test_extract_keywords_exact_boundary_length(self):
        """Test word exactly at 4-char boundary (should be included)."""
        result = extract_keywords("test word")
        # "test" is exactly 4 chars (should be included per len(word) > 3)
        # "word" is exactly 4 chars
        assert result == "test"

    def test_extract_keywords_less_than_4_chars_excluded(self):
        """Test that words with exactly 3 or fewer characters are excluded."""
        result = extract_keywords("cat and dog")
        # All words are <= 3 chars (plus stop words)
        assert result is None
