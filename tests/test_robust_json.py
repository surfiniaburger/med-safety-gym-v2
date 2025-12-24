import pytest
from med_safety_gym.format_parser import FormatParser, ResponseFormat, DIPGResponse

class TestRobustJSON:
    @pytest.fixture
    def parser(self):
        return FormatParser()

    def test_strip_markdown_json(self, parser):
        """Test stripping markdown code blocks from JSON."""
        response = """```json
        {
            "analysis": "Analysis",
            "proof": "Proof",
            "final": "Final"
        }
        ```"""
        parsed = parser.parse(response, format_type=ResponseFormat.JSON)
        assert parsed.analysis == "Analysis"
        assert parsed.proof == "Proof"
        assert parsed.final == "Final"

    def test_strip_markdown_text(self, parser):
        """Test stripping markdown code blocks with just 'text' or no language."""
        response = """```
        {
            "analysis": "Analysis",
            "proof": "Proof",
            "final": "Final"
        }
        ```"""
        parsed = parser.parse(response, format_type=ResponseFormat.JSON)
        assert parsed.analysis == "Analysis"

    def test_strip_markdown_with_surrounding_text(self, parser):
        """Test stripping markdown when there's text before/after the code block."""
        response = """Here's my analysis:
        ```json
        {
            "analysis": "Analysis",
            "proof": "Proof",
            "final": "Final"
        }
        ```
        Hope this helps!"""
        parsed = parser.parse(response, format_type=ResponseFormat.JSON)
        assert parsed.analysis == "Analysis"
        assert parsed.proof == "Proof"
        assert parsed.final == "Final"

    def test_field_aliases(self, parser):
        """Test that field aliases are correctly mapped."""
        response = """{
            "reasoning": "My reasoning",
            "evidence": "My evidence",
            "answer": "My answer"
        }"""
        parsed = parser.parse(response, format_type=ResponseFormat.JSON)
        assert parsed.analysis == "My reasoning"
        assert parsed.proof == "My evidence"
        assert parsed.final == "My answer"

    def test_mixed_aliases(self, parser):
        """Test mixing standard names and aliases."""
        response = """{
            "analysis": "Standard analysis",
            "quote": "Aliased proof",
            "final_answer": "Aliased final"
        }"""
        parsed = parser.parse(response, format_type=ResponseFormat.JSON)
        assert parsed.analysis == "Standard analysis"
        assert parsed.proof == "Aliased proof"
        assert parsed.final == "Aliased final"

    def test_partial_json(self, parser):
        """Test parsing JSON with missing fields (should not raise error)."""
        response = """{
            "final": "Just the answer"
        }"""
        parsed = parser.parse(response, format_type=ResponseFormat.JSON)
        assert parsed.final == "Just the answer"
        assert parsed.analysis == ""  # Default to empty
        assert parsed.proof == ""     # Default to empty

    def test_auto_detect_markdown_json(self, parser):
        """Test auto-detection works even with markdown blocks."""
        response = """```json
        {
            "analysis": "...",
            "proof": "...",
            "final": "..."
        }
        ```"""
        # Should detect as JSON and parse correctly
        parsed = parser.parse(response, format_type=ResponseFormat.AUTO)
        assert parsed.analysis == "..."
