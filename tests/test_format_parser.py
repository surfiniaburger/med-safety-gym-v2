"""
Unit tests for format_parser.py

Tests all supported formats (JSON, XML, YAML, custom tags) and auto-detection.
"""

import json
import pytest
from med_safety_gym.format_parser import FormatParser, ResponseFormat, DIPGResponse


class TestFormatParser:
    """Test suite for FormatParser"""
    
    @pytest.fixture
    def parser(self):
        """Create a FormatParser instance"""
        return FormatParser()
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        return {
            "analysis": "The sources present conflicting information about Drug A.",
            "proof": "Source A states 'Drug A shows efficacy.' Source B states 'Drug A shows toxicity.'",
            "final": "The provided sources present conflicting information."
        }
    
    # ==================================================================================
    # JSON Format Tests
    # ==================================================================================
    
    def test_parse_json_valid(self, parser, sample_data):
        """Test parsing valid JSON"""
        json_response = json.dumps(sample_data)
        
        result = parser.parse(json_response, ResponseFormat.JSON)
        
        assert isinstance(result, DIPGResponse)
        assert result.analysis == sample_data["analysis"]
        assert result.proof == sample_data["proof"]
        assert result.final == sample_data["final"]
    
    def test_parse_json_auto_detect(self, parser, sample_data):
        """Test auto-detection of JSON format"""
        json_response = '{"analysis": "test", "proof": "test", "final": "test"}'
        
        result = parser.parse(json_response, ResponseFormat.AUTO)
        
        assert isinstance(result, DIPGResponse)
        assert result.analysis == "test"
    
    def test_parse_json_invalid_syntax(self, parser):
        """Test parsing invalid JSON syntax"""
        invalid_json = '{"analysis": "test", "proof": "test"'  # Missing closing brace
        # Should now return FORMAT_ERROR instead of raising
        result = parser.parse(invalid_json, ResponseFormat.JSON)
        assert "FORMAT_ERROR" in result.final
    
    def test_parse_json_missing_field(self, parser):
        """Test parsing JSON with missing required field"""
        incomplete_json = '{"analysis": "test", "proof": "test"}'  # Missing 'final'
        # Should now succeed with error message in final
        result = parser.parse(incomplete_json, ResponseFormat.JSON)
        assert result.analysis == "test"
        assert result.proof == "test"
        assert "FORMAT_ERROR" in result.final
    
    def test_parse_json_empty_field(self, parser):
        """Test parsing JSON with empty field"""
        empty_field_json = '{"analysis": "test", "proof": "", "final": "test"}'
    
        # Should now succeed with empty proof
        result = parser.parse(empty_field_json, ResponseFormat.JSON)
        assert result.analysis == "test"
        assert result.proof == ""
        assert result.final == "test"
    
    # ==================================================================================
    # XML Format Tests
    # ==================================================================================
    
    def test_parse_xml_valid(self, parser, sample_data):
        """Test parsing valid XML"""
        xml_response = f'''<dipg_response>
            <analysis>{sample_data["analysis"]}</analysis>
            <proof>{sample_data["proof"]}</proof>
            <final>{sample_data["final"]}</final>
        </dipg_response>'''
        
        result = parser.parse(xml_response, ResponseFormat.XML)
        
        assert isinstance(result, DIPGResponse)
        assert result.analysis == sample_data["analysis"]
        assert result.proof == sample_data["proof"]
        assert result.final == sample_data["final"]
    
    def test_parse_xml_auto_detect(self, parser):
        """Test auto-detection of XML format"""
        xml_response = '<dipg_response><analysis>test</analysis><proof>test</proof><final>test</final></dipg_response>'
        
        result = parser.parse(xml_response, ResponseFormat.AUTO)
        
        assert isinstance(result, DIPGResponse)
        assert result.analysis == "test"
    
    def test_parse_xml_with_declaration(self, parser):
        """Test parsing XML with XML declaration"""
        xml_response = '''<?xml version="1.0" encoding="UTF-8"?>
        <dipg_response>
            <analysis>test</analysis>
            <proof>test</proof>
            <final>test</final>
        </dipg_response>'''
        
        result = parser.parse(xml_response, ResponseFormat.AUTO)
        
        assert isinstance(result, DIPGResponse)
    
    def test_parse_xml_invalid_syntax(self, parser):
        """Test parsing invalid XML syntax - Robust Parser should extract partials"""
        # Previously raised ValueError, now robust regex handles it
        invalid_xml = '<dipg_response><analysis>test</analysis>'  # Unclosed tags
        
        # Should NOT raise, but extract what it can
        result = parser.parse(invalid_xml, ResponseFormat.XML)
        
        assert isinstance(result, DIPGResponse)
        # It should extract analysis since the tag is technically closed </analysis> (regex pattern)
        assert result.analysis == "test"
        # Others should be empty (or error if final)
        assert result.proof == ""
        assert "FORMAT_ERROR" in result.final

    def test_parse_xml_with_placeholders_in_think_block(self, parser):
        """Tests that the parser correctly ignores placeholder tags inside a think block."""
        response_text = """
<think>
Some reasoning here.
The example structure shows:
<proof>
"[Exact quote from text]"
</proof>
<answer>
[Final Answer]
</answer>
More reasoning.
</think>

<proof>
"Real evidence from context"
</proof>
<answer>
Actual final answer.
</answer>
"""
        result = parser.parse(response_text, ResponseFormat.XML)
        
        assert result.final == "Actual final answer."
        assert result.proof == '"Real evidence from context"'

    def test_parse_xml_with_multiple_proofs(self, parser):
        """Tests that multiple proof tags are aggregated correctly."""
        response_text = """
<proof>
Evidence 1
</proof>
<answer>Partial answer</answer>
<proof>
Evidence 2
</proof>
<answer>
Final categorical answer.
</answer>
"""
        result = parser.parse(response_text, ResponseFormat.XML)
        
        # Should aggregate proofs
        assert "Evidence 1" in result.proof
        assert "Evidence 2" in result.proof
        # Should pick the LAST answer
        assert result.final == "Final categorical answer."

    def test_parse_xml_with_multiple_thought_blocks_stripping(self, parser):
        """Tests that all thought blocks are stripped before content extraction."""
        response_text = """
<think>Initial thoughts with <answer>Fake 1</answer></think>
Some text.
<thought>Correction with <answer>Fake 2</answer></thought>
<proof>Real Proof</proof>
<answer>Real Answer</answer>
"""
        result = parser.parse(response_text, ResponseFormat.XML)
        
        # Should pick the FIRST thought block for analysis
        assert "Initial thoughts" in result.analysis
        # Should NOT pick Fake 1 or Fake 2
        assert result.final == "Real Answer"
        assert result.proof == "Real Proof"

    def test_parse_xml_fallback_no_answer_tag(self, parser):
        """Tests fallback: extracting text after last closed tag when <answer> is missing."""
        response_text = """
<think>Thinking...</think>
<proof>Some proof</proof>
This is the final answer that should be captured by fallback.
"""
        result = parser.parse(response_text, ResponseFormat.XML)
        assert result.final == "This is the final answer that should be captured by fallback."
        assert result.proof == "Some proof"

    def test_parse_xml_malformed_tags(self, parser):
        """Tests resilience to malformed or unclosed tags."""
        response_text = """
<think>Incomplete thought...
<proof>Unclosed proof
<answer>But a valid answer</answer>
"""
        result = parser.parse(response_text, ResponseFormat.XML)
        # Should capture the valid answer
        assert result.final == "But a valid answer"
        # V4.6: Now supports unclosed tags for analysis
        assert result.analysis == "Incomplete thought..."
        assert result.proof == ""
    
    # ==================================================================================
    # YAML Format Tests
    # ==================================================================================
    
    def test_parse_yaml_valid(self, parser, sample_data):
        """Test parsing valid YAML"""
        yaml_response = f'''analysis: {sample_data["analysis"]}
proof: {sample_data["proof"]}
final: {sample_data["final"]}'''
        
        result = parser.parse(yaml_response, ResponseFormat.YAML)
        
        assert isinstance(result, DIPGResponse)
        assert result.analysis == sample_data["analysis"]
        assert result.proof == sample_data["proof"]
        assert result.final == sample_data["final"]
    
    def test_parse_yaml_auto_detect(self, parser):
        """Test auto-detection of YAML format"""
        yaml_response = '''analysis: test
proof: test
final: test'''
        
        result = parser.parse(yaml_response, ResponseFormat.AUTO)
        
        assert isinstance(result, DIPGResponse)
        assert result.analysis == "test"
    
    def test_parse_yaml_multiline(self, parser):
        """Test parsing YAML with multiline strings"""
        yaml_response = '''analysis: |
  This is a multiline
  analysis text
proof: |
  This is a multiline
  proof text
final: This is the final answer'''
        
        result = parser.parse(yaml_response, ResponseFormat.YAML)
        
        assert isinstance(result, DIPGResponse)
        assert "multiline" in result.analysis
    
    # ==================================================================================
    # Custom Tags Format Tests
    # ==================================================================================
    
    def test_parse_custom_tags_valid(self, parser, sample_data):
        """Test parsing valid custom tags format"""
        custom_response = f'''<|channel|>analysis<|message|>
{sample_data["analysis"]}
<|end|>
<|channel|>proof<|message|>
{sample_data["proof"]}
<|end|>
<|channel|>final<|message|>
{sample_data["final"]}
<|end|>'''
        
        result = parser.parse(custom_response, ResponseFormat.CUSTOM_TAGS)
        
        assert isinstance(result, DIPGResponse)
        assert result.analysis == sample_data["analysis"]
        assert result.proof == sample_data["proof"]
        assert result.final == sample_data["final"]
    
    def test_parse_custom_tags_auto_detect(self, parser):
        """Test auto-detection of custom tags format"""
        custom_response = '''<|channel|>analysis<|message|>
test
<|end|>
<|channel|>proof<|message|>
test
<|end|>
<|channel|>final<|message|>
test
<|end|>'''
        
        result = parser.parse(custom_response, ResponseFormat.AUTO)
        
        assert isinstance(result, DIPGResponse)
        assert result.analysis == "test"
    
    def test_parse_custom_tags_missing_channel(self, parser):
        """Test parsing custom tags with missing channel"""
        incomplete_custom = '''<|channel|>analysis<|message|>
test
<|end|>
<|channel|>proof<|message|>
test
<|end|>'''  # Missing 'final' channel

        # Should now succeed with error message in final
        result = parser.parse(incomplete_custom, ResponseFormat.CUSTOM_TAGS)
        assert result.analysis == "test"
        assert result.proof == "test"
        assert "FORMAT_ERROR" in result.final
    
    # ==================================================================================
    # Auto-Detection Tests
    # ==================================================================================
    
    def test_auto_detect_json(self, parser):
        """Test auto-detection correctly identifies JSON"""
        json_response = '{"analysis": "test", "proof": "test", "final": "test"}'
        detected_format = parser._detect_format(json_response)
        assert detected_format == ResponseFormat.JSON
    
    def test_auto_detect_xml(self, parser):
        """Test auto-detection correctly identifies XML"""
        xml_response = '<dipg_response><analysis>test</analysis></dipg_response>'
        detected_format = parser._detect_format(xml_response)
        assert detected_format == ResponseFormat.XML
    
    def test_auto_detect_yaml(self, parser):
        """Test auto-detection correctly identifies YAML"""
        yaml_response = 'analysis: test\nproof: test\nfinal: test'
        detected_format = parser._detect_format(yaml_response)
        assert detected_format == ResponseFormat.YAML
    
    def test_auto_detect_custom_tags(self, parser):
        """Test auto-detection correctly identifies custom tags"""
        custom_response = '<|channel|>analysis<|message|>test<|end|>'
        detected_format = parser._detect_format(custom_response)
        assert detected_format == ResponseFormat.CUSTOM_TAGS
    
    # ==================================================================================
    # Edge Cases and Error Handling
    # ==================================================================================
    
    def test_parse_empty_response(self, parser):
        """Test parsing empty response returns error message instead of raising."""
        result = parser.parse("", ResponseFormat.AUTO)
        assert "FORMAT_ERROR" in result.final

    def test_parse_whitespace_only(self, parser):
        """Test parsing whitespace-only response returns error message."""
        result = parser.parse("   \n\t   ", ResponseFormat.AUTO)
        assert "FORMAT_ERROR" in result.final
    
    def test_parse_unsupported_format(self, parser):
        """Test parsing with unsupported format type falls back to XML."""
        # Instead of ValueError, it should now fallback to XML parsing
        result = parser.parse('<answer>Fallback Work</answer>', "invalid_format")
        assert result.final == "Fallback Work"
    
    def test_dipg_response_validation(self):
        """Test DIPGResponse validation"""
        # Valid response
        valid = DIPGResponse(
            analysis="test analysis",
            proof="test proof",
            final="test final",
            original_response="test original"
        )
        assert valid.analysis == "test analysis"
        
        # Empty field should succeed now
        empty = DIPGResponse(
            analysis="",
            proof="",
            final="",
            original_response=""
        )
        assert empty.analysis == ""     
        
        # Whitespace-only should be stripped to empty
        whitespace = DIPGResponse(analysis="   ", proof="test", final="test", original_response="")
        assert whitespace.analysis == ""
    
    def test_parse_json_with_extra_fields(self, parser):
        """Test parsing JSON with extra fields (should be ignored)"""
        json_with_extra = '{"analysis": "test", "proof": "test", "final": "test", "extra": "ignored"}'
        
        result = parser.parse(json_with_extra, ResponseFormat.JSON)
        
        assert isinstance(result, DIPGResponse)
        assert result.analysis == "test"
        # Extra field should not be in the model
        assert not hasattr(result, 'extra')
