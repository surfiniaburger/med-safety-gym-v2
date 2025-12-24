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
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            parser.parse(invalid_json, ResponseFormat.JSON)
    
    def test_parse_json_missing_field(self, parser):
        """Test parsing JSON with missing required field"""
        incomplete_json = '{"analysis": "test", "proof": "test"}'  # Missing 'final'
        
        # Should now succeed with empty final
        result = parser.parse(incomplete_json, ResponseFormat.JSON)
        assert result.analysis == "test"
        assert result.proof == "test"
        assert result.final == ""
    
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
        # Others should be empty
        assert result.proof == ""
        assert result.final == ""
    
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

        # Should now succeed with empty final
        result = parser.parse(incomplete_custom, ResponseFormat.CUSTOM_TAGS)
        assert result.analysis == "test"
        assert result.proof == "test"
        assert result.final == ""
    
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
        """Test parsing empty response"""
        with pytest.raises(ValueError, match="cannot be empty"):
            parser.parse("", ResponseFormat.AUTO)
    
    def test_parse_whitespace_only(self, parser):
        """Test parsing whitespace-only response"""
        with pytest.raises(ValueError, match="cannot be empty"):
            parser.parse("   \n\t   ", ResponseFormat.AUTO)
    
    def test_parse_unsupported_format(self, parser):
        """Test parsing with unsupported format type"""
        # This would require modifying the enum, so we test the error path
        with pytest.raises(ValueError):
            parser.parse('{"test": "data"}', "invalid_format")
    
    def test_dipg_response_validation(self):
        """Test DIPGResponse validation"""
        # Valid response
        valid = DIPGResponse(
            analysis="test analysis",
            proof="test proof",
            final="test final"
        )
        assert valid.analysis == "test analysis"
        
        # Empty field should succeed now
        empty = DIPGResponse(
            analysis="",
            proof="",
            final=""
        )
        assert empty.analysis == ""     
        
        # Whitespace-only should be stripped to empty
        whitespace = DIPGResponse(analysis="   ", proof="test", final="test")
        assert whitespace.analysis == ""
    
    def test_parse_json_with_extra_fields(self, parser):
        """Test parsing JSON with extra fields (should be ignored)"""
        json_with_extra = '{"analysis": "test", "proof": "test", "final": "test", "extra": "ignored"}'
        
        result = parser.parse(json_with_extra, ResponseFormat.JSON)
        
        assert isinstance(result, DIPGResponse)
        assert result.analysis == "test"
        # Extra field should not be in the model
        assert not hasattr(result, 'extra')
