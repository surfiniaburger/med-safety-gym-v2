"""
Format parser for DIPG Safety Gym responses.

Supports multiple input formats (JSON, XML, YAML, custom tags) and normalizes
them to a common internal representation for evaluation.

The V3 hierarchical curriculum reward logic remains unchanged - this is purely
a normalization layer.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, ValidationError
import json
import yaml
import xml.etree.ElementTree as ET
import re
from enum import Enum


class ResponseFormat(str, Enum):
    """Supported response formats"""
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    CUSTOM_TAGS = "custom_tags"
    AUTO = "auto"


class DIPGResponse(BaseModel):
    """
    Normalized internal representation of a DIPG Safety response.
    
    This is format-agnostic - the same structure is used regardless of
    whether the input was JSON, XML, YAML, or custom tags.
    """
    analysis: str = Field(default="", description="Reasoning about the medical context")
    proof: str = Field(default="", description="Direct quote from context")
    final: str = Field(default="", description="Final answer to the question")
    
    @field_validator('analysis', 'proof', 'final')
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        if v is None:
            return ""
        return v.strip()


class FormatParser:
    """
    Parses DIPG Safety responses in multiple formats.
    
    Supports:
    - JSON: {"analysis": "...", "proof": "...", "final": "..."}
    - XML: <dipg_response><analysis>...</analysis>...</dipg_response>
    - YAML: analysis: ...\nproof: ...\nfinal: ...
    - Custom Tags: <|channel|>analysis<|message|>...<|end|>...
    """
    
    def __init__(self):
        # Regex pattern for custom tag format
        self.custom_tag_pattern = re.compile(
            r'<\|channel\|>(\w+)<\|message\|>(.*?)<\|end\|>',
            re.DOTALL
        )
    
    def parse(
        self,
        response: str,
        format_type: ResponseFormat = ResponseFormat.AUTO
    ) -> DIPGResponse:
        """
        Parse response in any supported format.
        
        Args:
            response: The LLM-generated response string
            format_type: Expected format (or AUTO to detect)
            
        Returns:
            Normalized DIPGResponse object
            
        Raises:
            ValueError: If format is invalid or required fields missing
        """
        if not response or not response.strip():
            raise ValueError("Response cannot be empty")
        
        if format_type == ResponseFormat.AUTO:
            format_type = self._detect_format(response)
        
        
        parser_map = {
            ResponseFormat.JSON: self._parse_json,
            ResponseFormat.XML: self._parse_xml,
            ResponseFormat.YAML: self._parse_yaml,
            ResponseFormat.CUSTOM_TAGS: self._parse_custom_tags,
        }
        parser_func = parser_map.get(format_type)
        if parser_func:
            return parser_func(response)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _detect_format(self, response: str) -> ResponseFormat:
        """Auto-detect the format of the response"""
        response_stripped = response.strip()
        
        
        # Check for Custom Tags (distinctive markers) - Prioritize this!
        if '<|channel|>' in response_stripped:
            return ResponseFormat.CUSTOM_TAGS

        # Check for JSON (starts with { or wrapped in markdown)
        if response_stripped.startswith('{') or '```json' in response_stripped.lower() or (response_stripped.startswith('```') and '{' in response_stripped):
            return ResponseFormat.JSON
        
        # Check for XML (starts with < and contains closing tags)
        # Relaxed check: Simply looking for plausible XML structure
        # We now check for closing tags anywhere, handling preamble text
        if '</' in response_stripped and '>' in response_stripped:
             # Basic heuristic: if it has tags, it's likely XML
             # This covers standard LLM output like <think>...</think>
             return ResponseFormat.XML
        
        # Fallback check for startup tags if closing tags are missing (rare but possible)
        # But we must ensure it's not actually custom tags (already checked above)
        if response_stripped.startswith('<') and '>' in response_stripped:
             return ResponseFormat.XML

        
        # Check for YAML (has key: value structure for required fields)
        if all(field in response_stripped for field in ['analysis:', 'proof:', 'final:']) or '```yaml' in response_stripped.lower():
            return ResponseFormat.YAML
        
        # Default to custom tags for backward compatibility
        return ResponseFormat.CUSTOM_TAGS
    
    def _parse_json(self, response: str) -> DIPGResponse:
        """Parse JSON format with robustness improvements"""
        cleaned_response = response.strip()
        
        # 1. Strip markdown code blocks (handles text before/after blocks)
        if '```' in cleaned_response:
            # Find content between first ``` and last ```
            # This handles cases like: "Here's the answer: ```json {...} ``` Hope this helps!"
            first_backtick = cleaned_response.find('```')
            last_backtick = cleaned_response.rfind('```')
            
            if first_backtick != -1 and last_backtick != -1 and first_backtick < last_backtick:
                # Extract content between the outermost backticks and strip the language specifier
                content_block = cleaned_response[first_backtick + 3 : last_backtick]
                cleaned_response = re.sub(r'^\w*\s*', '', content_block).strip()
            
        try:
            data = json.loads(cleaned_response)
            
            # 2. Normalize field aliases
            normalized_data = {}
            
            # Define aliases
            aliases = {
                'analysis': ['reasoning', 'thought', 'thoughts', 'explanation', 'analysis'],
                'proof': ['evidence', 'quote', 'reference', 'source', 'proof'],
                'final': ['answer', 'conclusion', 'result', 'final_answer', 'final']
            }
            
            # Map fields
            for target_field, source_fields in aliases.items():
                for source in source_fields:
                    if source in data:
                        normalized_data[target_field] = data[source]
                        break
                # If not found, default to empty string
                if target_field not in normalized_data:
                     normalized_data[target_field] = ""
            
            return DIPGResponse(**normalized_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except ValidationError as e:
            raise ValueError(f"JSON validation failed: {e}")
    
    def _parse_xml(self, response: str) -> DIPGResponse:
        """Parse XML-like format using Robust Regex (ignore strict XML rules)"""
        # We use regex to be resilient to malformed XML, attributes, or fragments
        cleaned_response = response.strip()

        def extract_content(tags: list[str]) -> str:
            # Try each tag alias
            for tag in tags:
                # Regex for <tag>CONTENT</tag> or <tag attr="...">CONTENT</tag>
                # Using dotall to capture newlines
                pattern = f"<{tag}(?:\\s+[^>]*)?>(.*?)</{tag}>"
                matches = re.findall(pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                if matches:
                    # Join all occurrences (e.g. multiple proofs)
                    # Use a space if it's single line, or newline if multiline? 
                    # Generally newline separation is safer for proofs.
                    return "\n".join(m.strip() for m in matches if m.strip())
            return ""

        # Map aliases
        analysis_text = extract_content(['analysis', 'think', 'reasoning', 'thought'])
        proof_text = extract_content(['proof', 'evidence', 'quote'])
        final_text = extract_content(['final', 'answer', 'conclusion', 'final_answer'])
        
        data = {
            "analysis": analysis_text,
            "proof": proof_text,
            "final": final_text
        }
        
        return DIPGResponse(**data)
    
    def _parse_yaml(self, response: str) -> DIPGResponse:
        """Parse YAML format"""
        try:
            data = yaml.safe_load(response.strip())
            if not isinstance(data, dict):
                raise ValueError("YAML must be a dictionary")
            return DIPGResponse(**data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        except ValidationError as e:
            raise ValueError(f"YAML validation failed: {e}")
    
    def _parse_custom_tags(self, response: str) -> DIPGResponse:
        """Parse custom tag format (backward compatibility)"""
        channels = {}
        
        # Extract all channels
        for match in self.custom_tag_pattern.finditer(response):
            channel_name = match.group(1)
            content = match.group(2).strip()
            channels[channel_name] = content
        
        # Map to expected fields
        data = {
            "analysis": channels.get("analysis", ""),
            "proof": channels.get("proof", ""),
            "final": channels.get("final", "")
        }
        
        try:
            return DIPGResponse(**data)
        except ValidationError as e:
            raise ValueError(f"Custom tag validation failed: {e}. Found channels: {list(channels.keys())}")
