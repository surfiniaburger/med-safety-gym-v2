"""
Format parser for handling various structured response formats from models.
"""
import re
import json
import yaml
from typing import Dict, Optional, Any

from .models import ParsedResponse, ResponseFormat

class FormatParser:
    """
    Parses a model's raw string response into a structured `ParsedResponse` object.
    
    It can handle different formats, like custom XML-like tags, JSON, and YAML,
    and is designed to be extensible.
    """

    def __init__(self):
        # Define tag aliases centrally for easier maintenance.
        self.tag_aliases = {
            "analysis": ["think", "analysis", "reasoning", "thought"],
            "proof": ["proof", "trace", "evidence", "quote"],
            "final": ["answer", "final", "conclusion", "result"],
        }
        
        # Pre-compile regex for efficiency, supporting multiple tag names for flexibility.
        self.tag_patterns = {
            key: re.compile(rf"<(?:{'|'.join(aliases)})>(.*?)</(?:{'|'.join(aliases)})>", re.DOTALL | re.IGNORECASE)
            for key, aliases in self.tag_aliases.items()
        }
        
        # Regex pattern for custom tag format (backward compatibility)
        self.custom_tag_pattern = re.compile(
            r'<\|channel\|>(\w+)<\|message\|>(.*?)<\|end\|>',
            re.DOTALL
        )
        
        # Build the fallback regex pattern for all closing tags
        # We exclude 'final' tags here because this regex is used when the <final> tag is MISSING.
        all_other_aliases = []
        for key, aliases in self.tag_aliases.items():
            if key != "final":
                all_other_aliases.extend(aliases)
        
        self.fallback_closing_tag_pattern = re.compile(
            rf"</(?:{'|'.join(all_other_aliases)})>", 
            re.IGNORECASE
        )

    def parse(
        self,
        response_text: str,
        format_type: ResponseFormat = ResponseFormat.AUTO
    ) -> ParsedResponse:
        """
        Public method to parse a response.
        
        Args:
            response_text: The raw string from the model.
            format_type: The expected format.
            
        Returns:
            A ParsedResponse object.
        """
        if not response_text or not response_text.strip():
            return ParsedResponse(
                final="FORMAT_ERROR: Empty response",
                original_response=response_text,
                format_error=True
            )

        if format_type == ResponseFormat.AUTO:
            format_type = self._detect_format(response_text)

        if format_type == ResponseFormat.CUSTOM_TAGS:
            return self._parse_custom_tags_legacy(response_text)
        elif format_type in [ResponseFormat.XML]:
            return self._parse_xml_tags(response_text)
        elif format_type == ResponseFormat.JSON:
            return self._parse_json(response_text)
        elif format_type == ResponseFormat.YAML:
            return self._parse_yaml(response_text)
        
        # Fallback to XML tags
        return self._parse_xml_tags(response_text)

    def _detect_format(self, response: str) -> ResponseFormat:
        """Auto-detect the format of the response"""
        response_stripped = response.strip()
        
        # Check for Custom Tags (distinctive markers)
        if '<|channel|>' in response_stripped:
            return ResponseFormat.CUSTOM_TAGS

        # Check for JSON (starts with { or wrapped in markdown)
        if response_stripped.startswith('{') or '```json' in response_stripped.lower() or (response_stripped.startswith('```') and '{' in response_stripped):
            return ResponseFormat.JSON
        
        # Check for YAML (has key: value structure for required fields)
        if all(field in response_stripped for field in ['analysis:', 'proof:', 'final:']) or '```yaml' in response_stripped.lower():
            return ResponseFormat.YAML

        # Check for XML/Custom Tags (contains closing tags)
        if '</' in response_stripped and '>' in response_stripped:
             return ResponseFormat.XML
        
        # Default to XML tags
        return ResponseFormat.XML

    def _parse_xml_tags(self, response_text: str) -> ParsedResponse:
        """
        Parses a response expected to contain XML-like tags.
        
        This implementation uses a 'Thought-Stripping' strategy:
        1. Extract the first thinking/analysis block.
        2. Remove all thinking blocks from the text to prevent extracting nested examples.
        3. Extract proof and final answer from the sanitized text.
        """
        extracted: Dict[str, Optional[str]] = {}
        
        # 1. Extract analysis (thought) from the original text (first match)
        analysis_pattern = self.tag_patterns["analysis"]
        analysis_match = analysis_pattern.search(response_text)
        extracted["analysis"] = analysis_match.group(1).strip() if analysis_match else None
        
        # 2. Strip ALL thinking blocks from the text to prevent nesting issues
        # Using a list of matches allows us to safely remove them without index shifts
        sanitized_text = response_text
        for m in analysis_pattern.finditer(response_text):
            sanitized_text = sanitized_text.replace(m.group(0), "")
            
        # 3. Extract other tags from the sanitized text
        for key in ["proof", "final"]:
            pattern = self.tag_patterns[key]
            if key == "proof":
                # RESTORE AGGREGATION: Collect all valid occurrences of proof
                # This ensures we don't regress if multiple proofs are provided.
                matches = [m.group(1).strip() for m in pattern.finditer(sanitized_text) if m.group(1).strip()]
                extracted[key] = "\n".join(matches) if matches else None
            else:
                # ROBUSTNESS: For the final answer, still take the LAST occurrence 
                # in the sanitized text as a safety against multi-stage logic.
                last_match = None
                for m in pattern.finditer(sanitized_text):
                    last_match = m
                extracted[key] = last_match.group(1).strip() if last_match else None

        # The 'final' answer is mandatory.
        if extracted.get("final") is None:
            # ROBUSTNESS FALLBACK: If <answer> is missing, look for text after the last closed tag
            # in the SANITIZED text (which now excludes thoughts).
            last_tag_match = list(self.fallback_closing_tag_pattern.finditer(sanitized_text))
            if last_tag_match:
                last_pos = last_tag_match[-1].end()
                remaining_text = sanitized_text[last_pos:].strip()
                if remaining_text and len(remaining_text) > 2:
                    return ParsedResponse(
                        analysis=extracted.get("analysis"),
                        proof=extracted.get("proof"),
                        final=remaining_text,
                        original_response=response_text,
                        format_error=False,
                    )

            return ParsedResponse(
                analysis=extracted.get("analysis"),
                proof=extracted.get("proof"),
                final=f"FORMAT_ERROR: Missing <answer> tag and no text after other tags. Original response: {response_text}",
                original_response=response_text,
                format_error=True,
            )

        return ParsedResponse(
            analysis=extracted.get("analysis"),
            proof=extracted.get("proof"),
            final=extracted["final"],
            original_response=response_text,
            format_error=False,
        )

    def _parse_custom_tags_legacy(self, response_text: str) -> ParsedResponse:
        """
        Parses a response using the legacy <|channel|> format.
        """
        channels = {
            match.group(1): match.group(2).strip()
            for match in self.custom_tag_pattern.finditer(response_text)
        }
        
        # Map to expected fields
        analysis = channels.get("analysis")
        proof = channels.get("proof")
        final = channels.get("final")

        if final is None:
            return ParsedResponse(
                analysis=analysis,
                proof=proof,
                final=f"FORMAT_ERROR: Missing final channel in custom tags. Original: {response_text}",
                original_response=response_text,
                format_error=True
            )

        return ParsedResponse(
            analysis=analysis,
            proof=proof,
            final=final,
            original_response=response_text,
            format_error=False
        )

    def _parse_json(self, response_text: str) -> ParsedResponse:
        """Parses a JSON response with robustness improvements."""
        cleaned_response = response_text.strip()
        
        # Strip markdown code blocks
        if '```' in cleaned_response:
            first_backtick = cleaned_response.find('```')
            last_backtick = cleaned_response.rfind('```')
            if first_backtick != -1 and last_backtick != -1 and first_backtick < last_backtick:
                content_block = cleaned_response[first_backtick + 3 : last_backtick]
                cleaned_response = re.sub(r'^\w*\s*', '', content_block).strip()
            
        try:
            data = json.loads(cleaned_response)
            
            # Normalize field aliases
            aliases = {
                'analysis': ['reasoning', 'thought', 'thoughts', 'explanation', 'analysis', 'think'],
                'proof': ['evidence', 'quote', 'reference', 'source', 'proof', 'trace'],
                'final': ['answer', 'conclusion', 'result', 'final_answer', 'final']
            }
            
            normalized = {}
            for target, sources in aliases.items():
                for s in sources:
                    if s in data:
                        normalized[target] = data[s]
                        break
            
            if not normalized.get('final'):
                return ParsedResponse(
                    analysis=normalized.get('analysis'),
                    proof=normalized.get('proof'),
                    final=f"FORMAT_ERROR: Missing final answer in JSON. Original: {response_text}",
                    original_response=response_text,
                    format_error=True
                )

            return ParsedResponse(
                analysis=normalized.get('analysis'),
                proof=normalized.get('proof'),
                final=str(normalized['final']),
                original_response=response_text,
                format_error=False
            )
        except Exception as e:
            return ParsedResponse(
                final=f"FORMAT_ERROR: JSON parse failed: {str(e)}",
                original_response=response_text,
                format_error=True
            )

    def _parse_yaml(self, response_text: str) -> ParsedResponse:
        """Parses a YAML response."""
        try:
            data = yaml.safe_load(response_text.strip())
            if not isinstance(data, dict):
                raise ValueError("YAML must be a dictionary")
            
            # Same alias logic as JSON
            aliases = {
                'analysis': ['reasoning', 'thought', 'thoughts', 'explanation', 'analysis', 'think'],
                'proof': ['evidence', 'quote', 'reference', 'source', 'proof', 'trace'],
                'final': ['answer', 'conclusion', 'result', 'final_answer', 'final']
            }
            
            normalized = {}
            for target, sources in aliases.items():
                for s in sources:
                    if s in data:
                        normalized[target] = data[s]
                        break

            if not normalized.get('final'):
                return ParsedResponse(
                    analysis=normalized.get('analysis'),
                    proof=normalized.get('proof'),
                    final=f"FORMAT_ERROR: Missing final answer in YAML. Original: {response_text}",
                    original_response=response_text,
                    format_error=True
                )

            return ParsedResponse(
                analysis=normalized.get('analysis'),
                proof=normalized.get('proof'),
                final=str(normalized['final']),
                original_response=response_text,
                format_error=False
            )
        except Exception as e:
            return ParsedResponse(
                final=f"FORMAT_ERROR: YAML parse failed: {str(e)}",
                original_response=response_text,
                format_error=True
            )