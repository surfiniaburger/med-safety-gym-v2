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
            "final": ["answer", "final", "conclusion", "result", "final_answer"],
        }
        
        # Shared template for XML-like tag extraction
        # V4.6: Support both <tag> and [tag].
        self.tag_pattern_template = r"(?:<|\[)(?:{tags})(?:\s+[^>\]]*)?(?:>|\])(.*?)(?:<|\[)/(?:{tags})(?:>|\])"
        
        # Pre-compile regex for efficiency, supporting multiple tag names for flexibility.
        self.tag_patterns = {
            key: re.compile(
                self.tag_pattern_template.format(tags='|'.join(re.escape(a) for a in aliases)),
                re.DOTALL | re.IGNORECASE
            )
            for key, aliases in self.tag_aliases.items()
        }
        
        # Unclosed tag fallback for analysis
        # V4.6: Stop at the next tag or end of string
        all_aliases = [a for aliases in self.tag_aliases.values() for a in aliases]
        self.unclosed_analysis_pattern = re.compile(
            r"(?:<|\[)(?:{tags})(?:\s+[^>\]]*)?(?:>|\])(.*?)(?=(?:<|\[)(?:{all_tags})|$)".format(
                tags='|'.join(re.escape(a) for a in self.tag_aliases["analysis"]),
                all_tags='|'.join(re.escape(a) for a in all_aliases)
            ),
            re.DOTALL | re.IGNORECASE
        )
        
        # Regex pattern for custom tag format (backward compatibility)
        self.custom_tag_pattern = re.compile(
            r'<\|channel\|>(\w+)<\|message\|>(.*?)<\|end\|>',
            re.DOTALL
        )

    def parse(
        self,
        response_text: str,
        format_type: ResponseFormat = ResponseFormat.AUTO
    ) -> ParsedResponse:
        """
        Public method to parse a response.
        """
        if not response_text or not response_text.strip():
            return ParsedResponse(
                final="FORMAT_ERROR: Empty response",
                original_response=response_text,
                format_error=True
            )

        if format_type == ResponseFormat.AUTO:
            format_type = self._detect_format(response_text)

        parser_map = {
            ResponseFormat.CUSTOM_TAGS: self._parse_custom_tags,
            ResponseFormat.XML: self._parse_xml,
            ResponseFormat.JSON: self._parse_json,
            ResponseFormat.YAML: self._parse_yaml,
        }
        
        parser_func = parser_map.get(format_type, self._parse_xml)
        return parser_func(response_text, original_response=response_text)

    def _detect_format(self, response: str) -> ResponseFormat:
        """Auto-detect the format of the response"""
        response_stripped = response.strip()
        
        if '<|channel|>' in response_stripped:
            return ResponseFormat.CUSTOM_TAGS

        if response_stripped.startswith('{') or '```json' in response_stripped.lower():
            return ResponseFormat.JSON
        
        if '<' in response_stripped and '>' in response_stripped:
             return ResponseFormat.XML

        if all(field in response_stripped for field in ['analysis:', 'proof:', 'final:']) or '```yaml' in response_stripped.lower():
            return ResponseFormat.YAML
        
        return ResponseFormat.XML

    def _parse_xml(self, response_text: str, original_response: str = "") -> ParsedResponse:
        """
        Parses a response expected to contain XML-like tags with V4.5 robustness.
        """
        extracted: Dict[str, Optional[str]] = {}
        
        # 1. Extract analysis (thought) block (first match)
        analysis_pattern = self.tag_patterns["analysis"]
        analysis_match = analysis_pattern.search(response_text)
        
        if analysis_match:
            extracted["analysis"] = analysis_match.group(1).strip()
            # 2. Sanitized text = original minus thinking blocks
            start, end = analysis_match.span()
            sanitized_text = response_text[:start] + response_text[end:]
        else:
            # Try unclosed fallback for analysis
            unclosed_match = self.unclosed_analysis_pattern.search(response_text)
            if unclosed_match:
                extracted["analysis"] = unclosed_match.group(1).strip()
                start, end = unclosed_match.span()
                sanitized_text = response_text[:start] + response_text[end:]
            else:
                extracted["analysis"] = None
                sanitized_text = response_text
            
        # 3. Extract proof and final answer from the sanitized text
        for key in ["proof", "final"]:
            pattern = self.tag_patterns[key]
            if key == "proof":
                # Aggregate multiple proof tags
                matches = [m.group(1).strip() for m in pattern.finditer(sanitized_text) if m.group(1).strip()]
                extracted[key] = "\n".join(matches) if matches else None
            else:
                # Take the last final answer
                last_match = None
                for m in pattern.finditer(sanitized_text):
                    last_match = m
                extracted[key] = last_match.group(1).strip() if last_match else None

        # 4. Fallback Handling if <final> is missing
        is_format_error = False
        if extracted.get("final") is None:
            # ROBUSTNESS FALLBACK A: "Dangling" Answer - look for substantial text outside of tags
            text_without_blocks = sanitized_text
            # Remove all well-formed tag blocks that we recognize
            for key_alias, aliases in self.tag_aliases.items():
                p = re.compile(rf"<(?:{'|'.join(re.escape(a) for a in aliases)})(?:\s+[^>]*)?>(.*?)</(?:{'|'.join(re.escape(a) for a in aliases)})>", re.DOTALL | re.IGNORECASE)
                text_without_blocks = p.sub("", text_without_blocks)
            
            # Remove any stray unclosed tags or top-level wrappers
            dangling_candidate = re.sub(r'<[^>]+>|[\[][^\]]+[\]]', '', text_without_blocks).strip()
            
            if len(dangling_candidate) > 5:
                # Clean up common Markdown headers or markers from the dangling answer
                cleaned_candidate = dangling_candidate
                markers = [
                    r"^(?:###?\s+)?(?:Final\s+)?Answer:?[\s\n]*",
                    r"^(?:###?\s+)?Conclusion:?[\s\n]*",
                    r"^(?:###?\s+)?Result:?[\s\n]*",
                    r"^\*\*Answer:\*\*[\s\n]*",
                    r"^Answer:?[\s\n]*"
                ]
                for m in markers:
                    match = re.search(m, cleaned_candidate, flags=re.IGNORECASE)
                    if match:
                        # Only strip if there's substantial text after the marker
                        potential_new = cleaned_candidate[match.end():].strip()
                        if len(potential_new) > 3:
                            cleaned_candidate = potential_new
                            break
                
                extracted["final"] = cleaned_candidate
                is_format_error = False # Dangling text is accepted as clear intent
            else:
                # ROBUSTNESS FALLBACK B: "Rescued" Answer - look inside the thinking block
                if extracted.get("analysis"):
                    thoughts = extracted["analysis"]
                    # Look for conclusion markers - find the LAST one for better accuracy
                    # We use a greedy .* at the start to push the match to the end of the text
                    # V4.6: Expanded markers
                    marker = re.search(r".*(\b(?:conclusion|answer|therefore|thus|so|consequently|final answer|result|summary)\b[\W\s]+.*?)$", thoughts, re.IGNORECASE | re.DOTALL)
                    if marker:
                        extracted["final"] = f"Rescued: {marker.group(1).strip()}"
                        is_format_error = True # Rescued from inside thoughts is still a format deviation
                    elif len(thoughts) > 50:
                        # Fallback to last sentence
                        sentences = [s.strip() for s in re.split(r'[\.\?\!\n]', thoughts) if s.strip()]
                        if sentences:
                            extracted["final"] = f"Rescued: {sentences[-1]}"
                            is_format_error = True

        if extracted.get("final") is None:
            return ParsedResponse(
                analysis=extracted.get("analysis"),
                proof=extracted.get("proof"),
                final=f"FORMAT_ERROR: Missing <answer> tag. Original response: {original_response or response_text}",
                original_response=original_response or response_text,
                format_error=True,
            )

        return ParsedResponse(
            analysis=extracted.get("analysis"),
            proof=extracted.get("proof"),
            final=extracted["final"],
            original_response=original_response or response_text,
            format_error=is_format_error,
        )

    def _parse_custom_tags(self, response_text: str, original_response: str = "") -> ParsedResponse:
        """Parses legacy <|channel|> format."""
        channels = {
            match.group(1): match.group(2).strip()
            for match in self.custom_tag_pattern.finditer(response_text)
        }
        
        analysis = channels.get("analysis")
        proof = channels.get("proof")
        final = channels.get("final")

        if final is None:
            return ParsedResponse(
                analysis=analysis,
                proof=proof,
                final=f"FORMAT_ERROR: Missing final channel. Original: {original_response or response_text}",
                original_response=original_response or response_text,
                format_error=True
            )

        return ParsedResponse(
            analysis=analysis,
            proof=proof,
            final=final,
            original_response=original_response or response_text,
            format_error=False
        )

    def _parse_json(self, response_text: str, original_response: str = "") -> ParsedResponse:
        """Parses JSON responses."""
        cleaned = response_text.strip()
        if '```' in cleaned:
            blocks = re.findall(r'```(?:json)?(.*?)```', cleaned, re.DOTALL | re.IGNORECASE)
            if blocks: cleaned = blocks[0].strip()
            
        try:
            data = json.loads(cleaned)
            aliases = {
                'analysis': ['reasoning', 'thought', 'explanation', 'analysis', 'think'],
                'proof': ['evidence', 'quote', 'reference', 'proof', 'trace', 'source'],
                'final': ['answer', 'conclusion', 'result', 'final', 'final_answer']
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
                    final=f"FORMAT_ERROR: Missing final answer in JSON.",
                    original_response=original_response or response_text,
                    format_error=True
                )
            return ParsedResponse(
                analysis=normalized.get('analysis'),
                proof=normalized.get('proof'),
                final=normalized['final'],
                original_response=original_response or response_text,
                format_error=False
            )
        except:
            return ParsedResponse(final="FORMAT_ERROR: Invalid JSON", original_response=response_text, format_error=True)

    def _parse_yaml(self, response_text: str, original_response: str = "") -> ParsedResponse:
        """Parses YAML responses."""
        cleaned = response_text.strip()
        if '```' in cleaned:
            blocks = re.findall(r'```(?:yaml)?(.*?)```', cleaned, re.DOTALL | re.IGNORECASE)
            if blocks: cleaned = blocks[0].strip()
        try:
            data = yaml.safe_load(cleaned)
            if not isinstance(data, dict): raise ValueError()
            return ParsedResponse(
                analysis=data.get('analysis'),
                proof=data.get('proof'),
                final=data.get('final') or "FORMAT_ERROR: Missing final in YAML",
                original_response=original_response or response_text,
                format_error=not bool(data.get('final'))
            )
        except:
            return ParsedResponse(final="FORMAT_ERROR: Invalid YAML", original_response=response_text, format_error=True)