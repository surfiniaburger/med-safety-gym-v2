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

# Import ParsedResponse from standalone library
from med_safety_eval.models import ParsedResponse as DIPGResponse, ResponseFormat
from med_safety_eval.format_parser import FormatParser as BaseFormatParser

class FormatParser(BaseFormatParser):
    """
    Parses DIPG Safety responses in multiple formats.
    
    Supports:
    - JSON: {"analysis": "...", "proof": "...", "final": "..."}
    - XML: <dipg_response><analysis>...</analysis>...</dipg_response>
    - YAML: analysis: ...\nproof: ...\nfinal: ...
    - Custom Tags: <|channel|>analysis<|message|>...<|end|>...
    """
    
    def parse(
        self,
        response: str,
        format_type: ResponseFormat = ResponseFormat.AUTO
    ) -> DIPGResponse:
        """
        Parse response in any supported format.

        Args:
            response: The LLM-generated response string.
            format_type: Expected format (or AUTO to detect).

        Returns:
            A normalized DIPGResponse object.
        """
        # The base class already handles dictionary-based dispatch and robust parsing logic.
        # We simply delegate to it, ensuring argument compatibility.
        return super().parse(response_text=response, format_type=format_type)
