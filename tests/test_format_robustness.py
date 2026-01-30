import pytest
from med_safety_eval.format_parser import FormatParser
from med_safety_eval.models import ResponseFormat

def test_unclosed_think_tag():
    """
    Test that an unclosed <think> tag is still parsed as analysis,
    and any following text is treated as the final answer.
    """
    parser = FormatParser()
    response = "<think>I am thinking about the patient's eligibility. The patient has a mutation. The answer is: The patient is eligible."
    
    parsed = parser.parse(response)
    
    # V4.6: Now supports unclosed tags for analysis
    assert parsed.analysis == "I am thinking about the patient's eligibility. The patient has a mutation. The answer is: The patient is eligible."
    assert "eligible" in parsed.final.lower()
    assert parsed.format_error is True

def test_markdown_header_answer():
    """
    Test that an answer provided under a Markdown header is recognized and cleaned.
    """
    parser = FormatParser()
    response = """<think>
Reasoning here...
</think>

### Final Answer
The patient is ineligible due to prior HDAC inhibitor exposure."""

    parsed = parser.parse(response)
    
    assert parsed.final == "The patient is ineligible due to prior HDAC inhibitor exposure."
    assert parsed.format_error is False

def test_bracketed_answer():
    """
    Test that an answer in [ANSWER] brackets is recognized.
    """
    parser = FormatParser()
    response = "<think>...</think> [ANSWER] The patient is eligible. [/ANSWER]"
    
    parsed = parser.parse(response)
    
    assert parsed.final == "The patient is eligible."
    assert parsed.format_error is False

def test_mid_thought_correction_rescue():
    """
    Reproduce the issue where a mid-thought "Therefore" is rescued as the final answer
    when the model is cut off or fails to provide a proper answer tag.
    """
    parser = FormatParser()
    # Simulating the cut-off response from index 3 in format_again.md
    response = """<think>
The patient has H3K27M mutation. 
Therefore, I must have made a mistake.
Wait, wait—the eligibility criterion (2) says "no prior exposure to any histone deacetyl
</think>"""

    parsed = parser.parse(response)
    
    # It currently rescues: "Rescued: Therefore, I must have made a mistake..."
    assert "Rescued: Therefore" in parsed.final
    assert parsed.format_error is True

def test_answer_marker_in_thoughts():
    """
    Test that if the model says "Answer: X" inside the thoughts, it's rescued.
    """
    parser = FormatParser()
    response = "<think> Reasoning... Answer: The patient is ineligible. </think>"
    parsed = parser.parse(response)
    
    assert "ineligible" in parsed.final.lower()
    assert "Rescued:" in parsed.final
    assert parsed.format_error is True

def test_text_after_think_block():
    """
    Ensure text after a closed think block is captured as the final answer.
    """
    parser = FormatParser()
    response = "<think>Reasoning</think> Final Answer"
    res = parser.parse(response)
    
    assert res.analysis == "Reasoning"
    assert res.final == "Final Answer"
    assert res.format_error is False

if __name__ == "__main__":
    pytest.main([__file__])