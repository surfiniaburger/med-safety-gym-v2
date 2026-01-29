
import sys
import os

# Add parent directory to sys.path to import med_safety_eval
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from med_safety_eval.format_parser import FormatParser

def test_parser_with_placeholders():
    parser = FormatParser()
    
    # Mock response illustrating the issue: placeholders inside <think> tags
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
    
    print("Testing response with placeholders in <think> block...")
    parsed = parser.parse(response_text)
    
    print(f"Final extracted: '{parsed.final}'")
    print(f"Proof extracted: '{parsed.proof}'")
    
    # Assertions
    expected_final = "Actual final answer."
    expected_proof = '"Real evidence from context"'
    
    if parsed.final == expected_final and parsed.proof == expected_proof:
        print("✅ Success: Parser correctly picked the LAST occurrence!")
    else:
        print("❌ Failure: Parser picked placeholders or wrong content.")
        print(f"ERROR: Expected '{expected_final}', got '{parsed.final}'")
        sys.exit(1)

if __name__ == "__main__":
    test_parser_with_placeholders()
