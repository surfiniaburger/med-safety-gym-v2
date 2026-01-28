import pytest
from med_safety_eval.rubrics.llm_judge import LLMJudge

class MockAction:
    def __init__(self, content):
        self.content = content

class MockObservation:
    def __init__(self, context, question=""):
        self.context = context
        self.question = question
    
    def __str__(self):
        return f"Context: {self.context}\nQuestion: {self.question}"

def test_llm_judge_initialization():
    judge = LLMJudge(prompt_template="{action}", inference_fn=lambda x: "Score: 1.0")
    assert judge.prompt_template == "{action}"

def test_llm_judge_forward_basic():
    """Test that it formats prompt and calls inference."""
    
    captured_prompt = None
    def mock_inference(prompt):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "Score: 0.8"

    judge = LLMJudge(
        prompt_template="Judge this: {action} given {observation}", 
        inference_fn=mock_inference
    )
    
    action = MockAction("Do X")
    obs = MockObservation("Situation Y")
    
    score = judge(action, obs)
    
    assert score == 0.8
    assert "Judge this: Do X" in captured_prompt
    assert "Situation Y" in captured_prompt

def test_llm_judge_score_parsing():
    """Test standard score parsing logic."""
    judge = LLMJudge("", lambda x: "")
    
    assert judge.score_parser("Score: 1.0") == 1.0
    assert judge.score_parser("Score: 0.5") == 0.5
    assert judge.score_parser("Rating: Score: 0.9 end") == 0.9
    
    # Fallback
    assert judge.score_parser("PASS") == 1.0
    assert judge.score_parser("FAIL") == 0.0
    assert judge.score_parser("Garbage") == 0.0

def test_llm_judge_custom_parser():
    """Test custom parsing logic."""
    def output_parser(text):
        if "Good" in text: return 1.0
        return 0.0
    
    judge = LLMJudge("", lambda x: "It is Good", score_parser=output_parser)
    assert judge("a", "o") == 1.0
