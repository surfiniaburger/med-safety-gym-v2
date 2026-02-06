import pytest
from med_safety_eval.logic import is_grounded

def test_synthesis_fallback_index_6_simulation():
    """Verify that a sentence synthesized from context fragments is grounded via entity fallback."""
    context = "Initial management: focal radiotherapy (54 Gy in 30 fractions). Patient enrolled in ONC201 (625 mg once weekly)."
    
    # Rephrased synthesis: Model summarizes two facts into one sentence
    # "The patient received 54 Gy of focal radiotherapy before starting ONC201 at 625 mg weekly."
    # Character similarity might be low, but "54", "Gy", "ONC201", "625", "mg" are all present.
    response = "The patient received 54 Gy of focal radiation before starting ONC201 at 625 mg weekly."
    assert is_grounded(response, context)

def test_entity_mismatch_fails():
    """Verify that if key entities (numbers/long words) are missing, grounding fails even if structure is similar."""
    context = "Dose was 50 Gy."
    # Similarity might be high due to "Dose was" but "60" is a hallucination
    assert not is_grounded("Dose was 60 Gy.", context)

def test_stopword_window_anchoring():
    """Verify that the sliding window anchors on meaningful words, not 'the' or 'with'."""
    context = "The primary tumor was located in the pons with significant expansion."
    # If it anchors on "the", it might find a false positive window.
    # We want it to anchor on "pons" or "expansion".
    assert is_grounded("tumor was in the pons with expansion", context)

def test_long_sentence_with_reordered_clauses():
    """Reordered clauses with identical entities should be grounded."""
    context = "Tazemetostat showed 30% response rate in H3K27M patients."
    response = "In H3K27M patients, a 30% response rate was seen with Tazemetostat."
    assert is_grounded(response, context)
