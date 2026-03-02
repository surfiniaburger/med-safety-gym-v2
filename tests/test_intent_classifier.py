import pytest
from med_safety_gym.intent_classifier import IntentClassifier, IntentCategory

"""
Test List (Canon TDD):
[ ] Test classifying a simple new topic (e.g. "What is Aspirin?")
[ ] Test classifying a refinement (e.g. "I meant the liquid form")
[ ] Test classifying an expansion (e.g. "What about for adults?")
[ ] Test classifying a follow-up (e.g. "What are its side effects?")
[ ] Test classifying a correction of model assumption (e.g. "No, I have an ulcer")
[ ] Test classifying a confirmation (e.g. "Yes, that's exactly right")
"""

def test_classify_new_topic():
    classifier = IntentClassifier()
    # A completely out-of-the-blue question with no prior context
    result = classifier.classify("What is Aspirin?")
    assert result.category == IntentCategory.NEW_TOPIC
    assert result.is_correction is False

def test_classify_refinement():
    classifier = IntentClassifier()
    # A refinement typically implies a previous topic is being narrowed down
    result = classifier.classify("I meant the liquid form")
    assert result.category == IntentCategory.REFINEMENT
    assert result.is_correction is False

def test_classify_expansion():
    classifier = IntentClassifier()
    # An expansion broadens the scope of the previous topic
    result = classifier.classify("What about for adults?")
    assert result.category == IntentCategory.EXPANSION
    assert result.is_correction is False

def test_classify_follow_up():
    classifier = IntentClassifier()
    # A follow-up asks for more details on the current topic
    result = classifier.classify("What are its side effects?")
    assert result.category == IntentCategory.FOLLOW_UP
    assert result.is_correction is False

def test_classify_recollection():
    classifier = IntentClassifier()
    result = classifier.classify("What did we say earlier about the dosage?")
    assert result.category == IntentCategory.RECOLLECTION
    assert result.is_correction is False

def test_classify_correction():
    classifier = IntentClassifier()
    result = classifier.classify("No, I meant the adult dosage, not pediatric.")
    assert result.category == IntentCategory.REFINEMENT
    assert result.is_correction is True

def test_classify_confirmation():
    classifier = IntentClassifier()
    result = classifier.classify("Yes, exactly right.")
    # Assuming confirmation might just be a continuation/follow-up or new topic but with is_correction=False
    # Let's say confirmation is just a FOLLOW_UP or NEW_TOPIC that is NOT a correction.
    # We might not even need a separate category, just check is_correction is False
    assert result.is_correction is False
