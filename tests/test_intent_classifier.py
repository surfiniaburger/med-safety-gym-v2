import pytest
from med_safety_gym.intent_classifier import IntentClassifier, IntentCategory

def test_intent_classification_refinements():
    classifier = IntentClassifier()
    
    # Test 'switching' refinement (distilled from failure)
    res = classifier.classify("The patient is switching to ONC201.")
    assert res.category == IntentCategory.REFINEMENT
    
    # Test 'i meant' refinement
    res = classifier.classify("No, I meant Panobinostat.")
    assert res.category == IntentCategory.REFINEMENT
    assert res.is_correction is True
    
    # Test 'change to' refinement
    res = classifier.classify("Can we change to the other drug?")
    assert res.category == IntentCategory.REFINEMENT

def test_intent_classification_follow_ups():
    classifier = IntentClassifier()
    
    # Test 'having trouble'
    res = classifier.classify("They are having trouble with the treatment.")
    assert res.category == IntentCategory.FOLLOW_UP
    
    # Test 'what are'
    res = classifier.classify("What are the side effects?")
    assert res.category == IntentCategory.FOLLOW_UP

def test_new_topic():
    classifier = IntentClassifier()
    res = classifier.classify("Tell me about DIPG.")
    assert res.category == IntentCategory.NEW_TOPIC
