import pytest
from med_safety_eval.logic import MedicalEntityExtractor

@pytest.fixture(scope="module")
def extractor():
    # We use the small model to keep tests fast
    return MedicalEntityExtractor(model_name="urchade/gliner_small-v2.1")

def test_gliner_extractor_ignores_conversational_words(extractor):
    text = "hi how does it differ from those we mentioned? it might impact or address the issue."
    entities = extractor.extract_entities(text)
    assert len(entities) == 0, f"Expected 0 entities, got {entities} from conversational text."

def test_gliner_extractor_finds_medical_terms(extractor):
    text = "The patient is taking Panobinostat for DIPG."
    entities = extractor.extract_entities(text)
    entities_lower = {e.lower() for e in entities}
    
    assert 'panobinostat' in entities_lower, f"Failed to extract 'panobinostat', got {entities_lower}"
    assert 'dipg' in entities_lower, f"Failed to extract 'dipg', got {entities_lower}"

def test_gliner_extractor_complex_sentence(extractor):
    text = "Swallowing difficulties might impact her ability to take oral meds, we need to address this."
    entities = extractor.extract_entities(text)
    entities_lower = {e.lower() for e in entities}
    
    # Check that medical terms are found
    assert any('swallow' in str(e) for e in entities_lower), f"Failed to extract swallowing-related term, got {entities_lower}"
    assert any('med' in str(e) for e in entities_lower), f"Failed to extract medication-related term, got {entities_lower}"
    
    # Check that conversational non-entities are ignored
    assert 'impact' not in entities_lower, "Conversational word 'impact' was incorrectly flagged as an entity."
    assert 'address' not in entities_lower, "Conversational word 'address' was incorrectly flagged as an entity."
