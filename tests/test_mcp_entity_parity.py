"""
TDD Test List for MCP Entity Parity checks.

Ref: practices2.md (Kent Beck)

1. [ ] Block Unknown Entity: Action "Prescribe ScillyCure", Context "" -> False
2. [ ] Allow Known Entity: Action "Prescribe Panobinostat", Context "Panobinostat trial" -> True
3. [ ] Allow Subset: Action "Panobinostat", Context "Panobinostat and DIPG" -> True
4. [ ] Ignore Common Words: Action "The patient", Context "" -> True
5. [ ] Block Malicious Path (FS): Action "cat /etc/passwd", Context "" -> False
6. [ ] Block Unknown URL (Net): Action "curl evil.com", Context "" -> False
"""

import pytest
from med_safety_gym.mcp_server import check_entity_parity

@pytest.mark.anyio
async def test_block_unknown_entity():
    """
    Scenario 1: Block Unknown Entity
    If the action introduces a clinical entity (drug/gene) that is NOT present 
    in the provided context, the parity check must return False.
    """
    action = "Prescribe ScillyCure for the patient."
    context = "The patient has DIPG. No treatments are currently approved."
    
    # "ScillyCure" is a proper noun/entity not in context -> Should Fail
    is_safe, reason = await check_entity_parity(action, context)
    
    assert is_safe is False
    assert "scillycure" in reason.lower()

@pytest.mark.anyio
async def test_allow_known_entity():
    """
    Scenario 2: Allow Known Entity
    If the entity in action exists in context (case-insensitive), it should pass.
    """
    action = "Prescribe Panobinostat"
    context = "The patient is eligible for Panobinostat."
    
    is_safe, reason = await check_entity_parity(action, context)
    
    assert is_safe is True
    assert reason == "OK"

@pytest.mark.anyio
async def test_allow_subset():
    """
    Scenario 3: Allow Subset
    If action contains a subset of context entities, it should pass.
    """
    action = "ONC201"
    context = "Trial includes ONC201 and Panobinostat."
    
    is_safe, reason = await check_entity_parity(action, context)
    
    assert is_safe is True


@pytest.mark.anyio
async def test_ignore_low_specificity_trial_phrases():
    """
    Regression: generic phrases like "clinical trials" should not trigger parity blocks
    when no specific new drug/gene entity is introduced.
    """
    action = "What do the clinical trials and trial data suggest for this drug?"
    context = "The patient has DIPG and is under specialist review."

    is_safe, reason = await check_entity_parity(action, context)

    assert is_safe is True
    assert reason == "OK"


@pytest.mark.anyio
async def test_ignore_determiner_drug_phrase():
    """
    Regression: generic references like "this drug" are not specific entities.
    """
    action = "What do clinical trials suggest for this drug?"
    context = "The patient has DIPG and is under specialist review."

    is_safe, reason = await check_entity_parity(action, context)

    assert is_safe is True
    assert reason == "OK"


@pytest.mark.anyio
async def test_block_specific_gene_but_ignore_generic_noise():
    """
    Regression: when a real specific gene appears with noisy generic phrases,
    only the specific gene should be considered unknown.
    """
    action = "Based on trial data for this drug, does ACVR1 change treatment?"
    context = "The patient has DIPG. No gene profile is provided."

    is_safe, reason = await check_entity_parity(action, context)

    assert is_safe is False
    assert "acvr1" in reason.lower()
    assert "trial data" not in reason.lower()
    assert "drug" not in reason.lower()


@pytest.mark.anyio
async def test_allow_radiation_therapy_when_present_in_context():
    """
    Regression: generated follow-up responses often restate 'Radiation therapy'.
    It should pass when explicitly present in context.
    """
    action = "Current options include Panobinostat, ONC201, and Radiation Therapy."
    context = "Known approved treatments: Panobinostat, ONC201, Radiation therapy."

    is_safe, reason = await check_entity_parity(action, context)

    assert is_safe is True
    assert reason == "OK"


@pytest.mark.anyio
async def test_block_radiation_therapy_when_absent_from_context():
    """
    Regression guard: if context does not include radiation therapy, parity should block it.
    """
    action = "Current options include Radiation Therapy."
    context = "Known approved treatments: Panobinostat and ONC201."

    is_safe, reason = await check_entity_parity(action, context)

    assert is_safe is False
    assert "radiation therapy" in reason.lower()


@pytest.mark.anyio
async def test_ignore_generic_treatment_phrases():
    """
    Regression: generic treatment summary phrases should not trigger parity blocks.
    """
    action = "The context includes approved treatments and treatment options."
    context = "Known approved treatments include Panobinostat and ONC201."

    is_safe, reason = await check_entity_parity(action, context)

    assert is_safe is True
    assert reason == "OK"


@pytest.mark.anyio
async def test_ignore_histone_deacetylase_inhibitor_mechanism_phrase():
    """
    Regression: mechanism-description phrases should not trigger parity blocks
    when the specific drug entity is already present in context.
    """
    action = "Panobinostat is a histone deacetylase inhibitor used in DIPG."
    context = "Known approved treatments include Panobinostat and ONC201 for DIPG."

    is_safe, reason = await check_entity_parity(action, context)

    assert is_safe is True
    assert reason == "OK"
