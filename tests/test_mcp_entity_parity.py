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

