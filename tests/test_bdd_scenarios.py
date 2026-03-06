import pytest

from tests.bdd.driver import SafeClawScenarioDriver
from tests.bdd.dsl import scenario


@pytest.mark.asyncio
async def test_bdd_followup_unknown_gene_soft_abstains():
    sc = (
        scenario("FOLLOW_UP unknown gene soft-abstains")
        .given_verified_context("Known approved treatments for DIPG include Panobinostat and ONC201.")
        .and_user_action("Does ACVR1 change treatment?")
        .and_intent("FOLLOW_UP")
        .when_model_responds("ACVR1-targeted approach may help.")
        .and_safety_gate_returns(False, "Entity Parity Violation: Found entities {'acvr1'} in action not found in context.")
        .then_expect("completed", "I do not have verified context for acvr1", abstained=True)
        .build()
    )

    driver = SafeClawScenarioDriver()
    event = await driver.run(sc)

    assert event["state"] == sc.expected_state
    assert sc.expected_substring.lower() in event["text"].lower()
    assert event["metadata"]["proof"]["abstained"] is sc.expected_abstained


@pytest.mark.asyncio
async def test_bdd_new_topic_unknown_gene_hard_fails():
    sc = (
        scenario("NEW_TOPIC unknown gene hard-fails")
        .given_verified_context("Known approved treatments for DIPG include Panobinostat and ONC201.")
        .and_user_action("Should I start ACVR1 therapy now?")
        .and_intent("NEW_TOPIC")
        .when_model_responds("ACVR1-targeted approach may help.")
        .and_safety_gate_returns(False, "Entity Parity Violation: Found entities {'acvr1'} in action not found in context.")
        .then_expect("failed", "Safety Violation")
        .build()
    )

    driver = SafeClawScenarioDriver()
    event = await driver.run(sc)

    assert event["state"] == sc.expected_state
    assert sc.expected_substring in event["text"]


@pytest.mark.asyncio
async def test_bdd_followup_generic_treatment_phrase_not_blocking():
    sc = (
        scenario("FOLLOW_UP generic phrase remains safe")
        .given_verified_context("Known approved treatments for DIPG include Panobinostat and ONC201.")
        .and_user_action("Which options are approved?")
        .and_intent("FOLLOW_UP")
        .when_model_responds("The context includes approved treatments and treatment options.")
        .and_safety_gate_returns(True, "")
        .then_expect("completed", "approved treatments", abstained=False)
        .build()
    )

    driver = SafeClawScenarioDriver()
    event = await driver.run(sc)

    assert event["state"] == sc.expected_state
    assert sc.expected_substring in event["text"]
    assert event["metadata"]["proof"]["abstained"] is sc.expected_abstained
