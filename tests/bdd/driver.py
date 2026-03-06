from unittest.mock import AsyncMock, MagicMock, patch

from med_safety_gym.claw_agent import SafeClawAgent
from med_safety_gym.intent_classifier import IntentCategory, IntentResult
from med_safety_gym.session_memory import SessionMemory

from tests.bdd.dsl import Scenario
from tests.bdd.stubs import CaptureUpdater, make_experience_client, make_llm_response


class SafeClawScenarioDriver:
    def __init__(self):
        self.agent = SafeClawAgent()
        self.session = SessionMemory("bdd-user", scope="base")
        self.updater = CaptureUpdater()

    async def run(self, sc: Scenario):
        exp_client = make_experience_client()
        self.agent.experience_client_factory = MagicMock(return_value=exp_client)
        self.agent._apply_safety_gate = AsyncMock(return_value=(sc.safety_is_safe, sc.safety_reason))

        intent = IntentResult(category=IntentCategory[sc.intent], is_correction=False)

        with patch("med_safety_gym.claw_agent.acompletion", AsyncMock(return_value=make_llm_response(sc.llm_response))):
            await self.agent.context_aware_action(
                action=sc.user_action,
                raw_text=sc.user_action,
                context=sc.verified_context,
                parity_context=sc.verified_context,
                updater=self.updater,
                session=self.session,
                intent=intent,
            )

        return self.updater.events[-1]
