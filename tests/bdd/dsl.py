from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    verified_context: str
    user_action: str
    llm_response: str
    intent: str
    safety_is_safe: bool
    safety_reason: str
    expected_state: str
    expected_substring: str
    expected_abstained: bool


class ScenarioBuilder:
    def __init__(self, name: str):
        self._name = name
        self._verified_context = ""
        self._user_action = ""
        self._llm_response = ""
        self._intent = "FOLLOW_UP"
        self._safety_is_safe = True
        self._safety_reason = ""
        self._expected_state = "completed"
        self._expected_substring = ""
        self._expected_abstained = False

    def given_verified_context(self, text: str):
        self._verified_context = text
        return self

    def and_user_action(self, text: str):
        self._user_action = text
        return self

    def and_intent(self, intent: str):
        self._intent = intent
        return self

    def when_model_responds(self, text: str):
        self._llm_response = text
        return self

    def and_safety_gate_returns(self, is_safe: bool, reason: str = ""):
        self._safety_is_safe = is_safe
        self._safety_reason = reason
        return self

    def then_expect(self, state: str, substring: str = "", abstained: bool = False):
        self._expected_state = state
        self._expected_substring = substring
        self._expected_abstained = abstained
        return self

    def build(self) -> Scenario:
        return Scenario(
            name=self._name,
            verified_context=self._verified_context,
            user_action=self._user_action,
            llm_response=self._llm_response,
            intent=self._intent,
            safety_is_safe=self._safety_is_safe,
            safety_reason=self._safety_reason,
            expected_state=self._expected_state,
            expected_substring=self._expected_substring,
            expected_abstained=self._expected_abstained,
        )


def scenario(name: str) -> ScenarioBuilder:
    return ScenarioBuilder(name)
