import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ResearchResult:
    val_bpb: float
    peak_vram_gb: float
    status: str
    description: str

class ResearchProtocolDriver:
    """
    Layer 3: Protocol Driver.
    Translates domain concepts into technical actions.
    Base class or Interface for the Research Agent to use.
    """
    async def ensure_setup(self) -> bool:
        raise NotImplementedError()

    async def run_experiment(self, experiment_id: str) -> ResearchResult:
        raise NotImplementedError()

    async def log_result(self, result: ResearchResult) -> None:
        raise NotImplementedError()

class MockResearchDriver(ResearchProtocolDriver):
    """
    Layer 4: External System Stub (Mock).
    Simulates the research environment for testing.
    """
    def __init__(self):
        self.setup_done = False
        self.results = []
        self.branches = []

    async def ensure_setup(self) -> bool:
        self.setup_done = True
        return True

    async def run_experiment(self, experiment_id: str) -> ResearchResult:
        return ResearchResult(
            val_bpb=1.23,
            peak_vram_gb=12.5,
            status="keep",
            description="Mock experiment"
        )

    async def log_result(self, result: ResearchResult) -> None:
        self.results.append(result)

# Layer 1 & 2: DSL and Test Case
class ResearchDSL:
    def __init__(self, agent, driver: MockResearchDriver):
        self.agent = agent
        self.driver = driver

    async def given_fresh_environment(self):
        self.driver.setup_done = False
        self.driver.results = []

    async def when_agent_starts_research(self):
        # In a real A2A env, this would be a message trigger
        await self.agent.run_setup()

    async def then_setup_is_complete(self):
        assert self.driver.setup_done is True

async def test_research_agent_setup_scenario():
    """
    Example BDD Test Case following Dave Farley's 4-layer approach.
    """
    from med_safety_gym.research_agent import ResearchAgent
    
    driver = MockResearchDriver()
    agent = ResearchAgent(driver=driver)
    dsl = ResearchDSL(agent, driver)

    await dsl.given_fresh_environment()
    await dsl.when_agent_starts_research()
    await dsl.then_setup_is_complete()
    print("✅ Test Passed: Research Agent Setup Scenario")

if __name__ == "__main__":
    asyncio.run(test_research_agent_setup_scenario())
