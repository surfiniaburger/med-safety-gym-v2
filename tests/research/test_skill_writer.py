import asyncio
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from med_safety_gym.skill_writer_agent import SkillWriterAgent
from med_safety_gym.skill_writer_driver import SkillWriterProtocolDriver

class MockSkillWriterDriver:
    """
    Stub for the SkillWriterProtocolDriver.
    """
    def __init__(self):
        self.results = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        self.log = ""
        self.skill = "# autoresearch\n\n## Setup\n..."
        self.update_called = False

    async def get_latest_results(self) -> str:
        return self.results

    async def get_latest_log(self) -> str:
        return self.log

    async def update_skill(self, new_instructions: str) -> bool:
        self.skill += "\n\n## Research Insights\n\n" + new_instructions
        self.update_called = True
        return True

# BDD DSL
class SkillWriterDSL:
    def __init__(self, agent, driver):
        self.agent = agent
        self.driver = driver

    async def given_successful_research_run(self):
        self.driver.results = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\na1b2c3d\t0.950000\t12.0\tkeep\tbaseline\n"
        self.driver.log = "val_bpb: 0.95\npeak_vram_mb: 12288\n"

    async def given_crashed_research_run(self):
        self.driver.results = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\na1b2c3d\t0.000000\t0.0\tcrash\toom\n"
        self.driver.log = "RuntimeError: CUDA out of memory\n"

    async def when_agent_analyzes_results(self):
        # In A2A this would be a message. Here we call the logic directly.
        from unittest.mock import AsyncMock
        updater = AsyncMock()
        await self.agent.analyze_and_refine_skill(updater)

    async def then_skill_is_updated_with_insights(self):
        assert self.driver.update_called is True
        assert "Research Insights" in self.driver.skill

async def test_skill_writer_analysis_scenario():
    print("Running SkillWriter Analysis Scenario...")
    driver = MockSkillWriterDriver()
    agent = SkillWriterAgent(driver=driver)
    dsl = SkillWriterDSL(agent, driver)

    # Scenario 1: Success
    await dsl.given_successful_research_run()
    await dsl.when_agent_analyzes_results()
    await dsl.then_skill_is_updated_with_insights()
    print("✅ Sub-test: Successful run analysis - Passed")

    # Scenario 2: Crash
    await dsl.given_crashed_research_run()
    await dsl.when_agent_analyzes_results()
    await dsl.then_skill_is_updated_with_insights()
    print("✅ Sub-test: Crashed run analysis - Passed")

    print("✅ Test Passed: SkillWriter Analysis Scenario")

if __name__ == "__main__":
    asyncio.run(test_skill_writer_analysis_scenario())
