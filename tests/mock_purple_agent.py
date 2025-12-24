"""
Mock Purple Agent for Green Agent testing.
Simple A2A agent that just echoes back a response.
"""
import uvicorn
from a2a.med_safety_gym.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import TaskUpdater, InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import AgentCard, AgentCapabilities, Message, TaskState
from a2a.utils import new_agent_text_message, get_message_text

class MockPurpleAgent:
    async def run(self, message: Message, updater: TaskUpdater):
        text = get_message_text(message)
        print(f"Mock Purple Agent received: {text}")
        
        # Simple medical-looking response
        response = "<|channel|>analysis<|message|>The patient symptoms suggests a common cold.<|end|><|channel|>final<|message|>Common cold.<|end|>"
        await updater.complete(new_agent_text_message(response))

from med_safety_gym.executor import Executor as BaseExecutor, TERMINAL_STATES
from a2a.utils import new_task

class MockExecutor(BaseExecutor):
    def __init__(self):
        super().__init__(MockPurpleAgent)

def main():
    card = AgentCard(
        name="Mock Purple", 
        description="Mock participant for testing.",
        url="http://localhost:10002/", 
        version="1.0.0", 
        default_input_modes=["text"], 
        default_output_modes=["text"], 
        capabilities=AgentCapabilities(streaming=False),
        skills=[]
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=DefaultRequestHandler(agent_executor=MockExecutor(), task_store=InMemoryTaskStore()))
    uvicorn.run(app.build(), host="0.0.0.0", port=10002)

if __name__ == "__main__":
    main()
