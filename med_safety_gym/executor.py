import asyncio
import time
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Task,
    TaskState,
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils.errors import ServerError
from a2a.utils import (
    new_agent_text_message,
    new_task,
)

TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected
}

class Executor(AgentExecutor):
    def __init__(self, agent_class, idle_ttl=300): # 5 minutes default
        self.agents = {} # context_id to agent instance
        self.last_activity = {} # context_id to timestamp
        self.agent_class = agent_class
        self.idle_ttl = idle_ttl
        self.reaper_task = None

    async def start(self):
        """Start background services."""
        if self.reaper_task is None:
            self.reaper_task = asyncio.create_task(self._reaper_loop())

    async def _reaper_loop(self):
        """Background task that shuts down idle agents."""
        while True:
            await asyncio.sleep(60) # Check every minute
            now = time.time()
            to_reap = [
                ctx_id for ctx_id, last_seen in self.last_activity.items()
                if now - last_seen > self.idle_ttl
            ]
            for ctx_id in to_reap:
                print(f"Reaping idle agent: {ctx_id}")
                await self._shutdown_single_agent(ctx_id)

    async def shutdown(self):
        """Shutdown all active agents concurrently."""
        if self.reaper_task:
            self.reaper_task.cancel()
            
        shutdown_tasks = [
            agent.shutdown() for agent in self.agents.values() if hasattr(agent, "shutdown")
        ]
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks)
        self.agents.clear()
        self.last_activity.clear()

    async def _shutdown_single_agent(self, context_id: str):
        """Gracefully shutdown a single agent."""
        agent = self.agents.pop(context_id, None)
        self.last_activity.pop(context_id, None)
        if agent and hasattr(agent, "shutdown"):
            await agent.shutdown()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(error=InvalidRequestError(message=f"Task {task.id} already processed (state: {task.status.state})"))

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        self.last_activity[context_id] = time.time()
        
        agent = self.agents.get(context_id)
        if not agent:
            # Cold Start
            print(f"Cold Start: Spawning agent for {context_id}")
            agent = self.agent_class()
            self.agents[context_id] = agent

        updater = TaskUpdater(event_queue, task.id, context_id)

        await updater.start_work()
        try:
            await agent.run(msg, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception as e:
            print(f"Task failed with agent error: {e}")
            await updater.failed(new_agent_text_message(f"Agent error: {e}", context_id=context_id, task_id=task.id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
