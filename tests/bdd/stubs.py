from unittest.mock import AsyncMock, MagicMock


def make_llm_response(text: str):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    return response


def make_experience_client():
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.call_tool = AsyncMock(return_value="ok")
    return client


class CaptureUpdater:
    def __init__(self):
        self.events = []

    async def update_status(self, state, message, metadata=None):
        self.events.append(
            {
                "state": state.name if hasattr(state, "name") else str(state),
                "text": message.parts[0].root.text,
                "metadata": metadata or {},
            }
        )
