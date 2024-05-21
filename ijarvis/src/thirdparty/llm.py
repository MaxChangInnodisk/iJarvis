from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor

from httpx import HTTPStatusError, RequestError

from thirdparty import core


class Mistral(core.AsyncApiModel):
    def __init__(self, host: str, port: str, route: str):
        super().__init__(host, port, route)

    def _input(self, content: str) -> dict:
        return {
            "Content-Type": "application/json",
            "messages": [{"role": "user", "content": content}],
            "mode": "chat",
            "character": "Example",
        }

    async def inference(self, content: str) -> str:
        try:
            response = await self.client.post(self.url, json=self._input(content))
            response.raise_for_status()  # Raise an HTTPStatusError for bad responses
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
        except HTTPStatusError as e:
            # Handle HTTP status errors
            return f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
        except RequestError as e:
            # Handle request errors (e.g., network problems)
            return f"Request error occurred: {str(e)}"
        except Exception as e:
            # Handle other possible exceptions
            return f"An unexpected error occurred: {str(e)}"


async def async_init_mistral(loop: AbstractEventLoop, executor: ThreadPoolExecutor):
    return await loop.run_in_executor(executor, Mistral)
