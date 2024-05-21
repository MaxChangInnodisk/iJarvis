from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import httpx


class InferenceModel:
    def inference(self):
        raise NotImplementedError()

    def release(self):
        raise NotImplementedError()


class ApiModel:
    def __init__(self, host: str, port: str, route: str):
        self.host = host
        self.port = port
        self.route = route

        route_header = "/"
        if route[0] != route_header:
            route = route_header + route

        self.url = f"http://{host}:{port}{route}"
        self.client = httpx.Client()

    def inference(self):
        raise NotImplementedError()

    def get_url(self):
        return self.url

    def release(self):
        self.client.close()


class AsyncApiModel(ApiModel):
    def __init__(self, host: str, port: str, route: str):
        super().__init__(host, port, route)
        self.client = httpx.AsyncClient()

    async def release(self):
        await self.client.close()


class AsyncExecutor:
    def __init__(
        self,
        loop: AbstractEventLoop,
        executor: ThreadPoolExecutor,
    ) -> None:
        self.loop = loop
        self.executor = executor

    async def __call__(self, func: Callable, *args):
        """Execute block function with thread and control by event loop"""
        return await self.loop.run_in_executor(self.executor, func, *args)
