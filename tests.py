from jarvisclient import JarvisClient
from jarvisdao import JarvisDAO
import asyncio
from typing import Any, AsyncGenerator

jdao = JarvisDAO()
jc = JarvisClient(dao = jdao, model = "rp", conversation_id=None, isAsync=True)

print("Successfully initialized the asynchronous Jarvis Client")

async def output_parts(prompt : str):
    asyncgen : AsyncGenerator[Any,Any] = jc.prompt_async(prompt=prompt)
    async for part in asyncgen:
        print(f"Part = {part}")

asyncio.run(output_parts("Good morning!  What can you tell me about the Great Wall of China?"))

print("End")
