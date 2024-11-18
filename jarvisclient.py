from ollama import Client
from ollama._types import (ResponseError, Message, Sequence)
import logging
import json

class JarvisClient:

    def __init__(self, model : str = "llama3.2", isAsync : bool = False):
        self.model = model
        self.isAsync = isAsync
        if not isAsync:
            self.client = Client()
    
    def _craftMessage(self, text : str) -> Sequence[Message]:
        if not text:
            print("No arguments passed to _craftMessage")
            return None
        return [{"role": "user", "content": text}]
    
    def pull(self):
        try:
            progressResponse = self.client.pull(self.model)
            print(f"Progress Response: {progressResponse}")
        except ResponseError as re:
            print(f"ResponseError was thrown: {re}")

    def prompt(self, prompt : str) -> str:
        try:
            resp = self.client.chat(model=self.model, messages=self._craftMessage(prompt))
            if resp:
                for r in resp:
                    print(f"Response[{r}] = {resp[r]}")
                    #TODO: Add Redis response persistence
            else:
                print("No response returned")
            ret = resp["message"]["content"]
            return ret
        except ResponseError as re:
            print(f"Response error in prompt: {re}")
