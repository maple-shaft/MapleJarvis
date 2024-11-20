from jarvisdao import JarvisDAO
from ollama import Client
from ollama._types import (ResponseError, Message, Sequence)
import logging
import json

class JarvisClient:

    def __init__(self, dao : JarvisDAO, model : str = "llama3.2", isAsync : bool = False):
        self.dao = dao
        self.model = model
        self.isAsync = isAsync
        if not isAsync:
            self.client = Client()
        self.messages = [{"role":"system", "content":"You are an AI assistant."}]
    
    def setModel(self, model : str):
        if model:
            self.model = model

    def createModel(self, filepath : str, modelName : str):
        modelfile = f'''
            FROM {filepath}
            SYSTEM You are mario from super mario bros.
            '''
        print(modelfile)
        self.client.create(model = modelName, modelfile = modelfile)

    def _craftMessage(self, text : str, role : str) -> Message:
        if not text or not role:
            print("No arguments passed to _craftMessage")
            return None
        return {"role": role, "content": text}
    
    def _assignMessage(self, text : str, role : str):
        mess = self._craftMessage(text, role)
        if mess:
            self.messages.append(mess)
    
    def pull(self):
        try:
            progressResponse = self.client.pull(self.model)
            print(f"Progress Response: {progressResponse}")
        except ResponseError as re:
            print(f"ResponseError was thrown: {re}")

    def prompt(self, prompt : str) -> str:
        try:
            self._assignMessage(prompt, "user")
            resp = self.client.chat(model=self.model, messages=self.messages, keep_alive=50)
            if resp:
                self.messages.append(resp["message"])
                for r in resp:
                    print(f"Response[{r}] = {resp[r]}")
                    #TODO: Add Redis response persistence
            else:
                print("No response returned")
            ret = resp["message"]["content"]
            return ret
        except ResponseError as re:
            print(f"Response error in prompt: {re}")
