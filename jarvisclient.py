from jarvisdao import JarvisDAO
from jarvisvoice import JarvisVoice
from ollama import Client
from ollama._types import (ResponseError, Message, Sequence)
from typing import List, Optional
import json

class JarvisClient:

    def __init__(self, dao : JarvisDAO, model : str = "llama3.2", conversation_id : Optional[str] = None, isAsync : bool = False):
        self.dao = dao
        self.model = model
        self.current_conversation = conversation_id
        self.isAsync = isAsync
        if not isAsync:
            self.client = Client()
        self.messages = []
        self.voice = JarvisVoice()
    
    def clearConversation(self):
        self.messages = []
        self.current_conversation = None

    def setModel(self, model : str):
        self.model = model

    def setSystemMessage(self, system_prompt : str):
        self.clearConversation()
        systemMessage : Message = self._craftMessage(text = system_prompt, role = "system")
        self.messages.append(systemMessage)

    @staticmethod
    def createModel(modelName : str):
        Client("http://localhost:11434").create(model = modelName, path = f"./models/{modelName}")

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
        conv_length : int = self.messages.__len__()
        print(f"Conversation length: {conv_length}")
        for i,v in enumerate(self.messages):
            print(f"Conversation Index[{i}], Value[{v}]")

        userPromptMessage : Message = self._craftMessage(text = prompt, role = "user")
        self.messages.append(userPromptMessage)
        self.current_conversation = JarvisDAO.save_message(model_name = self.model, conversation_id = self.current_conversation, message = userPromptMessage)
        try:
            resp = self.client.chat(model=self.model, messages=self.messages, keep_alive=50)
            if resp:
                for r in resp:
                    print(f"Response[{r}] = {resp[r]}")
                assistantMessage : Message = resp["message"]
                self.current_conversation = JarvisDAO.save_message(
                    model_name = self.model,
                    message = assistantMessage,
                    conversation_id=self.current_conversation)
            else:
                print("No response returned")
            
            # Speak the text
            self.voice.speak(resp["message"]["content"])
            return resp["message"]["content"]
        except ResponseError as re:
            print(f"Response error in prompt: {re}")

    def embed(self, document : str) -> Sequence[float]:
        response = self.client.embeddings(model = self.model, prompt = document)
        if response:
            return response["embedding"]
        
    def generate(self, prompt : str, embedding_data):
        embedded_prompt : str = f"Using this data: {embedding_data}. Respond to this prompt: {prompt}"
        output = self.client.generate(model = self.model, prompt = embedded_prompt)
        if output:
            return output["response"]