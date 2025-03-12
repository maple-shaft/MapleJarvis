from jarvisdao import JarvisDAO
from jarvisvoice import JarvisVoice
from ollama import Client, AsyncClient
from ollama._types import (ResponseError, Message, Sequence)
from typing import List, Optional
import asyncio

DEFAULT_SYSTEM_PROMPT : str = "You are a helpful AI assistant."

class JarvisClient:

    def __init__(self, dao : JarvisDAO, model : str = "llama3.2", host = "http://localhost:11434",
                 conversation_id : Optional[str] = None, isAsync : bool = False, voice = JarvisVoice(),
                 system_prompt : str = DEFAULT_SYSTEM_PROMPT):
        self.dao = dao
        self.model = model
        self.current_conversation = conversation_id
        self.isAsync = isAsync
        if not isAsync:
            self.client = Client(host=host)
        else:
            self.client = AsyncClient(host=host)
        self.messages = []
        self.voice = voice
        self.setSystemMessage(system_prompt=system_prompt)
    
    def clearConversation(self):
        self.messages = []
        self.current_conversation = None

    def setModel(self, model : str):
        self.model = model

    def setSystemMessage(self, system_prompt : str):
        self.clearConversation()
        self.system_prompt = system_prompt
        systemMessage : Message = self._craftMessage(text = system_prompt, role = "system")
        self.messages.append(systemMessage)

    @staticmethod
    def createModel(modelName : str, host : str = "http://localhost:11434"):
        Client(host=host).create(model = modelName, path = f"./models/{modelName}")

    def _craftMessage(self, text : str, role : str) -> Message:
        if not text or not role:
            print("No arguments passed to _craftMessage")
            return None
        return {"role": role, "content": text}
    
    def _assignMessage(self, text : str, role : str):
        mess = self._craftMessage(text, role)
        if mess:
            self.messages.put(mess)
    
    def pull(self):
        try:
            progressResponse = self.client.pull(self.model)
            print(f"Progress Response: {progressResponse}")
        except ResponseError as re:
            print(f"ResponseError was thrown: {re}")

    def _start_prompt(self, prompt : str):
        conv_length : int = self.messages.__len__()
        print(f"Conversation length: {conv_length}")
        for i,v in enumerate(self.messages):
            print(f"Conversation: Index[{i}], Value[{v}]")

        userPromptMessage : Message = self._craftMessage(text = prompt, role = "user")
        self.messages.append(userPromptMessage)
        self.current_conversation = JarvisDAO.save_message(model_name = self.model, conversation_id = self.current_conversation, message = userPromptMessage)

    def prompt(self, prompt : str) -> str:
        """Prompt the model with text, and return a str response."""
        self._start_prompt(prompt=prompt)
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
            return resp["message"]["content"]
        except ResponseError as re:
            print(f"Response error in prompt: {re}")

    def prompt_audio(self, prompt : str, play : bool = False) -> bytes:
        """Prompt the model with text and return a bytes response as audio data."""
        prompt_text = self.prompt(prompt=prompt)
        if prompt_text:
            return self.voice.speak(prompt_text, play = play)
        
    async def prompt_async(self, prompt : str):
        """Asynchronous prompt response generation of text"""
        if not self.isAsync:
            raise Exception("Asynchronous client is not initialized.")
        
        self._start_prompt(prompt=prompt)
        try:
            async for part in await self.client.chat(model=self.model, messages=self.messages, stream=True):
                if part and part["message"] and part["message"]["content"]:
                    part_word = part["message"]["content"]
                    #print(f"About to yield part: {part_word}")
                    yield part_word
        except ResponseError as re:
            print(f"Response error in prompt: {re}")
            pass

    def embed(self, document : str) -> Sequence[float]:
        response = self.client.embeddings(model = self.model, prompt = document)
        if response:
            return response["embedding"]
        
    def generate(self, prompt : str, embedding_data):
        embedded_prompt : str = f"Using this data: {embedding_data}. Respond to this prompt: {prompt}"
        output = self.client.generate(model = self.model, prompt = embedded_prompt)
        if output:
            return output["response"]