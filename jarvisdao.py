# Jarvis DAO - Requires localhost Redis container as a persistence layer running in a container.

from redis import Redis
from typing import Mapping, Any

class JarvisDAO:

    MODEL_KEY = "model"
    CONVERSATION_KEY = "conversation"
    MESSAGE_KEY = "message"
    SEPARATOR = ":"

    def __init__(self, host : str = "localhost", port = 6379):
        self.host = host
        self.port = port
        self.client = Redis(host = host, port = port, decode_responses = True)

    def __craftKey(self, model : str, conversationId : str, messageId : str) -> str:
        return model + JarvisDAO.SEPARATOR + conversationId + JarvisDAO.SEPARATOR + messageId

    def saveOrUpdateChat(self, chatData : Mapping[str, Any], conversationId : str):
        if not chatData:
            print("Null passed to saveChat")
            return
        model = "llama3.2"
        createdAt = None
        if JarvisDAO.MODEL_KEY in chatData:
            model = chatData[JarvisDAO.MODEL_KEY]
        if "created_at" in chatData:
            createdAt = chatData["created_at"]
        if JarvisDAO.MESSAGE_KEY not in chatData:
            print("No message found in chat data")
            return
        
        messageValue = repr(chatData[JarvisDAO.MESSAGE_KEY])
        prefix = JarvisDAO.__craftKey(self, model, conversationId, createdAt)
        
        ret = self.client.hset(name = model, key = prefix, value = messageValue)
        print(f"Redis hset returned status code: {ret}")



        

        