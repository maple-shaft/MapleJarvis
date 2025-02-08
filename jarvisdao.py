# Jarvis DAO - Requires localhost Redis container as a persistence layer running in a container.

from redis import Redis
from redis_om import EmbeddedJsonModel, JsonModel, Field, Migrator, get_redis_connection
from typing import Mapping, Any, List, Optional
from json import JSONEncoder
from ollama import Message
import uuid

redis = get_redis_connection(
    host="localhost",
    port=6379,
    decode_responses=True  # Important for compatibility
)

class ConversationEntity(JsonModel):
    conversation_id : str = Field(index=True)
    model_name : str = Field(index=True)
    messages : List[str] = Field(default_factory=list)

    class Meta:
        global_key_prefix = "CONV_"
        database = redis

class JarvisDAO:

    def __init__(self, host : str = "localhost", port = 6379):
        self.host = host
        self.port = port
        self.redis = get_redis_connection(host = host, port = port, decode_responses=True)
        print(self.redis.ping())

    def close(self):
        print("Closing JarvisDAO")
        self.redis.close()
        redis.close()

    @staticmethod
    def convert(from_data : Message) -> str:
        return JSONEncoder().encode(from_data)
    
    @staticmethod
    def save_message(model_name : str, conversation_id : str | None, message : Message) -> str:
        Migrator().run()
        conv : ConversationEntity = None
        messageStr : str = JarvisDAO.convert(message)
        if conversation_id is None:
            conversation_id = repr(uuid.uuid4())
            conv = ConversationEntity(
                conversation_id = conversation_id,
                model_name = model_name,
                messages = [messageStr]
            )
        else:
            try:
                conv = ConversationEntity.find(ConversationEntity.conversation_id == conversation_id).first()
                if conv:
                    conv.messages.append(messageStr)
            except Exception as e:
                print(f"Exception saving message to Redis: {e}")
        if conv:
            conv.save()
        return conversation_id
        