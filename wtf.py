from redis_om import HashModel, get_redis_connection, JsonModel, Migrator, Field
from typing import List

# Establish Redis connection
redis = get_redis_connection(
    host="localhost",
    port=6379,
    decode_responses=True
)

# Define the model
class MyEntity(JsonModel):
    name: str = Field(index = True)
    age: int
    active: bool
    messages : List[str]

    class Meta:
        database = redis

# Create an instance of the model
Migrator().run()
try:
    entity = MyEntity(name="Alice", age=30, active=True, messages = [])
    if entity:
        entity.save()
    fq = MyEntity.find(MyEntity.name == "Alice")
    if fq:
        ls = fq.all()
    print(ls)
except Exception as e:
    print("Error:", e)
