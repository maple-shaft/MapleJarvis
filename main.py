from jarvisclient import JarvisClient
from jarvisdao import JarvisDAO
from jarvisdocuments import JarvisDocuments
from threading import Event
import time
import asyncio

print("Starting JarvisGui")

#jarvisDocuments = JarvisDocuments(jarvis_client = jarvisClient, documents = JarvisDocuments.generateExampleDocuments())
#doc_vector_embeddings = jarvisDocuments.createDocumentVectorEmbeddings()
#relevant_docs = jarvisDocuments.findRelevantDocuments(col = doc_vector_embeddings, prompt = "Who is Rick Stevenson?")

print("Relevant Documents: ")

#convs = jarvisDAO.get_conversation("rp")
#for i in convs:
#    print(f"ConversationEntity {i}")

MODEL = "freddy_fazbear_alt"

async def main():
    print("Start main")
    shutdown_event = Event()
    jarvisDAO = JarvisDAO()
    jarvisClient = JarvisClient(dao = jarvisDAO, model = MODEL, isAsync=True, system_prompt="You are Freddy Fazbear from the Five Nights at Freddys franchise.")

    while not shutdown_event.is_set():
        try:
            time.sleep(1.0)
            user_prompt : str = input(f"{MODEL} > ")
            #response_text = jarvisClient.prompt(user_prompt)
            async for token in jarvisClient.prompt_async(user_prompt):
                print(token, end="")
        except KeyboardInterrupt:
            print("Keyboard interrupt encountered.")
            shutdown_event.set()


if __name__ == "__main__":
    asyncio.run(main())


