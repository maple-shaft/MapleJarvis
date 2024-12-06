from jarvisclient import JarvisClient
from jarvisgui import JarvisGui
from jarvisdao import JarvisDAO
from jarvisdocuments import JarvisDocuments

print("Starting JarvisGui")

jarvisDAO = JarvisDAO()
jarvisClient = JarvisClient(dao = jarvisDAO, model = "rp")
jarvisClient.createModel(modelName = "rp")
jarvisClient.setModel(model = "rp")
#jarvisDocuments = JarvisDocuments(jarvis_client = jarvisClient, documents = JarvisDocuments.generateExampleDocuments())
#doc_vector_embeddings = jarvisDocuments.createDocumentVectorEmbeddings()
#relevant_docs = jarvisDocuments.findRelevantDocuments(col = doc_vector_embeddings, prompt = "Who is Rick Stevenson?")

print("Relevant Documents: ")

#convs = jarvisDAO.get_conversation("rp")
#for i in convs:
#    print(f"ConversationEntity {i}")

g = JarvisGui(jarvisClient = jarvisClient)
if not g:
    print("Unable to setup the Jarvis client!")
    exit(1)
else:
    g.initializeGui()


