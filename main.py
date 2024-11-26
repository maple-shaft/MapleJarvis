from jarvisclient import JarvisClient
from jarvisgui import JarvisGui
from jarvisdao import JarvisDAO

print("Starting JarvisGui")

jarvisDAO = JarvisDAO()
jarvisClient = JarvisClient(dao = jarvisDAO, model = "Mario")

#jarvisClient.pull()

jarvisClient.createModel(modelName = "Mario")
jarvisClient.setModel(model = "Mario")

#convs = jarvisDAO.get_conversation("rp")
#for i in convs:
#    print(f"ConversationEntity {i}")

g = JarvisGui(jarvisClient = jarvisClient)
if not g:
    print("Unable to setup the Jarvis client!")
    exit(1)
else:
    g.initializeGui()


