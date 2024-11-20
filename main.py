from jarvisclient import JarvisClient
from jarvisgui import JarvisGui
from jarvisdao import JarvisDAO

print("Starting JarvisGui")

jarvisDAO = JarvisDAO()
jarvisClient = JarvisClient(dao = jarvisDAO, model = "rp")

#jarvisClient.pull()

jarvisClient.createModel(filepath = "./ollama/Blue-Orchid-2x7b-Q4_K_M.gguf", modelName = "rp")
jarvisClient.setModel(model = "rp")

g = JarvisGui(jarvisClient = jarvisClient)
if not g:
    print("Unable to setup the Jarvis client!")
    exit(1)
else:
    g.initializeGui()


