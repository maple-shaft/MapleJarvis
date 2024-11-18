from guizero import App, Text, PushButton, TextBox, Box
from jarvisclient import JarvisClient

class JarvisGui:

    def __init__(self, jarvisClient : JarvisClient):
        self.jarvisClient = jarvisClient
    
    def submitPrompt(self):
        try:
            promptText = self.promptInput.value
            promptResponse = self.jarvisClient.prompt(promptText)
            self.response.value = promptResponse
        except Exception as e:
            self.response.value = repr(e)

    def submitEmbedding(self):
        print("Not implemented yet")

    def initializeGui(self):
        self.app = App(title = "Maple Jarvis Control Center", width = 1024, height = 760)
        self.leftpadding = Box(master = self.app, width = 10, height = "fill", align = "left")
        self.rightpadding = Box(master = self.app, width = 10, height = "fill", align = "right")
        self.centerbox = Box(master = self.app, align = "top", width = "fill", height = "fill")
        