from guizero import App, Text, PushButton, TextBox, Box
from jarvisclient import JarvisClient

class JarvisGui:

    TITLE = "Maple Jarvis Control Center"

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
        self.app = App(title = JarvisGui.TITLE, width = 1024, height = 760)
        self.leftpadding = Box(master = self.app, width = 10, height = "fill", align = "left")
        self.rightpadding = Box(master = self.app, width = 10, height = "fill", align = "right")
        self.centerbox = Box(master = self.app, align = "top", width = "fill", height = "fill")
        self.padding1 = Box(master = self.centerbox, height = 10, width = "fill", align = "top")
        self.header = Text(master = self.centerbox, text = JarvisGui.TITLE, align = "top")
        self.padding2 = Box(master = self.centerbox, height = 10, width = "fill", align = "top")
        self.description = Text(master = self.centerbox, text = "Enter a prompt below and then click Submit to view the LLM output.", align = "top")
        self.padding3 = Box(master = self.centerbox, height = 10, width = "fill", align = "top")
        self.promptInput = TextBox(master = self.centerbox, width = "fill", multiline = True, scrollbar = True, align = "top")
        self.padding4 = Box(master = self.centerbox, height = 10, width = "fill", align = "top")
        self.submit = PushButton(master = self.centerbox, text = "Submit Prompt", command = self.submitPrompt, padx = 10, pady = 10, align = "top")
        self.padding5 = Box(master = self.centerbox, height = 10, width = "fill", align = "top")
        self.submitembed = PushButton(master = self.centerbox, text = "Submit Embedding", command = self.submitEmbedding, padx = 10, pady = 10, align = "top")
        self.padding6 = Box(master = self.centerbox, height = 10, width = "fill", align = "top")
        self.response = TextBox(master = self.centerbox, width = "fill", height = "fill", align = "top", multiline = True, scrollbar = True)
        self.app.display()