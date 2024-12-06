import ollama
from ollama._types import (ResponseError, Message, Sequence)
import chromadb
from typing import List, Optional
import json

from jarvisclient import JarvisClient

class JarvisDocuments:

    def __init__(self, jarvis_client : JarvisClient, documents : List[str] = []):
        self.documents = documents
        self.jarvis_client = jarvis_client
        self.client = chromadb.Client()
        
    @staticmethod
    def generateExampleDocuments() -> List[str]:
        ret = [
            "You are a Hollywood screen writer.",
            "You are tasked with writing a script for a new Batman movie.",
            "The producers name is Rick Stevenson.",
            "Your name is Jello Biafra.",
            "You have until January 2025 to complete a draft for your script.",
            "A fellow writer is helping you, named Steve Rickinson."
        ]
        return ret
    
    def createDocumentVectorEmbeddings(self) -> chromadb.Collection:
        col : chromadb.Collection = self.client.create_collection(name = "hollywood", get_or_create=True)
        
        for i, d in enumerate(self.documents):
            embedding : Sequence[float] = self.jarvis_client.embed(d)
            if embedding:
                col.add(ids=[str(i)], embeddings = [embedding], documents = [d])
        return col
    
    def findRelevantDocuments(self, col : chromadb.Collection, prompt : str):
        embedding : Sequence[float] = self.jarvis_client.embed(prompt)
        results = col.query(query_embeddings=[embedding], n_results=1)
        if results:
            return results["documents"][0][0]

