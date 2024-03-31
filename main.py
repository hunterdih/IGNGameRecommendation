import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from groq_model import *

model = GroqLanguageModel()

prompt = "Recommend an RPG video game with an amazing story"

print(model.get_non_rag_response(prompt))

print(f'-----------------------------------------------')
print(f'-------------------RAG-------------------------')

print(model.get_rag_response(prompt))


