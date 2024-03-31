import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from groq_model_faiss import *

model = GroqLanguageModel()

prompt = "Recommend three RPG video games with a rating of 3 or higher"

model.add_to_db(r'Dataset/games.csv', reset_db=True)
print(model.get_non_rag_response(prompt))

print(f'-----------------------------------------------')
print(f'-------------------RAG-------------------------')

print(model.get_rag_response(prompt))


