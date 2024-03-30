import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
chat = ChatGroq(temperature=0, groq_api_key = GROQ_API_KEY, model_name='mixtral-8x7b-32768')


system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
chain.invoke({"text": "Explain the importance of low latency LLMs."})


