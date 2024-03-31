import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from ctypes import windll

class GroqLanguageModel():
    def __init__(self, max_tokens = '4098'):
        GROQ_API_KEY = os.environ['GROQ_API_KEY']
        self.chat = ChatGroq(temperature=0,
                             groq_api_key = GROQ_API_KEY,
                             model_name='mixtral-8x7b-32768',
                             max_tokens = max_tokens)
        self.system = "You are a helpful assistant"
        self.human = "{text}"
        self.prompt = ChatPromptTemplate.from_messages([("system", self.system), ("human", self.human)])
        self.output_parser = StrOutputParser()
        self.response = "None"

        self.non_rag_chain = self.prompt | self.chat | self.output_parser

    def set_system_prompt(self, prompt):
        self.system = prompt

    def get_non_rag_response(self, prompt):
        self.response = self.non_rag_chain.invoke({"text": prompt})
        return self.response




