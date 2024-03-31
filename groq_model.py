import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from ctypes import windll
from pinecone import Pinecone
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_cohere.embeddings import CohereEmbeddings

from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os


class GroqLanguageModel():
    def __init__(self, max_tokens='4098'):
        GROQ_API_KEY = os.environ['GROQ_API_KEY']
        COHERE_API_KEY = os.environ['COHERE_API_KEY']
        PC_API_KEY = os.environ['PINECONE_API_KEY']
        self.index_name = 'general-index'
        self.embed_model_name = 'embed-multilingual-v3.0'

        self.llm = ChatGroq(temperature=0,
                             groq_api_key=GROQ_API_KEY,
                             model_name='mixtral-8x7b-32768',
                             max_tokens=max_tokens)

        self.embedding_model = CohereEmbeddings(model=self.embed_model_name,
                                                cohere_api_key=COHERE_API_KEY)

        self.pinecone_db = PineconeVectorStore(pinecone_api_key=PC_API_KEY,
                                               embedding=self.embedding_model,
                                               index_name = self.index_name)

        self.text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        self.output_parser = StrOutputParser()

        self.system = "You are a helpful assistant"
        self.human = "{text}"
        self.prompt = ChatPromptTemplate.from_messages([("system", self.system), ("human", self.human)])
        self.response = None


        self.rag_system = """Answer the following question based on the provided context. 
        Do not state that prior context was given to this question:
        
        <context>
        {context}
        </context>
        
        Question: {input}"""
        self.rag_prompt = ChatPromptTemplate.from_template(self.rag_system)
        self.document_chain = create_stuff_documents_chain(self.llm, self.rag_prompt)
        self.retriever = self.pinecone_db.as_retriever()
        self.rag_chain = create_retrieval_chain(self.retriever, self.document_chain)
        self.non_rag_chain = self.prompt | self.llm | self.output_parser
        self.rag_response = None

    def set_system_prompt(self, prompt):
        self.system = prompt

    def get_non_rag_response(self, prompt):
        self.response = self.non_rag_chain.invoke({"text": prompt})
        return self.response

    def get_rag_response(self, prompt):
        self.rag_response = self.rag_chain.invoke({"input": prompt})["answer"]
        return self.rag_response