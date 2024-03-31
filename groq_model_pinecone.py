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
    def __init__(self, max_tokens='4098',
                 llm_model_name='mixtral-8x7b-32768',
                 embed_model_name='embed-multilingual-v3.0',
                 index_name='general-index'):
        self.max_tokens = max_tokens
        self.llm_model_name = llm_model_name
        self.embed_model_name = embed_model_name
        self.index_name = index_name
        self.pinecone_db = None

        self.llm = None
        self.embedding_model = None

        self.system = "You are a helpful assistant"
        self.human = "{text}"
        self.prompt = ChatPromptTemplate.from_messages([("system", self.system), ("human", self.human)])
        self.output_parser = None
        self.non_rag_chain = None
        self.response = None

        self.rag_prompt = None
        self.retriever = None
        self.document_chain = None
        self.rag_system = """Answer the following question based on the provided context. 
                        Do not state that prior context was given to this question:

                        <context>
                        {context}
                        </context>

                        Question: {input}"""
        self.rag_chain = None
        self.rag_response = None

        self.text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

        self.initialize_llm_model()
        self.initialize_embed_model()
        self.initialize_db()

    def initialize_db(self):
        pc_api_key = os.environ['PINECONE_API_KEY']
        self.pinecone_db = PineconeVectorStore(pinecone_api_key=pc_api_key,
                                               embedding=self.embedding_model,
                                               index_name=self.index_name)

    def initialize_embed_model(self):
        cohere_api_key = os.environ['COHERE_API_KEY']
        self.embedding_model = CohereEmbeddings(model=self.embed_model_name,
                                                cohere_api_key=cohere_api_key)

    def initialize_llm_model(self):
        groq_api_key = os.environ.get('GROQ_API_KEY')
        self.llm = ChatGroq(temperature=0,
                            groq_api_key=groq_api_key,
                            model_name='mixtral-8x7b-32768',
                            max_tokens=self.max_tokens)

        self.output_parser = StrOutputParser()
        # Initialize all non-rag chain components
        self.prompt = ChatPromptTemplate.from_messages([("system", self.system), ("human", self.human)])
        self.non_rag_chain = self.prompt | self.llm | self.output_parser

        # Initialize all rag chain components
        self.rag_prompt = ChatPromptTemplate.from_template(self.rag_system)
        self.document_chain = create_stuff_documents_chain(self.llm, self.rag_prompt)
        self.retriever = self.pinecone_db.as_retriever()
        self.rag_chain = create_retrieval_chain(self.retriever, self.document_chain)

    def set_model(self, model_name):
        if model_name == 'mixtral-8x7b':
            self.llm_model_name = 'mixtral-8x7b-32768'
        elif model_name == 'llama-70b':
            self.llm_model_name = 'llama2-70b-4096'
        elif model_name == 'gemma-7b':
            self.llm_model_name = 'gemma-7b-it'
        else:
            self.llm_model_name = 'mixtral-8x7b-32768'

        self.initialize_llm_models()

    def set_system_prompt(self, prompt):
        self.system = prompt
        self.initialize_llm_models()

    def get_non_rag_response(self, prompt):
        self.response = self.non_rag_chain.invoke({"text": prompt})
        return self.response

    def get_rag_response(self, prompt):
        self.rag_response = self.rag_chain.invoke({"input": prompt})["answer"]
        return self.rag_response
