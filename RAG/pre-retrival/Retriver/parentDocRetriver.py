# https://github.com/athina-ai/rag-cookbooks?tab=readme-ov-file
# https://colab.research.google.com/github/athina-ai/rag-cookbooks/blob/main/advanced_rag_techniques/parent_document_retriever.ipynb
# ref : https://python.langchain.com/docs/how_to/parent_document_retriever/
# todo: verify
import os

# # load embedding model
# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings()

from langchain_community.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# load data
# from langchain.document_loaders import CSVLoader
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("./context.csv", encoding='UTF-8')
documents = loader.load()



from langchain.text_splitter import RecursiveCharacterTextSplitter

# create the parent documents - The big chunks
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# create the child documents - The small chunks
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# The storage layer for the parent chunks
from langchain.storage import InMemoryStore
store = InMemoryStore()

# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
vectorstore = Chroma(collection_name="split_parents", embedding_function=embeddings, persist_directory="./chroma_db")


# create retriever
from langchain.retrievers import ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)


# add documents to vectorstore
retriever.add_documents(documents)



from langchain_openai import ChatOpenAI
# llm = ChatOpenAI()


llm = ChatOpenAI(model_name="deepseek-v3",
                 openai_api_key=os.getenv("DASHSCOPE_API_KEY"), 
                 openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1")  #### 


# create document chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

template = """"
You are a helpful assistant that answers questions based on the following context
Context: {context}

Question: {input}

Answer:

"""
prompt = ChatPromptTemplate.from_template(template)

# Setup RAG pipeline
rag_chain = (
    {"context": retriever,  "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# response
response = rag_chain.invoke("who played the lead roles in the movie leaving las vegas")
print(response)