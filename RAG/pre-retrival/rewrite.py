### https://github.com/athina-ai/rag-cookbooks?tab=readme-ov-file
import os

# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings()

# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# print(embeddings)
print(embeddings)



######   Indexing
# load data
# from langchain.document_loaders import CSVLoader
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("./context.csv")
documents = loader.load()


# split documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(documents)


# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")


retriever = vectorstore.as_retriever()


from langchain_openai import ChatOpenAI
# llm = ChatOpenAI()  #### 


###### 报404异常
# llm = ChatOpenAI(model_name="deepseek/deepseek-r1:free",
#                  openai_api_key=os.getenv("OPENROUTER_API_KEY"), 
#                  openai_api_base="https://openrouter.ai/api/v1/chat/completions")  #### 
# print(os.getenv("OPENROUTER_API_KEY"))

######  可以调通，没钱了
# llm = ChatOpenAI(model_name="deepseek-reasoner",
#                  openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
#                  openai_api_base="https://api.deepseek.com")  #### 
# print(os.getenv("DEEPSEEK_API_KEY"))   

llm = ChatOpenAI(model_name="deepseek-v3",
                 openai_api_key=os.getenv("DASHSCOPE_API_KEY"), 
                 openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1")  #### 


# create document chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

template = """"
You are a helpful assistant that answers questions based on the following context.
If you don't find the answer in the context, just say that you don't know.
Context: {context}

Question: {input}

Answer:

"""
prompt = ChatPromptTemplate.from_template(template)
print(prompt)

rag_chain = (
    {"context": retriever,  "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)



# define simple query
simple_query = "who directed the matrix"
response = rag_chain.invoke(simple_query)
print(response)
### The directors of *The Matrix* are the Wachowskis.



# define distracted query
distracted_query = "who create the matrics"
response = rag_chain.invoke(distracted_query)
print(response)
### The Matrix was created by the Wachowskis. They are the directors of the film.


### Rewrite Retrieve Read
# define rewrite prompt for distracted query

template = """Provide a better search query for \
web search engine to answer the given question, end \
the queries with ’**’. Question: \
{x} Answer:"""

rewrite_prompt = ChatPromptTemplate.from_template(template)


# parse response
def _parse(text):
    return text.strip('"').strip("**")


# rewriter = rewrite_prompt | ChatOpenAI(temperature=0) | StrOutputParser() | _parse
rewriter = rewrite_prompt | llm | StrOutputParser() | _parse

# updated query
distracted_query=rewriter.invoke({"x": distracted_query})
print(distracted_query)
### Who created the matrix concept in mathematics?**" or "Who is the originator of the matrix theory?

# create rewrite retrieve read chain
rewrite_retrieve_read_chain = (
    {
        "context": {"x": RunnablePassthrough()} | rewriter | retriever,
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# final response
res=rewrite_retrieve_read_chain.invoke(distracted_query)
print(res)
### The context provided does not contain information about the originator or creator of the matrix concept in mathematics. Therefore, I don't know the answer to that question based on the given context. 