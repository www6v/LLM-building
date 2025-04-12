#

from langchain import OpenAI

# The vectorstore we'll be using
from langchain.vectorstores import FAISS

# The LangChain component we'll use to get the documents
from langchain.chains import RetrievalQA

# The easy document loader for text
from langchain.document_loaders import TextLoader

# The embedding engine that will convert our text to vectors
from langchain.embeddings.openai import OpenAIEmbeddings

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

#
loader = TextLoader('data/PaulGrahamEssays/worked.txt')
doc = loader.load()
print (f"You have {len(doc)} document")
print (f"You have {len(doc[0].page_content)} characters in that document")

#

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
docs = text_splitter.split_documents(doc)


#
# Get the total number of characters so we can see the average later
num_total_characters = sum([len(x.page_content) for x in docs])

print (f"Now you have {len(docs)} documents that have an average of {num_total_characters / len(docs):,.0f} characters (smaller pieces)")


#
# Get your embeddings engine ready
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Embed your documents and combine with the raw text in a pseudo db. Note: This will make an API call to OpenAI
docsearch = FAISS.from_documents(docs, embeddings)


#
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

#

query = "What does the author describe as good work?"
qa.run(query)