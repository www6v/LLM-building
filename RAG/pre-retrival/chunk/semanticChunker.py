# https://python.langchain.com/docs/how_to/semantic-chunker/

with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()



from langchain_community.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai.embeddings import OpenAIEmbeddings
# text_splitter = SemanticChunker(OpenAIEmbeddings())

text_splitter = SemanticChunker(embeddings, 
                                breakpoint_threshold_type="interquartile")
# breakpoint_threshold_type="percentile"
# breakpoint_threshold_type="standard_deviation"
# breakpoint_threshold_type="interquartile"
# breakpoint_threshold_type="gradient"

docs = text_splitter.create_documents([state_of_the_union])
# print(docs[0].page_content)

for i, doc in enumerate( docs):
    print(i,  doc.page_content)





