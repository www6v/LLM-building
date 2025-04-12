# https://python.langchain.com/docs/integrations/llms/huggingface_pipelines

from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="bigscience/bloom-1b7",
    task="text-generation",
    model_kwargs={"temperature": 0, "max_length": 64},
)



from langchain.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | llm

question = "What is electroencephalography?"

print(chain.invoke({"question": question}))