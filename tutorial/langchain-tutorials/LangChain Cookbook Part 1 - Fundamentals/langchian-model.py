
# from dotenv import load_dotenv
# import os

# load_dotenv()

# openai_api_key=os.getenv('OPENAI_API_KEY', '')

#####  Language Model

from langchain.llms import OpenAI

llm = OpenAI(model_name="text-ada-001", openai_api_key=openai_api_key)

llm("What day comes after Friday?")


##### Chat Model

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

#
chat = ChatOpenAI(temperature=1, openai_api_key=openai_api_key)

chat(
    [
        SystemMessage(content="You are an unhelpful AI bot that makes a joke at whatever the user says"),
        HumanMessage(content="I would like to go to New York, how should I do this?")
    ]
)

#
chat = ChatOpenAI(model='gpt-3.5-turbo-0613', temperature=1, openai_api_key=openai_api_key)

output = chat(messages=
     [
         SystemMessage(content="You are an helpful AI bot"),
         HumanMessage(content="What’s the weather like in Boston right now?")
     ],
     functions=[{
         "name": "get_current_weather",
         "description": "Get the current weather in a given location",
         "parameters": {
             "type": "object",
             "properties": {
                 "location": {
                     "type": "string",
                     "description": "The city and state, e.g. San Francisco, CA"
                 },
                 "unit": {
                     "type": "string",
                     "enum": ["celsius", "fahrenheit"]
                 }
             },
             "required": ["location"]
         }
     }
     ]
)

print(output)




##### Text Embedding Model

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

text = "Hi! It's time for the beach"

text_embedding = embeddings.embed_query(text)


print (f"Here's a sample: {text_embedding[:5]}...")
print (f"Your embedding is length {len(text_embedding)}")