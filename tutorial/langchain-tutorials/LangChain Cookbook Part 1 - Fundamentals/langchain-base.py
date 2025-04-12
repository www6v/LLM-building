
############# LangChain Cookbook Part 1 - Fundamentals

# from dotenv import load_dotenv
# import os

# load_dotenv()

# openai_api_key=os.getenv('OPENAI_API_KEY', '')



######   Chat Messages

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# This it the language model we'll use. We'll talk about what we're doing below in the next section
chat = ChatOpenAI(temperature=.7, openai_api_key=openai_api_key)

result1 = chat(
    [
        SystemMessage(content="You are a nice AI bot that helps a user figure out what to eat in one short sentence"),
        HumanMessage(content="I like tomatoes, what should I eat?")
    ]
)

result2 = chat(
    [
        SystemMessage(content="You are a nice AI bot that helps a user figure out where to travel in one short sentence"),
        HumanMessage(content="I like the beaches where should I go?"),
        AIMessage(content="You should go to Nice, France"),
        HumanMessage(content="What else should I do when I'm there?")
    ]
)

result3 = chat(
    [
        HumanMessage(content="What day comes after Thursday?")
    ]
)

print(result3)