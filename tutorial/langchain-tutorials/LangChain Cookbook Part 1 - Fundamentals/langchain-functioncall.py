
#

from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional

class Person(BaseModel):
    """Identifying information about a person."""

    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    fav_food: Optional[str] = Field(None, description="The person's favorite food")


#

from langchain.chains.openai_functions import create_structured_output_chain

llm = ChatOpenAI(model='gpt-4-0613', openai_api_key=openai_api_key)

chain = create_structured_output_chain(Person, llm, prompt)
chain.run(
    "Sally is 13, Joey just turned 12 and loves spinach. Caroline is 10 years older than Sally."
)

#
from typing import Sequence

class People(BaseModel):
    """Identifying information about all people in a text."""

    people: Sequence[Person] = Field(..., description="The people in the text")


#

chain = create_structured_output_chain(People, llm, prompt)
chain.run(
    "Sally is 13, Joey just turned 12 and loves spinach. Caroline is 10 years older than Sally."
)    

# output
People(people=[Person(name='Sally', age=13, fav_food=None), Person(name='Joey', age=12, fav_food='spinach'), Person(name='Caroline', age=23, fav_food=None)])