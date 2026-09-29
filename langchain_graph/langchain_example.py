from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

# 1. Define your desired output structure using Pydantic
class TestCase(BaseModel):
    title: str = Field(description="The title of the test case")
    steps: List[str] = Field(description="Step-by-step instructions to execute the test")
    expected_result: str = Field(description="The expected outcome if the test passes")

class TestCaseList(BaseModel):
    test_cases: List[TestCase] = Field(description="List of generated test cases")

# 2. Initialize the parser
parser = PydanticOutputParser(pydantic_object=TestCaseList)

# 3. Create the prompt, injecting the parser's format instructions
prompt = PromptTemplate(
    template=(
        "Generate test cases for the following user story:\n{user_story}\n\n"
        "{format_instructions}"
    ),
    input_variables=["user_story"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 4. Initialize your LLM
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    # Note: Please rotate this key in your OpenRouter dashboard as it was posted publicly!
    
    openai_api_key="", 
    openai_api_base=""
)

# 5. Build the chain: Prompt -> LLM -> Parser
chain = prompt | llm | parser 

user_story = "as user i want to login my bank application"
result = chain.invoke({"user_story": user_story})

# The result is now a properly structured Python object!
print(result.model_dump_json(indent=2))