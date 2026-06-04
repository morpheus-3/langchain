from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# -------- MODEL --------
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# -------- OUTPUT SCHEMA --------
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description="give the sentiment of feedback"
    )

classifier_parser = PydanticOutputParser(pydantic_object=Feedback)
text_parser = StrOutputParser()

# -------- CLASSIFICATION PROMPT --------
prompt = PromptTemplate(
    template="""
Classify the sentiment of following feedback into positive or negative.

Feedback: {feedback}

{format_instructions}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": classifier_parser.get_format_instructions()
    }
)

classifier_chain = prompt | model | classifier_parser

# -------- RESPONSE PROMPTS --------
prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback:\n{feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback:\n{feedback}",
    input_variables=["feedback"]
)

# -------- BRANCH --------
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | text_parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | text_parser),
    lambda x: "Could not find sentiment"
)

# -------- FINAL CHAIN --------
chain = classifier_chain | branch_chain

print(chain.invoke({"feedback": "this is the best ever phone"}))
chain.get_graph().print_ascii()