from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

import os

# Load API Key
load_dotenv()


# HuggingFace Model
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
)


model = ChatHuggingFace(llm=llm)
parser= JsonOutputParser()

template = PromptTemplate(
    template= 'Give me the 5 facts about the topic \ n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)
#prompt = template.format()
chain = template| model | parser
result = chain.invoke({'topic':'Gravity'})
#final_result = parser.parse(result.content)

print((result))