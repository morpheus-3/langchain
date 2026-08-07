from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

# Prompt 1 – Detailed Report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# Prompt 2 – Summary
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text:\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()
chain = template1 | model| parser | template2 | model | parser
result = chain.invoke({'topic':'black hole'})
print(result)