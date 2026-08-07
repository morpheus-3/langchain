from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load API Key
load_dotenv()

if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    raise ValueError("❌ HuggingFace API token not found in .env file")

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

# Run Prompt 1
prompt1 = template1.format(topic="Black Hole")
result = model.invoke(prompt1)

# Run Prompt 2
prompt2 = template2.format(text=result.content)
result1 = model.invoke(prompt2)

print("\n✅ SUMMARY:\n")
print(result1.content)