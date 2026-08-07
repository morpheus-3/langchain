from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel 
from dotenv import load_dotenv

load_dotenv()

# ----- Define Output Schema -----
class Report(BaseModel):
    topic: str
    explanation: str
    example: str

parser = PydanticOutputParser(pydantic_object=Report)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=300
)

model = ChatHuggingFace(llm=llm)

prompt = f"""
Explain Black Hole in simple words.

Return output in this format:
{parser.get_format_instructions()}
"""

response = model.invoke(prompt)

print(parser.parse(response.content))