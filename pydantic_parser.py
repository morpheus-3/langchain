from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ----- Model -----
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=300
)

model = ChatHuggingFace(llm=llm)

# ----- Output Schema -----
class Person(BaseModel):
    name: str = Field(description="Name of the Person")
    age: int = Field(gt=18, description="Age of the Person")
    city: str = Field(description="City person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

# ----- Prompt -----
template = PromptTemplate(
    template="""
Generate name, age and city of a fictional {place} person.

IMPORTANT:
Return ONLY valid JSON.
Do NOT add explanation.
Do NOT add text before or after JSON.

{format_instructions}
""",
    input_variables=["place"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)
prompt = template.invoke({"place": "Indian"})

result = model.invoke(prompt)
final_result = parser.parse_with_prompt(
    result.content,
    prompt=prompt.to_string()
)
print(final_result)



# chain = template | model| parser