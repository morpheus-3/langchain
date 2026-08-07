from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


model = ChatGroq(
    model="llama-3.1-8b-instant",   # REQUIRED
    temperature=0.7
)
prompt = PromptTemplate(
    template='Generate 5 interesting Facts about {topic}',
    input_variables= ['topic']
)

parser = StrOutputParser()
chain = prompt | model | parser
result = chain.invoke({'topic':'Indian Army'})
print(result)