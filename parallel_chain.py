from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

# -------- TWO WORKING GROQ MODELS --------
model_notes = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)

model_quiz = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text:\n{text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text:\n{text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document:\nNotes -> {notes}\nQuiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model_notes | parser,
    "quiz": prompt2 | model_quiz | parser
})

merge_chain = prompt3 | model_notes | parser
chain = parallel_chain | merge_chain

text = "Support Vector Machine is used for classification and regression."

result = chain.invoke({"text": text})
print(result)
chain.get_graph().print_ascii()