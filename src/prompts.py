from langchain.prompts import ChatPromptTemplate

SYSTEM_BASE = "You are a senior software engineer. Be concise, correct, and actionable."

EXPLAIN_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BASE + " Explain the given code in plain English."),
    ("human", "Question: {question}\n\nRelevant code:\n{context}")
])

BUGS_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BASE + " Review code for errors and risks."),
    ("human", "Question: {question}\n\nRelevant code:\n{context}")
])

REFACTOR_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BASE + " Suggest cleaner, efficient code."),
    ("human", "Question: {question}\n\nRelevant code:\n{context}")
])
