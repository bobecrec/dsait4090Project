from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1:8b", temperature=0.0)


def answer_question(question, context=""):
    prompt = f"""You are a knowledgeable assistant.

Use the following context to answer the question:
{context}

Question: {question}
Answer concisely with only the final answer."""

    response = llm([HumanMessage(content=prompt)])
    return response.content
