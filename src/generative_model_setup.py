from typing import List, Dict, Any

def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    """Simple RAG prompt. Contexts = list of {title, text, ...} dicts."""
    context_block = "\n\n".join(
        f"[DOC {i+1}] {c.get('title','')}\n{c.get('text','')}"
        for i, c in enumerate(contexts)
    )
    return (
        "You are a QA system.\n\n"
        f"Question:\n{question}\n\n"
        "Relevant documents:\n"
        f"{context_block}\n\n"
        "Answer the question as concisely as possible.\n"
    )

def generate_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Stub for now. Your teammates can plug in OpenAI / LLaMA here.
    For now, just return empty string or a dummy.
    """
    # TODO: replace with real LLM call
    return ""