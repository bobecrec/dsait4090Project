import os
from typing import List, Dict, Any

import anthropic


def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    """Simple RAG prompt. Contexts = list of {title, text, ...} dicts."""
    context_block = "\n\n".join(
        f"[DOC {i+1}] {c.get('title','')}\n{c.get('text','')}"
        for i, c in enumerate(contexts)
    )
    return (
        "You are a QA system. Answer questions based on the provided documents.\n\n"
        f"Question: {question}\n\n"
        "Relevant documents:\n"
        f"{context_block}\n\n"
        "Answer the question as concisely as possible. "
        "Provide only the answer, no explanation needed."
    )


def generate_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Generate answer using Anthropic Claude API.

    Requires ANTHROPIC_API_KEY environment variable to be set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Please set it with: export ANTHROPIC_API_KEY='your-key-here'"
        )

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_prompt(question, contexts)

    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""