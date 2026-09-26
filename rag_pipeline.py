"""
rag_pipeline.py
----------------
Orchestrates the retrieval-augmented generation flow:
retrieve relevant chunks -> build a grounded prompt -> call the LLM
-> return the answer plus the sources used.

Uses Groq by default (fast + free tier), since the project brief
lists Groq, Gemini, OpenAI, Hugging Face, or a local model as
acceptable options. Swap out `call_llm` to point at a different
provider if you prefer.
"""

import os
from groq import Groq

from prompt import SYSTEM_PROMPT, build_user_prompt, FALLBACK_MESSAGE

TOP_K = 4


def call_llm(system_prompt, user_prompt):
    """
    Calls the Groq chat completion API. Requires GROQ_API_KEY in the
    environment (see .env.example).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


def answer_question(vector_store, question, top_k=TOP_K):
    """
    Full RAG step for one question.

    Returns:
        {
            "answer": str,
            "sources": [{"source": str, "page": int, "score": float}, ...]
        }
    """
    retrieved = vector_store.search(question, top_k=top_k)

    if not retrieved:
        return {"answer": FALLBACK_MESSAGE, "sources": []}

    user_prompt = build_user_prompt(question, retrieved)

    try:
        answer = call_llm(SYSTEM_PROMPT, user_prompt)
    except RuntimeError as e:
        # Surface config errors (e.g. missing API key) clearly in the UI
        return {"answer": f"⚠️ {e}", "sources": []}

    sources = [
        {"source": r["source"], "page": r["page"], "score": round(r["score"], 3)}
        for r in retrieved
    ]
    return {"answer": answer, "sources": sources}
