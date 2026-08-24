from groq import Groq

from src.config import GROQ_API_KEY, GROQ_MODEL, TOP_K

SYSTEM_PROMPT = (
    "You are a research assistant for Reserve Bank of India monetary policy documents. "
    "Answer ONLY from the provided context. If the context does not contain the answer, "
    "say clearly that the documents do not cover it. Cite sources inline as [Source N]. "
    "Be precise with numbers, rates and dates. Keep answers under 200 words."
)


def build_context(hits):
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[Source {i}] file: {h['source_file']} | page: {h['page']}\n{h['text']}"
        )
    return "\n\n".join(blocks)


def generate_answer(question, retriever, top_k=TOP_K):
    hits = retriever.retrieve(question, top_k=top_k)
    if not hits:
        return "No relevant context found in the indexed documents.", []

    context = build_context(hits)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, with [Source N] citations."
    )

    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content, hits


if __name__ == "__main__":
    from src.retriever import RAGRetriever

    retriever = RAGRetriever()
    answer, hits = generate_answer("What did RBI say about inflation?", retriever)
    print(answer)
    print("\n--- sources ---")
    for h in hits:
        print(f"{h['source_file']} p{h['page']} (sim {h['similarity']})")
        