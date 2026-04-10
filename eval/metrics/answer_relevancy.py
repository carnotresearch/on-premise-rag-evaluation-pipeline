from eval.utils import cosine_similarity
from eval.llm_service import call_llm

def compute_answer_relevancy(question, answer, embeddings):

    qs_text = call_llm(f"""You are a helpful assistant. Generate EXACTLY 5 questions.

Rules:
- One question per line
- No explanation
- No numbering
- No empty lines

Answer:
{answer}
""")

    if not qs_text or not qs_text.strip():
        return 0

    qs = qs_text.split("\n")

    q_emb = embeddings.embed_query(question)

    sims = [
        cosine_similarity(q_emb, embeddings.embed_query(q))
        for q in qs if q.strip()
    ]

    if not sims:
        return 0

    sims = sorted(sims, reverse=True)[:2]

    return sum(sims) / len(sims)