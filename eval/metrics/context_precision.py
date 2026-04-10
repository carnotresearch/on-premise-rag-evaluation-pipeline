from eval.utils import cosine_similarity, normalize_relevant
from eval.llm_service import call_llm

def compute_context_precision(question, contexts, embeddings):
    
    q_emb = embeddings.embed_query(question)

    scored = [(c, cosine_similarity(q_emb, embeddings.embed_query(c))) for c in contexts]
    scored.sort(key=lambda x: x[1], reverse=True)

    contexts = scored[:3]

    relevant = sum(
        1 for c, _ in contexts
        if "not relevant" not in normalize_relevant(call_llm(f"""You are smart evaluator. Carefully check if the context is even remotely relevant to answer the question. Be generous in considering something as relevant.
Question: {question}
Context: {c}
Answer only: relevant / not relevant
"""))
    )

    return relevant / len(contexts) if contexts else 0