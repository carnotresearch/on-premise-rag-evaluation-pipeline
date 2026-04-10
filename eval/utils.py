import numpy as np
from numpy.linalg import norm
import logging
from eval.llm_service import call_llm

logger = logging.getLogger(__name__)

# -------------------------
# COSINE SIMILARITY
# -------------------------
def cosine_similarity(a, b):
    if norm(a) == 0 or norm(b) == 0:
        logger.debug("Zero vector encountered in cosine similarity")
        return 0.0
    return np.dot(a, b) / (norm(a) * norm(b))


# -------------------------
# NORMALIZATION
# -------------------------
def normalize_supported(text):
    text = text.lower()
    if "not supported" in text:
        return "not supported"
    if "supported" in text:
        return "supported"
    return "not supported"


def normalize_relevant(text):
    text = text.lower()
    if "not relevant" in text:
        return "not relevant"
    if "relevant" in text:
        return "relevant"
    return "not relevant"


# -------------------------
# REMOVE NOISE
# -------------------------
def remove_noise(contexts):
    logger.debug(f"Removing noise from {len(contexts)} contexts")
    clean = []
    for c in contexts:
        c_low = c.lower()

        if "contents" in c_low:
            continue
        if "introduction" in c_low:
            continue
        if len(c.split()) < 8:
            continue

        clean.append(c)

    logger.debug(f"Reduced to {len(clean)} contexts after cleaning")
    return clean


# -------------------------
# CLAIM EXTRACTION (EXACT PROMPT)
# -------------------------
def extract_claims(text):
    logger.debug("Extracting claims")

    res = call_llm(f"""Extract all factual claims from the following text. A claim is a statement that asserts something to be true. It should be specific and verifiable. Do not include opinions, questions, or vague statements. Return only the claims, one per line, without any numbering or explanation.

Rules:
- One claim per line
- No JSON
- No explanation
- No numbering
Text: {text}""")

    claims = [l.strip() for l in res.split("\n") if len(l.strip()) > 15]

    logger.debug(f"Extracted {len(claims)} claims")
    return claims


# -------------------------
# VERIFY CLAIM (EXACT PROMPT)
# -------------------------
def verify_claim(claim, contexts):
    logger.debug("Verifying claim")

    result = normalize_supported(call_llm(f""" You are a smart evaluator.
Context: {" ".join(contexts)}
Claim: {claim} Based on the context, is the claim supported?
Answer only: supported / not supported
"""))

    logger.info(f"Verification result: {result}")
    return result
def smart_filter(question, contexts, embeddings, min_sim=0.5, max_keep=3):
    logger.debug("Applying smart filter")

    q_emb = embeddings.embed_query(question)

    scored = []
    for ctx in contexts:
        sim = cosine_similarity(q_emb, embeddings.embed_query(ctx))
        scored.append((ctx, sim))

    scored.sort(key=lambda x: x[1], reverse=True)

    filtered = [c for c, s in scored if s >= min_sim]
    top_k = [c for c, _ in scored[:5]]

    filtered = list(dict.fromkeys(filtered + top_k))

    logger.debug(f"Filtered contexts count: {len(filtered[:max_keep])}")

    return filtered[:max_keep]