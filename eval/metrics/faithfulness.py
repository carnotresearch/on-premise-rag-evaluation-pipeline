from eval.utils import extract_claims, verify_claim

def compute_faithfulness(answer, contexts):
    claims = extract_claims(answer)

    supported = sum(
        1 for claim in claims
        if "not supported" not in verify_claim(claim, contexts)
    )

    return supported / len(claims) if claims else 0
