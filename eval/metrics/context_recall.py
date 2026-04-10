from eval.utils import extract_claims, normalize_supported
from eval.llm_service import call_llm

def compute_context_recall(gt, contexts):
    claims = extract_claims(gt)

    covered = 0

    for claim in claims:
        for ctx in contexts:
            if "not supported" not in normalize_supported(call_llm(f"""You are an expert evaluator measuring CONTEXT RECALL.

Your task is to determine whether the CLAIM is supported by the CONTEXT.

IMPORTANT GUIDELINES:
- Do NOT require exact wording match.
- Consider semantic similarity, paraphrasing, and implied meaning.
- If ANY part of the claim is supported, mark it as "SUPPORTED".
- Even partial or indirect support counts.
- Only mark "NOT SUPPORTED" if the context clearly does not contain the information.
Context: {ctx}
Claim: {claim}
Answer only: supported / not supported
""")):
                covered += 1
                break

    return covered / len(claims) if claims else 0