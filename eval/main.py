import json
import sys
import csv
import logging
import requests

# from rag.chain import answer_question

from eval.embeddings import embeddings
from eval.utils import remove_noise, smart_filter,extract_claims, verify_claim

from eval.metrics.faithfulness import compute_faithfulness
from eval.metrics.context_precision import compute_context_precision
from eval.metrics.context_recall import compute_context_recall
from eval.metrics.answer_relevancy import compute_answer_relevancy

# -------------------------
# LOGGING CONFIG (UNCHANGED)
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# -------------------------
# LOAD DATASET (UNCHANGED)
# -------------------------
print("Loading evaluation dataset...")

with open("eval/evaluation_dataset.json") as f:
    data = json.load(f)


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    logger.info("Starting evaluation pipeline")
    logger.info(f"Loaded dataset with {len(data)} items")

    # -------------------------
    # CSV FILES (UNCHANGED)
    # -------------------------
    per_query_file = open("per_query_results.csv", "w", newline="", encoding="utf-8")
    avg_file = open("average_results.csv", "w", newline="", encoding="utf-8")

    per_writer = csv.writer(per_query_file)
    avg_writer = csv.writer(avg_file)

    per_writer.writerow([
        "question", "response", "ground_truth",
        "context",
        "faithfulness", "answer_relevancy",
        "context_precision", "context_recall"
    ])

    # -------------------------
    # SCORE STORAGE
    # -------------------------
    f_scores, p_scores, r_scores, a_scores = [], [], [], []

    # -------------------------
    # LOOP OVER DATA
    # -------------------------
    for idx, item in enumerate(data):

        logger.info(f"Processing item {idx+1}/{len(data)}")

        q = item["question"]
        gt = item.get("ground_truth", item.get("answer"))

        # -------------------------
        # CALL RAG
        # -------------------------
        try:
         response = requests.post(
            "http://127.0.0.1:8000/ask",
             json={"question": q}
    )
         response.raise_for_status()
         result = response.json()

         ans = result.get("answer", "")
         ctxs = result.get("contexts", [])

        except Exception as e:
         logger.error(f"API failed: {e}")
         continue
        # -------------------------
        # DEBUG PRINTS (UNCHANGED)
        # -------------------------
        print("\n==============================")
        print(f"Question: {q}")
        print(f"Response: {ans}")
        print(f"Ground Truth: {gt}")
        # print("\n📚 Context used:")
        print("==============================\n")

        # -------------------------
        # CLEAN + FILTER
        # -------------------------
        ctxs_clean = remove_noise(ctxs)
        ctxs_filtered = smart_filter(q, ctxs_clean, embeddings)

        # -------------------------
        # STRING CONTEXT (FOR CSV)
        # -------------------------
        context_text = " || ".join(ctxs_filtered)

        # -------------------------
        # METRICS
        # -------------------------
        f = compute_faithfulness(ans, ctxs_filtered)
        p = compute_context_precision(q, ctxs_filtered, embeddings)
        r = compute_context_recall(gt, ctxs_clean)
        a = compute_answer_relevancy(q, ans, embeddings)

        # -------------------------
        # WRITE CSV (UNCHANGED)
        # -------------------------
        per_writer.writerow([q, ans, gt, context_text, f, a, p, r])

        # -------------------------
        # STORE SCORES
        # -------------------------
        f_scores.append(f)
        p_scores.append(p)
        r_scores.append(r)
        a_scores.append(a)

    # -------------------------
    # AVERAGE FUNCTION
    # -------------------------
    def avg(x):
        return sum(x) / len(x) if x else 0

    # -------------------------
    # WRITE AVERAGE CSV
    # -------------------------
    avg_writer.writerow([
        "avg_faithfulness",
        "avg_answer_relevancy",
        "avg_context_precision",
        "avg_context_recall"
    ])

    avg_writer.writerow([
        round(avg(f_scores), 2),
        round(avg(a_scores), 2),
        round(avg(p_scores), 2),
        round(avg(r_scores), 2)
    ])

    # -------------------------
    # PRINT FINAL SCORES (UNCHANGED)
    # -------------------------
    print("faithfulness:", round(avg(f_scores), 2))
    print("answer_relevancy:", round(avg(a_scores), 2))
    print("context_precision:", round(avg(p_scores), 2))
    print("context_recall:", round(avg(r_scores), 2))


    logger.info("Done")