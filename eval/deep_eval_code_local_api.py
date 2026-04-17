# evaluate.py
import json
import sys
import csv
import logging
import requests

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
import ollama

# -------------------------
# LOGGING CONFIG
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# -------------------------
# OLLAMA JUDGE
# -------------------------
class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "mistral:latest"):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0}
        )
        return response["message"]["content"]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"ollama/{self.model_name}"


# -------------------------
# LOAD DATASET
# -------------------------
print("Loading evaluation dataset...")
with open("eval/evaluation_dataset.json") as f:
    data = json.load(f)


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    logger.info("Starting DeepEval evaluation pipeline")
    logger.info(f"Loaded dataset with {len(data)} items")

    # -------------------------
    # JUDGE + METRICS SETUP
    # -------------------------
    judge = OllamaJudge(model_name="mistral:latest")

    metrics = [
        FaithfulnessMetric(threshold=0.5, model=judge, include_reason=True, async_mode=False),
        AnswerRelevancyMetric(threshold=0.5, model=judge, include_reason=True, async_mode=False),
        ContextualPrecisionMetric(threshold=0.5, model=judge, include_reason=True, async_mode=False),
        ContextualRecallMetric(threshold=0.5, model=judge, include_reason=True, async_mode=False),
    ]

    # -------------------------
    # CSV FILES
    # -------------------------
    per_query_file = open("per_query_results.csv", "w", newline="", encoding="utf-8")
    avg_file       = open("average_results.csv",   "w", newline="", encoding="utf-8")

    per_writer = csv.writer(per_query_file)
    avg_writer = csv.writer(avg_file)

    per_writer.writerow([
        "question", "response", "ground_truth", "context",
        "faithfulness", "answer_relevancy",
        "context_precision", "context_recall"
    ])

    # -------------------------
    # SCORE STORAGE
    # -------------------------
    f_scores, p_scores, r_scores, a_scores = [], [], [], []
    test_cases = []
    meta       = []

    # -------------------------
    # LOOP OVER DATASET + CALL RAG API
    # -------------------------
    for idx, item in enumerate(data):

        logger.info(f"Processing item {idx+1}/{len(data)}")

        q  = item["question"]
        gt = item.get("ground_truth", item.get("answer"))

        try:
            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json={"question": q}
            )
            response.raise_for_status()
            result = response.json()
            ans  = result.get("answer", "")
            ctxs = result.get("contexts", [])
            

        except Exception as e:
            logger.error(f"API failed for item {idx+1}: {e}")
            continue

        print("\n==============================")
        print(f"Question:     {q}")
        print(f"Response:     {ans}")
        print(f"Ground Truth: {gt}")
        print("==============================\n")

        test_case = LLMTestCase(
            input=q,
            actual_output=ans,
            expected_output=gt,
            retrieval_context=ctxs,
        )

        test_cases.append(test_case)
        meta.append({
            "question": q,
            "answer":   ans,
            "gt":       gt,
            "context":  " || ".join(ctxs),
        })

    # -------------------------
    # RUN METRICS MANUALLY
    # No async, no timeout, no crashes
    # -------------------------
    logger.info(f"Running DeepEval on {len(test_cases)} test cases...")

    all_results = []

    for i, test_case in enumerate(test_cases):
        logger.info(f"Evaluating test case {i+1}/{len(test_cases)}")
        case_scores = {}

        for metric in metrics:
            metric_name = metric.__class__.__name__
            try:
                logger.info(f"  Running {metric_name}...")
                metric.measure(test_case)        # ← synchronous, no timeout
                case_scores[metric_name] = {
                    "score":   metric.score,
                    "reason":  metric.reason,
                    "success": metric.score >= metric.threshold
                }
                logger.info(f"  {metric_name}: {metric.score:.2f}")
            except Exception as e:
                logger.error(f"  {metric_name} failed: {e}")
                case_scores[metric_name] = {
                    "score": 0.0,
                    "reason": str(e),
                    "success": False
                }

        all_results.append(case_scores)

    # -------------------------
    # PARSE RESULTS + WRITE CSV
    # -------------------------
    for i, case_scores in enumerate(all_results):

        f = case_scores.get("FaithfulnessMetric",        {}).get("score", 0)
        a = case_scores.get("AnswerRelevancyMetric",      {}).get("score", 0)
        p = case_scores.get("ContextualPrecisionMetric",  {}).get("score", 0)
        r = case_scores.get("ContextualRecallMetric",     {}).get("score", 0)

        f_scores.append(f)
        a_scores.append(a)
        p_scores.append(p)
        r_scores.append(r)

        m = meta[i]
        per_writer.writerow([
            m["question"], m["answer"], m["gt"], m["context"],
            round(f, 3), round(a, 3), round(p, 3), round(r, 3)
        ])

        print(f"\n--- Result {i+1}: {m['question'][:60]} ---")
        for metric_name, data in case_scores.items():
            status = "PASS" if data["success"] else "FAIL"
            print(f"  [{status}] {metric_name}: {data['score']:.2f}")
            if data["reason"]:
                print(f"         Reason: {data['reason']}")

    # -------------------------
    # AVERAGE SCORES
    # -------------------------
    def avg(x):
        return sum(x) / len(x) if x else 0

    avg_writer.writerow(["avg_faithfulness", "avg_answer_relevancy",
                         "avg_context_precision", "avg_context_recall"])
    avg_writer.writerow([
        round(avg(f_scores), 2),
        round(avg(a_scores), 2),
        round(avg(p_scores), 2),
        round(avg(r_scores), 2),
    ])

    per_query_file.close()
    avg_file.close()

    # -------------------------
    # FINAL SUMMARY
    # -------------------------
    print("\n========== AVERAGE SCORES ==========")
    print("faithfulness:       ", round(avg(f_scores), 2))
    print("answer_relevancy:   ", round(avg(a_scores), 2))
    print("context_precision:  ", round(avg(p_scores), 2))
    print("context_recall:     ", round(avg(r_scores), 2))
    print("=====================================")

    logger.info("Done")