# End-to-End On-Premise RAG Evaluation

> A fully **on-premises**, open-source framework for evaluating RAG systems —
> two evaluation methodologies, a human-in-the-loop
> dataset creation tool, and zero external API calls. Everything runs locally
> via Ollama.

---

## The Hidden Bottleneck in RAG Systems

Everyone has a tutorial on *building* a RAG pipeline.  
Nobody talks about *evaluating* one.

- How do you know if your answers are faithful to the documents?  
- How do you know if your retriever is fetching the right chunks?  
- How do you know if a change made things better or worse?

You need an evaluation pipeline.

This repo gives you one — fully local,  
no OpenAI, no cloud, no cost.

---

## Key Differentiators of This Repository

### 1. 100% on-premises

No data leaves your machine. Generation, judging, embeddings — all
running locally via Ollama. Designed for teams and use cases where
sending documents to an external API is not an option.

> ⚠️ **Judge model matters.** Both evaluation approaches require a
> high-parameter open-source model. Mistral 7B+ is tested and 
> recommended. Smaller models (1–3B) produce unreliable metric scores.

### 2. Human-in-the-loop dataset creation

No manual labelling. Upload a PDF or DOCX, and the local LLM generates
a balanced evaluation dataset across 8 question types. Then review,
edit, or delete any pair in real time via the web UI before saving.
You stay in control of the dataset quality — the model does the heavy
lifting.

### 3. Two evaluation Methodologies

Most repos give you one way to evaluate. This gives you two, so you can
compare and choose what fits your use case:

**Custom-built evaluator**  
Every metric written from scratch. No framework abstractions, no black
boxes. You can read exactly what is being measured and why. Uses
high-level evaluation logic with BGE-M3 embeddings and a local LLM judge.

**DeepEval with local OSS models**  
Uses DeepEval's individual metric classes — not their `evaluate()`
function. Why? Because `evaluate()` is async and open-source models
running locally via Ollama are slow — async timeouts kill the run.
Instead, metrics are called synchronously, with tuned hyperparameters
(timeout, retries, thresholds) to work reliably with local models.
This is the part most DeepEval tutorials skip entirely.

---

## Metrics

| Metric | What it measures |
|---|---|
| **Faithfulness** | Are the answer's claims grounded in retrieved context? Catches hallucination. |
| **Answer Relevancy** | Does the answer actually address the question? |
| **Context Precision** | Are retrieved chunks relevant to the question? Evaluates retriever accuracy. |
| **Context Recall** | Does retrieved context cover the ground truth? Evaluates retriever completeness. |

All scores in `[0.0, 1.0]`. Higher is better.

---

## vs. existing approaches

| | This repo | RAGAS | DeepEval (default) |
|---|---|---|---|
| Fully local / on-prem | ✅ | ❌ OpenAI default | ❌ OpenAI default |
| Custom metric logic | ✅ from scratch | ❌ black box | ❌ black box |
| DeepEval sync workaround | ✅ | ❌ | ❌ |
| Built-in dataset creator | ✅ 8 question types | ❌ | ❌ |
| Human-in-the-loop review | ✅ | ❌ | ❌ |
| Cost | ✅ free | ❌ API cost | ❌ API cost |

---

## How It Works

The system operates as a two-stage end-to-end pipeline:

**Step 1 — Create:** Upload a PDF or DOCX document through the web UI. A local LLM generates a balanced, high-quality Q&A evaluation dataset across 8 question types.

**Step 2 — Evaluate:** Run the evaluation pipeline against your live RAG API. It loads the generated dataset, retrieves answers and contexts, and scores across four metrics.

![High-Level Architecture](assets/architecture.png)

---

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Part 1: Dataset Creation (Flask App)](#part-1-dataset-creation-flask-app)
  - [Starting the App](#starting-the-app)
  - [Workflow](#workflow)
  - [Question Types](#question-types)
  - [Generation Pipeline](#generation-pipeline)
  - [Web UI Routes](#web-ui-routes)
  - [Quality Control](#quality-control)
- [Part 2: Evaluation Pipeline](#part-2-evaluation-pipeline)
  - [Metrics](#metrics)
  - [Usage](#usage)
  - [Output Files](#output-files)
- [API Reference](#api-reference)
- [Notes & Limitations](#notes--limitations)

---

## Project Structure

```
.
├── data_creation_app/
│   ├── app.py              # Flask app — dataset generation & review UI
│   ├── llm_service.py      # LLM call abstraction with retry + backoff
│   ├── config.py           # Loads API_URL and MODEL_NAME from .env
│   └── templates/
│       ├── index.html      # Upload form
│       └── review.html     # Dataset review & edit UI
│
├── eval/
│   ├── config.py           # Eval-side config (API_URL, MODEL_NAME)
│   ├── embeddings.py       # BGE-M3 embedding wrapper (via Ollama)
│   ├── llm_service.py      # LLM calls for metric computation
│   ├── utils.py            # Claim extraction, other utilities
│   ├── main.py             # Evaluation pipeline entry point
│   ├── evaluation_dataset.json   # Final dataset saved by web app
│   └── metrics/
│       ├── faithfulness.py
│       ├── context_precision.py
│       ├── context_recall.py
│       └── answer_relevancy.py
│
├── backup_dataset.json     # Auto-checkpoint during generation
├── per_query_results.csv   # Output: per-question scores
├── average_results.csv     # Output: averaged scores
├── assets/                 # Screenshots used in this README
│   ├── architecture.png
│   ├── qa_generator_main.png
│   ├── qa_generator_settings.png
│   ├── review_ui_1.png
│   ├── review_ui_2.png
│   ├── terminal_generation.png
│   └── terminal_eval.png
├── .env                    # Environment variables (not committed)
└── README.md
```

---

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) running locally with:
  - An LLM model (e.g., `mistral`, `llama3`) for generation and judging
  - `bge-m3:latest` for embeddings (evaluation pipeline only)
- Your RAG API running at `http://127.0.0.1:8000` with a `POST /ask` endpoint

---

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Required packages (`requirements.txt`):**

```
flask
requests
numpy
pypdf
python-docx
python-dotenv
```

---

## Configuration

Create a `.env` file in the project root:

```env
# Ollama chat completions endpoint
API_URL=http://localhost:11434/api/chat

# Ollama model for generation and evaluation judging
MODEL_NAME=mistral
```

> ⚠️ **Warning:** Both `data_creation_app/config.py` and `eval/config.py` load from this same `.env` file. Never commit it to version control.

---

## Part 1: Dataset Creation (Flask App)

### Starting the App

```bash
python -m data_creation_app.app

# Visit http://localhost:5000 in your browser
```

![QA Generator — Main UI](assets/qa_generator_main.png)

![QA Generator — Generation Settings](assets/qa_generator_settings.png)

### Workflow

1. Upload a PDF or DOCX document and set the number of Q&A pairs to generate
2. **Generate** — the app chunks the document, schedules generation across all question types, and runs parallel LLM calls
3. **Review** — inspect, edit, or delete generated pairs via the review UI
4. **Save** — approved pairs are written to `eval/evaluation_dataset.json`, ready for evaluation

### Question Types

The app generates questions balanced across 8 specific types:

| Type | Description |
|---|---|
| `procedural` | Step-by-step process questions (how to perform a task) |
| `compliance` | Rules, prohibitions, and mandatory requirements |
| `role_responsibility` | Who owns or must perform a specific action |
| `condition_trigger` | When or under what conditions a process activates |
| `definition_concept` | What a term, acronym, or concept means in the document |
| `exception_escalation` | Edge cases, deviations, and escalation procedures |
| `timeline_sla` | Timeframes, deadlines, response times, and SLAs |
| `multi_hop` | Complex questions requiring synthesis of 2+ facts from the text |

![Review UI — Q&A pairs](assets/review_ui_1.png)

![Review UI — Timeline SLA & Multi Hop](assets/review_ui_2.png)

### Generation Pipeline

**Chunking**
`semantic_chunk()` splits on paragraph/section boundaries rather than blind character slicing. Each chunk is 300–900 characters.

**Scheduling**
`build_generation_schedule()` distributes question types evenly across all chunks. A 2× buffer is generated (e.g., 40 attempts for 20 target pairs) to absorb rejections.

**Parallel Execution**
`ThreadPoolExecutor` with 4 workers runs LLM calls concurrently for speed.

**Checkpoint Saving**
Every accepted pair is immediately written to `backup_dataset.json` so no progress is lost on failure.

**Deduplication**
MD5-based hash deduplication on question text prevents near-identical questions from appearing in the final dataset.

### Web UI Routes

| Route | Method | Description |
|---|---|---|
| `/` | `GET` | Upload form (file + pair count + model selector) |
| `/generate` | `POST` | Runs generation pipeline, redirects to `/review` |
| `/review` | `GET` | Dataset review and editing UI |
| `/save` | `POST` | Saves approved dataset to `eval/evaluation_dataset.json` |
| `/stats` | `GET` | Returns JSON summary (type distribution, avg lengths) |

### Quality Control

Every generated Q&A pair passes through two validation stages before being accepted:

**`validate_qa()` — Structural checks:**
- All three keys present: `question`, `ground_truth`, `type`
- Question is at least 15 characters
- Answer is at least 12 words
- Type is one of the 8 defined SOP types

**`is_high_quality()` — Content quality checks:**
- Answer does not contain document-referencing phrases (e.g., "refer to section", "as mentioned")
- Question is not a vague document-meta question (e.g., "what is the document about")
- Question is not a yes/no question
- Answer is not a near-repetition of the question (word overlap threshold: 80%)

---

## Part 2A — Custom evaluation (built from scratch)

---

## Deep dive — how each metric works

All four metrics use a **local LLM as judge** and operate on atomic claims.
No black boxes. Here is exactly what happens under the hood.

---

### Faithfulness

**What it answers:** "Did the LLM make up anything not present in the retrieved context?"

```
Faithfulness = |Supported Claims| / |Total Claims in Answer|
```

**Step-by-step:**
1. The judge LLM reads the answer and extracts every atomic factual claim
2. For each claim, the judge checks all retrieved chunks — supported or not?
3. Score = supported claims / total claims

**Worked example:**

> Question: *Why am I seeing Invalid username and Password?*  
> Answer: *Your account is locked due to multiple failed login attempts. Contact the administrator. You can also reset via email which takes 24 hours.*  
> Context: *Account locked due to failed attempts. Contact administrator to unlock.*

| Claim | Verdict |
|---|---|
| Account is locked | ✅ supported |
| Due to multiple failed login attempts | ✅ supported |
| Contact the administrator | ✅ supported |
| Reset via email takes 24 hours | ❌ not supported |

```
Faithfulness = 3 / 4 = 0.75
```

The LLM invented the 24-hour email reset detail — that 25% is hallucination.

**Score interpretation:**

| Score | Meaning | Action |
|---|---|---|
| 0.9 – 1.0 | Excellent | LLM is well-grounded. Almost no hallucination. |
| 0.7 – 0.9 | Good | Minor hallucination. Review low-scoring answers. |
| 0.5 – 0.7 | Poor | Significant hallucination. Tighten your system prompt. |
| < 0.5 | Critical | LLM is largely ignoring the context. Check system prompt. |

---

### Answer Relevancy

**What it answers:** "Did the LLM actually answer what was asked — or did it go off-topic?"

```
AnswerRelevancy = (1/N) × Σ cos_sim(embed(q), embed(qᵢ))

cos_sim(A, B) = (A · B) / (||A|| × ||B||)
```

Where `q` is the original question and `qᵢ` are N reverse-generated questions.

**Step-by-step:**
1. The judge LLM reads the answer and generates N questions the answer could respond to
2. Both the original question and each generated question are embedded via BGE-M3
3. Cosine similarity is computed between the original and each generated question
4. Score = mean of all similarity scores

**Worked example:**

> Question: *Why am I seeing Invalid username and Password?*  
> Answer: *Your account has been locked. Please contact your administrator to unlock it.*

| Generated question | Similarity |
|---|---|
| What causes an account lockout? | 0.91 |
| How do I fix a locked account? | 0.88 |
| Why is login failing? | 0.93 |

```
AnswerRelevancy = (0.91 + 0.88 + 0.93) / 3 = 0.907
```

**Why cosine similarity and not exact match?**  
Cosine similarity captures semantic meaning, not word overlap. *"Why is login failing?"* and *"Why am I seeing Invalid username and Password?"* share almost no words but are semantically close — cosine similarity reflects this. BLEU would score them near zero.

---

### Context Precision

**What it answers:** "Is the retriever fetching relevant chunks, or is it adding noise?"

```
ContextPrecision = |Relevant Retrieved Chunks| / |Total Retrieved Chunks|
```

Rank-aware variant (rewards relevant chunks appearing earlier):

```
ContextPrecision@K = Σ (Precisionₖ × relevanceₖ) / |Relevant Chunks|
```

Where `Precisionₖ = relevant chunks in top-k / k` and `relevanceₖ = 1` if chunk k is relevant.

**Step-by-step:**
1. For each retrieved chunk, the judge reads both the question and the chunk
2. The judge decides: relevant or not relevant to answering this question?
3. Score = relevant chunks / total chunks

**Worked example:**

> Question: *How do I unlock my account?*

| Chunk | Verdict |
|---|---|
| "Account locked. Contact administrator to unlock." | ✅ relevant |
| "Password reset steps: go to forgot password page..." | ✅ relevant |
| "System maintenance scheduled for Sunday 2AM." | ❌ not relevant |
| "New features added to dashboard in v2.1." | ❌ not relevant |

```
ContextPrecision = 2 / 4 = 0.50
```

Half the retrieved chunks were noise. Fix: tighten your similarity threshold, adjust chunk size, or tune your embedding model.

---

### Context Recall

**What it answers:** "Did the retriever find *all* the information needed to answer correctly?"

```
ContextRecall = |GT Claims Covered by Context| / |Total GT Claims|
```

**Step-by-step:**
1. The judge LLM splits the **ground truth** answer into atomic claims
2. For each claim, check whether at least one retrieved chunk covers it
3. Score = covered claims / total claims

**Worked example:**

> Ground truth: *Account is locked. Enter correct credentials. Contact admin if locked. Wait 30 minutes after 5 failed attempts.*

| Claim | Covered? |
|---|---|
| Account is locked | ✅ chunk 1 |
| Enter correct credentials | ✅ chunk 1 |
| Contact admin if locked | ✅ chunk 2 |
| Wait 30 minutes after 5 failed attempts | ❌ not in any chunk |

```
ContextRecall = 3 / 4 = 0.75
```

The retriever missed one important fact. Fix: expand your knowledge base, lower the similarity threshold, or increase top-k retrieved chunks.

---

### Reading all four scores together

| Pattern | What it means |
|---|---|
| Low faithfulness, high recall | Retriever is finding the right chunks but LLM is hallucinating anyway |
| High faithfulness, low recall | LLM stays grounded but retriever is missing key information |
| Low precision, high recall | Retriever is fetching everything relevant — plus a lot of noise |
| High precision, low recall | Retriever is accurate but too conservative — missing important chunks |
| All four high | Your RAG pipeline is working well end-to-end |

### Metrics

All scores are in the range `[0.0, 1.0]`. Higher is better.

| Metric | Description |
|---|---|
| **Faithfulness** | Whether the answer's claims are supported by the retrieved contexts |
| **Answer Relevancy** | Semantic similarity between the question and the answer (via embeddings) |
| **Context Precision** | How relevant the retrieved contexts are to the question |
| **Context Recall** | How well the retrieved contexts cover the ground truth answer |

### Usage

**1. Ensure Ollama is running**

```bash
ollama serve
```

**2. Start your RAG API**

```bash
uvicorn your_rag_app:app --host 127.0.0.1 --port 8000
```

The RAG API must accept `POST /ask` with body `{"question": "..."}` and return:

```json
{
  "answer": "...",
  "contexts": ["chunk 1", "chunk 2", "..."]
}
```
## Adapting to Your RAG Pipeline

The evaluation pipeline calls your RAG system inside the function responsible for fetching responses
(e.g., `get_rag_response()` or the API call block in the evaluation script).

If your RAG API differs from the default setup, you can modify this function accordingly.

---

##  What You May Need to Adjust

* **Endpoint** → Change the API route (e.g., `/ask` → `/generate`)
* **Request Payload** → Update input fields (e.g., `question` → `query`)
* **Response Mapping** → Extract the correct fields from your API response

---

## Expected Output Format

Your function must return:

* **`ans`** → `str`

  * The final generated answer from your RAG system

* **`ctxs`** → `List[str]`

  * A list of retrieved context chunks used for generation

---

##  Note

As long as you correctly map your API response to these two variables (`ans`, `ctxs`),
the evaluation pipeline will work seamlessly with your RAG system.

**3. Run evaluation**

```bash
python -m eval.main
```
**Sample output:**

```
faithfulness: 0.83
answer_relevancy: 0.72
context_precision: 1.0
context_recall: 0.78
```

> 📋 **Sample run** on a Geo-Intelligence Platform SOP document: faithfulness `0.83` · answer_relevancy `0.72` · context_precision `1.0` · context_recall `0.78`

### Output Files

**`per_query_results.csv`** — one row per question:

| question | response | ground_truth | context | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|---|---|---|
| What is RAG? | RAG combines retrieval with generation | Retrieval-Augmented Generation combines retrieval + LLM | "RAG (Retrieval-Augmented Generation) is a framework that enhances LLM responses by retrieving relevant documents from a knowledge base and using them as context during generation." | 0.83 | 0.72 | 1.00 | 0.78 |

---

**`average_results.csv`** — aggregated scores:

| avg_faithfulness | avg_answer_relevancy | avg_context_precision | avg_context_recall |
|---|---|---|---|
| 0.83 | 0.72 | 1.00 | 0.78 |

---

## Part 2B — DeepEval evaluation 

This is the part most DeepEval tutorials get wrong.

DeepEval's `evaluate()` function is async. When your judge model is a
large open-source model running locally via Ollama, async calls pile up
faster than the model can respond — and you get timeout errors.

The fix: use DeepEval's individual metric classes directly, called
**synchronously**, with tuned hyperparameters.

```bash
python -m eval.deepeval.main
```
---
## Output files

**`average_results.csv`** — aggregated averages across the full dataset.

| avg_faithfulness | avg_answer_relevancy | avg_context_precision | avg_context_recall |
|---|---|---|---|
| 0.83 | 0.72 | 1.00 | 0.78 |

---

## API Reference

### `GET /stats` — Response Schema

```json
{
  "total": 20,
  "type_distribution": {
    "procedural": 3,
    "compliance": 3,
    "role_responsibility": 2,
    "condition_trigger": 3,
    "definition_concept": 2,
    "exception_escalation": 3,
    "timeline_sla": 2,
    "multi_hop": 2
  },
  "avg_question_len": 14.2,
  "avg_answer_len": 38.7
}
```

---


