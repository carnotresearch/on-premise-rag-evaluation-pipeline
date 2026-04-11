from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import requests
from pypdf import PdfReader
from docx import Document
import os
import time
import re
import logging
import concurrent.futures
import hashlib
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()
from data_creation_app.llm_service import call_llm

app = Flask(__name__)

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------
# API CONFIG
# -------------------------


HEADERS = {
    "Content-Type": "application/json"
}
# 🔥 NEW: runtime override model (user selected)
# USER_SELECTED_MODEL = None

# -------------------------
# SOP QUESTION TYPES
# -------------------------
SOP_QUESTION_TYPES = {
    "procedural": {
        "description": "Step-by-step process questions",
        "instruction": "Ask about HOW to perform a process, sequence of steps, or procedure described in the SOP.",
        "example_q": "What are the steps to escalate a critical incident?",
        "example_a": "To escalate a critical incident: (1) Notify the on-call engineer within 5 minutes, (2) Create a P1 ticket in the system, (3) Page the manager if unresolved after 15 minutes."
    },
    "compliance": {
        "description": "Rules, regulations, must-do / must-not-do",
        "instruction": "Ask about mandatory requirements, prohibitions, regulatory rules, or compliance obligations mentioned in the SOP.",
        "example_q": "What is prohibited when handling customer PII data?",
        "example_a": "Employees are strictly prohibited from storing customer PII on personal devices, sharing credentials, or transmitting data over unencrypted channels."
    },
    "role_responsibility": {
        "description": "Who is responsible for what",
        "instruction": "Ask about ownership, accountability, roles, teams, or who must perform a specific action in the SOP.",
        "example_q": "Who is responsible for approving vendor onboarding requests?",
        "example_a": "The Procurement Manager and the Information Security team jointly approve vendor onboarding requests after completing due diligence checks."
    },
    "condition_trigger": {
        "description": "When/under what conditions something happens",
        "instruction": "Ask about triggers, conditions, thresholds, or scenarios that activate a process or policy described in the SOP.",
        "example_q": "Under what conditions should a change request be treated as emergency?",
        "example_a": "A change request is treated as emergency when it addresses a production outage, a critical security breach, or a regulatory deadline within 24 hours."
    },
    "definition_concept": {
        "description": "What something means or is defined as in the document",
        "instruction": "Ask about the definition, meaning, or explanation of a term, acronym, or concept as defined in the SOP.",
        "example_q": "How does this SOP define a 'Major Incident'?",
        "example_a": "According to the SOP, a Major Incident is any unplanned service disruption affecting more than 100 users or causing revenue loss exceeding $10,000 per hour."
    },
    "exception_escalation": {
        "description": "Exceptions, edge cases, and escalation paths",
        "instruction": "Ask about what happens in exceptional cases, deviations from standard process, or escalation procedures described in the SOP.",
        "example_q": "What should be done if the standard approval chain is unavailable during an emergency?",
        "example_a": "If the standard approval chain is unavailable, the requestor must contact the designated backup approver listed in Appendix B and document the exception in the incident log."
    },
    "timeline_sla": {
        "description": "Timeframes, deadlines, SLAs, frequencies",
        "instruction": "Ask about specific timeframes, deadlines, response times, review frequencies, or SLA commitments mentioned in the SOP.",
        "example_q": "What is the maximum time allowed to resolve a Severity 2 incident?",
        "example_a": "Severity 2 incidents must be resolved within 4 business hours. If unresolved, automatic escalation to the department head is triggered at the 3-hour mark."
    },
    "multi_hop": {
        "description": "Complex questions requiring synthesis across the chunk",
        "instruction": "Ask a question that requires combining TWO or more facts from different parts of the text to answer correctly. The answer must not be a single isolated sentence from the text.",
        "example_q": "How does the incident classification affect both the response timeline and the escalation path?",
        "example_a": "P1 incidents require a 15-minute response and are directly escalated to the VP of Engineering, while P2 incidents allow a 1-hour response with escalation to the Engineering Manager only if unresolved."
    }
}


# -------------------------
# STRONG SOP PROMPT
# -------------------------
def build_sop_prompt(chunk: str, q_type: str) -> str:
    qt = SOP_QUESTION_TYPES[q_type]

    return f"""You are an expert QA dataset creator specializing in Standard Operating Procedure (SOP) documents.
Your task is to generate ONE high-quality question-answer pair of type: "{q_type}".

=== QUESTION TYPE DEFINITION ===
Type: {q_type}
Purpose: {qt["description"]}
How to write: {qt["instruction"]}

=== EXAMPLE OF GOOD OUTPUT ===
{{
  "question": "{qt["example_q"]}",
  "ground_truth": "{qt["example_a"]}",
  "type": "{q_type}"
}}

=== STRICT RULES — VIOLATIONS WILL CAUSE REJECTION ===

GROUNDING (CRITICAL):
- Every word in ground_truth MUST be directly supported by the TEXT below
- NEVER use outside knowledge, assumptions, or inferences beyond the text
- If the text does not contain enough information for this type, still extract what's relevant

SELF-CONTAINED OUTPUT:
- Do NOT say "refer to section X", "as mentioned above", "see document"
- The question and answer must be fully understandable without reading the document
- Do NOT use pronouns like "it", "they", "this" without clear referent

QUALITY STANDARDS:
- Question must be specific, not vague ("What should be done when X happens?" not "What is the process?")
- Ground truth must be at minimum 15 words and contain concrete, actionable information
- No one-word or one-phrase answers
- No yes/no questions
- Answers must state WHO, WHAT, WHEN, HOW — as applicable from the text

ANTI-PATTERNS (THESE WILL BE REJECTED):
- "The document states..." or "According to the SOP..."
- Answers containing only a section title
- Repeating the question in the answer
- Generic answers that could apply to any document

OUTPUT FORMAT:
- Return ONLY valid JSON
- No markdown fences, no explanation, no preamble
- Output must start with {{ and end with }}
- Keys must be exactly: "question", "ground_truth", "type"

=== SOP TEXT ===
{chunk}

=== YOUR OUTPUT (JSON only) ==="""


# -------------------------
# FILE READERS
# -------------------------
def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text


def read_docx(path: str) -> str:
    doc = Document(path)
    text = []
    for p in doc.paragraphs:
        if p.text.strip():
            text.append(p.text.strip())
    return "\n".join(text)


def read_file(path: str) -> str:
    if path.endswith(".pdf"):
        return read_pdf(path)
    elif path.endswith(".docx"):
        return read_docx(path)
    else:
        raise ValueError(f"Unsupported format: {path}")


# -------------------------
# SMART SEMANTIC CHUNKING
# (paragraph-aware, no mid-sentence cuts)
# -------------------------
def semantic_chunk(text: str, min_size: int = 300, max_size: int = 900) -> list[str]:
    """
    Splits text on paragraph/section boundaries first,
    then merges small paragraphs and splits oversized ones.
    Preserves meaning far better than blind character slicing.
    """
    # Split on double newline (paragraph boundary)
    raw_paragraphs = re.split(r"\n{2,}", text.strip())
    
    chunks = []
    buffer = ""

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this para stays under max_size, buffer it
        if len(buffer) + len(para) < max_size:
            buffer = (buffer + "\n\n" + para).strip()
        else:
            # Save current buffer if big enough
            if len(buffer) >= min_size:
                chunks.append(buffer)
            # Start new buffer
            buffer = para

        # If buffer is already big, flush it
        if len(buffer) >= max_size:
            chunks.append(buffer)
            buffer = ""

    if buffer and len(buffer) >= min_size:
        chunks.append(buffer)

    logger.info(f"📄 Chunked into {len(chunks)} semantic chunks (min={min_size}, max={max_size})")
    return chunks


# -------------------------
# JSON PARSER
# -------------------------
def safe_json_parse(content: str) -> dict | None:
    try:
        content = re.sub(r"```json|```", "", content).strip()
        return json.loads(content)
    except Exception:
        pass

    # Fallback: find first { ... } block
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    logger.warning("⚠️  JSON parse failed completely")
    return None


# -------------------------
# VALIDATION (strict)
# -------------------------
def validate_qa(item: dict) -> bool:
    if not isinstance(item, dict):
        return False

    required = {"question", "ground_truth", "type"}
    if not all(k in item for k in required):
        logger.debug(f"Missing keys: {set(item.keys())}")
        return False

    q = item["question"].strip()
    a = item["ground_truth"].strip()

    if len(q) < 15:
        return False
    if len(a.split()) < 12:
        return False
    if item["type"] not in SOP_QUESTION_TYPES:
        return False

    return True


# -------------------------
# QUALITY FILTER (SOP-specific)
# -------------------------
REJECT_PHRASES_IN_ANSWER = [
    "section", "refer to", "as mentioned", "see above", "the document",
    "the sop states", "as described", "follow steps", "refer the",
    "in the above", "the above", "per the document", "as per document"
]

REJECT_PHRASES_IN_QUESTION = [
    "title of the document", "name of the document", "what is the document about"
]


def is_high_quality(item: dict) -> bool:
    q = item["question"].lower().strip()
    a = item["ground_truth"].lower().strip()

    for phrase in REJECT_PHRASES_IN_ANSWER:
        if phrase in a:
            logger.debug(f"Rejected answer for phrase: '{phrase}'")
            return False

    for phrase in REJECT_PHRASES_IN_QUESTION:
        if phrase in q:
            logger.debug(f"Rejected question for phrase: '{phrase}'")
            return False

    # Reject yes/no questions
    if re.match(r"^(is |are |was |were |do |does |did |has |have |can |should |would |will )", q):
        logger.debug("Rejected yes/no question")
        return False

    # Reject if answer word count too low
    if len(a.split()) < 12:
        return False

    # Reject if answer is just a repetition of the question
    q_words = set(re.findall(r"\w+", q))
    a_words = set(re.findall(r"\w+", a))
    overlap = q_words & a_words
    if len(overlap) / max(len(q_words), 1) > 0.80:
        logger.debug("Rejected: answer too similar to question")
        return False

    return True


# -------------------------
# DEDUPLICATION (hash-based)
# -------------------------
def deduplicate(dataset: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for item in dataset:
        key = hashlib.md5(item["question"].strip().lower().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


# -------------------------
# API CALL (with retry + backoff)



# -------------------------
# SINGLE QA GENERATOR
# -------------------------
def generate_single_qa(chunk: str, q_type: str, model_name: str, attempt_label: str = "") -> dict | None:
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise JSON-only output machine. "
                    "You NEVER output anything except valid JSON. "
                    "No explanations. No markdown. Just JSON."
                )
            },
            {
                "role": "user",
                "content": build_sop_prompt(chunk, q_type)
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.4,   # lower = more faithful, less hallucination
            "num_predict": 300,   # enough for a full answer
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }
    }

    result = call_llm(payload)
    if not result:
        logger.warning(f"❌ No result [{attempt_label}]")
        return None

    content = result.get("message", {}).get("content", "")
    data = safe_json_parse(content)

    if not data:
        logger.debug(f"Parse fail [{attempt_label}]: {content[:100]}")
        return None

    # Force correct type in case model ignores it
    data["type"] = q_type

    if validate_qa(data) and is_high_quality(data):
        logger.info(f"✅ [{attempt_label}] {q_type}: {data['question'][:60]}...")
        return data
    else:
        logger.debug(f"Quality reject [{attempt_label}]: {data.get('question', '')[:60]}")
        return None


# -------------------------
# BALANCED TYPE SCHEDULER
# Ensures all SOP question types are evenly represented
# -------------------------
def build_generation_schedule(chunks: list[str], total_pairs: int) -> list[tuple[str, str, str]]:
    """
    Returns list of (chunk, q_type, label) tuples.
    Distributes question types evenly across chunks.
    """
    types = list(SOP_QUESTION_TYPES.keys())
    schedule = []

    for i in range(total_pairs):
        chunk = chunks[i % len(chunks)]
        q_type = types[i % len(types)]
        label = f"pair-{i+1}/{total_pairs}"
        schedule.append((chunk, q_type, label))

    return schedule


# -------------------------
# PARALLEL DATASET GENERATOR
# -------------------------
def generate_dataset(text: str, total_pairs: int, model_name: str) -> list[dict]:
    dataset = []
    chunks = semantic_chunk(text)

    if not chunks:
        logger.error("No chunks extracted from document!")
        return []

    logger.info(f"📊 Target: {total_pairs} QA pairs from {len(chunks)} chunks")
    schedule = build_generation_schedule(chunks, total_pairs * 2)  # 2x buffer for rejections

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(generate_single_qa, chunk, q_type, model_name, label)
            for chunk, q_type, label in schedule
        }

        for future in concurrent.futures.as_completed(futures):
            result = future.result()

            if result:
                dataset.append(result)
                dataset = deduplicate(dataset)

                logger.info(f"📈 Progress: {len(dataset)}/{total_pairs} unique QA pairs")

                # Checkpoint save
                with open("backup_dataset.json", "w") as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)

            if len(dataset) >= total_pairs:
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break

    final = dataset[:total_pairs]
    logger.info(f"🎯 Final dataset: {len(final)} pairs")
    _log_type_distribution(final)
    return final


# -------------------------
# TYPE DISTRIBUTION LOGGER
# -------------------------
def _log_type_distribution(dataset: list[dict]):
    dist = defaultdict(int)
    for item in dataset:
        dist[item["type"]] += 1
    logger.info("📊 Question type distribution:")
    for t, count in sorted(dist.items()):
        logger.info(f"   {t:25s}: {count}")


# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    file = request.files.get("file")
    pairs = int(request.form.get("pairs", 20))
    model_name = request.form.get("model")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    path = "temp_" + file.filename
    file.save(path)

    try:
        text = read_file(path)
        if len(text.strip()) < 100:
            return jsonify({"error": "Document too short or unreadable"}), 400

        dataset = generate_dataset(text, pairs, model_name)
    except Exception as e:
        logger.exception("Generation failed")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)

    with open("temp_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    return redirect(url_for("review"))


@app.route("/review")
def review():
    if not os.path.exists("temp_dataset.json"):
        return "No dataset found. Please generate first.", 404

    with open("temp_dataset.json") as f:
        dataset = json.load(f)

    return render_template("review.html", dataset=dataset)


@app.route("/save", methods=["POST"])
def save():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    with open("eval/evaluation_dataset.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return jsonify({"status": "saved", "count": len(data)})


@app.route("/stats")
def stats():
    """Quick stats endpoint for the current dataset."""
    if not os.path.exists("temp_dataset.json"):
        return jsonify({"error": "No dataset"}), 404

    with open("temp_dataset.json") as f:
        dataset = json.load(f)

    dist = defaultdict(int)
    for item in dataset:
        dist[item["type"]] += 1

    return jsonify({
        "total": len(dataset),
        "type_distribution": dict(dist),
        "avg_question_len": round(
            sum(len(i["question"].split()) for i in dataset) / max(len(dataset), 1), 1
        ),
        "avg_answer_len": round(
            sum(len(i["ground_truth"].split()) for i in dataset) / max(len(dataset), 1), 1
        )
    })


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)