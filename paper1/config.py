"""
Central Configuration & Prompt Templates
Paper 1: Evaluation Invariance under Multilingual Query Transformation

Experimental Conditions (Derived with honest lineage from q_native or passage):
- C0: Native reference query (q_ref)
- C1: Same-language LLM paraphrase (q_paraphrase)
- C2: Native -> English -> Native MT Round-Trip (q_mt_roundtrip)
- C3: Native -> English -> Native LLM Round-Trip (q_llm_roundtrip)
- C4: Controlled native-language simplification/rewording (q_simplified)
- C5: Passage-conditioned synthetic query (q_synthetic)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
CACHE_DIR = DATA_DIR / "cache"
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"

for d in [DATA_DIR, SPLITS_DIR, CACHE_DIR, OUTPUTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

# OpenRouter & API Config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Strict Model Selection
MODEL_PRIMARY_TRANSFORM = os.getenv("MODEL_PRIMARY_TRANSFORM", "google/gemini-3.7-flash")
MODEL_SECONDARY_TRANSFORM = os.getenv("MODEL_SECONDARY_TRANSFORM", "qwen/qwen3.8-flash")

MODEL_PRIMARY_JUDGE = os.getenv("MODEL_PRIMARY_JUDGE", "google/gemini-3.8-flash")
MODEL_SECONDARY_JUDGE = os.getenv("MODEL_SECONDARY_JUDGE", "qwen/qwen3.8-flash")
MODEL_ARBITER = os.getenv("MODEL_ARBITER", "deepseek/deepseek-chat")

# Offline Retrieval Models
MT_MODEL_NAME = os.getenv("MT_MODEL_NAME", "ai4bharat/indictrans2-en-indic-1B")
EMBEDDING_MODEL_BGE = os.getenv("EMBEDDING_MODEL_BGE", "BAAI/bge-m3")
EMBEDDING_MODEL_E5_BASE = os.getenv("EMBEDDING_MODEL_E5_BASE", "intfloat/multilingual-e5-base")
EMBEDDING_MODEL_E5_LARGE = os.getenv("EMBEDDING_MODEL_E5_LARGE", "intfloat/multilingual-e5-large")
EMBEDDING_MODEL_LABSE = os.getenv("EMBEDDING_MODEL_LABSE", "sentence-transformers/LaBSE")
EMBEDDING_MODEL_QWEN3_06B = os.getenv("EMBEDDING_MODEL_QWEN3_06B", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_MODEL_QWEN3_4B = os.getenv("EMBEDDING_MODEL_QWEN3_4B", "Qwen/Qwen3-Embedding-4B")

# Seed & Sizing
EXPERIMENT_SEED = int(os.getenv("EXPERIMENT_SEED", "20260921"))
DEFAULT_LANGUAGES = os.getenv("DEFAULT_LANGUAGES", "id,th").split(",")
PILOT_SAMPLE_SIZE = int(os.getenv("PILOT_SAMPLE_SIZE", "150"))
FINAL_SAMPLE_SIZE = int(os.getenv("FINAL_SAMPLE_SIZE", "800"))

# Language Full Name Mappings
LANGUAGE_MAP = {
    "id": "Indonesian",
    "th": "Thai",
    "as": "Assamese",
    "bn": "Bengali",
    "hi": "Hindi",
    "or": "Odia",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
}

# -------------------------------------------------------------------------
# Transformation Prompt Templates (Honest Lineage from q_native or passage)
# -------------------------------------------------------------------------

# C1: Same-Language LLM Paraphrase
PROMPT_PARAPHRASE = """Rewrite the following {LANGUAGE} search query using different wording while preserving exactly the same information need.

Requirements:
- Preserve all named entities, numbers, dates, units, negation, and temporal/spatial constraints.
- Do not answer the query.
- Do not add explanations.
- Output ONLY the paraphrased query in {LANGUAGE} on a single line.

QUERY:
{Q_REF}"""

# C2 & C3: Forward Translation to English (Step 1 of Round-Trip)
PROMPT_TO_ENGLISH = """Translate the following {LANGUAGE} information-seeking query into clear, natural English.

Requirements:
- Preserve the exact information need, entities, numbers, and constraints.
- Do not answer the question.
- Output ONLY the English query on a single line.

QUERY:
{Q_REF}"""

# C2 & C3: Back-Translation to Native Language (Step 2 of Round-Trip)
PROMPT_BACK_TO_NATIVE = """Translate the following English search query back into {LANGUAGE}.

Requirements:
- Translate accurately into {LANGUAGE} script.
- Preserve named entities, numbers, dates, and query intent.
- Do not answer the question.
- Output ONLY the translated query in {LANGUAGE} on a single line.

QUERY:
{Q_EN}"""

# C4: Controlled Simplification & Natural Rewording
PROMPT_SIMPLIFICATION = """Simplify and clarify the following {LANGUAGE} search query for a search engine while keeping its core information need intact.

Requirements:
- Use clear, direct, and concise {LANGUAGE} vocabulary.
- Preserve essential entities, numbers, and key concepts.
- Do not answer the query.
- Output ONLY the simplified query in {LANGUAGE} on a single line.

QUERY:
{Q_REF}"""

# C5: Passage-Conditioned Synthetic Query (Benchmark Construction Recipe)
PROMPT_SYNTHETIC = """Create one natural, standalone information-seeking search query in {LANGUAGE} that is directly answered by the provided passage.

Rules:
- The question must be answerable from the passage.
- Phrased naturally in {LANGUAGE} as a real user search query.
- Do not answer the question.
- Do not include explanations, labels, or notes.
- Output ONLY the generated question on a single line.

PASSAGE:
{PASSAGE}

TARGET ANSWER CONTEXT:
{ANSWER}"""

# -------------------------------------------------------------------------
# Dual-Judge Prompt Templates
# -------------------------------------------------------------------------

# Judge for Query-Preserving Transformations (C1, C2, C3, C4)
PROMPT_JUDGE_VALIDATION = """You are an expert evaluator assessing the quality of a transformed multilingual information-seeking query.

Original Reference Query ({LANGUAGE}): {Q_REF}
Transformed Query ({LANGUAGE}): {Q_TRANSFORMED}

Evaluate whether the Transformed Query preserves the exact information need and meaning of the Reference Query.

Score on the following dimensions:
1. Intent Preservation (1-5): Does the transformed query seek the same information?
2. Entity Fidelity (1-5): Are key entities, numbers, dates, and constraints preserved?
3. Fluency & Naturalness (1-5): Is the query grammatically natural in {LANGUAGE}?
4. Overall Validity (YES/NO): Is this transformed query a valid, faithful alternative query?

Respond strictly in pure JSON format:
{{
  "intent_preservation_score": 5,
  "entity_fidelity_score": 5,
  "fluency_score": 5,
  "overall_validity": "YES",
  "reason": "..."
}}"""

# Judge for Passage-Conditioned Synthetic Queries (C5)
PROMPT_JUDGE_SYNTHETIC = """You are an expert evaluator assessing a passage-conditioned synthetic query.

Reference Query ({LANGUAGE}): {Q_REF}
Synthetic Query ({LANGUAGE}): {Q_TRANSFORMED}
Passage Content: {PASSAGE}

Evaluate:
1. Passage Grounding (1-5): Is the synthetic query genuinely answerable by the passage?
2. Topic Alignment (1-5): Is the synthetic query topical and relevant to the domain of the reference query?
3. Fluency (1-5): Is the query natural and well-formed in {LANGUAGE}?
4. Overall Validity (YES/NO): Is this synthetic query a high-quality, valid benchmark test query for this passage?

Respond strictly in pure JSON format:
{{
  "intent_preservation_score": 5,
  "entity_fidelity_score": 5,
  "fluency_score": 5,
  "overall_validity": "YES",
  "reason": "..."
}}"""
