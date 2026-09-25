# Paper 1 Execution Protocol
## Evaluation Invariance under Multilingual Benchmark Transformation

### 0. One-sentence objective

Test whether common low-resource multilingual benchmark transformations (translation, post-editing, paraphrasing, and passage-conditioned synthetic question generation) preserve the **scientific conclusions** of retrieval evaluation, not merely sentence-level semantic quality.

---

## 1. Scientific question

### Main research question

> When a multilingual retrieval benchmark is transformed, do researchers still reach the same conclusions about retrieval systems?

### Central proposition to test

Let a benchmark be:

\[
D=\{(q_i, C, R_i)\}_{i=1}^{N}
\]

where:

- \(q_i\): information need / query,
- \(C\): fixed retrieval corpus,
- \(R_i\): relevance judgments.

Let \(T\) transform only the query:

\[
q_i' = T(q_i)
\]

The usual implicit assumption is:

\[
q_i \approx q_i'
\quad\Longrightarrow\quad
Eval(D) \approx Eval(T(D))
\]

This paper tests whether that implication actually holds.

### Do NOT assume the answer

The study must permit all outcomes:

1. transformations preserve evaluation almost perfectly;
2. only some transformations break evaluation;
3. linguistic quality predicts evaluation validity;
4. linguistic quality does **not** predict evaluation validity;
5. benchmark transformation changes absolute scores but not model ordering;
6. benchmark transformation changes model ordering / pairwise conclusions.

No result should be manufactured as a preferred outcome.

---

## 2. Hypotheses

### H1 — Information-need preservation

High-quality transformations preserve the original query intent.

### H2 — Absolute evaluation stability

For retrieval model \(m\), performance under transformation \(T\) is close to the reference:

\[
\Delta_{m,T}=S(m,T(D))-S(m,D)
\]

### H3 — Pairwise conclusion invariance

For models \(m_a,m_b\), the sign of the performance difference is preserved:

\[
sign(S(m_a,D)-S(m_b,D))
=
sign(S(m_a,T(D))-S(m_b,T(D)))
\]

### H4 — System-order invariance

The ranking of retrieval systems is preserved across benchmark transformations.

### H5 — Semantic quality is not necessarily evaluation validity

A transformation may receive high semantic-preservation scores while still changing retrieval-system conclusions.

### H6 — Construction procedures may favor retrieval families

Some transformations may systematically advantage lexical, dense, or late-interaction retrieval systems.

---

# 3. Dataset

## Primary dataset

Use:

```text
ai4bharat/IndicMSMARCO
```

Mandatory language subsets:

```text
as  # Assamese
bn  # Bengali
```

The dataset currently exposes approximately 1,000 examples per language and retains `query_id` values from the source MS MARCO examples.

### Final study size

Target:

```text
Assamese: 800 evaluation queries
Bengali:  800 evaluation queries
```

Use the same query IDs across languages whenever possible.

Recommended split:

```text
100 queries/language = pilot + prompt development
800 queries/language = frozen final evaluation
remaining usable items = diagnostics/replacement reserve
```

If fewer than 800 valid common IDs remain after filtering:

```text
minimum acceptable final N = 500 per language
```

Do not select examples based on retrieval results.

### Sampling

Use deterministic seed:

```text
SEED = 20260921
```

Stratify when feasible by `query_type` so that the sample is not dominated by one question type.

Store the exact selected query IDs in:

```text
data/splits/pilot_query_ids.json
data/splits/final_query_ids.json
```

Never resample after looking at results.

---

# 4. Reference query and source English query

For every IndicMSMARCO `query_id`:

1. keep the official target-language query as `q_ref`;
2. recover the corresponding original English MS MARCO query using `query_id`;
3. store both permanently.

Schema:

```json
{
  "query_id": "10219",
  "language": "as",
  "q_en_source": "...",
  "q_ref": "...",
  "positive_passage_id": "...",
  "positive_passage": "...",
  "answer": "...",
  "query_type": "...",
  "dataset_source": "ai4bharat/IndicMSMARCO"
}
```

### Critical rule

`q_ref` is the reference multilingual benchmark condition.

Do not call it “gold human translation” unless the dataset documentation explicitly supports that wording. In the paper use safer wording:

> reference / benchmark-provided target-language query

---

# 5. Query transformation conditions

Generate exactly the following conditions.

## C0 — REF

Benchmark-provided target-language query.

```text
q_ref
```

No modification.

---

## C1 — MT

Translate the source English query into the target language using:

```text
ai4bharat/indictrans2-en-indic-1B
```

Use deterministic decoding:

```text
num_beams = 5
do_sample = false
```

Save exact model revision / git commit.

---

## C2 — LLM-TRANSLATE

Translate `q_en_source` directly with one strong paid LLM.

Prompt:

```text
Translate the following information-seeking query into {LANGUAGE}.

Requirements:
- Preserve the exact information need.
- Preserve named entities, numbers, dates, units, negation, comparison direction,
  temporal constraints, spatial constraints, and requested answer type.
- Do not answer the question.
- Do not add explanations.
- Produce only the translated query.

SOURCE QUERY:
{QUERY}
```

Generation:

```text
temperature = 0
top_p = 1
```

Record provider, exact model ID, date, prompt hash, and raw response.

---

## C3 — MT+POSTEDIT

Input = C1 output.

Use a strong LLM to correct the MT output while preserving exact meaning.

Prompt:

```text
You are editing a machine-translated information-seeking query in {LANGUAGE}.

Correct grammar, fluency, terminology, and unnatural phrasing while preserving
the exact original information need.

You MUST preserve:
- named entities,
- numbers and units,
- dates and time constraints,
- negation,
- comparison direction,
- spatial relations,
- requested answer type,
- specificity.

Do not answer the query.
Do not add information.
Output only the corrected query.

ORIGINAL ENGLISH:
{Q_EN}

MACHINE TRANSLATION:
{Q_MT}
```

Temperature 0.

---

## C4 — PARAPHRASE

Paraphrase `q_ref` **within the same language**.

Prompt:

```text
Rewrite the following {LANGUAGE} search query using different wording while
preserving exactly the same information need.

Preserve all entities, numbers, dates, units, negation, temporal/spatial
constraints, comparison direction, and answer type.

Do not answer it.
Do not make it more general or more specific.
Output only one paraphrased query.

QUERY:
{Q_REF}
```

Temperature 0.

---

## C5 — SYNTHETIC

Passage-conditioned synthetic-query condition.

Input to generator:

- benchmark positive passage,
- benchmark answer when available.

Do NOT give the reference query to the generator.

Prompt:

```text
Create one natural information-seeking question in {LANGUAGE} that can be
answered by the provided passage and whose intended answer is exactly the
provided answer.

Rules:
- The question must be answerable from the passage.
- Target the provided answer.
- Do not copy a full sentence from the passage.
- Do not include the answer inside the question.
- Do not add facts not supported by the passage.
- Produce only the question.

PASSAGE:
{PASSAGE}

TARGET ANSWER:
{ANSWER}
```

This condition intentionally models a common synthetic benchmark construction process.

---

# 6. Provenance schema

Every generated query variant must be saved as JSONL.

Required fields:

```json
{
  "query_id": "...",
  "language": "as",
  "condition": "C3_MT_POSTEDIT",
  "query_text": "...",
  "parent_condition": "C1_MT",
  "source_query_id": "...",
  "source_english": "...",
  "reference_query": "...",
  "positive_passage_id": "...",
  "generator_type": "llm",
  "generator_provider": "...",
  "generator_model": "...",
  "generator_temperature": 0,
  "prompt_version": "v1",
  "prompt_sha256": "...",
  "generated_at": "...",
  "dataset_revision": "...",
  "model_revision": "..."
}
```

Nothing should be overwritten. Raw generations and cleaned outputs must both be retained.

---

# 7. Human-free validation protocol

No human validation is required for this fast paper, but **do not call the data human-verified**.

Use two independent strong judge-model families, with a third judge only for disagreement.

Example logic:

```text
Judge A = strong OpenAI model available to the project
Judge B = strong Gemini model available to the project
Judge C = independent Qwen/DeepSeek-class model, only for tie-breaking
```

Exact model IDs must be logged at runtime.

## Judge task

For C1–C4, compare transformed query against `q_ref`.

For C5, compare synthetic query against:

- positive passage,
- target answer,
- original/reference information need.

### Judge output format

Strict JSON:

```json
{
  "semantic_equivalence": 0,
  "answer_type_preserved": 0,
  "entities_preserved": 0,
  "numbers_units_preserved": 0,
  "negation_preserved": 0,
  "temporal_spatial_constraints_preserved": 0,
  "specificity_preserved": 0,
  "answerable_from_same_evidence": 0,
  "fatal_error": false,
  "error_types": [],
  "confidence": 0.0
}
```

All binary fields are 0/1.

### Fatal-error categories

Allowed labels:

```text
ENTITY_CHANGED
NUMBER_CHANGED
UNIT_CHANGED
NEGATION_CHANGED
TEMPORAL_CONSTRAINT_CHANGED
SPATIAL_CONSTRAINT_CHANGED
ANSWER_TYPE_CHANGED
SPECIFICITY_CHANGED
QUESTION_BECAME_UNANSWERABLE
QUESTION_ANSWER_LEAK
UNSUPPORTED_INFORMATION
LANGUAGE_FAILURE
OTHER
```

### Consensus rule

Accept transformation as semantically valid when:

```text
Judge A semantic_equivalence = 1
AND
Judge B semantic_equivalence = 1
AND
fatal_error = false for both
```

If judges disagree:

```text
call Judge C
majority vote decides
```

Keep both:

```text
ALL_SET      = every generated item
VALIDATED_SET = consensus-valid items only
```

Primary retrieval analysis must report **both**:

1. all transformations;
2. consensus-valid transformations.

This is important because it separates obvious translation failures from the deeper question:

> Can apparently valid transformations still alter evaluation?

---

# 8. Retrieval corpus

## Non-negotiable principle

For a language, **the retrieval corpus and relevance judgments must remain frozen across every query condition**.

Only the query text changes.

Never rebuild a corpus per condition.

---

## Preferred corpus setup

Use the language-specific INDIC-MARCO collection from:

```text
saifulhaq9/indicmarco
```

Languages:

```text
asm_Beng
ben_Beng
```

Use the same target-language corpus for all conditions in that language.

### If full corpus is too large

Create a fixed pooled evaluation corpus.

For each reference query:

1. include its known positive passage;
2. obtain hard negatives from the language corpus using BM25 on **reference queries only**;
3. take top 20 non-relevant passages;
4. union all positives and negatives across all selected queries;
5. deduplicate by passage ID;
6. freeze this pool before running transformed queries.

Recommended pool:

```text
minimum: 10,000 unique passages/language
preferred: 20,000–50,000 passages/language
```

Never generate hard negatives independently for C1–C5 because that would change the test environment.

Save:

```text
data/corpus/as_corpus_frozen.parquet
data/corpus/bn_corpus_frozen.parquet
```

---

# 9. Retrieval systems

Goal: compare **retrieval families**, not twenty nearly identical models.

Mandatory systems:

## R1 — BM25

Implementation:

```text
Pyserini / Lucene
```

Use one frozen tokenization/index configuration per language.

Tune nothing on final data.

Record:

```text
k1
b
analyzer/tokenizer
```

If using default BM25, report exact defaults.

---

## R2 — multilingual-e5-large

```text
intfloat/multilingual-e5-large
```

Use model-recommended query/passage formatting.

Cosine similarity.

Normalize embeddings.

---

## R3 — BGE-M3 dense

```text
BAAI/bge-m3
```

Dense retrieval mode.

Normalize embeddings.

---

## R4 — GTE multilingual

```text
Alibaba-NLP/gte-multilingual-base
```

Use retrieval configuration recommended by model card.

---

## R5 — Indic-ColBERT

```text
saifulhaq9/indiccolbert
```

Use the Assamese model for Assamese and Bengali model for Bengali.

This gives a language-specific late-interaction retrieval family.

---

## Optional R6 — BGE-M3 sparse / hybrid

Only add if implementation is stable before the final run.

Do not risk the paper for this system.

---

# 10. Reproducible model handling

For every Hugging Face model, write to:

```text
artifacts/model_manifest.json
```

Fields:

```json
{
  "model_name": "...",
  "revision": "...",
  "library": "...",
  "library_version": "...",
  "dtype": "...",
  "device": "...",
  "max_length": 512,
  "pooling": "...",
  "normalization": true
}
```

Cache document embeddings once.

Never recompute passage embeddings per condition.

---

# 11. Retrieval output

For every:

```text
language × query_condition × retrieval_system
```

save TREC-compatible runs:

```text
runs/{language}/{condition}/{system}.trec
```

Also save parquet:

```text
query_id
rank
doc_id
score
system
condition
language
```

Retrieve at least top 100.

---

# 12. Standard IR metrics

Primary metric:

\[
nDCG@10
\]

Secondary metrics:

\[
MRR@10
\]

\[
Recall@10
\]

\[
Recall@100
\]

\[
Success@10
\]

If graded qrels are genuinely available, use the graded relevance levels for nDCG.

If only binary relevance exists, do not invent grades.

Implementation:

```text
ir_measures
```

Cross-check core metrics with `pytrec_eval` if available.

---

# 13. Core new evaluation concepts

## 13.1 Absolute Score Shift (ASS)

For model \(m\), transformation \(T\), metric \(S\):

\[
ASS_{m,T}=S(m,T(D))-S(m,D)
\]

Also report relative shift:

\[
RASS_{m,T}=
\frac{S(m,T(D))-S(m,D)}
{\max(S(m,D),\epsilon)}
\]

Use \(\epsilon=10^{-8}\).

Interpretation:

- near 0 = stable absolute evaluation;
- positive = transformation makes evaluation easier for model;
- negative = transformation makes it harder.

---

## 13.2 Pairwise Conclusion Preservation (PCP)

For models \(a,b\):

\[
\Delta_{ab}^{ref}=S(a,D)-S(b,D)
\]

\[
\Delta_{ab}^{T}=S(a,T(D))-S(b,T(D))
\]

Define:

\[
PCP_{ab,T}
=
\mathbb{1}
[
sign_\tau(\Delta_{ab}^{ref})
=
sign_\tau(\Delta_{ab}^{T})
]
\]

where \(sign_\tau\) treats very small differences as ties.

Use:

\[
\tau = 0.005
\]

for nDCG scale by default, **but also run sensitivity analysis**:

```text
tau ∈ {0, 0.0025, 0.005, 0.01}
```

Aggregate:

\[
PCP_T=
\frac{2}{M(M-1)}
\sum_{a<b} PCP_{ab,T}
\]

This is the main evaluation-invariance quantity.

---

## 13.3 Statistically-aware Pairwise Conclusion Preservation

Raw sign flips can overreact to noise.

Therefore define the primary inferential version using paired bootstrap significance.

For every model pair \(a,b\), on reference data:

```text
bootstrap 10,000 resamples of query IDs
estimate 95% CI for metric difference
```

Classify reference relation:

```text
A>B  if CI excludes 0 positively
B>A  if CI excludes 0 negatively
TIE  otherwise
```

Repeat for transformed condition.

Then compute:

\[
PCP_T^{sig}
=
\frac{\# \text{pairwise relation labels preserved}}
{\# \text{model pairs}}
\]

Use this as the **primary conclusion-preservation metric**.

The threshold-based PCP is secondary/sensitivity analysis.

---

## 13.4 Rank-order stability

For each transformation:

\[
\rho_T = Spearman(rank_{ref}, rank_T)
\]

and

\[
\tau_T = Kendall\ \tau_b(rank_{ref}, rank_T)
\]

Because there are only ~5–6 systems, these are descriptive, not the sole inferential result.

Never build the paper around Spearman alone.

---

## 13.5 Query Difficulty Stability (QDS)

For every query \(i\), define:

\[
d_i =
\frac{1}{M}
\sum_m metric_i(m)
\]

where `metric_i` may be reciprocal rank or nDCG contribution.

For transformation \(T\):

\[
QDS_T =
Spearman(
\{d_i^{ref}\},
\{d_i^{T}\}
)
\]

This asks:

> Are the same information needs still easy/hard after transformation?

Also report mean absolute difficulty shift:

\[
MADS_T =
\frac{1}{N}
\sum_i
|d_i^{ref}-d_i^T|
\]

---

## 13.6 Top-k overlap stability

For model \(m\), query \(i\), transformation \(T\):

\[
J@k =
\frac{|Top_k^{ref} \cap Top_k^T|}
{|Top_k^{ref} \cup Top_k^T|}
\]

Report:

```text
Jaccard@10
Jaccard@100
```

Also compute Ranked Biased Overlap (RBO) if implementation is stable.

This measures behavioral change even when aggregate effectiveness is unchanged.

---

# 14. Statistical testing

## Query-level paired bootstrap

For every:

```text
system × transformation × language
```

compare reference and transformed metric at query level.

Use:

```text
10,000 bootstrap resamples
95% percentile confidence interval
```

Report:

\[
\Delta nDCG@10
\]

with CI.

## Randomization/permutation test

For important pairwise system conclusions use a paired randomization test over query-level metric differences.

Recommended:

```text
10,000 permutations
two-sided alpha = 0.05
```

## Multiple comparisons

Use Benjamini-Hochberg FDR correction within each family of hypotheses:

```text
q = 0.05
```

Do not use dozens of uncorrected p-values.

## Effect size

Report paired standardized effect size in addition to p-values where appropriate.

Preferred:

```text
paired Cohen's dz
```

and bootstrap CI.

But retrieval metric differences + CI remain more interpretable than effect size alone.

---

# 15. Semantic validation metrics

Do not rely only on LLM judges.

For every transformation calculate:

## Character / lexical diagnostics

```text
length ratio
token-count ratio
named-entity preservation
numeric-token preservation
unit preservation
question-mark / punctuation sanity
```

## Embedding semantic similarity

Use a semantic similarity encoder different from the main retrieval systems if feasible.

If not feasible, use one retrieval encoder but label it clearly as an auxiliary diagnostic.

Compute:

\[
Sim_i = cosine(e(q_{ref}), e(q_T))
\]

This is not a gold semantic metric.

It is an auxiliary feature.

## Judge consensus score

For item \(i\):

\[
J_i = mean(JudgeA, JudgeB[,JudgeC])
\]

Primary semantic-validity label = consensus rule described above.

---

# 16. Key analysis: semantic quality vs evaluation validity

This is one of the paper’s strongest analyses.

For each query transformation, define:

```text
semantic_validity
semantic_similarity
retrieval_behavior_change
```

Possible retrieval-change target:

\[
B_i =
\frac{1}{M}
\sum_m
|RR_{m,i}^{ref}-RR_{m,i}^{T}|
\]

or analogous per-query nDCG difference.

Then estimate:

```text
Spearman correlation:
semantic_similarity vs B_i
```

and compare:

```text
validated transformations vs rejected transformations
```

The most interesting subset is:

```text
semantic-valid according to all judges
BUT
large retrieval behavior change
```

Save these examples for qualitative analysis.

---

# 17. Architecture-bias analysis

Group retrieval systems by family:

```text
lexical
dense
late-interaction
hybrid (if included)
```

For transformation \(T\), compute each system’s relative shift:

\[
RASS_{m,T}
\]

Then test whether shifts differ systematically by architecture family.

Because the number of systems is small, treat this as descriptive/exploratory unless enough independent systems per family are available.

Do not claim causal architectural bias from one model per family.

---

# 18. Language comparison

Run every analysis independently for:

```text
Assamese
Bengali
```

Then report pooled results only as secondary summaries.

Do not hide one language behind macro averages.

Primary tables should have separate columns for AS and BN.

---

# 19. Required ablations

At minimum:

## A1 — All transformations vs validated-only

Does the evaluation-instability result remain after obvious semantic failures are removed?

## A2 — Translation vs same-language paraphrase

Compare:

```text
C1/C2/C3 vs C4
```

This separates multilingual translation effects from generic rewording sensitivity.

## A3 — Synthetic question generation

Compare C5 separately.

Do not merge C5 with translation conditions; it is a different benchmark-construction process.

## A4 — Metric robustness

Repeat central conclusion-preservation analysis for:

```text
nDCG@10
MRR@10
Recall@100
```

The main story should not depend on one metric.

## A5 — Tie-threshold sensitivity

Repeat raw PCP at:

```text
tau = 0
tau = .0025
tau = .005
tau = .01
```

---

# 20. Pilot stop/go rule

Before spending the full budget, run:

```text
150 queries/language
C0 + C1 + C4 + C5
5 retrieval systems
```

Proceed to full study if at least one of these occurs:

1. statistically meaningful absolute-score shifts for one transformation;
2. at least one pairwise model relation changes;
3. query difficulty stability is materially below perfect;
4. validated transformations still create substantial ranking-list divergence;
5. synthetic generation behaves systematically differently from reference/paraphrase.

Do NOT require a desired direction.

If every transformation produces essentially identical behavior, preserve the result and reconsider whether the full experiment is worth expanding.

---

# 21. File/folder structure

```text
paper1_eval_invariance/
│
├── README.md
├── config/
│   ├── experiment.yaml
│   ├── models.yaml
│   └── prompts.yaml
│
├── data/
│   ├── raw/
│   ├── source/
│   ├── splits/
│   ├── transformations/
│   ├── validation/
│   ├── corpus/
│   └── qrels/
│
├── cache/
│   ├── embeddings/
│   └── indexes/
│
├── runs/
│   ├── as/
│   └── bn/
│
├── results/
│   ├── standard_metrics.csv
│   ├── score_shifts.csv
│   ├── pairwise_conclusions.csv
│   ├── invariance_metrics.csv
│   ├── statistical_tests.csv
│   ├── semantic_validation.csv
│   ├── per_query_analysis.parquet
│   └── qualitative_cases.jsonl
│
├── artifacts/
│   ├── dataset_manifest.json
│   ├── model_manifest.json
│   ├── api_manifest.json
│   ├── environment.txt
│   └── run_manifest.json
│
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_build_source_table.py
│   ├── 03_generate_transformations.py
│   ├── 04_validate_transformations.py
│   ├── 05_build_corpus.py
│   ├── 06_encode_corpus.py
│   ├── 07_run_retrieval.py
│   ├── 08_compute_ir_metrics.py
│   ├── 09_compute_invariance.py
│   ├── 10_statistics.py
│   ├── 11_error_analysis.py
│   └── 12_make_tables_figures.py
│
└── logs/
```

---

# 22. Main result tables to generate automatically

## Table 1 — Dataset and transformation statistics

Columns:

```text
language
condition
N generated
N valid
valid %
mean query length
entity preservation %
number preservation %
judge agreement %
```

## Table 2 — Standard retrieval effectiveness

Rows:

```text
retrieval system
```

Columns grouped by:

```text
language × condition
```

Report:

```text
nDCG@10
MRR@10
Recall@100
```

## Table 3 — Evaluation invariance

Rows:

```text
transformation condition
```

Columns:

```text
ASS mean
|ASS| mean
PCP raw
PCP significant
Kendall tau-b
Spearman rho
QDS
Jaccard@10
Jaccard@100
```

Separate AS / BN.

## Table 4 — Validated-only analysis

Same core results using only multi-model-consensus-valid transformations.

## Table 5 — Pairwise conclusion matrix

For every model pair:

```text
reference relation
C1 relation
C2 relation
C3 relation
C4 relation
C5 relation
```

Values:

```text
A>B
B>A
TIE
```

with significance-aware labels.

---

# 23. Required figures

## Figure 1 — Study design

```text
source information need
        |
        ├─ reference query
        ├─ MT
        ├─ LLM translation
        ├─ MT + post-edit
        ├─ paraphrase
        └─ synthetic-from-evidence
                 |
          SAME FIXED CORPUS
                 |
        multiple retrievers
                 |
     evaluation invariance audit
```

## Figure 2 — Score-shift heatmap

Rows = retrieval systems.

Columns = transformation conditions.

Value:

```text
Δ nDCG@10
```

One figure per language or a clean faceted version in the paper.

## Figure 3 — Pairwise conclusion flips

Matrix/heatmap indicating which system-pair conclusions change.

## Figure 4 — Semantic quality vs retrieval behavior change

Scatter:

```text
x = semantic similarity / judge confidence
y = retrieval behavior change
```

Highlight consensus-valid transformations.

---

# 24. Qualitative error analysis

Automatically select examples from four buckets:

```text
A. semantic invalid + retrieval changed
B. semantic invalid + retrieval stable
C. semantic valid + retrieval stable
D. semantic valid + retrieval changed
```

Bucket D is the scientifically most interesting.

For each example store:

```text
reference query
transformed query
judge labels
top-5 reference retrieval
top-5 transformed retrieval
positive passage
rank change
error taxonomy
```

Select ~20 strongest examples per language for manual inspection by the authors.

This manual inspection is for paper interpretation only; it is not used as ground-truth annotation.

---

# 25. Reproducibility requirements

Freeze:

```text
Python version
PyTorch version
Transformers version
sentence-transformers version
datasets version
ir_measures version
Pyserini version
FAISS version
CUDA version
all HF model revisions
all dataset revisions
all prompts
all API model IDs
all seeds
```

Save:

```bash
pip freeze > artifacts/environment.txt
```

Every experiment run gets:

```text
run_id
git commit hash
timestamp
seed
config hash
input dataset hash
```

---

# 26. API rules

All API generations:

```text
temperature = 0
```

Retry only on transport/server failure.

Never silently regenerate a valid response because it “looks bad.”

Raw API response must be preserved.

Add cache keyed by:

```text
SHA256(model_id + prompt + input)
```

This avoids accidental duplicate billing and ensures exact reruns.

---

# 27. What NOT to do

Do not:

- train a new embedding model;
- tune systems on the final test set;
- select only examples where transformation differences look interesting;
- regenerate transformations after seeing retrieval results;
- call LLM validation “human validation”;
- compare unrelated datasets and attribute differences to translation;
- invent relevance labels;
- use aggregate accuracy only;
- report Spearman over five systems as the sole evidence;
- claim attention, embeddings, or LLM judge scores are ground truth;
- claim C5 represents the same information need unless validation supports it;
- merge Assamese and Bengali into one number without language-specific reporting;
- claim “benchmark bias” causally unless the evidence supports that wording.

---

# 28. Minimum deliverables before returning results

The coding agent should return the following files:

```text
results/standard_metrics.csv
results/invariance_metrics.csv
results/pairwise_conclusions.csv
results/statistical_tests.csv
results/semantic_validation.csv
results/per_query_analysis.parquet
results/qualitative_cases.jsonl
```

and the following summary block:

```text
N_AS =
N_BN =

Transformation validity:
C1 =
C2 =
C3 =
C4 =
C5 =

Reference nDCG@10 by model:
...

PCP_sig by transformation:
...

Kendall tau-b by transformation:
...

Largest absolute score shifts:
...

Number of statistically supported pairwise conclusion changes:
...

Validated-only pairwise conclusion changes:
...

Semantic-quality vs behavior-change correlation:
...

Most interesting 10 examples:
...
```

---

# 29. Decision criteria for interpreting results

## Strong result type A

High semantic validity but low evaluation invariance.

Interpretation:

> Sentence-level quality checks are insufficient to guarantee benchmark-level evaluation validity.

## Strong result type B

Translations preserve conclusions but passage-conditioned synthetic generation does not.

Interpretation:

> Not all benchmark transformations are equally safe; construction procedure matters.

## Strong result type C

Absolute scores move but model comparisons remain stable.

Interpretation:

> Transformed benchmarks may distort reported performance levels while remaining useful for relative system comparison.

## Strong result type D

Everything is highly invariant.

Interpretation:

> Under these controlled conditions, multilingual transformation is more robust for comparative retrieval evaluation than feared.

This is still a valid scientific result.

---

# 30. Short coding-agent instruction

> Implement the protocol exactly as specified. First execute only the pilot. Freeze query IDs, prompts, corpus, qrels, model revisions, and random seed before looking at retrieval outcomes. The independent variable is query transformation; the corpus and relevance judgments must be identical across conditions. Store every intermediate artifact and make all final tables reproducible from saved runs. Do not optimize for finding a positive result. The core endpoint is whether benchmark transformation preserves statistically supported retrieval-system conclusions.

