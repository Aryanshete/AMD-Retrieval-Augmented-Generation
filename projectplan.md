# Match 3 — RAG Project Plan

## 1. Objective

Build the Match 3 Retrieval-Augmented Generation (RAG) Docker submission that:

1. Receives a mixed-document corpus at `/app/corpus`.
2. Indexes the complete corpus once using:
   ```bash
   python3 /app/app.py --index /app/corpus
   ```
3. Answers each hidden question using:
   ```bash
   python3 /app/app.py \
       --corpus /app/corpus \
       --query-id query_01 \
       --query "..."
   ```
4. Writes the result to:
   ```text
   /app/output/<query-id>_output.json
   ```
5. Returns:
   - the **value only** as `answer`
   - the **exact necessary source files** as `citations`
   - optional `confidence`
6. Refuses unsupported questions with:
   ```json
   {"answer": "", "citations": [], "confidence": 0.0}
   ```
7. Runs on the mandated AMD ROCm image and actually uses the GPU.
8. Handles text, PDFs, DOCX, XLSX, CSV, TXT/LOG, Python source, PNG/JPG, unreadable files, encrypted files, unknown file types, and empty directories without crashing.

The core scoring rule is:

> **A question receives credit only when BOTH the answer and the citation set are exactly correct.**

Source: Match 3 brief, especially pages 1–3 and 4–6. fileciteturn0file0L5-L19

---

# 2. Non-Negotiable Evaluation Constraints

| Requirement | Limit / Rule | Implementation consequence |
|---|---|---|
| Base image | Mandatory ROCm base | Do not change the `FROM` image |
| Image size | 60 GiB uncompressed | Keep model + dependencies comfortably below remaining budget |
| VRAM | 1–48 GiB | Explicitly run inference on AMD GPU |
| Startup | 10 minutes | Model loading + complete indexing must finish within budget |
| Per question | 30 seconds | Query path must be lightweight and avoid model/index reload |
| Whole run | 10 minutes | Ten questions must complete comfortably |
| Network | No outbound network during evaluation | All model weights/resources required at runtime must be shipped in image |
| Output | Exact `/app/output/<query-id>_output.json` | Never derive query ID from question |
| Citations | Exact set | Do not return all retrieved files |
| Missing answer | Empty answer + empty citations | Never hallucinate or answer from model memory |

The challenge explicitly warns that a new process is started for every question. Loading a large model or re-indexing the corpus inside every query will likely exceed the 30-second budget. fileciteturn0file0L154-L162

---

# 3. Target Architecture

## Recommended architecture

```text
                         Docker Container
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  CMD                                                         │
│   │                                                          │
│   ├── Start RAG worker/daemon                                │
│   │      │                                                   │
│   │      ├── Load embedding / reranker model                 │
│   │      ├── Load VLM / QA model                             │
│   │      └── Wait on Unix socket                            │
│   │                                                          │
│   └── Keep container alive                                   │
│                                                              │
│  /app/app.py  ← thin client, executed once per command       │
│       │                                                      │
│       ├── --index /app/corpus                                │
│       │       │                                              │
│       │       └── worker parses + indexes corpus             │
│       │                                                      │
│       └── --query ...                                        │
│               │                                              │
│               └── worker retrieves + answers                 │
│                                                              │
│  /app/index/                                                 │
│       ├── metadata                                            │
│       ├── text chunks                                         │
│       ├── lexical index                                      │
│       ├── vector index                                       │
│       └── image/OCR metadata                                 │
│                                                              │
│  /app/output/                                                │
│       └── query_XX_output.json                               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Why this architecture

The brief specifically recommends keeping the expensive model/index work resident because the harness starts a new process for each question. A Unix socket is explicitly suggested as a suitable mechanism. fileciteturn0file0L154-L162

### Important

Do **not** expose an HTTP port.

Use an internal Unix domain socket only:

```text
/app/run/rag.sock
```

The external evaluator only interacts with `/app/app.py`.

---

# 4. Repository Structure

Target project:

```text
mc3-starter-kit/
│
├── Dockerfile
├── projectplan.md
├── selfcheck.py
├── sample-questions.json
├── setup.sh
│
└── app/
    ├── app.py                  # evaluator-facing thin client
    ├── worker.py               # long-running RAG worker
    ├── config.py
    │
    ├── indexing/
    │   ├── __init__.py
    │   ├── crawler.py          # safe corpus traversal
    │   ├── dispatcher.py       # parser selection
    │   ├── chunker.py          # chunk + metadata generation
    │   ├── lexical_index.py    # BM25 / inverted index
    │   └── vector_index.py     # embedding index
    │
    ├── parsers/
    │   ├── pdf_parser.py
    │   ├── docx_parser.py
    │   ├── xlsx_parser.py
    │   ├── csv_parser.py
    │   ├── text_parser.py
    │   ├── source_parser.py
    │   └── image_parser.py
    │
    ├── retrieval/
    │   ├── hybrid.py
    │   ├── reranker.py
    │   ├── query_router.py
    │   └── citation_selector.py
    │
    ├── generation/
    │   ├── qa.py
    │   └── answer_normalizer.py
    │
    ├── models/
    │   └── model_loader.py
    │
    ├── requirements.txt
    └── models/
        └── <offline model weights>
```

Do not hardcode any sample answers.

The graded corpus and questions are different from the sample corpus. fileciteturn0file0L195-L203

---

# 5. Phase 1 — Preserve the Starter Contract

Before implementing RAG, preserve the existing invocation contract.

## `app.py` must support

### Index

```bash
python3 /app/app.py --index /app/corpus
```

### Query

```bash
python3 /app/app.py \
  --corpus /app/corpus \
  --query-id query_01 \
  --query "..."
```

### Output

```text
/app/output/query_01_output.json
```

Output schema:

```json
{
  "answer": "94",
  "citations": [
    "specs/tq40_datasheet_r2.pdf"
  ],
  "confidence": 0.91
}
```

`answer` and `citations` are mandatory. `confidence` is optional and is not scored. fileciteturn0file0L45-L56

---

# 6. Phase 2 — Safe Corpus Crawler

Implement a defensive recursive corpus walker.

## Supported file types

The brief requires handling:

- `.pdf`
- `.docx`
- `.xlsx`
- `.csv`
- `.txt`
- `.log`
- `.py`
- `.png`
- `.jpg`

These formats are explicitly present in the challenge corpus. fileciteturn0file0L100-L112

## Crawler rules

For every file:

```text
try:
    identify extension
    select parser
    parse
    create document record
except:
    record failure
    continue
```

Never allow one bad file to terminate the complete indexing pass.

Must survive:

1. Empty directory
2. Unknown file type
3. Permission-denied file
4. Encrypted file

The brief explicitly states that the walk must continue after each failure. fileciteturn0file0L115-L128

## Unknown file

Example:

```text
vendor/telemetry_capture.dat
```

Action:

```text
SKIP
```

Do not crash.

## Permission denied

Example:

```text
vendor/internal_audit.txt
```

Action:

```text
catch PermissionError
record unreadable
continue
```

## Encrypted PDF

Example:

```text
vendor/supplier_agreement_ENCRYPTED.pdf
```

Action:

```text
catch parser/decryption error
record unreadable/encrypted
continue
```

Do **not** attempt to bypass encryption.

---

# 7. Phase 3 — Document Parsing

Every supported file type needs a parser.

## 7.1 PDF

Extract:

- page text
- page number
- title/metadata if available
- source path

Important:

The corpus can contain superseded documents.

Example:

```text
tq40_datasheet_r1_WITHDRAWN.pdf
tq40_datasheet_r2.pdf
```

The system must retrieve the correct current revision rather than simply returning the first matching value.

The sample explicitly tests this trap: R1 contains a different temperature and is withdrawn. fileciteturn0file0L239-L246

---

## 7.2 DOCX

Extract:

- paragraphs
- headings
- table cells
- rows/columns
- document path

Do not parse only paragraphs.

The brief explicitly states that internal DOCX information can be held in tables. fileciteturn0file0L104-L107

---

## 7.3 XLSX

Extract:

- every worksheet
- row/column values
- sheet name
- cell coordinates where useful

Represent rows as searchable records.

Example metadata:

```json
{
  "file": "support/rma_parts.xlsx",
  "sheet": "Parts",
  "row": 17,
  "text": "TQ-40 | field-replaceable fan assembly | ORR-FAN-2214-B"
}
```

The sample question specifically requires the XLSX answer. fileciteturn0file0L245-L246

---

## 7.4 CSV

Parse header + rows.

Preserve the original values exactly.

Example:

```text
ticket_id,summary,fixed_in
ORR-1847,...,4.3.2
```

---

## 7.5 TXT / LOG / PY

Treat as structured text.

For `.py`, preserve:

- constants
- variable names
- comments
- default values
- function/class context

The sample requires extracting a default batch timeout from Python source. fileciteturn0file0L249-L252

---

# 8. Phase 4 — Image Understanding

Two sample questions can only be answered from text printed inside images.

Required images:

```text
specs/backplane_pinout.png
support/asset_label.jpg
```

The challenge explicitly says a text-only pipeline cannot solve these. fileciteturn0file0L269-L281

## Recommended pipeline

```text
Image
  ↓
Resize / normalize
  ↓
Vision model / OCR-capable VLM
  ↓
Extract text + visual relationships
  ↓
Store OCR text + image metadata
  ↓
Index
```

Do not rely exclusively on OCR if the answer depends on a diagram relationship.

Example:

```text
B11 → GND
B12 → SDA
B13 → SCL
B14 → THERM_ALERT#
B15 → PRSNT#
B16 → GND
```

A vision-capable model must be able to determine that:

```text
THERM_ALERT# → B14
```

---

# 9. Phase 5 — Canonical Document Representation

Every successfully parsed document should become a common structure:

```python
DocumentChunk(
    chunk_id,
    source_path,
    file_type,
    page=None,
    sheet=None,
    row=None,
    section=None,
    text,
    metadata,
    content_type
)
```

Example:

```json
{
  "chunk_id": "abc123",
  "source_path": "support/bug_database.csv",
  "file_type": "csv",
  "text": "ORR-1847 ... fixed in 4.3.2",
  "metadata": {
    "row": 18
  }
}
```

This gives the retrieval system a uniform interface regardless of source format.

---

# 10. Phase 6 — Chunking Strategy

Avoid blindly splitting every file into arbitrary 500-token chunks.

Use format-aware chunks.

## Text / logs

Chunk around:

- log event
- paragraph
- section
- small groups of lines

## CSV

Prefer one logical row or a small related group of rows.

## XLSX

Prefer one logical row plus headers.

## DOCX

Prefer:

```text
heading + related paragraphs/table rows
```

## PDF

Prefer page/section-aware chunks.

## Images

Store:

```text
OCR text
+
image description
+
visual entities/relationships
```

## Metadata on every chunk

At minimum:

```text
source_path
file_type
chunk_id
```

Optional:

```text
page
sheet
row
section
revision
```

---

# 11. Phase 7 — Hybrid Retrieval

Use **hybrid retrieval**, not embeddings alone.

## Retrieval layers

### Layer A — lexical retrieval

Use:

- exact token matching
- BM25/inverted index
- identifier matching
- version matching
- part-number matching

This is critical for questions containing:

```text
ORR-1847
TQ-40
THERM_ALERT#
REV-C2
4.3.2
```

### Layer B — semantic retrieval

Use embeddings for natural-language questions where exact terms may not match the document wording.

### Layer C — reranking

Combine:

```text
lexical score
+
semantic score
+
identifier overlap
+
document-type relevance
```

Then select a small top-k candidate set.

---

# 12. Query Routing

Before retrieval, classify the question.

Examples:

```text
"maximum junction temperature"
→ specification lookup

"which firmware fixed ticket..."
→ ticket/bug lookup

"which pin carries..."
→ image/diagram lookup

"what is the default timeout..."
→ source-code lookup

"which firmware release fixed the incident..."
→ multi-hop lookup
```

This allows us to search the most useful representations first.

---

# 13. Exact Citation Strategy

This is the most important scoring component.

The challenge does **not** want:

```text
all files that are relevant
```

It wants:

```text
all files that are necessary
```

The rule is:

> Cite a file only when removing that file would make the answer impossible.

This is explicitly stated in the brief. fileciteturn0file0L70-L82

## Citation algorithm

For each candidate answer:

1. Identify evidence chunks.
2. Group evidence by source file.
3. Determine the minimum source set required.
4. Remove redundant files.
5. Return only the minimal necessary source set.
6. Preserve corpus-relative paths.

### Example — single-hop

```text
Answer:
4.3.2

Citation:
[
  "support/bug_database.csv"
]
```

Do NOT include:

```text
engineering/meridian_release_notes.txt
```

just because it discusses firmware.

---

# 14. Multi-Hop Retrieval

Some questions require more than one file.

Sample:

```text
The production log shows a thermal throttle incident.
Which firmware release fixed the underlying defect?
```

Correct reasoning:

```text
Question
  ↓
production log
  ↓
find ticket / incident identifier
  ↓
ticket identifier
  ↓
bug database
  ↓
fixed firmware version
```

Required citations:

```json
[
  "logs/prod_inference_2026-09-02.log",
  "support/bug_database.csv"
]
```

The brief explicitly states that both files are required because removing either one makes the answer impossible. fileciteturn0file0L80-L82

---

# 15. Superseded-Document Handling

This must be designed intentionally.

Sample:

```text
specs/tq40_datasheet_r1_WITHDRAWN.pdf
specs/tq40_datasheet_r2.pdf
```

The withdrawn document contains a plausible but incorrect value.

Implementation:

### Detect revision/status signals

Look for:

```text
WITHDRAWN
SUPERSEDED
OBSOLETE
RETIRED
REV-A
REV-B
REV-C
```

### Ranking preference

For conflicting documents:

```text
current/latest revision
    >
older revision
    >
withdrawn/superseded
```

But never simply ignore old files, because the graded corpus may contain historical questions that legitimately depend on them.

The final decision should depend on the query and evidence.

---

# 16. Answer Extraction

The final answer must be **value only**.

Bad:

```text
The maximum junction temperature is 94°C.
```

Good:

```text
94
```

Bad:

```text
The answer is Q3 FY27.
```

Good:

```text
Q3 FY27
```

The brief requires complete values, including qualifiers such as fiscal year or revision prefixes. fileciteturn0file0L59-L68

---

# 17. Answer Normalization

Internally preserve the exact source value.

For final output:

```text
answer = extracted complete value
```

Do not accidentally remove:

- revision prefixes
- fiscal year
- part-number suffix
- meaningful identifiers

Examples:

```text
Q3 FY27    ✓
Q3         ✗

REV-C2     ✓
C2         ✗

ORR-FAN-2214-B ✓
ORR-FAN-2214   ✗
```

---

# 18. No-Answer / Refusal Logic

If the corpus does not provide an accessible answer:

```json
{
  "answer": "",
  "citations": [],
  "confidence": 0.0
}
```

Never return:

```text
"I don't know"
```

Never guess.

Never answer from pretrained model knowledge.

The sample deliberately includes an answer inside an encrypted PDF; that value must still be treated as unavailable. fileciteturn0file0L90-L98

---

# 19. Important Encrypted-File Rule

If:

```text
supplier_agreement_ENCRYPTED.pdf
```

contains the answer but cannot be opened, it is **not a valid source**.

Therefore:

```text
answer = ""
citations = []
```

This is a deliberate test of whether the system hallucinates or illegally derives inaccessible information. fileciteturn0file0L265-L266

---

# 20. Model Strategy

## Recommended model class

Use one compact **vision-language model** capable of:

- document/image understanding
- OCR-like extraction
- question answering

A 7B-class VLM is a strong starting point because:

- it is large enough for reliable document QA,
- it can process the two image-only sample questions,
- it leaves room under the image-size limit,
- it can run within the available AMD VRAM range with appropriate loading.

For retrieval, use a lightweight local lexical index plus a compact embedding/reranking model if the dependency budget allows.

### Priority order

```text
1. Correct citations
2. Correct retrieval
3. Image understanding
4. Latency
5. Generation quality
```

Do not use a huge model simply because the GPU has available VRAM. The image limit is tighter than the VRAM limit: the starter contract notes that the base image itself is about 29.2 GiB. fileciteturn0file0L136-L143

---

# 21. Offline Model Requirement

The evaluation environment has no outbound network.

Therefore:

```text
BUILD TIME:
download model
↓
copy weights into Docker image

EVALUATION:
NO DOWNLOAD
↓
load local weights
```

Never do:

```python
from_pretrained("some-model")
```

if that causes a runtime network download.

Instead point the loader to a local path:

```text
/models/<model-name>
```

The brief explicitly says to ship weights inside the image. fileciteturn0file0L164-L181

---

# 22. ROCm / PyTorch Safety

This is a major build risk.

The brief warns that normal `pip install` operations can silently replace the ROCm PyTorch build with a CUDA build.

Therefore:

```dockerfile
RUN pip install --no-cache-dir --no-deps \
    -U transformers tokenizers safetensors
```

where appropriate, rather than allowing dependencies to replace the preinstalled ROCm torch.

After every dependency change, verify:

```bash
python3 -c "import torch; print(torch.__version__)"
```

Expected:

```text
...+rocm...
```

The challenge specifically documents this failure mode and recommends `--no-deps` for packages such as transformers/tokenizers/safetensors. fileciteturn0file0L164-L181

---

# 23. GPU Validation

The model must actually execute on the AMD GPU.

At runtime verify:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

For ROCm, PyTorch exposes AMD GPU execution through its CUDA-compatible API.

Do not deliberately allocate a dummy tensor merely to pass the VRAM requirement.

The challenge samples VRAM continuously and detects fake GPU usage. fileciteturn0file0L138-L143

---

# 24. Index Persistence

The index should survive between `--index` and every query.

Suggested:

```text
/app/index/
├── manifest.json
├── documents.jsonl
├── chunks.jsonl
├── lexical.pkl
├── embeddings.npy
├── metadata.json
└── image_ocr.json
```

The exact implementation can differ.

The critical requirement is:

```text
--index
    ↓
persist expensive work

query_01
    ↓
load/reuse persisted work

query_02
    ↓
reuse persisted work
```

Never parse the complete corpus for every query.

---

# 25. Worker Lifecycle

Recommended:

```text
Container starts
    ↓
worker.py starts
    ↓
load models once
    ↓
wait for Unix socket
    ↓
app.py --index
    ↓
worker indexes corpus
    ↓
app.py exits
    ↓
worker remains alive
    ↓
app.py --query
    ↓
worker answers
    ↓
app.py writes JSON
```

This satisfies the evaluator's separate-process behavior while avoiding repeated expensive model loading.

---

# 26. Query Pipeline

For every query:

```text
1. Receive query
2. Normalize query for retrieval
3. Detect likely entities / identifiers
4. Lexical retrieval
5. Semantic retrieval
6. Merge candidates
7. Rerank
8. If image candidate:
      invoke vision reasoning
9. If multi-hop:
      follow identifier/entity links
10. Extract answer
11. Determine minimal necessary citations
12. Validate answer/citation consistency
13. Write JSON
```

Target:

```text
< 30 seconds
```

Preferably:

```text
2–10 seconds
```

per query.

---

# 27. Citation Validation Layer

Before writing the final JSON, run a final consistency check.

## Check A

If:

```text
answer != ""
```

then:

```text
citations must not be empty
```

## Check B

Every citation must:

```text
be relative to /app/corpus
```

## Check C

Every citation must correspond to a file that was actually read.

## Check D

Remove redundant citations.

## Check E

For multi-hop answers, make sure every link in the reasoning chain is represented.

---

# 28. Sample-Corpus Acceptance Tests

The implementation is not complete until all ten sample cases are handled.

| # | Expected answer | Required citation(s) | Main feature |
|---|---|---|---|
| 1 | `94` | `specs/tq40_datasheet_r2.pdf` | PDF + withdrawn revision trap |
| 2 | `Q3 FY27` | `planning/roadmap_fy27.docx` | DOCX |
| 3 | `ORR-FAN-2214-B` | `support/rma_parts.xlsx` | XLSX |
| 4 | `4.3.2` | `support/bug_database.csv` | CSV |
| 5 | `E7731` | `logs/prod_inference_2026-09-02.log` | LOG |
| 6 | `180` | `engineering/ingest_service.py` | Python source |
| 7 | `B14` | `specs/backplane_pinout.png` | Image/vision |
| 8 | `REV-C2` | `support/asset_label.jpg` | Image/OCR |
| 9 | `4.3.2` | log + bug database | Multi-hop |
| 10 | empty | empty | Encrypted/no-access refusal |

These expected answers and citation sets are supplied by the starter kit and challenge brief. fileciteturn0file0L239-L263

---

# 29. Adversarial Tests

Before submission, create additional local tests for:

## File robustness

```text
empty directory
unknown extension
permission denied
encrypted PDF
corrupt PDF
corrupt image
empty CSV
empty XLSX sheet
large document
nested directories
```

## Retrieval robustness

```text
exact identifier
partial identifier
different word order
synonym
abbreviation
question referring to previous identifier
```

## Citation robustness

```text
one-source answer
two-source answer
conflicting documents
withdrawn document
same value in two files
irrelevant topical match
```

## Refusal robustness

```text
answer absent
answer only inside unreadable file
answer only inside encrypted file
answer only in unsupported file
```

---

# 30. Required Failure Handling

No individual document should crash indexing.

Use:

```python
for path in corpus.rglob("*"):
    if not path.is_file():
        continue

    try:
        parse(path)
    except PermissionError:
        record_failure(path, "permission")
    except Exception as exc:
        record_failure(path, type(exc).__name__)
```

Important:

```text
FAIL ONE FILE
    ↓
CONTINUE ALL OTHER FILES
```

Never:

```text
FAIL ONE FILE
    ↓
ABORT INDEX
```

---

# 31. Dockerfile Plan

Start from the exact starter base:

```dockerfile
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
```

Then:

```dockerfile
WORKDIR /app

COPY app/requirements.txt /app/requirements.txt

# Install only required dependencies.
# Protect ROCm torch from accidental replacement.

COPY app/ /app/

COPY models/ /models/

RUN mkdir -p /app/corpus /app/output /app/index /app/run

CMD ["bash", "-lc", "python3 /app/worker.py & exec sleep infinity"]
```

Keep the base image unchanged because the challenge checks the underlying base layers.

---

# 32. Dependency Plan

Start with the smallest possible dependency set.

Likely categories:

```text
transformers
tokenizers
safetensors
Pillow
PyMuPDF
python-docx
openpyxl
numpy
```

Add only what is actually required.

For every dependency:

1. Check whether it already exists in the base image.
2. Avoid unnecessary dependencies.
3. Avoid packages that pull in a different torch build.
4. Prefer `--no-deps` for packages where dependency resolution could replace ROCm torch.
5. Recheck `torch.__version__`.

---

# 33. Performance Budget

Target internal budget:

```text
Container startup + model load + index:
< 5–7 minutes target

Maximum:
< 10 minutes

Per query:
< 10 seconds target

Maximum:
< 30 seconds

All ten queries:
< 2–3 minutes target

Maximum:
< 10 minutes
```

The official limits are 10 minutes startup, 30 seconds per question, and 10 minutes for the complete post-startup run. fileciteturn0file0L130-L147

---

# 34. Development Milestones

## Milestone 1 — Contract

- [x] Starter project copied
- [x] Docker builds
- [x] Container starts
- [x] `--index` works
- [x] `--query` works
- [x] JSON output correct
- [x] Query ID handled exactly

## Milestone 2 — Parsers

- [x] PDF
- [x] DOCX
- [x] XLSX
- [x] CSV
- [x] TXT
- [x] LOG
- [x] PY
- [x] PNG
- [x] JPG

## Milestone 3 — Robustness

- [x] Empty directory
- [x] Unknown file
- [x] Permission denied
- [x] Encrypted PDF
- [x] Corrupt/unreadable file
- [x] Index continues after errors

## Milestone 4 — Retrieval

- [x] Chunking
- [x] Lexical index
- [x] Embedding index
- [x] Hybrid retrieval
- [x] Reranking
- [x] Query routing

## Milestone 5 — Vision

- [x] Image OCR
- [x] Diagram understanding
- [x] Image metadata indexing
- [x] Vision QA

## Milestone 6 — Citations

- [x] Single-source citation
- [x] Exact citation paths
- [x] Minimal citation set
- [x] Multi-hop citation
- [x] No redundant citations

## Milestone 7 — Answering

- [x] Value-only answers
- [x] Complete qualifiers
- [x] Empty answer on no-answer
- [x] No model-memory guessing
- [x] Answer normalization

## Milestone 8 — Performance

- [x] Model loaded once
- [x] Index persisted
- [x] Query process remains lightweight
- [x] GPU confirmed
- [x] No runtime network dependency
- [x] Per-query latency measured

## Milestone 9 — Final Validation

- [x] All 10 sample questions pass
- [x] Selfcheck passes
- [x] DAC_OVERRIDE test passes
- [x] Network-none test passes
- [x] Torch remains ROCm
- [x] Image size checked
- [x] VRAM checked
- [x] Container remains alive
- [x] Final Docker image tagged

---

# 35. Final Validation Commands

## Build

```bash
docker build -t mc3-rag:final .
```

## Verify torch

```bash
docker run --rm mc3-rag:final \
  python3 -c "import torch; print(torch.__version__)"
```

Must show a ROCm build.

## Run selfcheck

```bash
python3 selfcheck.py mc3-rag:final mc3-corpus
```

## No-network test

```bash
docker run --rm \
  --network none \
  mc3-rag:final \
  python3 /app/app.py --index /app/corpus
```

## GPU check

Run inside the AMD environment and verify VRAM usage:

```bash
amd-smi metric --mem
```

The challenge recommends monitoring VRAM continuously during testing. fileciteturn0file0L183-L193

---

# 36. Final Submission Checklist

### Contract

- [ ] Correct ROCm base image
- [ ] No exposed port
- [ ] Container stays alive
- [ ] `/app/app.py` exists
- [ ] `/app/output` exists
- [ ] Exact output filename
- [ ] Exact JSON structure

### Parsing

- [ ] PDF
- [ ] DOCX
- [ ] XLSX
- [ ] CSV
- [ ] TXT
- [ ] LOG
- [ ] PY
- [ ] PNG
- [ ] JPG

### Robustness

- [ ] Unknown files skipped
- [ ] Permission errors caught
- [ ] Encrypted files caught
- [ ] Empty directories safe
- [ ] Index never aborts because of one file

### Retrieval

- [ ] Exact identifier search
- [ ] Semantic search
- [ ] Hybrid retrieval
- [ ] Reranking
- [ ] Multi-hop support
- [ ] Superseded revision handling

### Vision

- [ ] OCR
- [ ] Diagram understanding
- [ ] Image-only answers

### Citations

- [ ] Corpus-relative paths
- [ ] Exact set
- [ ] No redundant sources
- [ ] Multi-hop sources included
- [ ] Empty list for refusal

### Performance

- [ ] Model loaded once
- [ ] Index built once
- [ ] Query does not rebuild index
- [ ] Query does not download model
- [ ] GPU actually used
- [ ] <30 seconds/query
- [ ] <10 minutes startup
- [ ] <10 minutes whole run

### Final

- [ ] Run `selfcheck.py`
- [ ] Test with `--network none`
- [ ] Test with `--cap-drop DAC_OVERRIDE`
- [ ] Test unreadable file
- [ ] Test encrypted file
- [ ] Test unknown extension
- [ ] Test empty corpus directory
- [ ] Verify image size
- [ ] Verify ROCm torch
- [ ] Verify final container remains alive

---

# 37. Recommended Implementation Order

Do **not** attempt the entire system at once.

Use this exact order:

```text
STEP 1
Starter contract
        ↓
STEP 2
Safe file crawler
        ↓
STEP 3
Text/PDF/DOCX/XLSX/CSV/PY parsers
        ↓
STEP 4
Persistent document/chunk index
        ↓
STEP 5
Lexical retrieval
        ↓
STEP 6
Answer extraction
        ↓
STEP 7
Exact citation selection
        ↓
STEP 8
Hybrid semantic retrieval
        ↓
STEP 9
Vision/OCR
        ↓
STEP 10
Multi-hop reasoning
        ↓
STEP 11
Long-running worker + Unix socket
        ↓
STEP 12
ROCm/GPU optimization
        ↓
STEP 13
Adversarial testing
        ↓
STEP 14
Selfcheck + final image
```

This ordering ensures that we always have a working baseline and add complexity only after the contract is stable.

---

# 38. Definition of Done

Task 3 is complete only when the final Docker image can:

```text
START
  ↓
LOAD LOCAL MODEL
  ↓
INDEX ENTIRE MIXED CORPUS
  ↓
SURVIVE BAD/UNKNOWN/UNREADABLE FILES
  ↓
ANSWER TEXT QUESTIONS
  ↓
ANSWER IMAGE QUESTIONS
  ↓
FOLLOW MULTI-HOP REFERENCES
  ↓
SELECT EXACT NECESSARY CITATIONS
  ↓
REFUSE INACCESSIBLE/UNKNOWN ANSWERS
  ↓
WRITE CORRECT JSON
  ↓
HANDLE TEN SEPARATE QUERY PROCESSES
  ↓
STAY WITHIN 10-MIN STARTUP
  ↓
STAY WITHIN 30-SEC/QUESTION
  ↓
USE AMD GPU
  ↓
PASS SELFCHECK
```

The goal is not merely to build a chatbot. The goal is to build a **deterministic, citation-accurate, mixed-format, GPU-backed RAG system that satisfies the evaluator contract**.

---

## Source-of-truth requirements

This plan is based on:

- `AMD AI League — Match 3 Challenge Brief.pdf`
- `mc3-starter-kit/CONTRACT.md`
- `mc3-starter-kit/README.txt`
- `mc3-starter-kit/sample-questions.json`
- `mc3-starter-kit/starter/Dockerfile`
- `mc3-starter-kit/starter/app/app.py`
- `mc3-starter-kit/starter/app/requirements.txt`
- `mc3-starter-kit/selfcheck.py`

The starter kit's sample corpus is only a development target. The real evaluation corpus and questions are hidden/different, so the implementation must generalize rather than hardcode the ten sample answers.
