## Legal Consultation RAG Chatbot

This project reconstructs a Retrieval-Augmented Generation (RAG) workflow for a legal
assistant. The focus is algorithm transparency: every stage—retrieval, semantic
re-ranking, and grounded response generation—is written in plain Python so you can trace
exactly how answers are produced.

---

### Architecture Overview

```mermaid
flowchart LR
    subgraph User Interaction
        U[User CLI / API] -->|問題| P
    end

    subgraph Pipeline
        P[RAGPipeline.run()]
        P -->|call| R[BM25Retriever]
        R -->|SearchResult[]| RR[LLMReranker]
        RR -->|SearchResult[]| G[AnswerGenerator]
        G -->|answer + context| P
    end

    subgraph Components
        DL[data_loader.load_documents]
        T(SimpleTokenizer)
        LM(HuggingFaceLLM\ngoogle/gemma-3-270m-it)
    end

    DL --> R
    R <-->|tokenize| T
    DL --> RR
    RR <-->|score_relevance| LM
    DL --> G
    T --> LM
    G <-->|generate| LM
    P -->|VerboseTracer| VT[(Tracing Logs)]
    P -->|PipelineOutput| U
```

---

### Repository Layout

| Path | Description |
| --- | --- |
| `data/legal_corpus.jsonl` | Sample bilingual legal knowledge base (JSONL). |
| `legal_rag/` | Python package with modular RAG components. |
| `legal_rag/data_loader.py` | Loads JSONL corpus into `Document` objects. |
| `legal_rag/preprocessing.py` | Tokenization and CJK bigram expansion. |
| `legal_rag/retriever.py` | BM25 lexical retriever (algorithm reproduction). |
| `legal_rag/ranker.py` | LLM-based semantic re-ranker with weighting. |
| `legal_rag/llm.py` | LLM abstraction + `HuggingFaceLLM` wrapper around `google/gemma-3-270m-it`. |
| `legal_rag/generator.py` | Prompt construction and guarded answer drafting. |
| `legal_rag/pipeline.py` | Orchestrates retrieval → ranking → generation. |
| `legal_rag/factory.py` | Convenience builder that wires all components. |
| `legal_rag/tracing.py` | Hierarchical verbose tracer for debugging. |
| `cli.py` | Interactive command-line chatbot demo. |
| `slides/business_value_slides.md` | Topic A briefing deck for the GC interview. |

---

### Quick Start

```bash
# install dependencies (first run)
pip install -U "transformers==4.44.2" "huggingface_hub>=0.24" accelerate sentencepiece torch --extra-index-url https://download.pytorch.org/whl/cpu

# authenticate with Hugging Face (once per environment)
export HF_TOKEN=<your huggingface token>
python - <<'PY'
from huggingface_hub import login
import os
token = os.getenv("HF_TOKEN")
if not token:
    raise SystemExit("Please set HF_TOKEN before running this script.")
login(token=token, new_session=False)
PY

# run the chatbot
python3 cli.py
# or enable detailed execution tracing
python3 cli.py --verbose
```

1. Type a legal question (Chinese or English).
2. The RAG pipeline retrieves, ranks, and assembles an answer with citations.
3. Enter `exit` to leave the chat.

To embed in your own code:

```python
from legal_rag import build_pipeline

pipeline = build_pipeline("data/legal_corpus.jsonl")
response = pipeline.run("公司違約時可以要求哪些救濟？")
print(response.answer)
```

---

### Stage-by-Stage Flow

1. **Retrieval (`BM25Retriever`)**  
   - Tokenizes the corpus using `SimpleTokenizer`.  
   - Computes BM25 scores to fetch top-k candidate documents.

2. **Semantic Re-Ranking (`LLMReranker` + `HuggingFaceLLM`)**  
   - Encodes question/contexts with Gemma embeddings (mean-pooled hidden states).  
   - Cosine similarity is blended with BM25 (`alpha` weighting) to refine ordering.

3. **Grounded Generation (`AnswerGenerator` + `HuggingFaceLLM`)**  
   - Builds an augmented prompt with ranked snippets and provenance.  
   - The Gemma model drafts a structured legal reply (guidance + citations + next steps).

The flow mirrors the provided pseudocode while keeping each component easily swappable.

---

### Verbose Tracing

Enable `--verbose` on the CLI (or inject `VerboseTracer(enabled=True)` into your own
entry point) to print a step-by-step execution log such as tokenizer output, BM25 scoring
details, and final prompt assembly. Useful for debugging, demos, and audits.

**Authentication note**: `build_pipeline` will read `HF_TOKEN` and `HF_MODEL_ID` from the
environment. At minimum export `HF_TOKEN` (或直接沿用上述指令) before the first run so the Gemma weights can be downloaded.

---

### Extending the Prototype

- **LLM Integration**: Implement `BaseLLM` against GPT, Claude, or a local model, then
  drop it into `factory.build_pipeline`.
- **Alternative Retrieval**: Replace `BM25Retriever` with vector search or hybrid recall.
- **Knowledge Expansion**: Append internal statutes, precedents, or policy memos to
  `legal_corpus.jsonl` (ensure provenance metadata is captured).
- **Evaluation Harness**: Add question–answer pairs and track Recall@K, nDCG, factual
  accuracy, and satisfaction metrics via the tracer hooks.

---

### Business Context

The accompanying deck (`slides/business_value_slides.md`) summarizes Topic A for the GC
Data Scientist Final Interview, walking through pseudocode alignment, component design,
and system extensibility.

---

> ⚠️ **Disclaimer:** The demo corpus provides generic legal guidance only. Always consult
> licensed counsel for real-world cases.
