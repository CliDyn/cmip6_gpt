
# RAG-Augmented Knowledge System for CMIP6 Climate Science: Merging Literature Intelligence with Multi-Agent Data Analysis

Dmitrii Pantiukhin  
Alfred Wegener Institute for Polar and Marine Research, Bremerhaven, Germany

[QUESTION: Co-authors? Same team as PANGAEA-GPT paper? Ivan Kuznetsov, Boris Shapkin, Antonia Anna Jost, Thomas Jung, Nikolay Koldunov?]

## Abstract

[QUESTION: Write after all sections are finalized. Should summarize: (1) the CMIP6 literature corpus, (2) the RAG architecture, (3) the agent system, (4) key results, (5) connection to multi-agent ecosystem.]

---

## 1 Introduction

The Coupled Model Intercomparison Project Phase 6 (CMIP6) has been the largest coordinated climate modeling effort to date, engaging over 50 modeling centers worldwide and generating output across more than 30 Model Intercomparison Projects (MIPs) spanning scenarios from deep paleoclimate to idealized future forcing pathways (Eyring et al., 2016). This unprecedented effort has produced thousands of peer-reviewed publications describing model configurations, evaluation procedures, emergent constraints, and projection uncertainties. Yet as the climate science community transitions toward CMIP7 (Balaji et al., 2025), a fundamental scalability challenge has emerged: the cumulative body of knowledge from CMIP6 — spanning model descriptions, variable definitions, experiment protocols, and inter-model comparisons — presents a substantial barrier for any individual researcher or working group attempting comprehensive review.

This knowledge reuse problem is acute. CMIP7 does not begin from zero; it inherits the infrastructure, parameterization experience, and documented biases of its predecessor. Climate models that participated in CMIP6 carry forward tuning histories, grid configurations, and known systematic errors that are extensively documented in the CMIP6 literature. Researchers designing new experiments, calibrating updated model versions, or interpreting projection spreads must locate and synthesize this prior art — a task that has traditionally relied on manual literature review and institutional memory.

The volume of relevant scientific output compounds this challenge. Conservative estimates place the number of open-access CMIP6-related publications at over 7,000 papers spanning more than a decade of activity. These publications are distributed across hundreds of journals with varying access models, use inconsistent terminology for equivalent physical quantities, and frequently reference model configurations that have undergone multiple revision cycles.

This fragmentation of knowledge creates a critical bottleneck not just for model developers, but particularly for downstream users of climate data. While the core physical science modeling community—such as the IPCC Working Group I (WG I)—is intimately familiar with the unique strengths, weaknesses, and structural biases of specific CMIP6 models, this nuanced understanding rarely propagates efficiently. Researchers in impacts, adaptation, and vulnerability (e.g., IPCC WG II) or mitigation strategy (WG III) are frequently forced to navigate complex model ensembles without the deeply specialized context required to select strictly appropriate models or apply necessary bias corrections. Consequently, there is an urgent need to streamline the transfer of highly technical, model-specific literature into accessible, actionable insights for non-modelers and downstream impact assessment groups.

The emergence of Large Language Models (LLMs) as scientific reasoning engines has opened a fundamentally new approach to this knowledge synthesis problem. LLMs have evolved from text generators into autonomous agents that reason over problems, decompose tasks, and invoke external tools (Boiko et al., 2023; Schick et al., 2023). This shift has given rise to Multi-Agent Systems (MAS) in which complex problems are partitioned across specialized agents (Hong et al., 2024; Li et al., 2023; Guo et al., 2024). Within geosciences and climate science, this progression has driven rapid innovation. Domain-specific foundation models such as K2 (Deng et al., 2024) and ClimateGPT (Thulke et al., 2024) have established robust baselines for Earth science knowledge extraction, while frameworks like GeoAgent (Chen et al., 2024) have demonstrated the capability of LLMs to conduct autonomous geospatial data analysis through code interpretation. Concurrently, specialized agentic systems have moved into operational deployment: ClimSight (Koldunov and Jung, 2024; Kuznetsov et al., 2025) demonstrated that augmenting LLMs with localized climate model output and retrieval-augmented generation (RAG) can deliver actionable, location-specific climate assessments, while PANGAEA-GPT (Pantiukhin et al., 2025a, 2025b) established a hierarchical multi-agent framework for autonomous data discovery and analysis in geoscientific data archives.

However, existing agent systems in climate science operate primarily on structured data resources — model output fields, reanalysis grids, and curated dataset repositories. None provides systematic access to the unstructured scientific literature that documents how these data were produced, what their known limitations are, and how they should be interpreted. Conversely, domain-general scientific RAG systems process text corpora without integration into the analytical workflows that climate scientists actually use. What is missing is a system that merges literature-grounded domain knowledge with live data analysis capabilities.

[QUESTION: The user mentioned "google and fast track" in the outline — is this referring to Google Scholar scraping? Or real-time web search? Need clarification on what "live extraction" means in the user's context vs. what the system actually does.]

Here, we present CMIP6 Forge, a hybrid retrieval-augmented generation and autonomous processing engine that bridges this gap. The system integrates a curated corpus of over 6,500 CMIP6-related scientific publications with a tool-augmented LLM agent capable of searching datasets, executing Python-based analyses, and connecting to external climate data services. By grounding the agent's responses in the peer-reviewed literature while simultaneously providing access to CMIP6 data catalogues and reanalysis fields, the system enables a workflow in which a researcher can ask a scientific question, receive a literature-informed answer, identify relevant datasets, and execute preliminary analyses within a single conversational interface. Furthermore, the architecture is designed as a composable module within a broader multi-agent ecosystem, where literature knowledge from CMIP6 Forge can be combined with PANGAEA-GPT's data analysis capabilities and ERA5 reanalysis access to support complex, cross-domain scientific workflows.

---

## 2 Methods

### 2.1 System Overview

CMIP6 Forge operates through three integrated subsystems (Fig. 1): (i) a Retrieval-Augmented Generation (RAG) pipeline that provides grounded access to the CMIP6 scientific literature, (ii) a CMIP6 metadata knowledge base that resolves natural language queries to structured dataset identifiers, and (iii) a ReAct-based LLM agent that orchestrates these resources through tool-augmented reasoning. The system is deployed as a web application with a React frontend communicating with a FastAPI backend via Server-Sent Events (SSE) for real-time streaming.

[QUESTION: Need a system architecture figure (Fig. 1). Should I generate one, or will you provide/sketch it?]

### 2.2 Literature Corpus Construction

#### 2.2.1 Corpus Scope and Acquisition

The literature corpus comprises 6,581 unique scientific publications related to CMIP6 climate science. Papers were identified through systematic queries of the OpenAlex academic metadata API, targeting publications that reference CMIP6 models, experiments, variables, or scenarios. The acquisition pipeline operated in three stages: (i) primary retrieval via standard HTTP requests to publisher endpoints, (ii) rendering-aware browser automation for dynamically loaded institutional pages, and (iii) multi-repository fallback through PubMed Central, Semantic Scholar, and the CORE open-access aggregator. This multi-stage approach achieved a 99.1% retrieval rate (7,101 of 7,162 identified publications).

A comprehensive open-access audit was conducted against the OpenAlex API to verify the legal distributional status of the corpus. Of the 6,581 parsed papers, 6,544 (99.4%) were confirmed as open-access publications from fully open journals (Geoscientific Model Development, Earth System Science Data, Nature Climate Change, etc.). A single closed-access document (an IPCC AR6 chapter published by Cambridge University Press) was identified, though this constitutes a government/intergovernmental publication effectively in the public domain. The remaining 36 papers (0.5%) were unresolvable against OpenAlex but originate from overwhelmingly open-access source journals. The RAG architecture stores only short text chunks (512–1,000 tokens) and mathematical embeddings, which constitutes fair use even for the negligible fraction of non-OA content.

#### 2.2.2 Document Parsing

Raw PDF documents were processed using the MinerU document understanding framework, deployed on an NVIDIA A100 GPU infrastructure utilizing a vLLM backend. MinerU performs layout-aware parsing using a 2.5-billion-parameter diffusion-based vision-language model (Dong et al., 2026) to extract structured content elements (headings, paragraphs, tables, mathematical formulas, and figure captions) with their hierarchical section paths. This advanced layout analysis preserves document structure that flat PDF text extraction destroys, enabling section-aware processing in downstream stages.

#### 2.2.3 Section-Aware Chunking

Parsed documents are segmented into semantically coherent chunks by a custom section-aware chunker (V6) that operates on the MinerU JSON output. The chunker implements the following design decisions:

- **Target chunk size**: 512–1,000 tokens, with a hard cap of 1,200 tokens for table chunks that require larger context windows.
- **Minimum quality threshold**: Chunks below 30 tokens are discarded; chunks below 80 tokens are merged with adjacent content.
- **Overlap**: 5% overlap ratio between consecutive chunks within a section to preserve boundary context.
- **Abstract isolation**: The abstract is always emitted as a standalone chunk, as it provides the highest-density summary signal.
- **Section exclusion**: Academic boilerplate sections (references, acknowledgements, author contributions, data availability, etc.) are automatically filtered via a curated exclusion list of >80 normalized section headings.

The chunking pipeline was systematically refined to ensure robust data quality by rigorously eliminating noise. Key extraction improvements included: the removal of HTML/URN boilerplate from web-scraped PDFs, suppression of OCR artifacts from figure-axis text, content-hash deduplication to eliminate redundant chunks, and context-dependent OCR corrections for common PDF glyph errors (e.g., "+" to "degrees C" in temperature contexts).

The pipeline produced 101,828 chunks across the 6,581 papers, with a mean of 15.5 chunks per paper.

#### 2.2.4 Embedding

Chunks are embedded using Google's Gemini Embedding 2 model (`gemini-embedding-2-preview`), a multimodal embedding model producing 768-dimensional dense vectors optimized for retrieval tasks. To ensure asymmetric optimization for the query-document similarity computation, the model explicitly differentiates between document-encoding mode during indexing and query-encoding mode during runtime search. All vectors are L2-normalized before insertion.

In parallel, each chunk is encoded as a sparse BM25 vector using the `Qdrant/bm25` model from the FastEmbed library, providing lexical matching capability that complements the semantic dense representations.

### 2.3 Vector Database and Hybrid Search

#### 2.3.1 Qdrant Index Architecture

The embedded corpus is indexed in a Qdrant vector database configured for hybrid retrieval. The primary literature collection (`cmip6_papers`, 101,828 points) stores both dense (768-dimensional cosine) and sparse (BM25) vector representations for each chunk, alongside structured metadata payloads containing: paper DOI, title, publication year, journal name, section path, chunk type (text/table/caption), and quality tier.

#### 2.3.2 Hybrid Search Pipeline

Literature retrieval follows a three-stage pipeline:

1. **Dual-channel query encoding**: The user query is simultaneously embedded as a dense vector (Gemini) and a sparse vector (BM25), capturing both semantic meaning and lexical specificity.

2. **Reciprocal Rank Fusion (RRF)**: Qdrant's native hybrid search executes parallel prefetch queries against the dense and sparse indices (default prefetch pool size: 50 per channel), then fuses the ranked lists using Reciprocal Rank Fusion to produce a unified ranking that leverages both channels.

3. **Neural reranking**: The fused candidate set is reranked using Google's Vertex AI Ranking API (`semantic-ranker-512`), a cross-encoder model that scores each query-document pair with full cross-attention. A local fallback using the `mxbai-rerank-base-v2` model (Shakir et al., 2024) provides CPU-based reranking when the API is unavailable.

The pipeline supports structured filtering by publication year ranges, journal names, quality tiers, chunk types, and DOI exclusion lists (to suppress dominant papers and diversify results).

#### 2.3.3 Citation Graph

A citation graph is constructed from the corpus metadata, capturing inter-paper citation relationships as a directed graph (stored as a NetworkX JSON serialization). The graph supports two traversal operations: forward citation lookup (papers that cite a given DOI within the corpus) and backward reference lookup (papers that a given DOI references, restricted to in-corpus documents). Results are ranked by global citation count, enabling the identification of high-impact foundational work.

### 2.4 Facet Metadata Enrichment and Indexing

Alongside the primary literature index, three supplementary collections are maintained to support the CMIP6 metadata knowledge base. These serve as the foundation for dynamic parameter matching during dataset search:
- `cmip6_variables` (1,313 points)
- `cmip6_experiments` (323 points)
- `cmip6_sources` (132 points)

A critical challenge in developing this component was the inherent terseness and inconsistency of native ESGF metadata. The raw administrative descriptions of CMIP6 parameters and experiment configurations are frequently abbreviated, highly opaque, or simply lacking context, rendering them inadequate for accurate semantic vector retrieval. To overcome this semantic gap, the entire metadata catalog for these three core facets was computationally rewritten and expanded using the `gemini-3.1-pro-preview` model. The LLM generated rich, contextually descriptive semantic profiles for each facet, effectively translating operational shorthand into comprehensive scientific text. These enriched, LLM-generated descriptions were then embedded and indexed in Qdrant. This preprocessing step ensures highly reliable semantic matching between a researcher's natural language request and the strictly formatted, rigid ESGF identifiers required for data access.

### 2.5 Agent Architecture

#### 2.5.1 ReAct Agent

The CMIP6 Forge agent is implemented as a LangGraph ReAct agent — a compiled directed cyclic graph that implements the ReAct (Reasoning + Acting) paradigm (Yao et al., 2023). The agent receives a natural language query, reasons about the required information, selects and invokes tools, observes results, and iterates until it can formulate a grounded response.

The system prompt instructs the agent to: (i) use the literature search tool for scientific methodology, model description, and research finding questions; (ii) use the dataset search tool for locating downloadable CMIP6 data; (iii) use the Python REPL for quantitative analysis and visualization; and (iv) cite sources with DOIs when providing literature-based answers.

#### 2.5.2 Tool Registry

The agent has access to seven specialized tools:

| Tool | Function |
|:-----|:---------|
| `cmip6_literature_search` | Hybrid search over the 101K-chunk literature corpus with year/journal/DOI filters and Vertex AI reranking. Returns ranked text excerpts with bibliographic metadata. |
| `cmip6_citation_graph` | Citation graph traversal (cited-by and references) for exploring paper relationships within the corpus. |
| `cmip6_datasets_search` | Resolves natural language queries to CMIP6 facet values (variable_id, source_id, experiment_id) via RAG over the metadata collections, then automatically chains into data availability checking. |
| `cmip6_datasets_access` | Checks data availability and retrieves download information for specific CMIP6 dataset configurations via the ESGF API. |
| `cmip6_adviser` | Answers questions about CMIP6 parameters, models, and experiments using the metadata knowledge base. |
| `Python_REPL` | A persistent, sandboxed Python execution environment with pre-loaded scientific libraries (NumPy, pandas, xarray, matplotlib). Figures are automatically saved and returned as file paths. |
| `analysis_guide` | Provides structured guidance for common CMIP6 analysis workflows. |

The dataset search tool implements a particularly important optimization: the agent's natural language query is decomposed into separate variable, source, and experiment components at the tool-call level, bypassing an intermediate LLM call for query splitting. Each component is independently embedded and searched against its corresponding Qdrant collection using Gemini embeddings, with exact-match injection for known identifiers and re-ranking boosts for query-mentioned source names.

#### 2.5.3 Session Management and Code Execution

Each user session receives an isolated Python REPL instance identified by a UUID, with a dedicated filesystem sandbox for generated artifacts (figures, data files). The REPL maintains persistent state (variables, loaded datasets, imported libraries) across tool calls within a session, enabling iterative multi-step analyses without data reloading.

### 2.6 Frontend and Deployment

The system is deployed as a web application:
- **Backend**: FastAPI server (Python) hosting the LangGraph agent, Qdrant client connections, and session management. SSE streaming provides real-time token delivery and tool-use status indicators.
- **Frontend**: Vite/React single-page application providing a conversational interface with rendered Markdown responses, inline figure display with code-behind capability, and RAG source attribution panels showing retrieved paper titles, DOIs, and relevance scores.
- **Infrastructure**: Qdrant runs as a persistent vector database service. The system supports configurable LLM backends.

[QUESTION: What model(s) does the system currently use? The config suggests OpenAI models — which one? GPT-4o? GPT-5.2? This is important for the paper. Also, are we using it on Levante or locally?]

---

## 3 Results

### 3.1 RAG Corpus Statistics

The constructed literature corpus provides comprehensive coverage of the CMIP6 scientific knowledge base:

| Metric | Value |
|:-------|:------|
| Total papers parsed | 6,581 |
| Total chunks indexed | 101,828 |
| Mean chunks per paper | 15.5 |
| Qdrant collections | 4 (papers, variables, experiments, sources) |
| Total indexed points | 103,596 |
| Storage footprint | ~905 MB |
| Confirmed open access | 99.4% (6,544 / 6,581) |
| Embedding model | Gemini Embedding 2 (768-dim) |
| Sparse model | Qdrant/bm25 |

The corpus spans publications from major climate science journals including Geoscientific Model Development, Journal of Advances in Modeling Earth Systems, Earth System Science Data, Nature Climate Change, and Geophysical Research Letters, covering the period from approximately 2016 to 2026.

### 3.2 Retrieval Quality

[QUESTION: We need benchmark results here. Possible approaches:
1. **Intrinsic evaluation**: Curate a set of N queries with known relevant papers, measure Precision at k, Recall at k, nDCG. Similar to what was done for PANGAEA-GPT's 100-query benchmark.
2. **Ablation study**: Compare retrieval quality across configurations — dense only vs. sparse only vs. hybrid vs. hybrid+rerank.
3. **Latency profiling**: Report embed_ms, search_ms, rerank_ms for typical queries.
4. **Qualitative examples**: Show 2-3 example queries with top-5 results demonstrating the system's ability to resolve CMIP6-specific terminology.

Which approach should we take? Should we run a formal benchmark, or are qualitative examples sufficient for this paper?]

### 3.3 System Use Case Scenarios

[QUESTION: What use cases should we demonstrate? The user's outline says "RAG (how good it is)" and "System Use Case scenarios". Possible scenarios:

1. **Model comparison query**: "What are the known biases of CESM2 vs. MPI-ESM1-2-LR in AMOC representation?" — Show literature retrieval + dataset discovery + analysis pipeline.
2. **Variable discovery**: "I need monthly sea surface temperature from a high-resolution model under SSP5-8.5" — Show metadata RAG to dataset search to data access chain.
3. **Literature review**: "What methods have been used to constrain equilibrium climate sensitivity in CMIP6?" — Show citation graph exploration.
4. **End-to-end analysis**: Start from a literature question, find relevant datasets, download and analyze them in the REPL.

Do you have specific scenarios in mind? Should we run actual sessions and include screenshots/outputs?]

### 3.4 [QUESTION: Something Else?]

[QUESTION: The user's outline includes "... Something else?" for results. Possible additional results sections:
- Comparison with baseline (no RAG) LLM responses (hallucination rate)?
- User study / expert evaluation?
- Chunk quality statistics (post-V6 audit)?
- Something else entirely?]

---

## 4 Discussion

### 4.1 Multi-Agent Integration

The CMIP6 Forge system is designed not as an isolated application but as a knowledge module within a broader multi-agent scientific workflow ecosystem. The architecture established in PANGAEA-GPT (Pantiukhin et al., 2025b) demonstrated that a Supervisor-Worker topology with data-type-aware routing can execute complex, multi-step Earth science analyses. CMIP6 Forge extends this ecosystem by providing a component that no prior agent possesses: grounded access to the scientific literature that documents how climate data were produced, validated, and should be interpreted.

In a multi-agent configuration, CMIP6 Forge serves as the domain knowledge authority. When a Supervisor Agent decomposes a research question, it can route literature-dependent subtasks to CMIP6 Forge (e.g., "What are the known biases of this model in this region?"), data retrieval tasks to the PANGAEA-GPT Oceanographer Agent, and reanalysis contextualization tasks to an ERA5 Agent. The literature-grounded responses from CMIP6 Forge can then inform the analytical parameters selected by downstream agents — for example, adjusting bias correction procedures based on documented model deficiencies, or selecting ensemble members known to best represent specific physical processes.

[QUESTION: Do we have concrete results of multi-agent integration to report? Or is this currently a design/architecture discussion? The user mentioned "Multi agentic way, so the other source like [in pangaea gpt previously multi ai agentic system]" — should this be framed as implemented or as architectural direction?]

### 4.2 ERA5 Integration

The ERA5 reanalysis agent provides a complementary data dimension to the CMIP6 literature knowledge base. While CMIP6 Forge grounds the agent in published scientific understanding of model behavior, the ERA5 agent provides access to observational constraint data through the Arraylake cloud-native data platform (Earthmover, 2025). This combination enables workflows that bridge model documentation and observational validation — for example, a researcher can query CMIP6 Forge for known temperature biases in a specific model's historical simulation, then task the ERA5 agent with retrieving the corresponding reanalysis fields for quantitative comparison.

[QUESTION: The user says "ERA5 [agent attached]" — do we have specific ERA5 integration results, or is this describing the Vostok/ClimSight ERA5 agent as a connected module? Need to clarify what "attached" means operationally.]

### 4.3 Limitations and Future Directions

The system's retrieval performance is fundamentally bounded by the quality and completeness of the underlying corpus. While 6,581 papers represent substantial coverage, the CMIP6 literature continues to grow, and the current corpus reflects a snapshot as of early 2026. Papers published after the corpus construction date, or those behind persistent publisher paywalls, are invisible to the retrieval layer.

Because the system relies on stochastic LLM inference, outputs are inherently non-deterministic. Repeated queries may yield different literature selections, varying emphasis on cited evidence, and marginally different interpretive framing. While the deterministic RAG retrieval and Python sandbox ensure that the evidence base and any generated code produce identical output, the upstream reasoning layer introduces variability in how evidence is synthesized into natural language responses.

The chunk-level retrieval granularity, while effective for locating specific claims and methodological details, can miss document-level arguments that span multiple sections. Future work should explore hierarchical retrieval strategies that combine chunk-level precision with paper-level summary representations.

[QUESTION: Other limitations to discuss? The user's outline focuses on the multi-agent discussion, but standard limitations should be included. Anything specific the user wants to highlight?]

---

## 5 Conclusion

[QUESTION: Write after all sections are finalized.]

---

## References

[QUESTION: Build reference list once all citations are finalized. Key references to include:

- Eyring et al. (2016) — CMIP6 overview
- Balaji et al. (2025) — CMIP7 design (if published)
- Pantiukhin et al. (2025a) — Frontiers in AI, PANGAEA-GPT architecture survey
- Pantiukhin et al. (2025b) — PANGAEA-GPT full system paper (arXiv:2602.21351)
- Koldunov and Jung (2024) — ClimSight
- Kuznetsov et al. (2025) — ClimSight v2
- Yao et al. (2023) — ReAct
- Schick et al. (2023) — Toolformer
- Boiko et al. (2023) — Autonomous scientific research
- Hong et al. (2024) — MetaGPT
- Li et al. (2023) — CAMEL
- Guo et al. (2024) — MAS survey
- Arias et al. (2021) — IPCC AR6 WG1 (if referencing assessment process)
- Earthmover (2025) — Arraylake
- LangChain (2024) — LangGraph
]
