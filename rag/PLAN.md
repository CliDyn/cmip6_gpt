# CMIP6 RAG — ФИНАЛЬНЫЙ ПЛАН 🔥

> AWI | CMIP6 Bot | Gemini API optimized | March 2026

---

## Текущее состояние

| Что есть | Статус |
|---|---|
| Корпус | 7,115 статей (PDF 55% + HTML 45% = 24 GB) |
| Классификация | 4,581 CORE + 2,532 EXTENDED (99.93%) |
| Лицензии | 7,047 resolved, 0 unknown, 100% для hosted RAG |
| Текущий RAG | Metadata-only ChromaDB, MRR=0.78, Hit@5=95% |
| Агент | LangGraph, GPT-5.2 / Gemini 3.1 Pro |

---

## Архитектура (полный пайплайн)

```
PDF/HTML
  → Docling (PDF: layout/tables/формулы, HTML: native)
  → Section-Aware Chunks (512-1000 tok) + Metadata Prefix
  → Gemini Embeddings (task_type, MRL 768-dim, L2-norm, Batch API)
  → Qdrant (Hybrid: Dense + BM25 + RRF)
  → Cross-Encoder Reranker (bge-reranker-v2-m3)
  → LangGraph Agent (Adaptive RAG + CRAG + Self-RAG)
  → LLM Response + Inline Citations
```

---

## ПОРЯДОК ИСПОЛНЕНИЯ

### 🟢 SPRINT 1: Парсинг + Чанкинг (2-3 дня)

**1. Docling парсинг** — `rag/parse_papers.py`
- Docling v2.76 — один конвертер для PDF И HTML
- Output: structured Markdown в `rag/parsed/{safe_doi}.md`
- Сначала тестируем на 50 CORE статей (топ по цитированиям)
- Потом масштабируем: multiprocessing на все 7,115

**2. Section-Aware chunking** — `rag/chunk_papers.py`
- Бьём по ## / ### заголовкам, target **512-1000 tokens**
- 10% overlap внутри секций
- Таблицы, подписи к рисункам, формулы → отдельные атомарные чанки
- Каждый чанк получает metadata prefix (БЕЗ LLM, просто поля):
  ```
  Paper: "{title}" ({year}, {journal})
  Section: {section_path}
  ---
  {chunk_text}
  ```
- Output: `rag/chunks.jsonl`

### 🟢 SPRINT 2: Embeddings + Qdrant (2-3 дня)

**3. Gemini Embeddings** — `rag/embed_and_index.py`
- Model: `gemini-embedding-001`
- `task_type="RETRIEVAL_DOCUMENT"` при индексации
- `task_type="RETRIEVAL_QUERY"` при запросе
- `output_dimensionality=768` (MRL, 4x меньше RAM)
- **L2-normalize ВСЕ** вектора перед записью в базу:
  ```python
  embedding = embedding / np.linalg.norm(embedding)
  ```
- Batch API для массовой обработки (50% дешевле)
- Стоимость: ~$60 за ~500K чанков

**4. Qdrant Hybrid Search** — `rag/search.py`
- Docker: `qdrant/qdrant:v1.15.0`
- Dense: 768-dim, cosine
- Sparse: BM25 (FastEmbed built-in)
- Payload: `paper_id`, `tier`, `year`, `journal`, `section`, `doi`
- RRF fusion dense + sparse
- Filter: `tier = "CORE"` для фокусного поиска

**5. Benchmark** — `rag/benchmark.py`
- 150-query benchmark (existing)
- Сравнение: new Qdrant full-text vs old ChromaDB metadata
- Target: **MRR > 0.85, Hit@5 > 97%**

### 🟡 SPRINT 3: Точность (3-4 дня)

**6. Cross-Encoder Reranker**
- Model: `bge-reranker-v2-m3` (local, OSS)
- Top-50 из Qdrant → rerank → top-8 в LLM
- Если MRR всё ещё < 0.85 после reranker:

**7. Contextual Retrieval (LLM prefix)**
- Добавляем 1-2 предложения LLM-саммари к каждому чанку
- Gemini 3.1 Pro: "Summarize this chunk in 1 sentence"
- Перевекторизуем с новыми prefix'ами
- Стоимость: ~$30 дополнительно

### 🔴 SPRINT 4: Knowledge Graph (1 неделя)

**8. GROBID → Citation Graph**
- Docker: `grobid/grobid:0.8.1`
- Extract references (TEI XML) → DOI matching
- NetworkX граф: `Paper --CITES--> Paper`
- PageRank, "кто цитирует X?", "противоречащие источники"

**9. LightRAG → Entity Graph**
- Extract: модели, эксперименты, переменные, регионы, SSP сценарии
- `Paper --USES--> CESM2`, `Paper --STUDIES--> Arctic sea ice`
- Multi-hop: "Какие статьи используют CESM2 для арктики под SSP5-8.5?"

### 🔴 SPRINT 5: Agentic RAG (ongoing)

**10. Adaptive RAG Router** (LangGraph)
- Классификация сложности запроса
- Simple → Qdrant hybrid
- Complex → Graph + Qdrant + iterative

**11. CRAG** (Corrective RAG)
- Критик проверяет чанки на релевантность
- Если мусор → переписывает запрос и ищет заново

**12. Self-RAG** (Citation Verification)
- Генератор сверяет КАЖДОЕ утверждение с контекстом
- Принудительные inline citations: `[Author et al., Year]`
- Флаг противоречий и low-confidence

---

## Инфраструктура

### Docker
```bash
# Qdrant (Sprint 2)
docker run -d --name qdrant -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:v1.15.0

# GROBID (Sprint 4)
docker run -d --name grobid -p 8070:8070 \
  grobid/grobid:0.8.1
```

### Dependencies
```
docling              # PDF/HTML parsing
qdrant-client        # Vector DB
fastembed            # BM25 sparse vectors
numpy                # L2 normalization
google-generativeai  # Gemini embeddings
```

---

## Стоимости

| Компонент | Разовая | Рекуррентная |
|---|---:|---:|
| Gemini embeddings (500K чанков) | ~$60 | ~$5/мес |
| Contextual prefixes (если нужно) | ~$30 | $0 |
| Qdrant Docker | $0 | $0 |
| GROBID Docker | $0 | $0 |
| Reranker (local) | $0 | $0 |
| **Итого** | **~$60-90** | **~$5/мес** |

---

## Файлы проекта

```
rag/                          ← НОВАЯ ДИРЕКТОРИЯ
├── parse_papers.py           ← Sprint 1: Docling → Markdown
├── chunk_papers.py           ← Sprint 1: Section-aware chunking
├── embed_and_index.py        ← Sprint 2: Gemini → Qdrant
├── search.py                 ← Sprint 2: Hybrid search API
├── benchmark.py              ← Sprint 2: MRR/Hit@5 comparison
├── parsed/                   ← Docling output (Markdown)
├── chunks.jsonl              ← Chunked corpus
└── qdrant_storage/           ← Qdrant data
```
