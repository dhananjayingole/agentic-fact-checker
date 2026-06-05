# 🔍 Agentic Fact Checker

An AI-powered autonomous fact verification system. Paste any claim — get a verdict with evidence in seconds. Built entirely on **free resources** (no paid APIs required).

---

## ✨ Features

- **Multi-source search** — DuckDuckGo + Wikipedia (no API keys needed)
- **LLM judgment** — Groq's Llama3 (1000+ tokens/sec, free tier: 10k req/day)
- **Evidence scoring** — relevance, stance detection, source credibility
- **Knowledge graph** — Neo4j AuraDB for claim history (optional, free tier)
- **REST API** — FastAPI with Swagger docs, batch verification, CORS

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourname/agentic-fact-checker.git
cd agentic-fact-checker

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set your **Groq API key** (free at https://console.groq.com):

```env
GROQ_API_KEY=gsk_your_key_here
```

> **No Groq key?** The app still works in heuristic mode — it uses keyword analysis instead of LLM.

### 3. Run

```bash
python run.py
```

Visit:
- **Swagger UI**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health

---

## 🐳 Docker (Optional)

```bash
# Build and run
docker-compose up --build

# Or with Docker directly
docker build -t fact-checker .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key fact-checker
```

---

## 📡 API Usage

### Verify a Claim

```bash
curl -X POST http://localhost:8000/verify/ \
  -H "Content-Type: application/json" \
  -d '{"claim": "Humans only use 10% of their brain", "max_sources": 5}'
```

**Response:**
```json
{
  "claim": "Humans only use 10% of their brain",
  "verdict": "FALSE",
  "confidence_score": 1.5,
  "evidence_count": 4,
  "evidence_summary": "Brain imaging shows all parts of the brain are active, disproving the 10% myth.",
  "reasoning": "Multiple high-credibility sources (Wikipedia, scientific journals) confirm this is a myth...",
  "processing_time_seconds": 3.8,
  "sources_searched": 5,
  "evidence_list": [...]
}
```

### Batch Verify (up to 10 claims)

```bash
curl -X POST http://localhost:8000/verify/batch \
  -H "Content-Type: application/json" \
  -d '{"claims": ["The Great Wall is visible from space", "Mount Everest is the tallest mountain"], "max_sources": 3}'
```

### Extract Claims from Text

```bash
curl -X POST http://localhost:8000/extract/claims \
  -H "Content-Type: application/json" \
  -d '{"text": "Brazil won the FIFA World Cup 8 times. The tournament started in 1930."}'
```

### Search Without Verdict

```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "climate change causes", "max_results": 5}'
```

---

## 🗄️ Neo4j Knowledge Graph (Optional)

1. Create a free cluster at https://neo4j.com/cloud/aura/
2. Add credentials to `.env`:

```env
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

The app will automatically store every verification result and build a graph of `(Claim)-[:HAS_EVIDENCE]->(Source)` relationships.

---

## 🧪 Running Tests

```bash
# Unit tests (no server needed)
pip install pytest pytest-asyncio
python -m pytest tests/ -v

# Integration tests (requires running server)
python run.py &     # start server first
python test_api.py
```

---

## 📁 Project Structure

```
agentic-fact-checker/
├── run.py                        # Entry point
├── test_api.py                   # Integration tests
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── api/
│   │   ├── main.py               # FastAPI app + lifespan
│   │   └── routes/
│   │       ├── verify.py         # POST /verify/, /verify/batch
│   │       ├── search.py         # POST /search/
│   │       ├── extract.py        # POST /extract/claims
│   │       └── health.py         # GET /, /health
│   ├── agents/
│   │   └── fact_checker_agent.py # Orchestration pipeline
│   ├── tools/
│   │   ├── search_tool.py        # DuckDuckGo + Wikipedia + scraping
│   │   └── evidence_analyzer.py  # Relevance + stance + credibility
│   ├── models/
│   │   └── schemas.py            # Pydantic request/response models
│   └── services/
│       ├── groq_service.py       # Groq LLM integration
│       └── neo4j_service.py      # Neo4j knowledge graph
└── tests/
    └── test_agents.py            # Unit tests
```

---

## 🔑 Free Resources Summary

| Service | Purpose | Free Tier |
|---|---|---|
| [Groq](https://console.groq.com) | LLM (Llama3) | 10,000 req/day |
| DuckDuckGo | Web search | Unlimited |
| Wikipedia | Encyclopedia | Unlimited |
| [Neo4j AuraDB](https://neo4j.com/cloud/aura/) | Knowledge graph | 50 MB forever |

---

## 🎯 Verdict Reference

| Verdict | Confidence | Meaning |
|---|---|---|
| `TRUE` | 7.0 – 10.0 | Evidence supports the claim |
| `FALSE` | 0.0 – 3.0 | Evidence contradicts the claim |
| `INCONCLUSIVE` | 3.1 – 6.9 | Mixed or ambiguous evidence |
| `UNVERIFIABLE` | — | No relevant evidence found |

---

## 🛠️ Tech Stack

- **FastAPI** — REST API framework
- **Groq** — Ultra-fast LLM inference (Llama3-8b)
- **DuckDuckGo Search** — Free web search
- **Wikipedia API** — Free encyclopedia search
- **BeautifulSoup4** — Web page content extraction
- **Neo4j** — Graph database for claim provenance
- **Loguru** — Structured logging
- **Pydantic v2** — Data validation
- **SlowAPI** — Rate limiting

---

## 📊 Performance

| Metric | Value |
|---|---|
| Avg. response time | 3–7 seconds |
| Concurrent requests | 10+ |
| Sources per claim | 3–10 |
| Cost per verification | **$0.00** |
