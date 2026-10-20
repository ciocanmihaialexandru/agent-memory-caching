# Agent Memory, Caching & Intent Classifier

Optimizarea agentului **Document Analyst**: îi adăugăm memorie de
conversație, prompt caching și un clasificator de intenție clasic — ca să fie
mai ieftin, mai rapid și cu context persistent, pe principiul:
*„nu folosi LLM unde un mecanism determinist e suficient"*.

## Ce face

Pleacă de la agentul RAG peste documentul CEC și adaugă trei straturi
de optimizare:

1. **Conversation Memory** — agentul ține minte conversația între request-uri,
   persistată în PostgreSQL.
2. **Prompt Caching** — system prompt-ul + documentul sunt cache-uite la nivel
   de provider (Anthropic), reducând tokenii reprocesați și latența.
3. **Intent Classifier** — un model scikit-learn (TF-IDF + LogisticRegression)
   prezice intenția (search / extract / summarize) fără LLM, cu un benchmark
   de latență, cost și accuracy față de LLM.

## Structura proiectului (adăugiri față de agentul RAG)

```
agent-memory-caching/
├── langchain_agent.py        # agentul LangGraph + memorie de conversație
├── memory/                   # Partea 1 — Conversation Memory
│   └── conversation.py       #   load_history / add_turn (persistat în Postgres)
├── prompt_cache_demo.py      # Partea 2 — demo prompt caching Anthropic
├── classifier/               # Partea 3 — Intent Classifier
│   ├── training_data.py      #   date de antrenare: query + label
│   ├── train.py              #   TF-IDF (word+char) + LogisticRegression, CV
│   ├── predict.py            #   predict_intent / predict_with_confidence
│   └── intent_model.joblib   #   modelul antrenat (salvat)
├── benchmark_intent.py       # compară clasificator vs LLM (latență/cost/accuracy)
│
│   # moștenite din agentul RAG:
├── extraction/               # încărcare + extracție structurată (pe doc_type)
├── db/                       # modele, repository, tranzacții (+ ChatMessage)
├── rag/                      # embeddings + similarity search
├── tools/                    # tool-uri agent + search_documents
├── prompts/                  # system prompturi (YAML + Jinja2)
├── alembic/                  # migrări (tabele, pgvector, HNSW, chat_messages)
└── data/                     # documentul sursă
```

## Concepte implementate

### Partea 1 — Conversation Memory
LLM-ul e **stateless**: fiecare apel e independent, deci memoria o gestionăm noi,
în cod. Ținem fiecare tur (user + assistant) în tabelul `chat_messages`, grupat pe
`session_id`. La fiecare rulare, `ConversationMemory.load_history()` reconstruiește
contextul din DB și-l retrimite agentului; după răspuns, `add_turn()` salvează turul.
Asta e „long-term memory = PostgreSQL" din analogia memoriei umane. Integrarea se
face în `langchain_agent.py`: istoricul se adaugă în lista de `messages` înainte de
`invoke`.

### Partea 2 — Prompt Caching
Cu `cache_control: {"type": "ephemeral"}` pe blocul de context fix (system prompt +
documentul), Anthropic cachează tot prefixul. Primul apel **scrie** cache-ul; apelurile
următoare cu același prefix îl **citesc** (tokenii cache-uiți costă ~10% și vin mai
repede). Se folosește `ChatAnthropic` direct (nu LiteLLM), fiindcă `cache_control` e o
funcție Anthropic. Măsurarea se face din `usage_metadata["input_token_details"]`
(`cache_read`, `ephemeral_5m_input_tokens`).

### Partea 3 — Intent Classifier
Regula de aur: dacă output-ul e **predictibil și finit** (3 clase), nu ai nevoie de LLM.
Antrenăm un `Pipeline(TF-IDF → LogisticRegression)` care clasifică **intenția întrebării**
(nu conținutul documentului): search / extract / summarize. TF-IDF combină n-grame de
**cuvinte** (1-2) și de **caractere** (`char_wb`, 2-4) — ultimele ajută la flexiunile din
română. Evaluarea folosește **cross-validation 5-fold** (accuracy stabil pe date puține),
iar modelul final se antrenează pe tot setul. Clasificatorul înlocuiește pasul de „detect
intent / routing" care altfel ar fi un apel LLM.

## Setup

1. Mediu virtual + dependințe:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Infrastructură (Postgres cu pgvector + LiteLLM), din folderul de infra:
```bash
docker-compose up -d
```

3. `.env` în proiect:
```
DATABASE_URL=postgresql+psycopg://skillab:skillab_dev@localhost:5432/skillab
LITELLM_BASE_URL=http://localhost:4000/v1
ANTHROPIC_API_KEY=sk-ant-...
```

4. Schema bazei de date (tabele, pgvector, HNSW, chat_messages):
```bash
alembic upgrade head
```

5. Ingestează documentul (dacă nu e deja):
```bash
python ingest.py
```

## Utilizare

**Agent cu memorie de conversație** (același `-s` = aceeași conversație):
```bash
python langchain_agent.py -s demo "Mă numesc Alex, mă interesează STUDENT FREE."
python langchain_agent.py -s demo "Cum mă numesc și ce pachet mă interesa?"
```

**Demo prompt caching:**
```bash
python prompt_cache_demo.py
```

**Antrenează clasificatorul de intenție:**
```bash
python -m classifier.train
```

**Benchmark clasificator vs LLM:**
```bash
python benchmark_intent.py
```

## Rezultate

**Conversation memory** — agentul își amintește între rulări (procese separate):

```text
$ python langchain_agent.py -s demo "Mă numesc Alex, mă interesează STUDENT FREE."
RĂSPUNS (sesiune: demo):
Salut, Alex! Pachetul STUDENT FREE are 0 lei comision de administrare lunară ...

$ python langchain_agent.py -s demo "Cum mă numesc și ce pachet mă interesa?"
RĂSPUNS (sesiune: demo):
Te numești Alex și te interesa pachetul STUDENT FREE.
```

Deși a doua comandă e un proces nou (LLM-ul, singur, n-ar avea context), agentul
răspunde corect pentru că `load_history("demo")` îi reîncarcă turul anterior din
tabelul `chat_messages`.

**Prompt caching** — context de ~13.000 tokeni (system + document):

```text
Apel 1 (write):  ephemeral_5m=13078  cache_read=0
Apel 2 (read):   cache_read=13078    latență -41%
```

**Intent classifier** — accuracy CV 5-fold: **0.86 ± 0.09**.

**Benchmark (9 întrebări de test):**

```text
CLASSIFIER  acc=100%  cost=$0 (local)
LLM         acc=78%   cost=$0.0023
```

Pe un task îngust cu 3 clase, clasificatorul clasic e mai rapid, gratuit și chiar
mai precis decât LLM-ul — și întoarce mereu exact una din cele 3 clase (determinist).

## Note

- Memoria și prompt caching-ul sunt complementare: memoria ține conversația, caching-ul
  reduce costul reprocesării contextului fix.
- Clasificatorul clasifică **intenția întrebării**, nu conținutul documentului —
  răspunsul despre document vine tot din RAG (`search_documents`).