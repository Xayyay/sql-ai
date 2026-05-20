"""
SQL AI Chatbot — Backend
Ask questions in plain language (English or Lao ພາສາລາວ).
The AI reads your MSSQL schema, writes the SQL, runs it, and answers you in natural language.

Flow: Your question → AI generates SQL → Execute on MSSQL → AI explains results → Chat reply
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import re, requests, json, os

app = FastAPI(title="SQL AI Chatbot", version="2.0.0")

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))


# ── Models ──────────────────────────────────────────────────────────────────

class DBConfig(BaseModel):
    server: str
    database: str
    username: str = ""
    password: str = ""
    use_windows_auth: bool = False
    port: int = 1433

class AIConfig(BaseModel):
    # Local
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    # External provider selection
    provider: str = "auto"      # "auto" | "ollama" | "claude" | "openai" | "groq" | "openrouter"
    # API keys
    claude_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    # Model selection per provider
    claude_model: str = "claude-haiku-4-5-20251001"
    openai_model: str = "gpt-4o-mini"
    groq_model: str = "llama-3.1-8b-instant"
    openrouter_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    # Legacy alias
    ai_mode: str = "auto"
    language: str = "en"    # "en" | "lo" | "both"

class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    db_config: DBConfig
    ai_config: AIConfig

class QueryRequest(BaseModel):
    sql: str
    db_config: DBConfig


# ── DB helpers ───────────────────────────────────────────────────────────────

def get_connection(config: DBConfig):
    try:
        import pyodbc
    except ImportError:
        raise HTTPException(500, "pyodbc not installed. Run: pip install pyodbc")

    drivers = [d for d in pyodbc.drivers() if "SQL Server" in d]
    if not drivers:
        raise HTTPException(500, "No SQL Server ODBC driver found. Install 'ODBC Driver 17 for SQL Server'.")
    driver = drivers[-1]

    if config.use_windows_auth:
        cs = (f"DRIVER={{{driver}}};SERVER={config.server},{config.port};"
              f"DATABASE={config.database};Trusted_Connection=yes;")
    else:
        cs = (f"DRIVER={{{driver}}};SERVER={config.server},{config.port};"
              f"DATABASE={config.database};UID={config.username};PWD={config.password};")
    try:
        return __import__("pyodbc").connect(cs, timeout=10)
    except Exception as e:
        raise HTTPException(400, f"Connection failed: {e}")


def serialize_val(v):
    if v is None:
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, (int, float, bool, str)):
        return v
    return str(v)


def run_sql(sql: str, config: DBConfig, limit: int = 500):
    """Execute a SELECT query and return (columns, rows)."""
    conn = get_connection(config)
    cur = conn.cursor()
    try:
        cur.execute(sql)
    except Exception as e:
        conn.close()
        raise HTTPException(400, f"SQL error: {e}")
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchmany(limit)
    data = [{c: serialize_val(v) for c, v in zip(cols, row)} for row in rows]
    conn.close()
    return cols, data


def is_safe_sql(sql: str) -> bool:
    """Only allow SELECT / WITH / read-only statements."""
    stripped = sql.strip().lstrip("(").upper()
    dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE",
                 "ALTER", "CREATE", "EXEC", "EXECUTE", "GRANT", "REVOKE"]
    for kw in dangerous:
        if re.search(rf"\b{kw}\b", stripped):
            return False
    return True


# ── Schema discovery ─────────────────────────────────────────────────────────

def get_schema(config: DBConfig) -> str:
    """
    Returns a compact text description of all tables & columns.
    This is fed to the AI so it knows what to query.
    """
    conn = get_connection(config)
    cur = conn.cursor()

    # Get tables
    cur.execute("""
        SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE IN ('BASE TABLE','VIEW')
        ORDER BY TABLE_SCHEMA, TABLE_NAME
    """)
    tables = cur.fetchall()

    # Get columns
    cur.execute("""
        SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE,
               CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
    """)
    all_cols = cur.fetchall()
    conn.close()

    # Group columns by table
    col_map: Dict[str, List[str]] = {}
    for schema, tbl, col, dtype, maxlen, nullable in all_cols:
        key = f"{schema}.{tbl}"
        type_str = dtype
        if maxlen:
            type_str += f"({maxlen})"
        col_map.setdefault(key, []).append(f"  - {col} ({type_str})")

    lines = [f"Database: {config.database}\n"]
    for schema, tbl, ttype in tables:
        key = f"{schema}.{tbl}"
        label = "VIEW" if ttype == "VIEW" else "TABLE"
        lines.append(f"[{label}] {key}")
        for c in col_map.get(key, []):
            lines.append(c)
        lines.append("")

    return "\n".join(lines)


# ── AI helpers ───────────────────────────────────────────────────────────────

def call_ollama(prompt: str, cfg: AIConfig, system: str = "") -> str:
    full = f"{system}\n\n{prompt}" if system else prompt
    try:
        r = requests.post(
            f"{cfg.ollama_url.rstrip('/')}/api/generate",
            json={"model": cfg.ollama_model, "prompt": full, "stream": False},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        return f"[Ollama error {r.status_code}: {r.text[:120]}]"
    except requests.exceptions.ConnectionError:
        return "[Ollama not running — start with: ollama serve]"
    except Exception as e:
        return f"[Ollama error: {e}]"


def call_claude(prompt: str, cfg: AIConfig, system: str = "") -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=cfg.claude_api_key)
        msg = client.messages.create(
            model=cfg.claude_model or "claude-haiku-4-5-20251001",
            max_tokens=1500,
            system=system or "You are a helpful SQL assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except ImportError:
        return "[anthropic package missing — run: pip install anthropic]"
    except Exception as e:
        return f"[Claude error: {e}]"


def call_openai_compat(prompt: str, api_key: str, model: str, base_url: str,
                        system: str = "", label: str = "OpenAI") -> str:
    """Generic OpenAI-compatible chat completion (works for OpenAI, Groq, OpenRouter)."""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if "openrouter" in base_url:
            headers["HTTP-Referer"] = "https://github.com/sql-ai-chat"
            headers["X-Title"] = "SQL AI Chatbot"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful SQL assistant."},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": 1500,
            "temperature": 0.3,
        }
        r = requests.post(f"{base_url.rstrip('/')}/chat/completions",
                          headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        return f"[{label} error {r.status_code}: {r.text[:200]}]"
    except Exception as e:
        return f"[{label} error: {e}]"


def call_provider(prompt: str, cfg: AIConfig, system: str = "") -> tuple[str, str]:
    """Call the active external provider. Returns (text, label)."""
    p = cfg.provider

    if p == "claude" and cfg.claude_api_key:
        t = call_claude(prompt, cfg, system)
        return t, f"☁️ Claude ({cfg.claude_model})"

    if p == "openai" and cfg.openai_api_key:
        t = call_openai_compat(prompt, cfg.openai_api_key, cfg.openai_model,
                               "https://api.openai.com/v1", system, "OpenAI")
        return t, f"☁️ OpenAI ({cfg.openai_model})"

    if p == "groq" and cfg.groq_api_key:
        t = call_openai_compat(prompt, cfg.groq_api_key, cfg.groq_model,
                               "https://api.groq.com/openai/v1", system, "Groq")
        return t, f"☁️ Groq ({cfg.groq_model})"

    if p == "openrouter" and cfg.openrouter_api_key:
        t = call_openai_compat(prompt, cfg.openrouter_api_key, cfg.openrouter_model,
                               "https://openrouter.ai/api/v1", system, "OpenRouter")
        return t, f"☁️ OpenRouter ({cfg.openrouter_model})"

    return None, ""


def call_ai(prompt: str, cfg: AIConfig, system: str = "", prefer_cloud: bool = False) -> tuple[str, str]:
    """
    Smart routing:
      - provider == "ollama" → always local
      - provider == specific cloud → always that provider
      - provider == "auto" → prefer cloud if Lao/complex, else try Ollama first
    Returns (response_text, label).
    """
    # Explicit Ollama only
    if cfg.provider == "ollama":
        return call_ollama(prompt, cfg, system), f"🖥️ Ollama ({cfg.ollama_model})"

    # Explicit cloud provider
    if cfg.provider not in ("auto", "ollama", ""):
        t, lbl = call_provider(prompt, cfg, system)
        if t and not t.startswith("["):
            return t, lbl
        # Fallback to Ollama
        return call_ollama(prompt, cfg, system), f"🖥️ Ollama [fallback]"

    # Auto routing
    needs_cloud = prefer_cloud or cfg.language in ("lo", "both")

    if needs_cloud:
        t, lbl = call_provider(prompt, cfg, system)
        if t and not t.startswith("["):
            return t, lbl
        # Fall back to Ollama
        r = call_ollama(prompt, cfg, system)
        return r, f"🖥️ Ollama [cloud fallback]"

    # Try Ollama first (free)
    r = call_ollama(prompt, cfg, system)
    if not r.startswith("[Ollama not running"):
        return r, f"🖥️ Ollama ({cfg.ollama_model})"

    # Ollama down → use cloud
    t, lbl = call_provider(prompt, cfg, system)
    if t and not t.startswith("["):
        return t, f"{lbl} [Ollama unavailable]"

    return r, "🖥️ Ollama (unavailable)"


# ── SQL generation ────────────────────────────────────────────────────────────

SQL_SYSTEM = """You are an expert Microsoft SQL Server (MSSQL) query writer.
Your job is to convert a user's natural language question into a single valid T-SQL SELECT query.

Rules:
- Output ONLY the raw SQL query — no markdown, no explanation, no code fences.
- Use only SELECT statements. Never use INSERT, UPDATE, DELETE, DROP, ALTER, EXEC.
- Use SQL Server syntax: TOP instead of LIMIT, GETDATE(), CONVERT(), etc.
- Always add TOP 500 unless the user asks for everything.
- Use square brackets [TableName] for table names.
- If the question is ambiguous, make a reasonable assumption and query anyway.
- If you genuinely cannot map the question to any table, output exactly: CANNOT_GENERATE
"""

def generate_sql(question: str, schema: str, history: List[ChatMessage], cfg: AIConfig) -> str:
    # Build conversation context from last few messages
    ctx = ""
    if history:
        recent = history[-6:]  # last 3 exchanges
        ctx = "\n\nRecent conversation:\n" + "\n".join(
            f"{m.role.upper()}: {m.content[:200]}" for m in recent
        )

    prompt = f"""Database schema:
{schema}
{ctx}

User question: {question}

Write the SQL Server SELECT query:"""

    raw, _ = call_ai(prompt, cfg, system=SQL_SYSTEM, prefer_claude=True)

    # Extract SQL from code fences if AI added them anyway
    fence = re.search(r"```(?:sql)?\s*([\s\S]+?)```", raw, re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()

    return raw.strip()


# ── Answer generation ─────────────────────────────────────────────────────────

def lang_instruction(lang: str) -> str:
    if lang == "lo":
        return "ກະລຸນາຕອບເປັນພາສາລາວທີ່ຊັດເຈນ ແລະ ເຂົ້າໃຈງ່າຍ. (Answer in clear, simple Lao language.)"
    if lang == "both":
        return "Answer in both English first, then repeat in Lao (ພາສາລາວ)."
    return "Answer in clear, plain English."


ANSWER_SYSTEM = """You are a friendly data analyst assistant.
You receive a user's question, the SQL that was run, and the data results.
Your job is to explain what the data means in plain, conversational language — like talking to a colleague.
Be concise but complete. Use numbers and specifics from the data. Do not repeat the SQL in your answer."""


def generate_answer(question: str, sql: str, cols: List[str], data: List[Dict],
                    cfg: AIConfig) -> tuple[str, str]:
    lang = lang_instruction(cfg.language)

    # Compact data representation
    if not data:
        data_text = "The query returned no rows."
    else:
        rows_shown = min(len(data), 30)
        lines = [f"Columns: {', '.join(cols)}", f"Total rows: {len(data)}", "---"]
        for i, row in enumerate(data[:rows_shown], 1):
            lines.append(f"{i}. " + " | ".join(f"{k}: {v}" for k, v in row.items()))
        if len(data) > rows_shown:
            lines.append(f"... and {len(data) - rows_shown} more rows")
        data_text = "\n".join(lines)

    prompt = f"""{lang}

User question: {question}

SQL that was run:
{sql}

Query results:
{data_text}

Please answer the user's question based on the data above."""

    return call_ai(prompt, cfg, system=ANSWER_SYSTEM, prefer_claude=cfg.language in ("lo", "both"))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/test-connection")
async def test_connection(config: DBConfig):
    conn = get_connection(config)
    cur = conn.cursor()
    cur.execute("SELECT @@VERSION")
    version = (cur.fetchone()[0] or "").split("\n")[0]
    conn.close()
    return {"success": True, "version": version}


@app.post("/api/schema")
async def get_schema_endpoint(config: DBConfig):
    schema_text = get_schema(config)
    return {"schema": schema_text}


@app.post("/api/tables")
async def get_tables(config: DBConfig):
    conn = get_connection(config)
    cur = conn.cursor()
    cur.execute("""
        SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE IN ('BASE TABLE','VIEW')
        ORDER BY TABLE_TYPE, TABLE_SCHEMA, TABLE_NAME
    """)
    tables = [{"schema": r[0], "name": r[1], "type": r[2]} for r in cur.fetchall()]
    conn.close()
    return {"tables": tables}


@app.post("/api/query")
async def execute_query(request: QueryRequest):
    if not is_safe_sql(request.sql):
        raise HTTPException(400, "Only SELECT queries are allowed.")
    cols, data = run_sql(request.sql, request.db_config)
    return {"columns": cols, "data": data, "row_count": len(data)}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    1. Load DB schema
    2. AI generates SQL from question
    3. Execute SQL
    4. AI generates natural-language answer
    5. Return answer + SQL + data preview
    """
    question = request.message.strip()
    if not question:
        raise HTTPException(400, "Empty message")

    # Step 1: Schema
    try:
        schema = get_schema(request.db_config)
    except HTTPException as e:
        return {
            "answer": f"❌ Could not connect to the database: {e.detail}",
            "sql": None,
            "data": [],
            "columns": [],
            "ai_used": "—",
            "error": True,
        }

    # Step 2: Generate SQL
    sql = generate_sql(question, schema, request.history, request.ai_config)

    if sql == "CANNOT_GENERATE" or not sql:
        return {
            "answer": (
                "ຂ້ອຍບໍ່ສາມາດຊອກຫາຂໍ້ມູນທີ່ກ່ຽວຂ້ອງໃນຖານຂໍ້ມູນໄດ້.\n\n"
                "I couldn't map your question to any table in the database. "
                "Try asking about a specific table or topic you know exists, "
                "or ask me to 'list all tables' first."
            ),
            "sql": None,
            "data": [],
            "columns": [],
            "ai_used": "—",
            "error": False,
        }

    if not is_safe_sql(sql):
        return {
            "answer": "⚠️ The AI tried to generate an unsafe query. Only SELECT queries are allowed.",
            "sql": sql,
            "data": [],
            "columns": [],
            "ai_used": "—",
            "error": True,
        }

    # Step 3: Execute SQL
    try:
        cols, data = run_sql(sql, request.db_config, limit=500)
    except HTTPException as e:
        # Tell the AI about the error so it can explain
        answer_text = (
            f"I generated this query but it failed to run:\n\n```sql\n{sql}\n```\n\n"
            f"Error: `{e.detail}`\n\n"
            "Please try rephrasing your question or check if the table/column names are correct."
        )
        return {
            "answer": answer_text,
            "sql": sql,
            "data": [],
            "columns": [],
            "ai_used": "—",
            "error": True,
        }

    # Step 4: Generate natural-language answer
    answer, ai_used = generate_answer(question, sql, cols, data, request.ai_config)

    return {
        "answer": answer,
        "sql": sql,
        "data": data[:50],        # preview (full data for display)
        "columns": cols,
        "row_count": len(data),
        "ai_used": ai_used,
        "error": False,
    }


@app.post("/api/check-ollama")
async def check_ollama(body: dict):
    url = body.get("url", "http://localhost:11434")
    try:
        r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return {"available": True, "models": models}
    except:
        pass
    return {"available": False, "models": []}


@app.post("/api/test-ai-provider")
async def test_ai_provider(body: dict):
    """
    Test an external AI provider and return available models.
    body: { provider, api_key, base_url? }
    """
    provider = body.get("provider", "")
    api_key  = body.get("api_key", "").strip()
    if not api_key:
        raise HTTPException(400, "API key is required")

    # ── Claude ──────────────────────────────────────────────────────────────
    if provider == "claude":
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            # Minimal API call to validate key
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
            models = [
                {"id": "claude-haiku-4-5-20251001",   "name": "Claude Haiku (fastest, cheapest)"},
                {"id": "claude-sonnet-4-6",  "name": "Claude Sonnet (balanced)"},
                {"id": "claude-opus-4-6",    "name": "Claude Opus (most powerful)"},
            ]
            return {"success": True, "models": models}
        except Exception as e:
            raise HTTPException(400, f"Claude error: {e}")

    # ── OpenAI ──────────────────────────────────────────────────────────────
    if provider == "openai":
        try:
            r = requests.get("https://api.openai.com/v1/models",
                             headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
            if r.status_code != 200:
                raise HTTPException(400, f"OpenAI error {r.status_code}: {r.text[:200]}")
            all_models = r.json().get("data", [])
            # Filter to chat models only
            chat_models = [m for m in all_models if "gpt" in m["id"].lower() and "instruct" not in m["id"]]
            chat_models.sort(key=lambda m: m["id"], reverse=True)
            models = [{"id": m["id"], "name": m["id"]} for m in chat_models[:20]]
            if not models:
                models = [
                    {"id": "gpt-4o-mini", "name": "GPT-4o Mini (cheapest)"},
                    {"id": "gpt-4o",      "name": "GPT-4o"},
                ]
            return {"success": True, "models": models}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"OpenAI error: {e}")

    # ── Groq ────────────────────────────────────────────────────────────────
    if provider == "groq":
        try:
            r = requests.get("https://api.groq.com/openai/v1/models",
                             headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
            if r.status_code != 200:
                raise HTTPException(400, f"Groq error {r.status_code}: {r.text[:200]}")
            all_models = r.json().get("data", [])
            models = [{"id": m["id"], "name": m["id"]} for m in all_models]
            if not models:
                models = [
                    {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B (fast, free tier)"},
                    {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B"},
                    {"id": "mixtral-8x7b-32768",   "name": "Mixtral 8x7B"},
                ]
            return {"success": True, "models": models}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Groq error: {e}")

    # ── OpenRouter ──────────────────────────────────────────────────────────
    if provider == "openrouter":
        try:
            r = requests.get("https://openrouter.ai/api/v1/models",
                             headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
            if r.status_code != 200:
                raise HTTPException(400, f"OpenRouter error {r.status_code}: {r.text[:200]}")
            all_models = r.json().get("data", [])
            # Show free models first, then popular paid
            free = [m for m in all_models if ":free" in m.get("id","")]
            paid = [m for m in all_models if ":free" not in m.get("id","")][:30]
            combined = (free + paid)[:40]
            models = [{"id": m["id"], "name": m.get("name", m["id"])} for m in combined]
            return {"success": True, "models": models}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"OpenRouter error: {e}")

    raise HTTPException(400, f"Unknown provider: {provider}")


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n" + "═" * 55)
    print("  🤖 SQL AI Chatbot — v2.0")
    print("═" * 55)
    print("  Open your browser: http://localhost:8000")
    print("  Press Ctrl+C to stop")
    print("═" * 55 + "\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
