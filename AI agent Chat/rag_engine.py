# =============================================================================
#  rag_engine.py — Dancing Numbers RAG Engine v8
#  Key fixes:
#   - Query lowercased BEFORE embedding (fixes case sensitivity at vector level)
#   - Hard negative prompt blocklist (blocks generic expert/upsell sentences)
#   - Wrong error number filter applied at CHUNK retrieval level too
#   - Stronger filler patterns
# =============================================================================

import sqlite3
import re
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

warnings.filterwarnings("ignore")

# =============================================================================
#  CONFIG
# =============================================================================

DB_PATH          = "blogs.db"
FAISS_INDEX_PATH = "blogs_faiss.index"
CHUNK_MAP_PATH   = "chunks_map.npy"
EMBED_MODEL      = "all-MiniLM-L6-v2"
RERANK_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FAISS_TOP_K      = 20
MMR_LAMBDA       = 0.65
MMR_TOP_K        = 10
FINAL_K          = 3
MAX_PER_URL      = 2
MIN_SCORE        = 0.20
HYBRID_ALPHA     = 0.7
MAX_WORDS        = 60
MIN_WORDS        = 35

FALLBACK = (
    "I couldn't find specific information on that topic in the knowledge base. "
    "Please connect with the support team for direct assistance."
)

# =============================================================================
#  QUERY NORMALIZATION
#  Must happen BEFORE embedding so identical queries always get identical vectors
# =============================================================================

def _normalize_query(query: str) -> str:
    """
    Normalize query text BEFORE it is embedded.
    - Strip whitespace
    - Lowercase  ← KEY FIX: 'Error 155' and 'error 155' → same embedding
    - Collapse spaces
    - Remove special chars except digits, letters, spaces
    """
    q = query.strip()
    q = q.lower()                              # ← lowercase first
    q = re.sub(r'[^a-z0-9\s]', ' ', q)        # remove punctuation
    q = re.sub(r'\s+', ' ', q).strip()
    return q


def _extract_numbers(query: str) -> set:
    """Extract all digit sequences from query for error-code filtering."""
    return set(re.findall(r'\b\d+\b', query))


def _extract_terms(query: str) -> list:
    """Extract meaningful keywords from query for boosting."""
    stopwords = {'how', 'to', 'the', 'a', 'an', 'in', 'on', 'at', 'is',
                 'are', 'was', 'fix', 'get', 'do', 'my', 'for', 'with',
                 'what', 'why', 'when', 'where', 'can', 'i', 'and', 'or'}
    return [w for w in query.lower().split()
            if len(w) > 2 and w not in stopwords]


# =============================================================================
#  NEGATIVE PROMPT BLOCKLIST
#  Any sentence matching these patterns is ALWAYS rejected from the answer
#  regardless of its semantic score
# =============================================================================

_NEGATIVE_PATTERNS = [
    # Generic expert/professional upsell
    r"(professional|certified|expert|specialist).{0,40}(help|assist|support|resolve|fix)",
    r"(help|assist).{0,20}(you|resolve|fix).{0,20}(efficiently|quickly|easily)",
    r"expert.{0,30}(equipped with|knowledge|tools|resolve)",
    r"(saving|save) you time.{0,30}(reducing|further|complications)",
    r"reducing further complications",
    r"minimal disruption to your workflow",
    r"ensure the issue is fixed efficiently",
    r"no troubleshooting steps have resolved",
    r"professional.{0,20}(are|is).{0,20}equipped",

    # Generic boilerplate closings
    r"in this (guide|article|post|blog)",
    r"we have (discussed|covered|shared|explained|listed)",
    r"all the necessary information",
    r"insightful enough",
    r"above (process|steps|information|guide)",
    r"sure that the above",
    r"(hope|trust|believe) (this|it|you|that)",
    r"we are sure",
    r"(will|would) (be|prove) (helpful|useful|insightful)",

    # Dancing Numbers brand upsell
    r"dancing numbers (you can|helps?|will|is|software|tool)",
    r"using dancing numbers",
    r"with (the help of )?dancing numbers",
    r"(save|saving) time and (increas|boost)",
    r"(easy to use|user.friendly)",
    r"no human errors?",
    r"works automatically",
    r"simplif(y|ies|ied) and automat",

    # Generic action prompts
    r"retrace your steps",
    r"make it error.free",
    r"just fill in a few (fields|details)",
    r"still not comfortable",
    r"further queries",
    r"you can avail",
    r"any time you want",
    r"at your convenience",
    r"(seek|get).{0,20}(help|assistance).{0,20}(expert|professional|specialist)",

    # Generic issue/article filler
    r"(import|export).{0,25}(delete|lists|transactions).{0,25}company file",
    r"everything in theory",
    r"(theory|practice) seems (very )?easy",
    r"(different|difficult) when practically",
    r"have been shared",
    r"related (situations?|issues?|queries)",
    r"(will help you out|help you out)",
    r"once you have finished",
    r"simply run your report",
    r"you can simply",
    r"in which these",
    r"consider seeking professional help",
    r"if you.ve tried all",
    r"suggested solutions and the issue persists",
    r"persistent after running this program",
    r"(contact|reach out to).{0,20}(quickbooks|intuit).{0,20}(support|team|experts)",

    # Vague resolution sentences
    r"(resolve|fix|solve).{0,20}(issue|problem|error).{0,20}(quickly|efficiently|easily)",
    r"(error|issue).{0,20}(continues to|will).{0,20}(disrupt|affect|persist)",
    r"an expert can provide",
    r"experts are equipped",
]
_NEGATIVE_RE = re.compile("|".join(_NEGATIVE_PATTERNS), re.IGNORECASE)


def _is_negative(s: str) -> bool:
    """Return True if sentence matches any negative prompt pattern."""
    return bool(_NEGATIVE_RE.search(s))


# =============================================================================
#  WRONG ERROR NUMBER FILTER
#  Applied at both chunk-scoring and sentence levels
# =============================================================================

def _has_wrong_error_number(text: str, query_numbers: set) -> bool:
    """
    Return True if text mentions error/code numbers NOT in the query.
    e.g. query='error 155', text mentions 'error 1334' → True (reject)
    """
    if not query_numbers:
        return False
    text_numbers    = set(re.findall(r'\b\d+\b', text))
    foreign_numbers = text_numbers - query_numbers
    for num in foreign_numbers:
        # Only reject if number appears next to error-related word
        if re.search(rf'\b(error|code|issue|#)\s*{re.escape(num)}\b',
                     text, re.IGNORECASE):
            return True
    return False


def _chunk_error_relevance(chunk_text: str, query_numbers: set) -> float:
    """
    Boost chunks that mention the exact error numbers from the query.
    Returns multiplier: 1.3 if query numbers present, 0.6 if wrong numbers, 1.0 otherwise.
    """
    if not query_numbers:
        return 1.0
    chunk_numbers = set(re.findall(r'\b\d+\b', chunk_text))
    if query_numbers & chunk_numbers:        # query numbers appear in chunk
        return 1.3
    if _has_wrong_error_number(chunk_text, query_numbers):
        return 0.6                           # penalise wrong error numbers
    return 1.0


# =============================================================================
#  ACTIONABLE SENTENCE CHECK
# =============================================================================

def _is_actionable(s: str) -> bool:
    if len(s.split()) < 6:
        return False
    if re.search(r'\b(step\s*\d+|\d+[\.\)]\s)', s, re.IGNORECASE):
        return True
    action_starts = {
        "go", "click", "open", "select", "choose", "navigate", "enter",
        "type", "press", "check", "uncheck", "enable", "disable", "create",
        "update", "change", "set", "verify", "confirm", "save", "delete",
        "add", "remove", "find", "search", "view", "access", "login",
        "download", "install", "run", "launch", "use", "make", "ensure",
        "first", "next", "then", "finally", "now", "sign", "restart",
        "repair", "reinstall", "reboot", "close", "clear", "reset",
        "error", "this", "if", "when", "the",
    }
    qb_terms = {
        "quickbooks", "invoice", "payment", "vendor", "customer", "account",
        "transaction", "reconcil", "report", "payroll", "tax", "bank",
        "deposit", "balance", "journal", "entry", "chart", "discount",
        "credit", "debit", "expense", "income", "budget", "error", "issue",
        "problem", "fix", "troubleshoot", "install", "update", "repair",
        "password", "license", "subscription", "file", "backup", "restore",
    }
    w0 = s.lower().split()[0]
    if w0 in action_starts:
        return True
    if any(t in s.lower() for t in qb_terms):
        return True
    return False


# =============================================================================
#  KEYWORD BOOST
# =============================================================================

def _keyword_boost(sentence: str, query_terms: list) -> float:
    if not query_terms:
        return 0.0
    sl   = sentence.lower()
    hits = sum(1 for t in query_terms if t in sl)
    return min(hits / len(query_terms), 1.0)


# =============================================================================
#  FTS5 BM25 KEYWORD SCORING
# =============================================================================

def _ensure_fts(db_path: str):
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    try:
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(chunk_text, content='chunks', content_rowid='chunk_id',
                       tokenize='porter ascii')
        """)
        cur.execute("SELECT COUNT(*) FROM chunks_fts")
        if cur.fetchone()[0] == 0:
            cur.execute(
                "INSERT INTO chunks_fts(rowid, chunk_text) "
                "SELECT chunk_id, chunk_text FROM chunks"
            )
        conn.commit()
    except Exception as e:
        print(f"FTS5 note: {e}")
    finally:
        conn.close()


def _bm25_scores(query: str, chunk_ids: list, db_path: str) -> dict:
    if not chunk_ids:
        return {}
    fts_q = re.sub(r'[^\w\s]', ' ', query).strip()
    if not fts_q:
        return {}
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    ph   = ",".join(str(c) for c in chunk_ids)
    try:
        cur.execute(f"""
            SELECT rowid, bm25(chunks_fts) AS score
            FROM   chunks_fts
            WHERE  chunks_fts MATCH ?
              AND  rowid IN ({ph})
            ORDER  BY score
        """, (fts_q,))
        rows = cur.fetchall()
    except Exception:
        conn.close()
        return {}
    conn.close()
    if not rows:
        return {}
    raw  = {r[0]: -r[1] for r in rows}
    maxv = max(raw.values()) or 1.0
    return {cid: v / maxv for cid, v in raw.items()}


# =============================================================================
#  QUERY EXPANSION
# =============================================================================

def _expand_query(query: str) -> list:
    """
    Generate variants of the (already normalized lowercase) query.
    All variants stay lowercase so embeddings remain consistent.
    """
    q        = query.strip().rstrip("?")
    variants = {q}
    if "quickbooks" not in q:
        variants.add(f"{q} quickbooks")
    if not q.startswith("how"):
        variants.add(f"how to {q} quickbooks")
    if not any(w in q for w in ["fix", "solve", "resolve"]):
        variants.add(f"fix {q} quickbooks")
    variants.add(f"resolve {q} quickbooks steps")
    return list(variants)[:4]


# =============================================================================
#  LOAD ASSETS  — returns exactly 4 values
# =============================================================================

def load_assets():
    """Returns: embed_model, reranker, faiss_index, chunk_id_map"""
    print("Loading embedding model...")
    embed_model  = SentenceTransformer(EMBED_MODEL)
    print("Loading cross-encoder reranker...")
    reranker     = CrossEncoder(RERANK_MODEL, max_length=512)
    print("Loading FAISS index...")
    faiss_index  = faiss.read_index(FAISS_INDEX_PATH)
    print("Loading chunk ID map...")
    chunk_id_map = np.load(CHUNK_MAP_PATH)
    print("Setting up FTS5 keyword index...")
    _ensure_fts(DB_PATH)
    print(f"Ready — {faiss_index.ntotal:,} vectors indexed")
    return embed_model, reranker, faiss_index, chunk_id_map


# =============================================================================
#  FAISS RETRIEVAL
# =============================================================================

def _faiss_retrieve(queries: list, embed_model, faiss_index, chunk_id_map) -> dict:
    """All queries must already be normalized/lowercased."""
    embs = embed_model.encode(
        queries, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)
    merged = {}
    for emb in embs:
        scores, indices = faiss_index.search(emb.reshape(1, -1), FAISS_TOP_K)
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or float(score) < MIN_SCORE:
                continue
            cid = int(chunk_id_map[idx])
            merged[cid] = max(merged.get(cid, 0.0), float(score))
    return merged


# =============================================================================
#  FETCH CHUNKS
# =============================================================================

def _fetch_chunks(chunk_ids: list, db_path: str) -> dict:
    if not chunk_ids:
        return {}
    ph   = ",".join(["?"] * len(chunk_ids))
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    try:
        cur.execute(
            f"""SELECT c.chunk_id, c.chunk_text, b.title, b.url,
                       COALESCE(c.h1, b.title, '') AS h1
                FROM chunks c JOIN blogs b ON c.blog_id = b.id
                WHERE c.chunk_id IN ({ph})""", chunk_ids)
    except sqlite3.OperationalError:
        cur.execute(
            f"""SELECT c.chunk_id, c.chunk_text, b.title, b.url, b.title AS h1
                FROM chunks c JOIN blogs b ON c.blog_id = b.id
                WHERE c.chunk_id IN ({ph})""", chunk_ids)
    rows = cur.fetchall()
    conn.close()
    return {
        r[0]: {"chunk_text": r[1], "title": r[2] or "",
               "url": r[3], "h1": r[4] or ""}
        for r in rows
    }


# =============================================================================
#  HYBRID SCORING + ERROR NUMBER BOOST
# =============================================================================

def _hybrid_score(semantic: dict, keyword: dict,
                  lookup: dict, query_numbers: set) -> dict:
    """
    Combine semantic + BM25 scores, then multiply by error-number relevance.
    Chunks with correct error number get boosted; wrong error number chunks penalised.
    """
    all_ids = set(semantic) | set(keyword)
    scores  = {}
    for cid in all_ids:
        base = (HYBRID_ALPHA * semantic.get(cid, 0.0)
                + (1 - HYBRID_ALPHA) * keyword.get(cid, 0.0))
        # Apply error-number multiplier at chunk level
        chunk_text = lookup.get(cid, {}).get("chunk_text", "")
        multiplier = _chunk_error_relevance(chunk_text, query_numbers)
        scores[cid] = base * multiplier
    return scores


# =============================================================================
#  URL DEDUPLICATION
# =============================================================================

def _limit_per_url(sorted_ids: list, lookup: dict,
                   max_per: int = MAX_PER_URL) -> list:
    url_count, out = {}, []
    for cid in sorted_ids:
        url = lookup.get(cid, {}).get("url", "")
        if url_count.get(url, 0) < max_per:
            out.append(cid)
            url_count[url] = url_count.get(url, 0) + 1
    return out


# =============================================================================
#  MMR RERANKING
# =============================================================================

def _mmr(chunk_ids: list, scores: dict, embed_model,
         lookup: dict, final_k: int = MMR_TOP_K) -> list:
    ids = [cid for cid in chunk_ids if cid in lookup]
    if not ids:
        return []
    embs = embed_model.encode(
        [lookup[cid]["chunk_text"] for cid in ids],
        convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False
    ).astype(np.float32)
    emb_map = {cid: emb for cid, emb in zip(ids, embs)}
    selected, sel_embs, remaining = [], [], list(ids)
    while len(selected) < final_k and remaining:
        best_id, best_sc = None, -np.inf
        for cid in remaining:
            rel  = scores.get(cid, 0.0)
            msim = float(np.max(np.dot(emb_map[cid], np.array(sel_embs).T))) \
                   if sel_embs else 0.0
            sc   = MMR_LAMBDA * rel - (1 - MMR_LAMBDA) * msim
            if sc > best_sc:
                best_sc, best_id = sc, cid
        if best_id is None:
            break
        selected.append(best_id)
        sel_embs.append(emb_map[best_id])
        remaining.remove(best_id)
    return selected


# =============================================================================
#  CROSS-ENCODER RERANKER
# =============================================================================

def _rerank(query: str, chunk_ids: list, lookup: dict,
            reranker, final_k: int = FINAL_K) -> list:
    ids = [cid for cid in chunk_ids if cid in lookup]
    if not ids:
        return []
    pairs  = [(query, lookup[cid]["chunk_text"]) for cid in ids]
    ce_sc  = reranker.predict(pairs)
    ranked = sorted(zip(ids, ce_sc), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in ranked[:final_k]]


# =============================================================================
#  ANSWER BUILDER — ≤60 words
# =============================================================================

def _clean_sent(s: str) -> str:
    s = re.sub(r'^[\.\,\:\-\•\*\s]+', '', s).strip()
    if not s:
        return ""
    s = s[0].upper() + s[1:]
    if s[-1] not in ".!?":
        s += "."
    return s


def _ngrams(words: list, n: int = 4) -> set:
    return set(" ".join(words[i:i+n]) for i in range(max(0, len(words)-n+1)))


def _build_answer(chunks: list, q_emb, embed_model,
                  query_numbers: set, query_terms: list) -> str:
    """
    Three-pass sentence selection:
    Pass 1: actionable + not negative + correct error numbers
    Pass 2: not negative + correct error numbers  (relaxed)
    Pass 3: just not negative (last resort)
    Then rank by semantic score + keyword boost, cap at 60 words.
    """
    def _passes(s, strict=True):
        if _is_negative(s):
            return False
        if _has_wrong_error_number(s, query_numbers):
            return False
        if len(s.split()) < 6:
            return False
        if strict and not _is_actionable(s):
            return False
        return True

    # Pass 1 — strict
    pool = [s.strip()
            for chunk in chunks
            for s in re.split(r'(?<=[.!?])\s+', chunk["chunk_text"])
            if _passes(s.strip(), strict=True)]

    # Pass 2 — relaxed
    if len(pool) < 3:
        pool = [s.strip()
                for chunk in chunks
                for s in re.split(r'(?<=[.!?])\s+', chunk["chunk_text"])
                if _passes(s.strip(), strict=False)]

    # Pass 3 — last resort (just block negatives)
    if len(pool) < 2:
        pool = [s.strip()
                for chunk in chunks
                for s in re.split(r'(?<=[.!?])\s+', chunk["chunk_text"])
                if len(s.strip().split()) >= 6 and not _is_negative(s.strip())]

    if not pool:
        return ""

    # Score sentences
    embs       = embed_model.encode(pool, convert_to_numpy=True,
                                    normalize_embeddings=True,
                                    show_progress_bar=False)
    sem_scores = np.dot(embs, q_emb.T).flatten()
    combined   = [
        sem + 0.2 * _keyword_boost(sent, query_terms)
        for sent, sem in zip(pool, sem_scores.tolist())
    ]
    ranked = sorted(zip(pool, combined), key=lambda x: x[1], reverse=True)

    # Greedily build answer within word budget
    chosen, wc, seen_ng = [], 0, set()
    for sent, _ in ranked:
        sent  = _clean_sent(sent)
        if not sent:
            continue
        words = sent.split()
        n     = len(words)
        if wc + n > MAX_WORDS and wc >= MIN_WORDS:
            break
        if wc + n > MAX_WORDS + 10:
            break
        ng = _ngrams(words)
        if len(ng & seen_ng) > 3:
            continue
        chosen.append(sent)
        seen_ng.update(ng)
        wc += n

    return " ".join(chosen)


# =============================================================================
#  FOLLOW-UP CONTEXT
# =============================================================================

_FOLLOWUP = {"it","this","that","same","above","how","why","what",
             "more","else","also","another","steps","fix"}


def _augment(query: str, history: list) -> str:
    if not history:
        return query
    if len(query.split()) <= 7 and bool(set(query.split()) & _FOLLOWUP):
        last_q = history[-1].get("user", "")
        return f"{last_q} {query}".strip()
    return query


# =============================================================================
#  MAIN ENTRY — generate_response()
# =============================================================================

def generate_response(query: str, embed_model, reranker,
                      faiss_index, chunk_id_map,
                      history: list = None,
                      db_path: str = DB_PATH) -> dict:
    """
    Returns:
        {"answer": str, "sources": list[dict], "found": bool}
    """
    if history is None:
        history = []

    # ── Normalize BEFORE everything — fixes case sensitivity ──────────
    normalized    = _normalize_query(query)
    effective     = _augment(normalized, history)

    query_numbers = _extract_numbers(normalized)
    query_terms   = _extract_terms(normalized)

    # ── Query expansion (all lowercase) ──────────────────────────────
    variants = _expand_query(effective)

    # ── FAISS retrieval ───────────────────────────────────────────────
    semantic = _faiss_retrieve(variants, embed_model, faiss_index, chunk_id_map)
    if not semantic:
        return {"answer": FALLBACK, "sources": [], "found": False}

    candidate_ids = list(semantic.keys())

    # ── Fetch chunk data ──────────────────────────────────────────────
    lookup = _fetch_chunks(candidate_ids, db_path)

    # ── BM25 keyword scoring ──────────────────────────────────────────
    bm25 = _bm25_scores(effective, candidate_ids, db_path)

    # ── Hybrid score (with error-number chunk boost) ──────────────────
    hybrid = _hybrid_score(semantic, bm25, lookup, query_numbers)

    # ── Sort + URL dedup ──────────────────────────────────────────────
    sorted_ids = sorted(hybrid, key=lambda c: hybrid[c], reverse=True)
    deduped    = _limit_per_url(sorted_ids, lookup)

    # ── MMR (20 → 10) ─────────────────────────────────────────────────
    mmr_ids = _mmr(deduped[:FAISS_TOP_K], hybrid, embed_model, lookup, MMR_TOP_K)

    # ── Cross-encoder reranker (10 → 3) ──────────────────────────────
    final_ids = _rerank(effective, mmr_ids, lookup, reranker, FINAL_K)
    if not final_ids:
        return {"answer": FALLBACK, "sources": [], "found": False}

    # ── Build answer ──────────────────────────────────────────────────
    top_chunks = [lookup[cid] for cid in final_ids if cid in lookup]

    q_emb = embed_model.encode(
        [effective], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)

    answer = _build_answer(top_chunks, q_emb, embed_model,
                           query_numbers, query_terms)

    if not answer:
        raw    = re.split(r'(?<=[.!?])\s+', top_chunks[0]["chunk_text"])
        answer = " ".join(
            _clean_sent(s) for s in raw
            if len(s.split()) >= 6
            and not _is_negative(s)
            and not _has_wrong_error_number(s, query_numbers)
        )

    # ── Top-2 source blogs ────────────────────────────────────────────
    seen_urls, sources = set(), []
    for cid in final_ids:
        ch  = lookup.get(cid, {})
        url = ch.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append({"title": ch.get("title", ""), "url": url})
        if len(sources) == 2:
            break

    return {"answer": answer, "sources": sources, "found": True}