import streamlit as st
import streamlit.components.v1 as components
import time, base64, os
from rag_engine import load_assets, generate_response

st.set_page_config(page_title="Dancing Numbers AI Support", page_icon="💃", layout="centered")

# ── Logo ──────────────────────────────────────────────────────────────────────
def get_logo_b64():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    return None

LOGO = get_logo_b64()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:#0f1117;color:#e8eaf0;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.8rem;padding-bottom:7rem;max-width:740px;}
.dn-header{display:flex;align-items:center;gap:14px;padding:16px 22px;
  background:linear-gradient(135deg,#0e2d1f,#091a0f);border:1px solid #1a4a2e;
  border-radius:14px;margin-bottom:24px;box-shadow:0 4px 20px rgba(0,0,0,.45);}
.dn-logo{width:44px;height:44px;border-radius:10px;object-fit:contain;}
.dn-logo-fb{font-size:1.9rem;}
.dn-title h1{margin:0;font-size:1.15rem;font-weight:700;color:#4ade80;}
.dn-title p{margin:2px 0 0;font-size:.76rem;color:#6b7280;}
.dn-badge{margin-left:auto;background:#052e16;border:1px solid #166534;color:#4ade80;
  font-size:.68rem;font-weight:600;padding:3px 10px;border-radius:20px;}
.user-row{display:flex;flex-direction:row-reverse;gap:11px;margin-bottom:10px;animation:fu .28s ease;}
@keyframes fu{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:translateY(0)}}
.u-av{width:32px;height:32px;border-radius:50%;background:#1e3a5f;overflow:hidden;
  display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:3px;}
.u-av img{width:100%;height:100%;object-fit:contain;}
.u-bub{max-width:88%;padding:12px 16px;border-radius:14px;border-top-right-radius:3px;
  font-size:.91rem;line-height:1.68;background:#1e3a5f;border:1px solid #1e40af22;color:#dbeafe;}
.typing-row{display:flex;gap:11px;margin-bottom:10px;}
.a-av{width:32px;height:32px;border-radius:50%;background:#052e16;border:1px solid #166534;
  overflow:hidden;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:3px;}
.a-av img{width:100%;height:100%;object-fit:contain;}
.t-bub{padding:14px 18px;background:#0d1f13;border:1px solid #1a4a2e;
  border-radius:14px;border-top-left-radius:3px;}
.td{display:inline-block;width:7px;height:7px;background:#4ade80;border-radius:50%;
  margin:0 2px;animation:blink 1.2s infinite;}
.td:nth-child(2){animation-delay:.2s}.td:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,80%,100%{opacity:.2;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}
.stChatInput>div{border:1px solid #1a4a2e !important;border-radius:12px !important;background:#0e1a12 !important;}
.stChatInput textarea{color:#d1fae5 !important;}
.es{text-align:center;padding:32px 20px 18px;}
.es h2{color:#374151;font-size:1.05rem;margin:8px 0 4px;}
.es p{font-size:.82rem;color:#6b7280;}
</style>
""", unsafe_allow_html=True)


# ── Cached assets ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_assets():
    return load_assets()   # returns (embed_model, reranker, faiss_index, chunk_id_map)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("messages",[]),("history",[]),("chip_query",None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Header ────────────────────────────────────────────────────────────────────
logo_tag = f'<img class="dn-logo" src="{LOGO}">' if LOGO else '<span class="dn-logo-fb">💃</span>'
st.markdown(f"""
<div class="dn-header">
  {logo_tag}
  <div class="dn-title">
    <h1>Dancing Numbers AI Support</h1>
    <p>QuickBooks knowledge base &nbsp;·&nbsp; Instant answers</p>
  </div>
  <div class="dn-badge">RAG v6 · Local AI</div>
</div>
""", unsafe_allow_html=True)


# ── Suggestion chips ──────────────────────────────────────────────────────────
CHIPS = [
    "Fix payment issues in QuickBooks?",
    "Reconcile accounts in QuickBooks?",
    "Import transactions into QuickBooks?",
    "QuickBooks backup not working?",
    "Integrate QuickBooks with Excel?",
]

if not st.session_state.messages:
    logo_big = (f'<img src="{LOGO}" style="width:60px;height:60px;object-fit:contain;border-radius:12px;">'
                if LOGO else '<span style="font-size:2.4rem">🤖</span>')
    st.markdown(f"""
    <div class="es">{logo_big}
      <h2>How can I help you today?</h2>
      <p>Ask anything about QuickBooks — powered by the Dancing Numbers knowledge base.</p>
    </div>""", unsafe_allow_html=True)
    cols = st.columns(len(CHIPS))
    for i, (col, sug) in enumerate(zip(cols, CHIPS)):
        with col:
            if st.button(sug, key=f"c{i}", use_container_width=True):
                st.session_state.chip_query = sug
    st.markdown('<hr style="border-color:#1a3a24;margin:14px 0 20px">', unsafe_allow_html=True)


# ── Agent bubble (components.html — bypasses sanitizer) ───────────────────────
def _agent_html(answer, sources, logo):
    av = (f'<img src="{logo}" style="width:100%;height:100%;object-fit:contain;border-radius:50%;">'
          if logo else "💃")

    ref = ""
    if sources:
        links = ""
        for i, s in enumerate(sources[:2], 1):
            t = s.get("title","Article")
            u = s.get("url","#")
            sh = (t[:62]+"…") if len(t)>62 else t
            links += f"""<a class="rl" href="{u}" target="_blank">
              <span class="rn">{i}</span><span>📄</span><span class="rt">{sh}</span></a>"""
        ref = f'<div class="rs"><div class="rb">📖 Reference Blogs</div>{links}</div>'

    sup = """<div class="sb">
      <span class="st">Need help?</span>
      <a class="sc" href="javascript:void(0);"
         onclick="try{{parent.$zopim.livechat.window.show();}}catch(e){{window.open('https://www.dancingnumbers.com/','_blank');}}">
        💬 Connect with Support Team</a></div>"""

    return f"""<!DOCTYPE html><html><head>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
*{{margin:0;padding:0;box-sizing:border-box;font-family:'DM Sans',sans-serif;}}
body{{background:transparent;padding:0 2px;}}
.row{{display:flex;gap:11px;align-items:flex-start;}}
.av{{width:32px;height:32px;min-width:32px;border-radius:50%;background:#052e16;
  border:1px solid #166534;overflow:hidden;display:flex;align-items:center;
  justify-content:center;margin-top:3px;font-size:1rem;}}
.bub{{flex:1;padding:14px 17px;background:#0d1f13;border:1px solid #1a4a2e;
  border-radius:14px;border-top-left-radius:3px;}}
.sm{{font-size:14.5px;line-height:1.72;color:#d1fae5;margin-bottom:14px;}}
.rs{{margin-bottom:14px;}}
.rb{{font-size:11px;font-weight:600;color:#6b7280;text-transform:uppercase;
  letter-spacing:.07em;margin-bottom:7px;}}
.rl{{display:flex;align-items:center;gap:8px;padding:9px 13px;background:#052e16;
  border:1px solid #166534;border-radius:10px;color:#4ade80;text-decoration:none;
  font-size:13px;font-weight:500;margin-bottom:7px;word-break:break-word;}}
.rl:last-child{{margin-bottom:0;}}
.rl:hover{{background:#064e24;}}
.rn{{background:#166534;color:#4ade80;font-size:11px;font-weight:700;
  width:18px;height:18px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;}}
.rt{{flex:1;}}
.sb{{display:flex;align-items:center;justify-content:space-between;gap:10px;
  flex-wrap:wrap;border-top:1px solid #1a3a24;padding-top:12px;}}
.st{{font-size:13px;color:#9ca3af;}}
.sc{{display:inline-flex;align-items:center;gap:6px;background:#15803d;color:#fff !important;
  text-decoration:none;border-radius:8px;padding:7px 16px;font-size:13px;
  font-weight:600;white-space:nowrap;cursor:pointer;}}
.sc:hover{{background:#166534;}}
</style></head><body>
<div class="row">
  <div class="av">{av}</div>
  <div class="bub">
    <div class="sm">{answer}</div>
    {ref}{sup}
  </div>
</div>
</body></html>"""


def _bubble_h(answer, n_sources):
    wc = len(answer.split())
    return 160 + max(1, wc//10)*24 + n_sources*60 + 30


# ── Render helpers ────────────────────────────────────────────────────────────
def render_user(text):
    av = (f'<div class="u-av"><img src="{LOGO}"></div>'
          if LOGO else '<div class="u-av" style="font-size:.95rem">👤</div>')
    st.markdown(f'<div class="user-row">{av}<div class="u-bub">{text}</div></div>',
                unsafe_allow_html=True)


def render_agent(answer, sources):
    components.html(_agent_html(answer, sources, LOGO),
                    height=_bubble_h(answer, len(sources or [])), scrolling=False)


def render_typing():
    av = (f'<div class="a-av"><img src="{LOGO}"></div>'
          if LOGO else '<div class="a-av" style="font-size:.95rem">💃</div>')
    ph = st.empty()
    ph.markdown(f"""<div class="typing-row">{av}
      <div class="t-bub">
        <span class="td"></span><span class="td"></span><span class="td"></span>
      </div></div>""", unsafe_allow_html=True)
    return ph


# ── Render history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        render_user(msg["content"])
    else:
        render_agent(msg["content"], msg.get("sources", []))


# ── Process query ─────────────────────────────────────────────────────────────
def process_query(q: str):
    if not q.strip():
        return

    st.session_state.messages.append({"role":"user","content":q,"sources":[]})
    render_user(q)
    ph = render_typing()

    embed_model, reranker, faiss_index, chunk_id_map = get_assets()

    result = generate_response(
        query        = q,
        embed_model  = embed_model,
        reranker     = reranker,
        faiss_index  = faiss_index,
        chunk_id_map = chunk_id_map,
        history      = st.session_state.history,
    )
    time.sleep(0.3)
    ph.empty()

    answer  = result["answer"]
    sources = result.get("sources", []) if result["found"] else []

    render_agent(answer, sources)
    st.session_state.messages.append({"role":"assistant","content":answer,"sources":sources})
    st.session_state.history.append({"user":q,"assistant":answer})
    if len(st.session_state.history) > 6:
        st.session_state.history = st.session_state.history[-6:]


if st.session_state.chip_query:
    q = st.session_state.chip_query
    st.session_state.chip_query = None
    process_query(q)

inp = st.chat_input("Ask a QuickBooks question…")
if inp:
    process_query(inp)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if LOGO:
        st.markdown(f'<img src="{LOGO}" style="width:80px;border-radius:10px;margin-bottom:8px;">',
                    unsafe_allow_html=True)
    st.markdown("### Dancing Numbers\n**AI Support Agent**\n---")
    st.markdown("""
**Embed** `all-MiniLM-L6-v2`
**Reranker** `ms-marco-MiniLM-L-6`
**Search** FAISS + BM25 + MMR
**DB** SQLite + FTS5
**Answer** ≤ 60 words
`100% local · no paid API`
    """)
    st.markdown("---")
    turns = len([m for m in st.session_state.messages if m["role"]=="user"])
    st.metric("Questions asked", turns)
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()