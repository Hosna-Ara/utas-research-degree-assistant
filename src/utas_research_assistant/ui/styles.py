"""Original, restrained visual system for the Streamlit application."""

APP_CSS = """
<style>
:root {
  --canvas: #f8f8f9; --surface: #ffffff; --ink: #20232b; --muted: #68717d;
  --line: #e5e5e7; --indigo: #762333; --indigo-soft: #f7edef;
}
html, body, [class*="css"] { font-family: Inter, "Avenir Next", -apple-system, BlinkMacSystemFont, sans-serif; color: var(--ink); }
.stApp { background: var(--canvas); }
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding: 1.25rem .85rem; }
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: var(--ink); }
[data-testid="stCaptionContainer"], .stCaption { color: #596780 !important; }
h1, h2, h3 { color: var(--ink) !important; letter-spacing: -.025em; }
a { color: #4c45c5 !important; }
.brand { display:flex; align-items:center; gap:.65rem; margin: .1rem .25rem 1.2rem; color:var(--ink); }
.brand-mark { width:2rem; height:2rem; display:grid; place-items:center; border-radius:10px; background:var(--indigo); color:#fff; font-weight:700; }
.brand strong { display:block; font-size:.9rem; letter-spacing:-.01em; }
.brand span { display:block; color:var(--muted); font-size:.73rem; margin-top:.08rem; }
.nav-section { color:#7a8496; font-size:.67rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; margin:1.15rem .35rem .4rem; }
.nav-item { display:flex; gap:.6rem; align-items:center; padding:.55rem .65rem; border-radius:9px; color:#526078; font-size:.85rem; margin:.12rem 0; }
.nav-item-active { color:var(--indigo); background:var(--indigo-soft); font-weight:700; }
.nav-item-muted { color:#7b8698; }
.nav-item-muted small { display:block; color:#9aa3b2; font-size:.6rem; margin-top:.1rem; }
.feature-list { color:#596780; font-size:.72rem; line-height:1.8; margin:0 .4rem; }
[data-testid="stSidebar"] .stCaption { color:#68758a !important; }
[data-testid="stSidebar"] div.stButton > button { min-height:2.15rem; text-align:left; font-size:.77rem; padding:.35rem .55rem; }
[data-testid="stSidebar"] div.stButton > button p { color:#26344c !important; }
[data-testid="stSidebar"] [data-testid="baseButton-primary"], [data-testid="stSidebar"] button[kind="primary"] { background:#762333 !important; border-color:#762333 !important; color:#fff !important; }
[data-testid="stSidebar"] [data-testid="baseButton-primary"] *, [data-testid="stSidebar"] button[kind="primary"] * { color:#fff !important; fill:#fff !important; }
[data-testid="stSidebar"] [data-testid="baseButton-primary"]:hover, [data-testid="stSidebar"] button[kind="primary"]:hover { background:#5f1b29 !important; border-color:#5f1b29 !important; }
.snapshot-date { color:#536078; font-size:.8rem; margin:0 .35rem .65rem; }
.snapshot-stats { display:grid; grid-template-columns:repeat(3,1fr); gap:.3rem; margin:.2rem .25rem .85rem; }
.snapshot-stats div { background:#fafbfe; border:1px solid var(--line); border-radius:9px; padding:.48rem .25rem; text-align:center; }
.snapshot-stats b { display:block; color:var(--ink); font-size:.95rem; }
.snapshot-stats span { color:#69768b; font-size:.61rem; }
.status-list { display:grid; gap:.42rem; color:#5d6a80; font-size:.75rem; margin:.75rem .35rem; }
.status-list i { display:inline-block; width:7px; height:7px; border-radius:50%; background:#5fbc91; margin-right:.42rem; }
.sidebar-footer { border-top:1px solid var(--line); color:#526078; font-size:.72rem; line-height:1.5; margin:1.5rem .25rem 0; padding-top:.8rem; }
.sidebar-footer strong { color:#762333; font-weight:800; }
.sidebar-footer span { color:#748096; }
.workspace-top { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin:.15rem 0 2.5rem; }
.workspace-title { display:flex; align-items:center; gap:.6rem; color:var(--ink); font-size:1rem; font-weight:750; }
.workspace-tagline { color:var(--muted); font-size:.78rem; margin:.35rem 0 0 2.4rem; }
.workspace-icon { display:grid; place-items:center; width:1.8rem; height:1.8rem; border-radius:8px; background:var(--indigo-soft); color:var(--indigo); }
.assignment-label { color:var(--indigo); font-size:.7rem; font-weight:750; letter-spacing:.05em; text-transform:uppercase; }
.empty-state { max-width:760px; margin:3rem auto 1.35rem; text-align:center; }
.empty-state .eyebrow { color:var(--indigo); font-size:.72rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
.empty-state h1 { font-size:clamp(2rem,4vw,3rem); margin:.5rem 0 .55rem; }
.empty-state p { color:var(--muted); font-size:1rem; margin:0 auto; max-width:560px; line-height:1.6; }
[data-testid="stForm"] { position:sticky; bottom:.65rem; z-index:10; background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:.35rem .5rem; box-shadow:0 10px 28px rgba(32,42,75,.08); margin:.8rem auto 1.2rem; max-width:900px; }
[data-testid="stForm"] [data-testid="stTextInput"] > div { background:#fff; border:1px solid #dfe3eb; border-radius:11px; }
[data-testid="stForm"] [data-testid="stTextInput"] > div:focus-within { border-color:var(--indigo); box-shadow:0 0 0 2px rgba(118,35,51,.14); }
[data-testid="stForm"] [data-testid="stTextInput"] input { background:#fff !important; color:#182338 !important; caret-color:var(--indigo); font-size:.98rem; }
[data-testid="stForm"] [data-testid="stTextInput"] input::placeholder { color:#7a8496 !important; opacity:1; }
[data-testid="stForm"] [data-testid="stTextInput"] input:disabled { background:#f1f3f7 !important; color:#8993a4 !important; cursor:not-allowed; }
[data-testid="stForm"] [data-testid="stTextInput"] input:focus { box-shadow:none !important; }
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button { background:var(--indigo) !important; border:1px solid var(--indigo) !important; color:#fff !important; font-weight:750; }
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button * { color:#fff !important; }
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover { background:#5f1b29 !important; border-color:#5f1b29 !important; }
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:disabled { background:#c9aeb4 !important; border-color:#c9aeb4 !important; color:#fff !important; }
[data-testid="stChatInput"] { background:#fff !important; border:1px solid #dfe3eb !important; border-radius:16px !important; box-shadow:0 10px 28px rgba(32,42,75,.10) !important; }
[data-testid="stChatInput"] textarea { background:#fff !important; color:#182338 !important; caret-color:#762333 !important; }
[data-testid="stChatInput"] textarea::placeholder { color:#7a8496 !important; opacity:1 !important; }
[data-testid="stChatInput"]:focus-within { border-color:#762333 !important; box-shadow:0 0 0 2px rgba(118,35,51,.14), 0 10px 28px rgba(32,42,75,.10) !important; }
[data-testid="stChatInput"] button { background:#762333 !important; color:#fff !important; border-radius:10px !important; }
[data-testid="stChatInput"] button:hover { background:#5f1b29 !important; }
div.stButton > button { border:1px solid var(--line); border-radius:12px; background:var(--surface); color:var(--ink); min-height:2.8rem; transition:all .15s ease; }
div.stButton > button:hover { border-color:#c799a3; background:var(--indigo-soft); color:var(--ink); }
[data-testid="baseButton-primary"], button[kind="primary"] { background:var(--indigo) !important; border-color:var(--indigo) !important; color:#fff !important; font-weight:700; }
[data-testid="baseButton-primary"] *, button[kind="primary"] * { color:#fff !important; }
[data-testid="baseButton-primary"]:hover, button[kind="primary"]:hover { background:#5f1b29 !important; border-color:#5f1b29 !important; }
[data-testid="stSidebar"] [data-testid="baseButton-primary"] p, [data-testid="stSidebar"] button[kind="primary"] p,
[data-testid="stSidebar"] [data-testid="baseButton-primary"] span, [data-testid="stSidebar"] button[kind="primary"] span { color:#fff !important; }
button:focus-visible, a:focus-visible, input:focus-visible { outline:3px solid #bd7d89 !important; outline-offset:2px; }
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"], .stBottomBlockContainer { background:transparent !important; }
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] > div { background:transparent !important; }
[data-testid="stChatMessage"] { border:0 !important; background:transparent !important; padding:.5rem 0; max-width:920px; margin:auto; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"], [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p, [data-testid="stChatMessage"] li { color:var(--ink) !important; line-height:1.68; }
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] { background:#f3e5e8 !important; color:#762333 !important; border:1px solid #e1c5cb; }
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] { width:2rem !important; height:2rem !important; min-width:2rem !important; display:grid !important; place-items:center !important; border-radius:50% !important; font-size:.61rem !important; font-weight:800 !important; letter-spacing:.02em; }
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] svg { display:none !important; }
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"]::before { content:"R-A"; color:#762333 !important; }
.message-label { font-size:.68rem; font-weight:750; letter-spacing:.04em; margin-bottom:.25rem; }
.user-label { color:#762333; align-self:flex-end; text-align:right; }
.assistant-label { color:#66717f; }
.user-bubble { display:inline-block; max-width:82%; background:#f5e9ec; border:1px solid #ead2d7; border-radius:14px 14px 4px 14px; color:#252832; padding:.62rem .82rem; line-height:1.55; }
[data-testid="stChatMessage"]:has(.user-bubble) [data-testid="stChatMessageContent"] { display:flex; flex-direction:column; align-items:flex-end; }
[data-testid="stChatMessage"]:has(.user-bubble) { justify-content:flex-end !important; }
[data-testid="stChatMessage"]:has(.user-bubble) [data-testid="stChatMessageAvatar"] { order:2; margin-left:.55rem !important; margin-right:0 !important; background:#762333 !important; border-color:#762333 !important; }
[data-testid="stChatMessage"]:has(.user-bubble) [data-testid="stChatMessageAvatar"]::before { content:"H-A"; color:#fff !important; }
[data-testid="stChatMessage"]:has(.user-bubble) [data-testid="stChatMessageContent"] { order:1; }
.about-view { max-width:850px; margin:3rem auto; }
.about-view h1 { font-size:2.25rem; margin:.5rem 0 .8rem; }
.about-view > p { color:#526078; line-height:1.7; max-width:720px; }
.about-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.8rem; margin-top:1.5rem; }
.about-grid > div { background:#fff; border:1px solid var(--line); border-radius:14px; padding:1rem; }
.about-grid strong { color:var(--indigo); font-size:.82rem; }
.about-grid p { color:#526078; font-size:.82rem; line-height:1.55; }
.finding-state { display:flex; align-items:center; gap:.42rem; color:#66717f; font-size:.82rem; padding:.15rem 0 .5rem; }
.finding-dot { width:7px; height:7px; border-radius:50%; background:#762333; animation:finding-pulse 1.1s ease-in-out infinite; }
.finding-ellipsis { display:inline-block; width:1.2rem; overflow:hidden; animation:finding-dots 1.2s steps(4,end) infinite; }
@keyframes finding-pulse { 0%,100% { opacity:.35; transform:scale(.85); } 50% { opacity:1; transform:scale(1.1); } }
@keyframes finding-dots { 0% { width:0; } 100% { width:1.2rem; } }
.latest-hint { color:#8a929e; font-size:.68rem; text-align:center; margin:.35rem auto .7rem; }
.result-heading { color:#526078; font-size:.74rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; margin:1rem 0 .45rem; }
.project-result { background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:.85rem 1rem; margin:.55rem 0; box-shadow:0 5px 16px rgba(32,42,75,.045); }
.project-top { display:flex; align-items:flex-start; gap:.65rem; }
.project-rank { color:var(--indigo); font-size:.77rem; font-weight:800; padding-top:.15rem; }
.project-title { color:var(--ink); font-size:1rem; font-weight:750; line-height:1.35; }
.project-id { color:#78849a; font-size:.7rem; margin-top:.14rem; }
.project-pills { display:flex; flex-wrap:wrap; gap:.35rem; margin:.65rem 0 .55rem 1.35rem; }
.project-pill { border:1px solid #deddf7; background:#f7f6ff; color:#4b4890; border-radius:999px; font-size:.68rem; padding:.25rem .48rem; }
.project-pill b { font-weight:750; }
.project-details { display:flex; flex-wrap:wrap; gap:.9rem; margin-left:1.35rem; }
.project-details div { min-width:130px; }
.project-details span { display:block; color:#758198; font-size:.66rem; text-transform:uppercase; letter-spacing:.04em; }
.project-details strong { display:block; color:#2b3951; font-size:.77rem; font-weight:650; margin-top:.1rem; }
.project-footer { border-top:1px solid #f0f1f5; margin:.7rem 0 0 1.35rem; padding-top:.55rem; }
.project-card-lower { display:grid; grid-template-columns:minmax(0,1fr) minmax(190px,.8fr); gap:1rem; align-items:start; }
.supervisor-mini { border-left:1px solid #eee8ea; padding-left:.9rem; display:grid; gap:.16rem; }
.supervisor-heading { color:#762333; font-size:.62rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
.supervisor-mini strong { color:#26344c; font-size:.78rem; }
.supervisor-mini span { color:#68758a; font-size:.68rem; line-height:1.35; }
.supervisor-fields { display:block; }
.supervisor-link { margin-top:.3rem; }
.project-link, .source-link { color:#5149c7 !important; font-size:.76rem; font-weight:700; text-decoration:none; }
.source-entry { color:var(--ink); margin:.45rem 0 .1rem; font-size:.83rem; }
.source-type { display:block; color:#68758a; font-size:.68rem; margin-top:.12rem; text-transform:capitalize; }
.structured-heading { color:#526078; font-size:.9rem; font-weight:800; margin:.7rem 0 .35rem; }
.structured-row { border-bottom:1px solid #f0f1f5; padding:.42rem .1rem; display:flex; flex-direction:column; gap:.12rem; color:#26344c; font-size:.82rem; }
.structured-row span { color:#68758a; font-size:.72rem; }
.graph-badge, .method-badge { display:inline-block; color:#5a54b8; background:var(--indigo-soft); border:1px solid #dfddff; border-radius:999px; font-size:.69rem; font-weight:750; padding:.27rem .55rem; margin-top:.65rem; }
.method-badge { color:#67748a; background:#f3f5fa; border-color:#e4e7ef; margin-left:.35rem; }
.evidence-warning, .no-match { background:#fffdf7; border:1px solid #ece5c8; color:#4c4a41; border-radius:13px; padding:.8rem .95rem; margin:.45rem 0 .8rem; line-height:1.55; }
.no-match { background:#fafbfe; border-color:var(--line); color:#4b5870; }
.evidence-warning strong, .no-match strong { color:#2b3548; }
@media (max-width:700px) { .workspace-top { margin-bottom:1.5rem; } .workspace-tagline { margin-left:0; } .assignment-label { display:none; } .empty-state { margin-top:2rem; } .project-details { gap:.55rem; } .project-card-lower { grid-template-columns:1fr; } .supervisor-mini { border-left:0; border-top:1px solid #eee8ea; padding: .7rem 0 0; } .about-grid { grid-template-columns:1fr; } }
</style>
"""
