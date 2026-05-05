"""
ShiftIQ — Workforce Shift Scheduling Optimizer
Dantzig (1954) Set-Covering MILP · ISO 22400-2 OEE · Lean Takt-Time

Run:  pip install streamlit numpy scipy pandas
      streamlit run shiftiq.py
"""

import math, json, http.client, ssl
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import milp, LinearConstraint, Bounds
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  — every number cited
# ─────────────────────────────────────────────────────────────────────────────
DAYS          = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

# Work-content (man-min/unit) per station.  At takt 0.5625 min/unit these yield
# 6–15 operators/station/day-shift — calibrated to battery module assembly norms.
STATIONS = {
    "Cell Stacking":    {"wc": 6.0, "desc": "Align, stack, press, inspect"},
    "Electrolyte Fill": {"wc": 4.0, "desc": "Fill, vacuum, seal, weigh"},
    "Laser Welding":    {"wc": 5.5, "desc": "Fixture, weld, inspect, log"},
    "Module Assembly":  {"wc": 6.0, "desc": "Bracket, connect, torque, test"},
    "Pack Integration": {"wc": 4.5, "desc": "Insert, BMS, seal, leak-test"},
    "QC & Testing":     {"wc": 3.0, "desc": "Electrical, thermal, cosmetic"},
}

# Shift demand multipliers — Ernst et al. (2004) EJOR 153(1):3–27
SHIFT_MULT    = {"Day": 1.00, "Evening": 0.75, "Night": 0.55}

# Weekend ratio 0.85 — Circadian Technologies Shiftwork Practices survey
WEEKEND_RATIO = 0.85

# Burden rate 1.30 — SHRM (2023) Benefits Survey, US manufacturing 28–32%
BURDEN        = 1.30
# Agency markup 1.15 — standard contractor bill-rate uplift
AGENCY        = 1.15

WTYPES = {
    "Full-Time":  {"hourly": 22.0, "hrs": 8, "agency": False},
    "Part-Time":  {"hourly": 18.0, "hrs": 8, "agency": False},
    "Contractor": {"hourly": 35.0, "hrs": 8, "agency": True},
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def shift_cost(wtype: str) -> float:
    """Loaded $/shift = base × hrs × 1.30 burden [× 1.15 agency].
    Ref: NetSuite Mfg Cost Guide; SHRM 2023."""
    w = WTYPES[wtype]
    c = w["hourly"] * w["hrs"] * BURDEN
    return round(c * AGENCY if w["agency"] else c, 2)

def takt(net_min: float, demand: int) -> float:
    """Takt = Net_Available / Daily_Demand.  Ref: Womack & Jones (1996)."""
    return net_min / demand

def ops_needed(wc: float, tk: float) -> int:
    """Operators = ⌈Work_Content / Takt⌉.  Ref: Womack & Jones (1996)."""
    return math.ceil(wc / tk)

def coverage_matrix() -> np.ndarray:
    """Dantzig (1954) 5-on-2-off coverage matrix.
    A[j,i]=1 iff pattern-i workers are on duty on day j.
    Ref: Dantzig OR 2(3):339-341; Van den Bergh et al. (2013) EJOR 226(3)."""
    A = np.zeros((7, 7))
    for i in range(7):
        for k in range(5):
            A[(i + k) % 7, i] = 1.0
    return A

def solve(demand_vec: np.ndarray, A: np.ndarray) -> dict:
    """Dantzig set-covering MILP per station-shift.
    min Σx[i]  s.t.  A@x >= demand_vec,  x ∈ Z+
    Post-solve: explicit validation all(A@x >= demand_vec)."""
    ub  = float(int(demand_vec.max()) * 3)
    res = milp(
        c           = np.ones(7),
        constraints = LinearConstraint(A, demand_vec.astype(float), np.full(7, np.inf)),
        integrality = np.ones(7),
        bounds      = Bounds(np.zeros(7), np.full(7, ub)),
        options     = {"disp": False, "time_limit": 5.0},
    )
    if res.status != 0:
        return {"ok": False}
    x   = np.round(res.x).astype(int)
    cov = (A @ x).astype(int)
    return {
        "ok": True, "valid": bool(np.all(cov >= demand_vec)),
        "hc": int(x.sum()), "patterns": x,
        "coverage": cov, "demand": demand_vec.astype(int),
        "surplus": (cov - demand_vec).astype(int),
        "wk_cost": int(x.sum()) * shift_cost("Full-Time"),
    }

def build_demand(daily_demand, shift_min, break_min, ramp_pct):
    net = shift_min - break_min
    adj = int(math.ceil(daily_demand * (1 + ramp_pct / 100)))
    tk  = takt(net, adj)
    dm  = {}
    for s, d in STATIONS.items():
        d0 = ops_needed(d["wc"], tk)
        dm[s] = {}
        for sh, m in SHIFT_MULT.items():
            wd = math.ceil(d0 * m)
            we = math.ceil(wd * WEEKEND_RATIO)
            dm[s][sh] = {"wd": wd, "we": we, "vec": np.array([wd,wd,wd,wd,wd,we,we])}
    return dm, tk, adj

def calc_oee(a, p, q):
    """OEE = Availability × Performance × Quality.  Ref: ISO 22400-2:2014."""
    return a * p * q

# ─────────────────────────────────────────────────────────────────────────────
# GROQ AI DEBRIEF
# ─────────────────────────────────────────────────────────────────────────────

def call_groq(prompt: str, api_key: str) -> str:
    payload = json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3, "max_tokens": 900,
    }).encode("utf-8")
    ctx  = ssl.create_default_context()
    conn = http.client.HTTPSConnection("api.groq.com", context=ctx, timeout=25)
    conn.request("POST", "/openai/v1/chat/completions", body=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Length": str(len(payload)),
    })
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    conn.close()
    if resp.status != 200:
        try:
            msg = json.loads(body).get("error", {}).get("message", body[:300])
        except Exception:
            msg = body[:300]
        raise RuntimeError(f"Groq {resp.status}: {msg}")
    return json.loads(body)["choices"][0]["message"]["content"].strip()

def build_prompt(sols, p, tk_, adj, ft_cost, mix_cost, oev, total_hc, all_ok):
    tightest = min(((s,sh) for s in STATIONS for sh in SHIFT_MULT if sols[s][sh]["ok"]),
                   key=lambda x: int(sols[x[0]][x[1]]["surplus"].sum()))
    loosest  = max(((s,sh) for s in STATIONS for sh in SHIFT_MULT if sols[s][sh]["ok"]),
                   key=lambda x: int(sols[x[0]][x[1]]["surplus"].sum()))
    costiest = max(((s,sh) for s in STATIONS for sh in SHIFT_MULT if sols[s][sh]["ok"]),
                   key=lambda x: sols[x[0]][x[1]]["wk_cost"])
    t_sur  = int(sols[tightest[0]][tightest[1]]["surplus"].sum())
    l_sur  = int(sols[loosest[0]][loosest[1]]["surplus"].sum())
    c_val  = sols[costiest[0]][costiest[1]]["wk_cost"]
    c_hc   = sols[costiest[0]][costiest[1]]["hc"]
    night  = sum(sols[s]["Night"]["hc"] for s in STATIONS if sols[s]["Night"]["ok"])
    day    = sum(sols[s]["Day"]["hc"]   for s in STATIONS if sols[s]["Day"]["ok"])
    return f"""You are an industrial engineering analyst writing a structured debrief for a workforce planner after running a shift scheduling optimisation.

CONTEXT: Dantzig (1954) set-covering MILP, HiGHS solver, 6 stations × 3 shifts, all feasible: {all_ok}

SOLVED NUMBERS:
- Takt: {tk_:.4f} min/unit (net {p['shift_dur']*60-p['brk']} min ÷ {adj} units/day)
- Total headcount: {total_hc} (Day: {day}, Night: {night})
- Weekly cost all-FT: ${ft_cost:,.0f} | Mixed ({p['ft_pct']}%FT/{p['pt_pct']}%PT/{p['ct_pct']}%CT): ${mix_cost:,.0f}
- OEE: {oev*100:.1f}% (A={p['avail']*100:.0f}%×P={p['perf']*100:.0f}%×Q={p['qual_']*100:.0f}%)
- Ramp: {p['ramp_pct']}%

COVERAGE:
- Tightest: {tightest[0]} / {tightest[1]} (total weekly surplus={t_sur} worker-days)
- Loosest: {loosest[0]} / {loosest[1]} (total weekly surplus={l_sur} worker-days)
- Costliest: {costiest[0]} / {costiest[1]} ({c_hc} workers, ${c_val:,.0f}/week)
- Night = {night/total_hc*100:.0f}% of total headcount

TASK — respond in exactly this format, under 280 words, use only the numbers above:

**Schedule Summary**
2–3 sentences: headcount, cost, feasibility.

**Coverage Flags**
- Tightest slot: what surplus={t_sur} means for absenteeism risk
- Loosest slot: what surplus={l_sur} means for rebalancing

**Cost Concentration**
1–2 sentences on where the weekly budget is concentrated and why.

**Night Shift Note**
1 sentence on whether {night} night workers vs {day} day workers looks proportionate.

**One Action Item**
Single most actionable recommendation based only on these numbers."""

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShiftIQ",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0D1117;
    color: #E2E8F0;
}
.block-container { padding: 1.6rem 2rem 3rem; max-width: 1400px; }
.stApp { background: #0D1117; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0A0E14;
    border-right: 1px solid #1E2530;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption { color: #94A3B8 !important; }
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stNumberInput > label,
[data-testid="stSidebar"] .stSelectbox > label { color: #CBD5E1 !important; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'DM Sans', sans-serif; font-weight: 600; }
.mono { font-family: 'IBM Plex Mono', monospace; }

/* ── Amber accent ── */
:root {
    --amber: #F59E0B;
    --amber-dim: #B45309;
    --amber-bg: rgba(245,158,11,0.08);
    --slate: #0D1117;
    --slate-1: #141B24;
    --slate-2: #1A2332;
    --slate-3: #1E2D3D;
    --border: #1E2D3D;
    --text: #E2E8F0;
    --muted: #64748B;
    --soft: #94A3B8;
}

/* ── Section header ── */
.shdr {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .70rem; font-weight: 600;
    color: var(--amber); letter-spacing: .12em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px; margin: 1.6rem 0 .8rem;
}

/* ── KPI cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: 10px; margin: .8rem 0; }
.kpi {
    background: var(--slate-1);
    border: 1px solid var(--border);
    border-top: 2px solid var(--amber);
    padding: 14px 16px; border-radius: 4px;
}
.kpi-l { font-size:.65rem; color:var(--muted); text-transform:uppercase;
          letter-spacing:.09em; font-family:'IBM Plex Mono',monospace; margin-bottom:6px; }
.kpi-v { font-family:'IBM Plex Mono',monospace; font-size:1.45rem;
          font-weight:600; color:var(--amber); line-height:1; }
.kpi-s { font-size:.70rem; color:var(--soft); margin-top:5px; }

/* ── OEE row ── */
.oee-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:.6rem 0; }
.oee-card {
    background:var(--slate-1); border:1px solid var(--border);
    padding:12px 14px; border-radius:4px;
}
.oee-l { font-size:.65rem; color:var(--muted); font-family:'IBM Plex Mono',monospace;
          text-transform:uppercase; letter-spacing:.08em; }
.oee-v { font-family:'IBM Plex Mono',monospace; font-size:1.3rem; font-weight:600;
          color:#34D399; margin:4px 0; }
.oee-bar { height:4px; background:var(--slate-3); border-radius:2px; overflow:hidden; }
.oee-fill { height:4px; background:#34D399; border-radius:2px; }

/* ── Status banner ── */
.status-ok {
    background: rgba(52,211,153,.08); border:1px solid rgba(52,211,153,.25);
    border-left:3px solid #34D399; padding:10px 16px; border-radius:4px;
    font-family:'IBM Plex Mono',monospace; font-size:.80rem; color:#6EE7B7;
    margin:.8rem 0;
}
.status-err {
    background: rgba(239,68,68,.08); border:1px solid rgba(239,68,68,.25);
    border-left:3px solid #EF4444; padding:10px 16px; border-radius:4px;
    font-family:'IBM Plex Mono',monospace; font-size:.80rem; color:#FCA5A5;
    margin:.8rem 0;
}

/* ── Reference chips ── */
.ref {
    display:inline-block; background:rgba(30,45,61,.6);
    border:1px solid var(--border); padding:3px 9px;
    border-radius:3px; font-size:.68rem; color:var(--soft);
    font-family:'IBM Plex Mono',monospace; margin:2px 3px;
}

/* ── Dataframe overrides ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; }
.stDataFrame th {
    background: var(--slate-2) !important;
    color: var(--amber) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .70rem !important; letter-spacing:.06em;
}
.stDataFrame td { color: var(--text) !important; font-size:.80rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background:var(--slate-1); border-bottom:1px solid var(--border); gap:0; }
.stTabs [data-baseweb="tab"] {
    background:transparent; color:var(--muted);
    font-family:'IBM Plex Mono',monospace; font-size:.72rem;
    padding:10px 18px; border-bottom:2px solid transparent;
    letter-spacing:.05em;
}
.stTabs [aria-selected="true"] {
    color: var(--amber) !important;
    border-bottom:2px solid var(--amber) !important;
    background:transparent !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--slate-1) !important;
    color: var(--soft) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .75rem !important; letter-spacing:.05em;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}
.streamlit-expanderContent {
    background: var(--slate-1) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--amber) !important;
    color: #0D1117 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important; font-size: .78rem !important;
    letter-spacing: .06em !important;
    border: none !important; border-radius: 3px !important;
}
.stButton > button:hover { background: #FBBF24 !important; }

/* ── Number input / selectbox / slider labels ── */
.stSlider > div > div > div > div { background: var(--amber) !important; }
.stProgress > div > div { background: var(--amber) !important; }

/* ── AI debrief box ── */
.debrief {
    background: var(--slate-1);
    border: 1px solid var(--border);
    border-left: 3px solid var(--amber);
    padding: 20px 24px; border-radius: 4px;
    font-size: .88rem; line-height: 1.7; color: var(--text);
    margin: .8rem 0;
}
.debrief strong { color: var(--amber); }

/* ── Formulation box ── */
.fml {
    background: #080C11;
    border: 1px solid var(--border);
    padding: 16px 20px; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .78rem; color: #7DD3FC; line-height: 1.8;
}
.fml b { color: var(--amber); }

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
div[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:12px 0 8px;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;font-weight:600;color:#F59E0B;letter-spacing:.04em;">⚙ ShiftIQ</div>
  <div style="font-size:.68rem;color:#475569;font-family:'IBM Plex Mono',monospace;letter-spacing:.06em;margin-top:2px;">WORKFORCE SCHEDULING ENGINE</div>
</div>
<div style="border-bottom:1px solid #1E2D3D;margin-bottom:16px;"></div>
""", unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.65rem;color:#F59E0B;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">LINE PARAMETERS</div>', unsafe_allow_html=True)
    daily_dem = st.number_input("Daily unit target", 100, 2000, 800, 50)
    ramp_pct  = st.slider("Production ramp (%)", 0, 80, 0, 5,
                           help="NPI ramp-up: scales station demand proportionally")
    shift_dur = st.selectbox("Shift length (hrs)", [8, 10, 12], 0)
    brk       = st.slider("Breaks / shift (min)", 20, 60, 30, 5)

    st.markdown('<div style="border-bottom:1px solid #1E2D3D;margin:12px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.65rem;color:#F59E0B;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">WORKER MIX</div>', unsafe_allow_html=True)
    ft_pct = st.slider("Full-Time %",  0, 100, 70, 5)
    pt_pct = st.slider("Part-Time %",  0, 100 - ft_pct, 20, 5)
    ct_pct = 100 - ft_pct - pt_pct
    st.markdown(f'<div style="font-size:.68rem;color:#475569;font-family:\'IBM Plex Mono\',monospace;">Contractor: {ct_pct}%</div>', unsafe_allow_html=True)

    st.markdown('<div style="border-bottom:1px solid #1E2D3D;margin:12px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.65rem;color:#F59E0B;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">OEE — ISO 22400-2</div>', unsafe_allow_html=True)
    avail = st.slider("Availability (%)", 50, 100, 85, 1) / 100
    perf  = st.slider("Performance (%)",  50, 100, 92, 1) / 100
    qual_ = st.slider("Quality (%)",      80, 100, 97, 1) / 100

    st.markdown('<div style="border-bottom:1px solid #1E2D3D;margin:12px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.65rem;color:#F59E0B;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">HOURLY WAGES (USD)</div>', unsafe_allow_html=True)
    w_ft = st.number_input("Full-Time $/hr",  value=22.0, step=1.0)
    w_pt = st.number_input("Part-Time $/hr",  value=18.0, step=1.0)
    w_ct = st.number_input("Contractor $/hr", value=35.0, step=1.0)
    WTYPES["Full-Time"]["hourly"]  = w_ft
    WTYPES["Part-Time"]["hourly"]  = w_pt
    WTYPES["Contractor"]["hourly"] = w_ct

    st.markdown('<div style="border-bottom:1px solid #1E2D3D;margin:12px 0;"></div>', unsafe_allow_html=True)
    run = st.button("RUN OPTIMIZER", use_container_width=True, type="primary")
    st.markdown('<div style="font-size:.62rem;color:#334155;font-family:\'IBM Plex Mono\',monospace;text-align:center;margin-top:8px;">HiGHS · 18 MILP sub-problems</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("""
<div style="width:52px;height:52px;background:#F59E0B;border-radius:4px;
     display:flex;align-items:center;justify-content:center;
     font-size:1.6rem;margin-top:4px;">⚙</div>
""", unsafe_allow_html=True)
with col_title:
    st.markdown("""
<div style="padding-top:2px;">
  <div style="font-family:'DM Sans',sans-serif;font-size:1.75rem;font-weight:600;
       color:#F1F5F9;letter-spacing:-.01em;line-height:1.1;">ShiftIQ</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:#64748B;
       letter-spacing:.08em;text-transform:uppercase;margin-top:2px;">
    Workforce Shift Scheduling Optimizer &nbsp;·&nbsp; Battery Assembly &nbsp;·&nbsp;
    Dantzig (1954) MILP &nbsp;·&nbsp; ISO 22400-2
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="border-bottom:1px solid #1E2D3D;margin:1rem 0 .5rem;"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ALWAYS-VISIBLE PANELS
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("MODEL & METHODOLOGY", expanded=False):
    mx1, mx2, mx3 = st.columns(3)
    with mx1:
        st.markdown("**MILP Formulation**")
        st.markdown("""<div class="fml">
<b>// Dantzig (1954) Set-Covering</b><br>
vars:   x[i] ∈ ℤ⁺  (i = Mon..Sun)<br>
        x[i] = workers on 5-on-2-off<br>
        pattern starting day i<br><br>
<b>// Coverage matrix</b><br>
A[j,i] = 1 if pattern-i on duty day j<br><br>
<b>// Objective</b><br>
min  Σᵢ x[i]<br><br>
<b>// Demand coverage</b><br>
A @ x >= d[j]  ∀ day j<br><br>
<b>// Integrality</b><br>
x[i] ≥ 0,  integer<br><br>
18 sub-problems · 7 vars each<br>
Solver: HiGHS (scipy.optimize.milp)<br>
TU matrix → LP relaxation optimal<br>
(Veinott & Wagner 1962)
</div>""", unsafe_allow_html=True)

    with mx2:
        st.markdown("**Demand Derivation**")
        st.markdown("""<div class="fml">
<b>// Takt time (Womack & Jones 1996)</b><br>
T = Net_Available / Daily_Demand<br>
Net = Shift_min − Break_min<br><br>
<b>// Operators per station</b><br>
Ops = ⌈ Work_Content / T ⌉<br><br>
<b>// Shift ratios (Ernst 2004 EJOR)</b><br>
Evening = ⌈ Ops_day × 0.75 ⌉<br>
Night   = ⌈ Ops_day × 0.55 ⌉<br><br>
<b>// Weekend (Circadian Tech.)</b><br>
Weekend = ⌈ Weekday × 0.85 ⌉<br><br>
<b>// OEE (ISO 22400-2:2014)</b><br>
OEE = Availability<br>
    × Performance<br>
    × Quality
</div>""", unsafe_allow_html=True)

    with mx3:
        st.markdown("**Labor Cost Model**")
        st.markdown("""<div class="fml">
<b>// Loaded shift cost</b><br>
<b>// (NetSuite 2024; SHRM 2023)</b><br><br>
FT = base × hrs × 1.30<br>
PT = base × hrs × 1.30<br>
CT = base × hrs × 1.30 × 1.15<br><br>
Burden  1.30 = 30% overhead<br>
Agency  1.15 = 15% bill-rate markup<br><br>
<b>// References</b><br>
[1] Dantzig OR 2(3) 1954<br>
[2] Van den Bergh EJOR 2013<br>
[3] Womack & Jones 1996<br>
[4] Ernst et al. EJOR 2004<br>
[5] ISO 22400-2:2014<br>
[6] Nakajima TPM 1988<br>
[7] SHRM Benefits Survey 2023<br>
[8] Veinott & Wagner 1962
</div>""", unsafe_allow_html=True)

with st.expander("INPUT DATA PREVIEW", expanded=False):
    ip1, ip2, ip3 = st.tabs(["Station Work Content", "Shift Demand Factors", "Loaded Cost Table"])
    with ip1:
        wc_df = pd.DataFrame([
            {"Station": k, "Work Content (man-min/unit)": v["wc"], "Description": v["desc"]}
            for k, v in STATIONS.items()
        ])
        st.dataframe(wc_df, use_container_width=True, hide_index=True)
        st.markdown('<span class="ref">Womack & Jones (1996) Lean Thinking</span><span class="ref">Ops = ⌈WC/Takt⌉</span>', unsafe_allow_html=True)
    with ip2:
        mdf = pd.DataFrame(
            [{"Shift": sh, "Multiplier": m, "vs Day Shift": f"{int(m*100)}%"} for sh, m in SHIFT_MULT.items()] +
            [{"Shift": "Weekend", "Multiplier": WEEKEND_RATIO, "vs Day Shift": "85% of weekday"}]
        )
        st.dataframe(mdf, use_container_width=True, hide_index=True)
        st.markdown('<span class="ref">Ernst et al. EJOR 153(1) 2004</span><span class="ref">Circadian Technologies Shiftwork Practices</span>', unsafe_allow_html=True)
    with ip3:
        cdf = pd.DataFrame([{
            "Type": wt, "$/hr": WTYPES[wt]["hourly"],
            "Base/Shift ($)": round(WTYPES[wt]["hourly"]*WTYPES[wt]["hrs"], 2),
            "×1.30 Burden":   round(WTYPES[wt]["hourly"]*WTYPES[wt]["hrs"]*BURDEN, 2),
            "Loaded/Shift ($)": shift_cost(wt),
        } for wt in WTYPES])
        st.dataframe(cdf, use_container_width=True, hide_index=True)
        st.markdown('<span class="ref">SHRM Benefits Survey 2023</span><span class="ref">NetSuite Manufacturing Cost Guide 2024</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SOLVER
# ─────────────────────────────────────────────────────────────────────────────
if run or "res" in st.session_state:

    if run:
        sm = shift_dur * 60
        dm_, tk_, adj_ = build_demand(daily_dem, sm, brk, ramp_pct)
        A_  = coverage_matrix()
        sols_ = {s: {sh: solve(dm_[s][sh]["vec"], A_) for sh in SHIFT_MULT} for s in STATIONS}
        st.session_state.update({
            "res": sols_, "tk": tk_, "adj": adj_, "dm": dm_, "A": A_,
            "oee_val": calc_oee(avail, perf, qual_),
            "p": dict(daily_dem=daily_dem, ramp_pct=ramp_pct, shift_dur=shift_dur,
                      brk=brk, ft_pct=ft_pct, pt_pct=pt_pct, ct_pct=ct_pct,
                      avail=avail, perf=perf, qual_=qual_,
                      w_ft=w_ft, w_pt=w_pt, w_ct=w_ct),
        })

    sols = st.session_state["res"]
    tk_  = st.session_state["tk"]
    adj  = st.session_state["adj"]
    dm_d = st.session_state["dm"]
    A_   = st.session_state["A"]
    oev  = st.session_state["oee_val"]
    p    = st.session_state["p"]

    total_hc  = sum(sols[s][sh]["hc"]      for s in STATIONS for sh in SHIFT_MULT if sols[s][sh]["ok"])
    ft_wkcost = sum(sols[s][sh]["wk_cost"] for s in STATIONS for sh in SHIFT_MULT if sols[s][sh]["ok"])
    mix_cost  = total_hc * (
        (p["ft_pct"]/100)*shift_cost("Full-Time") +
        (p["pt_pct"]/100)*shift_cost("Part-Time") +
        (p["ct_pct"]/100)*shift_cost("Contractor")
    )
    all_ok = all(sols[s][sh]["ok"] and sols[s][sh]["valid"] for s in STATIONS for sh in SHIFT_MULT)

    # Status
    if all_ok:
        st.markdown(f'<div class="status-ok">✓ OPTIMAL — all 18 sub-problems solved & constraint-validated &nbsp;·&nbsp; min headcount: {total_hc} &nbsp;·&nbsp; {datetime.now().strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
    else:
        bad = [f"{s}/{sh}" for s in STATIONS for sh in SHIFT_MULT if not (sols[s][sh].get("ok") and sols[s][sh].get("valid"))]
        st.markdown(f'<div class="status-err">✗ INFEASIBLE — {bad}</div>', unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    st.markdown('<div class="shdr">OUTPUTS</div>', unsafe_allow_html=True)
    kpi_html = '<div class="kpi-grid">'
    for lbl, val, sub in [
        ("Takt Time",            f"{tk_:.4f}",           f"min/unit · net {p['shift_dur']*60-p['brk']}min ÷ {adj}u"),
        ("Min Headcount",        str(total_hc),           "all stations × all shifts"),
        ("Weekly Cost — All FT", f"${ft_wkcost:,.0f}",   f"$22/hr × 8hr × 1.30 burden"),
        ("Weekly Cost — Mix",    f"${mix_cost:,.0f}",    f"FT{p['ft_pct']}% PT{p['pt_pct']}% CT{p['ct_pct']}%"),
        ("OEE",                  f"{oev*100:.1f}%",       f"A{p['avail']*100:.0f}%×P{p['perf']*100:.0f}%×Q{p['qual_']*100:.0f}%"),
    ]:
        kpi_html += f'<div class="kpi"><div class="kpi-l">{lbl}</div><div class="kpi-v">{val}</div><div class="kpi-s">{sub}</div></div>'
    kpi_html += '</div>'
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ── OEE row ───────────────────────────────────────────────────────────────
    st.markdown('<div class="shdr">OEE — ISO 22400-2:2014</div>', unsafe_allow_html=True)
    oee_html = '<div class="oee-grid">'
    for lbl, v in [("Availability", p["avail"]), ("Performance", p["perf"]), ("Quality", p["qual_"]), ("OEE = A×P×Q", oev)]:
        pct = int(v * 100)
        oee_html += f'''<div class="oee-card">
            <div class="oee-l">{lbl}</div>
            <div class="oee-v">{pct}%</div>
            <div class="oee-bar"><div class="oee-fill" style="width:{pct}%"></div></div>
        </div>'''
    oee_html += '</div>'
    st.markdown(oee_html, unsafe_allow_html=True)
    st.markdown('<span class="ref">ISO 22400-2:2014</span><span class="ref">Nakajima TPM 1988</span><span class="ref">World-class ≈ 85%</span>', unsafe_allow_html=True)

    # ── Results tabs ──────────────────────────────────────────────────────────
    st.markdown('<div class="shdr">RESULTS</div>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "SCHEDULE", "TAKT DERIVATION", "COST", "PATTERN DETAIL", "RAMP SCENARIOS", "CHARTS"
    ])

    # Tab 1 — Schedule
    with t1:
        rows = []
        for s in STATIONS:
            for sh in SHIFT_MULT:
                sol = sols[s][sh]
                rows.append({
                    "Station":         s,
                    "Shift":           sh,
                    "Weekday Demand":  int(dm_d[s][sh]["wd"]) if sol["ok"] else "—",
                    "Weekend Demand":  int(dm_d[s][sh]["we"]) if sol["ok"] else "—",
                    "Headcount":       sol["hc"]              if sol["ok"] else "FAIL",
                    "FT Weekly Cost":  f"${sol['wk_cost']:,.0f}" if sol["ok"] else "—",
                    "Validated":       ("YES" if sol["valid"] else "NO") if sol["ok"] else "FAIL",
                    "Mon Coverage":    int(sol["coverage"][0]) if sol["ok"] else "—",
                    "Sat Coverage":    int(sol["coverage"][5]) if sol["ok"] else "—",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=420, hide_index=True)
        st.markdown('<span class="ref">Dantzig OR 2(3) 1954</span><span class="ref">Post-solve: all(A@x >= demand_vec)</span><span class="ref">HiGHS solver</span>', unsafe_allow_html=True)

    # Tab 2 — Takt derivation
    with t2:
        st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:.78rem;
            color:#94A3B8;background:#141B24;border:1px solid #1E2D3D;
            padding:10px 14px;border-radius:4px;margin-bottom:12px;">
            Net = {p['shift_dur']*60}min − {p['brk']}min = {p['shift_dur']*60-p['brk']}min &nbsp;·&nbsp;
            Demand = {adj} units/day (+{p['ramp_pct']}% ramp) &nbsp;·&nbsp;
            <span style="color:#F59E0B;">Takt = {tk_:.4f} min/unit</span>
        </div>""", unsafe_allow_html=True)
        trows = []
        for s, d in STATIONS.items():
            d0 = ops_needed(d["wc"], tk_)
            for sh, m in SHIFT_MULT.items():
                wd = math.ceil(d0 * m)
                we = math.ceil(wd * WEEKEND_RATIO)
                trows.append({
                    "Station": s, "WC (man-min/unit)": d["wc"],
                    "Takt (min/unit)": round(tk_, 4),
                    "Day Ops": d0, "Shift": sh,
                    "Mult": m, "Weekday Ops": wd, "Weekend Ops": we,
                    "Formula": f"⌈{d['wc']}/{tk_:.4f}⌉×{m}={wd}",
                })
        st.dataframe(pd.DataFrame(trows), use_container_width=True, height=420, hide_index=True)
        st.markdown('<span class="ref">Womack & Jones (1996)</span><span class="ref">Ernst et al. EJOR 2004</span><span class="ref">Circadian Technologies</span>', unsafe_allow_html=True)

    # Tab 3 — Cost
    with t3:
        crows = []
        for wt, wk in [("Full-Time","ft_pct"),("Part-Time","pt_pct"),("Contractor","ct_pct")]:
            w    = WTYPES[wt]
            base = w["hourly"] * w["hrs"]
            burd = base * BURDEN
            fin  = burd * AGENCY if w["agency"] else burd
            heads= int(total_hc * p[wk] / 100)
            crows.append({
                "Type": wt, "$/hr": w["hourly"], "Hrs": w["hrs"],
                "Base/Shift": round(base,2), "×1.30": round(burd,2),
                "×1.15 (CT)": round(fin,2), "Mix%": p[wk],
                "Est. Heads": heads, "Weekly ($)": f"{heads*fin:,.0f}",
            })
        st.dataframe(pd.DataFrame(crows), use_container_width=True, hide_index=True)
        st.markdown('<span class="ref">SHRM 2023</span><span class="ref">NetSuite 2024</span>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:16px;font-family:\'IBM Plex Mono\',monospace;font-size:.68rem;color:#F59E0B;letter-spacing:.1em;text-transform:uppercase;border-bottom:1px solid #1E2D3D;padding-bottom:6px;margin-bottom:8px;">MIX SENSITIVITY</div>', unsafe_allow_html=True)
        srows = []
        for ft in [60,70,80,90,100]:
            for ct in [0,10,20]:
                pt = 100-ft-ct
                if pt < 0: continue
                c = total_hc*((ft/100)*shift_cost("Full-Time")+(pt/100)*shift_cost("Part-Time")+(ct/100)*shift_cost("Contractor"))
                srows.append({"FT%":ft,"PT%":pt,"CT%":ct,"Weekly Cost ($)":f"{c:,.0f}","vs All-FT":f"{c-ft_wkcost:+,.0f}"})
        st.dataframe(pd.DataFrame(srows), use_container_width=True, hide_index=True, height=260)

    # Tab 4 — Pattern detail
    with t4:
        cc1, cc2 = st.columns(2)
        with cc1:
            sel_s  = st.selectbox("Station", list(STATIONS.keys()), key="ps")
        with cc2:
            sel_sh = st.selectbox("Shift",   list(SHIFT_MULT.keys()), key="psh")
        ex = sols[sel_s][sel_sh]
        if ex["ok"]:
            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown('<div style="font-size:.70rem;color:#64748B;font-family:\'IBM Plex Mono\',monospace;margin-bottom:6px;">WORKERS PER START-DAY PATTERN</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame({
                    "Start Day":    DAYS,
                    "Workers x[i]": ex["patterns"],
                    "Works":        [f"{DAYS[i]}→{DAYS[(i+4)%7]}" for i in range(7)],
                    "Days Off":     [f"{DAYS[(i+5)%7]},{DAYS[(i+6)%7]}" for i in range(7)],
                }), use_container_width=True, hide_index=True)
            with pc2:
                st.markdown('<div style="font-size:.70rem;color:#64748B;font-family:\'IBM Plex Mono\',monospace;margin-bottom:6px;">DAILY COVERAGE VALIDATION</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame({
                    "Day":      DAYS,
                    "Required": ex["demand"],
                    "Coverage": ex["coverage"],
                    "Surplus":  ex["surplus"],
                    "Pass":     ["✓" if c >= d else "✗" for c,d in zip(ex["coverage"],ex["demand"])],
                }), use_container_width=True, hide_index=True)
            st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.75rem;color:#64748B;margin-top:8px;">Headcount: <span style="color:#F59E0B;">{ex["hc"]}</span> &nbsp;·&nbsp; FT Weekly: <span style="color:#F59E0B;">${ex["wk_cost"]:,.0f}</span></div>', unsafe_allow_html=True)
        st.markdown('<span class="ref">Dantzig (1954) 5-on-2-off patterns</span><span class="ref">A@x >= demand_vec validated ∀j</span>', unsafe_allow_html=True)

    # Tab 5 — Ramp
    with t5:
        st.markdown('<div style="font-size:.70rem;color:#64748B;font-family:\'IBM Plex Mono\',monospace;margin-bottom:8px;">Each row = 18 independently solved & validated MILPs. Headcount grows as takt shortens with demand.</div>', unsafe_allow_html=True)
        A_sc = coverage_matrix()
        scrows = []
        from scipy.optimize import milp as _milp, LinearConstraint as _LC, Bounds as _B
        for rp in [0,10,20,30,40,50,60,80]:
            dm_sc, t_sc, ad_sc = build_demand(p["daily_dem"], p["shift_dur"]*60, p["brk"], rp)
            hc_sc = 0; ok_sc = True
            for sn in STATIONS:
                for sh in SHIFT_MULT:
                    r = solve(dm_sc[sn][sh]["vec"], A_sc)
                    if not r["ok"] or not r["valid"]: ok_sc = False
                    else: hc_sc += r["hc"]
            scrows.append({
                "Ramp %": rp, "Adj Demand": ad_sc,
                "Takt (min/unit)": round(t_sc,4),
                "Min Headcount":   hc_sc if ok_sc else "—",
                "FT Weekly ($)":   f"{hc_sc*shift_cost('Full-Time'):,.0f}" if ok_sc else "—",
                "Validated":       "YES" if ok_sc else "NO",
            })
        st.dataframe(pd.DataFrame(scrows), use_container_width=True, hide_index=True)
        st.markdown('<span class="ref">Womack & Jones (1996) takt-time</span><span class="ref">All 18 MILPs validated per scenario</span>', unsafe_allow_html=True)

    # Tab 6 — Charts
    with t6:
        import altair as alt

        # Chart 1 — Headcount grouped bar
        st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.68rem;color:#F59E0B;letter-spacing:.1em;text-transform:uppercase;border-bottom:1px solid #1E2D3D;padding-bottom:5px;margin-bottom:10px;">HEADCOUNT BY STATION & SHIFT</div>', unsafe_allow_html=True)
        st.caption("Minimum workers needed per station-shift sub-problem. Grouped by shift.")
        hc_rows = [{"Station":s,"Shift":sh,"Headcount":sols[s][sh]["hc"]}
                   for s in STATIONS for sh in SHIFT_MULT if sols[s][sh]["ok"]]
        df_hc = pd.DataFrame(hc_rows)
        chart_hc = (
            alt.Chart(df_hc).mark_bar(cornerRadiusTopLeft=2,cornerRadiusTopRight=2)
            .encode(
                x=alt.X("Shift:N",title=None,sort=["Day","Evening","Night"],
                         axis=alt.Axis(labelAngle=0,labelColor="#64748B",labelFont="IBM Plex Mono",
                                       labelFontSize=10,gridColor="#1E2D3D",domainColor="#1E2D3D")),
                y=alt.Y("Headcount:Q",title="Workers",
                         axis=alt.Axis(labelColor="#64748B",titleColor="#64748B",
                                       labelFont="IBM Plex Mono",gridColor="#1E2D3D",domainColor="#1E2D3D")),
                color=alt.Color("Shift:N",scale=alt.Scale(
                    domain=["Day","Evening","Night"],range=["#F59E0B","#78716C","#292524"]),
                    legend=alt.Legend(labelColor="#94A3B8",titleColor="#94A3B8",
                                      labelFont="IBM Plex Mono",labelFontSize=10)),
                column=alt.Column("Station:N",title=None,
                    header=alt.Header(labelAngle=-30,labelAlign="right",labelColor="#94A3B8",
                                      labelFont="IBM Plex Mono",labelFontSize=10)),
                tooltip=["Station","Shift","Headcount"],
            )
            .properties(width=95,height=200,background="#0D1117")
            .configure_view(strokeColor="#1E2D3D")
        )
        st.altair_chart(chart_hc,use_container_width=False)

        st.markdown('<div style="border-bottom:1px solid #1E2D3D;margin:16px 0 12px;"></div>', unsafe_allow_html=True)

        # Chart 2 — Cost stacked
        st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.68rem;color:#F59E0B;letter-spacing:.1em;text-transform:uppercase;border-bottom:1px solid #1E2D3D;padding-bottom:5px;margin-bottom:10px;">WEEKLY COST BY STATION</div>', unsafe_allow_html=True)
        st.caption("Stacked by shift — Day + Evening + Night FT loaded cost per station.")
        cost_rows=[{"Station":s,"Shift":sh,"Cost ($)":round(sols[s][sh]["wk_cost"],0)}
                   for s in STATIONS for sh in SHIFT_MULT if sols[s][sh]["ok"]]
        df_cost=pd.DataFrame(cost_rows)
        chart_cost=(
            alt.Chart(df_cost).mark_bar()
            .encode(
                x=alt.X("Station:N",sort=list(STATIONS.keys()),title=None,
                         axis=alt.Axis(labelAngle=-25,labelColor="#64748B",labelFont="IBM Plex Mono",
                                       gridColor="#1E2D3D",domainColor="#1E2D3D")),
                y=alt.Y("Cost ($):Q",title="Weekly Cost ($)",
                         axis=alt.Axis(labelColor="#64748B",titleColor="#64748B",
                                       labelFont="IBM Plex Mono",gridColor="#1E2D3D",domainColor="#1E2D3D")),
                color=alt.Color("Shift:N",scale=alt.Scale(
                    domain=["Day","Evening","Night"],range=["#F59E0B","#92400E","#1C1917"]),
                    legend=alt.Legend(labelColor="#94A3B8",titleColor="#94A3B8",
                                      labelFont="IBM Plex Mono",labelFontSize=10)),
                order=alt.Order("Shift:N",sort="ascending"),
                tooltip=["Station","Shift","Cost ($)"],
            )
            .properties(height=280,background="#0D1117")
            .configure_view(strokeColor="#1E2D3D")
        )
        st.altair_chart(chart_cost,use_container_width=True)

        st.markdown('<div style="border-bottom:1px solid #1E2D3D;margin:16px 0 12px;"></div>', unsafe_allow_html=True)

        # Chart 3 — Surplus heatmap
        st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.68rem;color:#F59E0B;letter-spacing:.1em;text-transform:uppercase;border-bottom:1px solid #1E2D3D;padding-bottom:5px;margin-bottom:10px;">COVERAGE SURPLUS HEATMAP — DAY SHIFT</div>', unsafe_allow_html=True)
        st.caption("Extra workers scheduled beyond demand floor each day. Zero = no absenteeism buffer. Non-zero cells are artefacts of the 5-on-2-off pattern structure.")
        heat_rows=[]
        for s in STATIONS:
            if not sols[s]["Day"]["ok"]: continue
            for j,day in enumerate(DAYS):
                heat_rows.append({"Station":s,"Day":day,
                                   "Surplus":int(sols[s]["Day"]["surplus"][j]),
                                   "Demand":int(sols[s]["Day"]["demand"][j]),
                                   "Coverage":int(sols[s]["Day"]["coverage"][j])})
        df_heat=pd.DataFrame(heat_rows)
        base_heat=(
            alt.Chart(df_heat).mark_rect(stroke="#0D1117",strokeWidth=2)
            .encode(
                x=alt.X("Day:N",sort=DAYS,title=None,
                         axis=alt.Axis(labelColor="#64748B",labelFont="IBM Plex Mono",domainColor="#1E2D3D")),
                y=alt.Y("Station:N",sort=list(STATIONS.keys()),title=None,
                         axis=alt.Axis(labelColor="#64748B",labelFont="IBM Plex Mono",domainColor="#1E2D3D")),
                color=alt.Color("Surplus:Q",scale=alt.Scale(
                    domain=[0,5],range=["#1C2632","#F59E0B"]),
                    legend=alt.Legend(labelColor="#94A3B8",titleColor="#94A3B8",
                                      labelFont="IBM Plex Mono",title="Surplus")),
                tooltip=["Station","Day","Demand","Coverage","Surplus"],
            ).properties(height=220,background="#0D1117")
        )
        text_heat=(
            alt.Chart(df_heat).mark_text(fontSize=11,fontWeight="bold",font="IBM Plex Mono")
            .encode(
                x=alt.X("Day:N",sort=DAYS),
                y=alt.Y("Station:N",sort=list(STATIONS.keys())),
                text=alt.Text("Surplus:Q"),
                color=alt.condition(alt.datum.Surplus>2,alt.value("#0D1117"),alt.value("#F1F5F9")),
            )
        )
        st.altair_chart((base_heat+text_heat).configure_view(strokeColor="#1E2D3D"),
                        use_container_width=True)
        st.caption("Zero cells have no buffer — one absent worker drops below demand on that day.")

        st.markdown('<div style="border-bottom:1px solid #1E2D3D;margin:16px 0 12px;"></div>', unsafe_allow_html=True)

        # Chart 4 — Ramp curve
        st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.68rem;color:#F59E0B;letter-spacing:.1em;text-transform:uppercase;border-bottom:1px solid #1E2D3D;padding-bottom:5px;margin-bottom:10px;">PRODUCTION RAMP CURVE</div>', unsafe_allow_html=True)
        st.caption("Headcount and FT weekly cost vs. production ramp. Each point = 18 solved MILPs.")
        ramp_rows=[]
        A_ch=coverage_matrix()
        for rp in [0,10,20,30,40,50,60,80]:
            adj_r=int(math.ceil(p["daily_dem"]*(1+rp/100)))
            t_r=(p["shift_dur"]*60-p["brk"])/adj_r
            tot=0
            for wc in STATIONS.values():
                d0=math.ceil(wc["wc"]/t_r)
                for m in SHIFT_MULT.values():
                    wd=math.ceil(d0*m); we=math.ceil(wd*WEEKEND_RATIO)
                    dv=np.array([wd,wd,wd,wd,wd,we,we],dtype=float)
                    r2=milp(np.ones(7),constraints=LinearConstraint(A_ch,dv,np.full(7,np.inf)),
                            integrality=np.ones(7),bounds=Bounds(np.zeros(7),np.full(7,100.0)))
                    if r2.status==0: tot+=int(np.round(r2.x).sum())
            ramp_rows.append({"Ramp %":rp,"Headcount":tot,"Weekly Cost ($)":round(tot*shift_cost("Full-Time"),0)})
        df_ramp=pd.DataFrame(ramp_rows)
        base_r=alt.Chart(df_ramp).encode(x=alt.X("Ramp %:Q",title="Production Ramp (%)",
            axis=alt.Axis(labelColor="#64748B",titleColor="#64748B",labelFont="IBM Plex Mono",
                          gridColor="#1E2D3D",domainColor="#1E2D3D")))
        line_hc_r=base_r.mark_line(point=True,color="#F59E0B",strokeWidth=2).encode(
            y=alt.Y("Headcount:Q",title="Min Headcount",
                    axis=alt.Axis(titleColor="#F59E0B",labelColor="#64748B",
                                  labelFont="IBM Plex Mono",gridColor="#1E2D3D",domainColor="#1E2D3D")),
            tooltip=["Ramp %","Headcount"])
        line_cost_r=base_r.mark_line(point=True,color="#34D399",strokeWidth=2,strokeDash=[4,2]).encode(
            y=alt.Y("Weekly Cost ($):Q",title="Weekly Cost ($)",
                    axis=alt.Axis(titleColor="#34D399",labelColor="#64748B",
                                  labelFont="IBM Plex Mono",gridColor="#1E2D3D",domainColor="#1E2D3D")),
            tooltip=["Ramp %","Weekly Cost ($)"])
        st.altair_chart(
            alt.layer(line_hc_r,line_cost_r).resolve_scale(y="independent")
            .properties(height=280,background="#0D1117")
            .configure_view(strokeColor="#1E2D3D"),
            use_container_width=True)
        st.caption("Amber solid = headcount (left axis) · Green dashed = weekly cost (right axis)")

        st.markdown('<div style="border-bottom:1px solid #1E2D3D;margin:16px 0 12px;"></div>', unsafe_allow_html=True)

        # Chart 5 — Mix sensitivity
        st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.68rem;color:#F59E0B;letter-spacing:.1em;text-transform:uppercase;border-bottom:1px solid #1E2D3D;padding-bottom:5px;margin-bottom:10px;">WORKER MIX — COST SENSITIVITY</div>', unsafe_allow_html=True)
        st.caption("Weekly cost across FT/PT/CT ratios at current headcount. Sorted cheapest → most expensive. Coloured by contractor share.")
        mix_s=[]
        for _ft in [60,70,80,90,100]:
            for _ct in [0,10,20]:
                _pt=100-_ft-_ct
                if _pt<0: continue
                _c=total_hc*((_ft/100)*shift_cost("Full-Time")+(_pt/100)*shift_cost("Part-Time")+(_ct/100)*shift_cost("Contractor"))
                mix_s.append({"FT%":_ft,"CT%":_ct,"PT%":_pt,"Weekly Cost ($)":round(_c,0),"Mix":f"FT{_ft} PT{_pt} CT{_ct}"})
        df_mix=pd.DataFrame(mix_s)
        chart_mix=(
            alt.Chart(df_mix).mark_bar(cornerRadiusTopLeft=2,cornerRadiusTopRight=2)
            .encode(
                x=alt.X("Mix:N",title=None,
                         sort=alt.EncodingSortField("Weekly Cost ($)",order="ascending"),
                         axis=alt.Axis(labelAngle=-35,labelColor="#64748B",labelFont="IBM Plex Mono",
                                       gridColor="#1E2D3D",domainColor="#1E2D3D")),
                y=alt.Y("Weekly Cost ($):Q",title="Weekly Cost ($)",
                         axis=alt.Axis(labelColor="#64748B",titleColor="#64748B",
                                       labelFont="IBM Plex Mono",gridColor="#1E2D3D",domainColor="#1E2D3D")),
                color=alt.Color("CT%:O",scale=alt.Scale(
                    domain=[0,10,20],range=["#F59E0B","#B45309","#78350F"]),
                    legend=alt.Legend(title="CT%",labelColor="#94A3B8",titleColor="#94A3B8",
                                      labelFont="IBM Plex Mono")),
                tooltip=["FT%","PT%","CT%","Weekly Cost ($)"],
            )
            .properties(height=260,background="#0D1117")
            .configure_view(strokeColor="#1E2D3D")
        )
        st.altair_chart(chart_mix,use_container_width=True)
        st.caption("Contractors cost ~83% more per shift loaded ($419 vs $229). Amber = no contractors, brown = 20% contractor share.")

    # ── Export + AI debrief ────────────────────────────────────────────────────
    st.markdown('<div style="border-bottom:1px solid #1E2D3D;margin:1.4rem 0 .8rem;"></div>', unsafe_allow_html=True)
    ec1, ec2 = st.columns([1, 1])

    with ec1:
        export = {
            "app": "ShiftIQ — Workforce Shift Scheduling Optimizer",
            "model": "Dantzig (1954) OR 2(3):339-341. HiGHS via scipy.optimize.milp.",
            "generated": datetime.now().isoformat(),
            "parameters": p, "takt": round(tk_,4), "adj_demand": adj,
            "oee": round(oev,4), "total_headcount": total_hc,
            "weekly_cost_ft": round(ft_wkcost,2), "weekly_cost_mix": round(mix_cost,2),
            "all_validated": all_ok,
            "schedule": {s: {sh: {
                "hc": sols[s][sh]["hc"], "wk_cost": sols[s][sh]["wk_cost"],
                "valid": sols[s][sh]["valid"],
                "patterns": sols[s][sh]["patterns"].tolist(),
                "coverage": sols[s][sh]["coverage"].tolist(),
                "demand":   sols[s][sh]["demand"].tolist(),
            } for sh in SHIFT_MULT} for s in STATIONS},
        }
        st.download_button(
            "↓ EXPORT JSON",
            data=json.dumps(export, indent=2),
            file_name=f"shiftiq_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )

    with ec2:
        st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.65rem;color:#F59E0B;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">AI SCHEDULE DEBRIEF — GROQ · LLAMA 3.3 70B</div>', unsafe_allow_html=True)
        groq_key = ""
        try:
            groq_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
        if not groq_key:
            import os
            groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key:
            groq_key = st.text_input(
                "Groq API key",
                type="password",
                placeholder="gsk_...",
                help="Free at console.groq.com · Add as GROQ_API_KEY in Streamlit secrets for deployment",
                label_visibility="collapsed",
            )
        gen_btn = st.button("GENERATE DEBRIEF", use_container_width=True,
                            disabled=not bool(groq_key))
        if not groq_key:
            st.markdown('<div style="font-size:.68rem;color:#475569;font-family:\'IBM Plex Mono\',monospace;">Enter Groq key above · free at console.groq.com</div>', unsafe_allow_html=True)

    if gen_btn and groq_key:
        with st.spinner("Calling Groq …"):
            try:
                prompt  = build_prompt(sols, p, tk_, adj, ft_wkcost, mix_cost, oev, total_hc, all_ok)
                summary = call_groq(prompt, groq_key)
                st.session_state["ai_summary"] = summary
                st.session_state["ai_prompt"]  = prompt
            except Exception as e:
                err = str(e)
                st.session_state["ai_summary"] = (
                    f"**Error: {err}**\n\n"
                    "Common causes: key copied incorrectly (must start with `gsk_`), "
                    "key expired (regenerate at console.groq.com), "
                    "secret name wrong (must be exactly `GROQ_API_KEY`)."
                )
                st.session_state["ai_prompt"] = ""

    if "ai_summary" in st.session_state and st.session_state["ai_summary"]:
        st.markdown(f'<div class="debrief">{st.session_state["ai_summary"].replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True)
        with st.expander("VIEW PROMPT SENT TO MODEL", expanded=False):
            st.markdown('<div style="font-size:.70rem;color:#64748B;font-family:\'IBM Plex Mono\',monospace;margin-bottom:8px;">Every number in this prompt comes directly from the MILP solver output.</div>', unsafe_allow_html=True)
            if st.session_state.get("ai_prompt"):
                st.code(st.session_state["ai_prompt"], language="text")

# ─────────────────────────────────────────────────────────────────────────────
# PRE-RUN LANDING
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown('<div class="shdr">HOW IT WORKS</div>', unsafe_allow_html=True)
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.markdown("""<div class="fml">
<b>// Problem</b><br>
Most plants set shift schedules<br>
in Excel with no guarantee the<br>
schedule covers demand on every<br>
day of the week.<br><br>
<b>// This tool</b><br>
Derives demand from takt time,<br>
solves Dantzig (1954) MILP,<br>
validates every constraint,<br>
and reports loaded labor cost.
</div>""", unsafe_allow_html=True)
    with lc2:
        st.markdown("""<div class="fml">
<b>// Inputs</b><br>
Daily unit target<br>
Shift duration & break time<br>
Production ramp %<br>
Worker mix (FT / PT / CT)<br>
Hourly wages<br>
OEE components (A, P, Q)<br><br>
<b>// Outputs</b><br>
Minimum headcount schedule<br>
Post-solve constraint proof<br>
Loaded weekly cost<br>
5 interactive charts<br>
AI debrief (Groq)
</div>""", unsafe_allow_html=True)
    with lc3:
        st.markdown("""<div class="fml">
<b>// Verified references</b><br>
[1] Dantzig 1954<br>
[2] Van den Bergh 2013<br>
[3] Womack & Jones 1996<br>
[4] Ernst et al. 2004<br>
[5] ISO 22400-2:2014<br>
[6] Nakajima 1988<br>
[7] SHRM 2023<br>
[8] Veinott & Wagner 1962<br><br>
Configure in sidebar →<br>
RUN OPTIMIZER
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #1E2D3D;margin-top:2rem;padding-top:1rem;
     font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:#334155;
     display:flex;justify-content:space-between;align-items:center;">
  <span>ShiftIQ · Dantzig (1954) MILP · HiGHS Solver · ISO 22400-2 OEE</span>
  <span>Rutwik Satish · MS Engineering Management · Northeastern University</span>
</div>
""", unsafe_allow_html=True)
