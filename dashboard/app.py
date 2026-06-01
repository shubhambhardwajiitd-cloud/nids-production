import random
import os
import time
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from collections import defaultdict

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIDS — Threat Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Config ────────────────────────────────────────────────────────────────────
import os
API_URL = os.getenv("NIDS_API_URL", "http://localhost:8000")
API_KEY = os.getenv("NIDS_API_KEY", "dev-key-free-001")
HEADERS = {"x-api-key": API_KEY}

SEVERITY_COLOR = {
    "CRITICAL": "#FF3B3B",
    "HIGH":     "#FF9F0A",
    "MEDIUM":   "#FFD60A",
    "LOW":      "#32D74B",
}

SEVERITY_BG = {
    "CRITICAL": "rgba(255,59,59,0.12)",
    "HIGH":     "rgba(255,159,10,0.12)",
    "MEDIUM":   "rgba(255,214,10,0.12)",
    "LOW":      "rgba(50,215,75,0.12)",
}

# ── CSS — IDENTICAL to previous design ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #000000;
    color: #E5E5E5;
}
.stApp { background: #000000; }
#MainMenu, footer, header,
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"]        { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Expander — matches existing dark card style */
[data-testid="stExpander"] {
    background: #080808 !important;
    border: 1px solid #1A1A1A !important;
    border-radius: 8px !important;
    margin-bottom: 6px !important;
}
[data-testid="stExpander"]:hover { border-color: #2A2A2A !important; }
details summary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #CCC !important;
    padding: 10px 14px !important;
}
details summary:hover { color: #FFF !important; }

.topbar {
    background: #000;
    border-bottom: 1px solid #1A1A1A;
    padding: 12px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}
.brand-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: #32D74B;
    letter-spacing: 0.2em;
}
.brand-sub {
    font-size: 0.65rem;
    color: #404040;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
.live-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(50,215,75,0.08);
    border: 1px solid rgba(50,215,75,0.2);
    border-radius: 20px;
    padding: 4px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #32D74B;
}
.live-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #32D74B;
    animation: blink 1.5s infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; box-shadow: 0 0 6px #32D74B; }
    50% { opacity: 0.3; box-shadow: none; }
}
.topbar-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #404040;
}
.sev-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.sev-card {
    border-radius: 10px;
    padding: 18px 20px;
    border: 1px solid #1E2A3A;
    background: #080808;
    transition: transform 0.2s;
}
.sev-card:hover { transform: translateY(-2px); }
.sev-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
}
.sev-label {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 6px;
    color: #555;
}
.sev-bar {
    height: 2px;
    border-radius: 1px;
    margin-top: 14px;
    opacity: 0.4;
}
.panel-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #404040;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #111;
}
.offline-banner {
    background: rgba(255,59,59,0.08);
    border: 1px solid rgba(255,59,59,0.2);
    border-radius: 8px;
    padding: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #FF3B3B;
    text-align: center;
}
/* SHAP inside expander */
.shap-track {
    height: 5px;
    background: #111;
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 3px;
}
.shap-fill-pos {
    height: 100%;
    background: linear-gradient(90deg, #C0392B, #FF3B3B);
    border-radius: 3px;
}
.shap-fill-neg {
    height: 100%;
    background: linear-gradient(90deg, #1E8449, #32D74B);
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3)
def fetch_alerts(limit=100, severity=None):
    try:
        params = {"limit": limit}
        if severity and severity != "ALL":
            params["severity"] = severity
        r = requests.get(f"{API_URL}/alerts", headers=HEADERS,
                         params=params, timeout=5)
        return r.json() if r.status_code == 200 else {"alerts": [], "total": 0}
    except Exception:
        return {"alerts": [], "total": 0}


@st.cache_data(ttl=3)
def fetch_stats():
    try:
        r = requests.get(f"{API_URL}/stats", headers=HEADERS, timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def api_online():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Data ──────────────────────────────────────────────────────────────────────
online      = api_online()
stats       = fetch_stats() if online else {}
alert_data  = fetch_alerts(limit=100) if online else {"alerts": [], "total": 0}
alert_list  = alert_data.get("alerts", [])
attack_dist = stats.get("attack_distribution", {})
sev_dist    = stats.get("severity_breakdown", {})
total       = stats.get("total_alerts", 0)
now_str     = datetime.now().strftime("%H:%M:%S")

# ── Top bar ───────────────────────────────────────────────────────────────────
status_html = (
    '<span class="live-badge"><span class="live-dot"></span>LIVE</span>'
    if online else
    '<span style="color:#FF3B3B;font-family:JetBrains Mono,monospace;'
    'font-size:0.7rem;">● OFFLINE</span>'
)

st.markdown(f"""
<div class="topbar">
    <div style="display:flex;align-items:center;gap:16px;">
        <div>
            <div class="brand-name">🛡 NIDS</div>
            <div class="brand-sub">Network Intrusion Detection System</div>
        </div>
        {status_html}
    </div>
    <div class="topbar-meta">
        {total} TOTAL ALERTS &nbsp;·&nbsp;
        XGBoost + SHAP &nbsp;·&nbsp;
        CIC-IDS-2018 &nbsp;·&nbsp; {now_str}
    </div>
</div>
<div style="padding:24px 32px;">
""", unsafe_allow_html=True)

if not online:
    st.markdown("""
    <div class="offline-banner">
        ⚠ Cannot reach API at http://localhost:8000<br><br>
        <span style="color:#555;">
            py -3.11 -m uvicorn main:app --reload --port 8000
        </span>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Severity cards ────────────────────────────────────────────────────────────
c  = sev_dist.get("CRITICAL", 0)
h  = sev_dist.get("HIGH", 0)
m  = sev_dist.get("MEDIUM", 0)
l  = sev_dist.get("LOW", 0)
mx = max(c, 1)

st.markdown(f"""
<div class="sev-grid">
    <div class="sev-card">
        <div class="sev-count" style="color:#FF3B3B;">{c}</div>
        <div class="sev-label">Critical</div>
        <div class="sev-bar" style="background:#FF3B3B;width:100%;"></div>
    </div>
    <div class="sev-card">
        <div class="sev-count" style="color:#FF9F0A;">{h}</div>
        <div class="sev-label">High</div>
        <div class="sev-bar" style="background:#FF9F0A;width:{int(h/mx*100)}%;"></div>
    </div>
    <div class="sev-card">
        <div class="sev-count" style="color:#FFD60A;">{m}</div>
        <div class="sev-label">Medium</div>
        <div class="sev-bar" style="background:#FFD60A;width:{int(m/mx*100)}%;"></div>
    </div>
    <div class="sev-card">
        <div class="sev-count" style="color:#32D74B;">{l}</div>
        <div class="sev-label">Low</div>
        <div class="sev-bar" style="background:#32D74B;width:{int(l/mx*100)}%;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Controls ──────────────────────────────────────────────────────────────────
cc1, cc2, cc3, cc4 = st.columns([2, 2, 1, 5])
with cc1:
    sev_filter = st.selectbox(
        "Severity", ["ALL","CRITICAL","HIGH","MEDIUM","LOW"],
        label_visibility="collapsed"
    )
with cc2:
    limit = st.slider("Limit", 10, 100, 50, label_visibility="collapsed")
with cc3:
    refresh = st.button("🔄 Refresh")
with cc4:
    if st.button("🗑 Clear Alert Store"):
        try:
            requests.delete(f"{API_URL}/alerts/clear",
                            headers=HEADERS, timeout=5)
            st.cache_data.clear()
            st.rerun()
        except Exception:
            st.error("Failed")

# Re-fetch with filters
alert_data = fetch_alerts(
    limit=limit,
    severity=sev_filter if sev_filter != "ALL" else None
)
alert_list = alert_data.get("alerts", [])

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

# ── Main columns ──────────────────────────────────────────────────────────────
col_feed, col_mid, col_shap = st.columns([3, 3, 2])

# ════════════════════════════
# FEED — clickable expanders
# ════════════════════════════
with col_feed:
    st.markdown(
        f'<div class="panel-title">LIVE THREAT FEED '
        f'<span style="color:#222;">({len(alert_list)} shown)</span></div>',
        unsafe_allow_html=True
    )

    if not alert_list:
        st.markdown(
            '<div style="color:#333;text-align:center;padding:60px 0;'
            'font-family:JetBrains Mono,monospace;font-size:0.8rem;">'
            '// NO ALERTS YET<br><br>'
            '<span style="color:#222;">run streamer.py to simulate traffic</span>'
            '</div>', unsafe_allow_html=True
        )
    else:
        # ── Scrollable container — fixes blank space below ─────────
        feed_container = st.container(height=520, border=False)
        with feed_container:
            for alert in alert_list[:40]:
                sev       = alert.get("severity", "LOW")
                atype     = alert.get("attack_type", "?")
                conf      = alert.get("confidence", 0)
                ts        = alert.get("timestamp", "")
                shap_data = alert.get("shap_explanation", [])
                aid       = alert.get("alert_id", "")[:8]
                color     = SEVERITY_COLOR.get(sev, "#555")

                try:
                    ts_fmt = datetime.fromisoformat(ts).strftime("%H:%M:%S")
                except Exception:
                    ts_fmt = ts[:8]

                # ── Colored severity bar above each expander ───────
                st.markdown(
                    f'<div style="height:3px;background:{color};'
                    f'border-radius:2px 2px 0 0;margin-bottom:-8px;'
                    f'opacity:0.8;"></div>',
                    unsafe_allow_html=True
                )

                # ── Expander label ─────────────────────────────────
                label = f"● {atype}   {sev}   {conf:.0%}   {ts_fmt}"

                with st.expander(label):
                    st.markdown(f"""
                    <div style="padding:4px 0 12px;">
                        <div style="font-family:JetBrains Mono,monospace;
                                    font-size:0.9rem;font-weight:700;
                                    color:{color};margin-bottom:8px;">
                            {atype}
                        </div>
                        <div style="font-family:JetBrains Mono,monospace;
                                    font-size:0.68rem;color:#444;
                                    margin-bottom:16px;">
                            severity: {sev} &nbsp;·&nbsp;
                            confidence: {conf:.1%} &nbsp;·&nbsp;
                            id: #{aid} &nbsp;·&nbsp; {ts_fmt}
                        </div>
                        <div style="font-family:JetBrains Mono,monospace;
                                    font-size:0.6rem;color:#2A2A2A;
                                    text-transform:uppercase;
                                    letter-spacing:0.15em;margin-bottom:12px;">
                            Why this alert fired:
                        </div>
                    """, unsafe_allow_html=True)

                    if shap_data:
                        max_val = max(abs(s["shap_value"]) for s in shap_data) or 1
                        for s in shap_data:
                            val      = s["shap_value"]
                            pct      = int(abs(val) / max_val * 100)
                            sign     = "+" if val > 0 else ""
                            val_col  = "#FF3B3B" if val > 0 else "#32D74B"
                            fill_cls = "shap-fill-pos" if val > 0 \
                                       else "shap-fill-neg"
                            meaning  = "pushes toward ATTACK" \
                                       if val > 0 else "pushes toward BENIGN"

                            st.markdown(f"""
                            <div style="margin-bottom:12px;">
                                <div style="display:flex;
                                            justify-content:space-between;
                                            font-family:JetBrains Mono,monospace;
                                            font-size:0.72rem;margin-bottom:5px;">
                                    <span style="color:#888;">{s['feature']}</span>
                                    <span style="color:{val_col};font-weight:700;">
                                        {sign}{val:.4f}
                                    </span>
                                </div>
                                <div class="shap-track">
                                    <div class="{fill_cls}"
                                         style="width:{pct}%;"></div>
                                </div>
                                <div style="font-family:JetBrains Mono,monospace;
                                            font-size:0.6rem;color:#2A2A2A;
                                            margin-top:3px;">{meaning}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("""
                        <div style="margin-top:12px;padding:10px 12px;
                                    background:#060606;border-radius:6px;
                                    font-family:JetBrains Mono,monospace;
                                    font-size:0.6rem;color:#2A2A2A;
                                    line-height:2;">
                            <span style="color:#FF3B3B;">■</span>
                            red = increases attack probability &nbsp;&nbsp;
                            <span style="color:#32D74B;">■</span>
                            green = decreases attack probability
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div style="color:#333;font-family:JetBrains Mono,'
                            'monospace;font-size:0.75rem;">// no shap data</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════
# MIDDLE — charts (static)
# ════════════════════════════
STATIC_CFG = {"displayModeBar": False, "staticPlot": True}
STATIC_AXIS = dict(fixedrange=True)

with col_mid:
    st.markdown('<div class="panel-title">ATTACK BREAKDOWN</div>',
                unsafe_allow_html=True)

    if attack_dist:
        df_a = pd.DataFrame(
            list(attack_dist.items()),
            columns=["Attack", "Count"]
        ).sort_values("Count", ascending=True)

        fig = go.Figure(go.Bar(
            x=df_a["Count"], y=df_a["Attack"],
            orientation="h",
            marker=dict(
                color=df_a["Count"],
                colorscale=[[0,"#0D2B0D"],[0.5,"#1A5C1A"],[1,"#32D74B"]],
                showscale=False,
            ),
            text=df_a["Count"],
            textposition="outside",
            textfont=dict(color="#444", size=9, family="JetBrains Mono"),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=40, t=0, b=0),
            height=280,
            dragmode=False,
            xaxis=dict(showgrid=False, showticklabels=False,
                       zeroline=False, **STATIC_AXIS),
            yaxis=dict(showgrid=False, color="#555",
                       tickfont=dict(family="JetBrains Mono", size=9),
                       **STATIC_AXIS),
        )
        st.plotly_chart(fig, use_container_width=True, config=STATIC_CFG)
    else:
        st.markdown(
            '<div style="color:#222;text-align:center;padding:40px 0;'
            'font-family:JetBrains Mono,monospace;font-size:0.75rem;">'
            '// NO DATA</div>', unsafe_allow_html=True
        )

    st.markdown(
        '<div class="panel-title" style="margin-top:16px;">ALERT TIMELINE</div>',
        unsafe_allow_html=True
    )

    if alert_list:
        timeline = defaultdict(int)
        for a in alert_list:
            try:
                bucket = datetime.fromisoformat(
                    a["timestamp"]).strftime("%H:%M")
                timeline[bucket] += 1
            except Exception:
                pass

        if timeline:
            times  = sorted(timeline.keys())
            counts = [timeline[t] for t in times]

            fig2 = go.Figure(go.Scatter(
                x=times, y=counts,
                fill="tozeroy",
                fillcolor="rgba(50,215,75,0.06)",
                line=dict(color="#32D74B", width=1.5),
                mode="lines",
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=0, b=0),
                height=150,
                dragmode=False,
                xaxis=dict(showgrid=False, color="#2A2A2A",
                           tickfont=dict(family="JetBrains Mono", size=8),
                           **STATIC_AXIS),
                yaxis=dict(showgrid=False, showticklabels=False,
                           **STATIC_AXIS),
            )
            st.plotly_chart(fig2, use_container_width=True,
                            config=STATIC_CFG)
    else:
        st.markdown(
            '<div style="color:#222;text-align:center;padding:30px 0;'
            'font-family:JetBrains Mono,monospace;font-size:0.75rem;">'
            '// NO TIMELINE DATA</div>', unsafe_allow_html=True
        )

# ════════════════════════════
# RIGHT — SHAP latest alert
# ════════════════════════════
with col_shap:
    st.markdown('<div class="panel-title">SHAP ANALYSIS — LATEST</div>',
                unsafe_allow_html=True)

    if alert_list:
        latest    = alert_list[0]
        atype     = latest.get("attack_type", "?")
        conf      = latest.get("confidence", 0)
        sev       = latest.get("severity", "LOW")
        shap_data = latest.get("shap_explanation", [])
        color     = SEVERITY_COLOR.get(sev, "#555")
        bg        = SEVERITY_BG.get(sev, "rgba(85,85,85,0.1)")

        st.markdown(f"""
        <div style="background:#0D1321;border-radius:10px;
                    padding:14px 16px;margin-bottom:14px;
                    border:1px solid #1A1A1A;">
            <div style="font-family:JetBrains Mono,monospace;
                        font-size:0.9rem;font-weight:700;color:{color};">
                {atype}
            </div>
            <div style="font-size:0.72rem;color:#555;margin-top:4px;
                        font-family:JetBrains Mono,monospace;">
                confidence: {conf:.1%} &nbsp;·&nbsp; {sev}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if shap_data:
            max_val = max(abs(s["shap_value"]) for s in shap_data) or 1
            for s in shap_data:
                val      = s["shap_value"]
                pct      = int(abs(val) / max_val * 100)
                sign     = "+" if val > 0 else ""
                val_col  = "#FF3B3B" if val > 0 else "#32D74B"
                fill_cls = "shap-fill-pos" if val > 0 else "shap-fill-neg"
                meaning  = "↑ increases risk" if val > 0 else "↓ decreases risk"

                st.markdown(f"""
                <div style="margin-bottom:14px;">
                    <div style="display:flex;justify-content:space-between;
                                font-family:JetBrains Mono,monospace;
                                font-size:0.72rem;margin-bottom:5px;">
                        <span style="color:#888;">{s['feature']}</span>
                        <span style="color:{val_col};font-weight:700;">
                            {sign}{val:.3f}
                        </span>
                    </div>
                    <div class="shap-track">
                        <div class="{fill_cls}" style="width:{pct}%;"></div>
                    </div>
                    <div style="font-size:0.62rem;color:#333;margin-top:3px;
                                font-family:JetBrains Mono,monospace;">
                        {meaning}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div style="margin-top:16px;padding:10px;background:#0A0A0A;
                        border-radius:6px;font-family:JetBrains Mono,monospace;
                        font-size:0.62rem;color:#333;line-height:2;">
                <span style="color:#FF3B3B;">■</span> increases attack risk<br>
                <span style="color:#32D74B;">■</span> decreases attack risk<br>
                bar width = relative importance
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="color:#333;font-family:JetBrains Mono,monospace;'
                'font-size:0.75rem;">// no shap data</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div style="color:#222;text-align:center;padding:60px 0;'
            'font-family:JetBrains Mono,monospace;font-size:0.75rem;">'
            '// waiting for alerts</div>',
            unsafe_allow_html=True
        )


# ── Try It Yourself ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="panel-title">🎲 TRY IT YOURSELF</div>',
            unsafe_allow_html=True)

if st.button("Generate Random Flow & Predict", type="primary"):
    try:
        import pandas as pd
        
        # Load a random row from sample data
        df = pd.read_csv("data/sample_data.csv")
        row = df.sample(1).iloc[0]
        true_label = row["Label"]
        features = row.drop("Label").to_dict()
        
        # Send to API
        resp = requests.post(
            f"{API_URL}/predict?model_name=xgboost",
            headers={**HEADERS, "Content-Type": "application/json"},
            json={"features": features},
            timeout=10
        )
        result = resp.json()
        
        # Show result
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("True Label", true_label)
        with r2:
            st.metric("Predicted", result["prediction"])
        with r3:
            st.metric("Confidence", f"{result['confidence']:.1%}")
        
        if result.get("shap_explanation"):
            st.markdown("**Why this was flagged:**")
            for s in result["shap_explanation"]:
                direction = "⬆️ risk" if s["shap_value"] > 0 else "⬇️ risk"
                st.write(f"- **{s['feature']}**: `{s['shap_value']:+.4f}` {direction}")
        
        with st.expander("🔍 Raw API Response"):
            st.json(result)
            
    except Exception as e:
        st.error(f"Error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
</div>
<div style="border-top:1px solid #0F0F0F;padding:14px 32px;
            display:flex;justify-content:space-between;
            font-family:JetBrains Mono,monospace;
            font-size:0.62rem;color:#222;">
    <span>NIDS v1.0 · XGBoost · SHAP Explainability · CIC-IDS-2018</span>
    <span>Built by Shubham Bhardwaj · {now_str}</span>
</div>
""", unsafe_allow_html=True)

# ── Manual refresh ────────────────────────────────────────────────────────────
if refresh:
    st.cache_data.clear()
    st.rerun()
