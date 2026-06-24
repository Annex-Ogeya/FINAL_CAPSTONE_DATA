"""
Urban Traffic Congestion Dashboard
Streamlit app — mirrors the LSTM notebook pipeline with full interactivity.
"""

import warnings, io
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Congestion Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --muted:     #8b949e;
    --text:      #e6edf3;
    --blue:      #58a6ff;
    --green:     #3fb950;
    --orange:    #e3b341;
    --red:       #f78166;
    --purple:    #d2a8ff;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

.stApp { background-color: var(--bg); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label  { color: var(--muted) !important; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; font-family: 'Space Mono', monospace; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--blue) !important; font-family: 'Space Mono', monospace; font-size: 1.6rem; }
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: var(--green) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"]  { background: var(--surface); border-bottom: 1px solid var(--border); gap: 4px; }
.stTabs [data-baseweb="tab"]       { color: var(--muted); font-family: 'Space Mono', monospace; font-size: 13px; padding: 10px 18px; border-radius: 6px 6px 0 0; }
.stTabs [aria-selected="true"]     { background: var(--bg) !important; color: var(--blue) !important; border-bottom: 2px solid var(--blue); }

/* Headers */
h1 { font-family: 'Space Mono', monospace; color: var(--blue) !important; letter-spacing: -0.02em; }
h2, h3 { font-family: 'Space Mono', monospace; color: var(--text) !important; }

/* Section label */
.section-label { font-family: 'Space Mono', monospace; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .12em; margin-bottom: 8px; }

/* Congestion badges */
.badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:12px; font-family:'Space Mono',monospace; font-weight:700; }
.badge-ff  { background:#3fb95022; color:#3fb950; border:1px solid #3fb950; }
.badge-mod { background:#e3b34122; color:#e3b341; border:1px solid #e3b341; }
.badge-hvy { background:#f7816622; color:#f78166; border:1px solid #f78166; }
.badge-sev { background:#da363322; color:#da3633; border:1px solid #da3633; }

/* Slider & widgets */
.stSlider [data-baseweb="slider"] { padding: 0 6px; }
[data-testid="stSelectbox"] > div { background: var(--surface); border-color: var(--border); }
</style>
""", unsafe_allow_html=True)

# ─── Plotly dark template ──────────────────────────────────────────────────────
PTMPL = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    font=dict(family="DM Sans", color="#e6edf3"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#21262d"),
)
JCOLORS = ["#58a6ff","#3fb950","#f78166","#d2a8ff"]
CCOLORS = {"Free Flow":"#3fb950","Moderate":"#e3b341","Heavy":"#f78166","Severe":"#da3633"}

# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare(path="traffic.csv"):
    df = pd.read_csv(path, parse_dates=["DateTime"])
    df = df.sort_values(["Junction","DateTime"]).reset_index(drop=True)

    df["hour"]       = df["DateTime"].dt.hour
    df["dayofweek"]  = df["DateTime"].dt.dayofweek
    df["day_name"]   = df["DateTime"].dt.day_name()
    df["month"]      = df["DateTime"].dt.month
    df["month_name"] = df["DateTime"].dt.strftime("%b")
    df["year"]       = df["DateTime"].dt.year
    df["date"]       = df["DateTime"].dt.date
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_peak"]    = df["hour"].apply(lambda h: 1 if (7<=h<=9 or 17<=h<=19) else 0)

    # Cyclical
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    # Per-junction percentile thresholds
    thresholds = {}
    for junc in sorted(df["Junction"].unique()):
        v = df[df["Junction"]==junc]["Vehicles"]
        thresholds[junc] = dict(p33=v.quantile(.33), p66=v.quantile(.66), p90=v.quantile(.90))

    def label_row(row):
        t = thresholds[row["Junction"]]
        v = row["Vehicles"]
        if   v <= t["p33"]: return "Free Flow"
        elif v <= t["p66"]: return "Moderate"
        elif v <= t["p90"]: return "Heavy"
        else:               return "Severe"

    df["congestion"] = df.apply(label_row, axis=1)
    return df, thresholds

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚦 Traffic Intelligence")
    st.markdown('<p class="section-label">Data source</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload traffic.csv", type="csv")
    if uploaded:
        raw = pd.read_csv(uploaded, parse_dates=["DateTime"])
        raw.to_csv("/tmp/traffic_upload.csv", index=False)
        DATA_PATH = "/tmp/traffic_upload.csv"
    else:
        DATA_PATH = "traffic.csv"

    df, thresholds = load_and_prepare(DATA_PATH)

    st.markdown("---")
    st.markdown('<p class="section-label">Filters</p>', unsafe_allow_html=True)

    all_junctions = sorted(df["Junction"].unique())
    sel_junctions = st.multiselect("Junctions", all_junctions,
                                   default=all_junctions, key="jsel")
    if not sel_junctions:
        sel_junctions = all_junctions

    date_min = df["DateTime"].min().date()
    date_max = df["DateTime"].max().date()
    date_range = st.date_input("Date range",
        value=(date_min, date_max),
        min_value=date_min, max_value=date_max)
    if len(date_range) == 2:
        d0, d1 = date_range
    else:
        d0, d1 = date_min, date_max

    hour_range = st.slider("Hour of day", 0, 23, (0, 23))
    cong_filter = st.multiselect("Congestion levels",
        ["Free Flow","Moderate","Heavy","Severe"],
        default=["Free Flow","Moderate","Heavy","Severe"])

    st.markdown("---")
    st.markdown('<p class="section-label">Forecast settings</p>', unsafe_allow_html=True)
    forecast_junction = st.selectbox("Junction to forecast", all_junctions)
    forecast_hours    = st.slider("Forecast horizon (h)", 6, 72, 24, step=6)

# ─── Apply filters ────────────────────────────────────────────────────────────
mask = (
    df["Junction"].isin(sel_junctions) &
    (df["date"] >= d0) & (df["date"] <= d1) &
    (df["hour"] >= hour_range[0]) & (df["hour"] <= hour_range[1]) &
    df["congestion"].isin(cong_filter)
)
fdf = df[mask].copy()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🚦 Urban Traffic Congestion Intelligence")
st.markdown(
    f"<p style='color:#8b949e;font-family:Space Mono,monospace;font-size:13px'>"
    f"Showing <b style='color:#58a6ff'>{len(fdf):,}</b> records · "
    f"Junctions {sel_junctions} · "
    f"{d0} → {d1} · Hours {hour_range[0]:02d}:00–{hour_range[1]:02d}:59"
    f"</p>", unsafe_allow_html=True
)

# ─── KPI row ──────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
with k1: st.metric("Total Records",   f"{len(fdf):,}")
with k2: st.metric("Avg Vehicles/hr", f"{fdf['Vehicles'].mean():.1f}")
with k3: st.metric("Peak Hour",       f"{int(fdf.groupby('hour')['Vehicles'].mean().idxmax())}:00")
with k4:
    sev_pct = (fdf["congestion"]=="Severe").mean()*100
    st.metric("Severe Congestion", f"{sev_pct:.1f}%")
with k5:
    ff_pct = (fdf["congestion"]=="Free Flow").mean()*100
    st.metric("Free Flow",         f"{ff_pct:.1f}%")

st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈  Time Series",
    "🕐  Temporal Patterns",
    "🔴  Congestion",
    "📊  Distribution",
    "🔮  Forecast",
    "🗂  Raw Data",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — TIME SERIES
# ═══════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([3,1])
    with col_r:
        resample_opt = st.radio("Resample", ["Hourly","Daily","Weekly"], index=1, horizontal=False)
        show_congestion_band = st.checkbox("Colour congestion band", value=True)

    resample_map = {"Hourly":"h","Daily":"D","Weekly":"W"}
    freq = resample_map[resample_opt]

    fig = go.Figure()
    for i, junc in enumerate(sel_junctions):
        jdata = fdf[fdf["Junction"]==junc].set_index("DateTime")["Vehicles"]
        resampled = jdata.resample(freq).mean().dropna()
        fig.add_trace(go.Scatter(
            x=resampled.index, y=resampled.values,
            name=f"Junction {junc}",
            line=dict(color=JCOLORS[i%4], width=1.8),
            fill="tozeroy", fillcolor=JCOLORS[i%4].replace("ff","1a"),
            mode="lines", hovertemplate="<b>J%d</b><br>%%{x|%%b %%d %%H:00}<br>%%{y:.1f} veh<extra></extra>" % junc
        ))

    fig.update_layout(**PTMPL, height=400,
        title=dict(text=f"Vehicle Count — {resample_opt}", font=dict(size=14, family="Space Mono")),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        hovermode="x unified")
    with col_l:
        st.plotly_chart(fig, width='stretch')

    # Congestion overlay timeline
    if show_congestion_band and len(fdf) > 0:
        j_sel = st.selectbox("Junction for congestion timeline", sel_junctions, key="ts_junc")
        jdata2 = fdf[fdf["Junction"]==j_sel].copy()
        if freq != "h":
            jdata_d = jdata2.set_index("DateTime")["Vehicles"].resample(freq).mean().reset_index()
            jdata_d.columns = ["DateTime","Vehicles"]
        else:
            jdata_d = jdata2[["DateTime","Vehicles"]].copy()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=jdata_d["DateTime"], y=jdata_d["Vehicles"],
            fill="tozeroy", line=dict(color="#58a6ff", width=1.5),
            fillcolor="rgba(88,166,255,0.08)", name="Vehicles"
        ))
        # Horizontal threshold lines
        t = thresholds[j_sel]
        for val, label, color in [
            (t["p33"],"Free Flow/Moderate","#3fb950"),
            (t["p66"],"Moderate/Heavy",    "#e3b341"),
            (t["p90"],"Heavy/Severe",      "#f78166"),
        ]:
            fig2.add_hline(y=val, line_dash="dot", line_color=color,
                           annotation_text=label, annotation_position="bottom right",
                           annotation_font_size=10)
        fig2.update_layout(**PTMPL, height=280,
            title=dict(text=f"Junction {j_sel} with Congestion Thresholds",
                       font=dict(size=13, family="Space Mono")))
        st.plotly_chart(fig2, width='stretch')

# ═══════════════════════════════════════════════════════════════
# TAB 2 — TEMPORAL PATTERNS
# ═══════════════════════════════════════════════════════════════
with tab2:
    c1, c2 = st.columns(2)

    # Hourly profile
    with c1:
        hourly = fdf.groupby(["hour","Junction"])["Vehicles"].mean().reset_index()
        fig = px.line(hourly, x="hour", y="Vehicles", color="Junction",
                      color_discrete_sequence=JCOLORS,
                      markers=True, title="Average Vehicles by Hour of Day",
                      labels={"hour":"Hour","Vehicles":"Avg Vehicles"})
        fig.update_layout(**PTMPL, height=320)
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(fig, width='stretch')

    # Day-of-week
    with c2:
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = fdf.groupby(["day_name","Junction"])["Vehicles"].mean().reset_index()
        dow["day_name"] = pd.Categorical(dow["day_name"], categories=day_order, ordered=True)
        dow = dow.sort_values("day_name")
        fig = px.bar(dow, x="day_name", y="Vehicles", color="Junction",
                     barmode="group", color_discrete_sequence=JCOLORS,
                     title="Average Vehicles by Day of Week",
                     labels={"day_name":"Day","Vehicles":"Avg Vehicles"})
        fig.update_layout(**PTMPL, height=320)
        st.plotly_chart(fig, width='stretch')

    c3, c4 = st.columns(2)

    # Monthly trend
    with c3:
        monthly = fdf.groupby(["year","month","month_name"])["Vehicles"].mean().reset_index()
        monthly["period"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
        monthly = monthly.sort_values("period")
        fig = px.bar(monthly, x="period", y="Vehicles", color="Vehicles",
                     color_continuous_scale=[[0,"#3fb950"],[0.5,"#e3b341"],[1,"#f78166"]],
                     title="Monthly Average Vehicle Count",
                     labels={"period":"Month","Vehicles":"Avg Vehicles"})
        fig.update_layout(**PTMPL, height=320, coloraxis_showscale=False)
        fig.update_xaxes(tickangle=-40)
        st.plotly_chart(fig, width='stretch')

    # Heatmap Hour × Day
    with c4:
        hj = st.selectbox("Heatmap junction", sel_junctions, key="hmap_junc")
        pivot = (fdf[fdf["Junction"]==hj]
                 .groupby(["hour","day_name"])["Vehicles"].mean()
                 .unstack()
                 .reindex(columns=[d for d in day_order if d in fdf["day_name"].unique()]))
        fig = px.imshow(pivot, color_continuous_scale="YlOrRd",
                        title=f"Junction {hj} — Hour × Day Heatmap",
                        labels=dict(x="Day",y="Hour",color="Avg Vehicles"),
                        aspect="auto")
        fig.update_layout(**PTMPL, height=320)
        st.plotly_chart(fig, width='stretch')

    # Peak vs off-peak comparison
    st.markdown("#### Peak vs Off-Peak Comparison")
    peak_cmp = fdf.groupby(["Junction","is_peak"])["Vehicles"].mean().reset_index()
    peak_cmp["Period"] = peak_cmp["is_peak"].map({0:"Off-Peak",1:"Peak Hours"})
    fig = px.bar(peak_cmp, x="Junction", y="Vehicles", color="Period",
                 barmode="group", color_discrete_map={"Peak Hours":"#f78166","Off-Peak":"#58a6ff"},
                 title="Peak (7–9 AM, 5–7 PM) vs Off-Peak Average Vehicles")
    fig.update_layout(**PTMPL, height=300)
    st.plotly_chart(fig, width='stretch')

# ═══════════════════════════════════════════════════════════════
# TAB 3 — CONGESTION
# ═══════════════════════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns([1,2])

    with c1:
        counts = fdf["congestion"].value_counts().reindex(
            ["Free Flow","Moderate","Heavy","Severe"]).dropna()
        fig = go.Figure(go.Pie(
            labels=counts.index, values=counts.values,
            marker_colors=[CCOLORS[l] for l in counts.index],
            hole=0.45,
            textinfo="percent+label",
            textfont=dict(family="Space Mono", size=11),
        ))
        fig.update_layout(**PTMPL, height=340,
            title=dict(text="Congestion Distribution", font=dict(family="Space Mono",size=13)),
            showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with c2:
        # Stacked bar: congestion % by hour
        hc = (fdf.groupby(["hour","congestion"]).size()
              .unstack(fill_value=0)
              .reindex(columns=["Free Flow","Moderate","Heavy","Severe"], fill_value=0))
        hc_pct = hc.div(hc.sum(axis=1), axis=0) * 100

        fig = go.Figure()
        for level in ["Free Flow","Moderate","Heavy","Severe"]:
            if level in hc_pct.columns:
                fig.add_trace(go.Bar(
                    x=hc_pct.index, y=hc_pct[level],
                    name=level, marker_color=CCOLORS[level],
                    hovertemplate="%{x}:00 — " + level + ": %{y:.1f}%<extra></extra>"
                ))
        fig.update_layout(**PTMPL, barmode="stack", height=340,
            title=dict(text="Congestion Level % by Hour", font=dict(family="Space Mono",size=13)),
            yaxis_title="Share (%)", xaxis_title="Hour of Day",
            legend=dict(bgcolor="#161b22", bordercolor="#30363d"))
        st.plotly_chart(fig, width='stretch')

    # Per-junction congestion breakdown
    jc = (fdf.groupby(["Junction","congestion"]).size()
          .reset_index(name="Count"))
    jc_total = fdf.groupby("Junction").size().reset_index(name="Total")
    jc = jc.merge(jc_total, on="Junction")
    jc["Pct"] = jc["Count"] / jc["Total"] * 100

    fig = px.bar(jc, x="Junction", y="Pct", color="congestion",
                 color_discrete_map=CCOLORS, barmode="stack",
                 title="Congestion Breakdown per Junction",
                 labels={"Pct":"Share (%)","congestion":"Level"})
    fig.update_layout(**PTMPL, height=320,
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"))
    st.plotly_chart(fig, width='stretch')

    # Weekend vs weekday congestion
    st.markdown("#### Weekend vs Weekday Severity")
    wc = (fdf.groupby(["is_weekend","congestion"]).size()
          .reset_index(name="Count"))
    wc["Period"] = wc["is_weekend"].map({0:"Weekday",1:"Weekend"})
    wc_total = fdf.groupby("is_weekend").size().reset_index(name="Total")
    wc = wc.merge(wc_total, on="is_weekend")
    wc["Pct"] = wc["Count"] / wc["Total"] * 100

    fig = px.bar(wc, x="Period", y="Pct", color="congestion",
                 color_discrete_map=CCOLORS, barmode="stack",
                 labels={"Pct":"Share (%)","congestion":"Level"})
    fig.update_layout(**PTMPL, height=280)
    st.plotly_chart(fig, width='stretch')

# ═══════════════════════════════════════════════════════════════
# TAB 4 — DISTRIBUTION
# ═══════════════════════════════════════════════════════════════
with tab4:
    c1, c2 = st.columns(2)

    with c1:
        bins_n = st.slider("Histogram bins", 20, 100, 50)
        fig = px.histogram(fdf, x="Vehicles", color="Junction",
                           nbins=bins_n, barmode="overlay", opacity=0.65,
                           color_discrete_sequence=JCOLORS,
                           title="Vehicle Count Distribution by Junction",
                           labels={"Vehicles":"Vehicles/hr"})
        fig.update_layout(**PTMPL, height=360)
        st.plotly_chart(fig, width='stretch')

    with c2:
        fig = px.box(fdf, x="Junction", y="Vehicles",
                     color="Junction", color_discrete_sequence=JCOLORS,
                     title="Box Plot — Vehicle Count by Junction",
                     points="outliers")
        fig.update_layout(**PTMPL, height=360, showlegend=False)
        st.plotly_chart(fig, width='stretch')

    c3, c4 = st.columns(2)

    with c3:
        # Violin
        fig = px.violin(fdf, x="congestion", y="Vehicles", color="congestion",
                        color_discrete_map=CCOLORS, box=True,
                        category_orders={"congestion":["Free Flow","Moderate","Heavy","Severe"]},
                        title="Violin: Vehicles by Congestion Level")
        fig.update_layout(**PTMPL, height=340, showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with c4:
        # Scatter: hour vs vehicles with congestion colour
        sample = fdf.sample(min(3000, len(fdf)), random_state=42)
        fig = px.scatter(sample, x="hour", y="Vehicles", color="congestion",
                         color_discrete_map=CCOLORS, opacity=0.5,
                         title="Hour vs Vehicles (sample)",
                         labels={"hour":"Hour of Day"},
                         category_orders={"congestion":["Free Flow","Moderate","Heavy","Severe"]})
        fig.update_layout(**PTMPL, height=340,
            legend=dict(bgcolor="#161b22", bordercolor="#30363d"))
        st.plotly_chart(fig, width='stretch')

    # Stats table
    st.markdown("#### Descriptive Statistics")
    stats = fdf.groupby("Junction")["Vehicles"].agg(
        Count="count", Mean="mean", Std="std",
        Min="min", P25=lambda x:x.quantile(.25),
        Median="median", P75=lambda x:x.quantile(.75), Max="max"
    ).round(2)
    st.dataframe(stats, width='stretch')

# ═══════════════════════════════════════════════════════════════
# TAB 5 — FORECAST (heuristic autoregressive)
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("#### 🔮 Heuristic Autoregressive Forecast")
    st.caption(
        "Uses a weighted blend of recent history + hourly seasonal profile "
        "to project ahead — no GPU required. For full LSTM results, run the notebook."
    )

    jf = df[df["Junction"]==forecast_junction].set_index("DateTime")["Vehicles"].resample("h").mean().dropna()

    # Seasonal profile (mean by hour × day-of-week)
    prof = df[df["Junction"]==forecast_junction].groupby(["dayofweek","hour"])["Vehicles"].mean()

    # Build forecast
    seed = jf.iloc[-48:].values.copy()
    last_dt = jf.index[-1]
    preds, times = [], []

    for h in range(forecast_hours):
        target_dt = last_dt + pd.Timedelta(hours=h+1)
        seasonal  = prof.get((target_dt.dayofweek, target_dt.hour), seed[-1])
        ma3       = np.mean(seed[-3:])
        ma24      = np.mean(seed[-24:]) if len(seed) >= 24 else seasonal
        # Weighted blend
        pred = 0.35*ma3 + 0.30*ma24 + 0.35*seasonal + np.random.normal(0, 0.8)
        pred = max(0, pred)
        preds.append(pred)
        times.append(target_dt)
        seed = np.append(seed, pred)

    forecast_df = pd.DataFrame({
        "DateTime": times,
        "Predicted": np.round(preds, 1),
    })

    t_thresh = thresholds[forecast_junction]
    def cong_label(v):
        if   v <= t_thresh["p33"]: return "Free Flow"
        elif v <= t_thresh["p66"]: return "Moderate"
        elif v <= t_thresh["p90"]: return "Heavy"
        else:                      return "Severe"
    forecast_df["Congestion"] = forecast_df["Predicted"].apply(cong_label)

    # Plot
    hist72 = jf.iloc[-72:].reset_index()
    hist72.columns = ["DateTime","Vehicles"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.04)

    fig.add_trace(go.Scatter(
        x=hist72["DateTime"], y=hist72["Vehicles"],
        name="Historical", line=dict(color="#8b949e", width=1.5),
        fill="tozeroy", fillcolor="rgba(139,148,158,0.08)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=forecast_df["DateTime"], y=forecast_df["Predicted"],
        name="Forecast", line=dict(color="#58a6ff", width=2.5, dash="dot"),
        mode="lines+markers", marker=dict(size=4, color="#58a6ff")
    ), row=1, col=1)

    fig.add_vline(x=str(last_dt), line_color="#e3b341", line_dash="dash",
                  line_width=1.5, row=1, col=1)

    for _, row in forecast_df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["DateTime"]], y=[1],
            marker_color=CCOLORS[row["Congestion"]],
            showlegend=False, hovertext=row["Congestion"],
            hovertemplate="%{x}<br>" + row["Congestion"] + "<extra></extra>"
        ), row=2, col=1)

    fig.update_layout(**PTMPL, height=500,
        title=dict(text=f"Junction {forecast_junction} — {forecast_hours}h Forecast",
                   font=dict(family="Space Mono", size=14)),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        yaxis2=dict(showticklabels=False, gridcolor="#21262d"))
    fig.update_yaxes(title_text="Vehicles/hr", row=1, col=1)
    fig.update_yaxes(title_text="Level", row=2, col=1)

    st.plotly_chart(fig, width='stretch')

    # Summary cards
    st.markdown("#### Forecast Summary")
    fc_counts = forecast_df["Congestion"].value_counts()
    col_badges = st.columns(4)
    badge_classes = {"Free Flow":"badge-ff","Moderate":"badge-mod","Heavy":"badge-hvy","Severe":"badge-sev"}
    for i, level in enumerate(["Free Flow","Moderate","Heavy","Severe"]):
        cnt = fc_counts.get(level, 0)
        with col_badges[i]:
            st.markdown(
                f'<div style="text-align:center;padding:14px;background:#161b22;border:1px solid #30363d;border-radius:10px">'
                f'<span class="badge {badge_classes[level]}">{level}</span><br>'
                f'<span style="font-family:Space Mono,monospace;font-size:1.8rem;color:#e6edf3">{cnt}</span><br>'
                f'<span style="color:#8b949e;font-size:12px">hours</span></div>',
                unsafe_allow_html=True
            )

    st.markdown("#### Detailed Forecast Table")
    st.dataframe(
        forecast_df.style.map(
            lambda v: f"color:{CCOLORS.get(v,'#e6edf3')};font-weight:bold",
            subset=["Congestion"]
        ),
        width='stretch', height=300
    )

# ═══════════════════════════════════════════════════════════════
# TAB 6 — RAW DATA
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.markdown(f"#### Filtered Dataset — {len(fdf):,} rows")

    c1, c2 = st.columns([3,1])
    with c2:
        sort_col = st.selectbox("Sort by", ["DateTime","Junction","Vehicles","congestion"])
        ascending = st.radio("Order", ["Ascending","Descending"]) == "Ascending"
        n_rows    = st.slider("Rows to show", 50, 500, 100)

    display_df = (fdf[["DateTime","Junction","Vehicles","congestion","hour","dayofweek","is_peak","is_weekend"]]
                  .sort_values(sort_col, ascending=ascending)
                  .head(n_rows))

    st.dataframe(
        display_df.style.map(
            lambda v: f"color:{CCOLORS.get(v,'inherit')};font-weight:600"
                      if isinstance(v,str) and v in CCOLORS else "",
            subset=["congestion"]
        ),
        width='stretch', height=400
    )

    # Download
    csv_bytes = fdf.to_csv(index=False).encode()
    st.download_button("⬇  Download filtered CSV", csv_bytes,
                       file_name="traffic_filtered.csv", mime="text/csv")

    st.markdown("#### Congestion Thresholds by Junction")
    thresh_rows = []
    for j, t in thresholds.items():
        if j in sel_junctions:
            thresh_rows.append({
                "Junction": j,
                "Free Flow  ≤": round(t["p33"],1),
                "Moderate   ≤": round(t["p66"],1),
                "Heavy      ≤": round(t["p90"],1),
                "Severe     > ": round(t["p90"],1),
            })
    st.dataframe(pd.DataFrame(thresh_rows), width='stretch')
