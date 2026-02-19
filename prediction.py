import logging, time, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

ASSETS = {
    "Gold":   {"symbol": "GC=F", "color": "#B45309"},   # amber-700
    "Silver": {"symbol": "SI=F", "color": "#4B5563"},   # gray-600
}
TIME_RANGES = {
    "1 Hour":   {"period": "1d",  "interval": "1m"},
    "1 Day":    {"period": "5d",  "interval": "5m"},
    "1 Week":   {"period": "1mo", "interval": "30m"},
    "1 Month":  {"period": "3mo", "interval": "1h"},
    "3 Months": {"period": "6mo", "interval": "1d"},
    "6 Months": {"period": "1y",  "interval": "1d"},
    "1 Year":   {"period": "2y",  "interval": "1wk"},
    "5 Years":  {"period": "10y", "interval": "1mo"},
}
MODEL_COLORS = {
    "Actual":       "#0F172A",
    "Ridge":        "#0284C7",
    "SVR":          "#DC2626",
    "RandomForest": "#16A34A",
    "GBM":          "#D97706",
    "MLP":          "#7C3AED",
    "XGBoost":      "#EA580C",
    "Ensemble":     "#DB2777",
}
MODEL_NAMES = ["Ridge", "SVR", "RandomForest", "GBM", "MLP", "XGBoost"]
BO_CALLS, BO_INIT, MAX_TRAIN_ROWS = 8, 3, 400

# ── Light-theme Plotly layout base
# NOTE: 'margin' is intentionally NOT included here to avoid duplicate-keyword
# errors when individual chart functions pass their own margin values.
BG      = "#F8FAFC"   # slate-50
GRID    = "#E2E8F0"   # slate-200
PAPER   = "#FFFFFF"

_LAY_BASE = dict(
    paper_bgcolor=PAPER,
    plot_bgcolor=BG,
    font=dict(family="Inter, sans-serif", color="#1E293B"),
    hovermode="x unified",
    legend=dict(
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#CBD5E1",
        borderwidth=1,
        font=dict(size=10, color="#1E293B"),
    ),
    # margin deliberately omitted — each chart sets its own
)

def _ax(title=""):
    """Consistent light-theme axis dict."""
    d = dict(
        gridcolor=GRID,
        zerolinecolor="#CBD5E1",
        zerolinewidth=1,
        linecolor="#CBD5E1",
        tickfont=dict(color="#475569"),
    )
    if title:
        d["title"] = dict(text=title, font=dict(color="#334155"))
    return d

# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MetalPulse Analytics", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Share+Tech+Mono&family=Inter:wght@300;400;500;600&display=swap');

/* ── CSS variables — light palette ── */
:root {
    --bg:       #F1F5F9;
    --card:     #FFFFFF;
    --card2:    #F8FAFC;
    --border:   #CBD5E1;
    --border2:  #E2E8F0;
    --green:    #16A34A;
    --orange:   #EA580C;
    --red:      #DC2626;
    --blue:     #0284C7;
    --purple:   #7C3AED;
    --pink:     #DB2777;
    --amber:    #D97706;
    /* text */
    --t1:       #0F172A;
    --t2:       #1E293B;
    --t3:       #334155;
    --t4:       #64748B;
    --t5:       #94A3B8;
}

html, body, [class*="css"] {
    background: var(--bg) !important;
    color: var(--t1) !important;
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1rem .85rem; }
[data-testid="stSidebar"] * { color: var(--t2) !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p { color: var(--t3) !important; }

/* ── Main container ── */
.main .block-container { padding: 1rem 1.8rem 2rem; max-width: 100%; }

/* ── Header card ── */
.hdr {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 2rem 1.2rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 6px rgba(0,0,0,.06);
}
.hdr::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--green), var(--orange), var(--green));
}
.hdr-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem; font-weight: 700; letter-spacing: 3px;
    margin: 0; line-height: 1.1;
}
.hdr-title .word-metal  { color: var(--green);  }
.hdr-title .word-pulse  { color: var(--orange); }
.hdr-title .word-rest   { color: var(--t2);     }
.hdr-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: .58rem; color: var(--t4); letter-spacing: 2.5px; margin-top: 4px;
}
.price-val {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem; font-weight: 700; line-height: 1; color: var(--t1);
}

/* ── Prediction / Metric cards ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .85rem .9rem;
    text-align: center;
    transition: transform .18s, box-shadow .2s;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 18px rgba(0,0,0,.1);
}
.card-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: .55rem; letter-spacing: 2px;
    color: var(--t4); text-transform: uppercase; margin-bottom: 5px;
}
.card-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.45rem; font-weight: 700; line-height: 1.1;
}
.card-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: .6rem; color: var(--t4); margin-top: 2px;
}

/* ── Badges ── */
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 99px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .58rem; letter-spacing: 1.5px; font-weight: 600;
}
.live    { background: #DCFCE7; color: #15803D; border: 1px solid #86EFAC; }
.demo    { background: #FEE2E2; color: #B91C1C; border: 1px solid #FCA5A5; }
.trained { background: #DBEAFE; color: #1D4ED8; border: 1px solid #93C5FD; }

/* ── Section headers ── */
.sec {
    font-family: 'Rajdhani', sans-serif;
    font-size: .9rem; font-weight: 700; letter-spacing: 3px;
    color: var(--t3); text-transform: uppercase;
    border-bottom: 1px solid var(--border2);
    padding-bottom: 5px; margin: 1.2rem 0 .75rem;
    display: flex; align-items: center; gap: 8px;
}

/* ── Info / patience banner ── */
.patience-banner {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-left: 4px solid var(--green);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin: 1rem 0;
    font-family: 'Inter', sans-serif;
    font-size: .82rem;
    color: var(--t2);
    line-height: 1.7;
}
.patience-banner .banner-title {
    font-weight: 600; color: var(--t1);
    font-size: .88rem; margin-bottom: .45rem;
}
.patience-banner ul {
    margin: 0 0 .6rem 0;
    padding-left: 1.2rem;
    list-style: none;
}
.patience-banner ul li {
    position: relative;
    padding-left: 1.6rem;
    margin-bottom: .2rem;
}
.patience-banner ul li::before {
    content: '▸';
    position: absolute; left: 0;
    color: var(--green); font-size: .8rem;
}
.step {
    display: inline-block;
    background: #DCFCE7;
    color: var(--green);
    border-radius: 4px;
    padding: 1px 7px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .65rem;
    letter-spacing: 1px;
    margin-right: 4px;
    font-weight: 700;
}
.patience-footer {
    font-size: .78rem; color: var(--t3);
    border-top: 1px solid #BBF7D0; margin-top: .6rem; padding-top: .5rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card2); border-radius: 8px 8px 0 0;
    padding: 5px; gap: 4px; border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace; font-size: .65rem;
    letter-spacing: 1.5px; text-transform: uppercase;
    background: transparent; border-radius: 6px;
    color: var(--t4); padding: 7px 16px; transition: color .2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(22,163,74,.12), rgba(234,88,12,.08));
    color: var(--t1) !important;
    font-weight: 600;
}

/* ── Streamlit overrides ── */
div[data-testid="stSpinner"] > div { color: var(--green) !important; }
.stButton > button {
    background: linear-gradient(135deg, rgba(22,163,74,.08), rgba(234,88,12,.06)) !important;
    border: 1px solid var(--border) !important;
    color: var(--t2) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: .65rem !important;
    letter-spacing: 1.5px !important;
    border-radius: 8px !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    border-color: var(--green) !important;
    color: var(--green) !important;
}
.stSelectbox label, .stSlider label, .stToggle label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: .6rem !important;
    color: var(--t3) !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}
div[data-baseweb="select"] > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--t1) !important;
}
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px; }

/* ── Footer ── */
.footer {
    text-align: center; padding: 2rem 0 1rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: .55rem; color: var(--t5); letter-spacing: 2px;
    border-top: 1px solid var(--border2); margin-top: 2rem;
}

/* ── Keep toolbar visible ── */
[data-testid="stToolbar"]   { visibility: visible !important; }
header[data-testid="stHeader"] { visibility: visible !important; background: transparent !important; }
#MainMenu { visibility: visible !important; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────
def fetch_data(symbol, period, interval):
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.dropna(subset=["close"], inplace=True)
            return df, True
    except Exception:
        pass
    base = {"GC=F": 2350., "SI=F": 27.5}.get(symbol, 1000.)
    vol  = {"GC=F": .14,   "SI=F": .22 }.get(symbol, .15)
    n = 500; rng = np.random.default_rng(42)
    ret = (0.06 - .5 * vol ** 2) / 252 + vol / np.sqrt(252) * rng.standard_normal(n)
    prices = base * np.exp(np.cumsum(ret))
    idx = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="h")
    return pd.DataFrame({
        "open":   prices * (1 - .001),
        "high":   prices * (1 + .003),
        "low":    prices * (1 - .003),
        "close":  prices,
        "volume": rng.integers(1000, 50000, n).astype(float),
    }, index=idx), False


# ─────────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────────
def build_features(df):
    f = df[["close"]].copy()
    for lag in [1, 3, 5, 10, 20]:
        f[f"lag_{lag}"] = f["close"].shift(lag)
    for w in [5, 10, 20, 50]:
        f[f"rmean_{w}"] = f["close"].rolling(w).mean()
        f[f"rstd_{w}"]  = f["close"].rolling(w).std()
    f["ret1"]  = f["close"].pct_change(1)
    f["ret5"]  = f["close"].pct_change(5)
    f["ret20"] = f["close"].pct_change(20)
    f["vol5"]  = f["ret1"].rolling(5).std()
    f["vol20"] = f["ret1"].rolling(20).std()
    for span in [5, 12, 26]:
        f[f"ema{span}"] = f["close"].ewm(span=span, adjust=False).mean()
    e12 = f["close"].ewm(span=12, adjust=False).mean()
    e26 = f["close"].ewm(span=26, adjust=False).mean()
    f["macd"]   = e12 - e26
    f["macd_s"] = f["macd"].ewm(span=9, adjust=False).mean()
    d = f["close"].diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    f["rsi"]    = 100 - 100 / (1 + g / (l + 1e-9))
    bm = f["close"].rolling(20).mean()
    bs = f["close"].rolling(20).std()
    f["bb_pos"] = (f["close"] - bm) / (bs + 1e-9)
    if "high"   in df.columns:
        f["hl"] = df["high"] - df["low"]
        f["oc"] = df["close"] - df["open"]
    if "volume" in df.columns:
        f["vol"]    = df["volume"]
        f["vol_r5"] = df["volume"].rolling(5).mean()
    f["target"] = f["close"].shift(-1)
    f.dropna(inplace=True)
    return f


def temporal_split(feat, test_ratio=0.2):
    X = feat.drop(columns=["target"])
    y = feat["target"]
    split = int(len(X) * (1 - test_ratio))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# ─────────────────────────────────────────────────────────────────
# BAYESIAN OPTIMISATION
# ─────────────────────────────────────────────────────────────────
def bo_optimize(objective, space, n_calls=BO_CALLS, n_init=BO_INIT):
    try:
        from skopt import gp_minimize
        from skopt.callbacks import DeltaXStopper
        res = gp_minimize(
            objective, space, n_calls=n_calls, n_initial_points=n_init,
            acq_func="EI", random_state=42, noise=1e-8,
            callback=[DeltaXStopper(0.01)], verbose=False,
        )
        return res.x, res.fun
    except ImportError:
        pass
    rng = np.random.default_rng(42)
    best_p, best_s = None, np.inf
    for _ in range(n_calls):
        try:
            params = [d.rvs(random_state=int(rng.integers(9999)))[0] for d in space]
        except Exception:
            params = space
        s = objective(params)
        if s < best_s:
            best_s, best_p = s, params
    return best_p, best_s


# ─────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def _m(yt, yp):
    mse = mean_squared_error(yt, yp)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae":  float(mean_absolute_error(yt, yp)),
        "r2":   float(r2_score(yt, yp)),
        "mape": float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-9))) * 100),
    }


def _cap(a, n=MAX_TRAIN_ROWS):
    return a[-n:] if len(a) > n else a


def train_ridge(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import Ridge
    from skopt.space import Real
    sc = RobustScaler()
    Xs, Xts = sc.fit_transform(Xtr), sc.transform(Xte)
    Xc, yc  = _cap(Xs), _cap(ytr)
    def obj(p):
        m = Ridge(alpha=p[0]); m.fit(Xc, yc)
        return mean_squared_error(yte, m.predict(Xts))
    bp, _ = bo_optimize(obj, [Real(1e-3, 1e3, prior="log-uniform", name="a")])
    m = Ridge(alpha=bp[0]); m.fit(Xs, ytr)
    p = m.predict(Xts)
    return sc, m, p, _m(yte, p)


def train_svr(Xtr, ytr, Xte, yte):
    from sklearn.svm import SVR
    from skopt.space import Real
    sc = RobustScaler()
    Xs, Xts = sc.fit_transform(Xtr), sc.transform(Xte)
    Xc, yc  = Xs[-300:], ytr[-300:]
    def obj(p):
        m = SVR(C=p[0], epsilon=p[1], gamma="scale"); m.fit(Xc, yc)
        return mean_squared_error(yte, m.predict(Xts))
    bp, _ = bo_optimize(obj, [
        Real(0.1, 100, prior="log-uniform", name="C"),
        Real(1e-3, 1., prior="log-uniform", name="e"),
    ])
    m = SVR(C=bp[0], epsilon=bp[1], gamma="scale"); m.fit(Xc, yc)
    p = m.predict(Xts)
    return sc, m, p, _m(yte, p)


def train_rf(Xtr, ytr, Xte, yte):
    from sklearn.ensemble import RandomForestRegressor
    from skopt.space import Integer
    sc = RobustScaler()
    Xs, Xts = sc.fit_transform(Xtr), sc.transform(Xte)
    Xc, yc  = _cap(Xs), _cap(ytr)
    def obj(p):
        m = RandomForestRegressor(n_estimators=int(p[0]), max_depth=int(p[1]),
                                  random_state=42, n_jobs=-1)
        m.fit(Xc, yc)
        return mean_squared_error(yte, m.predict(Xts))
    bp, _ = bo_optimize(obj, [Integer(30, 150, name="n"), Integer(3, 12, name="d")])
    m = RandomForestRegressor(n_estimators=int(bp[0]), max_depth=int(bp[1]),
                              random_state=42, n_jobs=-1)
    m.fit(Xs, ytr); p = m.predict(Xts)
    return sc, m, p, _m(yte, p)


def train_gbm(Xtr, ytr, Xte, yte):
    from sklearn.ensemble import GradientBoostingRegressor
    from skopt.space import Integer, Real
    sc = RobustScaler()
    Xs, Xts = sc.fit_transform(Xtr), sc.transform(Xte)
    Xc, yc  = _cap(Xs), _cap(ytr)
    def obj(p):
        m = GradientBoostingRegressor(n_estimators=int(p[0]), learning_rate=float(p[1]),
                                      max_depth=int(p[2]), random_state=42)
        m.fit(Xc, yc)
        return mean_squared_error(yte, m.predict(Xts))
    bp, _ = bo_optimize(obj, [
        Integer(50, 200, name="n"), Real(0.05, 0.3, name="lr"), Integer(2, 6, name="d"),
    ])
    m = GradientBoostingRegressor(n_estimators=int(bp[0]), learning_rate=float(bp[1]),
                                  max_depth=int(bp[2]), random_state=42)
    m.fit(Xs, ytr); p = m.predict(Xts)
    return sc, m, p, _m(yte, p)


def train_mlp(Xtr, ytr, Xte, yte):
    from sklearn.neural_network import MLPRegressor
    from skopt.space import Integer, Real
    sc = RobustScaler()
    Xs, Xts = sc.fit_transform(Xtr), sc.transform(Xte)
    Xc, yc  = _cap(Xs), _cap(ytr)
    def obj(p):
        m = MLPRegressor(hidden_layer_sizes=(int(p[0]), int(p[0])), alpha=float(p[1]),
                         max_iter=100, random_state=42, early_stopping=True, n_iter_no_change=8)
        m.fit(Xc, yc)
        return mean_squared_error(yte, m.predict(Xts))
    bp, _ = bo_optimize(obj, [Integer(32, 128, name="sz"), Real(1e-4, 0.1, name="a")])
    m = MLPRegressor(hidden_layer_sizes=(int(bp[0]), int(bp[0])), alpha=float(bp[1]),
                     max_iter=300, random_state=42, early_stopping=True, n_iter_no_change=15)
    m.fit(Xs, ytr); p = m.predict(Xts)
    return sc, m, p, _m(yte, p)


def train_xgb(Xtr, ytr, Xte, yte):
    from skopt.space import Integer, Real
    sc = RobustScaler()
    Xs, Xts = sc.fit_transform(Xtr), sc.transform(Xte)
    Xc, yc  = _cap(Xs), _cap(ytr)
    try:
        from xgboost import XGBRegressor
        def obj(p):
            m = XGBRegressor(n_estimators=int(p[0]), learning_rate=float(p[1]),
                             max_depth=int(p[2]), random_state=42, verbosity=0, n_jobs=-1)
            m.fit(Xc, yc)
            return mean_squared_error(yte, m.predict(Xts))
        bp, _ = bo_optimize(obj, [
            Integer(50, 200, name="n"), Real(0.05, 0.3, name="lr"), Integer(2, 6, name="d"),
        ])
        m = XGBRegressor(n_estimators=int(bp[0]), learning_rate=float(bp[1]),
                         max_depth=int(bp[2]), random_state=42, verbosity=0, n_jobs=-1)
    except ImportError:
        from sklearn.ensemble import ExtraTreesRegressor
        def obj(p):
            m = ExtraTreesRegressor(n_estimators=int(p[0]), max_depth=int(p[1]),
                                    random_state=42, n_jobs=-1)
            m.fit(Xc, yc)
            return mean_squared_error(yte, m.predict(Xts))
        bp, _ = bo_optimize(obj, [Integer(50, 200, name="n"), Integer(3, 12, name="d")])
        m = ExtraTreesRegressor(n_estimators=int(bp[0]), max_depth=int(bp[1]),
                                random_state=42, n_jobs=-1)
    m.fit(Xs, ytr); p = m.predict(Xts)
    return sc, m, p, _m(yte, p)


def train_ensemble(sms, Xte, yte):
    bp = np.stack([m.predict(sc.transform(Xte)) for sc, m in sms], axis=1)
    n  = len(sms); w = np.ones(n) / n; p = bp @ w
    return w, p, _m(np.array(yte), p)


def predict_next_all(sms, w, Xte):
    last   = Xte.iloc[[-1]]
    indiv  = [float(m.predict(sc.transform(last))[0]) for sc, m in sms]
    return indiv, float(np.array(indiv) @ w)


@st.cache_resource(show_spinner=False)
def get_trained_models(asset_name, time_range_label, data_hash):
    tr     = TIME_RANGES[time_range_label]
    symbol = ASSETS[asset_name]["symbol"]
    df, _  = fetch_data(symbol, tr["period"], tr["interval"])
    feat   = build_features(df)
    if len(feat) < 50:
        return None
    Xtr, Xte, ytr, yte = temporal_split(feat)
    ytr_arr, yte_arr   = ytr.values, yte.values
    trainers = [train_ridge, train_svr, train_rf, train_gbm, train_mlp, train_xgb]
    results  = [None] * len(trainers)

    def _run(args):
        idx, fn = args
        return idx, fn(Xtr, ytr_arr, Xte, yte_arr)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_run, (i, fn)): i for i, fn in enumerate(trainers)}
        for future in as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
            except Exception as e:
                logging.warning(f"Model {futures[future]} failed: {e}")

    results = [r for r in results if r is not None]
    sms     = [(r[0], r[1]) for r in results]
    ew, ep, em = train_ensemble(sms, Xte, yte_arr)
    return {
        "df": df, "feat": feat, "Xte": Xte, "yte": yte_arr, "yte_idx": yte.index,
        "results": results, "ens_w": ew, "ens_preds": ep, "ens_met": em,
        "scalers_models": sms,
    }


# ─────────────────────────────────────────────────────────────────
# CHARTS
# Each chart passes its own `margin` to update_layout so there is
# no collision with the shared _LAY_BASE dict.
# ─────────────────────────────────────────────────────────────────
def chart_predictions(cache, asset_name, time_range_label):
    df, idx = cache["df"], cache["yte_idx"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"], name="Actual",
        line=dict(color="#0F172A", width=2), mode="lines",
    ))
    mclrs = list(MODEL_COLORS.values())[1:-1]
    for i, (name, clr) in enumerate(zip(MODEL_NAMES, mclrs)):
        if i >= len(cache["results"]):
            continue
        fig.add_trace(go.Scatter(
            x=idx, y=cache["results"][i][2], name=name,
            line=dict(color=clr, width=1.4), mode="lines", opacity=0.85,
        ))
    fig.add_trace(go.Scatter(
        x=idx, y=cache["ens_preds"], name="Ensemble",
        line=dict(color=MODEL_COLORS["Ensemble"], width=2.5, dash="dot"),
        mode="lines", opacity=0.95,
    ))
    fig.update_layout(
        **_LAY_BASE,
        title=dict(
            text=f"<b>{asset_name} — {time_range_label} · Predictions vs Actual</b>",
            font=dict(size=15, color="#0F172A"), x=0.01,
        ),
        height=500,
        margin=dict(l=60, r=160, t=55, b=50),
        xaxis=_ax("Date / Time"),
        yaxis=_ax("Price  (USD / oz)"),
    )
    return fig


def chart_candlestick(df, asset_name, time_range_label):
    fig = go.Figure(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="#16A34A",
        decreasing_line_color="#DC2626",
    ))
    fig.update_layout(
        **_LAY_BASE,
        title=dict(
            text=f"<b>{asset_name} OHLC — {time_range_label}</b>",
            font=dict(size=14, color="#0F172A"), x=0.01,
        ),
        height=420,
        margin=dict(l=60, r=40, t=55, b=50),
        xaxis_rangeslider_visible=False,
        xaxis=_ax("Date / Time"),
        yaxis=_ax("Price  (USD / oz)"),
    )
    return fig


def chart_metrics(all_metrics, metric="rmse"):
    names  = list(all_metrics.keys())
    values = [all_metrics[n].get(metric, 0) for n in names]
    colors = [MODEL_COLORS.get(n, "#888") for n in names]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(color="#334155", size=10),
    ))
    fig.update_layout(
        **_LAY_BASE,
        title=dict(
            text=f"<b>{metric.upper()} by Model</b>",
            font=dict(size=13, color="#0F172A"), x=0.01,
        ),
        height=310,
        # NOTE: chart_metrics uses its own margin — NOT the shared one
        margin=dict(l=120, r=100, t=45, b=35),
        xaxis=_ax(metric.upper()),
        yaxis=_ax(),
    )
    return fig


def chart_comparison(g_df, s_df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=g_df.index, y=g_df["close"], name="Gold",
        line=dict(color="#B45309", width=1.8),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=s_df.index, y=s_df["close"], name="Silver",
        line=dict(color="#6B7280", width=1.8),
    ), secondary_y=True)
    fig.update_layout(
        **_LAY_BASE,
        title=dict(
            text="<b>Gold vs Silver — Price Comparison</b>",
            font=dict(size=13, color="#0F172A"), x=0.01,
        ),
        height=300,
        margin=dict(l=60, r=80, t=50, b=40),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(
        title_text="Gold (USD/oz)",
        gridcolor=GRID, zerolinecolor=GRID,
        tickfont=dict(color="#475569"),
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Silver (USD/oz)",
        secondary_y=True,
        showgrid=False,
        tickfont=dict(color="#475569"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:.8rem 0 1.4rem;">
      <div style="font-family:'Rajdhani',sans-serif; font-size:1.6rem; font-weight:700;
                  letter-spacing:3px; line-height:1.1;">
        <span style="color:#16A34A;">METAL</span><span style="color:#EA580C;">PULSE</span>
      </div>
      <div style="font-family:'Share Tech Mono',monospace; font-size:.55rem;
                  color:#64748B; letter-spacing:2px; margin-top:4px;">
        ANALYTICS  v2.0
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-family:monospace;font-size:.65rem;color:#334155;'
                'letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;">Asset</p>',
                unsafe_allow_html=True)
    asset_name = st.selectbox("Asset", list(ASSETS.keys()), label_visibility="collapsed")

    st.markdown('<p style="font-family:monospace;font-size:.65rem;color:#334155;'
                'letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;margin-top:8px;">Time Range</p>',
                unsafe_allow_html=True)
    time_range = st.selectbox("Time Range", list(TIME_RANGES.keys()), index=1,
                              label_visibility="collapsed")

    st.divider()
    auto_refresh = st.toggle("🔄  Auto Refresh", value=False)
    refresh_sec  = st.slider("Interval (s)", 30, 300, 60, step=10) if auto_refresh else None
    force_retrain = st.button("⚡  Force Retrain All Models", use_container_width=True)

    st.divider()
    st.markdown('<p style="font-family:monospace;font-size:.6rem;color:#334155;'
                'letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">Model Legend</p>',
                unsafe_allow_html=True)
    for name, color in MODEL_COLORS.items():
        st.markdown(
            f'<span style="color:{color}; font-size:1rem;">●</span>&nbsp;'
            f'<span style="font-family:monospace; font-size:.72rem; color:#334155;">{name}</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        '<div style="font-family:monospace;font-size:.55rem;color:#94A3B8;'
        'text-align:center;line-height:1.9;">'
        'Bayesian Optimisation<br>Gaussian Process · EI Acquisition<br>'
        '6 Models · ThreadPoolExecutor<br>Yahoo Finance · No API Key</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
tr = TIME_RANGES[time_range]
with st.spinner("📡  Fetching live market data…"):
    df_now, is_live = fetch_data(ASSETS[asset_name]["symbol"], tr["period"], tr["interval"])

current = float(df_now["close"].iloc[-1])
prev    = float(df_now["close"].iloc[-2]) if len(df_now) > 1 else current
delta   = current - prev
dpct    = (delta / prev * 100) if prev else 0
sign    = "+" if delta >= 0 else ""
dclr    = "#16A34A" if delta >= 0 else "#DC2626"
arrow   = "▲" if delta >= 0 else "▼"
badge   = ('<span class="badge live">● &nbsp;LIVE DATA</span>' if is_live
           else '<span class="badge demo">⚡ &nbsp;DEMO DATA</span>')
aclr    = ASSETS[asset_name]["color"]
ts      = pd.Timestamp.now().strftime("%Y-%m-%d &nbsp; %H:%M:%S &nbsp; UTC")

st.markdown(f"""
<div class="hdr">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;
              flex-wrap:wrap;gap:1.2rem;">
    <div>
      <p class="hdr-title">
        <span class="word-metal">METAL</span><span class="word-pulse">PULSE</span>
        <span class="word-rest"> &nbsp;ANALYTICS</span>
      </p>
      <p class="hdr-sub">
        BAYESIAN OPTIMISATION &nbsp;·&nbsp; 6 PARALLEL MODELS + ENSEMBLE
        &nbsp;·&nbsp; REAL-TIME PREDICTIONS
      </p>
      <div style="margin-top:.6rem;">{badge}</div>
    </div>
    <div style="text-align:right;">
      <div class="price-val">
        <span style="color:{aclr};">{asset_name}</span>
        &nbsp;&nbsp;${current:,.2f}
      </div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.88rem;
                  color:{dclr};margin-top:4px;letter-spacing:1px;">
        {arrow}&nbsp;{sign}${delta:,.2f}&nbsp;&nbsp;({sign}{dpct:.2f}%)
      </div>
      <div style="font-family:monospace;font-size:.56rem;color:#94A3B8;margin-top:6px;">
        {ts}
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PATIENCE BANNER — properly formatted bullet list
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="patience-banner">
  <div class="banner-title">⏳ First-time load? The engine is warming up.</div>
  <ul>
    <li><span class="step">01 FETCH</span> Pulling live OHLCV data from Yahoo Finance.</li>
    <li><span class="step">02 FEATURES</span> Engineering 30+ technical indicators (RSI, MACD, Bollinger Bands, rolling stats…).</li>
    <li><span class="step">03 BO</span> Running Gaussian Process optimisation per model to find the best hyperparameters.</li>
    <li><span class="step">04 PARALLEL</span> All 6 models (Ridge, SVR, Random Forest, GBM, MLP, XGBoost) train simultaneously.</li>
    <li><span class="step">05 ENSEMBLE</span> Combining individual predictions into a single consensus forecast.</li>
  </ul>
  <div class="patience-footer">
    ⚡ Usually <strong>20–60 seconds</strong> total on first run.
    Results are <strong>cached</strong> — subsequent loads are instant until you change asset, time range, or force retrain.
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────
data_hash = f"{current:.2f}_{time_range}"
if force_retrain:
    st.cache_resource.clear()

with st.spinner("🤖  Bayesian Optimisation in progress — training all 6 models in parallel…"):
    cache = get_trained_models(asset_name, time_range, data_hash)

if cache is None:
    st.warning("⚠️  Not enough data to train. Try a longer time range (1 Month or more).")
    st.stop()

n_ok = len(cache["results"])
st.markdown(
    f'<span class="badge trained">✓ &nbsp;{n_ok} / 6 models trained &nbsp;+&nbsp; ensemble</span>'
    f'&nbsp;&nbsp;<span style="font-family:monospace;font-size:.6rem;color:#64748B;">'
    f'{len(cache["df"])} bars &nbsp;·&nbsp; {time_range}</span>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────
# PREDICTION CARDS
# ─────────────────────────────────────────────────────────────────
indiv_next, ens_next = predict_next_all(cache["scalers_models"], cache["ens_w"], cache["Xte"])
all_names = MODEL_NAMES[:len(indiv_next)] + ["Ensemble"]
all_next  = indiv_next + [ens_next]

st.markdown('<div class="sec"><span style="color:#0284C7">▸</span> Next Price Predictions</div>',
            unsafe_allow_html=True)
cols = st.columns(len(all_names))
for col, name, val in zip(cols, all_names, all_next):
    clr    = MODEL_COLORS.get(name, "#888")
    d      = val - current
    s      = "+" if d >= 0 else ""
    dc     = "#16A34A" if d >= 0 else "#DC2626"
    is_ens = (name == "Ensemble")
    bw     = "2px" if is_ens else "1px"
    bg     = ("background:linear-gradient(135deg,rgba(219,39,119,.06),"
              "rgba(124,58,237,.04));" if is_ens else "")
    with col:
        st.markdown(
            f'<div class="card" style="border:{bw} solid {clr}66;{bg}">'
            f'<div class="card-label" style="color:{clr};">{name}</div>'
            f'<div class="card-value" style="color:{clr};">${val:,.2f}</div>'
            f'<div class="card-sub" style="color:{dc};">{s}${d:,.2f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()


# ─────────────────────────────────────────────────────────────────
# METRICS CARDS
# ─────────────────────────────────────────────────────────────────
all_metrics = {
    n: cache["results"][i][3]
    for i, n in enumerate(MODEL_NAMES)
    if i < len(cache["results"])
}
all_metrics["Ensemble"] = cache["ens_met"]

st.markdown(
    '<div class="sec"><span style="color:#D97706">▸</span> Model Performance '
    '<span style="font-size:.7rem;color:#94A3B8;font-weight:400;letter-spacing:1px;">'
    '(test set)</span></div>',
    unsafe_allow_html=True,
)
mcols = st.columns(len(all_metrics))
for col, (name, m) in zip(mcols, all_metrics.items()):
    clr = MODEL_COLORS.get(name, "#888")
    r2c = "#16A34A" if m["r2"] > 0.9 else "#D97706" if m["r2"] > 0.7 else "#DC2626"
    with col:
        st.markdown(
            f'<div class="card" style="border-color:{clr}55;">'
            f'<div class="card-label" style="color:{clr};">{name}</div>'
            f'<div class="card-value" style="color:{clr};">{m["rmse"]:.2f}</div>'
            f'<div class="card-sub">RMSE</div>'
            f'<div style="font-family:monospace;font-size:.6rem;color:{r2c};margin-top:5px;">'
            f'R² {m["r2"]:.3f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()


# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Predictions", "🕯  Candlestick", "📊  Metrics", "🗂  Raw Data",
])

with tab1:
    try:
        g_df, _ = fetch_data("GC=F", tr["period"], tr["interval"])
        s_df, _ = fetch_data("SI=F", tr["period"], tr["interval"])
        st.plotly_chart(chart_comparison(g_df, s_df), use_container_width=True)
    except Exception:
        pass
    st.plotly_chart(chart_predictions(cache, asset_name, time_range), use_container_width=True)

with tab2:
    st.plotly_chart(chart_candlestick(cache["df"], asset_name, time_range), use_container_width=True)

with tab3:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(chart_metrics(all_metrics, "rmse"), use_container_width=True)
    with c2:
        st.plotly_chart(chart_metrics(all_metrics, "mae"),  use_container_width=True)
    with c3:
        st.plotly_chart(chart_metrics(all_metrics, "r2"),   use_container_width=True)
    rows = [
        {"Model": n, "RMSE": round(m["rmse"], 4), "MAE": round(m["mae"], 4),
         "R²": round(m["r2"], 4), "MAPE %": round(m["mape"], 3)}
        for n, m in all_metrics.items()
    ]
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

with tab4:
    st.dataframe(cache["df"].tail(300).style.format(precision=4), use_container_width=True)
    st.download_button(
        "⬇️  Download CSV",
        cache["df"].to_csv(),
        file_name=f"{asset_name}_{time_range.replace(' ', '_')}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  METALPULSE ANALYTICS &nbsp;v2.0 &nbsp;·&nbsp;
  BAYESIAN OPTIMISATION (GAUSSIAN PROCESS + EI) &nbsp;·&nbsp;
  DATA © YAHOO FINANCE &nbsp;·&nbsp;
  FOR INFORMATIONAL USE ONLY — NOT FINANCIAL ADVICE
</div>
""", unsafe_allow_html=True)

if auto_refresh and refresh_sec:
    time.sleep(refresh_sec)
    st.rerun()