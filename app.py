"""Streamlit interface for the trained laptop price model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "laptop_price_pipeline.joblib"
META_PATH = ROOT / "artifacts" / "metadata.json"

st.set_page_config(page_title="LapValue AI", page_icon="💻", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
.stApp {background: radial-gradient(circle at 15% 5%, #18244b 0%, #080b17 38%, #05060b 100%); color:#f7f8ff; font-family:'DM Sans',sans-serif;}
[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:1180px; padding-top:2.5rem;}
.eyebrow {color:#7dd3fc; letter-spacing:.18em; font-size:.75rem; font-weight:700; text-transform:uppercase;}
.hero-title {font-size:clamp(2.6rem,6vw,5.2rem); line-height:.98; font-weight:700; letter-spacing:-.055em; margin:.4rem 0 1rem; background:linear-gradient(110deg,#fff 25%,#a5b4fc 65%,#67e8f9); -webkit-background-clip:text; color:transparent;}
.hero-copy {max-width:720px;color:#adb6d2;font-size:1.08rem;line-height:1.7;margin-bottom:2rem;}
.glass {background:linear-gradient(145deg,rgba(24,32,61,.82),rgba(10,14,29,.78));border:1px solid rgba(148,163,184,.17);border-radius:22px;padding:1.35rem 1.5rem;box-shadow:0 22px 70px rgba(0,0,0,.28);}
.price {font-size:2.8rem;font-weight:700;letter-spacing:-.04em;color:#fff;margin:.2rem 0;}
.subtle {color:#8994b3;font-size:.9rem;}
div[data-testid="stMetric"] {background:rgba(15,23,42,.58);border:1px solid rgba(148,163,184,.14);padding:1rem;border-radius:16px;}
.stButton > button {width:100%;height:3.2rem;border-radius:12px;border:0;background:linear-gradient(100deg,#6366f1,#0ea5e9);font-weight:700;color:white;}
.stButton > button:hover {box-shadow:0 0 28px rgba(56,189,248,.25);color:white;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        from train import main as train_model
        with st.spinner("Preparing the ML model for its first prediction…"):
            train_model()
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, metadata


model, meta = load_assets()

st.markdown('<div class="eyebrow">Machine-learning price intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Know the value<br>before you buy.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-copy">Configure a laptop and get an instant data-driven price estimate. LapValue AI learns relationships between hardware, brand, form factor, and historical market price.</div>', unsafe_allow_html=True)

with st.container(border=True):
    st.subheader("Configure the laptop")
    c1, c2, c3 = st.columns(3)
    with c1:
        company = st.selectbox("Brand", meta["companies"], index=meta["companies"].index("Dell") if "Dell" in meta["companies"] else 0)
        laptop_type = st.selectbox("Form factor", meta["types"])
        cpu_family = st.selectbox("Processor family", meta["cpu_families"], index=meta["cpu_families"].index("Intel Core i5") if "Intel Core i5" in meta["cpu_families"] else 0)
        cpu_ghz = st.slider("CPU clock (GHz)", 0.8, 4.2, 2.4, 0.1)
    with c2:
        ram_gb = st.select_slider("Memory (GB)", options=[2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)
        ssd_gb = st.select_slider("SSD storage (GB)", options=[0, 64, 128, 256, 512, 1024, 2048], value=256)
        hdd_gb = st.select_slider("HDD storage (GB)", options=[0, 500, 1000, 2000], value=0)
        gpu_brand = st.selectbox("Graphics", meta["gpu_brands"])
    with c3:
        inches = st.slider("Display size (inches)", 10.0, 18.4, 15.6, 0.1)
        resolution_label = st.selectbox("Display resolution", ["1366 × 768", "1920 × 1080", "2560 × 1440", "3200 × 1800", "3840 × 2160"], index=1)
        touchscreen = st.toggle("Touchscreen")
        ips = st.toggle("IPS panel", value=True)
        weight_kg = st.slider("Weight (kg)", 0.7, 4.8, 1.8, 0.05)
        os_name = st.selectbox("Operating system", meta["operating_systems"], index=meta["operating_systems"].index("Windows 10") if "Windows 10" in meta["operating_systems"] else 0)

exchange_rate = st.number_input("EUR → INR rate used for display", min_value=50.0, max_value=150.0, value=92.0, step=0.5, help="The model predicts historical euro prices. Adjust this display conversion rate if needed.")
predict = st.button("Estimate price →", type="primary")

if predict:
    width, height = [int(value.strip()) for value in resolution_label.split("×")]
    sample = pd.DataFrame([{
        "company": company, "type": laptop_type, "inches": inches,
        "ram_gb": ram_gb, "weight_kg": weight_kg, "screen_width": width,
        "screen_height": height, "touchscreen": int(touchscreen), "ips": int(ips),
        "cpu_family": cpu_family, "cpu_ghz": cpu_ghz, "gpu_brand": gpu_brand,
        "ssd_gb": ssd_gb, "hdd_gb": hdd_gb, "flash_gb": 0, "os": os_name,
    }])
    price_eur = max(float(model.predict(sample)[0]), 0)
    price_inr = price_eur * exchange_rate
    low, high = price_inr * 0.88, price_inr * 1.12
    st.markdown(f'''<div class="glass"><div class="eyebrow">Estimated historical value</div><div class="price">₹{price_inr:,.0f}</div><div class="subtle">Model estimate: €{price_eur:,.0f} · Indicative range ₹{low:,.0f}–₹{high:,.0f}</div></div>''', unsafe_allow_html=True)
    st.info("Use this as an educational estimate, not a current retail quotation. Condition, release year, taxes, and live market changes can materially affect price.")

st.divider()
st.subheader("Model at a glance")
best = meta["leaderboard"][0]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Training rows", f'{meta["train_records"]:,}')
m2.metric("Held-out rows", f'{meta["test_records"]:,}')
m3.metric("Test R²", f'{best["r2"]:.3f}')
m4.metric("Test MAE", f'€{best["mae_eur"]:,.0f}')

with st.expander("How the prediction works"):
    st.write(f"The selected **{meta['winner']}** model was chosen by lowest mean absolute error on an untouched 20% test set. The complete preprocessing and estimator are stored as one reproducible scikit-learn pipeline.")
    st.dataframe(pd.DataFrame(meta["leaderboard"]), hide_index=True, use_container_width=True)

st.caption("LapValue AI · Educational ML demonstration · Dataset contains historical European laptop listings")
