"""Feature engineering shared by training and tests."""

from __future__ import annotations

import re
import pandas as pd


def _cpu_family(value: str) -> str:
    text = str(value)
    for pattern, label in [
        (r"Intel Core i7", "Intel Core i7"),
        (r"Intel Core i5", "Intel Core i5"),
        (r"Intel Core i3", "Intel Core i3"),
        (r"Intel Core M", "Intel Core M"),
        (r"Intel (Celeron|Pentium|Atom)", "Intel Entry"),
        (r"AMD Ryzen", "AMD Ryzen"),
        (r"AMD (A|E|FX)-?Series|AMD A\d|AMD E\d", "AMD Series"),
    ]:
        if re.search(pattern, text, flags=re.I):
            return label
    return "Other CPU"


def _gpu_brand(value: str) -> str:
    text = str(value).lower()
    if "nvidia" in text:
        return "Nvidia"
    if "amd" in text or "radeon" in text:
        return "AMD"
    if "intel" in text:
        return "Intel"
    return "Other"


def _storage_gb(value: str, kind: str) -> int:
    total = 0.0
    for part in str(value).split("+"):
        if kind.lower() not in part.lower():
            continue
        match = re.search(r"([\d.]+)\s*(TB|GB)", part, flags=re.I)
        if match:
            amount = float(match.group(1))
            total += amount * 1024 if match.group(2).upper() == "TB" else amount
    return int(total)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw dataset columns into stable, user-facing model features."""
    frame = raw.copy()
    resolution = frame["ScreenResolution"].astype(str).str.extract(r"(\d{3,4})x(\d{3,4})")
    features = pd.DataFrame(index=frame.index)
    features["company"] = frame["Company"].astype(str)
    features["type"] = frame["TypeName"].astype(str)
    features["inches"] = pd.to_numeric(frame["Inches"], errors="coerce")
    features["ram_gb"] = pd.to_numeric(frame["Ram"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    features["weight_kg"] = pd.to_numeric(frame["Weight"].astype(str).str.replace("kg", "", regex=False), errors="coerce")
    features["screen_width"] = pd.to_numeric(resolution[0], errors="coerce")
    features["screen_height"] = pd.to_numeric(resolution[1], errors="coerce")
    features["touchscreen"] = frame["ScreenResolution"].astype(str).str.contains("Touchscreen", case=False).astype(int)
    features["ips"] = frame["ScreenResolution"].astype(str).str.contains("IPS", case=False).astype(int)
    features["cpu_family"] = frame["Cpu"].map(_cpu_family)
    features["cpu_ghz"] = pd.to_numeric(frame["Cpu"].astype(str).str.extract(r"([\d.]+)GHz", flags=re.I)[0], errors="coerce")
    features["gpu_brand"] = frame["Gpu"].map(_gpu_brand)
    features["ssd_gb"] = frame["Memory"].map(lambda value: _storage_gb(value, "SSD"))
    features["hdd_gb"] = frame["Memory"].map(lambda value: _storage_gb(value, "HDD"))
    features["flash_gb"] = frame["Memory"].map(lambda value: _storage_gb(value, "Flash"))
    features["os"] = frame["OpSys"].astype(str)
    return features

