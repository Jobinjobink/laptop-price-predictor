import pandas as pd

from src.features import build_features


def test_feature_extraction():
    raw = pd.DataFrame([{
        "Company": "Dell", "TypeName": "Notebook", "Inches": 15.6,
        "ScreenResolution": "IPS Panel Full HD / Touchscreen 1920x1080",
        "Cpu": "Intel Core i7 7700HQ 2.8GHz", "Ram": "16GB",
        "Memory": "256GB SSD +  1TB HDD", "Gpu": "Nvidia GeForce GTX 1060",
        "OpSys": "Windows 10", "Weight": "2.65kg",
    }])
    result = build_features(raw).iloc[0]
    assert result["cpu_family"] == "Intel Core i7"
    assert result["gpu_brand"] == "Nvidia"
    assert result["ssd_gb"] == 256
    assert result["hdd_gb"] == 1024
    assert result["touchscreen"] == 1
    assert result["screen_width"] == 1920
