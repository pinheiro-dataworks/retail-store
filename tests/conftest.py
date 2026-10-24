"""Small synthetic panel shared by the tests (no raw data needed in CI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

STORES = {1: ("A", 200_000, "East"), 2: ("C", 40_000, "West")}


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=130, freq="7D")
    rows = []
    for store, (stype, size, region) in STORES.items():
        for dept in (1, 2):
            level = size / 4 + dept * 1_000
            for i, d in enumerate(dates):
                holiday = int(d.month == 12 and d.day >= 18)
                rows.append({
                    "store_id": store, "department": dept, "date": d,
                    "weekly_sales": level * (1.5 if holiday else 1.0) + rng.normal(0, 500) + i,
                    "is_holiday": holiday,
                    "holiday_name": "Christmas" if holiday else "None",
                    "store_type": stype, "store_size": size, "region": region,
                    "temperature": rng.uniform(20, 100), "fuel_price": 3.0, "cpi": 200.0,
                    "unemployment": 5.0, **{f"markdown_{k}": 0.0 for k in range(1, 6)},
                    "season": "Winter",
                })
    return pd.DataFrame(rows)
