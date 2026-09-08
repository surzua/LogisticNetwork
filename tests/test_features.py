"""
Pruebas unitarias para el módulo de ingeniería de características (src/data/features.py).
"""

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    add_price_promo_features,
    build_feature_pipeline,
)


@pytest.fixture
def sample_demand_df():
    """Genera un DataFrame sintético controlado para dos series durante 60 días."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    records = []
    for item in ["SKU_A", "SKU_B"]:
        for zone in ["ZONE_1"]:
            for i, d in enumerate(dates):
                records.append({
                    "date": d,
                    "item_id": item,
                    "zone_id": zone,
                    "units_sold": (i + 1) * 10,
                    "demand_target": (i + 1) * 10,
                    "price": 10.0,
                    "is_promo": 1 if i % 7 == 0 else 0,
                })
    return pd.DataFrame(records)


def test_calendar_features(sample_demand_df):
    df_feat = add_calendar_features(sample_demand_df)
    
    assert "day_of_week" in df_feat.columns
    assert "sin_dow" in df_feat.columns
    assert "cos_dow" in df_feat.columns
    assert "is_weekend" in df_feat.columns

    # Validar rangos trigonométricos
    assert df_feat["sin_dow"].between(-1.0, 1.0).all()
    assert df_feat["cos_dow"].between(-1.0, 1.0).all()

    # Validar fin de semana
    saturdays_sundays = df_feat[df_feat["day_of_week"].isin([5, 6])]
    assert (saturdays_sundays["is_weekend"] == 1).all()


def test_lag_features_no_leakage(sample_demand_df):
    lags = [7, 14]
    df_feat = add_lag_features(sample_demand_df, target_col="demand_target", lags=lags)
    
    # Filtrar un SKU específico ordenado
    sku_a = df_feat[df_feat["item_id"] == "SKU_A"].sort_values("date").reset_index(drop=True)
    
    # En t=7 (día 8), el lag_7 debe ser el valor de t=0 (día 1)
    assert sku_a.loc[7, "lag_7"] == sku_a.loc[0, "demand_target"]
    assert np.isnan(sku_a.loc[6, "lag_7"])  # t < 7 debe ser NaN
    
    # En t=14, lag_14 debe ser t=0
    assert sku_a.loc[14, "lag_14"] == sku_a.loc[0, "demand_target"]


def test_rolling_features_shift_protection(sample_demand_df):
    # Con shift=7 y ventana=7, la media en t=13 debe ser el promedio de t=0 hasta t=6
    df_feat = add_rolling_features(
        sample_demand_df, target_col="demand_target", windows=[7], shift=7
    )
    sku_a = df_feat[df_feat["item_id"] == "SKU_A"].sort_values("date").reset_index(drop=True)
    
    expected_mean_at_13 = sku_a.loc[0:6, "demand_target"].mean()
    assert np.isclose(sku_a.loc[13, "rolling_mean_7"], expected_mean_at_13)


def test_build_feature_pipeline_integration(sample_demand_df):
    df_res = build_feature_pipeline(
        sample_demand_df, forecast_horizon=7, lags=[7, 14], windows=[7, 14], drop_na=True
    )
    
    assert len(df_res) > 0
    assert df_res["item_id"].dtype.name == "category"
    assert df_res["zone_id"].dtype.name == "category"
    
    # Sin valores nulos en features críticas tras burn-in
    feature_cols = [c for c in df_res.columns if c.startswith(("lag_", "rolling_"))]
    assert not df_res[feature_cols].isna().any().any()
