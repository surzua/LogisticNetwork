"""
Módulo de Feature Engineering para series temporales de demanda logística.

Diseñado específicamente para modelos de Machine Learning (LightGBM Quantile)
con estricta prevención de fuga de datos (Lookahead Bias).
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd

from src.config import DEMAND_CLEAN_PATH, ITEMS_METADATA_PATH


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Agrega variables de calendario y codificaciones cíclicas (seno/coseno).
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_col])

    df["day_of_week"] = dates.dt.dayofweek
    df["day_of_month"] = dates.dt.day
    df["month"] = dates.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = dates.dt.is_month_start.astype(int)
    df["is_month_end"] = dates.dt.is_month_end.astype(int)

    # Codificación cíclica continua para capturar periodicidad
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "demand_target",
    lags: List[int] = [7, 14, 21, 28],
    group_cols: List[str] = ["item_id", "zone_id"],
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Genera rezagos temporales (lags) agrupados por SKU y Zona.
    
    IMPORTANTE: Los lags deben ser >= al horizonte de pronóstico (ej. 7 días)
    para evitar data leakage en predicción directa multi-paso.
    """
    df = df.sort_values(group_cols + [date_col]).copy()
    grouped = df.groupby(group_cols)[target_col]

    for lag in lags:
        df[f"lag_{lag}"] = grouped.shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "demand_target",
    windows: List[int] = [7, 14, 28],
    shift: int = 7,
    group_cols: List[str] = ["item_id", "zone_id"],
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Calcula estadísticas móviles (media, std, min, max) con un desplazamiento (shift)
    fijo igual o superior al horizonte de pronóstico para prevenir fuga de información.
    """
    df = df.sort_values(group_cols + [date_col]).copy()
    
    # Primero aplicamos el shift al target para crear la serie histórica segura
    shifted_target = df.groupby(group_cols)[target_col].shift(shift)

    for w in windows:
        # Usamos rolling sobre la serie desplazada por grupo
        rolling_obj = shifted_target.groupby([df[col] for col in group_cols]).rolling(
            window=w, min_periods=max(1, w // 2)
        )
        
        df[f"rolling_mean_{w}"] = rolling_obj.mean().reset_index(level=list(range(len(group_cols))), drop=True)
        df[f"rolling_std_{w}"] = rolling_obj.std().reset_index(level=list(range(len(group_cols))), drop=True).fillna(0.0)
        df[f"rolling_min_{w}"] = rolling_obj.min().reset_index(level=list(range(len(group_cols))), drop=True)
        df[f"rolling_max_{w}"] = rolling_obj.max().reset_index(level=list(range(len(group_cols))), drop=True)

    # Ratio de aceleración / momentum de demanda (corto plazo 7d vs mediano plazo 28d)
    if 7 in windows and 28 in windows:
        df["momentum_7_28"] = df["rolling_mean_7"] / (df["rolling_mean_28"] + 1e-4)

    return df


def add_price_promo_features(
    df: pd.DataFrame,
    items_metadata_path: Optional[Union[str, Path]] = None,
    group_cols: List[str] = ["item_id", "zone_id"],
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Calcula características de política comercial, elasticidad y promociones.
    """
    df = df.sort_values(group_cols + [date_col]).copy()

    meta_path = Path(items_metadata_path) if items_metadata_path else ITEMS_METADATA_PATH
    if meta_path.exists():
        items_meta = pd.read_parquet(meta_path)
        if "base_price" in items_meta.columns and "base_price" not in df.columns:
            df = df.merge(items_meta[["item_id", "base_price", "category"]], on="item_id", how="left")

    if "base_price" in df.columns and "price" in df.columns:
        # Descuento relativo respecto al precio de lista catálogo
        df["discount_pct"] = np.clip((df["base_price"] - df["price"]) / df["base_price"], 0.0, 1.0)
        df["price_ratio"] = df["price"] / df["base_price"]

    # Efecto contemporáneo y rezago de promoción
    if "is_promo" in df.columns:
        grouped = df.groupby(group_cols)["is_promo"]
        df["promo_lag_7"] = grouped.shift(7).fillna(0).astype(int)

    return df


def add_hierarchical_aggregations(
    df: pd.DataFrame,
    target_col: str = "demand_target",
    shift: int = 7,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Genera agregaciones jerárquicas macro con shift para capturar dinámica global:
    1. Demanda agregada por Zona geográfica a t-shift.
    2. Demanda agregada a nivel SKU para toda la red a t-shift.
    """
    df = df.copy()

    # 1. Demanda total por Zona en fecha t
    zone_daily = df.groupby(["zone_id", date_col])[target_col].sum().reset_index()
    zone_daily = zone_daily.sort_values(["zone_id", date_col])
    zone_daily[f"zone_total_demand_lag_{shift}"] = zone_daily.groupby("zone_id")[target_col].shift(shift)
    df = df.merge(
        zone_daily[["zone_id", date_col, f"zone_total_demand_lag_{shift}"]],
        on=["zone_id", date_col],
        how="left",
    )

    # 2. Demanda total del SKU a nivel de red nacional en fecha t
    sku_daily = df.groupby(["item_id", date_col])[target_col].sum().reset_index()
    sku_daily = sku_daily.sort_values(["item_id", date_col])
    sku_daily[f"sku_network_demand_lag_{shift}"] = sku_daily.groupby("item_id")[target_col].shift(shift)
    df = df.merge(
        sku_daily[["item_id", date_col, f"sku_network_demand_lag_{shift}"]],
        on=["item_id", date_col],
        how="left",
    )

    return df


def build_feature_pipeline(
    df: pd.DataFrame,
    forecast_horizon: int = 7,
    lags: List[int] = [7, 14, 21, 28],
    windows: List[int] = [7, 14, 28],
    target_col: str = "demand_target",
    drop_na: bool = True,
    items_metadata_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Pipeline orquestador completo de ingeniería de características.

    Aplica en orden estricto:
    1. Calendario y cíclicas.
    2. Dinámica de precios y promociones.
    3. Lags protegidos contra leakage (lags >= forecast_horizon).
    4. Estadísticas móviles con shift >= forecast_horizon.
    5. Agregaciones jerárquicas con shift >= forecast_horizon.
    6. Conversión de variables categóricas al formato optimizado de pandas.
    """
    # 1. Calendario
    df = add_calendar_features(df)

    # 2. Precios y promociones
    df = add_price_promo_features(df, items_metadata_path=items_metadata_path)

    # 3. Lags
    # Validamos que no existan lags menores al horizonte de predicción
    valid_lags = [lag for lag in lags if lag >= forecast_horizon]
    df = add_lag_features(df, target_col=target_col, lags=valid_lags)

    # 4. Ventanas móviles
    df = add_rolling_features(
        df, target_col=target_col, windows=windows, shift=forecast_horizon
    )

    # 5. Jerárquicas
    df = add_hierarchical_aggregations(df, target_col=target_col, shift=forecast_horizon)

    # 6. Tipado categórico
    cat_cols = ["item_id", "zone_id"]
    if "category" in df.columns:
        cat_cols.append("category")

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 7. Limpieza de filas iniciales de calentamiento (burn-in period)
    if drop_na:
        feature_cols = [c for c in df.columns if c.startswith(("lag_", "rolling_", "zone_total_", "sku_network_"))]
        df = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df


if __name__ == "__main__":
    print("Ejecutando pipeline de features sobre dataset limpio...")
    raw_df = pd.read_parquet(DEMAND_CLEAN_PATH)
    features_df = build_feature_pipeline(raw_df, forecast_horizon=7)
    print(f"Dataset generado exitosamente:")
    print(f"- Filas: {len(features_df):,}")
    print(f"- Columnas: {len(features_df.columns)}")
    print(f"- Features generadas: {[c for c in features_df.columns if c not in raw_df.columns]}")
