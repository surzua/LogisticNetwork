"""
Módulo de inferencia y generación de pronósticos probabilísticos de demanda.

Carga los modelos entrenados por cuantiles (P10, P50, P90) y genera la matriz
de demanda futura para alimentar el motor de optimización prescriptiva (MILP).
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from src.config import (
    DEMAND_CLEAN_PATH,
    FORECAST_HORIZON_DAYS,
    FORECAST_OUTPUT_PATH,
    MODELS_DIR,
    OUTPUT_DATA_DIR,
)
from src.data.features import build_feature_pipeline
from src.forecasting.train import QuantileForecaster


def generate_forecast_matrix(
    features_df: Optional[pd.DataFrame] = None,
    models_dir: Union[str, Path] = MODELS_DIR,
    output_path: Optional[Union[str, Path]] = None,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> pd.DataFrame:
    """
    Genera la matriz de predicción probabilística a nivel SKU-Zona-Fecha.

    Parámetros:
    -----------
    features_df: DataFrame con características procesadas. Si es None, se genera desde el dataset limpio.
    models_dir: Directorio donde residen los modelos .pkl serializados.
    output_path: Ruta parquet de destino. Si es None, usa FORECAST_OUTPUT_PATH.
    horizon_days: Cantidad de días hacia adelante a proyectar.

    Retorna:
    --------
    pd.DataFrame con columnas: ['date', 'item_id', 'zone_id', 'demand_p10', 'demand_p50', 'demand_p90']
    """
    print(f"🔮 Cargando modelos predictivos desde: {models_dir}")
    forecaster = QuantileForecaster.load(models_dir)

    if features_df is None:
        print("⚙️ Generando matriz de features desde datos limpios...")
        raw_df = pd.read_parquet(DEMAND_CLEAN_PATH)
        features_df = build_feature_pipeline(raw_df, forecast_horizon=7, drop_na=True)

    # Seleccionamos la ventana de evaluación / planificación (últimos horizon_days registros)
    dates_sorted = features_df["date"].sort_values().unique()
    target_dates = dates_sorted[-horizon_days:]
    horizon_mask = features_df["date"].isin(target_dates)

    horizon_df = features_df.loc[horizon_mask].copy().sort_values(["date", "item_id", "zone_id"]).reset_index(drop=True)

    print(f"📅 Generando predicciones para el horizonte: {target_dates[0].strftime('%Y-%m-%d')} a {target_dates[-1].strftime('%Y-%m-%d')} ({horizon_days} días)")

    # Inferencia con garantía de monotonicidad y no-negatividad
    preds = forecaster.predict(horizon_df, enforce_monotonic=True)

    result_df = pd.DataFrame({
        "date": horizon_df["date"],
        "item_id": horizon_df["item_id"].astype(str),
        "zone_id": horizon_df["zone_id"].astype(str),
        "demand_p10": preds[0.10].round(2),
        "demand_p50": preds[0.50].round(2),
        "demand_p90": preds[0.90].round(2),
    })

    # Si la demanda real está presente en el horizonte, la conservamos como referencia para benchmarking
    if "demand_target" in horizon_df.columns:
        result_df["actual_demand"] = horizon_df["demand_target"].values

    # Guardar en disco
    out_file = Path(output_path) if output_path else FORECAST_OUTPUT_PATH
    out_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(out_file, index=False)

    print(f"✅ Matriz de pronósticos exportada a: {out_file}")
    print(f"   - Total de registros: {len(result_df):,}")
    print(f"   - Columnas: {list(result_df.columns)}")
    print(f"   - Rango P50: [{result_df['demand_p50'].min():.1f}, {result_df['demand_p50'].max():.1f}]")
    print(f"   - Rango P90: [{result_df['demand_p90'].min():.1f}, {result_df['demand_p90'].max():.1f}]")

    return result_df


if __name__ == "__main__":
    generate_forecast_matrix()
