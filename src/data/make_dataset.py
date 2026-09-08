"""
Generador de demanda histórica y pipeline de limpieza/imputación de demanda censurada.
Produce:
  1. data/01_raw/raw_sales_demand.parquet: Demanda observada con stockouts simulados (demanda truncada a 0)
  2. data/02_intermediate/demand_clean.parquet: Demanda con indicador de stockout e imputación reconstructiva
  3. data/02_intermediate/items_metadata.parquet: Catálogo dimensional de SKUs (volumen, precios, penalizaciones)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Permitir ejecución directa y como módulo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# pyrefly: ignore [missing-import]
from src.config import (
    DEMAND_ZONES,
    SKU_CATALOG,
    SIMULATION_START_DATE,
    SIMULATION_END_DATE,
    RANDOM_SEED,
    RAW_SALES_PATH,
    DEMAND_CLEAN_PATH,
    ITEMS_METADATA_PATH,
    RAW_DATA_DIR,
    INTERMEDIATE_DATA_DIR,
)


def generate_items_metadata() -> pd.DataFrame:
    """Genera la tabla dimensional de SKUs con atributos físicos y de costos."""
    records = []
    for item_id, props in SKU_CATALOG.items():
        records.append({
            "item_id": item_id,
            "item_name": props["name"],
            "category": props["category"],
            "volume_m3": props["volume_m3"],
            "base_price": props["base_price"],
            "stockout_cost": props["stockout_cost"],
        })
    df_items = pd.DataFrame(records)
    return df_items


def simulate_raw_demand() -> pd.DataFrame:
    """
    Simula series de tiempo diarias de demanda para cada par (SKU, Zona) entre 2024-01-01 y 2025-12-31.
    Componentes incluidos:
      - Tendencia de crecimiento e-commerce anual (+10% interanual).
      - Estacionalidad semanal (picos jueves-sábado, valles lunes-martes).
      - Estacionalidad mensual/anual (picos de fin de mes / sueldos y festividades de diciembre/septiembre en Chile).
      - Efecto de promociones esporádicas (rebajas con salto de demanda).
      - Quiebres de stock (censura de demanda: venta observada = 0 cuando hay stockout).
    """
    np.random.seed(RANDOM_SEED)
    dates = pd.date_range(start=SIMULATION_START_DATE, end=SIMULATION_END_DATE, freq="D")
    n_days = len(dates)

    zone_ids = list(DEMAND_ZONES.keys())
    item_ids = list(SKU_CATALOG.keys())

    # Factores demográficos/socioeconómicos por zona (multiplicador de escala de demanda)
    # Por ejemplo, Santiago Centro o Maipú tienen mayor volumen absoluto por población
    zone_scale = {}
    for idx, zid in enumerate(zone_ids):
        # Escala entre 0.7 y 1.4 según tamaño comunal
        zone_scale[zid] = 0.75 + (hash(zid) % 70) / 100.0

    records = []

    # Factores temporales
    day_of_week = dates.dayofweek.values  # 0=Lunes, 6=Domingo
    # Patrón semanal: viernes(4) y sábado(5) tienen más compras online/supermercado
    weekly_multipliers = np.array([0.80, 0.85, 0.95, 1.10, 1.25, 1.30, 0.90])
    day_factors = weekly_multipliers[day_of_week]

    # Efecto quincena y fin de mes (pago de sueldos en Chile: días 28 al 31, y días 1 al 3)
    payday_effect = np.where((dates.day >= 28) | (dates.day <= 3) | (dates.day == 15), 1.20, 1.0)

    # Estacionalidad festiva chilena: Fiestas Patrias (septiembre) y Navidad/Año Nuevo (diciembre)
    seasonality = np.ones(n_days)
    seasonality[dates.month == 9] *= 1.18
    seasonality[dates.month == 12] *= 1.30

    # Tendencia lineal a 2 años (crecimiento sostenido del canal digital ~15% en 2 años)
    trend = np.linspace(1.0, 1.15, n_days)

    for item_id in item_ids:
        item_props = SKU_CATALOG[item_id]
        base_demand = item_props["base_daily_demand"]
        base_price = item_props["base_price"]
        elasticity = item_props["promo_elasticity"]

        for zone_id in zone_ids:
            scale = zone_scale[zone_id]

            # Probabilidad de promoción (5% de los días)
            is_promo = np.random.binomial(n=1, p=0.05, size=n_days)
            promo_mult = np.where(is_promo == 1, elasticity, 1.0)
            price_arr = np.where(is_promo == 1, round(base_price * 0.80, 2), base_price)

            # Demanda esperada (lambda de Poisson / media)
            lambda_demand = (
                base_demand
                * scale
                * day_factors
                * payday_effect
                * seasonality
                * trend
                * promo_mult
            )

            # Generar demanda latente real (Poisson con sobredispersión moderada)
            latent_demand = np.random.poisson(lam=lambda_demand)

            # Simulación de Stockout Operativo (censura de demanda):
            # Ocurre en días de alta demanda imprevista o interrupciones de suministro (~3% de los días)
            stockout_event = np.random.binomial(n=1, p=0.035, size=n_days)
            
            # En días de stockout, la venta registrada cae a 0 (o remanente mínimo)
            units_sold = np.where(stockout_event == 1, 0, latent_demand)

            for t in range(n_days):
                records.append({
                    "date": dates[t],
                    "item_id": item_id,
                    "zone_id": zone_id,
                    "units_sold": int(units_sold[t]),
                    "latent_demand_true": int(latent_demand[t]),  # Ground truth para validación de H1
                    "price": float(price_arr[t]),
                    "is_promo": int(is_promo[t]),
                    "is_stockout_flag": int(stockout_event[t]),
                })

    df_raw = pd.DataFrame(records)
    return df_raw


def process_clean_demand(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Tratamiento y limpieza de demanda censurada:
    1. Detecta ventas cero anómalas que corresponden a stockouts conocidos.
    2. Imputa la demanda no satisfecha calculando la demanda esperada histórica
       según el día de la semana, SKU, zona y status de promoción.
    3. Genera el target 'demand_imputed' listo para modelos de Quantile Regression.
    """
    df = df_raw.copy()

    # Cálculo de la demanda mediana histórica libre de quiebres por (item_id, zone_id, dayofweek, is_promo)
    valid_sales = df[df["is_stockout_flag"] == 0]
    baseline_lookup = (
        valid_sales.groupby(["item_id", "zone_id", valid_sales["date"].dt.dayofweek, "is_promo"])["units_sold"]
        .median()
        .reset_index()
        .rename(columns={"date": "dayofweek", "units_sold": "imputed_baseline"})
    )

    df["dayofweek"] = df["date"].dt.dayofweek
    df = df.merge(baseline_lookup, on=["item_id", "zone_id", "dayofweek", "is_promo"], how="left")

    # Si hubo stockout, la demanda corregida toma el valor de la demanda histórica ajustada; si no, la venta real
    df["demand_target"] = np.where(
        df["is_stockout_flag"] == 1,
        np.round(df["imputed_baseline"]).fillna(df["units_sold"]),
        df["units_sold"]
    ).astype(int)

    df.drop(columns=["dayofweek", "imputed_baseline"], inplace=True)
    return df


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("📦 Generando catálogo de metadatos de SKUs...")
    df_items = generate_items_metadata()
    df_items.to_parquet(ITEMS_METADATA_PATH, index=False)
    print(f"✓ Metadatos guardados en: {ITEMS_METADATA_PATH} ({len(df_items)} SKUs)")

    print("📊 Simulando demanda histórica (2024-01-01 a 2025-12-31)...")
    df_raw = simulate_raw_demand()
    
    # Guardar raw data (sin ground-truth para simular ambiente de producción)
    df_raw_save = df_raw.drop(columns=["latent_demand_true"])
    df_raw_save.to_parquet(RAW_SALES_PATH, index=False)
    print(f"✓ Demanda cruda guardada en: {RAW_SALES_PATH} ({len(df_raw_save):,} registros)")

    print("🧹 Procesando y tratando demanda censurada (imputación de stockouts)...")
    df_clean = process_clean_demand(df_raw)
    df_clean.to_parquet(DEMAND_CLEAN_PATH, index=False)
    print(f"✓ Demanda limpia guardada en: {DEMAND_CLEAN_PATH} ({len(df_clean):,} registros)")

    print("\nResumen estadístico de demanda limpia:")
    print(df_clean.groupby("item_id")[["units_sold", "demand_target"]].agg(["mean", "std", "max"]))


if __name__ == "__main__":
    main()
