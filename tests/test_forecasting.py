"""
Pruebas unitarias para el módulo predictivo por cuantiles (src/forecasting/train.py y predict.py).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import FORECAST_OUTPUT_PATH
from src.utils.metrics import (
    pinball_loss,
    wape_metric,
    mae_metric,
    rmse_metric,
    service_level_otif,
)
from src.forecasting.train import (
    enforce_monotonicity,
    QuantileForecaster,
    evaluate_forecasts,
)


def test_pinball_loss():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred_exact = np.array([10.0, 20.0, 30.0])
    
    # Pérdida debe ser 0 cuando la predicción es exacta
    assert pinball_loss(y_true, y_pred_exact, alpha=0.5) == 0.0
    assert pinball_loss(y_true, y_pred_exact, alpha=0.9) == 0.0

    # Sub-predicción (y_true > y_pred): pérdida = alpha * (y - y_hat)
    y_pred_under = np.array([5.0, 15.0, 25.0])  # diff = 5.0
    loss_p90 = pinball_loss(y_true, y_pred_under, alpha=0.9)
    assert np.isclose(loss_p90, 0.9 * 5.0)

    # Sobre-predicción (y_true < y_pred): pérdida = (1 - alpha) * (y_hat - y)
    y_pred_over = np.array([15.0, 25.0, 35.0])  # diff = -5.0
    loss_p10 = pinball_loss(y_true, y_pred_over, alpha=0.1)
    assert np.isclose(loss_p10, (1.0 - 0.1) * 5.0)


def test_wape_metric():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([90.0, 220.0])  # abs err = 10 + 20 = 30; sum(y) = 300
    assert np.isclose(wape_metric(y_true, y_pred), 10.0)


def test_additional_metrics_mae_rmse_otif():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])

    assert np.isclose(mae_metric(y_true, y_pred), 4.0 / 3.0)
    assert np.isclose(rmse_metric(y_true, y_pred), np.sqrt(8.0 / 3.0))

    # OTIF: Demanda total = 100, despachado = 90 -> 90%
    demand = np.array([50.0, 50.0])
    fulfilled = np.array([50.0, 40.0])
    assert np.isclose(service_level_otif(demand, fulfilled), 90.0)
    assert np.isclose(service_level_otif(np.array([0.0]), np.array([0.0])), 100.0)


def test_enforce_monotonicity_and_non_negativity():
    # Caso con cruce de cuantiles y valores negativos
    preds_crossed = {
        0.10: np.array([25.0, -5.0, 40.0]),
        0.50: np.array([20.0, 10.0, 35.0]),  # En pos 0, P10 (25) > P50 (20)
        0.90: np.array([30.0, 15.0, 30.0]),  # En pos 2, P50 (35) > P90 (30)
    }

    corrected = enforce_monotonicity(preds_crossed)

    # 1. No negatividad
    for q in [0.10, 0.50, 0.90]:
        assert (corrected[q] >= 0.0).all()

    # 2. Monotonicidad estricta P10 <= P50 <= P90
    assert (corrected[0.10] <= corrected[0.50]).all()
    assert (corrected[0.50] <= corrected[0.90]).all()


def test_quantile_forecaster_fit_and_predict():
    np.random.seed(42)
    n_samples = 150
    X = pd.DataFrame({
        "feat_num": np.random.randn(n_samples),
        "feat_cat": pd.Categorical(np.random.choice(["A", "B", "C"], size=n_samples)),
    })
    y = pd.Series(np.abs(X["feat_num"] * 10 + 20 + np.random.randn(n_samples)))

    forecaster = QuantileForecaster(
        quantiles=[0.10, 0.50, 0.90],
        params={"n_estimators": 20, "min_child_samples": 5},
    )
    forecaster.fit(X, y)

    preds = forecaster.predict(X, enforce_monotonic=True)

    assert set(preds.keys()) == {0.10, 0.50, 0.90}
    for q in [0.10, 0.50, 0.90]:
        assert len(preds[q]) == n_samples
        assert (preds[q] >= 0.0).all()

    assert (preds[0.10] <= preds[0.50]).all()
    assert (preds[0.50] <= preds[0.90]).all()


def test_forecast_matrix_file_schema():
    assert Path(FORECAST_OUTPUT_PATH).exists(), "El archivo de pronósticos parquet debe existir."
    df = pd.read_parquet(FORECAST_OUTPUT_PATH)

    required_cols = ["date", "item_id", "zone_id", "demand_p10", "demand_p50", "demand_p90"]
    for col in required_cols:
        assert col in df.columns, f"Columna requerida faltante: {col}"

    # Validar no-nulos
    assert not df[required_cols].isna().any().any(), "No deben existir valores NaN en las predicciones."

    # Validar monotonicidad en el archivo final
    assert (df["demand_p10"] <= df["demand_p50"] + 1e-4).all()
    assert (df["demand_p50"] <= df["demand_p90"] + 1e-4).all()
    assert (df["demand_p10"] >= 0.0).all()
