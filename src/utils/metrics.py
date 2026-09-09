"""
Módulo de métricas analíticas, de forecasting y de evaluación de servicio logístico.

Incluye métricas probabilísticas (Pinball Loss para cuantiles), porcentuales ponderadas (WAPE),
clásicas de regresión (MAE, RMSE) y de nivel de servicio en cadena de suministro (OTIF).
"""

from typing import Union
import numpy as np


def pinball_loss(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    alpha: float,
) -> float:
    """
    Calcula la pérdida Pinball Loss (Quantile Loss) para un cuantil alpha dado.

    Fórmula:
        L_alpha(y, y_hat) = max(alpha * (y - y_hat), (alpha - 1) * (y - y_hat))

    Parámetros:
        y_true: Valores reales observados.
        y_pred: Valores pronosticados para el cuantil alpha.
        alpha: Nivel del cuantil en el intervalo (0, 1) (ej: 0.1, 0.5, 0.9).

    Retorna:
        Pérdida media de pinball como float.
    """
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
    return float(np.mean(loss))


def wape_metric(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """
    Calcula el Weighted Absolute Percentage Error (WAPE).

    Fórmula:
        WAPE = (sum(|y - y_hat|) / sum(y)) * 100

    Es más robusto que MAPE ante valores cercanos o iguales a cero en series de demanda.

    Parámetros:
        y_true: Valores reales observados.
        y_pred: Valores pronosticados.

    Retorna:
        WAPE en porcentaje (0% a inf). Retorna 0.0 si la demanda total real es 0.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    total_y = float(np.sum(y_true_arr))
    if total_y == 0.0:
        return 0.0
    return float(np.sum(np.abs(y_true_arr - y_pred_arr)) / total_y * 100.0)


def mae_metric(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """
    Calcula el Mean Absolute Error (MAE).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def rmse_metric(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """
    Calcula el Root Mean Squared Error (RMSE).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def service_level_otif(
    demand: Union[np.ndarray, list],
    fulfilled: Union[np.ndarray, list],
) -> float:
    """
    Calcula el Nivel de Servicio On-Time In-Full (OTIF) en porcentaje.

    Fórmula:
        OTIF = (sum(min(fulfilled, demand)) / sum(demand)) * 100

    Parámetros:
        demand: Demanda requerida por los clientes.
        fulfilled: Cantidad despachada / satisfecha.

    Retorna:
        Porcentaje de servicio cumplido (0% a 100%).
    """
    demand_arr = np.asarray(demand, dtype=float)
    fulfilled_arr = np.asarray(fulfilled, dtype=float)
    total_demand = float(np.sum(demand_arr))
    if total_demand == 0.0:
        return 100.0
    effective_fulfilled = np.minimum(fulfilled_arr, demand_arr)
    return float(np.sum(effective_fulfilled) / total_demand * 100.0)
