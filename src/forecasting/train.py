"""
Módulo de entrenamiento de modelos predictivos de demanda con cuantiles.

Utiliza LightGBM con función de pérdida Pinball Loss (Quantile Regression)
para estimar los cuantiles P10, P50 y P90, con validación temporal cruzada
y prevención de cruce de cuantiles.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

from src.config import (
    DEMAND_CLEAN_PATH,
    MODELS_DIR,
    QUANTILES,
    RANDOM_SEED,
)
from src.data.features import build_feature_pipeline
from src.utils.metrics import pinball_loss, wape_metric


def enforce_monotonicity(preds_dict: Dict[float, np.ndarray]) -> Dict[float, np.ndarray]:
    """
    Garantiza no-negatividad y orden monótono estricto entre cuantiles:
    0 <= P10 <= P50 <= P90
    
    Evita el 'quantile crossing' característico de modelos de cuantiles desacoplados.
    """
    sorted_alphas = sorted(preds_dict.keys())
    
    # Matriz shape: (n_samples, n_quantiles)
    arr = np.column_stack([np.maximum(0.0, preds_dict[a]) for a in sorted_alphas])
    
    # Ordenar monótonamente a lo largo del eje de cuantiles
    arr_sorted = np.sort(arr, axis=1)
    
    result = {}
    for i, a in enumerate(sorted_alphas):
        result[a] = arr_sorted[:, i]
        
    return result


class QuantileForecaster:
    """
    Contenedor para entrenar, predecir y serializar modelos LightGBM por cuantiles.
    """

    def __init__(
        self,
        quantiles: List[float] = QUANTILES,
        params: Optional[Dict] = None,
        random_seed: int = RANDOM_SEED,
    ):
        self.quantiles = quantiles
        self.random_seed = random_seed
        self.models: Dict[float, lgb.LGBMRegressor] = {}
        self.feature_names_: List[str] = []

        self.default_params = {
            "n_estimators": 180,
            "learning_rate": 0.06,
            "num_leaves": 31,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_samples": 20,
            "random_state": self.random_seed,
            "verbosity": -1,
            "n_jobs": -1,
        }
        if params:
            self.default_params.update(params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileForecaster":
        """Entrena un modelo LightGBM para cada cuantil especificado."""
        self.feature_names_ = list(X.columns)

        for q in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                **self.default_params,
            )
            model.fit(X, y)
            self.models[q] = model

        return self

    def predict(
        self, X: pd.DataFrame, enforce_monotonic: bool = True
    ) -> Dict[float, np.ndarray]:
        """Genera predicciones para cada cuantil con ordenamiento monótono opcional."""
        if not self.models:
            raise ValueError("El modelo aún no ha sido entrenado. Llame a fit() primero.")

        preds = {}
        for q, model in self.models.items():
            preds[q] = model.predict(X[self.feature_names_])

        if enforce_monotonic:
            preds = enforce_monotonicity(preds)

        return preds

    def save(self, output_dir: Union[str, Path] = MODELS_DIR) -> None:
        """Serializa cada modelo a disco en output_dir."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for q, model in self.models.items():
            pct = int(q * 100)
            file_name = f"lgbm_quantile_p{pct}.pkl"
            with open(out_path / file_name, "wb") as f:
                pickle.dump(model, f)

        # Guardar lista de features y metadata
        metadata = {
            "quantiles": self.quantiles,
            "features": self.feature_names_,
            "random_seed": self.random_seed,
        }
        with open(out_path / "forecaster_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    @classmethod
    def load(cls, input_dir: Union[str, Path] = MODELS_DIR) -> "QuantileForecaster":
        """Carga los modelos y metadata desde disco."""
        in_path = Path(input_dir)
        with open(in_path / "forecaster_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        instance = cls(
            quantiles=metadata["quantiles"],
            random_seed=metadata.get("random_seed", RANDOM_SEED),
        )
        instance.feature_names_ = metadata["features"]

        for q in instance.quantiles:
            pct = int(q * 100)
            file_name = f"lgbm_quantile_p{pct}.pkl"
            with open(in_path / file_name, "rb") as f:
                instance.models[q] = pickle.load(f)

        return instance


def evaluate_forecasts(
    y_true: np.ndarray, preds_dict: Dict[float, np.ndarray]
) -> Dict[str, float]:
    """
    Calcula métricas clave de evaluación probabilística y puntual:
    - Pinball Loss por cuantil
    - WAPE y MAE en P50
    - Cobertura empírica del intervalo [P10, P90] (nominal 80%)
    - Ancho promedio del intervalo de incertidumbre (Sharpness)
    """
    metrics = {}
    for q, y_pred in preds_dict.items():
        pct = int(q * 100)
        metrics[f"pinball_loss_p{pct}"] = pinball_loss(y_true, y_pred, q)

    if 0.50 in preds_dict:
        metrics["wape_p50"] = wape_metric(y_true, preds_dict[0.50])
        metrics["mae_p50"] = float(mean_absolute_error(y_true, preds_dict[0.50]))

    if 0.10 in preds_dict and 0.90 in preds_dict:
        p10 = preds_dict[0.10]
        p90 = preds_dict[0.90]
        inside = (y_true >= p10) & (y_true <= p90)
        metrics["coverage_80_pct"] = float(np.mean(inside) * 100.0)
        metrics["sharpness_p90_p10"] = float(np.mean(p90 - p10))

    return metrics


def expanding_window_cv(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = "demand_target",
    horizon_days: int = 14,
    n_splits: int = 3,
) -> pd.DataFrame:
    """
    Realiza validación cruzada temporal por ventanas expansivas (Expanding Window CV).
    
    Avanza la ventana temporal en bloques de horizon_days sin fuga de información.
    """
    dates = pd.to_datetime(df["date"]).sort_values().unique()
    total_dates = len(dates)

    # Definir los puntos de corte para las últimas ventanas
    fold_results = []

    for i in range(n_splits, 0, -1):
        test_end_idx = total_dates - (i - 1) * horizon_days
        test_start_idx = test_end_idx - horizon_days
        
        train_dates = dates[:test_start_idx]
        test_dates = dates[test_start_idx:test_end_idx]
        
        train_mask = df["date"].isin(train_dates)
        test_mask = df["date"].isin(test_dates)
        
        X_tr = df.loc[train_mask, features]
        y_tr = df.loc[train_mask, target_col]
        X_te = df.loc[test_mask, features]
        y_te = df.loc[test_mask, target_col]
        
        forecaster = QuantileForecaster()
        forecaster.fit(X_tr, y_tr)
        preds = forecaster.predict(X_te)
        
        fold_metric = evaluate_forecasts(y_te.values, preds)
        fold_metric["fold"] = n_splits - i + 1
        fold_metric["train_start"] = train_dates[0].strftime("%Y-%m-%d")
        fold_metric["train_end"] = train_dates[-1].strftime("%Y-%m-%d")
        fold_metric["test_start"] = test_dates[0].strftime("%Y-%m-%d")
        fold_metric["test_end"] = test_dates[-1].strftime("%Y-%m-%d")
        fold_metric["test_records"] = len(X_te)
        fold_results.append(fold_metric)

    return pd.DataFrame(fold_results)


def train_production_models() -> Tuple[QuantileForecaster, Dict[str, float]]:
    """
    Pipeline completo:
    1. Carga y enriquecimiento de datos.
    2. Partición cronológica (Train + Val vs Hold-out Test final de 14 días).
    3. Entrenamiento de cuantiles P10, P50, P90.
    4. Evaluación de métricas.
    5. Serialización a MODELS_DIR.
    """
    print("🚀 Iniciando pipeline de entrenamiento de modelos predictivos por cuantiles...")
    raw_df = pd.read_parquet(DEMAND_CLEAN_PATH)
    features_df = build_feature_pipeline(raw_df, forecast_horizon=7, drop_na=True)

    ignore_cols = ["date", "demand_target", "units_sold", "latent_demand_true", "is_stockout_flag"]
    features = [c for c in features_df.columns if c not in ignore_cols]

    # Hold-out test: Últimos 14 días de 2025
    test_dates = features_df["date"].sort_values().unique()[-14:]
    test_start = test_dates[0]

    train_mask = features_df["date"] < test_start
    test_mask = features_df["date"] >= test_start

    X_train = features_df.loc[train_mask, features]
    y_train = features_df.loc[train_mask, "demand_target"]
    X_test = features_df.loc[test_mask, features]
    y_test = features_df.loc[test_mask, "demand_target"]

    print(f"📊 Partición temporal:")
    print(f"   - Entrenamiento: {len(X_train):,} filas ({features_df.loc[train_mask, 'date'].min().strftime('%Y-%m-%d')} a {features_df.loc[train_mask, 'date'].max().strftime('%Y-%m-%d')})")
    print(f"   - Test Hold-out: {len(X_test):,} filas ({test_start.strftime('%Y-%m-%d')} a {test_dates[-1].strftime('%Y-%m-%d')})")
    print(f"   - Features ({len(features)}): {features[:5]}... (+{len(features)-5} más)")

    forecaster = QuantileForecaster(quantiles=QUANTILES)
    forecaster.fit(X_train, y_train)

    test_preds = forecaster.predict(X_test)
    metrics = evaluate_forecasts(y_test.values, test_preds)

    print("\n📈 Métricas en Test Hold-out:")
    print(f"   - WAPE (P50):           {metrics.get('wape_p50', 0):.2f}%")
    print(f"   - MAE (P50):            {metrics.get('mae_p50', 0):.2f} unidades")
    print(f"   - Cobertura [P10, P90]: {metrics.get('coverage_80_pct', 0):.2f}% (Objetivo: ~80.0%)")
    print(f"   - Ancho Intervalo:      {metrics.get('sharpness_p90_p10', 0):.2f} unidades")
    print(f"   - Pinball Loss P10:     {metrics.get('pinball_loss_p10', 0):.4f}")
    print(f"   - Pinball Loss P50:     {metrics.get('pinball_loss_p50', 0):.4f}")
    print(f"   - Pinball Loss P90:     {metrics.get('pinball_loss_p90', 0):.4f}")

    # Guardar modelos
    forecaster.save(MODELS_DIR)
    print(f"\n💾 Modelos serializados con éxito en: {MODELS_DIR}")

    return forecaster, metrics


if __name__ == "__main__":
    train_production_models()
