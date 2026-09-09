"""
Pruebas unitarias para el motor de optimización prescriptiva MILP y benchmark
(src/optimization/problem_builder.py, solver.py y benchmark.py).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import OPTIMAL_PLAN_PATH, BENCHMARK_PLAN_PATH
from src.optimization.problem_builder import SupplyChainModel
from src.optimization.solver import SupplyChainSolver, solve_supply_chain
from src.optimization.benchmark import run_benchmark


@pytest.fixture(scope="module")
def base_solution():
    """Ejecuta y cachea la solución MILP base para las pruebas."""
    return solve_supply_chain(demand_quantile="p50", save=False)


def test_problem_builder_structure():
    """Verifica que el modelo cree las variables, conjuntos y restricciones esperadas."""
    model = SupplyChainModel(demand_quantile="p50")

    assert len(model.cds) == 2
    assert len(model.dark_stores) == 8
    assert len(model.zones) == 20
    assert len(model.items) == 5
    assert len(model.time_periods) == 14

    # Verificar que el problema PuLP se haya inicializado
    assert model.problem is not None
    assert len(model.problem.variables()) > 10000
    assert len(model.problem.constraints) > 2000


def test_solver_optimality(base_solution):
    """Verifica que el solver alcance la solución óptima y reporte KPIs válidos."""
    assert base_solution.is_optimal
    assert base_solution.status == "Optimal"
    assert base_solution.total_cost > 0.0
    assert base_solution.solve_time_seconds < 60.0

    kpis = base_solution.kpis
    assert kpis["otif_pct"] >= 95.0
    assert kpis["total_demand_units"] > 0.0
    assert kpis["fulfilled_units"] > 0.0
    assert np.isclose(kpis["fulfilled_units"] + kpis["stockout_units"], kpis["total_demand_units"], atol=1.0)


def test_capacity_constraints_never_violated(base_solution):
    """
    Verifica que en ningún nodo y en ninguna fecha la ocupación volumétrica
    supere la capacidad física disponible.
    """
    inv_df = base_solution.inventory_df
    assert not inv_df.empty, "El DataFrame de inventario no debe estar vacío."

    # Agrupar por fecha y nodo para calcular la suma de m3
    daily_node_vol = inv_df.groupby(["date", "node_id"]).agg({
        "volume_m3": "sum",
        "capacity_m3": "first",
    }).reset_index()

    # Tolerancia numérica por redondeo del solver
    tolerance = 1e-3
    violations = daily_node_vol[daily_node_vol["volume_m3"] > daily_node_vol["capacity_m3"] + tolerance]

    assert violations.empty, f"Se detectaron violaciones de capacidad en nodos:\n{violations}"


def test_demand_satisfaction_balance(base_solution):
    """
    Verifica que para cada SKU, zona y fecha la demanda sea cubierta por despachos o stockout.
    """
    model = SupplyChainModel(demand_quantile="p50")
    f_df = base_solution.fulfillment_df
    s_df = base_solution.stockouts_df

    # Sumar unidades despachadas por fecha, SKU y zona
    f_sum = f_df.groupby(["date", "item_id", "to_zone_id"])["units"].sum().to_dict() if not f_df.empty else {}
    s_sum = s_df.groupby(["date", "item_id", "zone_id"])["stockout_units"].sum().to_dict() if not s_df.empty else {}

    for (item, zone, t_str), dem_val in model.demand.items():
        date_ts = pd.to_datetime(t_str)
        dispatched = f_sum.get((date_ts, item, zone), 0.0)
        stockout = s_sum.get((date_ts, item, zone), 0.0)

        assert np.isclose(dispatched + stockout, dem_val, atol=1e-2), (
            f"Desbalance en {item}, {zone}, {t_str}: Demand={dem_val}, Dispatched={dispatched}, Stockout={stockout}"
        )


def test_disabled_node_scenario():
    """
    Verifica que al deshabilitar un nodo (ej. DS_PROVIDENCIA) no se genere flujo hacia o desde él.
    """
    disabled_node = "DS_PROVIDENCIA"
    sol_disrupted = solve_supply_chain(
        demand_quantile="p50",
        disabled_nodes=[disabled_node],
        save=False,
    )

    assert sol_disrupted.is_optimal

    # Verificar que el nodo inhabilitado no tenga despachos a clientes
    f_df = sol_disrupted.fulfillment_df
    assert (f_df["from_node_id"] == disabled_node).sum() == 0, (
        f"El nodo deshabilitado {disabled_node} generó despachos salientes."
    )

    # Verificar que no reciba reabastecimiento
    r_df = sol_disrupted.replenishment_df
    if not r_df.empty:
        assert (r_df["to_ds_id"] == disabled_node).sum() == 0, (
            f"El nodo deshabilitado {disabled_node} recibió reabastecimientos."
        )


def test_transport_cost_multiplier_what_if():
    """
    Verifica que un aumento en los costos de transporte incremente el costo logístico total.
    """
    sol_base = solve_supply_chain(demand_quantile="p50", transport_cost_multiplier=1.0, save=False)
    sol_high_cost = solve_supply_chain(demand_quantile="p50", transport_cost_multiplier=1.5, save=False)

    assert sol_high_cost.is_optimal
    assert sol_high_cost.kpis["total_transport_cost"] > sol_base.kpis["total_transport_cost"]
    assert sol_high_cost.total_cost > sol_base.total_cost


def test_milp_beats_naive_benchmark_hypothesis_h2():
    """
    Valida la Hipótesis H2 del proyecto:
    El modelo MILP reduce el costo logístico total en al menos un 8%
    respecto a la heurística naive de despacho miope.
    """
    sol_milp = solve_supply_chain(demand_quantile="p50", save=False)
    res_bench = run_benchmark(demand_quantile="p50", save=False)

    cost_milp = sol_milp.total_cost
    cost_bench = res_bench["total_cost"]

    assert cost_bench > cost_milp, "La heurística naive no debería superar al óptimo global MILP."

    savings_pct = ((cost_bench - cost_milp) / cost_bench) * 100.0
    assert savings_pct >= 8.0, (
        f"El ahorro del modelo MILP ({savings_pct:.2f}%) no superó el umbral del 8% de la hipótesis H2."
    )


def test_parquet_output_files_exist_and_valid():
    """
    Verifica que los archivos Parquet del plan óptimo y benchmark existan y sean válidos.
    """
    # Asegurar que se guarden
    solve_supply_chain(demand_quantile="p50", save=True)
    run_benchmark(demand_quantile="p50", save=True)

    assert Path(OPTIMAL_PLAN_PATH).exists(), "El archivo optimal_fulfillment_plan.parquet debe existir."
    assert Path(BENCHMARK_PLAN_PATH).exists(), "El archivo benchmark_fulfillment_plan.parquet debe existir."

    df_opt = pd.read_parquet(OPTIMAL_PLAN_PATH)
    expected_cols = [
        "date", "item_id", "from_node_id", "to_zone_id", "node_type",
        "units", "volume_m3", "distance_km", "lead_time_days", "transport_cost",
        "is_cross_fulfillment",
    ]
    for col in expected_cols:
        assert col in df_opt.columns, f"Columna requerida faltante en plan óptimo: {col}"

    assert not df_opt.empty
    assert not df_opt[expected_cols].isna().any().any(), "No deben haber valores nulos en el plan de fulfillment."
    assert (df_opt["units"] > 0).all(), "Todas las filas del plan deben tener unidades despachadas > 0."
