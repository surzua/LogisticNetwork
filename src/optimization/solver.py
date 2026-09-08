"""
Módulo de resolución del modelo de optimización y extracción de resultados.
Ejecuta el solver CBC sobre la formulación de PuLP, extrae flujos de despacho,
reabastecimiento, niveles de inventario y calcula los KPIs operativos y financieros.
"""

import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
import pulp

from src.config import (
    OPTIMAL_PLAN_PATH,
    DEFAULT_SOLVER_TIME_LIMIT_SECS,
    OUTPUT_DATA_DIR,
)
from src.optimization.problem_builder import SupplyChainModel


@dataclass
class SupplyChainSolution:
    """Contenedor de la solución completa de optimización de red."""
    status: str
    is_optimal: bool
    solve_time_seconds: float
    total_cost: float
    kpis: Dict[str, Any]
    fulfillment_df: pd.DataFrame
    replenishment_df: pd.DataFrame
    inventory_df: pd.DataFrame
    stockouts_df: pd.DataFrame


class SupplyChainSolver:
    """
    Orquestador de resolución de modelos SupplyChainModel y extracción de KPIs.
    """

    def __init__(self, model: SupplyChainModel):
        self.model = model
        self.solution: Optional[SupplyChainSolution] = None

    def solve(
        self,
        time_limit_secs: int = DEFAULT_SOLVER_TIME_LIMIT_SECS,
        verbose: bool = False,
    ) -> SupplyChainSolution:
        """
        Ejecuta el solver CBC y construye la solución estructurada.
        """
        solver = pulp.PULP_CBC_CMD(
            timeLimit=time_limit_secs,
            msg=verbose,
        )

        t_start = time.time()
        solver_status = self.model.problem.solve(solver)
        t_elapsed = time.time() - t_start

        status_str = pulp.LpStatus[solver_status]
        is_optimal = status_str in ("Optimal", "Integer Optimal")

        if not is_optimal:
            self.solution = SupplyChainSolution(
                status=status_str,
                is_optimal=False,
                solve_time_seconds=t_elapsed,
                total_cost=float("inf"),
                kpis={"status": status_str},
                fulfillment_df=pd.DataFrame(),
                replenishment_df=pd.DataFrame(),
                inventory_df=pd.DataFrame(),
                stockouts_df=pd.DataFrame(),
            )
            return self.solution

        # Extraer DataFrames de resultados
        fulfillment_df = self._extract_fulfillment()
        replenishment_df = self._extract_replenishment()
        inventory_df = self._extract_inventory()
        stockouts_df = self._extract_stockouts()

        # Calcular métricas globales de negocio
        kpis = self._compute_kpis(
            fulfillment_df=fulfillment_df,
            replenishment_df=replenishment_df,
            inventory_df=inventory_df,
            stockouts_df=stockouts_df,
            solve_time_seconds=t_elapsed,
            status=status_str,
        )

        total_cost = kpis["total_cost"]

        self.solution = SupplyChainSolution(
            status=status_str,
            is_optimal=True,
            solve_time_seconds=t_elapsed,
            total_cost=total_cost,
            kpis=kpis,
            fulfillment_df=fulfillment_df,
            replenishment_df=replenishment_df,
            inventory_df=inventory_df,
            stockouts_df=stockouts_df,
        )

        return self.solution

    def _extract_fulfillment(self) -> pd.DataFrame:
        """Extrae el flujo de despachos hacia las zonas de clientes."""
        records = []
        X = self.model.decision_vars["X"]

        # Cachear distancias y tipo de nodo
        top_idx = self.model.topology_df.set_index("node_id")

        # Mapeo de transporte
        tm_dict = {}
        for row in self.model.transport_df.to_dict("records"):
            tm_dict[(row["from_node_id"], row["to_node_id"])] = row

        for (item, k, j, t), var in X.items():
            qty = round(pulp.value(var), 4)
            if qty > 1e-4:
                vol_m3 = qty * self.model.volume_m3[item]
                tm_info = tm_dict.get((k, j), {})
                dist_km = tm_info.get("distance_km", 0.0)
                cost_m3 = self.model.transp_cost_per_m3_fulfillment[(k, j)]
                cost_transp = vol_m3 * cost_m3
                lead_time = self.model.lead_time_fulfillment.get((k, j), 0.2)
                is_cross = k in self.model.cds

                records.append({
                    "date": pd.to_datetime(t),
                    "item_id": item,
                    "from_node_id": k,
                    "to_zone_id": j,
                    "node_type": top_idx.loc[k, "node_type"],
                    "units": qty,
                    "volume_m3": round(vol_m3, 4),
                    "distance_km": dist_km,
                    "lead_time_days": lead_time,
                    "transport_cost": round(cost_transp, 2),
                    "is_cross_fulfillment": is_cross,
                })

        return pd.DataFrame(records)

    def _extract_replenishment(self) -> pd.DataFrame:
        """Extrae el flujo de reabastecimiento primario desde CDs hacia Dark Stores."""
        records = []
        R = self.model.decision_vars["R"]

        tm_dict = {}
        for row in self.model.transport_df.to_dict("records"):
            tm_dict[(row["from_node_id"], row["to_node_id"])] = row

        for (item, c, d, t), var in R.items():
            qty = round(pulp.value(var), 4)
            if qty > 1e-4:
                vol_m3 = qty * self.model.volume_m3[item]
                tm_info = tm_dict.get((c, d), {})
                dist_km = tm_info.get("distance_km", 0.0)
                cost_m3 = self.model.transp_cost_per_m3_replenishment[(c, d)]
                cost_transp = vol_m3 * cost_m3

                records.append({
                    "date": pd.to_datetime(t),
                    "item_id": item,
                    "from_cd_id": c,
                    "to_ds_id": d,
                    "units": qty,
                    "volume_m3": round(vol_m3, 4),
                    "distance_km": dist_km,
                    "transport_cost": round(cost_transp, 2),
                    "lead_time_days": 1.0,
                })

        return pd.DataFrame(records)

    def _extract_inventory(self) -> pd.DataFrame:
        """Extrae los niveles diarios de inventario remanente y ocupación volumétrica."""
        records = []
        I = self.model.decision_vars["I"]
        top_idx = self.model.topology_df.set_index("node_id")

        for (item, k, t), var in I.items():
            qty = round(pulp.value(var), 4)
            vol_m3 = qty * self.model.volume_m3[item]
            holding_cost = vol_m3 * self.model.holding_cost[k]

            records.append({
                "date": pd.to_datetime(t),
                "node_id": k,
                "node_type": top_idx.loc[k, "node_type"],
                "item_id": item,
                "units": qty,
                "volume_m3": round(vol_m3, 4),
                "holding_cost": round(holding_cost, 2),
                "capacity_m3": self.model.capacity[k],
            })

        df_inv = pd.DataFrame(records)
        if not df_inv.empty:
            # Calcular ocupación total diaria por nodo y % de utilización
            node_daily = df_inv.groupby(["date", "node_id"])["volume_m3"].transform("sum")
            df_inv["node_occupied_m3"] = round(node_daily, 4)
            df_inv["utilization_pct"] = np.where(
                df_inv["capacity_m3"] > 0,
                round(100.0 * df_inv["node_occupied_m3"] / df_inv["capacity_m3"], 2),
                0.0,
            )

        return df_inv

    def _extract_stockouts(self) -> pd.DataFrame:
        """Extrae los quiebres de stock / ventas perdidas."""
        records = []
        S = self.model.decision_vars["S"]

        for (item, j, t), var in S.items():
            qty = round(pulp.value(var), 4)
            if qty > 1e-4:
                cost = qty * self.model.stockout_cost[item]
                records.append({
                    "date": pd.to_datetime(t),
                    "item_id": item,
                    "zone_id": j,
                    "stockout_units": qty,
                    "stockout_cost": round(cost, 2),
                })

        return pd.DataFrame(records)

    def _compute_kpis(
        self,
        fulfillment_df: pd.DataFrame,
        replenishment_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        stockouts_df: pd.DataFrame,
        solve_time_seconds: float,
        status: str,
    ) -> Dict[str, Any]:
        """Calcula el conjunto completo de métricas operativas y de costo."""
        fulfillment_transp_cost = float(fulfillment_df["transport_cost"].sum()) if not fulfillment_df.empty else 0.0
        replenishment_transp_cost = float(replenishment_df["transport_cost"].sum()) if not replenishment_df.empty else 0.0
        total_transport_cost = fulfillment_transp_cost + replenishment_transp_cost

        total_holding_cost = float(inventory_df["holding_cost"].sum()) if not inventory_df.empty else 0.0
        total_stockout_cost = float(stockouts_df["stockout_cost"].sum()) if not stockouts_df.empty else 0.0

        total_cost = total_transport_cost + total_holding_cost + total_stockout_cost

        # Unidades demandadas, servidas y perdidas
        total_demand = sum(self.model.demand.values())
        fulfilled_units = float(fulfillment_df["units"].sum()) if not fulfillment_df.empty else 0.0
        stockout_units = float(stockouts_df["stockout_units"].sum()) if not stockouts_df.empty else 0.0

        # Nivel de Servicio (OTIF %)
        otif_pct = round(100.0 * fulfilled_units / max(1.0, total_demand), 2)

        # Tasa de Cross-Fulfillment
        cross_units = (
            float(fulfillment_df.loc[fulfillment_df["is_cross_fulfillment"], "units"].sum())
            if not fulfillment_df.empty
            else 0.0
        )
        cross_rate_pct = round(100.0 * cross_units / max(1.0, fulfilled_units), 2)

        # Utilización promedio de Dark Stores
        if not inventory_df.empty:
            ds_inv = inventory_df[inventory_df["node_type"] == "DarkStore"]
            ds_daily_util = ds_inv.groupby(["date", "node_id"])["utilization_pct"].first()
            avg_ds_utilization = round(float(ds_daily_util.mean()), 2)
        else:
            avg_ds_utilization = 0.0

        return {
            "status": status,
            "solve_time_seconds": round(solve_time_seconds, 3),
            "total_cost": round(total_cost, 2),
            "fulfillment_transport_cost": round(fulfillment_transp_cost, 2),
            "replenishment_transport_cost": round(replenishment_transp_cost, 2),
            "total_transport_cost": round(total_transport_cost, 2),
            "holding_cost": round(total_holding_cost, 2),
            "stockout_cost": round(total_stockout_cost, 2),
            "total_demand_units": round(total_demand, 1),
            "fulfilled_units": round(fulfilled_units, 1),
            "stockout_units": round(stockout_units, 1),
            "otif_pct": otif_pct,
            "cross_fulfillment_units": round(cross_units, 1),
            "cross_fulfillment_rate_pct": cross_rate_pct,
            "avg_ds_utilization_pct": avg_ds_utilization,
        }

    def save_solution(
        self,
        output_path: Path = OPTIMAL_PLAN_PATH,
    ) -> None:
        """
        Exporta la solución completa en archivos Parquet y JSON de métricas.
        """
        if self.solution is None or not self.solution.is_optimal:
            raise ValueError("No existe una solución óptima para exportar.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Guardar plan principal de fulfillment
        self.solution.fulfillment_df.to_parquet(output_path, index=False)

        # 2. Guardar reabastecimiento e inventario como artefactos complementarios
        base_dir = output_path.parent
        replenish_path = base_dir / "optimal_replenishment_plan.parquet"
        inventory_path = base_dir / "optimal_inventory_plan.parquet"
        kpis_path = base_dir / "optimal_kpis.json"

        self.solution.replenishment_df.to_parquet(replenish_path, index=False)
        self.solution.inventory_df.to_parquet(inventory_path, index=False)

        with open(kpis_path, "w", encoding="utf-8") as f:
            json.dump(self.solution.kpis, f, indent=2)


def solve_supply_chain(
    demand_quantile: str = "p50",
    transport_cost_multiplier: float = 1.0,
    demand_multiplier: float = 1.0,
    disabled_nodes: Optional[list] = None,
    capacity_multiplier: float = 1.0,
    save: bool = True,
    output_path: Path = OPTIMAL_PLAN_PATH,
) -> SupplyChainSolution:
    """
    Función de alto nivel para configurar, resolver y opcionalmente guardar el plan óptimo.
    """
    model = SupplyChainModel(
        demand_quantile=demand_quantile,
        transport_cost_multiplier=transport_cost_multiplier,
        demand_multiplier=demand_multiplier,
        disabled_nodes=disabled_nodes,
        capacity_multiplier=capacity_multiplier,
    )
    solver = SupplyChainSolver(model)
    sol = solver.solve()
    if save and sol.is_optimal:
        solver.save_solution(output_path=output_path)
    return sol


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 RESOLVIENDO MOTOR DE OPTIMIZACIÓN LOGÍSTICA PRESCRIPTIVA (MILP)")
    print("=" * 70)
    sol = solve_supply_chain(demand_quantile="p50", save=True)
    print(f"Estado del Solver:      {sol.status}")
    print(f"Tiempo de Resolución:   {sol.solve_time_seconds:.3f} s")
    print(f"Costo Logístico Total:  ${sol.total_cost:,.2f} USD")
    print(f"  ├─ Transporte Fulfillment:    ${sol.kpis['fulfillment_transport_cost']:,.2f} USD")
    print(f"  ├─ Transporte Reabastecimiento:${sol.kpis['replenishment_transport_cost']:,.2f} USD")
    print(f"  ├─ Almacenamiento (Holding):   ${sol.kpis['holding_cost']:,.2f} USD")
    print(f"  └─ Penalización Stockout:      ${sol.kpis['stockout_cost']:,.2f} USD")
    print(f"Nivel de Servicio (OTIF):       {sol.kpis['otif_pct']:.2f}%")
    print(f"Tasa de Cross-Fulfillment:      {sol.kpis['cross_fulfillment_rate_pct']:.2f}%")
    print(f"Utilización Promedio DS:        {sol.kpis['avg_ds_utilization_pct']:.2f}%")
    print(f"✅ Plan guardado exitosamente en: {OPTIMAL_PLAN_PATH}")
    print("=" * 70)

