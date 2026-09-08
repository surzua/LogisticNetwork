"""
Módulo de Benchmark: Heurística Naive de Despacho Descentralizado (Regla Miope).
Permite contrastar la solución óptima del modelo MILP contra una regla heurística estándar
(despacho al nodo más cercano sin coordinación global ni optimización prescriptiva),
validando formalmente la Hipótesis H2 del proyecto.
"""

from typing import Dict, Optional, Any
from pathlib import Path
import numpy as np
import pandas as pd
import json

from src.config import (
    BENCHMARK_PLAN_PATH,
    OUTPUT_DATA_DIR,
)
from src.optimization.problem_builder import SupplyChainModel


class NaiveHeuristicSimulator:
    """
    Simulador de regla de despacho miope / greedy:
    1. Despacha pedidos desde la Dark Store más cercana a la zona del cliente.
    2. Si la Dark Store agota inventario, recurre a cross-fulfillment de emergencia desde el CD más cercano.
    3. Si el CD no tiene disponibilidad, se registra quiebre de stock (stockout).
    4. Reabastece reactivamente desde los CDs al finalizar el día con lead time de 1 día.
    """

    def __init__(self, model: SupplyChainModel):
        self.model = model

    def run(self) -> Dict[str, Any]:
        """
        Ejecuta la simulación período a período.
        """
        # Inicializar inventarios locales
        inv = {
            (item, node): float(self.model.initial_inventory.get((item, node), 0.0))
            for item in self.model.items
            for node in self.model.supply_nodes
        }

        # Mapeo de Dark Store más cercana y CD más cercano para cada zona
        top_idx = self.model.topology_df.set_index("node_id")
        zone_nearest_ds = {}
        zone_nearest_cd = {}

        for zone in self.model.zones:
            # DS más cercana
            best_ds = None
            min_cost_ds = float("inf")
            for ds in self.model.dark_stores:
                if (ds, zone) in self.model.transp_cost_per_m3_fulfillment:
                    c = self.model.transp_cost_per_m3_fulfillment[(ds, zone)]
                    if c < min_cost_ds and ds not in self.model.disabled_nodes:
                        min_cost_ds = c
                        best_ds = ds
            zone_nearest_ds[zone] = best_ds

            # CD más cercano
            best_cd = None
            min_cost_cd = float("inf")
            for cd in self.model.cds:
                if (cd, zone) in self.model.transp_cost_per_m3_fulfillment:
                    c = self.model.transp_cost_per_m3_fulfillment[(cd, zone)]
                    if c < min_cost_cd and cd not in self.model.disabled_nodes:
                        min_cost_cd = c
                        best_cd = cd
            zone_nearest_cd[zone] = best_cd

        fulfillment_records = []
        stockout_records = []
        replenishment_records = []
        inventory_records = []

        total_fulfillment_cost = 0.0
        total_replenishment_cost = 0.0
        total_holding_cost = 0.0
        total_stockout_cost = 0.0

        total_demand_units = 0.0
        fulfilled_units = 0.0
        stockout_units = 0.0
        cross_units = 0.0

        # Órdenes en tránsito de reabastecimiento: {arrival_date_idx: {(item, cd, ds): qty}}
        in_transit_replenishments = {}

        for t_idx, t_str in enumerate(self.model.time_periods):
            # 1. Llegada de reabastecimientos planificados para hoy
            if t_idx in in_transit_replenishments:
                for (item, c, d), qty in in_transit_replenishments[t_idx].items():
                    inv[(item, d)] += qty

            # 2. Despacho de demanda del día
            orders_placed_for_tomorrow = {}

            for zone in self.model.zones:
                for item in self.model.items:
                    dem = self.model.demand.get((item, zone, t_str), 0.0)
                    total_demand_units += dem
                    rem_dem = dem

                    closest_ds = zone_nearest_ds.get(zone)
                    closest_cd = zone_nearest_cd.get(zone)

                    # Intento 1: Fulfillment desde Dark Store local
                    if closest_ds is not None and inv[(item, closest_ds)] > 0 and rem_dem > 0:
                        qty_ds = min(rem_dem, inv[(item, closest_ds)])
                        inv[(item, closest_ds)] -= qty_ds
                        rem_dem -= qty_ds
                        fulfilled_units += qty_ds

                        cost_m3 = self.model.transp_cost_per_m3_fulfillment[(closest_ds, zone)]
                        t_cost = qty_ds * self.model.volume_m3[item] * cost_m3
                        total_fulfillment_cost += t_cost

                        fulfillment_records.append({
                            "date": pd.to_datetime(t_str),
                            "item_id": item,
                            "from_node_id": closest_ds,
                            "to_zone_id": zone,
                            "units": round(qty_ds, 4),
                            "transport_cost": round(t_cost, 2),
                            "is_cross_fulfillment": False,
                        })

                    # Intento 2: Cross-fulfillment de emergencia desde CD
                    if rem_dem > 0 and closest_cd is not None:
                        # Si el CD tiene stock (o asume abastecimiento entrante)
                        qty_cd = rem_dem
                        inv[(item, closest_cd)] = max(0.0, inv[(item, closest_cd)] - qty_cd)
                        rem_dem = 0.0
                        fulfilled_units += qty_cd
                        cross_units += qty_cd

                        cost_m3 = self.model.transp_cost_per_m3_fulfillment[(closest_cd, zone)]
                        t_cost = qty_cd * self.model.volume_m3[item] * cost_m3
                        total_fulfillment_cost += t_cost

                        fulfillment_records.append({
                            "date": pd.to_datetime(t_str),
                            "item_id": item,
                            "from_node_id": closest_cd,
                            "to_zone_id": zone,
                            "units": round(qty_cd, 4),
                            "transport_cost": round(t_cost, 2),
                            "is_cross_fulfillment": True,
                        })

                    # Si aún queda demanda: Stockout
                    if rem_dem > 0:
                        stockout_units += rem_dem
                        s_cost = rem_dem * self.model.stockout_cost[item]
                        total_stockout_cost += s_cost
                        stockout_records.append({
                            "date": pd.to_datetime(t_str),
                            "item_id": item,
                            "zone_id": zone,
                            "stockout_units": round(rem_dem, 4),
                            "stockout_cost": round(s_cost, 2),
                        })

            # 3. Política de Reabastecimiento Reactivo (Order-up-to)
            # Cada DS ordena a su CD más cercano lo que vendió hoy si no excede su capacidad
            for ds in self.model.dark_stores:
                if ds in self.model.disabled_nodes:
                    continue

                best_cd = self.model.cds[0]  # CD predeterminado
                # Calcular espacio disponible
                current_vol = sum(inv[(it, ds)] * self.model.volume_m3[it] for it in self.model.items)
                space_avail = max(0.0, self.model.capacity[ds] * 0.70 - current_vol)

                for item in self.model.items:
                    init_target = self.model.initial_inventory.get((item, ds), 0.0)
                    deficit = max(0.0, init_target - inv[(item, ds)])
                    if deficit > 0 and space_avail > 0:
                        order_qty = min(deficit, space_avail / self.model.volume_m3[item])
                        if order_qty > 0.1:
                            space_avail -= order_qty * self.model.volume_m3[item]
                            next_t = t_idx + 1
                            if next_t not in in_transit_replenishments:
                                in_transit_replenishments[next_t] = {}
                            in_transit_replenishments[next_t][(item, best_cd, ds)] = order_qty

                            # Costo de transporte de reabastecimiento
                            cost_m3 = self.model.transp_cost_per_m3_replenishment.get((best_cd, ds), 0.045)
                            r_cost = order_qty * self.model.volume_m3[item] * cost_m3
                            total_replenishment_cost += r_cost

                            replenishment_records.append({
                                "date": pd.to_datetime(t_str),
                                "item_id": item,
                                "from_cd_id": best_cd,
                                "to_ds_id": ds,
                                "units": round(order_qty, 4),
                                "transport_cost": round(r_cost, 2),
                            })

            # 4. Cálculo de Holding Costs al final del día
            for node in self.model.supply_nodes:
                if node in self.model.disabled_nodes:
                    continue
                node_vol = 0.0
                for item in self.model.items:
                    qty = inv[(item, node)]
                    vol = qty * self.model.volume_m3[item]
                    node_vol += vol
                    h_cost = vol * self.model.holding_cost[node]
                    total_holding_cost += h_cost

                    inventory_records.append({
                        "date": pd.to_datetime(t_str),
                        "node_id": node,
                        "item_id": item,
                        "units": round(qty, 4),
                        "volume_m3": round(vol, 4),
                        "holding_cost": round(h_cost, 2),
                    })

        total_transp = total_fulfillment_cost + total_replenishment_cost
        total_cost = total_transp + total_holding_cost + total_stockout_cost

        otif_pct = round(100.0 * fulfilled_units / max(1.0, total_demand_units), 2)
        cross_rate = round(100.0 * cross_units / max(1.0, fulfilled_units), 2)

        results = {
            "total_cost": round(total_cost, 2),
            "fulfillment_transport_cost": round(total_fulfillment_cost, 2),
            "replenishment_transport_cost": round(total_replenishment_cost, 2),
            "total_transport_cost": round(total_transp, 2),
            "holding_cost": round(total_holding_cost, 2),
            "stockout_cost": round(total_stockout_cost, 2),
            "total_demand_units": round(total_demand_units, 1),
            "fulfilled_units": round(fulfilled_units, 1),
            "stockout_units": round(stockout_units, 1),
            "otif_pct": otif_pct,
            "cross_fulfillment_units": round(cross_units, 1),
            "cross_fulfillment_rate_pct": cross_rate,
            "fulfillment_df": pd.DataFrame(fulfillment_records),
            "replenishment_df": pd.DataFrame(replenishment_records),
            "inventory_df": pd.DataFrame(inventory_records),
            "stockouts_df": pd.DataFrame(stockout_records),
        }
        return results

    def save_results(self, results: Dict[str, Any], output_path: Path = BENCHMARK_PLAN_PATH) -> None:
        """Exporta los resultados del benchmark a parquet y json."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results["fulfillment_df"].to_parquet(output_path, index=False)

        kpis = {k: v for k, v in results.items() if not isinstance(v, pd.DataFrame)}
        kpis_path = output_path.parent / "benchmark_kpis.json"
        with open(kpis_path, "w", encoding="utf-8") as f:
            json.dump(kpis, f, indent=2)


def run_benchmark(
    demand_quantile: str = "p50",
    transport_cost_multiplier: float = 1.0,
    demand_multiplier: float = 1.0,
    disabled_nodes: Optional[list] = None,
    save: bool = True,
    output_path: Path = BENCHMARK_PLAN_PATH,
) -> Dict[str, Any]:
    """Función de alto nivel para ejecutar el benchmark naive."""
    model = SupplyChainModel(
        demand_quantile=demand_quantile,
        transport_cost_multiplier=transport_cost_multiplier,
        demand_multiplier=demand_multiplier,
        disabled_nodes=disabled_nodes,
    )
    sim = NaiveHeuristicSimulator(model)
    res = sim.run()
    if save:
        sim.save_results(res, output_path=output_path)
    return res


if __name__ == "__main__":
    print("=" * 70)
    print("📊 EJECUTANDO BENCHMARK LOGÍSTICO (HEURÍSTICA NAIVE DESCENTRALIZADA)")
    print("=" * 70)
    res = run_benchmark(demand_quantile="p50", save=True)
    print(f"Costo Logístico Total:  ${res['total_cost']:,.2f} USD")
    print(f"  ├─ Transporte Fulfillment:    ${res['fulfillment_transport_cost']:,.2f} USD")
    print(f"  ├─ Transporte Reabastecimiento:${res['replenishment_transport_cost']:,.2f} USD")
    print(f"  ├─ Almacenamiento (Holding):   ${res['holding_cost']:,.2f} USD")
    print(f"  └─ Penalización Stockout:      ${res['stockout_cost']:,.2f} USD")
    print(f"Nivel de Servicio (OTIF):       {res['otif_pct']:.2f}%")
    print(f"Tasa de Cross-Fulfillment:      {res['cross_fulfillment_rate_pct']:.2f}%")
    print(f"✅ Resultados guardados en: {BENCHMARK_PLAN_PATH}")
    print("=" * 70)

