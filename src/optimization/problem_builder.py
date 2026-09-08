"""
Módulo de construcción del modelo de Programación Lineal Entera Mixta (MILP)
para la red de fulfillment y distribución logística en Santiago de Chile.
Utiliza PuLP para formular variables, restricciones y función objetivo.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import pulp

from src.config import (
    TOPOLOGY_PATH,
    TRANSPORT_MATRIX_PATH,
    ITEMS_METADATA_PATH,
    FORECAST_OUTPUT_PATH,
    DEFAULT_INITIAL_INVENTORY_COVERAGE_DAYS,
)


class SupplyChainModel:
    """
    Constructor y formulador del modelo de optimización de red logística multi-período.
    """

    def __init__(
        self,
        demand_quantile: str = "p50",
        transport_cost_multiplier: float = 1.0,
        demand_multiplier: float = 1.0,
        disabled_nodes: Optional[List[str]] = None,
        capacity_multiplier: float = 1.0,
        initial_inventory_coverage_days: float = DEFAULT_INITIAL_INVENTORY_COVERAGE_DAYS,
        initial_inventory_custom: Optional[Dict[Tuple[str, str], float]] = None,
        topology_df: Optional[pd.DataFrame] = None,
        transport_df: Optional[pd.DataFrame] = None,
        items_df: Optional[pd.DataFrame] = None,
        forecast_df: Optional[pd.DataFrame] = None,
    ):
        """
        Inicializa los parámetros del modelo.
        """
        self.demand_quantile = demand_quantile.lower()
        self.transport_cost_multiplier = float(transport_cost_multiplier)
        self.demand_multiplier = float(demand_multiplier)
        self.disabled_nodes = set(disabled_nodes or [])
        self.capacity_multiplier = float(capacity_multiplier)
        self.initial_inventory_coverage_days = float(initial_inventory_coverage_days)
        self.initial_inventory_custom = initial_inventory_custom

        # Cargar DataFrames
        self.topology_df = topology_df if topology_df is not None else pd.read_parquet(TOPOLOGY_PATH)
        self.transport_df = transport_df if transport_df is not None else pd.read_parquet(TRANSPORT_MATRIX_PATH)
        self.items_df = items_df if items_df is not None else pd.read_parquet(ITEMS_METADATA_PATH)
        self.forecast_df = forecast_df if forecast_df is not None else pd.read_parquet(FORECAST_OUTPUT_PATH)

        self._prepare_sets_and_parameters()
        self._build_pulp_problem()

    def _prepare_sets_and_parameters(self) -> None:
        """
        Extrae conjuntos, índices y mapas de parámetros a partir de los datos.
        """
        # 1. Conjuntos de Nodos
        cd_mask = self.topology_df["node_type"] == "CD"
        ds_mask = self.topology_df["node_type"] == "DarkStore"
        zone_mask = self.topology_df["node_type"] == "DemandZone"

        self.cds = sorted(self.topology_df.loc[cd_mask, "node_id"].tolist())
        self.dark_stores = sorted(self.topology_df.loc[ds_mask, "node_id"].tolist())
        self.supply_nodes = self.cds + self.dark_stores
        self.zones = sorted(self.topology_df.loc[zone_mask, "node_id"].tolist())

        # Capacidades y costos de holding
        top_indexed = self.topology_df.set_index("node_id")
        self.capacity = {}
        self.holding_cost = {}
        for node in self.supply_nodes:
            if node in self.disabled_nodes:
                self.capacity[node] = 0.0
            else:
                self.capacity[node] = float(top_indexed.loc[node, "capacity_m3"]) * self.capacity_multiplier
            self.holding_cost[node] = float(top_indexed.loc[node, "holding_cost_unit"])

        # 2. Catálogo de SKUs
        items_indexed = self.items_df.set_index("item_id")
        self.items = sorted(self.items_df["item_id"].tolist())
        self.volume_m3 = {item: float(items_indexed.loc[item, "volume_m3"]) for item in self.items}
        self.stockout_cost = {item: float(items_indexed.loc[item, "stockout_cost"]) for item in self.items}
        self.base_price = {item: float(items_indexed.loc[item, "base_price"]) for item in self.items}

        # 3. Horizonte Temporal
        dates_sorted = sorted(self.forecast_df["date"].unique())
        self.time_periods = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates_sorted]
        self.num_periods = len(self.time_periods)

        # 4. Parámetros de Demanda
        col_name = "demand_p50"
        if self.demand_quantile == "p10":
            col_name = "demand_p10"
        elif self.demand_quantile == "p90":
            col_name = "demand_p90"
        elif self.demand_quantile in ("actual", "actual_demand"):
            col_name = "actual_demand"

        self.demand = {}
        for _, row in self.forecast_df.iterrows():
            d_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
            item = row["item_id"]
            zone = row["zone_id"]
            val = float(row[col_name]) * self.demand_multiplier
            self.demand[(item, zone, d_str)] = max(0.0, val)

        # 5. Costos y Enlaces de Transporte
        # Fulfillment arcs (CD -> ZONE y DS -> ZONE)
        self.fulfillment_arcs = []
        self.transp_cost_per_m3_fulfillment = {}
        self.lead_time_fulfillment = {}

        # Replenishment arcs (CD -> DS)
        self.replenishment_arcs = []
        self.transp_cost_per_m3_replenishment = {}

        tm_records = self.transport_df.to_dict("records")
        for row in tm_records:
            u, w = row["from_node_id"], row["to_node_id"]
            cost_m3 = float(row["transport_cost_per_m3"]) * self.transport_cost_multiplier
            lead_time = float(row["lead_time_days"])

            # Arcos hacia Zonas de Clientes
            if w in self.zones and u in self.supply_nodes:
                self.fulfillment_arcs.append((u, w))
                self.transp_cost_per_m3_fulfillment[(u, w)] = cost_m3
                self.lead_time_fulfillment[(u, w)] = lead_time

            # Arcos de Reabastecimiento CD -> DS
            elif u in self.cds and w in self.dark_stores:
                self.replenishment_arcs.append((u, w))
                self.transp_cost_per_m3_replenishment[(u, w)] = cost_m3

        # 6. Inventario Inicial I_{i, k, 0}
        self.initial_inventory = self._calculate_initial_inventory()

    def _calculate_initial_inventory(self) -> Dict[Tuple[str, str], float]:
        """
        Determina el nivel de inventario inicial para cada SKU y nodo.
        Si se suministra un diccionario custom, se utiliza directamente.
        En caso contrario, se computa una cobertura proporcional a la demanda.
        """
        if self.initial_inventory_custom is not None:
            return self.initial_inventory_custom

        init_inv = {}

        # Calcular demanda media diaria por SKU y Zona
        zone_sku_avg = {}
        for (item, zone, t), dem in self.demand.items():
            zone_sku_avg[(item, zone)] = zone_sku_avg.get((item, zone), 0.0) + dem / max(1, self.num_periods)

        # Mapear cada zona a su Dark Store más cercana
        zone_to_nearest_ds = {}
        for zone in self.zones:
            best_ds = None
            best_cost = float("inf")
            for ds in self.dark_stores:
                if (ds, zone) in self.transp_cost_per_m3_fulfillment:
                    cost = self.transp_cost_per_m3_fulfillment[(ds, zone)]
                    if cost < best_cost:
                        best_cost = cost
                        best_ds = ds
            zone_to_nearest_ds[zone] = best_ds

        # Inventario en Dark Stores: cobertura de días de las zonas asignadas
        for ds in self.dark_stores:
            if ds in self.disabled_nodes:
                for item in self.items:
                    init_inv[(item, ds)] = 0.0
                continue

            # Demanda agregada de zonas asignadas a este DS
            for item in self.items:
                assigned_demand = sum(
                    zone_sku_avg.get((item, z), 0.0)
                    for z in self.zones
                    if zone_to_nearest_ds.get(z) == ds
                )
                qty = assigned_demand * self.initial_inventory_coverage_days
                init_inv[(item, ds)] = round(qty, 2)

            # Control de volumen máximo inicial (no superar 50% de la capacidad del DS)
            total_vol = sum(init_inv[(item, ds)] * self.volume_m3[item] for item in self.items)
            max_allowed = self.capacity[ds] * 0.50
            if total_vol > max_allowed and total_vol > 0:
                scale = max_allowed / total_vol
                for item in self.items:
                    init_inv[(item, ds)] = round(init_inv[(item, ds)] * scale, 2)

        # Inventario en Centros de Distribución (CDs)
        # Ample stock inicial para soportar reabastecimientos del horizonte
        total_daily_network = {
            item: sum(zone_sku_avg.get((item, z), 0.0) for z in self.zones)
            for item in self.items
        }
        for cd in self.cds:
            if cd in self.disabled_nodes:
                for item in self.items:
                    init_inv[(item, cd)] = 0.0
                continue

            for item in self.items:
                # 7 días de demanda de red repartidos entre los 2 CDs
                cd_qty = (total_daily_network[item] * 7.0) / len(self.cds)
                init_inv[(item, cd)] = round(cd_qty, 2)

        return init_inv

    def _build_pulp_problem(self) -> None:
        """
        Construye el problema de optimización en PuLP con variables, restricciones y objetivo.
        """
        prob = pulp.LpProblem("SupplyChainNetworkOptimizer", pulp.LpMinimize)

        # --- 1. VARIABLES DE DECISIÓN ---
        # X[i, k, j, t]: Cantidad de producto i despachado desde nodo k a zona j en día t
        X = {}
        for item in self.items:
            for (k, j) in self.fulfillment_arcs:
                if k in self.disabled_nodes:
                    continue
                for t in self.time_periods:
                    X[(item, k, j, t)] = pulp.LpVariable(
                        f"X_{item}_{k}_{j}_{t}", lowBound=0, cat=pulp.LpContinuous
                    )

        # R[i, c, d, t]: Reabastecimiento de producto i desde CD c hacia Dark Store d en día t
        R = {}
        for item in self.items:
            for (c, d) in self.replenishment_arcs:
                if c in self.disabled_nodes or d in self.disabled_nodes:
                    continue
                for t in self.time_periods:
                    R[(item, c, d, t)] = pulp.LpVariable(
                        f"R_{item}_{c}_{d}_{t}", lowBound=0, cat=pulp.LpContinuous
                    )

        # I[i, k, t]: Nivel de inventario de producto i en nodo k al final del día t
        I = {}
        for item in self.items:
            for k in self.supply_nodes:
                if k in self.disabled_nodes:
                    continue
                for t in self.time_periods:
                    I[(item, k, t)] = pulp.LpVariable(
                        f"I_{item}_{k}_{t}", lowBound=0, cat=pulp.LpContinuous
                    )

        # S[i, j, t]: Stockout / venta perdida de producto i en zona j en día t
        S = {}
        for item in self.items:
            for j in self.zones:
                for t in self.time_periods:
                    S[(item, j, t)] = pulp.LpVariable(
                        f"S_{item}_{j}_{t}", lowBound=0, cat=pulp.LpContinuous
                    )

        # Q[i, c, t]: Inbound de reposición externa hacia CD c en día t
        Q = {}
        for item in self.items:
            for c in self.cds:
                if c in self.disabled_nodes:
                    continue
                for t in self.time_periods:
                    Q[(item, c, t)] = pulp.LpVariable(
                        f"Q_{item}_{c}_{t}", lowBound=0, cat=pulp.LpContinuous
                    )

        # --- 2. FUNCIÓN OBJETIVO ---
        objective_terms = []

        # Costo de transporte de fulfillment (k -> j)
        for (item, k, j, t), x_var in X.items():
            unit_cost = self.transp_cost_per_m3_fulfillment[(k, j)] * self.volume_m3[item]
            objective_terms.append(unit_cost * x_var)

        # Costo de transporte de reabastecimiento (c -> d)
        for (item, c, d, t), r_var in R.items():
            unit_cost = self.transp_cost_per_m3_replenishment[(c, d)] * self.volume_m3[item]
            objective_terms.append(unit_cost * r_var)

        # Costo de almacenamiento diario (holding cost en nodo k)
        for (item, k, t), i_var in I.items():
            unit_cost = self.holding_cost[k] * self.volume_m3[item]
            objective_terms.append(unit_cost * i_var)

        # Penalización por stockout (quiebre en zona j)
        for (item, j, t), s_var in S.items():
            unit_cost = self.stockout_cost[item]
            objective_terms.append(unit_cost * s_var)

        # Costo marginal mínimo de inbound a CD para evitar compras infinitas innecesarias
        for (item, c, t), q_var in Q.items():
            objective_terms.append(0.01 * q_var)

        prob += pulp.lpSum(objective_terms), "TotalLogisticCost"

        # --- 3. RESTRICCIONES ---

        # 3.1 Satisfacción de Demanda
        for item in self.items:
            for j in self.zones:
                for t in self.time_periods:
                    dem_val = self.demand.get((item, j, t), 0.0)
                    available_x = [
                        X[(item, k, j, t)]
                        for k in self.supply_nodes
                        if (item, k, j, t) in X
                    ]
                    s_var = S[(item, j, t)]
                    prob += (
                        pulp.lpSum(available_x) + s_var == dem_val,
                        f"DemandSatisfaction_{item}_{j}_{t}",
                    )

        # 3.2 Balance de Inventario en Dark Stores (Lead time de reabastecimiento = 1 día)
        for t_idx, t in enumerate(self.time_periods):
            for item in self.items:
                for d in self.dark_stores:
                    if d in self.disabled_nodes:
                        continue

                    # Inventario previo
                    if t_idx == 0:
                        prev_inv = self.initial_inventory.get((item, d), 0.0)
                        replenish_in = 0.0
                    else:
                        t_prev = self.time_periods[t_idx - 1]
                        prev_inv = I[(item, d, t_prev)]
                        # Reabastecimiento enviado en t-1 llega en t
                        replenish_in = pulp.lpSum(
                            R[(item, c, d, t_prev)]
                            for c in self.cds
                            if (item, c, d, t_prev) in R
                        )

                    # Despachos salientes a clientes
                    outflows = pulp.lpSum(
                        X[(item, d, j, t)]
                        for j in self.zones
                        if (item, d, j, t) in X
                    )

                    prob += (
                        I[(item, d, t)] == prev_inv + replenish_in - outflows,
                        f"InventoryBalance_DS_{item}_{d}_{t}",
                    )

        # 3.3 Balance de Inventario en Centros de Distribución (CDs)
        for t_idx, t in enumerate(self.time_periods):
            for item in self.items:
                for c in self.cds:
                    if c in self.disabled_nodes:
                        continue

                    if t_idx == 0:
                        prev_inv = self.initial_inventory.get((item, c), 0.0)
                    else:
                        t_prev = self.time_periods[t_idx - 1]
                        prev_inv = I[(item, c, t_prev)]

                    inbound_q = Q.get((item, c, t), 0.0)

                    # Reabastecimientos despachados a Dark Stores en t
                    replenish_out = pulp.lpSum(
                        R[(item, c, d, t)]
                        for d in self.dark_stores
                        if (item, c, d, t) in R
                    )

                    # Despachos directos a clientes (cross-fulfillment)
                    fulfillment_out = pulp.lpSum(
                        X[(item, c, j, t)]
                        for j in self.zones
                        if (item, c, j, t) in X
                    )

                    prob += (
                        I[(item, c, t)] == prev_inv + inbound_q - replenish_out - fulfillment_out,
                        f"InventoryBalance_CD_{item}_{c}_{t}",
                    )

        # 3.4 Capacidad de Almacenamiento en Nodos (m3)
        for k in self.supply_nodes:
            if k in self.disabled_nodes:
                continue
            cap_m3 = self.capacity[k]
            for t in self.time_periods:
                vol_sum = pulp.lpSum(
                    self.volume_m3[item] * I[(item, k, t)]
                    for item in self.items
                    if (item, k, t) in I
                )
                prob += (vol_sum <= cap_m3, f"StorageCapacity_{k}_{t}")

        # Guardar referencias
        self.problem = prob
        self.decision_vars = {"X": X, "R": R, "I": I, "S": S, "Q": Q}
