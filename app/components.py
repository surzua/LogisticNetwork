"""
Componentes auxiliares y utilidades de visualización para el Dashboard de Streamlit.
Incluye carga de datos, ejecución de simulaciones What-If, construcción de mapas PyDeck
y generación de gráficos interactivos con Plotly.
"""

import sys
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Asegurar que el directorio raíz del proyecto esté en sys.path para resolver 'src'
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.config import (
    TOPOLOGY_PATH,
    TRANSPORT_MATRIX_PATH,
    ITEMS_METADATA_PATH,
    FORECAST_OUTPUT_PATH,
    OPTIMAL_PLAN_PATH,
    BENCHMARK_PLAN_PATH,
    DEMAND_ZONES,
    CD_NODES,
    DARK_STORE_NODES,
    SKU_CATALOG,
)
from src.optimization.solver import solve_supply_chain, SupplyChainSolution
from src.optimization.benchmark import run_benchmark


@st.cache_data(show_spinner=False)
def load_network_data() -> Dict[str, Any]:
    """Carga y estructura los datos maestros de la red y catálogo."""
    topology_df = pd.read_parquet(TOPOLOGY_PATH)
    transport_df = pd.read_parquet(TRANSPORT_MATRIX_PATH)
    items_df = pd.read_parquet(ITEMS_METADATA_PATH)
    forecast_df = pd.read_parquet(FORECAST_OUTPUT_PATH)

    # Construir DataFrame georreferenciado de Zonas de Demanda
    zones_records = [
        {
            "zone_id": z_id,
            "zone_name": z_info["name"],
            "lat": z_info["lat"],
            "lon": z_info["lon"],
        }
        for z_id, z_info in DEMAND_ZONES.items()
    ]
    zones_df = pd.DataFrame(zones_records)

    # Mapeo rápido de coordenadas para nodos y zonas
    coords_dict = {}
    for _, row in topology_df.iterrows():
        coords_dict[row["node_id"]] = (row["lat"], row["lon"])
    for _, row in zones_df.iterrows():
        coords_dict[row["zone_id"]] = (row["lat"], row["lon"])

    return {
        "topology_df": topology_df,
        "transport_df": transport_df,
        "items_df": items_df,
        "forecast_df": forecast_df,
        "zones_df": zones_df,
        "coords_dict": coords_dict,
    }


@st.cache_data(show_spinner=False)
def load_baseline_results() -> Dict[str, Any]:
    """Carga los resultados precalculados del caso base (MILP vs Naive)."""
    optimal_fulfillment = pd.read_parquet(OPTIMAL_PLAN_PATH)
    benchmark_fulfillment = pd.read_parquet(BENCHMARK_PLAN_PATH)

    kpis_optimal_path = OPTIMAL_PLAN_PATH.parent / "optimal_kpis.json"
    kpis_benchmark_path = BENCHMARK_PLAN_PATH.parent / "benchmark_kpis.json"
    inventory_path = OPTIMAL_PLAN_PATH.parent / "optimal_inventory_plan.parquet"
    replenishment_path = OPTIMAL_PLAN_PATH.parent / "optimal_replenishment_plan.parquet"

    with open(kpis_optimal_path, "r", encoding="utf-8") as f:
        optimal_kpis = json.load(f)

    with open(kpis_benchmark_path, "r", encoding="utf-8") as f:
        benchmark_kpis = json.load(f)

    inventory_df = pd.read_parquet(inventory_path) if inventory_path.exists() else pd.DataFrame()
    replenishment_df = pd.read_parquet(replenishment_path) if replenishment_path.exists() else pd.DataFrame()

    return {
        "optimal_fulfillment": optimal_fulfillment,
        "benchmark_fulfillment": benchmark_fulfillment,
        "optimal_kpis": optimal_kpis,
        "benchmark_kpis": benchmark_kpis,
        "inventory_df": inventory_df,
        "replenishment_df": replenishment_df,
    }


@st.cache_data(show_spinner=False)
def run_scenario_optimization(
    demand_quantile: str = "p50",
    transport_cost_multiplier: float = 1.0,
    demand_multiplier: float = 1.0,
    disabled_nodes_tuple: Tuple[str, ...] = (),
    capacity_multiplier: float = 1.0,
) -> Dict[str, Any]:
    """
    Ejecuta el modelo MILP y el benchmark naive para un escenario What-If determinado.
    Usa tupla para `disabled_nodes` para permitir el cacheo nativo de Streamlit.
    """
    disabled_nodes_list = list(disabled_nodes_tuple) if disabled_nodes_tuple else None

    # Resolver MILP
    sol_milp: SupplyChainSolution = solve_supply_chain(
        demand_quantile=demand_quantile,
        transport_cost_multiplier=transport_cost_multiplier,
        demand_multiplier=demand_multiplier,
        disabled_nodes=disabled_nodes_list,
        capacity_multiplier=capacity_multiplier,
        save=False,
    )

    # Resolver Benchmark Naive
    res_bench: Dict[str, Any] = run_benchmark(
        demand_quantile=demand_quantile,
        transport_cost_multiplier=transport_cost_multiplier,
        demand_multiplier=demand_multiplier,
        disabled_nodes=disabled_nodes_list,
        save=False,
    )

    return {
        "sol_milp": sol_milp,
        "res_bench": res_bench,
        "optimal_kpis": sol_milp.kpis if sol_milp.is_optimal else {},
        "benchmark_kpis": {k: v for k, v in res_bench.items() if not isinstance(v, pd.DataFrame)},
        "fulfillment_df": sol_milp.fulfillment_df,
        "benchmark_fulfillment_df": res_bench.get("fulfillment_df", pd.DataFrame()),
        "replenishment_df": sol_milp.replenishment_df,
        "inventory_df": sol_milp.inventory_df,
        "stockouts_df": sol_milp.stockouts_df,
    }


def prepare_map_layers_data(
    fulfillment_df: pd.DataFrame,
    coords_dict: Dict[str, Tuple[float, float]],
    date_filter: Optional[str] = None,
    item_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Enriquece el DataFrame de despachos con coordenadas geográficas de origen y destino,
    colores RGBA y anchos para las capas de arcos de PyDeck.
    """
    if fulfillment_df.empty:
        return pd.DataFrame()

    df = fulfillment_df.copy()
    if date_filter and "date" in df.columns:
        df = df[df["date"].astype(str) == str(date_filter)]
    if item_filter and item_filter != "TODOS" and "item_id" in df.columns:
        df = df[df["item_id"] == item_filter]

    if df.empty:
        return pd.DataFrame()

    # Agregar flujos acumulados por par origen-destino
    flows = df.groupby(["from_node_id", "to_zone_id", "node_type", "is_cross_fulfillment"]).agg({
        "units": "sum",
        "volume_m3": "sum",
        "transport_cost": "sum",
        "distance_km": "first",
    }).reset_index()

    # Mapear coordenadas
    flows["from_lat"] = flows["from_node_id"].map(lambda x: coords_dict.get(x, (0, 0))[0])
    flows["from_lon"] = flows["from_node_id"].map(lambda x: coords_dict.get(x, (0, 0))[1])
    flows["to_lat"] = flows["to_zone_id"].map(lambda x: coords_dict.get(x, (0, 0))[0])
    flows["to_lon"] = flows["to_zone_id"].map(lambda x: coords_dict.get(x, (0, 0))[1])

    # Colores:
    # Verde esmeralda para despacho regular desde Dark Store [0, 230, 118, 160]
    # Naranja / Ámbar fuego para Cross-Fulfillment desde CD [255, 109, 0, 200]
    def assign_color(row):
        if row["is_cross_fulfillment"]:
            return [255, 109, 0, 200]
        elif row["node_type"] == "CD":
            return [30, 136, 229, 180]
        else:
            return [0, 230, 118, 160]

    flows["color"] = flows.apply(assign_color, axis=1)

    # Ancho relativo para visualización en PyDeck
    max_units = flows["units"].max() if not flows.empty and flows["units"].max() > 0 else 1.0
    flows["stroke_width"] = (flows["units"] / max_units * 5.0 + 1.5).round(1)

    return flows


def create_network_map(
    flows_df: pd.DataFrame,
    topology_df: pd.DataFrame,
    zones_df: pd.DataFrame,
    disabled_nodes: Optional[List[str]] = None,
) -> pdk.Deck:
    """
    Construye la visualización cartográfica completa con PyDeck.
    Combina capas de Nodos CDs, Dark Stores (activas/inactivas), Zonas de demanda
    y Arcos de despacho volumétrico 3D.
    """
    disabled_set = set(disabled_nodes or [])

    # Preparar datos de nodos logísticos de suministro (CDs y Dark Stores)
    nodes_df = topology_df[topology_df["node_type"].isin(["CD", "DarkStore"])].copy()
    nodes_df["is_disabled"] = nodes_df["node_id"].isin(disabled_set)

    def node_color(row):
        if row["is_disabled"]:
            return [244, 67, 54, 220]  # Rojo contingencia
        if row["node_type"] == "CD":
            return [33, 150, 243, 220]  # Azul CD
        return [0, 200, 83, 220]  # Verde Dark Store

    def node_radius(row):
        if row["node_type"] == "CD":
            return 800
        return 500

    nodes_df["color"] = nodes_df.apply(node_color, axis=1)
    nodes_df["radius"] = nodes_df.apply(node_radius, axis=1)

    # Capa 1: Nodos de Suministro (CDs y Dark Stores)
    supply_layer = pdk.Layer(
        "ScatterplotLayer",
        data=nodes_df,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
    )

    # Capa 2: Zonas de Demanda (Clientes)
    zones_plot_df = zones_df.copy()
    zones_plot_df["color"] = [[255, 179, 0, 180] for _ in range(len(zones_plot_df))]
    zones_plot_df["radius"] = 350

    zones_layer = pdk.Layer(
        "ScatterplotLayer",
        data=zones_plot_df,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
    )

    # Capa 3: Arcos de Flujo Logístico
    layers = [zones_layer, supply_layer]

    if not flows_df.empty:
        arc_layer = pdk.Layer(
            "ArcLayer",
            data=flows_df,
            get_source_position=["from_lon", "from_lat"],
            get_target_position=["to_lon", "to_lat"],
            get_source_color="color",
            get_target_color="color",
            get_width="stroke_width",
            pickable=True,
            auto_highlight=True,
        )
        layers.append(arc_layer)

    # Vista inicial centrada en el Gran Santiago
    view_state = pdk.ViewState(
        latitude=-33.4500,
        longitude=-70.6500,
        zoom=10.8,
        pitch=45,
        bearing=15,
    )

    tooltip = {
        "html": """
        <div style="font-family: sans-serif; font-size: 12px; color: #fff; background: rgba(20,25,35,0.9); padding: 8px 12px; border-radius: 6px; border: 1px solid #444;">
            <b>{node_name}</b> {zone_name} <br/>
            <span><b>ID:</b> {node_id}{zone_id}</span><br/>
            <span><b>Tipo:</b> {node_type}</span><br/>
            <span><b>Capacidad:</b> {capacity_m3} m³</span><br/>
            <span><b>Despacho:</b> {units} unid. ({volume_m3} m³)</span><br/>
            <span><b>Ruta:</b> {from_node_id} ➔ {to_zone_id}</span><br/>
            <span><b>Distancia:</b> {distance_km} km</span>
        </div>
        """,
        "style": {"backgroundColor": "transparent", "color": "white"},
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v11",
        tooltip=tooltip,
    )

    return deck


def create_cost_comparison_chart(
    optimal_kpis: Dict[str, Any],
    benchmark_kpis: Dict[str, Any],
) -> go.Figure:
    """Genera gráfico comparativo de barras agrupadas con el desglose de costos."""
    categories = [
        "Fulfillment",
        "Reabastecimiento",
        "Almacenamiento (Holding)",
        "Quiebres (Stockout)",
        "Costo Total Red",
    ]

    opt_values = [
        optimal_kpis.get("fulfillment_transport_cost", 0.0),
        optimal_kpis.get("replenishment_transport_cost", 0.0),
        optimal_kpis.get("holding_cost", 0.0),
        optimal_kpis.get("stockout_cost", 0.0),
        optimal_kpis.get("total_cost", 0.0),
    ]

    bench_values = [
        benchmark_kpis.get("fulfillment_transport_cost", 0.0),
        benchmark_kpis.get("replenishment_transport_cost", 0.0),
        benchmark_kpis.get("holding_cost", 0.0),
        benchmark_kpis.get("stockout_cost", 0.0),
        benchmark_kpis.get("total_cost", 0.0),
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="MILP Prescriptivo (Óptimo)",
        x=categories,
        y=opt_values,
        marker_color="#00E676",
        text=[f"${v:,.1f}" for v in opt_values],
        textposition="auto",
    ))

    fig.add_trace(go.Bar(
        name="Heurística Naive (Regla Miope)",
        x=categories,
        y=bench_values,
        marker_color="#FF5252",
        text=[f"${v:,.1f}" for v in bench_values],
        textposition="auto",
    ))

    fig.update_layout(
        title="<b>Comparación de Costos Logísticos: Modelo MILP vs Heurística Naive</b>",
        barmode="group",
        template="plotly_dark",
        yaxis_title="Costo en USD",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
    )

    return fig


def create_inventory_utilization_chart(inventory_df: pd.DataFrame) -> go.Figure:
    """Genera gráfico temporal de ocupación de capacidad (%) en Dark Stores."""
    if inventory_df.empty:
        return go.Figure()

    ds_inv = inventory_df[inventory_df["node_type"] == "DarkStore"].copy()
    if ds_inv.empty:
        return go.Figure()

    fig = go.Figure()

    for node_id, group in ds_inv.groupby("node_id"):
        group_sorted = group.sort_values("date")
        fig.add_trace(go.Scatter(
            x=group_sorted["date"].astype(str),
            y=group_sorted["utilization_pct"],
            mode="lines+markers",
            name=node_id,
            line=dict(width=2),
            marker=dict(size=4),
        ))

    fig.add_hline(
        y=80.0,
        line_dash="dash",
        line_color="#FFA726",
        annotation_text="Alerta Saturación (80%)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="<b>Evolución Temporal de Utilización de Capacidad (%) por Dark Store</b>",
        xaxis_title="Fecha",
        yaxis_title="Utilización (%)",
        yaxis=dict(range=[0, 105]),
        template="plotly_dark",
        margin=dict(l=40, r=20, t=60, b=40),
        height=360,
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
    )

    return fig


def create_demand_forecast_chart(
    forecast_df: pd.DataFrame,
    selected_sku: str,
    selected_zone: str,
) -> go.Figure:
    """Muestra la serie de pronósticos con banda de incertidumbre P10 - P90."""
    sub = forecast_df[
        (forecast_df["item_id"] == selected_sku) & (forecast_df["zone_id"] == selected_zone)
    ].sort_values("date")

    if sub.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sub["date"].astype(str),
        y=sub["demand_p90"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="P90",
    ))

    fig.add_trace(go.Scatter(
        x=sub["date"].astype(str),
        y=sub["demand_p10"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(33, 150, 243, 0.25)",
        name="Intervalo de Confianza (P10 - P90)",
    ))

    fig.add_trace(go.Scatter(
        x=sub["date"].astype(str),
        y=sub["demand_p50"],
        mode="lines+markers",
        line=dict(color="#00E676", width=3),
        marker=dict(size=6),
        name="Pronóstico P50 (Mediana)",
    ))

    fig.update_layout(
        title=f"<b>Pronóstico Probabilístico: {selected_sku} en {selected_zone}</b>",
        xaxis_title="Fecha",
        yaxis_title="Unidades Demandadas",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=60, b=40),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
