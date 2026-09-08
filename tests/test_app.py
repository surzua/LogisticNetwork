"""
Pruebas unitarias para los componentes y utilidades de la aplicación Streamlit (app/components.py).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import pydeck as pdk
import plotly.graph_objects as go

from app.components import (
    load_network_data,
    load_baseline_results,
    run_scenario_optimization,
    prepare_map_layers_data,
    create_network_map,
    create_cost_comparison_chart,
    create_inventory_utilization_chart,
    create_demand_forecast_chart,
)


def test_load_network_data():
    """Verifica que los datos maestros de red se carguen con la estructura esperada."""
    data = load_network_data()
    assert "topology_df" in data
    assert "transport_df" in data
    assert "items_df" in data
    assert "forecast_df" in data
    assert "zones_df" in data
    assert "coords_dict" in data

    assert len(data["topology_df"]) == 30  # 2 CDs + 8 DS + 20 Zonas en grafo
    supply_nodes = data["topology_df"][data["topology_df"]["node_type"].isin(["CD", "DarkStore"])]
    assert len(supply_nodes) == 10         # 2 CDs + 8 DS
    assert len(data["zones_df"]) == 20     # 20 Zonas
    assert len(data["coords_dict"]) == 30  # 10 nodos suministro + 20 zonas


def test_load_baseline_results():
    """Verifica la carga de resultados base de optimización y benchmark."""
    base = load_baseline_results()
    assert "optimal_fulfillment" in base
    assert "benchmark_fulfillment" in base
    assert "optimal_kpis" in base
    assert "benchmark_kpis" in base

    assert not base["optimal_fulfillment"].empty
    assert base["optimal_kpis"]["total_cost"] > 0
    assert base["benchmark_kpis"]["total_cost"] > base["optimal_kpis"]["total_cost"]


def test_prepare_map_layers_data():
    """Verifica que los arcos de flujo se enriquezcan con coordenadas y atributos PyDeck."""
    data = load_network_data()
    base = load_baseline_results()

    flows_df = prepare_map_layers_data(
        fulfillment_df=base["optimal_fulfillment"],
        coords_dict=data["coords_dict"],
        date_filter=None,
        item_filter="TODOS",
    )

    assert not flows_df.empty
    expected_cols = [
        "from_node_id", "to_zone_id", "from_lat", "from_lon",
        "to_lat", "to_lon", "units", "color", "stroke_width",
    ]
    for col in expected_cols:
        assert col in flows_df.columns

    # Verificar que las coordenadas no sean cero
    assert (flows_df["from_lat"] != 0).all()
    assert (flows_df["to_lat"] != 0).all()


def test_create_network_map():
    """Verifica que PyDeck construya el Deck con las capas esperadas."""
    data = load_network_data()
    base = load_baseline_results()

    flows_df = prepare_map_layers_data(
        fulfillment_df=base["optimal_fulfillment"],
        coords_dict=data["coords_dict"],
    )

    deck = create_network_map(
        flows_df=flows_df,
        topology_df=data["topology_df"],
        zones_df=data["zones_df"],
        disabled_nodes=["DS_PROVIDENCIA"],
    )

    assert isinstance(deck, pdk.Deck)
    assert len(deck.layers) == 3  # zones_layer, supply_layer, arc_layer


def test_plotly_chart_builders():
    """Verifica que los generadores de gráficos Plotly produzcan figuras válidas."""
    base = load_baseline_results()
    data = load_network_data()

    # 1. Gráfico comparativo de costos
    cost_fig = create_cost_comparison_chart(base["optimal_kpis"], base["benchmark_kpis"])
    assert isinstance(cost_fig, go.Figure)
    assert len(cost_fig.data) == 2  # MILP trace y Benchmark trace

    # 2. Gráfico de utilización de inventario
    inv_fig = create_inventory_utilization_chart(base["inventory_df"])
    assert isinstance(inv_fig, go.Figure)
    assert len(inv_fig.data) == 8  # 8 Dark Stores

    # 3. Gráfico de pronóstico con banda de incertidumbre
    first_sku = data["items_df"]["item_id"].iloc[0]
    first_zone = data["zones_df"]["zone_id"].iloc[0]
    fc_fig = create_demand_forecast_chart(data["forecast_df"], first_sku, first_zone)
    assert isinstance(fc_fig, go.Figure)
    assert len(fc_fig.data) >= 3  # P90, P10 intervalo y P50


def test_run_scenario_optimization():
    """Verifica que la ejecución de simulación What-If dinámica sea exitosa."""
    sim = run_scenario_optimization(
        demand_quantile="p50",
        transport_cost_multiplier=1.2,
        demand_multiplier=1.1,
        disabled_nodes_tuple=("DS_PROVIDENCIA",),
        capacity_multiplier=1.0,
    )

    assert "sol_milp" in sim
    assert "res_bench" in sim
    assert sim["sol_milp"].is_optimal
    assert sim["optimal_kpis"]["total_cost"] > 0
    assert not sim["fulfillment_df"].empty
