"""
LogisticNetwork: Dashboard Interactivo de Optimización Prescriptiva y Simulador What-If.
Aplicación principal de Streamlit con visualización geoespacial PyDeck,
comparador prescriptivo vs naive y analítica de inventario/pronóstico.
"""

import sys
from pathlib import Path

# Asegurar que el directorio raíz del proyecto esté en sys.path para resolver 'src' y 'app'
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np

from src.config import (
    CD_NODES,
    DARK_STORE_NODES,
    DEMAND_ZONES,
    SKU_CATALOG,
    CIRCUITY_FACTOR,
)
try:
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
except ModuleNotFoundError:
    from components import (
        load_network_data,
        load_baseline_results,
        run_scenario_optimization,
        prepare_map_layers_data,
        create_network_map,
        create_cost_comparison_chart,
        create_inventory_utilization_chart,
        create_demand_forecast_chart,
    )

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="LogisticNetwork | Fulfillment Optimizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inyección de CSS para diseño premium
st.markdown(
    """
    <style>
    /* Estilos generales */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #9E9E9E;
        margin-bottom: 1.5rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background: rgba(0, 230, 118, 0.15);
        color: #00E676;
        border: 1px solid rgba(0, 230, 118, 0.3);
        margin-bottom: 0.8rem;
    }
    .kpi-card {
        background: #1E232F;
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
        border: 1px solid #2D3748;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F7FAFC;
        margin-top: 0.2rem;
    }
    .kpi-sub {
        font-size: 0.82rem;
        margin-top: 0.3rem;
    }
    .positive-delta {
        color: #00E676;
        font-weight: 600;
    }
    .negative-delta {
        color: #FF5252;
        font-weight: 600;
    }
    .legend-box {
        background: #1A202C;
        border: 1px solid #2D3748;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
        display: flex;
        gap: 1.5rem;
        flex-wrap: wrap;
        font-size: 0.85rem;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .dot {
        height: 12px;
        width: 12px;
        border-radius: 50%;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 2. CARGA DE DATOS
# ------------------------------------------------------------------------------
net_data = load_network_data()
topology_df = net_data["topology_df"]
zones_df = net_data["zones_df"]
items_df = net_data["items_df"]
forecast_df = net_data["forecast_df"]
coords_dict = net_data["coords_dict"]

baseline = load_baseline_results()

# ------------------------------------------------------------------------------
# 3. SIDEBAR: SIMULADOR WHAT-IF Y PARÁMETROS
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/delivery.png", width=64)
    st.markdown("## ⚙️ Simulador What-If")
    st.markdown("Ajusta las variables operativas de la red metropolitana en tiempo real.")

    # Presets rápidos
    st.markdown("### 🎯 Escenarios Predefinidos")
    preset = st.selectbox(
        "Cargar escenario rápido:",
        [
            "Personalizado",
            "Línea Base (Baseline)",
            "Shock Combustible (+20%)",
            "Falla Dark Store (Providencia cerrada)",
            "Pico de Demanda (+30%)",
            "Estrés Severo Combinado",
        ],
        index=0,
    )

    # Valores por defecto según preset
    default_quantile = "p50"
    default_transp = 1.0
    default_demand = 1.0
    default_cap = 1.0
    default_disabled = []

    if preset == "Línea Base (Baseline)":
        default_quantile = "p50"
        default_transp = 1.0
        default_demand = 1.0
        default_disabled = []
    elif preset == "Shock Combustible (+20%)":
        default_transp = 1.20
    elif preset == "Falla Dark Store (Providencia cerrada)":
        default_disabled = ["DS_PROVIDENCIA"]
    elif preset == "Pico de Demanda (+30%)":
        default_demand = 1.30
    elif preset == "Estrés Severo Combinado":
        default_transp = 1.35
        default_demand = 1.25
        default_disabled = ["DS_PROVIDENCIA", "DS_LAS_CONDES"]
        default_quantile = "p90"

    st.markdown("---")
    st.markdown("### 🎛️ Controles del Modelo")

    selected_quantile = st.radio(
        "Aversión al Riesgo (Pronóstico de Demanda):",
        options=["p10", "p50", "p90", "actual"],
        format_func=lambda x: {
            "p10": "P10 (Demanda Baja / Conservador)",
            "p50": "P50 (Mediana / Escenario Central)",
            "p90": "P90 (Demanda Alta / Stock de Seguridad)",
            "actual": "Demanda Observada (Perfect Foresight)",
        }.get(x, x),
        index=["p10", "p50", "p90", "actual"].index(default_quantile),
    )

    transp_mult = st.slider(
        "Multiplicador Costo Transporte (Combustible):",
        min_value=0.50,
        max_value=2.50,
        value=float(default_transp),
        step=0.05,
        format="%.2fx",
        help="Simula shocks de combustible, peajes o congestión urbana.",
    )

    demand_mult = st.slider(
        "Multiplicador de Demanda Global:",
        min_value=0.50,
        max_value=2.00,
        value=float(default_demand),
        step=0.05,
        format="%.2fx",
        help="Simula eventos de alta demanda tipo CyberDay o estacionalidad.",
    )

    cap_mult = st.slider(
        "Factor de Capacidad en Bodegas:",
        min_value=0.50,
        max_value=1.50,
        value=float(default_cap),
        step=0.05,
        format="%.2fx",
        help="Simula restricciones o ampliaciones de espacio de almacenamiento.",
    )

    all_supply_nodes = list(CD_NODES.keys()) + list(DARK_STORE_NODES.keys())
    disabled_nodes = st.multiselect(
        "Nodos Logísticos Inhabilitados (Contingencias):",
        options=all_supply_nodes,
        default=default_disabled,
        help="Simula cierres repentinos de Dark Stores o huelgas en CDs.",
    )

    is_baseline = (
        selected_quantile == "p50"
        and transp_mult == 1.0
        and demand_mult == 1.0
        and cap_mult == 1.0
        and not disabled_nodes
    )

    st.markdown("---")
    if is_baseline:
        st.success("✅ Ejecutando sobre Línea Base precalculada.")
    else:
        st.info("⚡ Simulación What-If dinámica activada.")

# ------------------------------------------------------------------------------
# 4. RESOLUCIÓN / CARGA DE ESCENARIO
# ------------------------------------------------------------------------------
if is_baseline:
    # Usar datos base precalculados
    optimal_kpis = baseline["optimal_kpis"]
    benchmark_kpis = baseline["benchmark_kpis"]
    fulfillment_df = baseline["optimal_fulfillment"]
    benchmark_fulfillment_df = baseline["benchmark_fulfillment"]
    inventory_df = baseline["inventory_df"]
    replenishment_df = baseline["replenishment_df"]
    solve_status = "Optimal (Precalculado)"
    solve_time = optimal_kpis.get("solve_time_seconds", 0.15)
else:
    # Resolver en caliente con caché
    with st.spinner("Resolviendo modelo MILP y simulador benchmark..."):
        sim_res = run_scenario_optimization(
            demand_quantile=selected_quantile,
            transport_cost_multiplier=transp_mult,
            demand_multiplier=demand_mult,
            disabled_nodes_tuple=tuple(sorted(disabled_nodes)),
            capacity_multiplier=cap_mult,
        )
        sol_milp = sim_res["sol_milp"]
        optimal_kpis = sim_res["optimal_kpis"]
        benchmark_kpis = sim_res["benchmark_kpis"]
        fulfillment_df = sim_res["fulfillment_df"]
        benchmark_fulfillment_df = sim_res["benchmark_fulfillment_df"]
        inventory_df = sim_res["inventory_df"]
        replenishment_df = sim_res["replenishment_df"]
        solve_status = sol_milp.status
        solve_time = sol_milp.solve_time_seconds

# ------------------------------------------------------------------------------
# 5. ENCABEZADO Y TARJETAS KPI DE NEGOCIO
# ------------------------------------------------------------------------------
st.markdown('<div class="badge-pill">🚀 Supply Chain & Fulfillment Network Optimizer | MILP Prescriptive</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header">📦 LogisticNetwork: Optimizador & Simulador de Red</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Red Metropolitana de Santiago de Chile (2 CDs • 8 Dark Stores • 20 Zonas de Clientes • 14 Días)</div>',
    unsafe_allow_html=True,
)

# Métricas principales
opt_cost = optimal_kpis.get("total_cost", 0.0)
bench_cost = benchmark_kpis.get("total_cost", 0.0)
savings_usd = bench_cost - opt_cost if bench_cost > opt_cost else 0.0
savings_pct = (savings_usd / bench_cost * 100.0) if bench_cost > 0 else 0.0

col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)

with col_kpi1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Costo Total MILP</div>
            <div class="kpi-value">${opt_cost:,.0f} <span style="font-size: 1rem; color:#A0AEC0;">USD</span></div>
            <div class="kpi-sub positive-delta">⚡ Solver: {solve_time:.2f}s ({solve_status})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_kpi2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Ahorro vs Heurística</div>
            <div class="kpi-value positive-delta">-{savings_pct:.1f}%</div>
            <div class="kpi-sub positive-delta">💵 Ahorro: ${savings_usd:,.0f} USD</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_kpi3:
    otif = optimal_kpis.get("otif_pct", 100.0)
    otif_color = "positive-delta" if otif >= 98.0 else "negative-delta"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Nivel de Servicio (OTIF)</div>
            <div class="kpi-value {otif_color}">{otif:.1f}%</div>
            <div class="kpi-sub" style="color: #A0AEC0;">Quiebres: {optimal_kpis.get('stockout_units', 0.0):.0f} unid.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_kpi4:
    cross_rate = optimal_kpis.get("cross_fulfillment_rate_pct", 0.0)
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Cross-Fulfillment</div>
            <div class="kpi-value" style="color: #FFB300;">{cross_rate:.1f}%</div>
            <div class="kpi-sub" style="color: #A0AEC0;">Despachos desde CDs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_kpi5:
    ds_util = optimal_kpis.get("avg_ds_utilization_pct", 0.0)
    util_color = "negative-delta" if ds_util > 80.0 else "positive-delta"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Utilización Media DS</div>
            <div class="kpi-value {util_color}">{ds_util:.1f}%</div>
            <div class="kpi-sub" style="color: #A0AEC0;">Capacidad volumétrica</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br/>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 6. PESTAÑAS PRINCIPALES DE VISUALIZACIÓN
# ------------------------------------------------------------------------------
tab_map, tab_comparison, tab_inventory, tab_forecast = st.tabs([
    "🗺️ Vista Geográfica de Flujos (PyDeck)",
    "📊 Comparador MILP vs Heurística Naive",
    "🏭 Inventario y Utilización de Nodos",
    "📈 Pronóstico de Demanda e Incertidumbre",
])

# ==============================================================================
# PESTAÑA 1: VISTA GEOGRÁFICA Y ARCOS PYDECK
# ==============================================================================
with tab_map:
    st.markdown("### 🌐 Red de Distribución Urbana y Arcos de Despacho")
    st.markdown(
        "Visualización en 3D de los flujos óptimos de última milla y cross-fulfillment "
        "desde centros de suministro hacia clusters de clientes en el Gran Santiago."
    )

    # Leyenda visual
    st.markdown(
        """
        <div class="legend-box">
            <div class="legend-item"><span class="dot" style="background:#2196F3;"></span> <b>Centros de Distribución (CDs)</b></div>
            <div class="legend-item"><span class="dot" style="background:#00C853;"></span> <b>Dark Stores Activas</b></div>
            <div class="legend-item"><span class="dot" style="background:#F44336;"></span> <b>Dark Stores Inhabilitadas</b></div>
            <div class="legend-item"><span class="dot" style="background:#FFB300;"></span> <b>Zonas de Demanda (Clientes)</b></div>
            <div class="legend-item"><span class="dot" style="background:#00E676;"></span> <b>Flujo Local (Dark Store ➔ Zona)</b></div>
            <div class="legend-item"><span class="dot" style="background:#FF6D00;"></span> <b>Cross-Fulfillment (CD ➔ Zona)</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Filtros sobre el mapa
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        date_options = ["Todos los Días"]
        if not fulfillment_df.empty and "date" in fulfillment_df.columns:
            date_options.extend(sorted(fulfillment_df["date"].astype(str).unique().tolist()))
        selected_date = st.selectbox("Filtrar por fecha:", date_options, index=0)
        filter_date = None if selected_date == "Todos los Días" else selected_date

    with col_f2:
        item_options = ["TODOS"] + list(SKU_CATALOG.keys())
        selected_item = st.selectbox(
            "Filtrar por Producto (SKU):",
            item_options,
            format_func=lambda x: f"{x} - {SKU_CATALOG[x]['name']}" if x in SKU_CATALOG else "Todos los SKUs",
            index=0,
        )

    # Preparar datos enriquecidos para PyDeck
    flows_df = prepare_map_layers_data(
        fulfillment_df=fulfillment_df,
        coords_dict=coords_dict,
        date_filter=filter_date,
        item_filter=selected_item,
    )

    # Renderizar mapa PyDeck
    deck_map = create_network_map(
        flows_df=flows_df,
        topology_df=topology_df,
        zones_df=zones_df,
        disabled_nodes=disabled_nodes,
    )
    st.pydeck_chart(deck_map, use_container_width=True)

    # Resumen de arcos renderizados
    if not flows_df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rutas Activas", f"{len(flows_df):,}")
        c2.metric("Unidades en Tránsito", f"{flows_df['units'].sum():,.0f}")
        c3.metric("Volumen Despachado", f"{flows_df['volume_m3'].sum():,.2f} m³")
        c4.metric("Distancia Promedio", f"{flows_df['distance_km'].mean():.1f} km")
    else:
        st.warning("No hay flujos de despacho para el filtro seleccionado.")

# ==============================================================================
# PESTAÑA 2: COMPARADOR MILP VS BENCHMARK NAIVE
# ==============================================================================
with tab_comparison:
    st.markdown("### 🏆 Evaluación Económica: Prescripción MILP vs. Regla Miope")
    st.markdown(
        "Contraste formal del costo total y estructura de costos entre el modelo de optimización coordinado "
        "y la regla heurística naive descentralizada (**Validación Hipótesis H2**)."
    )

    # Gráfico de barras comparativo
    cost_fig = create_cost_comparison_chart(optimal_kpis, benchmark_kpis)
    st.plotly_chart(cost_fig, use_container_width=True)

    # Tabla comparativa de métricas
    st.markdown("#### 📋 Matriz Detallada de Rendimiento Operativo y Financiero")
    comp_data = [
        {
            "Métrica": "Costo Logístico Total",
            "MILP Prescriptivo": f"${optimal_kpis.get('total_cost', 0):,.2f} USD",
            "Heurística Naive": f"${benchmark_kpis.get('total_cost', 0):,.2f} USD",
            "Diferencia / Ahorro": f"-{savings_pct:.2f}%",
        },
        {
            "Métrica": "Transporte Fulfillment",
            "MILP Prescriptivo": f"${optimal_kpis.get('fulfillment_transport_cost', 0):,.2f} USD",
            "Heurística Naive": f"${benchmark_kpis.get('fulfillment_transport_cost', 0):,.2f} USD",
            "Diferencia / Ahorro": f"{((optimal_kpis.get('fulfillment_transport_cost', 0) - benchmark_kpis.get('fulfillment_transport_cost', 0)) / max(benchmark_kpis.get('fulfillment_transport_cost', 1), 1) * 100):+.2f}%",
        },
        {
            "Métrica": "Transporte Reabastecimiento",
            "MILP Prescriptivo": f"${optimal_kpis.get('replenishment_transport_cost', 0):,.2f} USD",
            "Heurística Naive": f"${benchmark_kpis.get('replenishment_transport_cost', 0):,.2f} USD",
            "Diferencia / Ahorro": f"{((optimal_kpis.get('replenishment_transport_cost', 0) - benchmark_kpis.get('replenishment_transport_cost', 0)) / max(benchmark_kpis.get('replenishment_transport_cost', 1), 1) * 100):+.2f}%",
        },
        {
            "Métrica": "Almacenamiento (Holding Cost)",
            "MILP Prescriptivo": f"${optimal_kpis.get('holding_cost', 0):,.2f} USD",
            "Heurística Naive": f"${benchmark_kpis.get('holding_cost', 0):,.2f} USD",
            "Diferencia / Ahorro": f"{((optimal_kpis.get('holding_cost', 0) - benchmark_kpis.get('holding_cost', 0)) / max(benchmark_kpis.get('holding_cost', 1), 1) * 100):+.2f}%",
        },
        {
            "Métrica": "Penalización por Quiebre (Stockout)",
            "MILP Prescriptivo": f"${optimal_kpis.get('stockout_cost', 0):,.2f} USD",
            "Heurística Naive": f"${benchmark_kpis.get('stockout_cost', 0):,.2f} USD",
            "Diferencia / Ahorro": "$0.00 USD",
        },
        {
            "Métrica": "Nivel de Servicio (OTIF)",
            "MILP Prescriptivo": f"{optimal_kpis.get('otif_pct', 100):.2f}%",
            "Heurística Naive": f"{benchmark_kpis.get('otif_pct', 100):.2f}%",
            "Diferencia / Ahorro": f"{optimal_kpis.get('otif_pct', 100) - benchmark_kpis.get('otif_pct', 100):+.2f} pp",
        },
        {
            "Métrica": "Tasa de Cross-Fulfillment",
            "MILP Prescriptivo": f"{optimal_kpis.get('cross_fulfillment_rate_pct', 0):.2f}%",
            "Heurística Naive": f"{benchmark_kpis.get('cross_fulfillment_rate_pct', 0):.2f}%",
            "Diferencia / Ahorro": f"{optimal_kpis.get('cross_fulfillment_rate_pct', 0) - benchmark_kpis.get('cross_fulfillment_rate_pct', 0):+.2f} pp",
        },
    ]
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

# ==============================================================================
# PESTAÑA 3: INVENTARIO Y UTILIZACIÓN DE NODOS
# ==============================================================================
with tab_inventory:
    st.markdown("### 🏢 Dinámica de Inventario y Saturación de Capacidad")
    st.markdown(
        "Monitoreo de ocupación volumétrica ($m^3$) y tasa de utilización en las Dark Stores urbanas "
        "a lo largo del horizonte de 14 días para prevenir cuellos de botella físicos."
    )

    inv_fig = create_inventory_utilization_chart(inventory_df)
    st.plotly_chart(inv_fig, use_container_width=True)

    col_inv1, col_inv2 = st.columns(2)
    with col_inv1:
        st.markdown("#### 📦 Resumen de Reabastecimiento (CDs ➔ Dark Stores)")
        if not replenishment_df.empty:
            replenish_summary = replenishment_df.groupby(["from_cd_id", "to_ds_id"]).agg({
                "units": "sum",
                "volume_m3": "sum",
                "transport_cost": "sum",
            }).reset_index()
            st.dataframe(
                replenish_summary.style.format({
                    "units": "{:,.0f}",
                    "volume_m3": "{:,.2f} m³",
                    "transport_cost": "${:,.2f} USD",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No se requirió reabastecimiento en este escenario.")

    with col_inv2:
        st.markdown("#### 📊 Capacidad Física y Costo de Almacenamiento")
        st.dataframe(
            topology_df[["node_id", "node_name", "node_type", "capacity_m3", "holding_cost_unit"]].style.format({
                "capacity_m3": "{:,.0f} m³",
                "holding_cost_unit": "${:,.2f} USD/m³·d",
            }),
            use_container_width=True,
            hide_index=True,
        )

# ==============================================================================
# PESTAÑA 4: PRONÓSTICO DE DEMANDA E INCERTIDUMBRE
# ==============================================================================
with tab_forecast:
    st.markdown("### 🔮 Demanda Predictiva por Cuantiles (LightGBM)")
    st.markdown(
        "Exploración de la distribución de demanda futura ($P_{10}, P_{50}, P_{90}$) por producto y zona, "
        "alimentando el stock de seguridad y las decisiones de abastecimiento del optimizador."
    )

    col_fc1, col_fc2 = st.columns(2)
    with col_fc1:
        sku_choice = st.selectbox(
            "Seleccionar Producto (SKU):",
            list(SKU_CATALOG.keys()),
            format_func=lambda x: f"{x} - {SKU_CATALOG[x]['name']}",
        )
    with col_fc2:
        zone_choice = st.selectbox(
            "Seleccionar Zona de Demanda:",
            list(DEMAND_ZONES.keys()),
            format_func=lambda x: f"{x} ({DEMAND_ZONES[x]['name']})",
        )

    forecast_fig = create_demand_forecast_chart(forecast_df, sku_choice, zone_choice)
    st.plotly_chart(forecast_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📥 Descarga de Planes Operativos Generados")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if not fulfillment_df.empty:
            csv_fulfillment = fulfillment_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 Descargar Plan de Fulfillment (CSV)",
                data=csv_fulfillment,
                file_name="plan_fulfillment_optimo.csv",
                mime="text/csv",
            )
    with col_d2:
        if not replenishment_df.empty:
            csv_replenish = replenishment_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 Descargar Plan de Reabastecimiento (CSV)",
                data=csv_replenish,
                file_name="plan_reabastecimiento_optimo.csv",
                mime="text/csv",
            )

# ------------------------------------------------------------------------------
# 7. PIE DE PÁGINA
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #718096; font-size: 0.85rem;">
        <b>LogisticNetwork</b> • Proyecto End-to-End de Analítica Predictiva y Prescriptiva (Machine Learning + MILP)<br/>
        Desarrollado para optimización logística de última milla y redes omnicanal en Santiago de Chile.
    </div>
    """,
    unsafe_allow_html=True,
)
