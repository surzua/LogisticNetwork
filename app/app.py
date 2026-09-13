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
        margin-bottom: 1.2rem;
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
        font-size: 0.82rem;
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
        font-size: 0.80rem;
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
    /* Tarjetas explicativas y pedagógicas */
    .edu-card {
        background: linear-gradient(145deg, #1A202C, #202736);
        border: 1px solid #323F52;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    .edu-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #60A5FA;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .edu-text {
        font-size: 0.88rem;
        color: #D1D5DB;
    }
    .preset-narrative {
        background: rgba(30, 41, 59, 0.7);
        border-left: 3px solid #38BDF8;
        padding: 0.6rem 0.8rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #CBD5E1;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }
    .insight-box {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-top: 1rem;
        color: #E2E8F0;
        font-size: 0.88rem;
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
    st.markdown("## ⚙️ Simulador de Red (What-If)")
    st.markdown(
        "Modifica las condiciones del mercado y de la infraestructura para ver "
        "en tiempo real cómo reacciona la red inteligente."
    )

    # Presets rápidos
    st.markdown("### 🎯 Escenarios Predefinidos")
    preset = st.selectbox(
        "Elige una situación de negocio:",
        [
            "Línea Base (Baseline)",
            "Shock Combustible (+20%)",
            "Falla Dark Store (Providencia cerrada)",
            "Pico de Demanda (+30%)",
            "Estrés Severo Combinado",
            "Personalizado",
        ],
        index=0,
    )

    # Historias explicativas para público general
    preset_descriptions = {
        "Línea Base (Baseline)": "📌 **Operación Normal:** Red 100% operativa con sus 2 Centros Regionales y 8 Dark Stores de barrio abasteciendo a las 20 comunas.",
        "Shock Combustible (+20%)": "⛽ **Alza de Bencina/Diésel:** Simula un encarecimiento del transporte. La optimización busca rutas más cortas y camiones más llenos.",
        "Falla Dark Store (Providencia cerrada)": "⚠️ **Cierre de Emergencia:** La bodega de Providencia queda inhabilitada (corte de energía/huelga). El sistema reasigna los pedidos a comunas vecinas.",
        "Pico de Demanda (+30%)": "🛍️ **Evento Especial (CyberDay):** Incremento súbito de pedidos en toda la ciudad, poniendo a prueba el espacio físico en las bodegas.",
        "Estrés Severo Combinado": "🔥 **Tormenta Perfecta:** Combustible un 35% más caro, demanda +25% y dos bodegas cerradas (Providencia y Las Condes).",
        "Personalizado": "🛠️ **Modo Libre:** Ajusta manualmente los parámetros inferiores a tu gusto.",
    }
    st.markdown(f'<div class="preset-narrative">{preset_descriptions.get(preset, "")}</div>', unsafe_allow_html=True)

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
    st.markdown("### 🎛️ Palancas Operativas")

    selected_quantile = st.radio(
        "Estrategia de Demanda Futura:",
        options=["p10", "p50", "p90", "actual"],
        format_func=lambda x: {
            "p10": "📉 P10: Demanda Baja (Ahorro/Conservador)",
            "p50": "⚖️ P50: Demanda Normal (Escenario Medio)",
            "p90": "📈 P90: Demanda Alta (Stock de Seguridad)",
            "actual": "👁️ Demanda Real (Pronóstico Perfecto)",
        }.get(x, x),
        index=["p10", "p50", "p90", "actual"].index(default_quantile),
        help="Define qué nivel de incertidumbre asume el planificador: prepararse para pocas ventas (P10), promedio (P50) o protegerse ante ventas muy altas (P90).",
    )

    transp_mult = st.slider(
        "Costo de Transporte (Combustible / Peajes):",
        min_value=0.50,
        max_value=2.50,
        value=float(default_transp),
        step=0.05,
        format="%.2fx",
        help="1.0x es costo normal. 1.5x simula un aumento del 50% en fletes, congestión urbana o combustible.",
    )

    demand_mult = st.slider(
        "Volumen de Pedidos de Clientes:",
        min_value=0.50,
        max_value=2.00,
        value=float(default_demand),
        step=0.05,
        format="%.2fx",
        help="1.0x es demanda habitual. Valores mayores a 1.2x simulan días de alta venta como CyberMonday.",
    )

    cap_mult = st.slider(
        "Capacidad de Espacio en Bodegas:",
        min_value=0.50,
        max_value=1.50,
        value=float(default_cap),
        step=0.05,
        format="%.2fx",
        help="Espacio disponible para almacenar productos en las Dark Stores y Centros de Distribución.",
    )

    all_supply_nodes = list(CD_NODES.keys()) + list(DARK_STORE_NODES.keys())
    
    # Nombres amigables para selección de contingencias
    node_name_lookup = {**{k: v["name"] for k, v in CD_NODES.items()}, **{k: v["name"] for k, v in DARK_STORE_NODES.items()}}

    disabled_nodes = st.multiselect(
        "Inhabilitar Bodegas (Contingencias):",
        options=all_supply_nodes,
        default=default_disabled,
        format_func=lambda x: node_name_lookup.get(x, x),
        help="Selecciona instalaciones que no puedan operar (por ejemplo por mantención o imprevistos).",
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
        st.success("✅ Modo Base: Resultados precalculados de máxima velocidad.")
    else:
        st.info("⚡ Simulación What-If activa: Resolviendo en tiempo real.")

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
    '<div class="sub-header">Red Metropolitana de Santiago de Chile • 2 Centros Regionales (CDs) • 8 Bodegas Urbanas (Dark Stores) • 20 Comunas de Clientes • Horizonte de 14 Días</div>',
    unsafe_allow_html=True,
)

# Panel introductorio pedagógico para público general
with st.expander("💡 ¿Primera vez en la plataforma? Haz clic aquí para entender cómo funciona la red en 2 minutos", expanded=False):
    col_intro1, col_intro2, col_intro3 = st.columns(3)
    with col_intro1:
        st.markdown(
            """
            <div class="edu-card">
                <div class="edu-title">🏭 1. ¿Dónde están los productos?</div>
                <div class="edu-text">
                    <b>• Centros de Distribución (CDs):</b> Grandes bodegas en la periferia (Quilicura y San Bernardo). Tienen espacio casi ilimitado y almacenan los productos que llegan de los proveedores.<br/><br/>
                    <b>• Dark Stores:</b> Pequeñas bodegas express dentro de los barrios (Providencia, Las Condes, Santiago Centro...). Permiten entregar rápido al cliente final, pero su espacio físico es muy acotado.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_intro2:
        st.markdown(
            """
            <div class="edu-card">
                <div class="edu-title">🚚 2. ¿Cómo viaja cada compra?</div>
                <div class="edu-text">
                    <b>• Despacho Local Ágil (🟢 Verde):</b> El escenario ideal. Tu compra sale directamente desde la Dark Store de tu comuna. Es rápido, barato y contamina menos.<br/><br/>
                    <b>• Cross-Fulfillment / Respaldo (🟠 Naranja):</b> Si la Dark Store de tu comuna se quedó sin stock, el producto debe viajar de emergencia desde el CD periférico. Cuesta más flete, pero evita que el cliente pierda su compra.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_intro3:
        st.markdown(
            """
            <div class="edu-card">
                <div class="edu-title">🧠 3. ¿Por qué usamos Optimización IA?</div>
                <div class="edu-text">
                    <b>• Regla Tradicional Simple:</b> Despacha siempre desde la bodega más cercana sin mirar el futuro. Con frecuencia colapsa el espacio o agota los productos.<br/><br/>
                    <b>• Optimización Prescriptiva (MILP):</b> Anticipa la demanda de los próximos 14 días y decide exactamente cuándo reabastecer cada noche y desde dónde despachar para minimizar costos y maximizar entregas a tiempo.
                </div>
            </div>
            """,
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
            <div class="kpi-title" title="Gasto total estimado en los 14 días considerando fletes, bodegaje y penalizaciones por falta de stock.">💰 Costo Total Red</div>
            <div class="kpi-value">${opt_cost:,.0f} <span style="font-size: 1rem; color:#A0AEC0;">USD</span></div>
            <div class="kpi-sub positive-delta">⚡ Optimizado en {solve_time:.2f}s ({solve_status})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_kpi2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title" title="Ahorro financiero directo frente a la regla tradicional de enviar siempre desde la bodega más cercana.">🏆 Ahorro vs Regla Simple</div>
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
            <div class="kpi-title" title="On-Time In-Full: Porcentaje de compras entregadas a tiempo y completas. Meta internacional: >98%.">📦 Pedidos a Tiempo (OTIF)</div>
            <div class="kpi-value {otif_color}">{otif:.1f}%</div>
            <div class="kpi-sub" style="color: #A0AEC0;">Quiebres: {optimal_kpis.get('stockout_units', 0.0):.0f} unidades</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_kpi4:
    cross_rate = optimal_kpis.get("cross_fulfillment_rate_pct", 0.0)
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title" title="Porcentaje de pedidos que tuvieron que ser rescatados desde los Centros periféricos por falta de stock local.">🚚 Envíos de Respaldo</div>
            <div class="kpi-value" style="color: #FFB300;">{cross_rate:.1f}%</div>
            <div class="kpi-sub" style="color: #A0AEC0;">Cross-Fulfillment desde CDs</div>
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
            <div class="kpi-title" title="Porcentaje del espacio físico ocupado en las bodegas urbanas express. Sobre el 80% existe riesgo de congestión.">🏢 Ocupación Bodegas</div>
            <div class="kpi-value {util_color}">{ds_util:.1f}%</div>
            <div class="kpi-sub" style="color: #A0AEC0;">Capacidad volumétrica usada</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br/>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 6. PESTAÑAS PRINCIPALES DE VISUALIZACIÓN
# ------------------------------------------------------------------------------
tab_map, tab_comparison, tab_inventory, tab_forecast, tab_learning = st.tabs([
    "🗺️ Vista Geográfica de Rutas (PyDeck)",
    "📊 Comparador de Ahorro y Eficiencia",
    "🏭 Inventario y Espacio en Bodegas",
    "📈 Pronóstico de Demanda de Clientes",
    "📚 Centro de Aprendizaje & Casos de Negocio",
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

    st.markdown(
        """
        <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #00E676; padding: 10px 14px; border-radius: 0 8px 8px 0; margin-bottom: 1rem; font-size: 0.88rem; color: #E2E8F0;">
            💡 <b>Cómo explorar el mapa:</b> Pasa el cursor por encima de cualquier bodega o ruta para ver su información en tiempo real sin códigos difíciles.<br/>
            • <b>Círculos azules y verdes:</b> Centros de Distribución y bodegas express urbanas.<br/>
            • <b>Puntos ámbar:</b> Comunas de clientes que reciben pedidos.<br/>
            • <b>Líneas verdes:</b> Entregas directas desde la bodega de tu barrio (rápidas y baratas).<br/>
            • <b>Líneas naranjas:</b> Envíos de respaldo desde la periferia (salvan pedidos cuando falta stock local).
        </div>
        """,
        unsafe_allow_html=True,
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
    st.markdown("### 🏆 Evaluación Económica: Prescripción Inteligente vs. Regla Simple")
    st.markdown(
        "Contraste del costo total y estructura de gastos entre el modelo de optimización coordinado "
        "y la regla simple de enviar siempre desde la bodega más cercana."
    )

    st.markdown(
        f"""
        <div class="insight-box">
            <b>💡 ¿Cómo logra el optimizador este ahorro del {savings_pct:.1f}%?</b><br/>
            Mientras una regla intuitiva despacha a ciegas desde la bodega más cercana (provocando quiebres de stock o saturación de espacio en días posteriores),
            el modelo de optimización <b>coordina los envíos con 14 días de anticipación</b>. Reabastece las Dark Stores en horarios nocturnos eficientes y decide exactamente
            cuándo conviene absorber un flete desde el Centro Regional para no agotar la bodega del barrio.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br/>", unsafe_allow_html=True)

    # Gráfico de barras comparativo
    cost_fig = create_cost_comparison_chart(optimal_kpis, benchmark_kpis)
    st.plotly_chart(cost_fig, use_container_width=True)

    # Tabla comparativa de métricas
    st.markdown("#### 📋 Desglose Financiero y Operativo Detallado")
    comp_data = [
        {
            "Métrica de Gestión": "Costo Logístico Total",
            "Optimización Inteligente (MILP)": f"${optimal_kpis.get('total_cost', 0):,.2f} USD",
            "Regla Simple (Heurística)": f"${benchmark_kpis.get('total_cost', 0):,.2f} USD",
            "Ahorro Conseguido": f"-{savings_pct:.2f}%",
        },
        {
            "Métrica de Gestión": "Flete de Entrega a Clientes",
            "Optimización Inteligente (MILP)": f"${optimal_kpis.get('fulfillment_transport_cost', 0):,.2f} USD",
            "Regla Simple (Heurística)": f"${benchmark_kpis.get('fulfillment_transport_cost', 0):,.2f} USD",
            "Ahorro Conseguido": f"{((optimal_kpis.get('fulfillment_transport_cost', 0) - benchmark_kpis.get('fulfillment_transport_cost', 0)) / max(benchmark_kpis.get('fulfillment_transport_cost', 1), 1) * 100):+.2f}%",
        },
        {
            "Métrica de Gestión": "Flete de Reabastecimiento",
            "Optimización Inteligente (MILP)": f"${optimal_kpis.get('replenishment_transport_cost', 0):,.2f} USD",
            "Regla Simple (Heurística)": f"${benchmark_kpis.get('replenishment_transport_cost', 0):,.2f} USD",
            "Ahorro Conseguido": f"{((optimal_kpis.get('replenishment_transport_cost', 0) - benchmark_kpis.get('replenishment_transport_cost', 0)) / max(benchmark_kpis.get('replenishment_transport_cost', 1), 1) * 100):+.2f}%",
        },
        {
            "Métrica de Gestión": "Almacenamiento (Holding)",
            "Optimización Inteligente (MILP)": f"${optimal_kpis.get('holding_cost', 0):,.2f} USD",
            "Regla Simple (Heurística)": f"${benchmark_kpis.get('holding_cost', 0):,.2f} USD",
            "Ahorro Conseguido": f"{((optimal_kpis.get('holding_cost', 0) - benchmark_kpis.get('holding_cost', 0)) / max(benchmark_kpis.get('holding_cost', 1), 1) * 100):+.2f}%",
        },
        {
            "Métrica de Gestión": "Pérdida por Falta de Stock",
            "Optimización Inteligente (MILP)": f"${optimal_kpis.get('stockout_cost', 0):,.2f} USD",
            "Regla Simple (Heurística)": f"${benchmark_kpis.get('stockout_cost', 0):,.2f} USD",
            "Ahorro Conseguido": "$0.00 USD",
        },
        {
            "Métrica de Gestión": "Cumplimiento a Tiempo (OTIF)",
            "Optimización Inteligente (MILP)": f"{optimal_kpis.get('otif_pct', 100):.2f}%",
            "Regla Simple (Heurística)": f"{benchmark_kpis.get('otif_pct', 100):.2f}%",
            "Ahorro Conseguido": f"{optimal_kpis.get('otif_pct', 100) - benchmark_kpis.get('otif_pct', 100):+.2f} pp",
        },
        {
            "Métrica de Gestión": "Tasa de Envíos de Respaldo",
            "Optimización Inteligente (MILP)": f"{optimal_kpis.get('cross_fulfillment_rate_pct', 0):.2f}%",
            "Regla Simple (Heurística)": f"{benchmark_kpis.get('cross_fulfillment_rate_pct', 0):.2f}%",
            "Ahorro Conseguido": f"{optimal_kpis.get('cross_fulfillment_rate_pct', 0) - benchmark_kpis.get('cross_fulfillment_rate_pct', 0):+.2f} pp",
        },
    ]
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

# ==============================================================================
# PESTAÑA 3: INVENTARIO Y UTILIZACIÓN DE NODOS
# ==============================================================================
with tab_inventory:
    st.markdown("### 🏢 Dinámica de Inventario y Espacio en Bodegas")
    st.markdown(
        "Monitoreo de ocupación física ($m^3$) y porcentaje de saturación en las Dark Stores urbanas "
        "a lo largo de los 14 días para prevenir colapsos de espacio."
    )

    st.markdown(
        """
        <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #F59E0B; padding: 10px 14px; border-radius: 0 8px 8px 0; margin-bottom: 1rem; font-size: 0.88rem; color: #E2E8F0;">
            💡 <b>El desafío del espacio urbano:</b> A diferencia de un gran centro en la periferia, una Dark Store en Providencia o Las Condes tiene pocos metros cúbicos.
            Si la bodega supera el <b>80% de ocupación</b> (zona de advertencia), corre el riesgo de no poder recibir camiones de reabastecimiento o generar sobrecostos por congestión.
        </div>
        """,
        unsafe_allow_html=True,
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

            # Nombres amigables para el reabastecimiento
            name_map_rep = {**{k: v["name"] for k, v in CD_NODES.items()}, **{k: v["name"] for k, v in DARK_STORE_NODES.items()}}
            replenish_summary["Origen"] = replenish_summary["from_cd_id"].map(lambda x: name_map_rep.get(x, x))
            replenish_summary["Destino"] = replenish_summary["to_ds_id"].map(lambda x: name_map_rep.get(x, x))

            disp_rep = replenish_summary[["Origen", "Destino", "units", "volume_m3", "transport_cost"]].rename(columns={
                "units": "Unidades Enviadas",
                "volume_m3": "Volumen (m³)",
                "transport_cost": "Costo Flete",
            })

            st.dataframe(
                disp_rep.style.format({
                    "Unidades Enviadas": "{:,.0f}",
                    "Volumen (m³)": "{:,.2f} m³",
                    "Costo Flete": "${:,.2f} USD",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No se requirió reabastecimiento en este escenario.")

    with col_inv2:
        st.markdown("#### 📊 Capacidad Física y Costo Diario de Almacenamiento")
        disp_top = topology_df[["node_name", "node_type", "capacity_m3", "holding_cost_unit"]].rename(columns={
            "node_name": "Instalación",
            "node_type": "Tipo",
            "capacity_m3": "Capacidad Máx.",
            "holding_cost_unit": "Costo Bodegaje",
        })
        st.dataframe(
            disp_top.style.format({
                "Capacidad Máx.": "{:,.0f} m³",
                "Costo Bodegaje": "${:,.2f} USD/m³·d",
            }),
            use_container_width=True,
            hide_index=True,
        )

# ==============================================================================
# PESTAÑA 4: PRONÓSTICO DE DEMANDA E INCERTIDUMBRE
# ==============================================================================
with tab_forecast:
    st.markdown("### 🔮 Pronóstico Inteligente de Demanda Futura (Machine Learning)")
    st.markdown(
        "Visualización de las predicciones probabilísticas de ventas ($P_{10}, P_{50}, P_{90}$) "
        "que alimentan el stock de seguridad del modelo de optimización."
    )

    st.markdown(
        """
        <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #60A5FA; padding: 10px 14px; border-radius: 0 8px 8px 0; margin-bottom: 1rem; font-size: 0.88rem; color: #E2E8F0;">
            💡 <b>¿Cómo entender este pronóstico probabilístico?</b><br/>
            El algoritmo no predice un solo valor ciego, sino un abanico de posibilidades:<br/>
            • <b>Línea P50 (Escenario Medio):</b> Lo que estadísticamente se espera vender en un día promedio.<br/>
            • <b>Franja Superior P90:</b> Escenario de ventas altas (días de promociones o festivos). Permite protegerse ante quiebres.<br/>
            • <b>Franja Inferior P10:</b> Escenario de ventas bajas. Ayuda a no sobrecargar bodegas con productos de baja rotación.
        </div>
        """,
        unsafe_allow_html=True,
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
            "Seleccionar Comuna / Zona de Clientes:",
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
                label="📄 Descargar Plan de Despachos a Clientes (CSV)",
                data=csv_fulfillment,
                file_name="plan_despachos_optimo.csv",
                mime="text/csv",
            )
    with col_d2:
        if not replenishment_df.empty:
            csv_replenish = replenishment_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 Descargar Plan de Reabastecimiento a Bodegas (CSV)",
                data=csv_replenish,
                file_name="plan_reabastecimiento_optimo.csv",
                mime="text/csv",
            )

# ==============================================================================
# PESTAÑA 5: CENTRO DE APRENDIZAJE & CASOS DE NEGOCIO (NUEVA)
# ==============================================================================
with tab_learning:
    st.markdown("### 📚 Centro de Aprendizaje & Casos de Negocio")
    st.markdown(
        "Espacio didáctico diseñado para que cualquier persona, sin importar su formación técnica, "
        "pueda comprender cómo opera la distribución urbana moderna y qué valor aporta la optimización matemática."
    )

    st.markdown("#### 🎮 3 Experimentos Guiados para Probar en el Simulador")
    st.markdown("Te invitamos a interactuar con la barra lateral izquierda para explorar estas 3 situaciones cotidianas:")

    exp1, exp2, exp3 = st.columns(3)
    with exp1:
        st.markdown(
            """
            <div class="edu-card">
                <div class="edu-title">🧪 1. El Alza de Combustibles</div>
                <div class="edu-text">
                    <b>Situación:</b> La bencina y el diésel suben repentinamente (+20% a +40%).<br/><br/>
                    <b>Qué hacer:</b> En el menú lateral, selecciona el preset <i>'Shock Combustible (+20%)'</i> o sube el slider a 1.3x.<br/><br/>
                    <b>Qué observar:</b> Ve a la <b>Pestaña 2</b>. Verás cómo el costo de flete sube en ambos modelos, pero la Inteligencia Artificial reduce el impacto al consolidar viajes nocturnos más eficientes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with exp2:
        st.markdown(
            """
            <div class="edu-card">
                <div class="edu-title">🧪 2. Contingencia en Bodega Urbana</div>
                <div class="edu-text">
                    <b>Situación:</b> La bodega express de Providencia sufre un corte eléctrico o huelga y debe cerrar.<br/><br/>
                    <b>Qué hacer:</b> Elige el preset <i>'Falla Dark Store (Providencia cerrada)'</i> en la barra lateral.<br/><br/>
                    <b>Qué observar:</b> Ve a la <b>Pestaña 1 (Mapa)</b>: Providencia aparecerá en color rojo. Verás cómo automáticamente se encienden líneas naranjas desde el Centro de Distribución para despachar a los clientes sin perder ventas.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with exp3:
        st.markdown(
            """
            <div class="edu-card">
                <div class="edu-title">🧪 3. Evento Masivo (CyberDay)</div>
                <div class="edu-text">
                    <b>Situación:</b> Las compras online se disparan un 30% en toda la capital.<br/><br/>
                    <b>Qué hacer:</b> Elige el preset <i>'Pico de Demanda (+30%)'</i> o sube el multiplicador de demanda a 1.30x.<br/><br/>
                    <b>Qué observar:</b> Ve a la <b>Pestaña 3 (Inventario)</b> y observa cómo la ocupación de las Dark Stores se acerca al 90%, demostrando la necesidad de reabastecimientos continuos desde los CDs periféricos.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 📖 Glosario Rápido de Términos Logísticos")
    st.markdown("Los conceptos clave de la industria explicados en palabras cotidianas:")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(
            """
            - 🏭 **Centro de Distribución (CD):** Grandes naves industriales en la periferia (Quilicura y San Bernardo). Tienen espacio casi ilimitado y almacenan los productos que llegan de los proveedores. Su costo de bodegaje es muy bajo.
            - 🏬 **Dark Store (Bodega Express):** Pequeña bodega dentro de los barrios (Providencia, Las Condes, Santiago Centro). No atiende público en mesón; su único objetivo es preparar pedidos para entregarlos en bicicleta, moto o furgón en menos de 2 horas.
            - 🛵 **Última Milla (Last-Mile):** El viaje final del paquete: desde la bodega urbana hasta la puerta de tu casa. Suele ser el tramo más caro y contaminante de toda la cadena logística (hasta el 50% del costo total).
            """
        )
    with g2:
        st.markdown(
            """
            - 🔀 **Cross-Fulfillment (Despacho de Respaldo):** Cuando la bodega de tu barrio se queda sin el producto que compraste, el sistema envía el paquete directamente desde el Centro periférico. Cuesta más flete, pero salva al cliente de un pedido cancelado.
            - 🎯 **OTIF (On-Time In-Full):** El indicador estrella del e-commerce. Mide el porcentaje de pedidos que llegaron a la hora prometida y con el 100% de los productos solicitados.
            - 📉 **Quiebre de Stock (Stockout):** Ocurre cuando un cliente quiere comprar algo y no hay stock en ninguna parte de la red, perdiendo la venta y dañando la fidelidad del cliente.
            """
        )

    st.markdown("---")
    st.markdown("#### 🔄 El Ciclo de Distribución en 4 Etapas")
    c_step1, c_step2, c_step3, c_step4 = st.columns(4)
    with c_step1:
        st.markdown(
            """
            <div style="background: #1A202C; border-top: 3px solid #60A5FA; padding: 14px; border-radius: 8px; font-size: 0.86rem; min-height: 160px;">
                <div style="font-weight: 700; color: #60A5FA; font-size: 0.95rem; margin-bottom: 6px;">1. Pronóstico con IA</div>
                Modelos de Machine Learning analizan el historial de ventas y patrones de consumo para predecir cuántas unidades comprará cada comuna en los próximos 14 días.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_step2:
        st.markdown(
            """
            <div style="background: #1A202C; border-top: 3px solid #34D399; padding: 14px; border-radius: 8px; font-size: 0.86rem; min-height: 160px;">
                <div style="font-weight: 700; color: #34D399; font-size: 0.95rem; margin-bottom: 6px;">2. Reabastecimiento Nocturno</div>
                Camiones de carga viajan de madrugada desde los grandes Centros Regionales (CDs) hacia las Dark Stores urbanas para rellenar los estantes antes de que empiece el día.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_step3:
        st.markdown(
            """
            <div style="background: #1A202C; border-top: 3px solid #FBBF24; padding: 14px; border-radius: 8px; font-size: 0.86rem; min-height: 160px;">
                <div style="font-weight: 700; color: #FBBF24; font-size: 0.95rem; margin-bottom: 6px;">3. Despacho Express Diurno</div>
                Apenas el cliente compra por la app, el pedido se arma en la Dark Store de su barrio y sale en viaje corto hacia su domicilio.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_step4:
        st.markdown(
            """
            <div style="background: #1A202C; border-top: 3px solid #F87171; padding: 14px; border-radius: 8px; font-size: 0.86rem; min-height: 160px;">
                <div style="font-weight: 700; color: #F87171; font-size: 0.95rem; margin-bottom: 6px;">4. Respaldo de Emergencia</div>
                Si una bodega urbana se agota o sufre una falla, el optimizador activa de inmediato despachos de respaldo desde los CDs para que el cliente nunca se quede sin su compra.
            </div>
            """,
            unsafe_allow_html=True,
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
