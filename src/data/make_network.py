"""
Generador de topología de red logística y matriz de transporte para Santiago de Chile.
Genera los nodos (CDs, Dark Stores, Zonas) y calcula distancias y costos entre todos los pares permitidos.
"""
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Permitir ejecución tanto directa (python src/data/make_network.py) como módulo (-m src.data.make_network)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# pyrefly: ignore [missing-import]
from src.config import (
    CD_NODES,
    DARK_STORE_NODES,
    DEMAND_ZONES,
    CIRCUITY_FACTOR,
    TRANSPORT_COST_PER_M3_KM,
    LEAD_TIME_DAYS,
    TOPOLOGY_PATH,
    TRANSPORT_MATRIX_PATH,
    INTERMEDIATE_DATA_DIR,
)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula la distancia Haversine en kilómetros entre dos coordenadas geográficas."""
    R = 6371.0  # Radio medio de la Tierra en km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def build_network_topology() -> pd.DataFrame:
    """Construye el DataFrame con todos los nodos de la red (CDs, Dark Stores y Zonas de Demanda)."""
    records = []

    # 1. CDs
    for node_id, data in CD_NODES.items():
        records.append({
            "node_id": node_id,
            "node_name": data["name"],
            "comuna": data["comuna"],
            "node_type": "CD",
            "lat": data["lat"],
            "lon": data["lon"],
            "capacity_m3": data["capacity_m3"],
            "holding_cost_unit": data["holding_cost_unit"],
        })

    # 2. Dark Stores
    for node_id, data in DARK_STORE_NODES.items():
        records.append({
            "node_id": node_id,
            "node_name": data["name"],
            "comuna": data["comuna"],
            "node_type": "DarkStore",
            "lat": data["lat"],
            "lon": data["lon"],
            "capacity_m3": data["capacity_m3"],
            "holding_cost_unit": data["holding_cost_unit"],
        })

    # 3. Zonas de Demanda
    for zone_id, data in DEMAND_ZONES.items():
        records.append({
            "node_id": zone_id,
            "node_name": data["name"],
            "comuna": zone_id.replace("ZONE_", "").title(),
            "node_type": "DemandZone",
            "lat": data["lat"],
            "lon": data["lon"],
            "capacity_m3": 0.0,
            "holding_cost_unit": 0.0,
        })

    df_topology = pd.DataFrame(records)
    return df_topology


def build_transport_matrix(df_topology: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la matriz completa de transporte con todas las combinaciones operativas:
      - CD -> DarkStore (abastecimiento primario)
      - CD -> DemandZone (cross-fulfillment de respaldo)
      - DarkStore -> DemandZone (última milla estándar)
      - DarkStore -> DarkStore (balanceo y transferencias laterales)
    """
    node_dict = df_topology.set_index("node_id").to_dict(orient="index")
    
    cds = [nid for nid, d in node_dict.items() if d["node_type"] == "CD"]
    dark_stores = [nid for nid, d in node_dict.items() if d["node_type"] == "DarkStore"]
    zones = [nid for nid, d in node_dict.items() if d["node_type"] == "DemandZone"]

    allowed_connections = []

    # 1. CD -> DS
    for cd in cds:
        for ds in dark_stores:
            allowed_connections.append((cd, ds, ("CD", "DS")))

    # 2. CD -> ZONE
    for cd in cds:
        for zone in zones:
            allowed_connections.append((cd, zone, ("CD", "ZONE")))

    # 3. DS -> ZONE
    for ds in dark_stores:
        for zone in zones:
            allowed_connections.append((ds, zone, ("DS", "ZONE")))

    # 4. DS -> DS (bidireccional, sin autoloop)
    for ds1 in dark_stores:
        for ds2 in dark_stores:
            if ds1 != ds2:
                allowed_connections.append((ds1, ds2, ("DS", "DS")))

    records = []
    for from_id, to_id, link_type in allowed_connections:
        from_data = node_dict[from_id]
        to_data = node_dict[to_id]

        haversine_km = haversine_distance(
            from_data["lat"], from_data["lon"], to_data["lat"], to_data["lon"]
        )
        road_distance_km = round(haversine_km * CIRCUITY_FACTOR, 2)

        unit_cost_km = TRANSPORT_COST_PER_M3_KM[link_type]
        base_cost_unit = round(road_distance_km * unit_cost_km, 3)
        lead_time = LEAD_TIME_DAYS[link_type]

        records.append({
            "from_node_id": from_id,
            "to_node_id": to_id,
            "link_type": f"{link_type[0]}_to_{link_type[1]}",
            "haversine_km": round(haversine_km, 2),
            "distance_km": road_distance_km,
            "cost_per_m3_km": unit_cost_km,
            "transport_cost_per_m3": base_cost_unit,
            "lead_time_days": lead_time,
        })

    df_transport = pd.DataFrame(records)
    return df_transport


def main():
    INTERMEDIATE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("🛰️  Generando topología de red logística para Santiago de Chile...")
    df_topology = build_network_topology()
    df_topology.to_parquet(TOPOLOGY_PATH, index=False)
    print(f"✓ Topología guardada en: {TOPOLOGY_PATH} ({len(df_topology)} nodos: "
          f"{len(CD_NODES)} CDs, {len(DARK_STORE_NODES)} Dark Stores, {len(DEMAND_ZONES)} Zonas)")

    print("🛣️  Calculando matriz de transporte con distancias y costos viales...")
    df_transport = build_transport_matrix(df_topology)
    df_transport.to_parquet(TRANSPORT_MATRIX_PATH, index=False)
    print(f"✓ Matriz de transporte guardada en: {TRANSPORT_MATRIX_PATH} ({len(df_transport)} arcos de transporte)")

    print("\nResumen de arcos por tipo de conexión:")
    print(df_transport["link_type"].value_counts().to_string())

    print("\nEjemplo de arcos calculados:")
    print(df_transport[["from_node_id", "to_node_id", "link_type", "distance_km", "transport_cost_per_m3", "lead_time_days"]].head(8))


if __name__ == "__main__":
    main()
