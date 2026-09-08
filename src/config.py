"""
Configuraciones globales, rutas de archivos, parámetros de simulación y costos de red.
"""
from pathlib import Path

# ------------------------------------------------------------------------------
# 1. RUTAS DEL PROYECTO
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "01_raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "02_intermediate"
OUTPUT_DATA_DIR = DATA_DIR / "03_output"

TOPOLOGY_PATH = INTERMEDIATE_DATA_DIR / "network_topology.parquet"
TRANSPORT_MATRIX_PATH = INTERMEDIATE_DATA_DIR / "transport_matrix.parquet"
DEMAND_CLEAN_PATH = INTERMEDIATE_DATA_DIR / "demand_clean.parquet"

# ------------------------------------------------------------------------------
# 2. REPRODUCIBILIDAD
# ------------------------------------------------------------------------------
RANDOM_SEED = 42

# ------------------------------------------------------------------------------
# 3. TOPOLOGÍA SANTIAGO DE CHILE
# ------------------------------------------------------------------------------
# 2 Centros de Distribución (CDs) periféricos / industriales
CD_NODES = {
    "CD_PUDAHUEL": {
        "name": "CD Central Pudahuel (Enea)",
        "comuna": "Pudahuel",
        "lat": -33.4250,
        "lon": -70.7850,
        "capacity_m3": 25000.0,
        "holding_cost_unit": 0.40,  # USD / m3 día (bajo)
    },
    "CD_SAN_BERNARDO": {
        "name": "CD Sur San Bernardo (Nos)",
        "comuna": "San Bernardo",
        "lat": -33.6050,
        "lon": -70.7150,
        "capacity_m3": 30000.0,
        "holding_cost_unit": 0.35,  # USD / m3 día (bajo)
    }
}

# 8 Dark Stores / Micro-Hubs Urbanos
DARK_STORE_NODES = {
    "DS_SANTIAGO_CENTRO": {
        "name": "DS Santiago Centro",
        "comuna": "Santiago",
        "lat": -33.4489,
        "lon": -70.6693,
        "capacity_m3": 1200.0,
        "holding_cost_unit": 1.40,
    },
    "DS_PROVIDENCIA": {
        "name": "DS Providencia",
        "comuna": "Providencia",
        "lat": -33.4310,
        "lon": -70.6120,
        "capacity_m3": 1000.0,
        "holding_cost_unit": 1.65,
    },
    "DS_LAS_CONDES": {
        "name": "DS Las Condes",
        "comuna": "Las Condes",
        "lat": -33.4080,
        "lon": -70.5670,
        "capacity_m3": 1500.0,
        "holding_cost_unit": 1.80,
    },
    "DS_MAIPU": {
        "name": "DS Maipú",
        "comuna": "Maipú",
        "lat": -33.5110,
        "lon": -70.7580,
        "capacity_m3": 1800.0,
        "holding_cost_unit": 1.10,
    },
    "DS_LA_FLORIDA": {
        "name": "DS La Florida",
        "comuna": "La Florida",
        "lat": -33.5225,
        "lon": -70.5980,
        "capacity_m3": 1600.0,
        "holding_cost_unit": 1.20,
    },
    "DS_NUNOA": {
        "name": "DS Ñuñoa",
        "comuna": "Ñuñoa",
        "lat": -33.4540,
        "lon": -70.5985,
        "capacity_m3": 1100.0,
        "holding_cost_unit": 1.50,
    },
    "DS_QUILICURA": {
        "name": "DS Quilicura Norte",
        "comuna": "Quilicura",
        "lat": -33.3650,
        "lon": -70.7300,
        "capacity_m3": 2000.0,
        "holding_cost_unit": 0.95,
    },
    "DS_SAN_MIGUEL": {
        "name": "DS San Miguel / Gran Avenida",
        "comuna": "San Miguel",
        "lat": -33.4950,
        "lon": -70.6540,
        "capacity_m3": 1300.0,
        "holding_cost_unit": 1.25,
    }
}

# 20 Zonas de Demanda (Clusters de Clientes) en Santiago
DEMAND_ZONES = {
    "ZONE_SANTIAGO_CENTRO": {"name": "Santiago Centro Histórico", "lat": -33.4410, "lon": -70.6540},
    "ZONE_PROVIDENCIA_ALTA": {"name": "Providencia / Manuel Montt", "lat": -33.4280, "lon": -70.6180},
    "ZONE_LAS_CONDES_BAJA": {"name": "El Golf / Alcántara", "lat": -33.4150, "lon": -70.5930},
    "ZONE_LAS_CONDES_ALTA": {"name": "Los Dominicos / San Carlos", "lat": -33.4020, "lon": -70.5420},
    "ZONE_VITACURA": {"name": "Vitacura / Lo Curro", "lat": -33.3850, "lon": -70.5820},
    "ZONE_LO_BARNECHEA": {"name": "La Dehesa", "lat": -33.3550, "lon": -70.5180},
    "ZONE_NUNOA_ORIENTE": {"name": "Ñuñoa Plaza Egaña", "lat": -33.4530, "lon": -70.5720},
    "ZONE_LA_REINA": {"name": "La Reina Alta", "lat": -33.4420, "lon": -70.5360},
    "ZONE_MACUL": {"name": "Macul Residencial", "lat": -33.4860, "lon": -70.5990},
    "ZONE_PENALOLEN": {"name": "Peñalolén Alto", "lat": -33.4820, "lon": -70.5310},
    "ZONE_LA_FLORIDA_NORTE": {"name": "La Florida Walker Martínez", "lat": -33.5130, "lon": -70.6020},
    "ZONE_LA_FLORIDA_SUR": {"name": "La Florida Bellavista", "lat": -33.5410, "lon": -70.5890},
    "ZONE_PUENTE_ALTO": {"name": "Puente Alto Plaza", "lat": -33.6120, "lon": -70.5750},
    "ZONE_SAN_MIGUEL": {"name": "San Miguel El Llano", "lat": -33.4910, "lon": -70.6510},
    "ZONE_ESTACION_CENTRAL": {"name": "Estación Central Alameda", "lat": -33.4560, "lon": -70.6920},
    "ZONE_MAIPU_CENTRO": {"name": "Maipú Plaza / Pajaritos", "lat": -33.5100, "lon": -70.7570},
    "ZONE_PUDAHUEL_SUR": {"name": "Pudahuel Sur", "lat": -33.4550, "lon": -70.7520},
    "ZONE_QUINTA_NORMAL": {"name": "Quinta Normal / Matucana", "lat": -33.4350, "lon": -70.6890},
    "ZONE_INDEPENDENCIA": {"name": "Independencia / Hospitales", "lat": -33.4180, "lon": -70.6580},
    "ZONE_RECOLETA": {"name": "Recoleta Bellavista / El Salto", "lat": -33.4120, "lon": -70.6380},
}

# ------------------------------------------------------------------------------
# 4. PARÁMETROS DE COSTO DE TRANSPORTE Y LEAD TIME
# ------------------------------------------------------------------------------
# Factor de tortuosidad vial promedio en Santiago (distancia en ruta vs. Haversine)
CIRCUITY_FACTOR = 1.28

# Costos unitarios de transporte por m3 transportado por km ($/m3·km)
TRANSPORT_COST_PER_M3_KM = {
    ("CD", "DS"): 0.045,        # Camiones pesados / consolidado (económico por m3)
    ("CD", "ZONE"): 0.120,      # Despacho directo / cross-fulfillment de emergencia desde CD
    ("DS", "ZONE"): 0.160,      # Última milla urbana en vans / motos
    ("DS", "DS"): 0.080,        # Rebalanceo / transbordo entre Dark Stores
}

# Lead time en días según tipo de arco
LEAD_TIME_DAYS = {
    ("CD", "DS"): 1.0,          # Abastecimiento programado (Next-Day)
    ("CD", "ZONE"): 2.0,        # Entrega estándar lejana
    ("DS", "ZONE"): 0.2,        # Same-day / express última milla (horas)
    ("DS", "DS"): 0.5,          # Transferencia inter-hub (medio día)
}
