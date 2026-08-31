# 📦 Supply Chain & Fulfillment Network Optimizer
> **Proyecto End-to-End de Analítica Predictiva y Prescriptiva (Machine Learning + MILP)**

---

## 1. Resumen Ejecutivo y Problema de Negocio

### Contexto
Un minorista omnicanal opera una red de distribución logística compuesta por:
* **2 Centros de Distribución Principales (CDs):** Alta capacidad de almacenamiento, costo de almacenamiento bajo por $m^3$, pero mayor distancia/tiempo hacia el cliente final.
* **8 Dark Stores / Micro-Hubs Urbanos:** Capacidad limitada, costo de almacenamiento por $m^3$ más alto, ubicados estratégicamente cerca de las zonas de mayor densidad de demanda (entregas Same-Day / Next-Day).
* **20 Zonas de Demanda de Clientes (Clusters Urbanos):** Generan órdenes diarias de múltiples categorías de SKUs.

### El Reto Operativo
* Si una Dark Store se queda sin stock de un SKU de alta rotación, el pedido debe ser despachado desde un CD lejano (**Cross-Fulfillment**) o se pierde la venta (**Stockout/Quiebre**), disparando los costos de última milla o penalizaciones de servicio.
* Si se sobreabastece una Dark Store, se satura el espacio físico disponible y se incurre en altos costos de mantenimiento de inventario (*holding costs*).
* La demanda futura es inherentemente estocástica y asimétrica.

### Objetivo del Proyecto
Diseñar e implementar un sistema integral que:
1. **Pronostique la demanda futura con incertidumbre cuantificada** a nivel SKU-Zona usando modelos de regresión por cuantiles ($P_{10}, P_{50}, P_{90}$).
2. **Optimice las decisiones operativas de inventario y despacho** mediante un modelo de Programación Lineal Entera Mixta (MILP) que minimice el costo logístico total sujeto a restricciones de capacidad y nivel de servicio.
3. **Ofrezca un simulador What-If interactivo** para evaluar la resiliencia de la red ante escenarios de disrupción de costos, cuellos de botella y picos de demanda.

---

## 2. Hipótesis de Negocio y Modelado

* **$H_1$ (Incertidumbre en Stock de Seguridad):** Utilizar pronósticos probabilísticos por cuantiles ($P_{90}$) para dimensionar el inventario de seguridad reduce los quiebres de stock en al menos un **15%** en comparación con métodos deterministas basados en media ($P_{50}$) con márgenes fijos.
* **$H_2$ (Optimización Prescriptiva vs. Heurísticas):** La asignación de inventario y fulfillment mediante un modelo MILP coordinado reduce el costo total de la red (transporte + almacenamiento + penalizaciones) en más de un **8%** respecto a una regla de despacho miope de "nodo más cercano sin restricción de capacidad".
* **$H_3$ (Resiliencia de Red ante Alzas de Transporte):** Existe un umbral crítico en el costo por kilómetro donde la red prefiere consolidar inventario de seguridad en Dark Stores urbanas a pesar del mayor costo de metro cúbico de almacenamiento, evitando despachos de larga distancia desde los CDs.

---

## 3. Fuentes de Datos y Esquema

Se utilizará como base el dataset público **M5 Forecasting (Walmart)** o **Corporación Favorita Grocery Sales**, combinado con una topología de red simulada geométricamente realista.

### Tablas y Entidades:
1. **`raw_sales_demand.parquet`:**
   * `date`: Fecha del registro.
   * `item_id`: Identificador del SKU.
   * `zone_id`: Zona de demanda del cliente.
   * `units_sold`: Unidades vendidas (demanda observada).
   * `price`: Precio unitario.
   * `is_promo`: Indicador binario de promoción.
2. **`network_topology.parquet`:**
   * `node_id`: Identificador del nodo logístico (ej. `CD_1`, `DS_1` a `DS_8`).
   * `node_type`: Tipo de nodo (`CD` o `DarkStore`).
   * `capacity_m3`: Capacidad máxima de almacenamiento volumétrico.
   * `holding_cost_unit`: Costo de almacenamiento por unidad de volumen por periodo ($/m^3 \cdot \text{día}$).
3. **`transport_matrix.parquet`:**
   * `from_node_id`, `to_zone_id`: Par origen-destino.
   * `distance_km`: Distancia calculada en carretera / Haversine.
   * `cost_per_unit_km`: Costo variable de transporte por unidad transportada por kilómetro.
   * `lead_time_days`: Tiempo de tránsito (ej. 0.5 días para DS, 2 días para CD).

---

## 4. Fases Detalladas de Ejecución

┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   FASE 1    │ ──> │   FASE 2    │ ──> │   FASE 3    │ ──> │   FASE 4    │
│ Topología & │     │ Demand ML   │     │ MILP Engine │     │ Dashboard & │
│ Datos       │     │ Forecasting │     │ Solver      │     │ Packaging   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘


---

### FASE 1: Ingeniería de Datos y Configuración de Topología
* **Entregables:**
  * Script `src/data/make_dataset.py`: Limpieza de series temporales y tratamiento de demanda censurada (quiebres históricos imputados).
  * Script `src/data/make_network.py`: Generación determinista de nodos, capacidades, coordenadas geográficas y matriz de distancias.
  * Pipeline de validación de esquemas con `Great Expectations` o validaciones nativas de tipos y valores nulos.
* **Archivos generados:**
  * `data/02_intermediate/demand_clean.parquet`
  * `data/02_intermediate/network_graph.parquet`

---

### FASE 2: Modelado Predictivo de Demanda con Incertidumbre
* **Entregables:**
  * Feature Engineering (`src/data/features.py`):
    * Lags temporales ($t-1, t-7, t-14, t-28$).
    * Rolling stats (media y desviación estándar móvil a 7 y 28 días).
    * Calendario (día de semana, mes, quincena, festivos).
  * Entrenamiento (`src/forecasting/train.py`):
    * Modelo base: **LightGBM Regressor** entrenado con función de pérdida Pinball Loss (`objective='quantile'`) para cuantiles $\alpha \in \{0.10, 0.50, 0.90\}$.
    * Validación temporal cruzada (*Time-Series Split / Expanding Window* sin data leakage).
  * Inferencia (`src/forecasting/predict.py`):
    * Generación de la matriz de predicción a 7/14 días hacia adelante.
* **Archivos generados:**
  * `models/lgbm_quantile_p10.pkl`, `models/lgbm_quantile_p50.pkl`, `models/lgbm_quantile_p90.pkl`
  * `data/03_output/forecast_demand_quantiles.parquet`

---

### FASE 3: Motor de Optimización Prescriptiva (MILP)
* **Formulación Matemática:**
  * **Conjuntos:**
    * $\mathcal{I}$: SKUs / Productos.
    * $\mathcal{K}$: Nodos de suministro (CDs + Dark Stores).
    * $\mathcal{J}$: Zonas de demanda.
    * $\mathcal{T}$: Horizonte temporal de planificación.
  * **Parámetros:**
    * $\hat{D}_{i,j,t}$: Demanda pronosticada (usando cuantil $P_{50}$ o $P_{90}$ según aversión al riesgo).
    * $C^{\text{transp}}_{k,j}$: Costo de transporte por unidad del nodo $k$ a la zona $j$.
    * $C^{\text{hold}}_{k}$: Costo unitario de almacenamiento en el nodo $k$.
    * $C^{\text{stockout}}_{i}$: Penalización por unidad no satisfecha del producto $i$.
    * $\text{Cap}_k$: Capacidad máxima volumétrica del nodo $k$.
    * $v_i$: Volumen cúbico del producto $i$.
  * **Variables de Decisión:**
    * $X_{i,k,j,t} \ge 0$: Cantidad de producto $i$ despachada desde el nodo $k$ a la zona $j$ en el tiempo $t$.
    * $I_{i,k,t} \ge 0$: Inventario remanente del producto $i$ en el nodo $k$ al final del tiempo $t$.
    * $S_{i,j,t} \ge 0$: Ventas perdidas / quiebre de stock del producto $i$ en la zona $j$ en el tiempo $t$.
    * $R_{i,k,t} \ge 0$: Reabastecimiento enviado desde CDs a la Dark Store $k$.
* **Función Objetivo:**
  $$\min \sum_{t \in \mathcal{T}} \left[ \sum_{i,k,j} \left( C^{\text{transp}}_{k,j} \cdot X_{i,k,j,t} \right) + \sum_{i,k} \left( C^{\text{hold}}_k \cdot v_i \cdot I_{i,k,t} \right) + \sum_{i,j} \left( C^{\text{stockout}}_i \cdot S_{i,j,t} \right) \right]$$
* **Restricciones Principales:**
  1. **Balance de Inventario en Nodos:** $I_{i,k,t} = I_{i,k,t-1} + R_{i,k,t} - \sum_{j} X_{i,k,j,t} \quad \forall i, k, t$
  2. **Satisfacción de Demanda:** $\sum_{k} X_{i,k,j,t} + S_{i,j,t} = \hat{D}_{i,j,t} \quad \forall i, j, t$
  3. **Capacidad de Almacenamiento:** $\sum_{i} v_i \cdot I_{i,k,t} \le \text{Cap}_k \quad \forall k, t$
* **Implementación:**
  * Módulo `src/optimization/problem_builder.py` con `PuLP` / `Google OR-Tools`.
  * Solver `src/optimization/solver.py` que exporta la solución óptima y el status de factibilidad (`Optimal`, `Infeasible`).
* **Archivos generados:**
  * `data/03_output/optimal_fulfillment_plan.parquet`

---

### FASE 4: Dashboard Interactivo, Pruebas y Presentación
* **Entregables:**
  * **Dashboard en Streamlit (`app/app.py`):**
    * *Vista Geográfica:* Mapa interactivo (`pydeck`) con arcos de flujo desde CDs/Dark Stores hacia las zonas demandantes.
    * *Simulador What-If:* Controles dinámicos para modificar costos de combustible (+20%), fallas en Dark Stores (nodo inactivo) o aumentos repentinos de demanda (+30%).
    * *Comparador de Costos:* Gráfico de barras comparando el costo total de la solución óptima vs. solución heurística naive.
  * **Suite de Tests Unitarios (`tests/`):**
    * `test_forecasting.py`: Verifica ausencia de valores negativos y monotonicidad de cuantiles ($P_{10} \le P_{50} \le P_{90}$).
    * `test_optimization.py`: Verifica que la solución del solver nunca supere la capacidad máxima de ningún nodo.
  * **Documentación Ejecutiva:**
    * `README.md` con arquitectura, instrucciones de ejecución (`make run`, `docker run`) y resumen de resultados.

---

## 5. Métricas de Evaluación e Impacto de Negocio

### A. Métricas Técnicas de Machine Learning
1. **WAPE (Weighted Absolute Percentage Error):**
   $$\text{WAPE} = \frac{\sum |y - \hat{y}|}{\sum y}$$
2. **Pinball Loss (para cuantiles $\alpha$):** Mide la calibración de los pronósticos probabilísticos.

### B. Métricas Operativas y Financieras (Impacto de Negocio)
1. **Costo Logístico Total por Pedido ($/orden):** Suma ponderada de transporte, almacenamiento y penalizaciones dividida por el total de pedidos.
2. **Service Level / On-Time In-Full (OTIF %):**
   $$\text{OTIF} = \frac{\sum \text{Demanda Satisfecha}}{\text{Demanda Total}} \times 100$$
3. **Cross-Fulfillment Rate (%):** Porcentaje de órdenes que debieron ser cubiertas por un CD en vez del micro-hub local asignado.
4. **Utilización de Capacidad (%):** Porcentaje promedio del volumen cúbico utilizado en cada Dark Store.
