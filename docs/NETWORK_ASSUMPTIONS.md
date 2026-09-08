# 📘 Fundamentos y Supuestos de la Red Logística (Santiago de Chile)

> **Documento de Arquitectura y Negocio**  
> *Racional detrás de la elección de ubicaciones, capacidades, factores viales, costos y tiempos de respuesta.*

---

## 1. Contexto Geográfico: ¿Por qué Santiago de Chile?

Santiago concentra aproximadamente el **40% de la población de Chile** y cerca del **60% de las transacciones de comercio electrónico** a nivel nacional (fuente: *Cámara de Comercio de Santiago - CCS*). 

La cuenca metropolitana presenta una configuración geográfica particular:
1. **Periferia industrial accesible:** Conectada por autopistas urbanas de alta velocidad (Costanera Norte, Vespucio Norte, Vespucio Sur, Autopista Central).
2. **Alta densidad urbana central y oriente:** Comunas como Santiago Centro, Providencia y Las Condes con alta densidad demográfica y demanda de despachos Same-Day / Next-Day.
3. **Fricción de congestión urbana:** El tráfico intraurbano encarece la última milla si se opera exclusivamente desde bodegas remotas.

---

## 2. Nodos Logísticos y Supuestos Operativos

### 2.1 Centros de Distribución Principales (CDs)

Los CDs están ubicados en los dos polos logísticos e industriales más consolidados de la Región Metropolitana:

| Nodo ID | Ubicación Real | Comuna | Justificación Territorial | Capacidad ($m^3$) | Costo Almacenamiento ($/m^3 \cdot \text{día}$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `CD_PUDAHUEL` | Parque de Negocios Enea / Aeropuerto | Pudahuel | Polo logístico e industrial consolidado del poniente. Salida inmediata a Autopista Costanera Norte y Vespucio Norte. | $25.000$ | $\$0.40$ |
| `CD_SAN_BERNARDO` | Eje Panamericana Sur / Nos | San Bernardo | Polo de grandes centros de distribución del sur (ej. Walmart El Peñón, CCU, Falabella). Acceso a Autopista Central / Ruta 5 Sur. | $30.000$ | $\$0.35$ |

#### Fundamento de los Parámetros:
* **Capacidades ($25.000 - 30.000\ m^3$):** Corresponden a naves industriales Clase A de aproximadamente $3.500\text{ a }4.500\ m^2$ con racks de $7\text{ a }8$ metros de altura útil.
* **Costo de Almacenamiento ($\$0.35 - \$0.40\ \text{USD}/m^3\cdot\text{día}$):** 
  En arriendos industriales periféricos en Santiago, el valor promedio bordea los **$0.12 - $0.16\ \text{UF}/m^2/\text{mes}$** (aprox. $\$4.5 - \$6.0\ \text{USD}/m^2/\text{mes}$). Al prorratear a volumen cúbico y sumar costos de operación, seguros, refrigeración base y amortización, el costo diario por metro cúbico resulta en torno a los $\$0.35 - \$0.40\ \text{USD}$.

---

### 2.2 Dark Stores / Micro-Hubs Urbanos (DS)

Ubicados en zonas residenciales y mixtas de alta densidad para cumplir promesas de entrega ultra-rápida (2 a 4 horas o Same-Day):

| Nodo ID | Comuna | Racional de Ubicación | Capacidad ($m^3$) | Costo Almacenamiento ($/m^3 \cdot \text{día}$) |
| :--- | :--- | :--- | :--- | :--- |
| `DS_SANTIAGO_CENTRO` | Santiago | Densidad residencial en edificios de alta altura; alta rotación de pedidos pequeños. | $1.200$ | $\$1.40$ |
| `DS_PROVIDENCIA` | Providencia | Clientes profesionales, alta frecuencia de compra y adopción de delivery digital. | $1.000$ | $\$1.65$ |
| `DS_LAS_CONDES` | Las Condes | Mayor ticket promedio; arriendos comerciales de alto costo. | $1.500$ | $\$1.80$ |
| `DS_MAIPU` | Maipú | Segunda comuna más poblada de Santiago; gran volumen del poniente. | $1.800$ | $\$1.10$ |
| `DS_LA_FLORIDA` | La Florida | Eje estructurante de la zona sur-oriente; alta demanda familiar. | $1.600$ | $\$1.20$ |
| `DS_NUNOA` | Ñuñoa | Segmento joven, rápido crecimiento en torres residenciales y consumo on-demand. | $1.100$ | $\$1.50$ |
| `DS_QUILICURA` | Quilicura | Cobertura norte del área metropolitana y puente con comunas aledañas. | $2.000$ | $\$0.95$ |
| `DS_SAN_MIGUEL` | San Miguel | Eje Gran Avenida; conectividad con el anillo intermedio sur. | $1.300$ | $\$1.25$ |

#### Fundamento de los Parámetros:
* **Capacidades ($1.000 - 2.000\ m^3$):** Galpones o locales comerciales acondicionados de $300\text{ a }600\ m^2$ con estanterías de baja altura ($2.5 - 3.5\ m$).
* **Costo de Almacenamiento ($\$0.95 - \$1.80\ \text{USD}/m^3\cdot\text{día}$):** 
  El suelo urbano comercial en comunas como Las Condes o Providencia supera con creces el suelo industrial periférico (**$3\times$ a $5\times$ más caro por metro cuadrado**). Además, los costos fijos (personal de picking intensivo, climatización, seguridad) se dividen entre mucho menos metros cúbicos disponibles, elevando drásticamente el *holding cost* unitario.

---

### 2.3 Zonas de Demanda de Clientes (Clusters Urbanos)

Se definieron **20 centroides de demanda** representativos de los distintos estratos socioeconómicos y geográficos del Gran Santiago:
* **Cono Oriente (Ticket alto / frecuencia media-alta):** El Golf, Los Dominicos, La Dehesa, Vitacura, La Reina, Plaza Egaña.
* **Zona Centro (Volumen alto / densidad vertical):** Santiago Centro, Estación Central, Independencia, Recoleta, Quinta Normal.
* **Zona Sur y Sur-Oriente (Familias / alta masividad):** San Miguel, Macul, Peñalolén, La Florida (Norte/Sur), Puente Alto.
* **Zona Poniente (Grandes conglomerados suburbanos):** Maipú Centro, Pudahuel Sur.

---

## 3. Matriz de Transporte y Factores Viales

### 3.1 Factor de Tortuosidad (`CIRCUITY_FACTOR = 1.28`)
* **Definición:** Relación entre la distancia real en ruta por carretera ($d_{\text{ruta}}$) y la distancia euclidiana o geodésica lineal ($d_{\text{haversine}}$):
  $$\text{Circuity Factor} = \frac{d_{\text{ruta}}}{d_{\text{haversine}}}$$
* **¿Por qué 1.28?**
  * Estudios empíricos de ingeniería de transporte urbano (ej. *Ballou et al.*, mediciones empíricas sobre redes viales de ciudades latinoamericanas y la trama ortogonal/diagonal de Santiago) sitúan el factor urbano entre **1.25 y 1.35**. 
  * Un valor de **1.28** ajusta con alta precisión los trazados que combinan autopistas urbanas de circunvalación (Américo Vespucio) con la grilla de calles interiores.

---

### 3.2 Estructura de Costos de Transporte (`TRANSPORT_COST_PER_M3_KM`)

Los costos varían drásticamente según el vehículo y la escala del envío:

$$\text{Costo Unitario} = \text{Distancia (km)} \times \text{Tarifa ($/m^3\cdot\text{km})}$$

| Tramo | Vehículo Típico | Tarifa ($/m^3\cdot\text{km}$) | Racional Logístico |
| :--- | :--- | :--- | :--- |
| **`CD -> DS`** | Camión Rampla / 10-15 Ton ($40 - 70\ m^3$) | **$\$0.045$** | **Economías de escala masivas:** Despachos de abastecimiento primario consolidados en pallets completos y autopistas de alta velocidad. |
| **`DS -> DS`** | Camión 3/4 o Van cerrada ($10 - 20\ m^3$) | **$\$0.080$** | **Transferencias laterales intermedias:** Traslado de stock entre tiendas urbanas sin pasar por el CD. Mayor costo que camión pesado pero menor que despacho a cliente. |
| **`CD -> ZONE`** | Camión liviano urbano | **$\$0.120$** | **Cross-fulfillment de emergencia:** Ocurre cuando la Dark Store quiebra stock y el CD asume la entrega al cliente. Es costoso por recorrer largas distancias en camiones medianos. |
| **`DS -> ZONE`** | Vans urbanas, furgones, motos | **$\$0.160$** | **Última milla capilar:** Entrega individualizada puerta a puerta, paradas frecuentes, tiempo ocioso en semáforos y accesos a conserjerías. |

---

### 3.3 Tiempos de Tránsito (`LEAD_TIME_DAYS`)

Representan el tiempo total de ciclo de la orden:

* **`DS -> ZONE` ($0.2\ \text{días} \approx 4.8\ \text{horas}$):** Entrega *Same-Day* express desde el micro-hub más cercano.
* **`DS -> DS` ($0.5\ \text{días} = 12\ \text{horas}$):** Traslado interdiario entre sucursales urbanas para rebalancear inventario.
* **`CD -> DS` ($1.0\ \text{día} = 24\ \text{horas}$):** Abastecimiento nocturno consolidado (*Next-Day Replenishment*).
* **`CD -> ZONE` ($2.0\ \text{días} = 48\ \text{horas}$):** Despacho estándar directo desde el CD cuando no hay disponibilidad local.

---

## 4. Dinámica de Negocio que Inducen estos Parámetros en el Modelo MILP

Esta calibración no es arbitraria; está diseñada para generar las **tensiones de decisión reales** que enfrenta un Director de Supply Chain:

1. **Trade-off Almacenamiento vs. Transporte:**
   * Almacenar en una Dark Store cuesta hasta **$4.5\times$ más** que en un CD ($\$1.80$ vs $\$0.40$).
   * Despachar desde el CD al cliente cuesta casi el **doble en transporte** y tarda **$10\times$ más tiempo** ($48$ hrs vs $4.8$ hrs).
2. **Rol de los Cuantiles ($P_{10}, P_{50}, P_{90}$):**
   * El modelo MILP no puede simplemente "llenar" todas las Dark Stores con stock de seguridad excesivo porque violará la capacidad cúbica ($1.000 - 1.500\ m^3$) o disparará el *holding cost*.
   * Un pronóstico de demanda $P_{90}$ para productos de alta rotación permite dimensionar el stock óptimo, mientras que productos lentos deberán permanecer centralizados en el CD.
