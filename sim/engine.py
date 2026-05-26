"""
GNL Logistics Simulation Engine
Motor de simulación de eventos discretos con SimPy.

Modela la cadena logística completa:
  Punto de llenado → Transporte terrestre → Punto de descarga → Retorno
"""

import simpy
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal, Optional
from scipy import stats


# ─── PARÁMETROS ───────────────────────────────────────────────────────────────

@dataclass
class SimParams:
    # Flota
    n_trucks: int = 20
    truck_capacity_m3: float = 50.0

    # Punto de llenado
    fill_bays: int = 3
    fill_time_mean_h: float = 1.5
    fill_time_std_h: float = 0.3

    # Transporte
    travel_dist: Literal["gamma", "erlang", "lognormal"] = "gamma"
    travel_outbound_mean_h: float = 4.0
    travel_cv: float = 0.25          # coeficiente de variación
    travel_return_mean_h: float = 3.5

    # Punto de descarga
    unload_bays: int = 2
    unload_time_mean_h: float = 1.2
    unload_time_std_h: float = 0.24

    # Destino: inventario y demanda
    initial_inventory_m3: float = 800.0
    demand_rate_m3h: float = 150.0   # consumo constante en destino

    # Simulación
    sim_duration_h: float = 168.0    # 7 días por defecto
    random_seed: Optional[int] = 42
    n_replicas: int = 1


# ─── DISTRIBUCIONES ───────────────────────────────────────────────────────────

class TravelTimeDistribution:
    """Muestrea tiempos de viaje según distribución elegida."""

    def __init__(self, mean: float, cv: float,
                 dist: str = "gamma", rng: np.random.Generator = None):
        self.mean = mean
        self.cv = cv
        self.dist = dist
        self.rng = rng or np.random.default_rng()

    def sample(self) -> float:
        mu, cv = self.mean, self.cv
        if self.dist == "gamma":
            k = 1 / (cv ** 2)
            theta = mu * cv ** 2
            return max(0.1, self.rng.gamma(shape=k, scale=theta))
        elif self.dist == "erlang":
            k = max(1, round(1 / (cv ** 2)))
            rate = k / mu
            return max(0.1, sum(-np.log(self.rng.uniform()) / rate for _ in range(k)))
        elif self.dist == "lognormal":
            sigma2 = np.log(1 + cv ** 2)
            mu_ln = np.log(mu) - sigma2 / 2
            return max(0.1, self.rng.lognormal(mean=mu_ln, sigma=np.sqrt(sigma2)))
        else:
            raise ValueError(f"Distribución desconocida: {self.dist}")


# ─── MÉTRICAS ─────────────────────────────────────────────────────────────────

@dataclass
class TruckMetrics:
    truck_id: int
    trips_completed: int = 0
    total_fill_wait_h: float = 0.0
    total_unload_wait_h: float = 0.0
    total_fill_time_h: float = 0.0
    total_unload_time_h: float = 0.0
    total_travel_out_h: float = 0.0
    total_travel_ret_h: float = 0.0
    volume_delivered_m3: float = 0.0


@dataclass
class SimResults:
    params: SimParams
    replica: int = 0

    # Series temporales
    time_series: list = field(default_factory=list)          # h
    inventory_series: list = field(default_factory=list)     # m³
    delivered_cum_series: list = field(default_factory=list) # m³ acumulado
    fill_queue_series: list = field(default_factory=list)    # camiones en cola llenado
    unload_queue_series: list = field(default_factory=list)  # camiones en cola descarga
    state_series: list = field(default_factory=list)         # dict estados por paso

    # Eventos puntuales
    delivery_log: list = field(default_factory=list)   # (time, truck_id, vol, inventory)
    stockout_log: list = field(default_factory=list)   # (time, deficit_m3)
    queue_events: list = field(default_factory=list)   # (time, location, wait_h)

    # Métricas por camión
    truck_metrics: list = field(default_factory=list)

    # KPIs resumen (calculados al final)
    kpi: dict = field(default_factory=dict)

    def compute_kpis(self):
        p = self.params
        total_h = p.sim_duration_h
        total_delivered = sum(d[2] for d in self.delivery_log)
        stockout_count = len(self.stockout_log)
        stockout_h = sum(s[2] for s in self.stockout_log) if self.stockout_log and len(self.stockout_log[0]) > 2 else 0

        all_fill_waits = [e[2] for e in self.queue_events if e[1] == "fill"]
        all_unload_waits = [e[2] for e in self.queue_events if e[1] == "unload"]
        trips = sum(tm.trips_completed for tm in self.truck_metrics)

        # Utilización: tiempo activo / (n_trucks * sim_duration)
        total_active = sum(
            tm.total_fill_time_h + tm.total_unload_time_h +
            tm.total_travel_out_h + tm.total_travel_ret_h
            for tm in self.truck_metrics
        )
        util = total_active / (p.n_trucks * total_h) if total_h > 0 else 0

        self.kpi = {
            "total_trips": trips,
            "total_delivered_m3": round(total_delivered, 1),
            "throughput_m3h": round(total_delivered / total_h, 2) if total_h > 0 else 0,
            "stockout_events": stockout_count,
            "truck_utilization_pct": round(util * 100, 1),
            "avg_fill_wait_h": round(np.mean(all_fill_waits), 3) if all_fill_waits else 0,
            "p95_fill_wait_h": round(np.percentile(all_fill_waits, 95), 3) if all_fill_waits else 0,
            "avg_unload_wait_h": round(np.mean(all_unload_waits), 3) if all_unload_waits else 0,
            "p95_unload_wait_h": round(np.percentile(all_unload_waits, 95), 3) if all_unload_waits else 0,
            "avg_cycle_h": round(total_h * p.n_trucks / trips, 2) if trips > 0 else 0,
            "final_inventory_m3": round(self.inventory_series[-1], 1) if self.inventory_series else 0,
            "min_inventory_m3": round(min(self.inventory_series), 1) if self.inventory_series else 0,
        }
        return self.kpi

    def to_dataframes(self):
        """Devuelve diccionario de DataFrames para análisis y export."""
        ts = pd.DataFrame({
            "time_h": self.time_series,
            "time_d": [t / 24 for t in self.time_series],
            "inventory_m3": self.inventory_series,
            "delivered_cum_m3": self.delivered_cum_series,
            "fill_queue": self.fill_queue_series,
            "unload_queue": self.unload_queue_series,
        })
        deliveries = pd.DataFrame(self.delivery_log,
                                  columns=["time_h", "truck_id", "volume_m3", "inventory_after_m3"])
        stockouts = pd.DataFrame(self.stockout_log,
                                 columns=["time_h", "deficit_m3", "duration_h"] if self.stockout_log else [])
        queues = pd.DataFrame(self.queue_events,
                              columns=["time_h", "location", "wait_h"])
        trucks = pd.DataFrame([
            {
                "truck_id": tm.truck_id + 1,
                "trips": tm.trips_completed,
                "volume_m3": tm.volume_delivered_m3,
                "fill_wait_h": round(tm.total_fill_wait_h, 2),
                "unload_wait_h": round(tm.total_unload_wait_h, 2),
                "fill_time_h": round(tm.total_fill_time_h, 2),
                "unload_time_h": round(tm.total_unload_time_h, 2),
                "travel_out_h": round(tm.total_travel_out_h, 2),
                "travel_ret_h": round(tm.total_travel_ret_h, 2),
                "utilization_pct": round(
                    (tm.total_fill_time_h + tm.total_unload_time_h +
                     tm.total_travel_out_h + tm.total_travel_ret_h)
                    / self.params.sim_duration_h * 100, 1
                ),
            }
            for tm in self.truck_metrics
        ])
        return {"timeseries": ts, "deliveries": deliveries,
                "stockouts": stockouts, "queues": queues, "trucks": trucks}


# ─── PROCESO DE CAMIÓN ────────────────────────────────────────────────────────

def truck_process(env, truck_id, p, fill_resource, unload_resource,
                  state, metrics, results, rng):
    """
    Proceso SimPy para un camión individual.
    Ciclo: esperar bahía llenado → llenar → viajar → esperar bahía descarga
           → descargar → regresar → repetir
    """
    tm = metrics[truck_id]
    travel_out = TravelTimeDistribution(p.travel_outbound_mean_h, p.travel_cv, p.travel_dist, rng)
    travel_ret = TravelTimeDistribution(p.travel_return_mean_h, p.travel_cv, p.travel_dist, rng)

    # Escalonar arranque para evitar sobrecarga inicial
    yield env.timeout(rng.uniform(0, min(2.0, p.travel_return_mean_h)))

    while True:
        # ── COLA LLENADO ─────────────────────────────────────────
        state["trucks"][truck_id] = "waiting_fill"
        t_arrive_fill = env.now
        with fill_resource.request() as req:
            yield req
            wait_fill = env.now - t_arrive_fill
            tm.total_fill_wait_h += wait_fill
            results.queue_events.append((env.now, "fill", wait_fill))

            # ── LLENADO ─────────────────────────────────────────
            state["trucks"][truck_id] = "loading"
            t_fill = max(0.1, rng.normal(p.fill_time_mean_h, p.fill_time_std_h))
            tm.total_fill_time_h += t_fill
            yield env.timeout(t_fill)

        # ── VIAJE DE IDA ─────────────────────────────────────────
        state["trucks"][truck_id] = "traveling_out"
        t_travel = travel_out.sample()
        tm.total_travel_out_h += t_travel
        yield env.timeout(t_travel)

        # ── COLA DESCARGA ────────────────────────────────────────
        state["trucks"][truck_id] = "waiting_unload"
        t_arrive_unload = env.now
        with unload_resource.request() as req:
            yield req
            wait_unload = env.now - t_arrive_unload
            tm.total_unload_wait_h += wait_unload
            results.queue_events.append((env.now, "unload", wait_unload))

            # ── DESCARGA ─────────────────────────────────────────
            state["trucks"][truck_id] = "unloading"
            t_unload = max(0.1, rng.normal(p.unload_time_mean_h, p.unload_time_std_h))
            tm.total_unload_time_h += t_unload
            yield env.timeout(t_unload)

            # Registrar entrega
            state["inventory"] += p.truck_capacity_m3
            state["total_delivered"] += p.truck_capacity_m3
            tm.trips_completed += 1
            tm.volume_delivered_m3 += p.truck_capacity_m3
            results.delivery_log.append((
                env.now, truck_id,
                p.truck_capacity_m3,
                round(state["inventory"], 1)
            ))

        # ── VIAJE DE REGRESO ──────────────────────────────────────
        state["trucks"][truck_id] = "returning"
        t_return = travel_ret.sample()
        tm.total_travel_ret_h += t_return
        yield env.timeout(t_return)


# ─── PROCESO DE DEMANDA ───────────────────────────────────────────────────────

def demand_process(env, p, state, results, dt=0.25):
    """
    Consume inventario continuamente según demanda.
    Registra series temporales y eventos de desabasto.
    """
    stockout_start = None
    last_snapshot = -dt

    while True:
        yield env.timeout(dt)

        # Consumo en este intervalo
        consumed = p.demand_rate_m3h * dt
        state["inventory"] -= consumed

        # Detección de desabasto
        if state["inventory"] < 0:
            deficit = abs(state["inventory"])
            state["inventory"] = 0
            state["stockouts"] += 1
            if stockout_start is None:
                stockout_start = env.now
            results.stockout_log.append((env.now, deficit, dt))
        else:
            stockout_start = None

        # Snapshot periódico (cada dt horas)
        if env.now - last_snapshot >= dt - 1e-6:
            n_s = state["trucks"].count("waiting_fill") + state["trucks"].count("loading")
            n_d = state["trucks"].count("waiting_unload") + state["trucks"].count("unloading")

            results.time_series.append(round(env.now, 4))
            results.inventory_series.append(round(max(0, state["inventory"]), 2))
            results.delivered_cum_series.append(round(state["total_delivered"], 2))
            results.fill_queue_series.append(
                state["trucks"].count("waiting_fill"))
            results.unload_queue_series.append(
                state["trucks"].count("waiting_unload"))
            results.state_series.append({
                "loading": state["trucks"].count("loading"),
                "traveling_out": state["trucks"].count("traveling_out"),
                "unloading": state["trucks"].count("unloading"),
                "returning": state["trucks"].count("returning"),
                "waiting_fill": state["trucks"].count("waiting_fill"),
                "waiting_unload": state["trucks"].count("waiting_unload"),
            })
            last_snapshot = env.now


# ─── FUNCIÓN PRINCIPAL ────────────────────────────────────────────────────────

def run_simulation(p: SimParams, replica: int = 0,
                   progress_callback=None) -> SimResults:
    """
    Ejecuta una réplica de la simulación.

    Args:
        p: Parámetros de simulación
        replica: Número de réplica (afecta semilla aleatoria)
        progress_callback: función(pct_float) para barra de progreso

    Returns:
        SimResults con todas las métricas y series
    """
    seed = (p.random_seed or 0) + replica * 1000
    rng = np.random.default_rng(seed)

    env = simpy.Environment()
    results = SimResults(params=p, replica=replica)

    # Estado compartido mutable
    state = {
        "inventory": p.initial_inventory_m3,
        "total_delivered": 0.0,
        "stockouts": 0,
        "trucks": ["idle"] * p.n_trucks,
    }

    # Recursos SimPy
    fill_resource = simpy.Resource(env, capacity=p.fill_bays)
    unload_resource = simpy.Resource(env, capacity=p.unload_bays)

    # Métricas individuales por camión
    metrics = [TruckMetrics(truck_id=i) for i in range(p.n_trucks)]

    # Lanzar procesos de camiones
    for i in range(p.n_trucks):
        env.process(truck_process(
            env, i, p, fill_resource, unload_resource,
            state, metrics, results, rng
        ))

    # Proceso de demanda y registro
    env.process(demand_process(env, p, state, results, dt=0.25))

    # Ejecutar
    env.run(until=p.sim_duration_h)

    results.truck_metrics = metrics
    results.compute_kpis()
    return results


def run_replicas(p: SimParams, progress_callback=None) -> list[SimResults]:
    """Ejecuta n_replicas y devuelve lista de resultados."""
    all_results = []
    for i in range(p.n_replicas):
        if progress_callback:
            progress_callback(i / p.n_replicas)
        r = run_simulation(p, replica=i)
        all_results.append(r)
    if progress_callback:
        progress_callback(1.0)
    return all_results


def aggregate_replicas(results: list[SimResults]) -> pd.DataFrame:
    """
    Agrega KPIs de múltiples réplicas en un DataFrame con
    media, desviación estándar e intervalos de confianza al 95%.
    """
    kpis = [r.kpi for r in results]
    df = pd.DataFrame(kpis)
    agg = pd.DataFrame({
        "KPI": df.columns,
        "Media": df.mean().round(3),
        "Std": df.std().round(3),
        "Min": df.min().round(3),
        "Max": df.max().round(3),
        "IC95_inf": (df.mean() - 1.96 * df.std() / np.sqrt(len(df))).round(3),
        "IC95_sup": (df.mean() + 1.96 * df.std() / np.sqrt(len(df))).round(3),
    }).reset_index(drop=True)
    return agg
