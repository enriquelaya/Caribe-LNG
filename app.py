"""
GNL Logistics Simulator — Streamlit Dashboard
Ejecutar con: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim.engine import SimParams, run_simulation, run_replicas, aggregate_replicas

# ─── CONFIGURACIÓN PÁGINA ────────────────────────────────────────────────────

st.set_page_config(
    page_title="GNL Logistics Simulator",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 600; }
  .block-container { padding-top: 1rem; padding-bottom: 1rem; }
  h1 { font-size: 1.4rem !important; }
  h2 { font-size: 1.1rem !important; }
  .stTabs [data-baseweb="tab"] { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ─── COLORES ─────────────────────────────────────────────────────────────────

C = {
    "teal":    "#26a641",
    "blue":    "#58a6ff",
    "amber":   "#d29922",
    "purple":  "#bc8cff",
    "red":     "#f85149",
    "gray":    "#8b949e",
    "bg":      "#0d1117",
    "bg2":     "#161b22",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="monospace", size=11, color="#c9d1d9"),
    margin=dict(l=40, r=16, t=32, b=32),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d", borderwidth=1),
)

# ─── SIDEBAR: PARÁMETROS ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚛 GNL Logistics Sim")
    st.markdown("---")

    st.markdown("### Flota")
    n_trucks = st.slider("Número de camiones", 1, 70, 20)
    truck_cap = st.selectbox("Capacidad (m³)", [10, 20, 30, 40, 50], index=4)

    st.markdown("### Punto de Llenado")
    fill_bays = st.slider("Bahías de carga", 1, 10, 3)
    fill_mean = st.slider("T. llenado μ (h)", 0.5, 4.0, 1.5, 0.1)
    fill_std  = st.slider("T. llenado σ (h)", 0.05, 1.0, 0.3, 0.05)

    st.markdown("### Ruta de Transporte")
    dist_map = {
        "Gamma (paradas variables)": "gamma",
        "Erlang (paradas fijas)": "erlang",
        "Lognormal": "lognormal",
    }
    dist_label = st.selectbox("Distribución", list(dist_map.keys()))
    travel_dist = dist_map[dist_label]
    travel_mean = st.slider("T. viaje ida μ (h)", 1.0, 16.0, 9.2, 0.5)
    travel_cv   = st.slider("Coef. variación (cv)", 0.05, 0.80, 0.25, 0.05)
    return_mean = st.slider("T. regreso μ (h)", 1.0, 16.0, 9.2, 0.5)

    st.markdown("### Punto de Descarga")
    unload_bays = st.slider("Bahías de descarga", 1, 8, 2)
    unload_mean = st.slider("T. descarga μ (h)", 0.3, 3.0, 1.2, 0.1)
    demand      = st.slider("Demanda destino (m³/h)", 20, 600, 150, 10)
    inv0        = st.slider("Inventario inicial (m³)", 100, 3000, 800, 100)

    st.markdown("### Simulación")
    sim_days    = st.slider("Duración (días)", 1, 30, 15)
    n_replicas  = st.slider("Réplicas", 1, 20, 1)
    seed        = st.number_input("Semilla aleatoria", value=42, step=1)

    run_btn = st.button("▶ EJECUTAR SIMULACIÓN", use_container_width=True, type="primary")

# ─── ENCABEZADO ───────────────────────────────────────────────────────────────

st.markdown("# 🏭 GNL Logistics — Simulación de Cadena de Suministro")
st.markdown(
    f"**{n_trucks} camiones × {truck_cap} m³** | "
    f"Llenado: {fill_bays} bahías | Descarga: {unload_bays} bahías | "
    f"Demanda: {demand} m³/h | Duración: {sim_days} d"
)

# ─── EJECUCIÓN ────────────────────────────────────────────────────────────────

if run_btn:
    p = SimParams(
        n_trucks=n_trucks,
        truck_capacity_m3=float(truck_cap),
        fill_bays=fill_bays,
        fill_time_mean_h=fill_mean,
        fill_time_std_h=fill_std,
        travel_dist=travel_dist,
        travel_outbound_mean_h=travel_mean,
        travel_cv=travel_cv,
        travel_return_mean_h=return_mean,
        unload_bays=unload_bays,
        unload_time_mean_h=unload_mean,
        unload_time_std_h=unload_mean * 0.2,
        demand_rate_m3h=float(demand),
        initial_inventory_m3=float(inv0),
        sim_duration_h=float(sim_days * 24),
        random_seed=int(seed),
        n_replicas=n_replicas,
    )

    progress_bar = st.progress(0, text="Ejecutando simulación...")

    def cb(pct):
        progress_bar.progress(pct, text=f"Simulando... {int(pct*100)}%")

    all_results = run_replicas(p, progress_callback=cb)
    progress_bar.empty()

    st.session_state["results"] = all_results
    st.session_state["params"] = p

# ─── VISUALIZACIÓN ────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.info("👈 Configure los parámetros en el panel izquierdo y presione **EJECUTAR SIMULACIÓN**.")
    st.stop()

all_results = st.session_state["results"]
p = st.session_state["params"]
# Usar primera réplica para gráficas detalladas
r = all_results[0]
dfs = r.to_dataframes()
ts = dfs["timeseries"]

tabs = st.tabs(["📊 KPIs & Inventario", "🚛 Análisis de Flota",
                "📈 Series Temporales", "🔁 Réplicas", "📥 Exportar datos"])

# ── TAB 1: KPIs ───────────────────────────────────────────────────────────────
with tabs[0]:
    kpi = r.kpi
    cols = st.columns(5)
    cols[0].metric("Viajes completados", f"{kpi['total_trips']:,}")
    cols[1].metric("GNL entregado (m³)", f"{kpi['total_delivered_m3']:,.0f}")
    cols[2].metric("Throughput (m³/h)", f"{kpi['throughput_m3h']:.1f}")
    cols[3].metric("Utilización flota", f"{kpi['truck_utilization_pct']:.1f}%")
    cols[4].metric("Eventos desabasto", str(kpi["stockout_events"]),
                   delta="⚠ CRÍTICO" if kpi["stockout_events"] > 0 else "✓ OK",
                   delta_color="inverse")

    cols2 = st.columns(4)
    cols2[0].metric("Ciclo promedio (h)", f"{kpi['avg_cycle_h']:.2f}")
    cols2[1].metric("Espera prom. llenado (h)", f"{kpi['avg_fill_wait_h']:.3f}")
    cols2[2].metric("Espera prom. descarga (h)", f"{kpi['avg_unload_wait_h']:.3f}")
    cols2[3].metric("Inventario mínimo (m³)", f"{kpi['min_inventory_m3']:,.0f}")

    st.markdown("---")

    # Inventario + línea de alarma
    col_inv, col_pie = st.columns([3, 1])
    with col_inv:
        fig_inv = go.Figure()
        # Zona crítica (< 4 h autonomía)
        alarm = p.demand_rate_m3h * 4
        fig_inv.add_hrect(y0=0, y1=alarm, fillcolor="rgba(248,81,73,0.08)",
                          line_width=0, annotation_text="Zona crítica",
                          annotation_position="top left",
                          annotation_font_color=C["red"])
        fig_inv.add_trace(go.Scatter(
            x=ts["time_d"], y=ts["inventory_m3"],
            mode="lines", name="Inventario",
            line=dict(color=C["blue"], width=1.5),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.08)"
        ))
        fig_inv.add_hline(y=alarm, line_dash="dot", line_color=C["red"],
                          annotation_text=f"Alarma 4h ({alarm:.0f} m³)")
        fig_inv.add_trace(go.Scatter(
            x=ts["time_d"], y=ts["inventory_m3"].rolling(8).mean(),
            mode="lines", name="Media móvil 2h",
            line=dict(color=C["amber"], width=1, dash="dash")
        ))
        fig_inv.update_layout(title="Inventario en punto de destino (m³)",
                              xaxis_title="Tiempo (días)",
                              yaxis_title="Inventario (m³)",
                              **PLOTLY_LAYOUT)
        st.plotly_chart(fig_inv, use_container_width=True)

    with col_pie:
        # Tiempo promedio de ciclo breakdown
        tm_list = r.truck_metrics
        avg = {
            "Llenado": np.mean([t.total_fill_time_h / max(t.trips_completed,1) for t in tm_list]),
            "Viaje ida": np.mean([t.total_travel_out_h / max(t.trips_completed,1) for t in tm_list]),
            "Descarga": np.mean([t.total_unload_time_h / max(t.trips_completed,1) for t in tm_list]),
            "Retorno": np.mean([t.total_travel_ret_h / max(t.trips_completed,1) for t in tm_list]),
            "Espera fill": np.mean([t.total_fill_wait_h / max(t.trips_completed,1) for t in tm_list]),
            "Espera desc": np.mean([t.total_unload_wait_h / max(t.trips_completed,1) for t in tm_list]),
        }
        fig_pie = go.Figure(go.Pie(
            labels=list(avg.keys()),
            values=[round(v, 2) for v in avg.values()],
            hole=0.5,
            marker=dict(colors=[C["teal"], C["blue"], C["amber"],
                                 C["purple"], C["red"], "#d04060"]),
            textfont_size=10,
        ))
        fig_pie.update_layout(title="Desglose ciclo promedio (h)",
                              showlegend=True,
                              legend=dict(font=dict(size=9)),
                              **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "legend")})
        st.plotly_chart(fig_pie, use_container_width=True)

    # Delivered cumulative
    fig_del = go.Figure()
    expected_rate = p.demand_rate_m3h
    fig_del.add_trace(go.Scatter(
        x=ts["time_d"], y=ts["delivered_cum_m3"],
        mode="lines", name="Entregado acum.",
        line=dict(color=C["teal"], width=2)
    ))
    fig_del.add_trace(go.Scatter(
        x=ts["time_d"],
        y=[expected_rate * t * 24 for t in ts["time_d"]],
        mode="lines", name="Demanda acum.",
        line=dict(color=C["red"], width=1, dash="dash")
    ))
    fig_del.update_layout(title="Entregas acumuladas vs demanda acumulada (m³)",
                          xaxis_title="Tiempo (días)", yaxis_title="m³",
                          **PLOTLY_LAYOUT)
    st.plotly_chart(fig_del, use_container_width=True)

# ── TAB 2: FLOTA ───────────────────────────────────────────────────────────────
with tabs[1]:
    trucks_df = dfs["trucks"]

    col_a, col_b = st.columns(2)
    with col_a:
        fig_trips = px.bar(
            trucks_df, x="truck_id", y="trips",
            color="utilization_pct",
            color_continuous_scale=[[0, "#21262d"], [0.5, C["blue"]], [1, C["teal"]]],
            labels={"truck_id": "Camión", "trips": "Viajes", "utilization_pct": "Util. (%)"},
            title="Viajes completados por camión",
        )
        fig_trips.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_trips, use_container_width=True)

    with col_b:
        fig_util = px.histogram(
            trucks_df, x="utilization_pct", nbins=15,
            color_discrete_sequence=[C["blue"]],
            labels={"utilization_pct": "Utilización (%)"},
            title="Distribución de utilización de camiones",
        )
        fig_util.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_util, use_container_width=True)

    # Tiempo desglosado por camión
    truck_melt = trucks_df.melt(
        id_vars="truck_id",
        value_vars=["fill_wait_h", "fill_time_h", "travel_out_h",
                    "unload_wait_h", "unload_time_h", "travel_ret_h"],
        var_name="actividad", value_name="horas"
    )
    label_map = {
        "fill_wait_h": "Espera llenado", "fill_time_h": "Llenado",
        "travel_out_h": "Viaje ida", "unload_wait_h": "Espera descarga",
        "unload_time_h": "Descarga", "travel_ret_h": "Retorno",
    }
    color_map = {
        "Espera llenado": C["red"], "Llenado": C["teal"],
        "Viaje ida": C["blue"], "Espera descarga": "#d04060",
        "Descarga": C["amber"], "Retorno": C["purple"],
    }
    truck_melt["actividad"] = truck_melt["actividad"].map(label_map)
    fig_stack = px.bar(
        truck_melt, x="truck_id", y="horas", color="actividad",
        color_discrete_map=color_map,
        labels={"truck_id": "Camión", "horas": "Horas", "actividad": "Actividad"},
        title="Desglose de tiempo por camión (horas totales simuladas)",
        barmode="stack",
    )
    fig_stack.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_stack, use_container_width=True)

    # Tabla resumen
    st.markdown("#### Tabla de métricas por camión")
    st.dataframe(trucks_df.style.format({
        "volume_m3": "{:,.0f}", "fill_wait_h": "{:.2f}", "unload_wait_h": "{:.2f}",
        "utilization_pct": "{:.1f}%",
    }), use_container_width=True, height=250)

# ── TAB 3: SERIES TEMPORALES ──────────────────────────────────────────────────
with tabs[2]:
    fig_queues = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=("Cola en punto de llenado (camiones)",
                        "Cola en punto de descarga (camiones)")
    )
    fig_queues.add_trace(go.Scatter(
        x=ts["time_d"], y=ts["fill_queue"],
        line=dict(color=C["teal"], width=1.2), name="Cola llenado",
        fill="tozeroy", fillcolor="rgba(38,166,65,0.08)"
    ), row=1, col=1)
    fig_queues.add_trace(go.Scatter(
        x=ts["time_d"], y=ts["unload_queue"],
        line=dict(color=C["amber"], width=1.2), name="Cola descarga",
        fill="tozeroy", fillcolor="rgba(210,153,34,0.08)"
    ), row=2, col=1)
    fig_queues.update_layout(xaxis2_title="Tiempo (días)", **PLOTLY_LAYOUT)
    st.plotly_chart(fig_queues, use_container_width=True)

    # Histograma de tiempos de espera
    q_df = dfs["queues"]
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fill_waits = q_df[q_df["location"] == "fill"]["wait_h"]
        fig_wf = px.histogram(fill_waits, nbins=40,
                              color_discrete_sequence=[C["teal"]],
                              title="Distribución tiempo de espera — Llenado (h)",
                              labels={"value": "Espera (h)", "count": "Frecuencia"})
        fig_wf.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_wf, use_container_width=True)
    with col_h2:
        unload_waits = q_df[q_df["location"] == "unload"]["wait_h"]
        fig_wu = px.histogram(unload_waits, nbins=40,
                              color_discrete_sequence=[C["amber"]],
                              title="Distribución tiempo de espera — Descarga (h)",
                              labels={"value": "Espera (h)", "count": "Frecuencia"})
        fig_wu.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_wu, use_container_width=True)

    # Estadísticas descriptivas tiempos de espera
    if len(q_df) > 0:
        st.markdown("#### Estadísticas de tiempos de espera en cola")
        wait_stats = q_df.groupby("location")["wait_h"].agg(
            N="count", Media="mean", Mediana="median",
            P95=lambda x: np.percentile(x, 95),
            Máximo="max"
        ).round(3)
        wait_stats.index = ["Llenado" if i == "fill" else "Descarga" for i in wait_stats.index]
        st.dataframe(wait_stats, use_container_width=True)

    # Distribución muestral de tiempos de viaje
    st.markdown("#### Distribución muestral del tiempo de viaje configurado")
    rng_demo = np.random.default_rng(42)
    from sim.engine import TravelTimeDistribution
    td = TravelTimeDistribution(p.travel_outbound_mean_h, p.travel_cv, p.travel_dist, rng_demo)
    samples = [td.sample() for _ in range(2000)]
    fig_td = go.Figure()
    fig_td.add_trace(go.Histogram(x=samples, nbinsx=50,
                                  marker_color=C["blue"], opacity=0.7,
                                  name=f"{dist_label}"))
    fig_td.add_vline(x=np.mean(samples), line_dash="dash", line_color=C["amber"],
                    annotation_text=f"μ={np.mean(samples):.2f}h")
    fig_td.update_layout(title=f"Muestra de 2000 tiempos de viaje — {dist_label}",
                         xaxis_title="Tiempo (h)", yaxis_title="Frecuencia",
                         **PLOTLY_LAYOUT)
    st.plotly_chart(fig_td, use_container_width=True)

# ── TAB 4: RÉPLICAS ───────────────────────────────────────────────────────────
with tabs[3]:
    if len(all_results) < 2:
        st.info("Ejecuta con **2 o más réplicas** para ver el análisis estadístico comparativo.")
    else:
        st.markdown("### Intervalos de confianza al 95% — KPIs entre réplicas")
        agg = aggregate_replicas(all_results)
        st.dataframe(agg.style.format({
            "Media": "{:.3f}", "Std": "{:.3f}",
            "IC95_inf": "{:.3f}", "IC95_sup": "{:.3f}"
        }), use_container_width=True)

        st.markdown("### Inventario por réplica")
        fig_rep = go.Figure()
        colors_rep = px.colors.qualitative.Plotly
        for i, res in enumerate(all_results):
            ts_r = res.to_dataframes()["timeseries"]
            fig_rep.add_trace(go.Scatter(
                x=ts_r["time_d"], y=ts_r["inventory_m3"],
                mode="lines", name=f"Réplica {i+1}",
                line=dict(color=colors_rep[i % len(colors_rep)], width=1),
                opacity=0.7
            ))
        fig_rep.update_layout(title="Inventario en destino — todas las réplicas",
                              xaxis_title="Tiempo (días)", yaxis_title="Inventario (m³)",
                              **PLOTLY_LAYOUT)
        st.plotly_chart(fig_rep, use_container_width=True)

        # Box plots KPIs clave
        kpi_keys = ["total_trips", "throughput_m3h", "truck_utilization_pct",
                    "avg_fill_wait_h", "avg_unload_wait_h", "stockout_events"]
        kpi_vals = {k: [res.kpi[k] for res in all_results] for k in kpi_keys}
        kpi_labels = {
            "total_trips": "Viajes totales",
            "throughput_m3h": "Throughput (m³/h)",
            "truck_utilization_pct": "Util. flota (%)",
            "avg_fill_wait_h": "Espera llenado (h)",
            "avg_unload_wait_h": "Espera descarga (h)",
            "stockout_events": "Stockouts",
        }
        cols_bp = st.columns(3)
        for idx, key in enumerate(kpi_keys):
            fig_bp = go.Figure(go.Box(
                y=kpi_vals[key], name=kpi_labels[key],
                marker_color=list(C.values())[idx % 6],
                boxmean="sd"
            ))
            fig_bp.update_layout(title=kpi_labels[key], showlegend=False,
                                 margin=dict(l=20, r=20, t=40, b=20),
                                 **{k: v for k, v in PLOTLY_LAYOUT.items()
                                    if k not in ("xaxis", "yaxis", "margin")})
            cols_bp[idx % 3].plotly_chart(fig_bp, use_container_width=True)

# ── TAB 5: EXPORTAR ───────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### Exportar resultados a CSV / Excel")

    dfs_export = r.to_dataframes()

    col_e1, col_e2 = st.columns(2)

    # CSV individual por dataset
    for name, label in [
        ("timeseries", "Serie temporal (inventario, colas)"),
        ("deliveries", "Log de entregas"),
        ("trucks", "Métricas por camión"),
        ("queues", "Tiempos de espera en cola"),
        ("stockouts", "Eventos de desabasto"),
    ]:
        df_e = dfs_export[name]
        if df_e.empty:
            continue
        csv_bytes = df_e.to_csv(index=False).encode("utf-8")
        col_e1.download_button(
            label=f"⬇ {label}",
            data=csv_bytes,
            file_name=f"gnl_sim_{name}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Excel completo
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        for name, df_e in dfs_export.items():
            if not df_e.empty:
                df_e.to_excel(writer, sheet_name=name[:31], index=False)
        # Hoja de KPIs
        kpi_df = pd.DataFrame([r.kpi]).T.reset_index()
        kpi_df.columns = ["KPI", "Valor"]
        kpi_df.to_excel(writer, sheet_name="kpis_resumen", index=False)
        # Hoja de parámetros
        params_dict = {k: str(v) for k, v in vars(p).items()}
        pd.DataFrame(list(params_dict.items()), columns=["Parámetro", "Valor"])\
          .to_excel(writer, sheet_name="parametros", index=False)
        # Réplicas agregadas (si hay más de 1)
        if len(all_results) > 1:
            aggregate_replicas(all_results).to_excel(
                writer, sheet_name="replicas_agregado", index=False)

    col_e2.download_button(
        label="⬇ Exportar TODO como Excel (.xlsx)",
        data=excel_buf.getvalue(),
        file_name="gnl_sim_completo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("#### Vista previa — Serie temporal")
    st.dataframe(dfs_export["timeseries"].head(50), use_container_width=True)

    st.markdown("#### Vista previa — KPIs resumen")
    kpi_preview = pd.DataFrame([r.kpi]).T.reset_index()
    kpi_preview.columns = ["KPI", "Valor"]
    st.dataframe(kpi_preview, use_container_width=True)
