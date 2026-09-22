"""Streamlit dashboard for the Neftecode MAS.

The dashboard is deliberately tied to CycleResult / Tick objects produced by the
same agent pipeline as the CLI. No business logic is duplicated in the UI.

Run:
    PYTHONPATH=src streamlit run src/oilhack/dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from oilhack.agents import impact_agent
from oilhack.config import CONTROLLABLE_PARAMS, SULFUR_LIMIT_MG_KG
from oilhack.dashboard.demo_stubs import examples as demo_examples
from oilhack.harness import runner
from oilhack.harness.scenarios import SCENARIOS, load_environment

st.set_page_config(
    page_title="Нефтекод · МАС дизельного топлива",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root{
  --bg:#f7f9fc; --panel:#ffffff; --line:#d9e1ea;
  --text:#1f2937; --muted:#64748b; --blue:#0b5cad;
  --green:#168052; --amber:#b77800; --red:#c92a2a;
}
.stApp{background:var(--bg); color:var(--text);}
header[data-testid="stHeader"]{background:transparent;}
section[data-testid="stSidebar"]{background:#eef3f8; border-right:1px solid var(--line);}
.block-container{padding-top:1.3rem; max-width:1600px;}
h1,h2,h3,h4{color:var(--text)!important; letter-spacing:-.3px;}
.small-muted{color:var(--muted); font-size:.86rem;}
.hero{padding:18px 22px; background:linear-gradient(135deg,#0b5cad,#2d7ed0); color:white;
      border-radius:16px; margin-bottom:14px; box-shadow:0 8px 28px rgba(11,92,173,.18);}
.hero h1{color:white!important; margin:0; font-size:2rem;}
.hero p{margin:.35rem 0 0; opacity:.9;}
.card{background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:16px 18px; height:100%;}
.banner{background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 18px;
        display:flex; align-items:center; gap:16px; margin-bottom:12px;}
.banner .mark{font-size:2rem; font-weight:800;}
.banner .title{font-size:1.2rem; font-weight:800; margin:0;}
.banner .sub{color:var(--muted); margin:.25rem 0 0;}
.step{background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:11px 12px; min-height:110px;}
.step .n{font-size:.72rem; text-transform:uppercase; color:var(--muted); letter-spacing:.1em;}
.step .name{font-weight:800; margin:.15rem 0 .25rem;}
.step .val{font-size:.96rem;}
.step .note{font-size:.79rem; color:var(--muted); margin-top:.25rem;}
.step.ok{border-left:4px solid var(--green);}.step.warn{border-left:4px solid var(--amber);}
.step.bad{border-left:4px solid var(--red);}.step.info{border-left:4px solid var(--blue);}
.metric-card{background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px 14px;}
.metric-label{color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em;}
.metric-value{font-size:1.5rem; font-weight:800; margin-top:.2rem;}
.badge{display:inline-block; border-radius:999px; padding:.22rem .58rem; font-size:.76rem; font-weight:700;}
.badge.ok{background:#e9f7ef; color:#166b41;}.badge.warn{background:#fff5df; color:#8a5b00;}
.badge.bad{background:#fdeaea; color:#a82424;}.badge.info{background:#e8f1fb; color:#0b5cad;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

STATUS = {
    "no_action": ("✓", "Менять ничего не нужно", "ok",
                   "Режим в допуске, и система не нашла материального улучшения."),
    "action": ("→", "Есть рекомендация оператору", "info",
                "Найден допустимый сценарий изменения параметров."),
    "no_feasible_solution": ("!", "Безопасного решения нет", "bad",
                              "Все кандидаты не прошли жёсткие ограничения или не могут быть проверены."),
    "insufficient_data": ("?", "Данных недостаточно", "warn",
                           "Цикл остановлен: входы неполные, устаревшие или аномальные."),
}


def _status_badge(status: str) -> str:
    _, title, tone, _ = STATUS.get(status, ("•", status, "info", ""))
    return f'<span class="badge {tone}">{title}</span>'


def banner(rec) -> None:
    mark, title, tone, plain = STATUS.get(rec.status, ("•", rec.status, "info", ""))
    color = {"ok": "#168052", "info": "#0b5cad", "bad": "#c92a2a", "warn": "#b77800"}[tone]
    st.markdown(
        f"""<div class="banner" style="border-left:6px solid {color}">
        <div class="mark" style="color:{color}">{mark}</div>
        <div>
          <div class="title">{title}</div>
          <div class="sub">{plain} · момент <b>{rec.timestamp}</b></div>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )


def metric_cards(result) -> None:
    rec = result.recommendation
    dr = result.data_report
    q = result.quality
    r = result.reliability
    sf = result.sulfur_forecast
    values = [
        ("Статус", STATUS.get(rec.status, ("", rec.status, "info", ""))[1], ""),
        ("Сера сейчас", "н/д" if q is None or q.sulfur_ppm is None else f"{q.sulfur_ppm:.2f} мг/кг",
         "ЛИМС / ПАК" if q else ""),
        ("Forecast upper", "н/д" if sf is None else f"{sf.upper:.2f} мг/кг",
         "жёсткий gate: ≤ 10"),
        ("Тяжесть режима", "н/д" if r is None else f"{r.severity_index:.2f}",
         "0 = привычный режим"),
        ("Полнота КИП", f"{dr.tag_coverage:.0%}", f"пропусков: {len(dr.missing_tags)}"),
        ("Свежесть", _freshness_text(dr), "ЛИМС / ПАК"),
    ]
    cols = st.columns(len(values), gap="small")
    for col, (label, value, note) in zip(cols, values):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div><div class="small-muted">{note}</div></div>',
                unsafe_allow_html=True,
            )


def _freshness_text(dr) -> str:
    ages = [v for v in list(dr.lims_age_hours.values()) + list(dr.pak_age_hours.values()) if v is not None]
    if not ages:
        return "нет"
    age = min(ages)
    if age < 1:
        return f"{age:.0f} ч"
    if age < 24:
        return f"{age:.1f} ч"
    return f"{age / 24:.1f} д"


def _step(number: int, name: str, value: str, note: str, tone: str) -> str:
    return (
        f'<div class="step {tone}"><div class="n">Шаг {number}</div>'
        f'<div class="name">{name}</div><div class="val">{value}</div>'
        f'<div class="note">{note}</div></div>'
    )


def agent_chain(result) -> None:
    st.markdown("### Цепочка решения")
    st.caption("Каждый блок соответствует реальному выходу агента из одного CycleResult.")
    dr, q, sf, r, gate, opt, blend = (
        result.data_report, result.quality, result.sulfur_forecast,
        result.reliability, result.constraint_report, result.optimization, result.blending,
    )
    cols = st.columns(7, gap="small")
    with cols[0]:
        st.markdown(_step(1, "Data Agent", f"{dr.tag_coverage:.0%}",
                          f"аномалий {len(dr.anomalous_tags)} · пропусков {len(dr.missing_tags)}",
                          "ok" if dr.is_usable else "bad"), unsafe_allow_html=True)
    with cols[1]:
        if q is None:
            st.markdown(_step(2, "Quality Agent", "не запускался", "данные непригодны", "warn"), unsafe_allow_html=True)
        else:
            tone = "bad" if q.sulfur_ppm is not None and q.sulfur_ppm > SULFUR_LIMIT_MG_KG else "ok"
            st.markdown(_step(2, "Quality Agent",
                              "н/д" if q.sulfur_ppm is None else f"{q.sulfur_ppm:.2f} мг/кг",
                              f"{q.sulfur_source} · conf {q.confidence:.0%}", tone),
                        unsafe_allow_html=True)
    with cols[2]:
        if sf is None:
            st.markdown(_step(3, "Sulfur Forecast", "нет", "surrogate недоступен", "warn"), unsafe_allow_html=True)
        else:
            st.markdown(_step(3, "Sulfur Forecast", f"upper {sf.upper:.2f}",
                              f"H={sf.horizon_minutes} мин · MAE {sf.calibration_mae:.2f}",
                              "ok" if sf.upper <= SULFUR_LIMIT_MG_KG else "bad"), unsafe_allow_html=True)
    with cols[3]:
        if r is None:
            st.markdown(_step(4, "Reliability", "—", "не запускался", "warn"), unsafe_allow_html=True)
        else:
            tone = "bad" if r.severity_index >= 0.8 else "warn" if r.severity_index >= 0.6 else "ok"
            st.markdown(_step(4, "Reliability", f"{r.severity_index:.2f}", r.severity_class, tone), unsafe_allow_html=True)
    with cols[4]:
        if gate is None:
            gate_text, gate_note, tone = "результат в optimizer", "финальная проверка в оркестраторе", "info"
        else:
            gate_text = "PASS" if gate.feasible else "REJECT"
            gate_note = "; ".join(gate.violations[:2]) if gate.violations else "hard constraints выполнены"
            tone = "ok" if gate.feasible else "bad"
        st.markdown(_step(5, "Safety Gate", gate_text, gate_note, tone), unsafe_allow_html=True)
    with cols[5]:
        if opt is None:
            st.markdown(_step(6, "Optimization", "—", "не запускался", "warn"), unsafe_allow_html=True)
        else:
            st.markdown(_step(6, "Optimization", str(len(opt.candidates)),
                              f"отклонено {opt.infeasible_count}",
                              "ok" if opt.candidates else "bad"), unsafe_allow_html=True)
    with cols[6]:
        st.markdown(_step(7, "Orchestrator", STATUS.get(result.recommendation.status, ("", result.recommendation.status, "info", ""))[1],
                          "финальный ответ оператору", "ok" if result.recommendation.status == "no_action" else "info"),
                    unsafe_allow_html=True)

    if blend is not None and not blend.available:
        st.caption(f"Blending Agent: {blend.reason}")


def sulfur_chart(series: pd.DataFrame, current_ts=None) -> go.Figure:
    fig = go.Figure()
    if "upper" in series.columns:
        fig.add_trace(go.Scatter(x=series["timestamp"], y=series["upper"], name="Forecast upper",
                                 mode="lines", line=dict(color="#c92a2a", dash="dot", width=2)))
    fig.add_trace(go.Scatter(x=series["timestamp"], y=series["sulfur"], name="Сера",
                             mode="lines+markers", line=dict(color="#0b5cad", width=2)))
    fig.add_hline(y=SULFUR_LIMIT_MG_KG, line=dict(color="#c92a2a", dash="dash"),
                  annotation_text="предел 10 мг/кг", annotation_position="top left")
    if current_ts is not None:
        fig.add_vline(x=current_ts, line=dict(color="#64748b", dash="dot"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      yaxis_title="мг/кг", hovermode="x unified")
    return fig


def severity_chart(series: pd.DataFrame, current_ts=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["timestamp"], y=series["severity"],
                             mode="lines+markers", name="Тяжесть", line=dict(color="#64748b", width=2)))
    fig.add_hline(y=0.6, line=dict(color="#b77800", dash="dash"), annotation_text="высокая тяжесть")
    fig.add_hline(y=0.8, line=dict(color="#c92a2a", dash="dash"), annotation_text="критическая тяжесть")
    if current_ts is not None:
        fig.add_vline(x=current_ts, line=dict(color="#0b5cad", dash="dot"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis=dict(range=[0, 1], title="индекс"))
    return fig


def status_timeline(series: pd.DataFrame, current_ts=None) -> go.Figure:
    order = {"insufficient_data": 0, "no_feasible_solution": 1, "no_action": 2, "action": 3}
    labels = {v: k for k, v in order.items()}
    y = series["status"].map(order)
    fig = go.Figure(go.Scatter(
        x=series["timestamp"], y=y, mode="lines+markers", name="Статус",
        text=series["status"].map({k: STATUS[k][1] for k in STATUS}),
        hovertemplate="%{x}<br>%{text}<extra></extra>",
        line=dict(color="#0b5cad", width=2), marker=dict(size=7),
    ))
    fig.update_yaxes(tickmode="array", tickvals=list(order.values()), ticktext=[STATUS[labels[i]][1] for i in range(4)])
    if current_ts is not None:
        fig.add_vline(x=current_ts, line=dict(color="#64748b", dash="dot"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def recommendation_card(rec) -> None:
    st.markdown("### Рекомендация оператору")
    rows = [
        ("Состояние", rec.state_summary),
        ("Проблема / риск", rec.problem),
        ("Предлагаемое действие", rec.action),
        ("Ожидаемый эффект", rec.expected_effect),
        ("Проверка ограничений", rec.constraints_checked),
        ("Уверенность", rec.confidence),
        ("Объяснение", rec.explanation),
    ]
    for key, value in rows:
        with st.container(border=True):
            c1, c2 = st.columns([1, 3])
            c1.markdown(f"**{key}**")
            c2.write(value)


def candidate_table(result) -> pd.DataFrame:
    opt = result.optimization
    if opt is None:
        return pd.DataFrame()
    rows = []
    for idx, c in enumerate(opt.candidates, start=1):
        changes = []
        for tag, new_value in c.deltas.items():
            param = CONTROLLABLE_PARAMS.get(tag)
            label = param.description if param else tag
            current = None
            if result.timestamp is not None:
                current = None
            changes.append(f"{label}: {new_value:.2f}")
        rows.append({
            "№": idx,
            "Сценарий": c.label,
            "Изменения": "; ".join(changes) if changes else "Без изменений",
            "Сера p50": None if c.predicted_sulfur_ppm is None else round(c.predicted_sulfur_ppm, 2),
            "Сера upper": None if c.predicted_sulfur_upper_ppm is None else round(c.predicted_sulfur_upper_ppm, 2),
            "Тяжесть": round(c.reliability_severity, 3),
            "Выпуск-прокси": round(c.throughput_proxy, 3),
            "Энерго-прокси": round(c.energy_proxy, 3),
            "PASS": "✓" if c.feasible else "✗",
        })
    return pd.DataFrame(rows)


def frontier_chart(result) -> go.Figure | None:
    opt = result.optimization
    if opt is None or not opt.candidates:
        return None
    df = pd.DataFrame([{
        "label": c.label,
        "severity": c.reliability_severity,
        "sulfur": c.predicted_sulfur_upper_ppm,
        "throughput": c.throughput_proxy,
    } for c in opt.candidates])
    df = df.dropna(subset=["severity", "sulfur"])
    if df.empty:
        return None
    fig = px.scatter(df, x="severity", y="sulfur", size="throughput", hover_name="label",
                     title="Допустимые сценарии: тяжесть × консервативная сера")
    fig.add_hline(y=SULFUR_LIMIT_MG_KG, line_dash="dash", line_color="#c92a2a")
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="Тяжесть режима", yaxis_title="Сера upper, мг/кг")
    return fig


def current_process_table(tick) -> pd.DataFrame:
    rows = []
    for tag, value in tick.avt_row.items():
        if tag in CONTROLLABLE_PARAMS and CONTROLLABLE_PARAMS[tag].dataset == "avt":
            p = CONTROLLABLE_PARAMS[tag]
            rows.append({"Установка": "АВТ", "Тег": tag, "Параметр": p.description,
                         "Значение": value, "Ед.": p.unit, "Модельный диапазон": f"{p.lo:.2f} … {p.hi:.2f}"})
    for tag, value in tick.u242000_row.items():
        if tag in CONTROLLABLE_PARAMS and CONTROLLABLE_PARAMS[tag].dataset == "242000":
            p = CONTROLLABLE_PARAMS[tag]
            rows.append({"Установка": "24-2000", "Тег": tag, "Параметр": p.description,
                         "Значение": value, "Ед.": p.unit, "Модельный диапазон": f"{p.lo:.2f} … {p.hi:.2f}"})
    return pd.DataFrame(rows).sort_values(["Установка", "Тег"])


def quality_table(result) -> pd.DataFrame:
    if result.quality is None:
        return pd.DataFrame()
    return pd.DataFrame([
        {"Показатель": k, "Прогноз / оценка": v}
        for k, v in sorted(result.quality.predictions.items())
    ])


def detailed_agent_table(result) -> pd.DataFrame:
    dr = result.data_report
    rows = [
        {"Агент": "Data Agent", "Статус": "PASS" if dr.is_usable else "STOP",
         "Ключевое число": f"{dr.tag_coverage:.1%}", "Комментарий": "; ".join(dr.reasons) or "данные пригодны"},
    ]
    if result.quality is not None:
        rows.append({"Агент": "Quality Agent", "Статус": "PASS",
                     "Ключевое число": "н/д" if result.quality.sulfur_ppm is None else f"{result.quality.sulfur_ppm:.2f} мг/кг",
                     "Комментарий": f"источник: {result.quality.sulfur_source}; conf={result.quality.confidence:.0%}"})
    if result.sulfur_forecast is not None:
        sf = result.sulfur_forecast
        rows.append({"Агент": "Sulfur Forecast Agent", "Статус": "PASS" if sf.upper <= SULFUR_LIMIT_MG_KG else "WARN",
                     "Ключевое число": f"{sf.upper:.2f} мг/кг",
                     "Комментарий": f"H={sf.horizon_minutes} мин; MAE={sf.calibration_mae:.2f}; samples={sf.training_samples}"})
    if result.reliability is not None:
        rows.append({"Агент": "Reliability Agent", "Статус": result.reliability.severity_class,
                     "Ключевое число": f"{result.reliability.severity_index:.2f}",
                     "Комментарий": "; ".join(result.reliability.risk_factors[:3])})
    if result.constraint_report is not None:
        rows.append({"Агент": "Constraint Agent", "Статус": "PASS" if result.constraint_report.feasible else "REJECT",
                     "Ключевое число": str(len(result.constraint_report.violations)),
                     "Комментарий": "; ".join(result.constraint_report.violations) or "hard constraints выполнены"})
    if result.optimization is not None:
        rows.append({"Агент": "Optimization Agent", "Статус": "PASS" if result.optimization.candidates else "NO-SOLUTION",
                     "Ключевое число": str(len(result.optimization.candidates)),
                     "Комментарий": f"отброшено {result.optimization.infeasible_count}"})
    if result.blending is not None:
        rows.append({"Агент": "Blending Agent", "Статус": "AVAILABLE" if result.blending.available else "UNAVAILABLE",
                     "Ключевое число": "—",
                     "Комментарий": result.blending.reason})
    return pd.DataFrame(rows)


def details(result) -> None:
    tabs = st.tabs(["Данные", "Качество", "Оптимизация", "Трасса агентов"])
    with tabs[0]:
        dr = result.data_report
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Data Agent**")
            st.write(f"Полнота КИП: {dr.tag_coverage:.1%}")
            st.write(f"Пропущено: {len(dr.missing_tags)} · аномалий: {len(dr.anomalous_tags)} · dead tags: {len(dr.dead_tags)}")
            if dr.missing_tags:
                st.caption("Молчат: " + ", ".join(dr.missing_tags[:30]))
            if dr.anomalous_tags:
                st.caption("Аномалии: " + ", ".join(dr.anomalous_tags[:30]))
        with c2:
            st.markdown("**Возраст источников**")
            age_rows = []
            for source, ages in (("ЛИМС", dr.lims_age_hours), ("ПАК", dr.pak_age_hours)):
                for tag, age in ages.items():
                    age_rows.append({"Источник": source, "Сигнал": tag,
                                     "Возраст, ч": None if age is None else round(age, 2)})
            st.dataframe(pd.DataFrame(age_rows), hide_index=True, use_container_width=True)

    with tabs[1]:
        qdf = quality_table(result)
        if not qdf.empty:
            st.dataframe(qdf, hide_index=True, use_container_width=True)
        if result.quality is not None:
            st.caption(f"Сера: {result.quality.sulfur_ppm} мг/кг · источник {result.quality.sulfur_source} · возраст {result.quality.sulfur_age_hours} ч")
        if result.sulfur_forecast is not None:
            sf = result.sulfur_forecast
            st.write(f"Forecast: p50={sf.p50:.2f}, upper={sf.upper:.2f}, горизонт={sf.horizon_minutes} мин")
            st.write(f"Калибровка: MAE={sf.calibration_mae:.2f}, RMSE={sf.calibration_rmse:.2f}, q90={sf.calibration_q90:.2f}")

    with tabs[2]:
        ctab = candidate_table(result)
        if ctab.empty:
            st.info("Допустимых сценариев нет.")
        else:
            st.dataframe(ctab, hide_index=True, use_container_width=True)
            fig = frontier_chart(result)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            st.download_button("Скачать таблицу кандидатов",
                               ctab.to_csv(index=False).encode("utf-8-sig"),
                               file_name="candidates.csv", mime="text/csv")
        if result.constraint_report is not None:
            gate = result.constraint_report
            if gate.feasible:
                st.success("Safety Gate: PASS")
            else:
                st.error("Safety Gate: REJECT")
            if gate.violations:
                for item in gate.violations:
                    st.caption(f"• {item}")

    with tabs[3]:
        st.dataframe(detailed_agent_table(result), hide_index=True, use_container_width=True)
        with st.expander("Полная трасса принятия решения"):
            st.code("\n".join(result.recommendation.react_trace), language=None)
            st.download_button("Скачать trace",
                               "\n".join(result.recommendation.react_trace).encode("utf-8"),
                               file_name="agent_trace.txt", mime="text/plain")


def hitl(rec, key: str, later=None) -> None:
    if rec.status != "action":
        return
    st.markdown("#### Human-in-the-loop")
    st.caption("Система только предлагает действие; оператор подтверждает или отклоняет его.")
    store = st.session_state.setdefault("impacts", {})
    done = store.get(key)
    if done is None:
        c1, c2, c3 = st.columns(3)
        if c1.button("Подтвердить", key=f"ok_{key}", use_container_width=True):
            store[key] = impact_agent.apply(rec, "confirmed"); st.rerun()
        if c2.button("Отклонить", key=f"no_{key}", use_container_width=True):
            store[key] = impact_agent.apply(rec, "rejected"); st.rerun()
        if c3.button("Другой вариант", key=f"alt_{key}", use_container_width=True):
            store[key] = impact_agent.apply(rec, "alternative_requested"); st.rerun()
    else:
        st.success(f"Решение оператора: {done.decision}")
        if later is not None:
            st.write("Мониторинг отклика: " + impact_agent.observe_response(done, later))


@st.cache_resource(show_spinner="Читаю телеметрию, ЛИМС и ПАК…")
def _environment():
    return load_environment()


@st.cache_data(show_spinner="Прогоняю сценарий через агентов…")
def _ticks(scenario_key: str):
    return list(runner.run_scenario(_environment(), SCENARIOS[scenario_key]))


def _real_series(ticks) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": [t.timestamp for t in ticks],
        "sulfur": [t.recommendation.sulfur_ppm for t in ticks],
        "upper": [t.recommendation.predicted_sulfur_upper_ppm for t in ticks],
        "severity": [t.recommendation.reliability_severity for t in ticks],
        "status": [t.recommendation.status for t in ticks],
    })


def main() -> None:
    st.markdown(
        '<div class="hero"><h1>🛢️ МАС управления производством дизельного топлива</h1>'
        '<p>АВТ → гидроочистка 24-2000 → прогноз качества → Safety Gate → оптимизация → оператор</p></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Режим")
        mode = st.radio("Источник", ["Реальные данные", "Демо"], label_visibility="collapsed")
        st.divider()
        if mode == "Реальные данные":
            skey = st.selectbox("Сценарий", list(SCENARIOS),
                                format_func=lambda k: SCENARIOS[k].title, label_visibility="visible")
            scenario = SCENARIOS[skey]
            st.caption(f"{scenario.start} — {scenario.end}")
            if scenario.inject_degraded:
                st.warning("Сценарий содержит намеренную порчу данных.")
        else:
            ex = demo_examples()
            dkey = st.radio("Пример", list(ex), format_func=lambda k: ex[k].title, label_visibility="visible")
        st.divider()
        st.markdown("### Правила системы")
        st.caption("Сера: жёсткий предел 10 мг/кг")
        st.caption("ЛИМС → ПАК для факта качества")
        st.caption("Модельный диапазон ≠ промышленный предел")
        st.caption("Экономический proxy не может обойти Safety Gate")

    if mode == "Реальные данные":
        ticks = _ticks(skey)
        if not ticks:
            st.error("В выбранном сценарии нет доступных временных точек.")
            return
        idx = st.slider("Момент времени", 0, len(ticks) - 1, min(len(ticks) - 1, 6), key=f"idx_{skey}")
        tick = ticks[idx]
        result = tick.result
        series = _real_series(ticks)
        current_ts = tick.timestamp
        later = ticks[idx + 1].recommendation if idx + 1 < len(ticks) else None
        hitl_key = f"{skey}:{idx}"
    else:
        chosen = demo_examples()[dkey]
        result = chosen.result
        series = chosen.series.copy()
        series["upper"] = series["sulfur"]
        series["status"] = "no_action" if chosen.key == "calm" else "no_feasible_solution" if chosen.key == "sulfur" else "insufficient_data"
        tick = None
        current_ts = chosen.result.timestamp
        later = None
        hitl_key = f"demo:{dkey}"

    rec = result.recommendation
    banner(rec)
    metric_cards(result)
    st.write("")
    agent_chain(result)

    overview, process, recommendation, analytics = st.tabs([
        "Обзор", "Технологические данные", "Рекомендация", "Аналитика агентов"
    ])

    with overview:
        c1, c2 = st.columns([1.55, 1], gap="large")
        with c1:
            st.markdown("### Динамика качества")
            st.plotly_chart(sulfur_chart(series, current_ts), use_container_width=True)
        with c2:
            st.markdown("### Решение")
            st.markdown(_status_badge(rec.status), unsafe_allow_html=True)
            st.write(rec.problem)
            st.write(rec.expected_effect)
            if rec.predicted_sulfur_upper_ppm is not None:
                margin = SULFUR_LIMIT_MG_KG - rec.predicted_sulfur_upper_ppm
                if margin >= 0:
                    st.success(f"Safety margin по сере: +{margin:.2f} мг/кг")
                else:
                    st.error(f"Safety deficit по сере: {abs(margin):.2f} мг/кг")
            st.plotly_chart(severity_chart(series, current_ts), use_container_width=True)
        st.markdown("### Статус цикла во времени")
        st.plotly_chart(status_timeline(series, current_ts), use_container_width=True)

    with process:
        if tick is None:
            st.info("Демо-режим содержит только агрегированные данные; перейдите в «Реальные данные» для снимка телеметрии.")
        else:
            st.markdown("### Текущие управляемые параметры")
            ptab = current_process_table(tick)
            st.dataframe(ptab, hide_index=True, use_container_width=True)
            st.markdown("### Все текущие значения выбранного тика")
            a, b = st.columns(2)
            with a:
                avt_df = pd.DataFrame({"Тег": list(tick.avt_row), "Значение": list(tick.avt_row.values())})
                st.dataframe(avt_df, hide_index=True, use_container_width=True, height=420)
            with b:
                hyd_df = pd.DataFrame({"Тег": list(tick.u242000_row), "Значение": list(tick.u242000_row.values())})
                st.dataframe(hyd_df, hide_index=True, use_container_width=True, height=420)

    with recommendation:
        recommendation_card(rec)
        hitl(rec, hitl_key, later)

    with analytics:
        details(result)

    st.divider()
    st.caption(
        "Источник численных результатов — тот же pipeline, что используется в CLI. "
        "UI не пересчитывает технологическую логику отдельно."
    )


if __name__ == "__main__":
    main()
