"""Streamlit-дашборд харнесса. Запуск:

    PYTHONPATH=src streamlit run src/oilhack/dashboard/app.py

Показывает по каждому тику: данные, работу всех агентов, финальную
рекомендацию в 7-блочном формате ТЗ, Парето-сравнение сценариев и
HITL-подтверждение с последующим мониторингом отклика.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oilhack.agents import impact_agent
from oilhack.config import SULFUR_LIMIT_MG_KG
from oilhack.harness import runner
from oilhack.harness.scenarios import SCENARIOS, load_environment

st.set_page_config(page_title="Нефтекод — МАС дашборд", layout="wide")


@st.cache_resource(show_spinner="Загрузка телеметрии, ЛИМС и ПАК...")
def _environment():
    return load_environment()


@st.cache_data(show_spinner="Прогон сценария через агентов...")
def _ticks(scenario_key: str):
    env = _environment()
    scenario = SCENARIOS[scenario_key]
    return list(runner.run_scenario(env, scenario))


STATUS_LABELS = {
    "action": "Действие",
    "no_action": "Без действий",
    "no_feasible_solution": "Нет допустимого решения",
    "insufficient_data": "Недостаточно данных",
}
STATUS_COLORS = {
    "action": "#2563eb",
    "no_action": "#16a34a",
    "no_feasible_solution": "#dc2626",
    "insufficient_data": "#6b7280",
}


def _timeline_chart(ticks, current_idx: int) -> go.Figure:
    ts = [t.timestamp for t in ticks]
    sulfur = [t.recommendation.sulfur_ppm for t in ticks]
    severity = [t.recommendation.reliability_severity for t in ticks]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=sulfur, name="Сера, мг/кг", mode="lines+markers",
                              line=dict(color="#b45309"), yaxis="y1"))
    fig.add_hline(y=SULFUR_LIMIT_MG_KG, line=dict(color="#dc2626", dash="dash"),
                  annotation_text="лимит 10 мг/кг", yref="y1")
    fig.add_trace(go.Scatter(x=ts, y=severity, name="Индекс тяжести режима", mode="lines+markers",
                              line=dict(color="#2563eb"), yaxis="y2"))
    if 0 <= current_idx < len(ts):
        fig.add_vline(x=ts[current_idx], line=dict(color="#111827", dash="dot"))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Сера, мг/кг"),
        yaxis2=dict(title="Тяжесть режима", overlaying="y", side="right", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _pareto_scatter(optimization) -> go.Figure:
    fig = go.Figure()
    if optimization is None:
        return fig
    pts = [optimization.baseline] + optimization.candidates
    fig.add_trace(go.Scatter(
        x=[c.energy_proxy for c in pts],
        y=[c.reliability_severity for c in pts],
        mode="markers+text",
        text=[c.label.split(" (")[0] for c in pts],
        textposition="top center",
        marker=dict(size=12, color=["#111827"] + ["#2563eb"] * len(optimization.candidates)),
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Энергопрокси (изменение расхода/темп., у.е.)",
        yaxis_title="Индекс тяжести режима (ниже — лучше)",
    )
    return fig


def main() -> None:
    st.title("МАС управления производством дизтоплива — харнесс и дашборд")
    st.caption("АВТ → гидроочистка 24-2000 · ReAct-оркестратор (детерминированный) · Human-in-the-loop")

    with st.sidebar:
        st.header("Сценарий демонстрации")
        scenario_key = st.selectbox(
            "Выберите сценарий", options=list(SCENARIOS),
            format_func=lambda k: SCENARIOS[k].title,
        )
        scenario = SCENARIOS[scenario_key]
        st.caption(f"{scenario.start} — {scenario.end}")
        if scenario.inject_degraded:
            st.warning("В этом сценарии данные синтетически испорчены (см. docs/concept.md).")

        if st.session_state.get("_scenario_key") != scenario_key:
            st.session_state["_scenario_key"] = scenario_key
            st.session_state["tick_idx"] = 0
            st.session_state["impacts"] = {}

    ticks = _ticks(scenario_key)
    n = len(ticks)
    st.session_state.setdefault("tick_idx", 0)

    tick_idx = st.slider("Тик", 0, max(n - 1, 0), key="tick_idx")
    tick = ticks[tick_idx]
    rec = tick.recommendation
    result = tick.result

    st.plotly_chart(_timeline_chart(ticks, tick_idx), use_container_width=True)

    badge_color = STATUS_COLORS.get(rec.status, "#6b7280")
    st.markdown(
        f"### Тик {tick.timestamp} &nbsp; "
        f"<span style='background:{badge_color};color:white;padding:2px 10px;"
        f"border-radius:10px;font-size:0.8em'>{STATUS_LABELS.get(rec.status, rec.status)}</span>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.subheader("Рекомендация оператору")
        for label, value in [
            ("Время и состояние", rec.state_summary),
            ("Проблема / риск", rec.problem),
            ("Предлагаемое действие", rec.action),
            ("Ожидаемый эффект", rec.expected_effect),
            ("Проверка ограничений", rec.constraints_checked),
            ("Уверенность", rec.confidence),
            ("Объяснение", rec.explanation),
        ]:
            st.markdown(f"**{label}:** {value}")

        if rec.status == "action":
            st.write("")
            c1, c2, c3 = st.columns(3)
            key_prefix = f"{scenario_key}:{tick_idx}"
            already = st.session_state.get("impacts", {}).get(key_prefix)
            if already is None:
                if c1.button("✅ Подтвердить", key=f"confirm_{key_prefix}"):
                    st.session_state.setdefault("impacts", {})[key_prefix] = impact_agent.apply(rec, "confirmed")
                    st.rerun()
                if c2.button("❌ Отклонить", key=f"reject_{key_prefix}"):
                    st.session_state.setdefault("impacts", {})[key_prefix] = impact_agent.apply(rec, "rejected")
                    st.rerun()
                if c3.button("🔁 Запросить альтернативу", key=f"alt_{key_prefix}"):
                    st.session_state.setdefault("impacts", {})[key_prefix] = impact_agent.apply(
                        rec, "alternative_requested"
                    )
                    st.rerun()
            else:
                st.info(f"Решение оператора: **{already.decision}**")
                later_candidates = [t for t in ticks[tick_idx + 1:] if t.recommendation.reliability_severity is not None]
                if later_candidates:
                    note = impact_agent.observe_response(already, later_candidates[0].recommendation)
                    st.write(f"Мониторинг отклика (на тике {later_candidates[0].timestamp}): {note}")
                else:
                    st.write("Мониторинг отклика: следующих тиков в окне сценария больше нет.")

    with col_b:
        st.subheader("ReAct-трасса оркестратора")
        st.code("\n".join(rec.react_trace), language=None)

    st.divider()
    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.markdown("#### Агент данных")
        dr = result.data_report
        st.metric("Полнота КИП", f"{dr.tag_coverage:.0%}")
        st.write(f"Аномальных тегов: {len(dr.anomalous_tags)}")
        st.write(f"Пропущено тегов: {len(dr.missing_tags)}")
        if dr.missing_tags:
            st.caption(", ".join(dr.missing_tags[:15]) + (" ..." if len(dr.missing_tags) > 15 else ""))
        st.write("Возраст ЛИМС/ПАК:")
        for k, v in {**dr.lims_age_hours, **dr.pak_age_hours}.items():
            st.caption(f"{k}: {'н/д' if v is None else f'{v:.1f} ч'}")

    with d2:
        st.markdown("#### Агент качества")
        q = result.quality
        if q is None:
            st.write("Не выполнялось (нет данных).")
        else:
            st.metric("Сера, мг/кг", f"{q.sulfur_ppm:.2f}" if q.sulfur_ppm is not None else "н/д",
                      help=f"Источник: {q.sulfur_source}")
            st.write(f"Риск спецификации: {'да' if q.spec_risk else 'нет'}")
            st.write(f"Уверенность: {q.confidence:.0%}")
            if q.confidence_reasons:
                st.caption("; ".join(q.confidence_reasons))
            with st.expander("ВАК-прогнозы"):
                st.dataframe(pd.DataFrame(sorted(q.predictions.items()), columns=["Показатель", "Значение"]),
                             hide_index=True, use_container_width=True)

    with d3:
        st.markdown("#### Агент надёжности")
        r = result.reliability
        if r is None:
            st.write("Не выполнялось (нет данных).")
        else:
            st.metric("Индекс тяжести", f"{r.severity_index:.2f}", help=r.severity_class)
            for f in r.risk_factors:
                st.caption(f)

    with d4:
        st.markdown("#### Агент оптимизации")
        opt = result.optimization
        if opt is None:
            st.write("Не выполнялось.")
        elif not opt.candidates:
            st.write(f"Допустимых вариантов нет (отброшено: {opt.infeasible_count}).")
        else:
            st.write(f"Допустимых: {len(opt.candidates)}, отброшено: {opt.infeasible_count}")
            rows = [{
                "Вариант": c.label,
                "Тяжесть": round(c.reliability_severity, 2),
                "Выпуск-прокси": round(c.throughput_proxy, 2),
                "Энерго-прокси": round(c.energy_proxy, 2),
            } for c in opt.candidates[:8]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if result.optimization is not None and result.optimization.candidates:
        st.subheader("Парето-сравнение допустимых вариантов")
        st.plotly_chart(_pareto_scatter(result.optimization), use_container_width=True)


if __name__ == "__main__":
    main()
