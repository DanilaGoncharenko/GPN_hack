"""Дашборд МАС: что система увидела, что решила и почему.

Запуск:
    PYTHONPATH=src streamlit run src/oilhack/dashboard/app.py

Два режима:
  • «Демо» — три готовых примера из `demo_stubs`, открывается мгновенно;
  • «Реальные данные» — прогон харнесса по историческим рядам.
Оба режима рисуются одними и теми же функциями из одного `CycleResult`.
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
from oilhack.dashboard.demo_stubs import examples as demo_examples
from oilhack.harness import runner
from oilhack.harness.scenarios import SCENARIOS, load_environment

st.set_page_config(
    page_title="МАС · производство дизтоплива",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root{
  --bg:#0d1117; --panel:#161b22; --line:#2a313c;
  --text:#e6edf3; --muted:#9aa4b2;
  --ok:#2ea043; --warn:#d29922; --bad:#f85149; --info:#4c8dff;
}
.stApp{background:var(--bg); color:var(--text);}
header[data-testid="stHeader"]{background:transparent;}
section[data-testid="stSidebar"]{background:#11161d; border-right:1px solid var(--line);}
.block-container{padding-top:1.6rem; max-width:1500px;}
h1,h2,h3,h4{letter-spacing:-.4px; color:var(--text)!important;}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{color:var(--text);}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{color:var(--muted)!important;}
.lead{color:var(--muted); font-size:1.02rem; margin:-.4rem 0 1.1rem;}

.banner{border-radius:14px; padding:18px 22px; margin:6px 0 18px;
  border:1px solid var(--line); background:var(--panel); display:flex; gap:18px; align-items:center;}
.banner .mark{font-size:2.1rem; line-height:1;}
.banner .title{font-size:1.45rem; font-weight:800; margin:0;}
.banner .sub{color:var(--muted); margin:.25rem 0 0; font-size:.98rem;}

.step{background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:12px 14px; height:100%;}
.step .n{color:var(--muted); font-size:.72rem; letter-spacing:.12em; text-transform:uppercase;}
.step .name{font-weight:700; margin:.15rem 0 .35rem; font-size:1.02rem;}
.step .val{font-size:.94rem; color:var(--text);}
.step .note{color:var(--muted); font-size:.82rem; margin-top:.3rem;}
.step.ok{border-left:4px solid var(--ok);}
.step.warn{border-left:4px solid var(--warn);}
.step.bad{border-left:4px solid var(--bad);}
.step.info{border-left:4px solid var(--info);}
.step.off{border-left:4px solid #3a414c; opacity:.55;}

.card{background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px 20px;}
.row{display:flex; gap:14px; padding:10px 0; border-bottom:1px solid var(--line);}
.row:last-child{border-bottom:none;}
.row .k{color:var(--muted); min-width:190px; font-size:.9rem; padding-top:2px;}
.row .v{font-size:1rem;}
.big{font-size:1.15rem; font-weight:700;}

.gauge{height:10px; border-radius:6px; background:#22272e; overflow:hidden; margin:.45rem 0 .2rem;}
.gauge > span{display:block; height:100%;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

STATUS = {
    "no_action": ("✓", "Менять ничего не нужно", "ok",
                   "Режим в допуске, и ни один вариант изменений не даёт значимого выигрыша."),
    "action": ("→", "Есть рекомендация оператору", "info",
                "Система нашла изменение уставок, которое улучшает показатели и не нарушает ограничения."),
    "no_feasible_solution": ("!", "Безопасного решения нет", "bad",
                              "Система отказывается советовать: последствия изменений проверить нечем."),
    "insufficient_data": ("?", "Данных недостаточно", "off",
                           "Входные данные неполные или устаревшие — цикл остановлен."),
}


def banner(rec) -> None:
    mark, title, tone, plain = STATUS.get(rec.status, ("•", rec.status, "info", ""))
    color = {"ok": "var(--ok)", "info": "var(--info)", "bad": "var(--bad)",
             "warn": "var(--warn)", "off": "var(--muted)"}[tone]
    st.markdown(
        f"""<div class="banner" style="border-left:6px solid {color}">
              <div class="mark" style="color:{color}">{mark}</div>
              <div>
                <p class="title">{title}</p>
                <p class="sub">{plain}<br>Момент времени: <b>{rec.timestamp}</b></p>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )


def _step(n: str, name: str, value: str, note: str, tone: str) -> str:
    return (f'<div class="step {tone}"><div class="n">{n}</div>'
            f'<div class="name">{name}</div><div class="val">{value}</div>'
            f'<div class="note">{note}</div></div>')


def agent_chain(result) -> None:
    st.markdown("#### Как отработали агенты")
    st.caption("Цикл идёт слева направо. Каждый агент отвечает за свой вопрос и передаёт вывод дальше.")
    dr, q, r, opt = result.data_report, result.quality, result.reliability, result.optimization
    cols = st.columns(5, gap="small")

    with cols[0]:
        tone = "ok" if dr.is_usable else "bad"
        st.markdown(_step(
            "Шаг 1", "Агент данных",
            f"{dr.tag_coverage:.0%} датчиков на месте",
            f"Пропущено {len(dr.missing_tags)}, аномалий {len(dr.anomalous_tags)}",
            tone), unsafe_allow_html=True)

    with cols[1]:
        if q is None:
            st.markdown(_step("Шаг 2", "Агент качества", "не запускался",
                              "Нет пригодных данных", "off"), unsafe_allow_html=True)
        else:
            s = q.sulfur_ppm
            tone = "bad" if q.spec_risk else "ok"
            val = f"сера {s:.2f} из {SULFUR_LIMIT_MG_KG:.0f} мг/кг" if s is not None else "сера неизвестна"
            st.markdown(_step("Шаг 2", "Агент качества", val,
                              f"Источник: {q.sulfur_source}, уверенность {q.confidence:.0%}",
                              tone), unsafe_allow_html=True)

    with cols[2]:
        if r is None:
            st.markdown(_step("Шаг 3", "Агент надёжности", "не запускался",
                              "Нет пригодных данных", "off"), unsafe_allow_html=True)
        else:
            tone = {"normal": "ok", "warn": "warn", "critical": "bad"}[r.severity_class]
            human = {"normal": "спокойный", "warn": "напряжённый", "critical": "тяжёлый"}[r.severity_class]
            st.markdown(_step("Шаг 3", "Агент надёжности", f"режим {human}",
                              f"Индекс тяжести {r.severity_index:.2f} из 1.00",
                              tone), unsafe_allow_html=True)

    with cols[3]:
        if opt is None:
            st.markdown(_step("Шаг 4", "Агент оптимизации", "не запускался",
                              "Нет пригодных данных", "off"), unsafe_allow_html=True)
        else:
            n_ok = len([c for c in opt.candidates if c.deltas])
            tone = "info" if n_ok else "ok"
            val = f"{n_ok} допустимых вариантов" if n_ok else "нет вариантов лучше текущего"
            st.markdown(_step("Шаг 4", "Агент оптимизации", val,
                              f"Отклонено ограничениями: {opt.infeasible_count}",
                              tone), unsafe_allow_html=True)

    with cols[4]:
        mark, title, tone, _ = STATUS.get(result.recommendation.status, ("•", "—", "info", ""))
        st.markdown(_step("Шаг 5", "Оркестратор", title,
                          "Собрал выводы и сформулировал ответ оператору",
                          tone), unsafe_allow_html=True)


def sulfur_gauge(value: float | None) -> None:
    if value is None:
        st.markdown("**Сера в товарном ДТ:** нет свежего измерения")
        return
    pct = min(value / SULFUR_LIMIT_MG_KG, 1.0) * 100
    color = "var(--bad)" if value >= SULFUR_LIMIT_MG_KG else ("var(--warn)" if pct >= 90 else "var(--ok)")
    st.markdown(
        f"""<div><b>Сера в товарном ДТ: {value:.2f} мг/кг</b> <span style="color:var(--muted)">
            · предел {SULFUR_LIMIT_MG_KG:.0f} мг/кг</span>
            <div class="gauge"><span style="width:{pct:.0f}%; background:{color}"></span></div></div>""",
        unsafe_allow_html=True,
    )


def recommendation_card(rec) -> None:
    rows = [
        ("Что происходит", rec.state_summary),
        ("Проблема или риск", rec.problem),
        ("Что делать", f'<span class="big">{rec.action}</span>'),
        ("Что это даст", rec.expected_effect),
        ("Какие ограничения проверены", rec.constraints_checked),
        ("Насколько уверены", rec.confidence),
        ("Почему именно так", rec.explanation),
    ]
    body = "".join(f'<div class="row"><div class="k">{k}</div><div class="v">{v}</div></div>'
                    for k, v in rows)
    st.markdown(f'<div class="card">{body}</div>', unsafe_allow_html=True)


def timeline(series: pd.DataFrame, current_ts=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["timestamp"], y=series["sulfur"], name="Сера, мг/кг",
                              mode="lines+markers", line=dict(color="#e3a008", width=2)))
    fig.add_hline(y=SULFUR_LIMIT_MG_KG, line=dict(color="#f85149", dash="dash"),
                  annotation_text="предел 10 мг/кг", annotation_position="top left")
    fig.add_trace(go.Scatter(x=series["timestamp"], y=series["severity"], name="Тяжесть режима",
                              mode="lines+markers", line=dict(color="#4c8dff", width=2), yaxis="y2"))
    if current_ts is not None:
        fig.add_vline(x=current_ts, line=dict(color="#9aa4b2", dash="dot"))
    fig.update_layout(
        height=330, margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3"),
        yaxis=dict(title="Сера, мг/кг", gridcolor="#222932"),
        yaxis2=dict(title="Тяжесть", overlaying="y", side="right", range=[0, 1], showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def details(result) -> None:
    with st.expander("Как агенты рассуждали (полная трасса)"):
        st.code("\n".join(result.recommendation.react_trace), language=None)

    with st.expander("Детали по агентам"):
        c1, c2 = st.columns(2)
        dr = result.data_report
        with c1:
            st.markdown("**Агент данных**")
            st.write(f"Полнота КИП: {dr.tag_coverage:.0%}")
            if dr.missing_tags:
                st.caption("Молчат: " + ", ".join(dr.missing_tags[:20]) +
                           (" …" if len(dr.missing_tags) > 20 else ""))
            if dr.anomalous_tags:
                st.caption("Аномальные значения: " + ", ".join(dr.anomalous_tags))
            for k, v in {**dr.lims_age_hours, **dr.pak_age_hours}.items():
                st.caption(f"Возраст {k}: {'нет данных' if v is None else f'{v:.1f} ч'}")
        with c2:
            if result.quality is not None:
                st.markdown("**Агент качества — прогноз ВАК**")
                st.dataframe(
                    pd.DataFrame(sorted(result.quality.predictions.items()),
                                 columns=["Показатель", "Значение"]),
                    hide_index=True, use_container_width=True)
            if result.reliability is not None:
                st.markdown("**Агент надёжности — факторы риска**")
                for f in result.reliability.risk_factors:
                    st.caption(f)

    opt = result.optimization
    if opt is not None and opt.candidates:
        with st.expander("Рассмотренные варианты (Парето-фронт)"):
            rows = [{"Вариант": c.label,
                      "Тяжесть режима": round(c.reliability_severity, 2),
                      "Выпуск-прокси": round(c.throughput_proxy, 2),
                      "Энерго-прокси": round(c.energy_proxy, 2)} for c in opt.candidates[:10]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def hitl(rec, key: str, later=None) -> None:
    if rec.status != "action":
        return
    st.markdown("#### Решение оператора")
    st.caption("Система не меняет режим сама: последнее слово за человеком.")
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
        st.success(f"Решение: {done.decision}")
        if later is not None:
            st.write("Мониторинг отклика: " + impact_agent.observe_response(done, later))


@st.cache_resource(show_spinner="Читаю телеметрию, ЛИМС и ПАК…")
def _environment():
    return load_environment()


@st.cache_data(show_spinner="Прогоняю сценарий через агентов…")
def _ticks(scenario_key: str):
    return list(runner.run_scenario(_environment(), SCENARIOS[scenario_key]))


def main() -> None:
    st.title("Мультиагентная система управления производством дизтоплива")
    st.markdown(
        '<p class="lead">Цепочка АВТ → гидроочистка 24-2000. Система читает телеметрию и анализы, '
        'оценивает качество и износ оборудования, перебирает варианты изменения режима и отдаёт '
        'оператору одно объяснимое решение — либо честно говорит, что решения нет.</p>',
        unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Что показать")
        mode = st.radio("Режим", ["Демо: готовые примеры", "Реальные данные"], label_visibility="collapsed")
        st.divider()
        with st.expander("Словарь терминов"):
            st.caption("**ЛИМС** — лаборатория, самый точный, но редкий источник качества.")
            st.caption("**ПАК** — поточный анализатор, меряет непрерывно, но менее точен.")
            st.caption("**ВАК** — формула, оценивающая качество по датчикам, когда анализа нет.")
            st.caption("**Сера ≤ 10 мг/кг** — жёсткое требование к товарному ДТ.")
            st.caption("**Тяжесть режима** — насколько режим далёк от привычного: прокси износа оборудования.")

    if mode.startswith("Демо"):
        ex = demo_examples()
        with st.sidebar:
            st.markdown("### Пример")
            key = st.radio("Пример", list(ex), format_func=lambda k: ex[k].title,
                           label_visibility="collapsed")
        chosen = ex[key]
        st.info(chosen.one_liner)
        result, series, current_ts = chosen.result, chosen.series, chosen.result.timestamp
        hitl_key = f"demo:{key}"
        later = None
    else:
        with st.sidebar:
            st.markdown("### Сценарий")
            skey = st.selectbox("Сценарий", list(SCENARIOS),
                                format_func=lambda k: SCENARIOS[k].title, label_visibility="collapsed")
            st.caption(f"{SCENARIOS[skey].start} — {SCENARIOS[skey].end}")
            if SCENARIOS[skey].inject_degraded:
                st.warning("Данные в этом сценарии испорчены намеренно.")
        ticks = _ticks(skey)
        if st.session_state.get("_skey") != skey:
            st.session_state["_skey"] = skey
            st.session_state["tick_idx"] = 0
        st.session_state.setdefault("tick_idx", 0)
        idx = st.slider("Момент времени (шаг 10 минут)", 0, max(len(ticks) - 1, 0), key="tick_idx")
        tick = ticks[idx]
        result, current_ts = tick.result, tick.timestamp
        series = pd.DataFrame({
            "timestamp": [t.timestamp for t in ticks],
            "sulfur": [t.recommendation.sulfur_ppm for t in ticks],
            "severity": [t.recommendation.reliability_severity for t in ticks],
        })
        hitl_key = f"{skey}:{idx}"
        later = ticks[idx + 1].recommendation if idx + 1 < len(ticks) else None

    rec = result.recommendation
    banner(rec)
    agent_chain(result)
    st.write("")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Что система говорит оператору")
        recommendation_card(rec)
        hitl(rec, hitl_key, later)
    with right:
        st.markdown("#### Ключевые показатели")
        sulfur_gauge(rec.sulfur_ppm)
        if rec.reliability_severity is not None:
            st.caption(f"Тяжесть режима: {rec.reliability_severity:.2f} из 1.00 "
                        "(0 — привычный режим, 1 — далёкий от нормы)")
        if series["sulfur"].notna().any():
            st.plotly_chart(timeline(series, current_ts), use_container_width=True)
        else:
            st.caption("График не строится: в этот период нет ни одного измерения серы — "
                        "именно поэтому система и остановила цикл.")

    details(result)


if __name__ == "__main__":
    main()
