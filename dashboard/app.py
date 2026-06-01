from __future__ import annotations
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from ru_liquidity_sentinel.aggregation.lsi_points import format_active_flags, lsi_points_frame
from ru_liquidity_sentinel.config import get_settings
from ru_liquidity_sentinel.core.dates import truncate_index_to_today
from ru_liquidity_sentinel.llm.comment import LlmCommentError, generate_llm_comment, llm_comment_configured, summarize_upcoming
from ru_liquidity_sentinel.llm.rag_chat import ChatSession, SentinelRAG
from ru_liquidity_sentinel.pipeline.runner import Pipeline
st.set_page_config(page_title='RU Liquidity Sentinel', layout='wide')
settings = get_settings()
PROC = settings.processed_dir
GREEN_MAX = float(settings.get('aggregation', 'lsi_green_max', default=40))
YELLOW_MAX = float(settings.get('aggregation', 'lsi_yellow_max', default=70))

@st.cache_data(ttl=3600)
def load_or_run_pipeline() -> bool:
    if (PROC / 'lsi.parquet').exists():
        return True
    try:
        Pipeline().run()
        return True
    except Exception as e:
        st.error(f'Ошибка пайплайна: {e}')
        return False

def load_parquet(name: str) -> pd.DataFrame:
    p = PROC / name
    if p.exists():
        df = pd.read_parquet(p)
        if 'date' in df.columns and (not isinstance(df.index, pd.DatetimeIndex)):
            df = df.set_index('date')
        return df
    return pd.DataFrame()

def status_color(status: str) -> str:
    return {'green': '#22c55e', 'yellow': '#eab308', 'red': '#ef4444'}.get(status, '#94a3b8')

def add_lsi_zones(fig: go.Figure) -> None:
    fig.add_hrect(y0=0, y1=GREEN_MAX, fillcolor='#22c55e', opacity=0.07, line_width=0)
    fig.add_hrect(y0=GREEN_MAX, y1=YELLOW_MAX, fillcolor='#eab308', opacity=0.07, line_width=0)
    fig.add_hrect(y0=YELLOW_MAX, y1=100, fillcolor='#ef4444', opacity=0.07, line_width=0)
    fig.add_hline(y=GREEN_MAX, line_dash='dot', line_color='#64748b', annotation_text='Норма / Напряжение')
    fig.add_hline(y=YELLOW_MAX, line_dash='dot', line_color='#64748b', annotation_text='Напряжение / Стресс')

def page_overview() -> None:
    st.title('RU Liquidity Sentinel')
    st.caption('Система раннего выявления стресса ликвидности рублёвого денежного рынка')
    if not load_or_run_pipeline():
        st.stop()
    lsi = truncate_index_to_today(load_parquet('lsi.parquet'))
    if lsi.empty:
        st.warning('Нет данных LSI. Запустите: `sentinel run`')
        st.stop()
    last = lsi.iloc[-1]
    lsi_val = float(last.get('lsi', 0))
    status = str(last.get('status', 'green'))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('LSI', f'{lsi_val:.1f}')
    c2.metric('Статус', last.get('status_ru', status))
    c3.metric('Seasonal Factor', f"{float(last.get('seasonal_factor', 1)):.2f}")
    c4.metric('Дата', str(lsi.index[-1])[:10])
    st.markdown(f"<div style='height:10px;background:{status_color(status)};border-radius:5px'></div>", unsafe_allow_html=True)
    fig = px.line(lsi.reset_index(), x=lsi.index.name or 'date', y='lsi', title='Индекс стресса ликвидности (0–100)')
    fig.update_traces(line_color='#1d4ed8', name='LSI')
    fig.update_layout(xaxis_title='Дата', yaxis_title='LSI')
    add_lsi_zones(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader('Алерты')
    if lsi_val >= YELLOW_MAX:
        st.error(f'🔴 Критический уровень стресса ликвидности (LSI = {lsi_val:.1f} ≥ {YELLOW_MAX:.0f})')
    elif lsi_val >= GREEN_MAX:
        st.warning(f'🟡 Повышенное напряжение ({GREEN_MAX:.0f} ≤ LSI = {lsi_val:.1f} < {YELLOW_MAX:.0f})')
    else:
        st.success(f'🟢 Нормальное состояние ликвидности (LSI = {lsi_val:.1f} < {GREEN_MAX:.0f})')
    flags = format_active_flags(last.get('active_flags'))
    if flags != 'нет':
        st.info(f'Активные флаги: {flags}')
    points = lsi_points_frame(lsi)
    if not points.empty:
        left, right = st.columns(2)
        with left:
            st.subheader('Вклад модулей (последняя дата)')
            last_pts = points.iloc[-1]
            mods = list(points.columns)
            fig_w = go.Figure(go.Waterfall(
                orientation='v',
                measure=['relative'] * len(mods) + ['total'],
                x=mods + ['LSI'],
                y=[float(last_pts[m]) for m in mods] + [float(last.get('lsi', last_pts.sum()))],
                text=[f'{float(last_pts[m]):.1f}' for m in mods] + [f"{float(last.get('lsi', 0)):.1f}"],
                connector={'line': {'color': '#94a3b8'}},
            ))
            fig_w.update_layout(
                title='Разложение LSI по модулям (баллы)',
                xaxis_title='Модуль',
                yaxis_title='Баллы LSI',
            )
            st.plotly_chart(fig_w, use_container_width=True)

        with right:
            st.subheader('Динамика вклада модулей')
            fig_a = px.area(points.tail(365))
            fig_a.update_layout(
                title='Вклад модулей в LSI за год (баллы)',
                legend_title='Модуль',
                xaxis_title='Дата',
                yaxis_title='Баллы LSI',
            )
            st.plotly_chart(fig_a, use_container_width=True)
    st.subheader('Аналитический комментарий')
    if not llm_comment_configured():
        st.warning(
            'Комментарий генерируется только через Yandex GPT. '
            'Укажите `YANDEX_API_KEY` и `YANDEX_FOLDER_ID` в `.env`, '
            'включите `llm.enabled: true` в `config/settings.yaml`.'
        )
    else:
        tax_str, ofz_str = summarize_upcoming(None, None, asof=lsi.index[-1])
        try:
            with st.spinner('Генерация комментария (Yandex GPT)…'):
                comment = generate_llm_comment(last, tax_str, ofz_str, lsi_history=lsi)
            st.markdown(comment)
        except LlmCommentError as exc:
            st.error(str(exc))
    with st.expander('Калибровка, backtest и sensitivity (±20% весов)'):
        cal_path = PROC / 'lsi_calibration.json'
        if cal_path.exists():
            st.caption('Параметры калибровки')
            st.json(json.loads(cal_path.read_text(encoding='utf-8')))
        bt_path = PROC / 'backtest_report.json'
        if bt_path.exists():
            st.caption('Backtest на исторических стресс-эпизодах')
            st.json(json.loads(bt_path.read_text(encoding='utf-8')))
        ov_path = PROC / 'overlap_report.json'
        if ov_path.exists():
            st.caption('Отчёт по двойному счёту M4')
            st.json(json.loads(ov_path.read_text(encoding='utf-8')))
        sens_path = PROC / 'sensitivity.csv'
        if sens_path.exists():
            st.dataframe(pd.read_csv(sens_path), use_container_width=True)

def _dual_axis(title: str, x, y1, n1, y2, n2, c1='#1d4ed8', c2='#dc2626') -> go.Figure:
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=x, y=y1, name=n1, line=dict(color=c1)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=y2, name=n2, line=dict(color=c2)), secondary_y=True)
    fig.update_layout(title=title, legend=dict(orientation='h'), xaxis_title='Дата')
    fig.update_yaxes(title_text=n1, secondary_y=False)
    fig.update_yaxes(title_text=n2, secondary_y=True)
    return fig

def chart_m1(df: pd.DataFrame) -> None:
    if {'reserve_spread', 'ruonia'} <= set(df.columns):
        d = df.tail(1500)
        st.plotly_chart(_dual_axis('Спред усреднения резервов + RUONIA', d.index, d['reserve_spread'], 'Спред усреднения, млрд ₽', d['ruonia'], 'RUONIA, %'), use_container_width=True)
    if 'flag_end_of_period' in df.columns:
        eop = int(df['flag_end_of_period'].sum())
        ofl = int(df.get('flag_overfulfillment', pd.Series(dtype=bool)).sum())
        st.caption(f'Дней «конец периода усреднения»: {eop} - паттерн «перебор нормы»: {ofl}')

def chart_m2(df: pd.DataFrame) -> None:
    rate_col = 'cutoff_rate' if 'cutoff_rate' in df.columns else 'weighted_rate'
    if 'cover_ratio' in df.columns and rate_col in df.columns:
        d = df.tail(400)
        fig = _dual_axis(
            'Аукционы репо ЦБ 7д: коэффициент покрытия и ставка',
            d.index,
            d['cover_ratio'],
            'Коэффициент покрытия',
            d[rate_col],
            'Ставка отсечения, %',
        )
        fig.add_hline(y=2.0, line_dash='dot', line_color='#dc2626', secondary_y=False, annotation_text='переспрос, cover>2')
        st.plotly_chart(fig, use_container_width=True)
    if 'flag_demand' in df.columns:
        st.caption(f"Аукционов с переспросом, cover>2: {int(df['flag_demand'].sum())}")

def chart_m3(df: pd.DataFrame) -> None:
    if 'cover_ratio' in df.columns:
        d = df.tail(400).reset_index()
        xcol = d.columns[0]
        fig = px.scatter(d, x=xcol, y='cover_ratio', title='Аукционы ОФЗ: коэффициент покрытия')
        fig.update_traces(mode='lines+markers', line_color='#1d4ed8')
        fig.update_layout(yaxis_title='Коэффициент покрытия', xaxis_title='Дата')
        fig.add_hline(y=1.2, line_dash='dot', line_color='#dc2626', annotation_text='недоспрос, cover<1.2')
        fig.add_hline(y=2.0, line_dash='dot', line_color='#16a34a', annotation_text='переспрос, cover>2.0')
        st.plotly_chart(fig, use_container_width=True)
    nd = int(df.get('flag_nedospros', pd.Series(dtype=bool)).sum())
    pr = int(df.get('flag_perespros', pd.Series(dtype=bool)).sum())
    st.caption(f'Недоспрос: {nd} аукционов - Переспрос: {pr} аукционов')

def _add_tax_week_bands(fig: go.Figure, d: pd.DataFrame) -> None:
    if 'tax_week_flag' not in d.columns:
        return
    active = d['tax_week_flag'].fillna(0).astype(bool)
    if not active.any():
        return
    prev = active.shift(1, fill_value=False)
    nxt = active.shift(-1, fill_value=False)
    for start, end in zip(d.index[active & ~prev], d.index[active & ~nxt], strict=False):
        fig.add_vrect(
            x0=start,
            x1=end + pd.Timedelta(days=1),
            fillcolor='#fbbf24',
            opacity=0.18,
            layer='below',
            line_width=0,
        )


def chart_m4(df: pd.DataFrame) -> None:
    if 'seasonal_factor' in df.columns:
        d = truncate_index_to_today(df).tail(900)
        xcol = d.index.name or 'date'
        fig = px.line(
            d.reset_index(),
            x=xcol,
            y='seasonal_factor',
            title='Сезонный множитель и налоговые недели',
        )
        fig.update_traces(line_color='#7c3aed', name='Сезонный множитель')
        fig.update_layout(xaxis_title='Дата', yaxis_title='Сезонный множитель')
        _add_tax_week_bands(fig, d)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color='#fbbf24'),
                name='Налоговая неделя',
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

def chart_m5(df: pd.DataFrame) -> None:
    delta_col = 'budget_delta' if 'budget_delta' in df.columns else None
    if delta_col:
        d = df.resample('MS').last().dropna(subset=[delta_col]).tail(60)
        colors = ['#16a34a' if v >= 0 else '#dc2626' for v in d[delta_col]]
        fig = go.Figure(go.Bar(x=d.index, y=d[delta_col], marker_color=colors, name='Изменение бюджетных средств, млрд ₽'))
        if 'budget_balance_bn' in d.columns:
            fig.add_trace(go.Scatter(x=d.index, y=d['budget_balance_bn'], name='Остатки, млрд ₽', line=dict(color='#1d4ed8'), yaxis='y2'))
            fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='Остатки, млрд ₽'))
        fig.update_layout(
            title='Приток / отток средств казначейства',
            legend=dict(orientation='h'),
            xaxis_title='Дата',
            yaxis_title='Изменение бюджетных средств, млрд ₽',
        )
        st.plotly_chart(fig, use_container_width=True)
    if 'flag_budget_drain' in df.columns:
        st.caption(f"Дней с флагом Budget_Drain: {int(df['flag_budget_drain'].sum())}")

def page_modules() -> None:
    st.header('Модули M1–M5')
    tabs = st.tabs(['M1 Резервы', 'M2 Репо', 'M3 ОФЗ', 'M4 Налоги', 'M5 Казначейство'])
    charts = [chart_m1, chart_m2, chart_m3, chart_m4, chart_m5]
    files = [f'm{i}_signals.parquet' for i in range(1, 6)]
    for tab, f, chart in zip(tabs, files, charts, strict=True):
        with tab:
            df = load_parquet(f)
            if df.empty:
                st.write('Нет данных')
                continue
            chart(df)
            with st.expander('Сырые сигналы (последние 30 строк)'):
                st.dataframe(df.tail(30), use_container_width=True)

def page_analyst() -> None:
    st.header('Интерактивный чат')
    if 'chat' not in st.session_state:
        st.session_state.chat = ChatSession()
    rag = SentinelRAG()
    st.sidebar.caption(f'RAG backend: {rag.backend}, документов: {len(rag.documents)}')
    st.markdown('**Например:** Что было с ликвидностью в марте 2022?')
    for msg in st.session_state.chat.messages:
        with st.chat_message(msg['role']):
            st.write(msg['content'])
    if (prompt := st.chat_input('Спросите о ликвидности, например: Что было в марте 2022?')):
        with st.chat_message('user'):
            st.write(prompt)
        answer = rag.ask(prompt, st.session_state.chat)
        with st.chat_message('assistant'):
            st.write(answer)

def main() -> None:
    page = st.sidebar.radio('Навигация', ['Обзор LSI', 'Модули', 'Аналитик (чат)'])
    if st.sidebar.button('Пересчитать пайплайн'):
        st.cache_data.clear()
        Pipeline().run()
        st.rerun()
    if page == 'Обзор LSI':
        page_overview()
    elif page == 'Модули':
        page_modules()
    else:
        page_analyst()
if __name__ == '__main__':
    main()
