"""Streaming no Brasil - dashboard interativo (Streamlit + Plotly)."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Streaming no Brasil", page_icon="▶", layout="wide")

CSV = Path(__file__).parent / "dados" / "simulacao_streaming_brasil.csv"
CAT_COLS = ["plataforma", "categoria", "genero", "titulo", "horario_pico"]
LABELS = {"plataforma": "Plataforma", "categoria": "Categoria", "genero": "Gênero",
          "titulo": "Título", "horario_pico": "Horário de pico"}
METRICS = {"reproducoes": "Reproduções", "usuarios_ativos": "Usuários ativos",
           "assinaturas": "Assinaturas", "receita_plataforma": "Receita",
           "tempo_medio_consumo": "Tempo médio de consumo (min)",
           "avaliacao_media": "Avaliação média"}
SUM_METRICS = {"reproducoes", "receita_plataforma"}  # demais métricas: média
COLORS = ["#2563EB", "#38BDF8", "#7C3AED", "#22D3EE", "#94A3B8"]

CSS = """
<style>
.stApp{background:#080B12;color:#F8FAFC}
[data-testid=stSidebar]{background:#101522;border-right:1px solid #273449}
[data-testid=stHeader]{background:transparent}
.hdr{padding:1.2rem 0 .6rem}.tag{color:#38BDF8;font-size:.78rem;letter-spacing:.12em;text-transform:uppercase}
.hdr h1{font-size:2.3rem;margin:.2rem 0;background:linear-gradient(90deg,#F8FAFC,#38BDF8);-webkit-background-clip:text;color:transparent}
.hdr p{color:#94A3B8;margin:0}
.kpi{background:#151B29;border:1px solid #273449;border-radius:14px;padding:1rem 1.1rem;box-shadow:0 6px 20px #0006;transition:.2s}
.kpi:hover{border-color:#2563EB;transform:translateY(-2px)}
.kpi span{color:#94A3B8;font-size:.78rem;text-transform:uppercase;letter-spacing:.06em}
.kpi b{display:block;font-size:1.55rem;margin-top:.25rem}
.ins{background:#101522;border-left:3px solid #2563EB;border-radius:8px;padding:.7rem 1rem;color:#cbd5e1;margin:.5rem 0}
h2,h3{color:#F8FAFC!important}
@media(max-width:640px){.hdr h1{font-size:1.6rem}.kpi b{font-size:1.2rem}}
</style>"""


def fmt(v, money=False):
    """Formata números grandes de forma compacta (pt-BR)."""
    for lim, s in ((1e9, " bi"), (1e6, " mi"), (1e3, " mil")):
        if abs(v) >= lim:
            v, suf = v / lim, s
            break
    else:
        suf = ""
    txt = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return (f"R$ {txt}" if money else txt) + suf


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")  # utf-8-sig remove o BOM do cabeçalho


@st.cache_data
def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes, textos, tipos; remove duplicatas, nulos e valores inválidos."""
    df = raw.copy()
    df.columns = df.columns.str.strip().str.lower()
    for c in [c for c in CAT_COLS if c in df]:
        df[c] = df[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    num = [c for c in METRICS if c in df]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    if {"ano", "mes"} <= set(df):  # recupera datas inválidas a partir de ano/mês
        alt = pd.to_datetime(dict(year=df["ano"], month=df["mes"], day=1), errors="coerce")
        df["data"] = df["data"].fillna(alt)
    df = df.drop_duplicates().dropna(subset=["data"] + num)
    df = df[(df[num] >= 0).all(axis=1)]
    if "avaliacao_media" in df:
        df = df[df["avaliacao_media"].between(0, 5)]
    # atributos derivados
    df["ano"], df["mes"] = df["data"].dt.year, df["data"].dt.month
    df["periodo"] = df["data"].dt.to_period("M").dt.to_timestamp()
    df["receita_por_assinatura"] = (df["receita_plataforma"] / df["assinaturas"].replace(0, np.nan))
    df["minutos_totais_estimados"] = df["reproducoes"] * df["tempo_medio_consumo"]
    return df.reset_index(drop=True)


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Cria filtros apenas para as colunas existentes e aplica ao DataFrame."""
    cats = [c for c in CAT_COLS if c in df]
    y0, y1 = int(df["ano"].min()), int(df["ano"].max())

    def reset():
        for c in cats:
            st.session_state[f"f_{c}"] = sorted(df[c].unique())
        st.session_state["f_ano"] = (y0, y1)

    st.sidebar.markdown("### Filtros")
    for c in cats:
        st.sidebar.multiselect(LABELS[c], sorted(df[c].unique()), default=sorted(df[c].unique()), key=f"f_{c}")
    if y0 < y1:
        st.sidebar.slider("Período (ano)", y0, y1, (y0, y1), key="f_ano")
    st.sidebar.button("Limpar filtros", on_click=reset, use_container_width=True)
    return apply_filters(df, cats)


def apply_filters(df: pd.DataFrame, cats) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for c in cats:
        mask &= df[c].isin(st.session_state.get(f"f_{c}", df[c].unique()))
    a, b = st.session_state.get("f_ano", (df["ano"].min(), df["ano"].max()))
    return df[mask & df["ano"].between(a, b)]


def calculate_kpis(df: pd.DataFrame) -> dict:
    top = df.groupby("plataforma")["receita_plataforma"].sum().idxmax()
    return {
        "Registros": f"{len(df):,}".replace(",", "."),
        "Reproduções": fmt(df["reproducoes"].sum()),
        "Usuários ativos (média)": fmt(df["usuarios_ativos"].mean()),
        "Assinaturas (média)": fmt(df["assinaturas"].mean()),
        "Receita total": fmt(df["receita_plataforma"].sum(), True),
        "Consumo médio (min)": f"{df['tempo_medio_consumo'].mean():.1f}",
        "Avaliação média": f"{df['avaliacao_media'].mean():.2f}",
        "Maior receita": top,
    }


def style(fig, h=380):
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", colorway=COLORS, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(color="#F8FAFC"), legend_title_text="")
    fig.update_xaxes(gridcolor="#273449")
    fig.update_yaxes(gridcolor="#273449")
    return fig


def create_platform_chart(df):
    g = df.groupby("plataforma", as_index=False)["receita_plataforma"].sum().sort_values("receita_plataforma")
    return style(px.bar(g, x="receita_plataforma", y="plataforma", orientation="h",
                        title="Receita total por plataforma", labels={"receita_plataforma": "Receita (R$)", "plataforma": ""}))


def create_share_chart(df):
    g = df.groupby("plataforma", as_index=False)["reproducoes"].sum()
    fig = px.pie(g, names="plataforma", values="reproducoes", hole=.6, title="Participação nas reproduções")
    return style(fig)


def create_consumption_chart(df):
    return style(px.box(df, x="plataforma", y="tempo_medio_consumo", color="plataforma",
                        title="Distribuição do tempo médio de consumo (min)",
                        labels={"tempo_medio_consumo": "Minutos", "plataforma": ""}))


def create_time_chart(df, metric):
    g = df.groupby(["periodo", "plataforma"], as_index=False)[metric].agg("sum" if metric in SUM_METRICS else "mean")
    g["média móvel 12m"] = g.groupby("plataforma")[metric].transform(lambda s: s.rolling(12, min_periods=3).mean())
    fig = px.line(g, x="periodo", y="média móvel 12m", color="plataforma",
                  title=f"{METRICS[metric]} - média móvel de 12 meses", labels={"periodo": "", "média móvel 12m": METRICS[metric]})
    return style(fig)


def create_profile_chart(df, col, metric):
    g = df.groupby(col, as_index=False)[metric].agg("sum" if metric in SUM_METRICS else "mean")
    return style(px.bar(g.sort_values(metric), x=col, y=metric, title=f"{METRICS[metric]} por {LABELS[col].lower()}",
                        labels={col: "", metric: METRICS[metric]}))


def generate_insights(df: pd.DataFrame) -> dict:
    """Gera textos interpretativos a partir dos dados filtrados."""
    p = df.groupby("plataforma").agg(rec=("receita_plataforma", "sum"), rep=("reproducoes", "sum"),
                                     ten=("tempo_medio_consumo", "mean"), av=("avaliacao_media", "mean"),
                                     usr=("usuarios_ativos", "mean"))
    out = {"plataforma": [
        f"**{p.rec.idxmax()}** concentra a maior receita ({fmt(p.rec.max(), True)}, {p.rec.max() / p.rec.sum():.1%} do total) "
        f"e **{p.rep.idxmax()}** lidera em reproduções ({p.rep.max() / p.rep.sum():.1%}).",
        f"**{p.ten.idxmax()}** tem o maior tempo médio de consumo ({p.ten.max():.1f} min) e **{p.av.idxmax()}** a melhor avaliação média ({p.av.max():.2f}). "
        f"A diferença entre maior e menor receita é de {fmt(p.rec.max() - p.rec.min(), True)}."]}
    h = df.groupby("horario_pico")["reproducoes"].sum()
    c = df.groupby("categoria")["reproducoes"].sum()
    g = df.groupby("genero")["avaliacao_media"].mean()
    out["perfil"] = [f"O horário de pico com mais reproduções é **{h.idxmax()}**; a categoria com mais reproduções é **{c.idxmax()}** "
                     f"e o gênero mais bem avaliado é **{g.idxmax()}** ({g.max():.2f})."]
    a = df.groupby("ano")["receita_plataforma"].sum()
    if len(a) > 1:
        var = (a.iloc[-1] / a.iloc[0] - 1) if a.iloc[0] else np.nan
        out["tempo"] = [f"A receita anual atinge o pico em **{a.idxmax()}** ({fmt(a.max(), True)}). Entre {a.index[0]} e {a.index[-1]} a variação foi de {var:+.1%}. "
                        "Anos com menos plataformas ou registros nos filtros influenciam essa comparação."]
    else:
        out["tempo"] = ["Selecione mais de um ano para observar a evolução temporal."]
    return out


def show(texts):
    for t in texts:
        st.markdown(f'<div class="ins">{t}</div>', unsafe_allow_html=True)


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="hdr"><div class="tag">Data Analytics | Python | Streamlit</div>'
                '<h1>Streaming no Brasil</h1><p>Painel interativo de análise de consumo e comportamento</p></div>',
                unsafe_allow_html=True)
    try:
        df = clean_data(load_data(CSV))
    except Exception as e:
        st.error(f"Não foi possível carregar a base em {CSV.name}: {e}")
        st.stop()
    with st.expander("Sobre o problema analisado"):
        st.write("Como se distribuem consumo, receita, assinaturas e avaliações entre plataformas, categorias e gêneros "
                 "de streaming ao longo do tempo? A base é simulada e não possui região, estado, plano ou preço, "
                 "por isso esses recortes não são analisados.")
    f = sidebar_filters(df)
    if f.empty:
        st.warning("Nenhum registro encontrado para os filtros selecionados.")
        st.stop()

    cols = st.columns(4)
    for i, (k, v) in enumerate(calculate_kpis(f).items()):
        cols[i % 4].markdown(f'<div class="kpi"><span>{k}</span><b>{v}</b></div>', unsafe_allow_html=True)
        if i == 3:
            st.write("")

    st.header("Visão por plataforma")
    a, b = st.columns(2)
    a.plotly_chart(create_platform_chart(f), use_container_width=True)
    b.plotly_chart(create_share_chart(f), use_container_width=True)
    ins = generate_insights(f)
    show(ins["plataforma"])

    st.header("Consumo e perfil")
    a, b = st.columns(2)
    a.plotly_chart(create_consumption_chart(f), use_container_width=True)
    dim = b.selectbox("Dimensão", [c for c in ["categoria", "genero", "horario_pico", "titulo"] if c in f],
                      format_func=LABELS.get)
    met = b.selectbox("Métrica", list(METRICS), format_func=METRICS.get, key="m_perfil")
    b.plotly_chart(create_profile_chart(f, dim, met), use_container_width=True)
    show(ins["perfil"])

    st.header("Análise temporal")
    m = st.selectbox("Métrica da série", list(METRICS), format_func=METRICS.get, key="m_tempo")
    st.plotly_chart(create_time_chart(f, m), use_container_width=True)
    show(ins["tempo"])

    st.header("Comparativo entre plataformas")
    plats = sorted(f["plataforma"].unique())
    sel = st.multiselect("Plataformas", plats, default=plats, key="cmp_p")
    ms = st.multiselect("Métricas", list(METRICS), default=["receita_plataforma", "tempo_medio_consumo", "avaliacao_media"],
                        format_func=METRICS.get, key="cmp_m")
    if sel and ms:
        t = f[f["plataforma"].isin(sel)].groupby("plataforma")[ms].agg(
            {c: "sum" if c in SUM_METRICS else "mean" for c in ms})
        norm = (t / t.max()).reset_index().melt("plataforma", var_name="métrica", value_name="índice (máx = 1)")
        norm["métrica"] = norm["métrica"].map(METRICS)
        st.plotly_chart(style(px.bar(norm, x="plataforma", y="índice (máx = 1)", color="métrica", barmode="group",
                                     title="Métricas normalizadas pelo maior valor")), use_container_width=True)
        st.dataframe(t.rename(columns=METRICS).round(2), use_container_width=True)
        st.caption("Reproduções e receita são somas; as demais métricas são médias.")

    st.header("Dados filtrados")
    q = st.text_input("Pesquisar (qualquer coluna)")
    t = f.drop(columns=["periodo"])
    if q:
        t = t[t.astype(str).apply(lambda s: s.str.contains(q, case=False, regex=False)).any(axis=1)]
    st.caption(f"{len(t):,} registros encontrados".replace(",", "."))
    st.dataframe(t, use_container_width=True, height=360)
    st.download_button("Baixar dados filtrados", t.to_csv(index=False).encode("utf-8-sig"),
                       "streaming_filtrado.csv", "text/csv")

    st.header("Conclusão Executiva")
    p = f.groupby("plataforma")["receita_plataforma"].sum()
    show([f"Principal plataforma: **{p.idxmax()}** ({p.max() / p.sum():.1%} da receita filtrada).",
          *ins["perfil"], *ins["tempo"],
          f"Base filtrada: {len(f):,} registros de {f['ano'].min()} a {f['ano'].max()}.".replace(",", ".")])


main()
