import threading
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64

from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc

from src.simulacao import simular_mercado_streaming
from src.ativos import carregar_ifix
from funcoes.fatos_estilizados import (
    plot_intermitencia, plot_acf_retornos, plot_power_law,
    plot_gaussianidade_agregacional, calcular_residuos_garch,
)

PARAMETROS_SISTEMA = [0.7074504, 0.87356055, 0.67826246, 0.2438783, 0.08593365]
PRECO_BASE = (3_000_000 + 50_000) / 100_000 * 0.65

lock = threading.Lock()
estado_sim = {
    "rodando": False, "pausado": False, "concluido": False,
    "gerador": None, "mercado_obj": None,
    "historico_precos": [], "sentimentos": [],
    "hist_patrimonio": [], "hist_dividendos": [],
    "usuario": {"caixa": 10_000, "cotas": 100, "total_dividendos": 0.0},
    "dia_atual": 0, "num_dias": 100, "window_vol": 20,
    "choques_noticias": [], "choques_econ": [], "choques_fii": [],
    "cfg_not": [], "cfg_econ": [], "cfg_fii": [],
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def calc_vol_rolante(precos, window):
    arr = np.array(precos)
    if len(arr) < window + 2:
        return np.array([np.nan])
    lr = np.diff(np.log(arr))
    vp = np.full(len(lr), np.nan)
    for i in range(window, len(lr)):
        vp[i] = np.std(lr[i-window:i]) * np.sqrt(252)
    return vp

def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
    buf.seek(0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def build_fig_mercado(hp, sen, dia, window, ch_not, ch_econ, ch_fii):
    N = 252
    n_sim  = len(hp) - N
    n_sent = len(sen)
    vol    = calc_vol_rolante(hp, window)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=False,
                        vertical_spacing=0.07, row_heights=[0.40, 0.30, 0.30],
                        subplot_titles=(f"Preço da cota (dia {dia})",
                                        "Sentimento médio dos agentes",
                                        f"Volatilidade rolante ({window} dias)"))

    fig.add_trace(go.Scatter(x=list(range(-N, 1)), y=hp[:N+1], mode="lines",
                             line=dict(color="#adb5bd", width=1.0), name="IFIX"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(0, n_sim+1)), y=hp[N:], mode="lines",
                             line=dict(color="#1a1a2e", width=1.5), name="Simulação"), row=1, col=1)
    fig.add_vline(x=0, line_dash="dot", line_color="#888780", line_width=1, opacity=0.7, row=1, col=1)

    if n_sent > 0:
        sa = np.array(sen)
        xs = list(range(1, n_sent+1))
        fig.add_trace(go.Scatter(x=xs, y=list(np.where(sa>=0, sa, 0)), mode="lines",
                                 line=dict(width=0), fill="tozeroy",
                                 fillcolor="rgba(69,123,157,0.15)", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=xs, y=list(np.where(sa<0, sa, 0)), mode="lines",
                                 line=dict(width=0), fill="tozeroy",
                                 fillcolor="rgba(230,57,70,0.15)", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=xs, y=list(sa), mode="lines",
                                 line=dict(color="#457b9d", width=1.2), name="Sentimento"), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#adb5bd", line_width=0.8, row=2, col=1)
    fig.update_xaxes(range=[-N-5, n_sim+5], row=2, col=1)

    if len(vol) > 1:
        xv = list(range(-N, -N+len(vol)))
        fig.add_trace(go.Scatter(x=xv, y=list(vol), mode="lines",
                                 line=dict(color="#e76f51", width=1.2), fill="tozeroy",
                                 fillcolor="rgba(231,111,81,0.08)", name="Vol."), row=3, col=1)
        fig.add_vline(x=0, line_dash="dot", line_color="#888780", line_width=1, opacity=0.6, row=3, col=1)
        fig.update_xaxes(range=[-N-5, n_sim+5], row=3, col=1)

    for c in ch_not:
        if c.get("dia", 0) <= n_sim:
            cor = "#e63946" if c.get("tipo") == "negativo" else "#2a9d8f"
            for r in [1, 2, 3]:
                fig.add_vline(x=c["dia"], line_dash="dash", line_color=cor, line_width=1, opacity=0.7, row=r, col=1)
            fig.add_annotation(x=c["dia"]+1, y=1.01, yref="paper",
                               text=f"📰{c.get('tipo','')[:3]}", showarrow=False,
                               font=dict(size=9, color=cor), xanchor="left")
    for c in ch_econ:
        if c.get("dia", 0) <= n_sim:
            for r in [1, 2, 3]:
                fig.add_vline(x=c["dia"], line_dash="dot", line_color="#e9c46a", line_width=1.5, opacity=0.9, row=r, col=1)
            fig.add_annotation(x=c["dia"]+1, y=0.96, yref="paper",
                               text="📊econ", showarrow=False,
                               font=dict(size=9, color="#c9a84c"), xanchor="left")
    for c in ch_fii:
        if c.get("dia", 0) <= n_sim:
            for r in [1, 2, 3]:
                fig.add_vline(x=c["dia"], line_dash="dashdot", line_color="#7f77dd", line_width=1, opacity=0.7, row=r, col=1)
            fig.add_annotation(x=c["dia"]+1, y=0.91, yref="paper",
                               text="🏠FII", showarrow=False,
                               font=dict(size=9, color="#7f77dd"), xanchor="left")

    fig.update_layout(height=600, showlegend=False,
                      margin=dict(l=50, r=20, t=50, b=40),
                      paper_bgcolor="white", plot_bgcolor="white",
                      hovermode="x unified")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.07)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.07)", zeroline=False)
    fig.update_yaxes(range=[-1.15, 1.15], row=2, col=1)
    fig.update_yaxes(title_text="Preço (R$)",   title_font_size=10, row=1, col=1)
    fig.update_yaxes(title_text="Sentimento",   title_font_size=10, row=2, col=1)
    fig.update_yaxes(title_text="Volatilidade", title_font_size=10, row=3, col=1)
    fig.update_xaxes(title_text="Dias",         title_font_size=10, row=3, col=1)
    return fig

def build_fig_patrimonio(hist, divs, dia):
    if not hist:
        return go.Figure(layout=dict(height=180, paper_bgcolor="white", plot_bgcolor="white"))
    arr = np.array(hist)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(hist)+1)), y=hist, mode="lines",
                             line=dict(color="#2a9d8f", width=1.5),
                             hovertemplate="Dia %{x}<br>R$ %{y:,.2f}<extra></extra>"))
    for d in divs:
        fig.add_annotation(x=d["dia"], y=d["patrimonio"],
                           text=f"💰R${d['valor']:.0f}",
                           showarrow=True, arrowhead=2, arrowcolor="#e9c46a",
                           font=dict(size=8, color="#c9a84c"),
                           bgcolor="rgba(255,255,255,0.8)")
    fig.update_layout(height=200, showlegend=False,
                      margin=dict(l=50, r=20, t=30, b=40),
                      paper_bgcolor="white", plot_bgcolor="white",
                      title=dict(text=f"Patrimônio do investidor (dia {dia})", font=dict(size=11)),
                      yaxis=dict(range=[arr.min()*0.995, arr.max()*1.005]))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.07)", title_text="Dias")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.07)", title_text="R$")
    return fig

# ── Funções de configuração de choques ─────────────────────────────────────────
def make_cfg_not(n):
    defs = [{"dia":30,"tipo":"negativo","int":0.7,"dur":2,"delta":0.8},
            {"dia":50,"tipo":"positivo","int":0.5,"dur":3,"delta":0.8},
            {"dia":80,"tipo":"negativo","int":0.4,"dur":5,"delta":0.7}]
    rows = []
    for i in range(3):
        d = defs[i]
        visible = i < n
        rows.append(html.Div(style={"display":"block" if visible else "none"}, children=[
            dbc.Card(className="mb-1 p-1", children=[
                html.Small(f"Choque {i+1}", className="fw-bold"),
                html.Label("Dia", className="small"),
                dcc.Input(id=f"not-dia-{i}",   type="number", value=d["dia"], min=1, style={"width":"100%"}),
                html.Label("Tipo", className="small"),
                dcc.Dropdown(id=f"not-tipo-{i}",
                             options=[{"label":"Negativo","value":"negativo"},
                                      {"label":"Positivo","value":"positivo"}],
                             value=d["tipo"], clearable=False),
                html.Label("Intensidade", className="small"),
                dcc.Slider(id=f"not-int-{i}",   min=0.1, max=1.0, step=0.05, value=d["int"],
                           marks={0.1:"0.1", 1.0:"1.0"}, tooltip={"always_visible":True}),
                html.Label("Duração (dias)", className="small"),
                dcc.Input(id=f"not-dur-{i}",   type="number", value=d["dur"], min=1, max=30, style={"width":"100%"}),
                html.Label("Delta", className="small"),
                dcc.Slider(id=f"not-delta-{i}", min=0.1, max=1.0, step=0.05, value=d["delta"],
                           marks={0.1:"0.1", 1.0:"1.0"}, tooltip={"always_visible":True}),
            ])
        ]))
    return rows

def make_cfg_econ(n):
    defs = [{"dia":50,"inf":0.05,"premio":0.06},
            {"dia":150,"inf":0.06,"premio":0.07},
            {"dia":300,"inf":0.07,"premio":0.08}]
    rows = []
    for i in range(3):
        d = defs[i]
        visible = i < n
        rows.append(html.Div(style={"display":"block" if visible else "none"}, children=[
            dbc.Card(className="mb-1 p-1", children=[
                html.Small(f"Choque {i+1}", className="fw-bold"),
                html.Label("Dia", className="small"),
                dcc.Input(id=f"econ-dia-{i}", type="number", value=d["dia"], min=1, style={"width":"100%"}),
                html.Label("Inflação", className="small"),
                dcc.Slider(id=f"econ-inf-{i}", min=0.01, max=0.20, step=0.01, value=d["inf"],
                           marks={0.01:"1%", 0.20:"20%"}, tooltip={"always_visible":True}),
                html.Label("Prêmio", className="small"),
                dcc.Slider(id=f"econ-premio-{i}", min=0.01, max=0.20, step=0.01, value=d["premio"],
                           marks={0.01:"1%", 0.20:"20%"}, tooltip={"always_visible":True}),
            ])
        ]))
    return rows

def make_cfg_fii(n):
    defs = [{"dia":30,"vac":50,"custo":20},
            {"dia":100,"vac":-30,"custo":50},
            {"dia":200,"vac":20,"custo":-20}]
    rows = []
    for i in range(3):
        d = defs[i]
        visible = i < n
        rows.append(html.Div(style={"display":"block" if visible else "none"}, children=[
            dbc.Card(className="mb-1 p-1", children=[
                html.Small(f"Choque {i+1}", className="fw-bold"),
                html.Label("Dia", className="small"),
                dcc.Input(id=f"fii-dia-{i}", type="number", value=d["dia"], min=1, style={"width":"100%"}),
                html.Label("Vacância (%)", className="small"),
                dcc.Slider(id=f"fii-vac-{i}", min=-80, max=300, step=5, value=d["vac"],
                           marks={-80:"-80%", 0:"0", 300:"300%"}, tooltip={"always_visible":True}),
                html.Label("Custo (%)", className="small"),
                dcc.Slider(id=f"fii-custo-{i}", min=-80, max=300, step=5, value=d["custo"],
                           marks={-80:"-80%", 0:"0", 300:"300%"}, tooltip={"always_visible":True}),
            ])
        ]))
    return rows

# ── App ────────────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)

app.layout = dbc.Container(fluid=True, children=[
    dcc.Interval(id="intervalo", interval=300, n_intervals=0, disabled=True),
    dcc.Store(id="store-tick",   data=0),
    dcc.Store(id="store-pausado",  data=False),
    dcc.Store(id="store-acao-msg",  data=""),
    dcc.Store(id="store-venda-msg", data=""),

    dbc.Row([
        # ── Sidebar esquerda ───────────────────────────────────────────────────
        dbc.Col(width=2, style={"overflowY":"auto","height":"100vh","fontSize":"13px"}, children=[
            html.H5("⚙ Configurações", className="mt-3"),
            html.Hr(),
            html.Label("Dias de simulação"),
            dcc.Slider(id="sl-dias", min=50, max=500, step=10, value=100,
                       marks={50:"50",250:"250",500:"500"}, tooltip={"always_visible":True}),
            html.Label("Preço de referência (R$)", className="mt-2"),
            dcc.Input(id="inp-preco", type="number", value=20.15, min=1, max=10000,
                      step=0.5, style={"width":"100%"}),
            html.Label("Janela volatilidade", className="mt-2"),
            dcc.Slider(id="sl-vol", min=5, max=60, step=5, value=20,
                       marks={5:"5",30:"30",60:"60"}, tooltip={"always_visible":True}),
            html.Label("Velocidade (ms/dia)", className="mt-2"),
            dcc.Slider(id="sl-vel", min=100, max=2000, step=100, value=300,
                       marks={100:"rápido",1000:"médio",2000:"lento"}, tooltip={"always_visible":True}),
            html.Hr(),
            html.H6("🏦 Banco Central"),
            html.Label("Selic inicial"),
            dcc.Slider(id="sl-selic", min=0.01, max=0.30, step=0.01, value=0.15,
                       marks={0.01:"1%",0.15:"15%",0.30:"30%"}, tooltip={"always_visible":True}),
            html.Label("Inflação inicial"),
            dcc.Slider(id="sl-inf", min=0.01, max=0.20, step=0.01, value=0.07,
                       marks={0.01:"1%",0.10:"10%",0.20:"20%"}, tooltip={"always_visible":True}),
            html.Label("Prêmio de risco"),
            dcc.Slider(id="sl-premio", min=0.01, max=0.20, step=0.01, value=0.08,
                       marks={0.01:"1%",0.10:"10%",0.20:"20%"}, tooltip={"always_visible":True}),
            html.Label("Prob. choque aleatório"),
            dcc.Slider(id="sl-prob", min=0.0, max=0.2, step=0.005, value=0.025,
                       marks={0:"0",0.1:"0.1",0.2:"0.2"}, tooltip={"always_visible":True}),
            html.Hr(),
            html.H6("📰 Choques notícias"),
            dcc.RadioItems(id="rd-not",
                           options=[{"label":f" {i}","value":i} for i in range(4)],
                           value=0, inline=True, className="mb-1"),
            html.Div(id="div-cfg-not", children=make_cfg_not(0)),
            html.Hr(),
            html.H6("📊 Choques econômicos"),
            dcc.RadioItems(id="rd-econ",
                           options=[{"label":f" {i}","value":i} for i in range(4)],
                           value=0, inline=True, className="mb-1"),
            html.Div(id="div-cfg-econ", children=make_cfg_econ(0)),
            html.Hr(),
            html.H6("🏠 Choques FII"),
            dcc.RadioItems(id="rd-fii",
                           options=[{"label":f" {i}","value":i} for i in range(4)],
                           value=0, inline=True, className="mb-1"),
            html.Div(id="div-cfg-fii", children=make_cfg_fii(0)),
        ]),

        # ── Área principal ─────────────────────────────────────────────────────
        dbc.Col(width=8, children=[
            html.H3("🏢 FiiLMA: Laboratório de Mercados Artificiais de Fiis",
                    className="mt-3 mb-2"),
            dbc.Row([
                dbc.Col(dbc.Button("▶ Rodar",   id="btn-rodar",   color="success"), width="auto"),
                dbc.Col(dbc.Button("⏸ Pausar",  id="btn-pausar",  color="warning"), width="auto"),
                dbc.Col(dbc.Button("▶ Retomar", id="btn-retomar", color="primary"), width="auto"),
                dbc.Col(dbc.Button("⏹ Parar",   id="btn-parar",   color="danger"),  width="auto"),
            ], className="mb-2 g-2"),
            html.Div(id="div-status"),
            dbc.Progress(id="prog-bar", value=0, className="mb-2"),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.P("Preço",        className="small mb-0"), html.H5(id="met-preco", children="—")]))),
                dbc.Col(dbc.Card(dbc.CardBody([html.P("Sentimento",   className="small mb-0"), html.H5(id="met-sent",  children="—")]))),
                dbc.Col(dbc.Card(dbc.CardBody([html.P("Volatilidade", className="small mb-0"), html.H5(id="met-vol",   children="—")]))),
                dbc.Col(dbc.Card(dbc.CardBody([html.P("Dia",          className="small mb-0"), html.H5(id="met-dia",   children="—")]))),
            ], className="mb-2 g-2"),
            dcc.Tabs(value="tab-sim", children=[
                dcc.Tab(label="🔄 Simulação", value="tab-sim", children=[
                    dcc.Graph(id="grafico-mercado",    figure=go.Figure(), config={"displayModeBar":False}),
                    dcc.Graph(id="grafico-patrimonio", figure=go.Figure(), config={"displayModeBar":False}),
                ]),
                dcc.Tab(label="📊 Fatos Estilizados", value="tab-fatos", children=[
                    html.Div(className="mt-3 mb-2", children=[
                        dbc.Row([
                            dbc.Col([html.Label("Lags"),
                                     dcc.Slider(id="sl-lags", min=10, max=60, step=5, value=40,
                                                marks={10:"10",40:"40",60:"60"}, tooltip={"always_visible":True})]),
                            dbc.Col([html.Label("Alpha"),
                                     dcc.Slider(id="sl-alpha", min=0.01, max=0.10, step=0.01, value=0.05,
                                                marks={0.01:"0.01",0.05:"0.05",0.10:"0.10"}, tooltip={"always_visible":True})]),
                            dbc.Col([html.Label("Cor"),
                                     dcc.Dropdown(id="dd-cor",
                                                  options=[{"label":c,"value":c} for c in
                                                           ["tab:blue","tab:orange","tab:green","steelblue"]],
                                                  value="tab:blue", clearable=False)]),
                            dbc.Col(dbc.Button("📊 Calcular", id="btn-fatos", color="primary",
                                               className="mt-4 w-100")),
                        ]),
                    ]),
                    html.Div(id="div-fatos"),
                ]),
            ]),
        ]),

        # ── Sidebar direita — Investidor ───────────────────────────────────────
        dbc.Col(width=2, style={"overflowY":"auto","height":"100vh","fontSize":"13px"}, children=[
            html.H5("👤 Investidor", className="mt-3"),
            html.Hr(),
            html.Div(id="div-preco-pause"),
            dbc.Card(className="mb-2", children=dbc.CardBody([
                html.P("💵 Caixa",      className="mb-0 small"), html.H6(id="inv-caixa", children="R$ —"),
                html.P("📦 Cotas",      className="mb-0 small mt-1"), html.H6(id="inv-cotas", children="—"),
                html.P("💼 Patrimônio", className="mb-0 small mt-1"), html.H6(id="inv-patr",  children="R$ —"),
                html.P("💰 Dividendos", className="mb-0 small mt-1"), html.H6(id="inv-div",   children="R$ —"),
            ])),
            html.Div(id="div-acao", style={"display":"none"}, children=[
                html.H6("Comprar"),
                html.Label("Preço (R$)", className="small"),
                dcc.Input(id="inp-pc", type="number", value=0, min=0, step=0.01, style={"width":"100%"}),
                html.Label("Quantidade", className="small"),
                dcc.Input(id="inp-qc", type="number", value=0, min=0, step=1, style={"width":"100%"}),
                html.P(id="txt-custo", children="Custo: R$ 0,00", className="small text-muted"),
                dbc.Button("✅ Comprar", id="btn-comprar", color="success", size="sm", className="w-100 mb-1"),
                html.Div(id="div-msg-acao",  className="mt-1"),
                html.Div(id="div-msg-venda", className="mb-2"),
                html.H6("Vender"),
                html.Label("Preço (R$)", className="small"),
                dcc.Input(id="inp-pv", type="number", value=0, min=0, step=0.01, style={"width":"100%"}),
                html.Label("Quantidade", className="small"),
                dcc.Input(id="inp-qv", type="number", value=0, min=0, step=1, style={"width":"100%"}),
                html.P(id="txt-receita", children="Receita: R$ 0,00", className="small text-muted"),
                dbc.Button("💸 Vender", id="btn-vender", color="danger", size="sm", className="w-100"),
            ]),
        ]),
    ]),
])


# ── Callbacks choques (mostrar/ocultar) ────────────────────────────────────────
@app.callback(Output("div-cfg-not",  "children"), Input("rd-not",  "value"))
def cfg_not(n):  return make_cfg_not(n)

@app.callback(Output("div-cfg-econ", "children"), Input("rd-econ", "value"))
def cfg_econ(n): return make_cfg_econ(n)

@app.callback(Output("div-cfg-fii",  "children"), Input("rd-fii",  "value"))
def cfg_fii(n):  return make_cfg_fii(n)


# ── Callback custo/receita ─────────────────────────────────────────────────────
@app.callback(
    Output("txt-custo",   "children"),
    Output("txt-receita", "children"),
    Input("inp-pc", "value"), Input("inp-qc", "value"),
    Input("inp-pv", "value"), Input("inp-qv", "value"),
)
def totais(pc, qc, pv, qv):
    return f"Custo: R$ {(pc or 0)*(qc or 0):,.2f}", f"Receita: R$ {(pv or 0)*(qv or 0):,.2f}"


# ── Callback controle simulação ────────────────────────────────────────────────
@app.callback(
    Output("intervalo",    "disabled"),
    Output("intervalo",    "interval"),
    Output("store-pausado","data"),
    Input("btn-rodar",   "n_clicks"),
    Input("btn-pausar",  "n_clicks"),
    Input("btn-retomar", "n_clicks"),
    Input("btn-parar",   "n_clicks"),
    State("sl-dias",   "value"), State("inp-preco", "value"),
    State("sl-vol",    "value"), State("sl-vel",    "value"),
    State("sl-selic",  "value"), State("sl-inf",    "value"),
    State("sl-premio", "value"), State("sl-prob",   "value"),
    State("rd-not",    "value"), State("rd-econ",   "value"), State("rd-fii", "value"),
    State("not-dia-0",   "value"), State("not-dia-1",   "value"), State("not-dia-2",   "value"),
    State("not-tipo-0",  "value"), State("not-tipo-1",  "value"), State("not-tipo-2",  "value"),
    State("not-int-0",   "value"), State("not-int-1",   "value"), State("not-int-2",   "value"),
    State("not-dur-0",   "value"), State("not-dur-1",   "value"), State("not-dur-2",   "value"),
    State("not-delta-0", "value"), State("not-delta-1", "value"), State("not-delta-2", "value"),
    State("econ-dia-0",    "value"), State("econ-dia-1",    "value"), State("econ-dia-2",    "value"),
    State("econ-inf-0",    "value"), State("econ-inf-1",    "value"), State("econ-inf-2",    "value"),
    State("econ-premio-0", "value"), State("econ-premio-1", "value"), State("econ-premio-2", "value"),
    State("fii-dia-0",   "value"), State("fii-dia-1",   "value"), State("fii-dia-2",   "value"),
    State("fii-vac-0",   "value"), State("fii-vac-1",   "value"), State("fii-vac-2",   "value"),
    State("fii-custo-0", "value"), State("fii-custo-1", "value"), State("fii-custo-2", "value"),
    prevent_initial_call=True,
)
def controlar(r,p,ret,stop,
              n_dias,preco_ref,window_vol,vel,selic,inflacao,premio,prob,
              n_not,n_econ,n_fii,
              nd0,nd1,nd2, nt0,nt1,nt2, ni0,ni1,ni2, ndu0,ndu1,ndu2, ndel0,ndel1,ndel2,
              ed0,ed1,ed2, ei0,ei1,ei2, ep0,ep1,ep2,
              fd0,fd1,fd2, fv0,fv1,fv2, fc0,fc1,fc2):
    global estado_sim
    triggered = ctx.triggered_id
    vel = vel or 300

    with lock:
        if triggered == "btn-rodar":
            escala = (preco_ref or PRECO_BASE) / PRECO_BASE
            n_not  = n_not  or 0
            n_econ = n_econ or 0
            n_fii  = n_fii  or 0

            cfg_not = [{"dia":d,"tipo":t or "negativo","intensidade":i or 0.5,
                        "duracao":du or 2,"delta":de or 0.8}
                       for d,t,i,du,de in zip([nd0,nd1,nd2],[nt0,nt1,nt2],
                           [ni0,ni1,ni2],[ndu0,ndu1,ndu2],[ndel0,ndel1,ndel2])][:n_not]

            cfg_econ = [{"dia":d,"inflacao":i or 0.07,"premio":p or 0.08}
                        for d,i,p in zip([ed0,ed1,ed2],[ei0,ei1,ei2],[ep0,ep1,ep2])][:n_econ]

            cfg_fii = [{"dia":d,"vac":v or 0,"custo":c or 0}
                       for d,v,c in zip([fd0,fd1,fd2],[fv0,fv1,fv2],[fc0,fc1,fc2])][:n_fii]

            estado_sim.update({
                "rodando":True,"pausado":False,"concluido":False,
                "historico_precos":[],"sentimentos":[],
                "hist_patrimonio":[],"hist_dividendos":[],
                "dia_atual":0,"num_dias":n_dias or 100,"window_vol":window_vol or 20,
                "choques_noticias":[],"choques_econ":[],"choques_fii":[],
                "cfg_not":cfg_not,"cfg_econ":cfg_econ,"cfg_fii":cfg_fii,
                "usuario":{"caixa":10_000*escala,"cotas":100,"total_dividendos":0.0},
            })
            overrides = {
                "taxa_selic":selic or 0.15,"expectativa_inflacao":inflacao or 0.07,
                "premio_risco":premio or 0.08,"escala":escala,
                "prob_choque_diario":prob or 0.025,
                "dia_choque":-1,"intensidade":0.0,"duracao":1,"delta":0.8,
                "tipo_choque":"negativo",
                "imoveis":[{"valor":1_000_000,"vacancia":0.1,"custo_manutencao":200},
                           {"valor":2_000_000,"vacancia":0.2,"custo_manutencao":500}],
            }
            estado_sim["gerador"] = simular_mercado_streaming(
                parametros_sistema=PARAMETROS_SISTEMA,
                num_dias=n_dias or 100, overrides=overrides, seed=None)
            return False, vel, False

        elif triggered == "btn-pausar":
            estado_sim["pausado"] = True
            return True, vel, True

        elif triggered == "btn-retomar":
            estado_sim["pausado"] = False
            estado_sim["rodando"] = True
            return False, vel, False

        elif triggered == "btn-parar":
            estado_sim.update({"rodando":False,"pausado":False,"gerador":None})
            return True, vel, False

    return True, vel, False


# ── Callback tick ──────────────────────────────────────────────────────────────
@app.callback(
    Output("store-tick", "data"),
    Input("intervalo",   "n_intervals"),
    State("store-tick",  "data"),
    prevent_initial_call=True,
)
def tick(n, tick_atual):
    global estado_sim
    with lock:
        if not estado_sim["rodando"] or estado_sim["pausado"]:
            return no_update
        gerador = estado_sim.get("gerador")
        if gerador is None:
            return no_update
        try:
            estado = next(gerador)
        except StopIteration:
            estado_sim.update({"rodando":False,"concluido":True,"gerador":None})
            return tick_atual + 1

        dia         = estado["dia"]
        preco_atual = estado["preco"]
        mercado     = estado["mercado"]
        dividendo   = estado.get("dividendo_dia", 0.0)
        usuario     = estado_sim["usuario"]

        estado_sim["historico_precos"] = estado["historico_precos"]
        estado_sim["sentimentos"].append(estado["sentimento"])
        estado_sim["dia_atual"]   = dia
        estado_sim["mercado_obj"] = mercado

        # Dividendos
        if dividendo > 0 and usuario["cotas"] > 0:
            val = dividendo * usuario["cotas"]
            usuario["caixa"] += val
            usuario["total_dividendos"] += val
            pat = usuario["caixa"] + usuario["cotas"] * preco_atual
            estado_sim["hist_dividendos"].append({"dia":dia,"valor":val,"patrimonio":pat})

        # Choques notícias
        for c in estado_sim.get("cfg_not", []):
            if c.get("dia") and dia == c["dia"]:
                for ag in mercado.agentes:
                    ag.aplicar_choque(tipo_choque=c["tipo"], intensidade=c["intensidade"],
                                      duracao=c["duracao"], delta=c["delta"])

        # Choques econômicos
        for c in estado_sim.get("cfg_econ", []):
            if c.get("dia") and dia == c["dia"]:
                mercado.banco_central.expectativa_inflacao = c["inflacao"]
                mercado.banco_central.premio_risco         = c["premio"]

        # Choques FII
        for c in estado_sim.get("cfg_fii", []):
            if c.get("dia") and dia == c["dia"]:
                for im in mercado.fii.imoveis:
                    im.vacancia         *= (1 + c["vac"]   / 100)
                    im.custo_manutencao *= (1 + c["custo"] / 100)

        estado_sim["choques_noticias"] = [{"dia":c["dia"],"tipo":c["tipo"]} for c in estado_sim.get("cfg_not",[]) if c.get("dia")]
        estado_sim["choques_econ"]     = [{"dia":c["dia"]} for c in estado_sim.get("cfg_econ",[]) if c.get("dia")]
        estado_sim["choques_fii"]      = [{"dia":c["dia"]} for c in estado_sim.get("cfg_fii",[])  if c.get("dia")]

        pat = usuario["caixa"] + usuario["cotas"] * preco_atual
        estado_sim["hist_patrimonio"].append(pat)
        estado_sim["usuario"] = usuario

    return tick_atual + 1


# ── Callback atualizar UI ──────────────────────────────────────────────────────
@app.callback(
    Output("grafico-mercado",    "figure"),
    Output("grafico-patrimonio", "figure"),
    Output("met-preco",  "children"), Output("met-sent",   "children"),
    Output("met-vol",    "children"), Output("met-dia",    "children"),
    Output("inv-caixa",  "children"), Output("inv-cotas",  "children"),
    Output("inv-patr",   "children"), Output("inv-div",    "children"),
    Output("div-status", "children"), Output("prog-bar",   "value"),
    Output("div-acao",   "style"),    Output("div-preco-pause", "children"),
    Output("inp-pc",     "value"),    Output("inp-pv",     "value"),
    Input("store-tick",   "data"),
    Input("store-pausado","data"),
    prevent_initial_call=True,
)
def atualizar_ui(tick, pausado_store):
    with lock:
        hp        = list(estado_sim["historico_precos"])
        sen       = list(estado_sim["sentimentos"])
        dia       = estado_sim["dia_atual"]
        num_dias  = estado_sim["num_dias"]
        window    = estado_sim["window_vol"]
        usuario   = dict(estado_sim["usuario"])
        pausado   = estado_sim["pausado"]
        concluido = estado_sim["concluido"]
        hist_pat  = list(estado_sim["hist_patrimonio"])
        hist_div  = list(estado_sim["hist_dividendos"])
        ch_not    = list(estado_sim["choques_noticias"])
        ch_econ   = list(estado_sim["choques_econ"])
        ch_fii    = list(estado_sim["choques_fii"])

    vazio = (go.Figure(), go.Figure(),
             "—","—","—","—","R$ —","—","R$ —","R$ —",
             "Aguardando...", 0, {"display":"none"}, "", 0, 0)

    if not hp or len(hp) <= 252:
        return vazio

    preco  = float(hp[-1])
    patr   = usuario["caixa"] + usuario["cotas"] * preco
    vol_s  = calc_vol_rolante(hp[252:], window)
    vol_v  = vol_s[~np.isnan(vol_s)]
    vol_c  = float(vol_v[-1]) if len(vol_v) > 0 else 0.0
    sent   = float(sen[-1]) if sen else 0.0
    prog   = min(100, int(dia / num_dias * 100))

    fig_m = build_fig_mercado(hp, sen, dia, window, ch_not, ch_econ, ch_fii)
    fig_p = build_fig_patrimonio(hist_pat, hist_div, dia)

    if concluido:
        status = dbc.Alert("✓ Simulação concluída!", color="success", className="mb-0 py-1")
    elif pausado:
        status = dbc.Alert(f"⏸ Pausado no dia {dia}", color="warning", className="mb-0 py-1")
    else:
        status = dbc.Alert(f"Simulando... dia {dia}/{num_dias}", color="info", className="mb-0 py-1")

    acao_style = {"display":"block"} if pausado else {"display":"none"}
    preco_txt  = html.P(f"💹 R$ {preco:,.2f}", className="fw-bold text-warning") if pausado else ""

    return (fig_m, fig_p,
            f"R$ {preco:,.2f}", f"{sent:+.3f}", f"{vol_c:.2%}", f"{dia}/{num_dias}",
            f"R$ {usuario['caixa']:,.2f}", f"{usuario['cotas']}",
            f"R$ {patr:,.2f}", f"R$ {usuario['total_dividendos']:,.2f}",
            status, prog, acao_style, preco_txt,
            round(preco, 2), round(preco, 2))


# ── Callback compra/venda ──────────────────────────────────────────────────────
@app.callback(
    Output("store-acao-msg",  "data"),
    Output("store-venda-msg", "data"),
    Input("btn-comprar", "n_clicks"), Input("btn-vender", "n_clicks"),
    State("inp-pc","value"), State("inp-qc","value"),
    State("inp-pv","value"), State("inp-qv","value"),
    prevent_initial_call=True,
)
def acao(nc, nv, pc, qc, pv, qv):
    global estado_sim
    with lock:
        u = estado_sim["usuario"]
        if ctx.triggered_id == "btn-comprar":
            custo = (pc or 0)*(qc or 0)
            if (qc or 0) > 0 and u["caixa"] >= custo:
                u["caixa"] -= custo; u["cotas"] += (qc or 0)
                estado_sim["usuario"] = u
                return f"✓ Comprou {qc} cotas a R${pc:,.2f}", ""
            return "❌ Caixa insuficiente.", ""
        elif ctx.triggered_id == "btn-vender":
            rec = (pv or 0)*(qv or 0)
            if (qv or 0) > 0 and u["cotas"] >= (qv or 0):
                u["caixa"] += rec; u["cotas"] -= (qv or 0)
                estado_sim["usuario"] = u
                return "", f"✓ Vendeu {qv} cotas a R${pv:,.2f}"
            return "", "❌ Cotas insuficientes."
    return "", ""


@app.callback(Output("div-msg-acao",  "children"), Input("store-acao-msg",  "data"))
def msg_compra(msg):
    if not msg: return ""
    color = "success" if msg.startswith("✓") else "danger"
    return dbc.Alert(msg, color=color, className="py-1 small mt-1")

@app.callback(Output("div-msg-venda", "children"), Input("store-venda-msg", "data"))
def msg_venda(msg):
    if not msg: return ""
    color = "success" if msg.startswith("✓") else "danger"
    return dbc.Alert(msg, color=color, className="py-1 small mt-1")


# ── Callback fatos estilizados ─────────────────────────────────────────────────
@app.callback(
    Output("div-fatos", "children"),
    Input("btn-fatos",  "n_clicks"),
    State("sl-lags",  "value"),
    State("sl-alpha", "value"),
    State("dd-cor",   "value"),
    prevent_initial_call=True,
)
def fatos(n, lags, alpha, cor):
    with lock:
        hp = list(estado_sim["historico_precos"])

    if not hp or len(hp) <= 252:
        return dbc.Alert("Rode a simulação primeiro.", color="warning")

    lr      = np.diff(np.log(np.array(hp[252:])))
    n_dias  = len(lr)
    lags_pl = min(40, max(10, int(n_dias * 0.10)))

    try:
        df_ifix = carregar_ifix()
        lr_ifix = np.diff(np.log(df_ifix['IFIX'].values[-(n_dias+1):]))
    except Exception:
        lr_ifix = None

    def row2(titulo, fs, fi=None, cap_s="", cap_i=""):
        cols = [dbc.Col([html.P("Simulado", className="small text-muted"),
                         html.Img(src=fig_b64(fs), style={"width":"100%"}),
                         html.P(cap_s, className="small text-muted")])]
        if fi is not None:
            cols.append(dbc.Col([html.P("IFIX", className="small text-muted"),
                                  html.Img(src=fig_b64(fi), style={"width":"100%"}),
                                  html.P(cap_i, className="small text-muted")]))
        return html.Div([html.Hr(), html.H5(titulo), dbc.Row(cols)])

    def metricas(curt):
        return dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.P("Diário",  className="small mb-0"), html.H6(f"{curt[0]:.4f}")]))),
            dbc.Col(dbc.Card(dbc.CardBody([html.P("Semanal", className="small mb-0"), html.H6(f"{curt[1]:.4f}")]))),
            dbc.Col(dbc.Card(dbc.CardBody([html.P("Mensal",  className="small mb-0"), html.H6(f"{curt[2]:.4f}")]))),
        ])

    items = []

    # 01 Intermitência
    fi = plot_intermitencia(lr_ifix, "IFIX", "tab:gray") if lr_ifix is not None else None
    items.append(row2("01 — Intermitência", plot_intermitencia(lr, "Simulado", cor), fi))

    # 02 ACF
    fi = plot_acf_retornos(lr_ifix, "ACF IFIX", lags, alpha, "tab:gray") if lr_ifix is not None else None
    items.append(row2("02 — Autocorrelação", plot_acf_retornos(lr, "ACF Simulado", lags, alpha, cor), fi))

    # 03 Power law
    fs, ps = plot_power_law(lr, "Power law Simulado", lags_pl, alpha, cor)
    cap_s  = f"a={ps[0]:.3f} b={ps[1]:.3f} lags={lags_pl}" if not np.isnan(ps).any() else ""
    if lr_ifix is not None:
        fi, pi = plot_power_law(lr_ifix, "Power law IFIX", lags_pl, alpha, "tab:gray")
        cap_i  = f"a={pi[0]:.3f} b={pi[1]:.3f}" if not np.isnan(pi).any() else ""
    else:
        fi, cap_i = None, ""
    items.append(row2("03 — Decaimento em lei de potência", fs, fi, cap_s, cap_i))

    # 04 Gaussianidade
    fs, cs = plot_gaussianidade_agregacional(lr, [1,5,21], ["Diário","Semanal","Mensal"], "Gauss. Simulado", cor)
    if lr_ifix is not None:
        fi, ci = plot_gaussianidade_agregacional(lr_ifix, [1,5,21], ["Diário","Semanal","Mensal"], "Gauss. IFIX", "tab:gray")
    else:
        fi, ci = None, None
    items.append(html.Div([html.Hr(), html.H5("04 — Gaussianidade agregacional"),
        dbc.Row([
            dbc.Col([html.P("Simulado", className="small text-muted"),
                     html.Img(src=fig_b64(fs), style={"width":"100%"}), metricas(cs)]),
            dbc.Col([html.P("IFIX", className="small text-muted"),
                     html.Img(src=fig_b64(fi), style={"width":"100%"}), metricas(ci)] if fi else []),
        ])
    ]))

    # 05 GARCH
    try:
        rs = calcular_residuos_garch(lr, [1,5,21])
        fs, cgs = plot_gaussianidade_agregacional(rs[0], [1,5,21], ["Diário","Semanal","Mensal"], "GARCH Simulado", cor)
        if lr_ifix is not None:
            ri = calcular_residuos_garch(lr_ifix, [1,5,21])
            fi, cgi = plot_gaussianidade_agregacional(ri[0], [1,5,21], ["Diário","Semanal","Mensal"], "GARCH IFIX", "tab:gray")
        else:
            fi, cgi = None, None
        items.append(html.Div([html.Hr(), html.H5("05 — Caudas pesadas (resíduos GARCH)"),
            dbc.Row([
                dbc.Col([html.P("Simulado", className="small text-muted"),
                         html.Img(src=fig_b64(fs), style={"width":"100%"}), metricas(cgs)]),
                dbc.Col([html.P("IFIX", className="small text-muted"),
                         html.Img(src=fig_b64(fi), style={"width":"100%"}), metricas(cgi)] if fi else []),
            ])
        ]))
    except Exception as e:
        items.append(dbc.Alert(f"Erro GARCH: {e}", color="warning"))

    return items


if __name__ == "__main__":
    app.run(debug=False, port=8050)