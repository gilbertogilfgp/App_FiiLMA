import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from src.simulacao import simular_mercado_streaming
from funcoes.fatos_estilizados import (
    plot_intermitencia,
    plot_acf_retornos,
    plot_power_law,
    plot_gaussianidade_agregacional,
    calcular_residuos_garch,
)

st.set_page_config(
    page_title="FiiLMA",
    page_icon="🏢",
    layout="wide",
)

PARAMETROS_SISTEMA = [0.7074504, 0.87356055, 0.67826246, 0.2438783, 0.08593365]
PRECO_BASE = (3_000_000 + 50_000) / 100_000 * 0.65

st.title("🏢 FiiLMA: Laboratório de Mercados Artificiais de Fiis")

# ── Inicializa session_state ───────────────────────────────────────────────────
defaults = {
    "lr_final": None, "hp_final": None, "sent_final": None,
    "rodando": False, "pausado": False,
    "gerador": None, "mercado_obj": None,
    "historico_precos": [], "sentimentos": [],
    "hist_patrimonio": [], "hist_dividendos_usuario": [],
    "usuario": None, "dia_atual": 0,
    "concluido": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

aba_sim, aba_fatos = st.tabs(["🔄 Simulação", "📊 Fatos Estilizados"])

# ══════════════════════════════════════════════════════════════════════════════
# ABA 1 — SIMULAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
with aba_sim:

    col_btn, col_dias, col_preco, col_slider, col_vol = st.columns([1, 2, 2, 2, 2])
    with col_btn:
        rodar    = st.button("▶  Rodar",   use_container_width=True)
        pausar   = st.button("⏸  Pausar",  use_container_width=True)
        retomar  = st.button("▶  Retomar", use_container_width=True)
        parar    = st.button("⏹  Parar",   use_container_width=True)
    with col_dias:
        NUM_DIAS = st.slider("Dias de simulação", min_value=50, max_value=500, value=100, step=10)
    with col_preco:
        preco_ref = st.number_input("Preço de referência (R$)",
                                    min_value=1.0, max_value=10000.0,
                                    value=20.15, step=0.5, format="%.2f")
        escala = preco_ref / PRECO_BASE
    with col_slider:
        atualizar_cada = st.slider("Atualizar a cada N dias",
                                   min_value=5, max_value=50, value=10, step=5)
    with col_vol:
        window_vol = st.slider("Janela volatilidade (dias)",
                               min_value=5, max_value=60, value=20, step=5)

    st.markdown("---")

    with st.expander("🏦 Parâmetros econômicos iniciais", expanded=False):
        vi_a, vi_b, vi_c = st.columns(3)
        selic_inicial    = vi_a.slider("Selic inicial (a.a.)",      0.01, 0.30, 0.15, step=0.01, format="%.2f")
        inflacao_inicial = vi_b.slider("Inflação inicial esperada", 0.01, 0.20, 0.07, step=0.01, format="%.2f")
        premio_inicial   = vi_c.slider("Prêmio de risco inicial",  0.01, 0.20, 0.08, step=0.01, format="%.2f")

    with st.expander("🏠 Choques nos FIIs", expanded=False):
        num_choques_fii = st.radio("Número de choques", [0, 1, 2, 3], horizontal=True, key="num_ch_fii")
        choques_fii = []
        defaults_fii = [
            {"dia": 30,  "vac": 50,  "custo": 20},
            {"dia": 100, "vac": -30, "custo": 50},
            {"dia": 200, "vac": 20,  "custo": -20},
        ]
        for i in range(num_choques_fii):
            d = defaults_fii[i]
            st.markdown(f"**Choque FII {i+1}**")
            ca, cb, cc = st.columns(3)
            choques_fii.append({
                "dia":         ca.number_input("Dia", min_value=1, max_value=NUM_DIAS, value=d["dia"], key=f"fii{i}_dia"),
                "choque_vac":  cb.slider("Choque vacância (%)", -80, 300, d["vac"], step=5, format="%d%%", key=f"fii{i}_vac"),
                "choque_custo":cc.slider("Choque custo (%)", -80, 300, d["custo"], step=5, format="%d%%", key=f"fii{i}_custo"),
            })

    with st.expander("📰 Choques de notícias", expanded=False):
        num_choques_not = st.radio("Número de choques", [0, 1, 2, 3], horizontal=True, key="num_ch_not")
        choques_noticias = []
        defaults_not = [
            {"dia": 30,  "tipo": "negativo", "int": 0.7, "dur": 2,  "delta": 0.8},
            {"dia": 50,  "tipo": "positivo", "int": 0.5, "dur": 3,  "delta": 0.8},
            {"dia": 60,  "tipo": "negativo", "int": 0.4, "dur": 5,  "delta": 0.7},
        ]
        for i in range(num_choques_not):
            d = defaults_not[i]
            st.markdown(f"**Choque {i+1}**")
            ca, cb, cc, cd, ce = st.columns(5)
            choques_noticias.append({
                "dia":         ca.number_input("Dia", min_value=1, max_value=NUM_DIAS, value=d["dia"], key=f"not{i}_dia"),
                "tipo":        cb.selectbox("Tipo", ["negativo", "positivo"], index=0 if d["tipo"]=="negativo" else 1, key=f"not{i}_tipo"),
                "intensidade": cc.slider("Intensidade", 0.1, 1.0, d["int"], step=0.05, key=f"not{i}_int"),
                "duracao":     cd.slider("Duração (dias)", 1, 30, d["dur"], key=f"not{i}_dur"),
                "delta":       ce.slider("Delta", 0.1, 1.0, d["delta"], step=0.05, key=f"not{i}_delta"),
            })
        st.markdown("---")
        prob_choque_diario = st.slider("Prob. choque aleatório diário", 0.0, 0.2, 0.025, step=0.005, format="%.3f")

    with st.expander("📊 Choques econômicos", expanded=False):
        num_choques_econ = st.radio("Número de choques", [0, 1, 2, 3], horizontal=True, key="num_ch_econ")
        choques_economicos = []
        defaults_econ = [
            {"dia": 50,  "inf": 0.05, "premio": 0.06},
            {"dia": 150, "inf": 0.06, "premio": 0.07},
            {"dia": 300, "inf": 0.07, "premio": 0.08},
        ]
        for i in range(num_choques_econ):
            d = defaults_econ[i]
            st.markdown(f"**Choque econômico {i+1}**")
            ca, cb, cc = st.columns(3)
            choques_economicos.append({
                "dia":     ca.number_input("Dia", min_value=1, max_value=NUM_DIAS, value=d["dia"], key=f"econ{i}_dia"),
                "inflacao":cb.slider("Nova inflação esperada", 0.01, 0.20, d["inf"], step=0.01, format="%.2f", key=f"econ{i}_inf"),
                "premio":  cc.slider("Novo prêmio de risco", 0.01, 0.20, d["premio"], step=0.01, format="%.2f", key=f"econ{i}_premio"),
            })

    st.markdown("---")

    ph_status   = st.empty()
    ph_progress = st.empty()

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    ph_preco      = col_m1.empty()
    ph_sentimento = col_m2.empty()
    ph_vol_metric = col_m3.empty()
    ph_dia        = col_m4.empty()
    ph_grafico    = st.empty()

    st.markdown("---")
    st.subheader("👤 Investidor")
    col_u1, col_u2, col_u3, col_u4 = st.columns(4)
    ph_u_caixa      = col_u1.empty()
    ph_u_cotas      = col_u2.empty()
    ph_u_patrimonio = col_u3.empty()
    ph_u_dividendos = col_u4.empty()
    ph_painel_pause = st.empty()
    ph_grafico_usuario = st.empty()

    if parar:
        st.session_state.rodando  = False
        st.session_state.pausado  = False
        st.session_state.gerador  = None

    # ── Helpers ────────────────────────────────────────────────────────────────
    def calc_vol_rolante(precos, window):
        arr = np.array(precos)
        if len(arr) < window + 2:
            return np.array([np.nan])
        log_ret  = np.diff(np.log(arr))
        vol_plot = np.full(len(log_ret), np.nan)
        for i in range(window, len(log_ret)):
            vol_plot[i] = np.std(log_ret[i - window:i]) * np.sqrt(252)
        return vol_plot

    def render_grafico(historico_precos, sentimentos, dia_atual, window,
                       choques_not, choques_econ, choques_fii_lista):
        N_MEM  = 252
        n_sim  = len(historico_precos) - N_MEM
        n_sent = len(sentimentos)
        dias_mem  = list(range(-N_MEM, 1))
        dias_sim  = list(range(0, n_sim + 1))
        dias_sent = list(range(1, n_sent + 1))
        vol       = calc_vol_rolante(historico_precos, window)
        dias_vol  = list(range(-N_MEM, -N_MEM + len(vol)))

        fig = make_subplots(rows=3, cols=1, shared_xaxes=False,
                            vertical_spacing=0.07, row_heights=[0.40, 0.30, 0.30],
                            subplot_titles=(f"Preço da cota  (dia {dia_atual})",
                                            "Sentimento médio dos agentes",
                                            f"Volatilidade rolante (janela {window} dias)"))

        fig.add_trace(go.Scatter(x=dias_mem, y=historico_precos[:N_MEM+1], mode="lines",
                                 line=dict(color="#adb5bd", width=1.0),
                                 hovertemplate="Dia %{x}<br>Preço: R$ %{y:.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dias_sim, y=historico_precos[N_MEM:], mode="lines",
                                 line=dict(color="#1a1a2e", width=1.5),
                                 hovertemplate="Dia %{x}<br>Preço: R$ %{y:.2f}<extra></extra>"), row=1, col=1)
        fig.add_vline(x=0, line_dash="dot", line_color="#888780", line_width=1.2, opacity=0.8, row=1, col=1)
        fig.add_annotation(x=2, xref="x1", y=max(historico_precos)*1.005, yref="y1",
                           text="início simulação", showarrow=False,
                           font=dict(size=9, color="#888780"), xanchor="left")

        if n_sent > 0:
            sent_arr = np.array(sentimentos)
            fig.add_trace(go.Scatter(x=dias_sent, y=list(np.where(sent_arr>=0, sent_arr, 0)),
                                     mode="lines", line=dict(width=0), fill="tozeroy",
                                     fillcolor="rgba(69,123,157,0.15)", showlegend=False, hoverinfo="skip"), row=2, col=1)
            fig.add_trace(go.Scatter(x=dias_sent, y=list(np.where(sent_arr<0, sent_arr, 0)),
                                     mode="lines", line=dict(width=0), fill="tozeroy",
                                     fillcolor="rgba(230,57,70,0.15)", showlegend=False, hoverinfo="skip"), row=2, col=1)
            fig.add_trace(go.Scatter(x=dias_sent, y=list(sent_arr), mode="lines",
                                     line=dict(color="#457b9d", width=1.2),
                                     hovertemplate="Dia %{x}<br>Sentimento: %{y:.3f}<extra></extra>"), row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#adb5bd", line_width=0.8, row=2, col=1)
        fig.update_xaxes(range=[-N_MEM-5, n_sim+5], row=2, col=1)

        if len(vol) > 1:
            fig.add_trace(go.Scatter(x=dias_vol, y=list(vol), mode="lines",
                                     line=dict(color="#e76f51", width=1.2), fill="tozeroy",
                                     fillcolor="rgba(231,111,81,0.08)",
                                     hovertemplate="Dia %{x}<br>Vol.: %{y:.2%}<extra></extra>"), row=3, col=1)
            fig.add_vline(x=0, line_dash="dot", line_color="#888780", line_width=1.0, opacity=0.6, row=3, col=1)
            fig.update_xaxes(range=[-N_MEM-5, n_sim+5], row=3, col=1)

        for c in choques_not:
            if c["dia"] <= n_sim:
                cor = "#e63946" if c["tipo"]=="negativo" else "#2a9d8f"
                for row in [1,2,3]:
                    fig.add_vline(x=c["dia"], line_dash="dash", line_color=cor, line_width=1, opacity=0.7, row=row, col=1)
                fig.add_annotation(x=c["dia"]+1, y=1.01, yref="paper",
                                   text=f"📰 {c['tipo'][:3]}.", showarrow=False,
                                   font=dict(size=9, color=cor), xanchor="left")
        for alt in choques_econ:
            if alt["dia"] <= n_sim:
                for row in [1,2,3]:
                    fig.add_vline(x=alt["dia"], line_dash="dot", line_color="#e9c46a", line_width=1.5, opacity=0.9, row=row, col=1)
                fig.add_annotation(x=alt["dia"]+1, y=0.96, yref="paper",
                                   text="📊 econ.", showarrow=False,
                                   font=dict(size=9, color="#c9a84c"), xanchor="left")
        for c in choques_fii_lista:
            if c["dia"] <= n_sim:
                for row in [1,2,3]:
                    fig.add_vline(x=c["dia"], line_dash="dashdot", line_color="#7f77dd", line_width=1, opacity=0.7, row=row, col=1)
                fig.add_annotation(x=c["dia"]+1, y=0.91, yref="paper",
                                   text="🏠 FII", showarrow=False,
                                   font=dict(size=9, color="#7f77dd"), xanchor="left")

        fig.update_layout(height=680, showlegend=False,
                          margin=dict(l=50, r=30, t=60, b=40),
                          paper_bgcolor="white", plot_bgcolor="white",
                          hovermode="x unified")
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.07)", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.07)", zeroline=False)
        fig.update_yaxes(range=[-1.15, 1.15], row=2, col=1)
        fig.update_yaxes(title_text="Preço (R$)",   title_font_size=10, row=1, col=1)
        fig.update_yaxes(title_text="Sentimento",   title_font_size=10, row=2, col=1)
        fig.update_yaxes(title_text="Volatilidade", title_font_size=10, row=3, col=1)
        fig.update_xaxes(title_text="Dias",         title_font_size=10, row=3, col=1)
        ph_grafico.plotly_chart(fig, use_container_width=True)

    def render_grafico_usuario(hist_patrimonio, hist_dividendos, dia_atual):
        if len(hist_patrimonio) == 0:
            return
        
        arr = np.array(hist_patrimonio)
        y_min = arr.min() * 0.995
        y_max = arr.max() * 1.005

        dias = list(range(1, len(hist_patrimonio) + 1))
        fig  = go.Figure()
        fig.add_trace(go.Scatter(
            x=dias, y=hist_patrimonio, mode="lines", name="Patrimônio",
            line=dict(color="#2a9d8f", width=1.5),
            hovertemplate="Dia %{x}<br>Patrimônio: R$ %{y:,.2f}<extra></extra>"
        ))
        for d in hist_dividendos:
            fig.add_annotation(
                x=d["dia"], y=d["patrimonio"],
                text=f"💰 R${d['valor']:.2f}",
                showarrow=True, arrowhead=2, arrowcolor="#e9c46a",
                font=dict(size=9, color="#c9a84c"),
                bgcolor="rgba(255,255,255,0.8)"
            )
        fig.update_layout(
            height=250, showlegend=False,
            margin=dict(l=50, r=30, t=30, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
            title=dict(text=f"Patrimônio do investidor  (dia {dia_atual})",
                    font=dict(size=11), x=0)
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.07)",
                        title_text="Dias", title_font_size=10)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.07)",
                        title_text="R$", title_font_size=10,
                        range=[y_min, y_max])
        ph_grafico_usuario.plotly_chart(fig, use_container_width=True)

    def atualizar_metricas_usuario(usuario, preco_atual):
        patrimonio = usuario["caixa"] + usuario["cotas"] * preco_atual
        ph_u_caixa.metric("💵 Caixa",               f"R$ {usuario['caixa']:,.2f}")
        ph_u_cotas.metric("📦 Cotas",               f"{usuario['cotas']}")
        ph_u_patrimonio.metric("💼 Patrimônio",      f"R$ {patrimonio:,.2f}")
        ph_u_dividendos.metric("💰 Dividendos",      f"R$ {usuario['total_dividendos']:,.2f}")
        return patrimonio

    def executar_lote(gerador, n_dias, choques_noticias, choques_economicos,
                      choques_fii, atualizar_cada, window_vol):
        """Avança o gerador até pausar ou terminar."""
        historico_precos = st.session_state.historico_precos
        sentimentos      = st.session_state.sentimentos
        usuario          = st.session_state.usuario

        for estado in gerador:
            if not st.session_state.rodando:
                break
            if st.session_state.pausado:
                # Salva estado e para
                st.session_state.historico_precos = estado["historico_precos"]
                st.session_state.sentimentos      = sentimentos
                st.session_state.usuario          = usuario
                st.session_state.dia_atual        = estado["dia"]
                st.session_state.gerador          = gerador
                st.session_state.preco_pause      = estado["preco"]
                return False  # não concluiu

            dia              = estado["dia"]
            preco_atual      = estado["preco"]
            sentimento_atual = estado["sentimento"]
            mercado          = estado["mercado"]
            dividendo_dia    = estado.get("dividendo_dia", 0.0)
            usuario = st.session_state.usuario

            historico_precos = estado["historico_precos"]
            sentimentos.append(sentimento_atual)

            # Dividendos ao usuário
            if dividendo_dia > 0 and usuario["cotas"] > 0:
                valor_div = dividendo_dia * usuario["cotas"]
                usuario["caixa"]           += valor_div
                usuario["total_dividendos"] += valor_div
                patrimonio = usuario["caixa"] + usuario["cotas"] * preco_atual
                st.session_state.hist_dividendos_usuario.append({
                    "dia": dia, "valor": valor_div, "patrimonio": patrimonio
                })

            # Choques adicionais
            for choque in choques_noticias[1:]:
                if dia == choque["dia"]:
                    for agente in mercado.agentes:
                        agente.aplicar_choque(tipo_choque=choque["tipo"],
                                              intensidade=choque["intensidade"],
                                              duracao=choque["duracao"],
                                              delta=choque["delta"])
            for alt in choques_economicos:
                if dia == alt["dia"]:
                    mercado.banco_central.expectativa_inflacao = alt["inflacao"]
                    mercado.banco_central.premio_risco         = alt["premio"]
            for c in choques_fii:
                if dia == c["dia"]:
                    for imovel in mercado.fii.imoveis:
                        imovel.vacancia         *= (1 + c["choque_vac"]   / 100)
                        imovel.custo_manutencao *= (1 + c["choque_custo"] / 100)

            patrimonio = usuario["caixa"] + usuario["cotas"] * preco_atual
            st.session_state.hist_patrimonio.append(patrimonio)
            usuario = st.session_state.usuario

            if dia % atualizar_cada == 0 or dia == n_dias:
                vol_s = calc_vol_rolante(historico_precos[252:], window_vol)
                vol_v = vol_s[~np.isnan(vol_s)]
                vol_c = vol_v[-1] if len(vol_v) > 0 else 0.0

                ph_progress.progress(dia / n_dias, text=f"Dia {dia} / {n_dias}")
                ph_preco.metric("Preço",             f"R$ {preco_atual:,.2f}")
                ph_sentimento.metric("Sentimento",   f"{sentimento_atual:+.3f}")
                ph_vol_metric.metric("Volatilidade", f"{vol_c:.2%}")
                ph_dia.metric("Dia",                 f"{dia} / {n_dias}")

                atualizar_metricas_usuario(usuario, preco_atual)
                render_grafico(historico_precos, sentimentos, dia,
                               window_vol, choques_noticias, choques_economicos, choques_fii)
                render_grafico_usuario(st.session_state.hist_patrimonio,
                                       st.session_state.hist_dividendos_usuario, dia)

        # Concluiu
        st.session_state.historico_precos = historico_precos
        st.session_state.sentimentos      = sentimentos
        st.session_state.usuario          = usuario
        st.session_state.dia_atual        = n_dias
        st.session_state.gerador          = None
        return True  # concluiu

    # ── Botão RODAR ────────────────────────────────────────────────────────────
    if rodar:
        st.session_state.rodando          = True
        st.session_state.pausado          = False
        st.session_state.concluido        = False
        st.session_state.historico_precos = []
        st.session_state.sentimentos      = []
        st.session_state.hist_patrimonio  = []
        st.session_state.hist_dividendos_usuario = []
        st.session_state.usuario = {
            "caixa": 10_000 * escala, "cotas": 100, "total_dividendos": 0.0
        }

        choque_principal = choques_noticias[0] if choques_noticias else None
        overrides = {
            "prob_choque_diario":   prob_choque_diario,
            "dia_choque":           choque_principal["dia"]         if choque_principal else -1,
            "tipo_choque":          choque_principal["tipo"]        if choque_principal else "negativo",
            "intensidade":          choque_principal["intensidade"] if choque_principal else 0.0,
            "duracao":              choque_principal["duracao"]     if choque_principal else 1,
            "delta":                choque_principal["delta"]       if choque_principal else 0.8,
            "imoveis": [
                {"valor": 1_000_000, "vacancia": 0.1, "custo_manutencao": 200},
                {"valor": 2_000_000, "vacancia": 0.2, "custo_manutencao": 500},
            ],
            "taxa_selic":           selic_inicial,
            "expectativa_inflacao": inflacao_inicial,
            "premio_risco":         premio_inicial,
            "escala":               escala,
        }

        ph_status.info("Inicializando o modelo...")
        gerador = simular_mercado_streaming(
            parametros_sistema=PARAMETROS_SISTEMA,
            num_dias=NUM_DIAS, overrides=overrides, seed=None)
        st.session_state.gerador = gerador
        ph_status.success("Simulando...")

        concluiu = executar_lote(gerador, NUM_DIAS, choques_noticias,
                                 choques_economicos, choques_fii,
                                 atualizar_cada, window_vol)
        if concluiu:
            st.session_state.concluido = True
            st.session_state.rodando   = False

    # ── Botão PAUSAR ───────────────────────────────────────────────────────────
    if pausar and st.session_state.rodando:
        st.session_state.pausado = True

    # ── Painel de pausa ────────────────────────────────────────────────────────
    if st.session_state.pausado and st.session_state.gerador is not None:
        hp = st.session_state.historico_precos
        
        # Pega o último preço do histórico simulado (após os 252 de memória)
        preco_pause = float(hp[-1]) if len(hp) > 252 else 0.0
        usuario     = st.session_state.usuario or {}
        dia_pause   = st.session_state.dia_atual

        st.warning(f"⏸ Pausado no dia {dia_pause} — Preço atual da cota: R$ {preco_pause:,.2f}")

        with ph_painel_pause.container():
            st.markdown("### 💼 Sua ação")
            col_compra, col_venda = st.columns(2)

            with col_compra:
                preco_compra = st.number_input("Preço de compra (R$)", 
                                                min_value=0.0, value=preco_pause,
                                                step=0.01, format="%.2f", key="preco_compra")
                qtd_compra = st.number_input("Quantidade (cotas)", min_value=0,
                                            value=0, step=1, key="qtd_compra")
                custo_total = preco_compra * qtd_compra
                st.caption(f"Custo total: R$ {custo_total:,.2f} | Caixa: R$ {usuario.get('caixa',0):,.2f}")
                if st.button("✅ Comprar", key="btn_comprar"):
                    if qtd_compra > 0 and usuario["caixa"] >= custo_total:
                        usuario["caixa"]  -= custo_total
                        usuario["cotas"]  += qtd_compra
                        st.session_state.usuario = usuario
                        st.success(f"Comprou {qtd_compra} cotas a R$ {preco_compra:,.2f} — Total: R$ {custo_total:,.2f}")
                    else:
                        st.error("Caixa insuficiente ou quantidade inválida.")

            with col_venda:
                preco_venda = st.number_input("Preço de venda (R$)",
                                            min_value=0.0, value=preco_pause,
                                            step=0.01, format="%.2f", key="preco_venda")
                qtd_venda = st.number_input("Quantidade (cotas)", min_value=0,
                                            max_value=usuario.get("cotas", 0),
                                            value=0, step=1, key="qtd_venda")
                receita_total = preco_venda * qtd_venda
                st.caption(f"Receita total: R$ {receita_total:,.2f} | Cotas: {usuario.get('cotas',0)}")
                if st.button("💸 Vender", key="btn_vender"):
                    if qtd_venda > 0 and usuario["cotas"] >= qtd_venda:
                        usuario["caixa"]  += receita_total
                        usuario["cotas"]  -= qtd_venda
                        st.session_state.usuario = usuario
                        st.success(f"Vendeu {qtd_venda} cotas a R$ {preco_venda:,.2f} — Total: R$ {receita_total:,.2f}")
                    else:
                        st.error("Cotas insuficientes ou quantidade inválida.")

    # ── Botão RETOMAR ──────────────────────────────────────────────────────────
    if retomar and st.session_state.pausado and st.session_state.gerador is not None:
        st.session_state.pausado = False
        ph_painel_pause.empty()
        ph_status.success("Retomando simulação...")

        concluiu = executar_lote(
            st.session_state.gerador, NUM_DIAS,
            choques_noticias, choques_economicos, choques_fii,
            atualizar_cada, window_vol
        )
        if concluiu:
            st.session_state.concluido = True
            st.session_state.rodando   = False

    # ── Conclusão ──────────────────────────────────────────────────────────────
    if st.session_state.concluido:
        hp  = st.session_state.historico_precos
        sen = st.session_state.sentimentos
        if hp and len(hp) > 252:
            log_ret = np.diff(np.log(np.array(hp[252:])))
            st.session_state.lr_final   = log_ret
            st.session_state.hp_final   = hp[252:]
            st.session_state.sent_final = sen

            ph_progress.progress(1.0, text="Simulação concluída ✓")
            ph_status.success(
                f"Concluída ✓  |  Preço final: R$ {hp[-1]:,.2f}"
                f"  |  Sentimento: {sen[-1]:+.3f}" if sen else "")

            with st.expander("📥 Exportar resultados"):
                n_sim   = len(hp) - 252
                vol_exp = calc_vol_rolante(hp[252:], window_vol)
                df = pd.DataFrame({
                    "dia":          np.arange(n_sim),
                    "preco":        hp[252:],
                    "log_return":   np.concatenate([[np.nan], log_ret]),
                    "volatilidade": np.concatenate([[np.nan], vol_exp]),
                    "sentimento":   [np.nan]*(n_sim-len(sen)) + sen,
                })
                st.dataframe(df.tail(20), use_container_width=True)
                st.download_button("⬇ Baixar CSV",
                                   data=df.to_csv(index=False).encode("utf-8"),
                                   file_name="simulacao_fii.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# ABA 2 — FATOS ESTILIZADOS
# ══════════════════════════════════════════════════════════════════════════════
with aba_fatos:
    st.header("Fatos Estilizados")

    if st.session_state.get("lr_final") is None:
        st.info("Rode a simulação na aba **Simulação** para visualizar os fatos estilizados.")
    else:
        col_lags, col_alpha, col_cor = st.columns(3)
        lags_acf  = col_lags.slider("Lags ACF / Power law", 10, 60, 40, step=5)
        alpha_acf = col_alpha.slider("Alpha (bandas de confiança)", 0.01, 0.10, 0.05, step=0.01)
        cor_sim   = col_cor.selectbox("Cor", ["tab:blue", "tab:orange", "tab:green", "steelblue"])
        cor_ifix  = "tab:gray"

        calcular = st.button("📊 Calcular fatos estilizados")

        if calcular:
            log_ret = np.array(st.session_state.lr_final)
            n_dias  = len(log_ret)

            from src.ativos import carregar_ifix
            df_ifix      = carregar_ifix()
            precos_ifix  = df_ifix['IFIX'].values[-(n_dias + 1):]
            log_ret_ifix = np.diff(np.log(precos_ifix))
            lags_pl      = min(40, max(10, int(n_dias * 0.10)))

            st.caption(f"Comparando {n_dias} dias simulados vs {len(log_ret_ifix)} dias do IFIX")

            st.markdown("---")
            st.subheader("01 — Intermitência")
            col_s, col_i = st.columns(2)
            with col_s:
                st.caption("Simulado")
                fig = plot_intermitencia(log_ret, titulo="Retorno diário — Simulado", cor=cor_sim)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
            with col_i:
                st.caption("IFIX")
                fig = plot_intermitencia(log_ret_ifix, titulo="Retorno diário — IFIX", cor=cor_ifix)
                st.pyplot(fig, use_container_width=True); plt.close(fig)

            st.markdown("---")
            st.subheader("02 — Autocorrelação dos retornos")
            col_s, col_i = st.columns(2)
            with col_s:
                st.caption("Simulado")
                fig = plot_acf_retornos(log_ret, titulo="ACF — Simulado",
                                        lags=lags_acf, alpha=alpha_acf, cor=cor_sim)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
            with col_i:
                st.caption("IFIX")
                fig = plot_acf_retornos(log_ret_ifix, titulo="ACF — IFIX",
                                        lags=lags_acf, alpha=alpha_acf, cor=cor_ifix)
                st.pyplot(fig, use_container_width=True); plt.close(fig)

            st.markdown("---")
            st.subheader("03 — Decaimento em lei de potência")
            col_s, col_i = st.columns(2)
            with col_s:
                st.caption("Simulado")
                fig, popt_s = plot_power_law(log_ret, titulo="Power law — Simulado",
                                             lags=lags_pl, alpha=alpha_acf, cor=cor_sim)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                if not np.isnan(popt_s).any():
                    st.caption(f"a={popt_s[0]:.4f} | b={popt_s[1]:.4f} | lags={lags_pl}")
            with col_i:
                st.caption("IFIX")
                fig, popt_i = plot_power_law(log_ret_ifix, titulo="Power law — IFIX",
                                             lags=lags_pl, alpha=alpha_acf, cor=cor_ifix)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                if not np.isnan(popt_i).any():
                    st.caption(f"a={popt_i[0]:.4f} | b={popt_i[1]:.4f} | lags={lags_pl}")

            st.markdown("---")
            st.subheader("04 — Gaussianidade agregacional")
            col_s, col_i = st.columns(2)
            with col_s:
                st.caption("Simulado")
                fig, curtoses_s = plot_gaussianidade_agregacional(
                    log_ret, escalas=[1, 5, 21], labels_escalas=["Diário", "Semanal", "Mensal"],
                    titulo="Gaussianidade — Simulado", cor=cor_sim)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                c1, c2, c3 = st.columns(3)
                c1.metric("Curtose diária",  f"{curtoses_s[0]:.4f}")
                c2.metric("Curtose semanal", f"{curtoses_s[1]:.4f}")
                c3.metric("Curtose mensal",  f"{curtoses_s[2]:.4f}")
            with col_i:
                st.caption("IFIX")
                fig, curtoses_i = plot_gaussianidade_agregacional(
                    log_ret_ifix, escalas=[1, 5, 21], labels_escalas=["Diário", "Semanal", "Mensal"],
                    titulo="Gaussianidade — IFIX", cor=cor_ifix)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                c1, c2, c3 = st.columns(3)
                c1.metric("Curtose diária",  f"{curtoses_i[0]:.4f}")
                c2.metric("Curtose semanal", f"{curtoses_i[1]:.4f}")
                c3.metric("Curtose mensal",  f"{curtoses_i[2]:.4f}")

            st.markdown("---")
            st.subheader("05 — Caudas pesadas condicionais (resíduos GARCH)")
            with st.spinner("Ajustando modelos GARCH..."):
                residuos_s = calcular_residuos_garch(log_ret,      escalas=[1, 5, 21])
                residuos_i = calcular_residuos_garch(log_ret_ifix, escalas=[1, 5, 21])
            col_s, col_i = st.columns(2)
            with col_s:
                st.caption("Simulado")
                fig, curtoses_gs = plot_gaussianidade_agregacional(
                    residuos_s[0], escalas=[1, 5, 21], labels_escalas=["Diário", "Semanal", "Mensal"],
                    titulo="GARCH — Simulado", cor=cor_sim)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                c1, c2, c3 = st.columns(3)
                c1.metric("Curtose diária",  f"{curtoses_gs[0]:.4f}")
                c2.metric("Curtose semanal", f"{curtoses_gs[1]:.4f}")
                c3.metric("Curtose mensal",  f"{curtoses_gs[2]:.4f}")
            with col_i:
                st.caption("IFIX")
                fig, curtoses_gi = plot_gaussianidade_agregacional(
                    residuos_i[0], escalas=[1, 5, 21], labels_escalas=["Diário", "Semanal", "Mensal"],
                    titulo="GARCH — IFIX", cor=cor_ifix)
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                c1, c2, c3 = st.columns(3)
                c1.metric("Curtose diária",  f"{curtoses_gi[0]:.4f}")
                c2.metric("Curtose semanal", f"{curtoses_gi[1]:.4f}")
                c3.metric("Curtose mensal",  f"{curtoses_gi[2]:.4f}")