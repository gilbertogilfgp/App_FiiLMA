import random
import numpy as np
import matplotlib.pyplot as plt

from src.ativos import Imovel, FII
from src.agentes import Agente, gerar_literacia_financeira
from src.mercado import BancoCentral, Midia, Mercado


def calcular_sentimento_medio(agentes):
    return np.mean([agente.sentimento for agente in agentes])


def _construir_sim_params(parametros_sistema, num_dias, overrides=None):
    """
    Monta o dicionário sim_params canônico.
    overrides: dict opcional com valores que substituem os defaults
               (usado pelo Streamlit para passar os sliders).
    """

    p = overrides or {}

    escala = p.get("escala", 1.0)

    return {
        "num_dias": num_dias,
        "total_cota": 100_000,

        "fii": {
            "num_cotas": 100_000,
            "caixa_inicial": int(50_000 * escala),
            "dividendos_taxa": 0.95,
            "dividendos_caixa_taxa": 0.05,
            "investimento_fracao": 0.50,
            "memoria": True,
        },

        "imoveis": [
            {
                **im,
                "valor": im["valor"] * escala,
                "custo_manutencao": im.get("custo_manutencao", 200) * escala,
                "params": im.get("params", {"aluguel_factor": 0.005, "desvio_normal": 0.01})
            }
            for im in p.get("imoveis", [
                {"valor": 1_000_000, "vacancia": 0.1, "custo_manutencao": 200},
                {"valor": 2_000_000, "vacancia": 0.2, "custo_manutencao": 500},
            ])
        ],
        

        "num_agentes_pf": p.get("num_agentes_pf", 600),
        "prop_cota_agente": 0.6,
        "agente_pf": {
            "caixa_inicial": int(10_000 * escala),
            "cotas_iniciais_primeiro": 100,
            "cotas_iniciais_outros": 100,
            "num_vizinhos": 30,
            "expectativa_inflacao": 0.05,
            "expectativa_premio": 0.08,
            "params": {
                "window_chart": 21,
                "alpha_chart_short": 0.3,
                "alpha_chart_long": 0.1,
                "ruido_std": p.get("ruido_std", 0.1),
                "peso_retorno": 0.6,
                "peso_riqueza": 0.4,
            }
        },

        "num_agentes_pj": p.get("num_agentes_pj", 200),
        "prop_cota_agente_pj": 0.2,
        "agente_pj": {
            "caixa_inicial": int(10_000 * escala),
            "cotas_iniciais_primeiro": 100,
            "cotas_iniciais_outros": 100,
            "num_vizinhos": 30,
            "expectativa_inflacao": 0.05,
            "expectativa_premio": 0.08,
            "params": {
                "window_chart": 21,
                "alpha_chart_short": 0.3,
                "alpha_chart_long": 0.1,
                "ruido_std": p.get("ruido_std", 0.1) * 0.5,
                "peso_retorno": 0.3,
                "peso_riqueza": 0.7,
            }
        },

        "banco_central": {
            "taxa_selic":            p.get("taxa_selic", 0.15),
            "expectativa_inflacao":  p.get("expectativa_inflacao", 0.07),
            "premio_risco":          p.get("premio_risco", 0.08),
        },

        "midia": {
            "valor_inicial": 0,
            "sigma": 0.1,
            "valores_fixos": {},
        },

        "parametros_sentimento": {
            "a0":                        parametros_sistema[0],
            "b0":                        parametros_sistema[1],
            "c0":                        parametros_sistema[2],
            "beta":                      parametros_sistema[3],
            "peso_preco_esperado":       parametros_sistema[4],
            "ruido_std":                 p.get("ruido_std", 0.1),
            "sigma_midia":               p.get("sigma_midia", 0.8),
            "piso_prob_negociar":        0.1,
            "peso_sentimento_inflacao":  0.4,
            "peso_sentimento_expectativa": 0.4,
            "quantidade_compra_min":     1,
            "quantidade_compra_max":     30,
            # choque fixo
            "dia":                       p.get("dia_choque", 30),
            "tipo":                      p.get("tipo_choque", "negativo"),
            "intensidade":               p.get("intensidade", 0.7),
            "duracao":                   p.get("duracao", 2),
            "delta":                     p.get("delta", 0.8),
            "prob_choque_diario":        p.get("prob_choque_diario", 0.025),
        },

        "mercado": {
            "volatilidade_inicial": 0.1,
            "dividendos_frequencia": 21,
            "atualizacao_imoveis_frequencia": 126,
        },

        "order_book": {},

        "plot": {
            "window_volatilidade": 200,
        },
    }


def _inicializar(sim_params, seed=None):
    """Inicializa todos os objetos do modelo. Retorna (mercado, historico_inicial)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    fii = FII(num_cotas=sim_params["fii"]["num_cotas"],
              caixa=sim_params["fii"]["caixa_inicial"],
              params=sim_params["fii"])

    for ip in sim_params["imoveis"]:
        fii.adicionar_imovel(Imovel(
            valor=ip["valor"],
            vacancia=ip["vacancia"],
            custo_manutencao=ip["custo_manutencao"],
            params=ip.get("params")
        ))

    historia = fii.inicializar_historico(memoria=sim_params["fii"]["memoria"])
    fii.preco_cota = fii.historico_precos[-1]

    num_agentes_pf = sim_params["num_agentes_pf"]
    agentes_pf = []
    for i in range(num_agentes_pf):
        cotas = (sim_params["agente_pf"]["cotas_iniciais_primeiro"] if i == 0
                 else sim_params["agente_pf"]["cotas_iniciais_outros"])
        agentes_pf.append(Agente(
            id=i,
            literacia_financeira=gerar_literacia_financeira(minimo=0.2, maximo=0.7),
            caixa=sim_params["agente_pf"]["caixa_inicial"],
            cotas=cotas,
            expectativa_inflacao=sim_params["agente_pf"]["expectativa_inflacao"],
            expectativa_premio=sim_params["agente_pf"]["expectativa_premio"],
            historico_precos=historia,
            params=sim_params["agente_pf"].get("params")
        ))

    num_agentes_pj = sim_params["num_agentes_pj"]
    agentes_pj = []
    for i in range(num_agentes_pj):
        cotas = (sim_params["agente_pj"]["cotas_iniciais_primeiro"] if i == 0
                 else sim_params["agente_pj"]["cotas_iniciais_outros"])
        agentes_pj.append(Agente(
            id=i + num_agentes_pf,
            literacia_financeira=gerar_literacia_financeira(minimo=0.7, maximo=1.0),
            caixa=sim_params["agente_pj"]["caixa_inicial"],
            cotas=cotas,
            expectativa_inflacao=sim_params["agente_pj"]["expectativa_inflacao"],
            expectativa_premio=sim_params["agente_pj"]["expectativa_premio"],
            historico_precos=historia,
            params=sim_params["agente_pj"].get("params")
        ))

    agentes = agentes_pf + agentes_pj

    for ag in agentes_pf:
        ag.definir_vizinhos(agentes, num_vizinhos=sim_params["agente_pf"]["num_vizinhos"])
    for ag in agentes_pj:
        ag.definir_vizinhos(agentes_pj, num_vizinhos=sim_params["agente_pj"]["num_vizinhos"])

    print(f"Total de Agentes: {len(agentes)}")

    bc = BancoCentral(sim_params["banco_central"])
    midia = Midia(
        dias=sim_params["num_dias"],
        valor_inicial=sim_params["midia"]["valor_inicial"],
        sigma=sim_params["parametros_sentimento"]["sigma_midia"],
        valores_fixos=sim_params["midia"]["valores_fixos"]
    )

    mercado = Mercado(
        agentes=agentes,
        imoveis=fii.imoveis,
        fii=fii,
        banco_central=bc,
        midia=midia,
        params=sim_params["mercado"]
    )

    return mercado, list(historia)


# ─────────────────────────────────────────────────────────────────────────────
# Função original — mantida intacta para uso nos notebooks
# ─────────────────────────────────────────────────────────────────────────────
def simular_mercado_e_plotar(parametros_sistema, num_dias, imprimir=False, overrides=None):
    sim_params = _construir_sim_params(parametros_sistema, num_dias, overrides)
    mercado, historico_precos_fii = _inicializar(sim_params)
    parametros_sentimento = sim_params["parametros_sentimento"]
    sentimento_medio_ao_longo_dos_dias = []

    for dia in range(1, num_dias + 1):
        mercado.executar_dia(parametros_sentimento)
        sentimento_medio_ao_longo_dos_dias.append(calcular_sentimento_medio(mercado.agentes))
        historico_precos_fii.append(mercado.fii.preco_cota)

    historico_precos_fii = np.array(historico_precos_fii)
    log_returns = np.diff(np.log(historico_precos_fii))

    window = sim_params["plot"]["window_volatilidade"]
    volatilidade_rolante = np.full_like(log_returns, np.nan)
    for i in range(window, len(log_returns)):
        volatilidade_rolante[i] = np.std(log_returns[i - window:i]) * (252 ** 0.5)

    if imprimir:
        print(f"Preço Final da Cota: R${mercado.fii.preco_cota:,.2f}")
        print(f"Caixa Final do FII: R${mercado.fii.caixa:,.2f}")
        for agente in mercado.agentes:
            print(f"Agente {agente.id}: Caixa: R${agente.caixa:,.2f}, "
                  f"Sentimento: {agente.sentimento:.2f}, "
                  f"Riqueza: R${agente.historico_riqueza[-1]:,.2f}")

        dias_array = np.arange(len(historico_precos_fii))
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax[0].plot(dias_array, historico_precos_fii, label="Preço da Cota do FII")
        ax[0].set_title("Evolução do Preço do FII")
        ax[0].set_ylabel("Preço")
        ax[0].legend()
        ax[1].plot(dias_array[1:], volatilidade_rolante,
                   label="Volatilidade Rolante (200 dias)", color="orange")
        ax[1].set_title("Volatilidade Rolante dos Retornos Logarítmicos")
        ax[1].set_ylabel("Volatilidade")
        ax[1].set_xlabel("Dias")
        ax[1].legend()
        plt.tight_layout()
        plt.show()

    return historico_precos_fii, log_returns, volatilidade_rolante, mercado.midia, sentimento_medio_ao_longo_dos_dias


# ─────────────────────────────────────────────────────────────────────────────
# Função geradora — usada pelo Streamlit para atualização em tempo real
# ─────────────────────────────────────────────────────────────────────────────
def simular_mercado_streaming(parametros_sistema, num_dias, overrides=None, seed=None):
    """
    Gerador que executa a simulação dia a dia e cede o estado a cada iteração.
    Usa exatamente a mesma inicialização de simular_mercado_e_plotar.

    Yields
    ------
    dict com:
        dia            : int
        preco          : float
        sentimento     : float
        volatilidade   : float
        news           : float
        historico_precos : list  (acumulado)
    """
    sim_params = _construir_sim_params(parametros_sistema, num_dias, overrides)
    mercado, historico_precos = _inicializar(sim_params, seed=seed)
    parametros_sentimento = sim_params["parametros_sentimento"]

    for dia in range(1, num_dias + 1):
        mercado.executar_dia(parametros_sentimento)

        preco = mercado.fii.preco_cota
        historico_precos.append(preco)

        yield {
            "dia":             dia,
            "preco":           preco,
            "sentimento":      float(np.mean([ag.sentimento for ag in mercado.agentes])),
            "volatilidade":    mercado.volatilidade_historica,
            "news":            mercado.news,
            "historico_precos": historico_precos,
            "mercado":         mercado,
        }