import random
import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import expon

from src.microestrutura import Ordem


def gerar_literacia_financeira(minimo=0.2, maximo=1.0, lambda_=4):
    """
    Gera um valor de literacia financeira usando uma distribuição exponencial truncada no intervalo [minimo, maximo].

    A densidade é decrescente: valores mais baixos são mais prováveis.
    """
    while True:
        valor = expon(scale=1/lambda_).rvs()
        if minimo <= valor <= maximo:
            return valor



class Agente:
    def __init__(self, id: int, literacia_financeira: float, caixa: float, cotas: int,
                 expectativa_inflacao: float, expectativa_premio: float, historico_precos: list,
                 params: dict=None) -> None:
        self.id = id
        self.LF = literacia_financeira
        self.caixa = caixa
        self.cotas = cotas

        ################################################################# INSERI em 06/06/25
        self.params = params if params is not None else {}
        # Obtém 'piso_prob_negociar' dos parâmetros, com um valor padrão seguro
        piso_prob_negociar_val = self.params.get("piso_prob_negociar", 0.1) # Use 0.1 como default, ou seu padrão desejado
        # self.prob_negociar = np.clip(piso_prob_negociar_val + 0.9 * ((1 - self.LF) ** 2), 0.1, 1.0)
        self.prob_negociar = np.clip(0.8 - 0.875 * (self.LF - 0.2), 0.1, 0.8)
        ####################################################################



        #self.prob_negociar = np.clip(0.3 + 0.9 * ((1 - self.LF) ** 2), 0.1, 1.0) ################# -----------> Parâmetro sistêmico: piso da probabilidade de negociação (0.3)

        self.saldo = caixa
        self.carteira = {"FII": cotas}

        self.sentimento = 0
        self.RD = 0
        self.percentual_alocacao = 0

        self.expectativa_inflacao = expectativa_inflacao
        self.expectativa_premio = expectativa_premio

        # Converter para numpy array na inicialização para operações vetorizadas
        self.historico_precos = np.array(historico_precos)
        self.retornos_dia = []
        self.historico_riqueza = np.array([caixa + cotas * historico_precos[-1]])

        self.historico_sentimentos = []
        self.vizinhos = []
        self.params = params if params is not None else {}

    def definir_vizinhos(self, todos_agentes: list, num_vizinhos: int = 3) -> None:
        candidatos = [agente for agente in todos_agentes if agente.id != self.id]
        self.vizinhos = random.sample(candidatos, min(num_vizinhos, len(candidatos)))

    def calcular_preco_esperado(self, literacia_financeira: float, beta: float, dividendos: float) -> float:
        # Já usa np.exp, np.log (vetorizadas)
        x = literacia_financeira / (np.exp(1) ** beta)
        z = (1 - beta) * (1 - literacia_financeira)
        y = 1 - x - z

        preco_fundamentalista = dividendos * 12 * (1 + self.expectativa_inflacao) / self.expectativa_premio

        # Garante que historico_precos seja um array NumPy para operações vetorizadas
        if not isinstance(self.historico_precos, np.ndarray):
            self.historico_precos = np.array(self.historico_precos)

        # Evita log(0) ou log(negativo) e log(último preço) se não houver histórico suficiente
        if len(self.historico_precos) > 0 and self.historico_precos[-1] > 0 and preco_fundamentalista > 0:
            retorno_fundamentalista = np.log(preco_fundamentalista) - np.log(self.historico_precos[-1])
        else:
            retorno_fundamentalista = 0.0 # Ou algum valor padrão/erro

        #Estratégia Chartista: usa janela e alfas configuráveis

        def calcular_ema_curta_e_longa_otimizado(precos_historicos, LF):
          # Verifica se fica esse ou do Mercado, Verificar como ficaria com média aritimética.
            omega = int(LF * 252)
            if omega < 2: # Garantir janela mínima para EMA
                omega = 2

            janela_curta = max(2, int(omega / 4))

            alpha_short = 2 / (janela_curta + 1)
            alpha_long = 2 / (omega + 1)

            # Usar apenas a parte relevante do histórico e garantir que seja numpy array
            serie_precos = pd.Series(precos_historicos[-omega:]) # pd.Series é eficiente para EMA

            if len(serie_precos) < 2: # Evitar erros para séries muito curtas
                return precos_historicos[-1], precos_historicos[-1] # Ou outro valor padrão

            ema_short = serie_precos.ewm(alpha=alpha_short, adjust=False).mean().iloc[-1]
            ema_long = serie_precos.ewm(alpha=alpha_long, adjust=False).mean().iloc[-1]

            return ema_short, ema_long

        ####################################

        def calcular_sma_curta_e_longa_otimizado(precos_historicos, LF):
            """
            Calcula as Médias Móveis Simples (SMA) curta e longa com base na literacia financeira.

            Parâmetros:
            - precos_historicos (np.ndarray ou list): Histórico de preços.
            - LF (float): Nível de literacia financeira do agente.

            Retorna:
            - tuple: Uma tupla contendo (sma_short, sma_long).
            """
            # Converter para numpy array se ainda não for, para fatiamento e cálculo eficientes
            if not isinstance(precos_historicos, np.ndarray):
                precos_historicos = np.array(precos_historicos)

            # Definir as janelas de cálculo
            omega = int(LF * 252) # Janela longa (ex: 252 dias úteis como base)
            if omega < 2: # Garantir janela mínima
                omega = 2

            janela_curta = max(2, int(omega / 4)) # Janela curta, tipicamente 1/4 da longa

            # Inicializar SMAs com o último preço, caso não haja histórico suficiente
            sma_short = precos_historicos[-1] if len(precos_historicos) > 0 else 0.0
            sma_long = precos_historicos[-1] if len(precos_historicos) > 0 else 0.0

            # Calcular SMA Curta
            if len(precos_historicos) >= janela_curta:
                sma_short = np.mean(precos_historicos[-janela_curta:])

            # Calcular SMA Longa
            if len(precos_historicos) >= omega:
                sma_long = np.mean(precos_historicos[-omega:])

            # Tratamento para evitar log(0) ou divisão por zero no retorno especulativo
            if sma_long == 0 and sma_short == 0:
                return 0.0, 0.0
            elif sma_long == 0: # Se a SMA longa for zero, mas a curta não, use a curta como referência
                return sma_short, sma_short
            elif sma_short == 0: # Se a SMA curta for zero, mas a longa não, use a longa como referência
                return sma_long, sma_long

            return sma_short, sma_long

        #####################################



        ema_short, ema_long = calcular_sma_curta_e_longa_otimizado(self.historico_precos, self.LF)

        # Evita log(0) ou log(negativo)
        if ema_long > 0:
             retorno_expeculador = np.log(ema_short / ema_long)
        else:
            retorno_expeculador = 0.0 # Ou algum valor padrão/erro

        ruido_std = self.params.get("ruido_std", 0.1)
        retorno_ruido = np.random.normal(0, ruido_std)

        retorno_expectativa = (x * retorno_fundamentalista) + (y * retorno_expeculador) + (z * retorno_ruido)

        # Evita multiplicar por 0 ou NaN
        if len(self.historico_precos) > 0 and self.historico_precos[-1] > 0:
            preco_esperado = self.historico_precos[-1] * np.exp(retorno_expectativa)
        else:
            preco_esperado = 0.0 # Ou algum valor padrão/erro

        return preco_esperado

    def atualizar_caixa(self, taxa_selic: float, dividendos: float) -> None:
        taxa_selic_mes = (1 + taxa_selic) ** (1 / 12) - 1
        # Operações matemáticas diretas
        self.caixa *= (1 + taxa_selic_mes)
        self.caixa += dividendos

    def calcular_I_privada(self, n: int = 5, beta: int = 0, dividendos: float = 0.0) -> float:
        peso_retorno = self.params.get("peso_retorno", 0.8)
        peso_riqueza = self.params.get("peso_riqueza", 0.4)

        preco_esperado = self.calcular_preco_esperado(self.LF, beta, dividendos)
        # print(f"\tPreco esperado: {preco_esperado:.2f}") # Removido para desempenho

        # Garante que historico_precos seja um array NumPy para operações vetorizadas
        if not isinstance(self.historico_precos, np.ndarray):
            self.historico_precos = np.array(self.historico_precos)

        preco_atual = self.historico_precos[-1] if len(self.historico_precos) > 0 else 0.0

        if preco_atual > 0 and preco_esperado > 0:
            componente_retorno = np.log(preco_esperado / preco_atual)
        else:
            componente_retorno = 0.0

        # Garante que historico_riqueza seja um array NumPy para operações vetorizadas
        if not isinstance(self.historico_riqueza, np.ndarray):
            self.historico_riqueza = np.array(self.historico_riqueza)

        n = int(self.LF * 252)

        if len(self.historico_riqueza) >= n and self.historico_riqueza[-n] != 0:
            variacao_riqueza = (self.historico_riqueza[-1] - self.historico_riqueza[-n]) / self.historico_riqueza[-n]
            componente_riqueza = variacao_riqueza
        else:
            componente_riqueza = 0.0

        I_privada = peso_retorno * componente_retorno + peso_riqueza * componente_riqueza + np.random.normal(0, 0.05)
        return I_privada

    def calcular_I_social(self, vizinhos: list) -> float:
        # Vetorização da coleta e cálculo da média de sentimentos dos vizinhos
        # Cria um array NumPy com os sentimentos para cálculo da média
        sentimentos_vizinhos = np.array([
            np.mean(v.historico_sentimentos[-3:]) if len(v.historico_sentimentos) >= 3 else (np.mean(v.historico_sentimentos) if v.historico_sentimentos else 0.0)
            for v in vizinhos
        ])

        # Filtra NaNs que podem surgir de .mean() em listas vazias (já tratado acima com o condicional)
        sentimentos_vizinhos = np.nan_to_num(sentimentos_vizinhos)

        if len(sentimentos_vizinhos) == 0:
            return 0.0

        I_social = np.mean(sentimentos_vizinhos)
        return I_social

    def calcular_sentimento_risco_alocacao(self, mercado: "Mercado", vizinhos: list, parametros: dict) -> None:
        I_privado = self.calcular_I_privada(n=5, beta = parametros["beta"], dividendos = mercado.fii.historico_dividendos[-1])
        I_social = self.calcular_I_social(vizinhos)
        # Removido para desempenho
        # print(f"\nAgente {self.id}:")
        # print(f"\tI_privado: {I_privado:.2f}")
        # print(f"\tVizinhos: {[v.id for v in vizinhos]}")
        # print(f"\tSentimentos dos vizinhos: {[v.sentimento for v in vizinhos]}")
        # print(f"\tI_social: {I_social:.2f}")

        volatilidade_percebida = mercado.volatilidade_historica
        a_i = parametros["a0"] * self.LF
        b_i = parametros["b0"] * (1 - self.LF)
        c_i = parametros["c0"] * (1 - self.LF)

        S_bruto = round(a_i * I_privado + b_i * I_social + c_i * mercado.news, 4)
        self.sentimento = max(min(S_bruto, 1), -1)
        self.historico_sentimentos.append(self.sentimento)

        # Evita divisão por zero
        self.RD = (self.sentimento + 1) / 2 * volatilidade_percebida
        self.percentual_alocacao = self.RD / volatilidade_percebida if volatilidade_percebida > 0 else 0

        self.calcular_expectativa_inflacao(mercado.banco_central, mercado.news, mercado, parametros)
        self.calcular_expectativa_premio(mercado, parametros)

        # Removido para desempenho
        # print(f"\tSentimento final: {self.sentimento:.2f}")
        # print(f"\tRisco percebido (RD): {self.RD:.2f}")
        # print(f"\tPercentual de alocação: {self.percentual_alocacao:.2f}%")

    def calcular_expectativa_inflacao(self, banco_central: "BancoCentral", noticias: float, mercado: "Mercado", parametros: float ) -> None:

        self.expectativa_inflacao = banco_central.expectativa_inflacao * (1 - self.sentimento * parametros["peso_sentimento_inflacao"]) ###### -------------> Parâmetro peso_sentimento_inflacao

    def calcular_expectativa_premio(self, mercado: "Mercado", parametros: float) -> None:
        self.expectativa_premio = mercado.banco_central.premio_risco * (1 - self.sentimento * parametros["peso_sentimento_expectativa"]) ###### -------------> Parâmetro peso_sentimento_expectativa

    def calcular_estatisticas_retoricas(self) -> Optional[dict]:
        if len(self.historico_precos) < 2:
            return None
        # Garante que historico_precos é um array NumPy antes de calcular retornos
        if not isinstance(self.historico_precos, np.ndarray):
            self.historico_precos = np.array(self.historico_precos)

        # Cálculo vetorizado de retornos
        retornos = (self.historico_precos[1:] - self.historico_precos[:-1]) / self.historico_precos[:-1]

        media_retorno = np.mean(retornos)
        volatilidade = np.std(retornos)
        sharpe_ratio = media_retorno / volatilidade if volatilidade > 0 else 0
        return {"media_retorno": media_retorno, "volatilidade": volatilidade, "sharpe_ratio": sharpe_ratio}

    def calcular_retornos_dia(self, preco_atual: float) -> None:
        if len(self.historico_precos) > 0:
            preco_anterior = self.historico_precos[-1]
            if preco_anterior > 0:
                retorno = (preco_atual - preco_anterior) / preco_anterior
                self.retornos_dia.append(retorno)

        # Adiciona o novo preço ao histórico
        # Para manter historico_precos como np.ndarray, podemos concatenar ou usar uma lista temporária
        self.historico_precos = np.append(self.historico_precos, preco_atual)

    def atualizar_historico(self, preco_fii: float) -> None:
        # Garante que carteira.get("FII", 0) seja numérico para o cálculo
        riqueza_atual = self.caixa + self.carteira.get("FII", 0) * preco_fii
        # Para manter historico_riqueza como np.ndarray, concatene
        self.historico_riqueza = np.append(self.historico_riqueza, riqueza_atual)


    def criar_ordem(self, mercado: "Mercado", parametros: float) -> Optional["Ordem"]:
            ativo = "FII"
            preco_mercado = mercado.fii.preco_cota
            if preco_mercado <= 0:
                return None

            preco_esperado = self.calcular_preco_esperado(
                literacia_financeira=self.LF,
                beta=parametros["beta"],
                dividendos= mercado.fii.historico_dividendos[-1]
                #dividendos=mercado.fii.calcular_fluxo_aluguel() * 0.95 / mercado.fii.num_cotas
            )

            peso_base = parametros.get("peso_preco_esperado", 0.3)
            #peso_preco_esperado = parametros.get("peso_preco_esperado", 0.3)

            if hasattr(self, "choque_ativo") and self.choque_ativo:
                peso_preco_esperado = peso_base * 2 #(1 + self.choque_ativo["intensidade_atual"])
            else:
                peso_preco_esperado = peso_base



            if preco_mercado < preco_esperado:
                qtd_min = parametros.get("quantidade_compra_min", 1)
                qtd_max = parametros.get("quantidade_compra_max", 10)
                cotas_desejadas = random.randint(qtd_min, qtd_max)

                valor_necessario = preco_mercado * cotas_desejadas
                if self.saldo >= valor_necessario:
                    preco_limite = (1 - peso_preco_esperado) * preco_mercado + peso_preco_esperado * preco_esperado
                    return Ordem(tipo="compra", agente=self, ativo=ativo, preco_limite=preco_limite, quantidade=cotas_desejadas)

            elif preco_mercado > preco_esperado:
                cotas_possuidas = self.carteira.get(ativo, 0)
                if cotas_possuidas > 0:
                    preco_limite = (1 - peso_preco_esperado) * preco_mercado + peso_preco_esperado * preco_esperado
                    divisor_venda = parametros.get("divisor_quantidade_venda", 5)
                    quantidade_venda_max = int(cotas_possuidas / divisor_venda) + 1
                    return Ordem(tipo="venda", agente=self, ativo=ativo, preco_limite=preco_limite,
                                quantidade=random.randint(1, quantidade_venda_max))
            return None


    def aplicar_choque(self, tipo_choque: str, intensidade: float = 0.2, duracao: int = 5, delta: float = 0.5) -> None:
        """Aplica um choque ao agente e inicia dissipação temporal."""
        # Salva o valor original apenas uma vez
        if not hasattr(self, "_prob_negociar_base"):
            self._prob_negociar_base = self.prob_negociar

        self.choque_ativo = {
            "tipo": tipo_choque,
            "intensidade_inicial": intensidade,
            "dias_restantes": duracao,
            "delta": delta,
            "intensidade_atual": intensidade
        }

        # Aplica o impacto inicial
        self._aplicar_impacto_choque()

    def _aplicar_impacto_choque(self) -> None:
        choque = getattr(self, "choque_ativo", None)
        if not choque:
            return

        intensidade = choque["intensidade_atual"]
        tipo = choque["tipo"]

        if tipo == "negativo":
            self.expectativa_inflacao *= (1 + intensidade/100)
            self.expectativa_premio *= (1 + intensidade/100)
            #self.sentimento -= intensidade
            # Aumenta chance de negociação durante pânico
            self.prob_negociar = np.clip(self._prob_negociar_base * (1 + 0.3 * intensidade), 0.1, 1.0)

        elif tipo == "positivo":
            self.expectativa_inflacao *= (1 - intensidade)
            self.expectativa_premio *= (1 - intensidade)
            #self.sentimento += intensidade
            # Reduz chance de negociação (confiança e menor reação)
            self.prob_negociar = np.clip(self._prob_negociar_base * (1 - 0.3 * intensidade), 0.1, 1.0)

        self.sentimento = np.clip(self.sentimento, -1, 1)

    def atualizar_choque(self) -> None:
        """Atualiza dissipação do choque e retorna ao comportamento original."""
        choque = getattr(self, "choque_ativo", None)
        if not choque:
            return

        if choque["dias_restantes"] > 0:
            choque["dias_restantes"] -= 1
            choque["intensidade_atual"] *= choque["delta"]
            self._aplicar_impacto_choque()
        else:
            # Dissipou completamente: retorna gradualmente ao valor base
            self.prob_negociar = self._prob_negociar_base
            self.choque_ativo = None
