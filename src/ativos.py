import numpy as np
import pandas as pd
from typing import Optional, Sequence

# 2025 incompleto

_df_ifix_cache = None

def carregar_ifix():
    global _df_ifix_cache
    if _df_ifix_cache is None:
        url = "https://docs.google.com/spreadsheets/d/1bBVbP7GH3JPcRPPn0qydQJ6H16I3yDbz/export?format=csv"
        df = pd.read_csv(url, index_col="Date", parse_dates=True, dayfirst=True)
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
        df["Close"] = df["Close"].astype(str).str.replace(".", "")
        df["Close"] = df["Close"].astype(str).str.replace(",", ".").astype(float)
        df = df.sort_index(ascending=True)
        _df_ifix_cache = df.rename(columns={'Close': 'IFIX'})[-1008:]
    return _df_ifix_cache

def reconstruir_precos(precos_hist, P0, n):
    """
    Reconstrói uma série de preços a partir dos retornos logarítmicos dos últimos n preços,
    preservando as variações percentuais e terminando exatamente em P0.

    Parâmetros
    ----------
    precos_hist : list ou np.array
        Lista ou array com os preços históricos.
    P0 : float
        Preço final desejado da nova série.
    n : int
        Tamanho da janela (número de períodos recentes a considerar).

    Retorna
    -------
    np.array
        Nova série de preços (mantendo variações percentuais e terminando em P0).
    """
    precos_hist = np.array(precos_hist)

    # Retornos logarítmicos
    retornos_log = np.diff(np.log(precos_hist))

    # Pega os últimos n retornos (os mais recentes)
    retornos_log = retornos_log[-n:]

    # Reconstrói os preços para trás a partir de P0
    precos = [P0]
    for r in reversed(retornos_log):
        precos.append(precos[-1] / np.exp(r))

    # Inverte para a ordem cronológica correta (do passado ao presente)
    precos = precos[::-1]

    return np.array(precos)


class Imovel:
    def __init__(self, valor: float, vacancia: float, custo_manutencao: float, params: dict=None) -> None:
        self.valor = valor
        self.vacancia = vacancia
        self.custo_manutencao = custo_manutencao
        # Recebe os parâmetros customizáveis ou usa os valores default
        self.params = params if params is not None else {}
        self.aluguel_factor = self.params.get("aluguel_factor", 0.005)
        self.desvio_normal = self.params.get("desvio_normal", 0.1)
        self.aluguel = self.valor * self.aluguel_factor

    def gerar_fluxo_aluguel(self) -> float:
        return self.aluguel * (1 - self.vacancia * (1 + np.random.normal(0, self.desvio_normal)))




class FII:
    def __init__(self, num_cotas: int, caixa: float, params: list = None) -> None:
        self.num_cotas = num_cotas
        self.caixa = caixa
        self.imoveis = []
        self.retornos_diarios = []
        self.preco_cota = 0
        self.historico_precos = []
        self.historico_dividendos = [self.valor_patrimonial_por_acao()*0.25]
        # Guarda os parâmetros passados ou usa um dicionário vazio se não forem fornecidos
        self.params = params if params is not None else {}

    def valor_patrimonial_por_acao(self) -> float:
        # Converter para array de valores se self.imoveis for grande e homogêneo
        # Caso contrário, sum() com generator expression é eficiente para listas de objetos
        total_valor_imoveis = sum(imovel.valor for imovel in self.imoveis)
        return (self.caixa + total_valor_imoveis) / self.num_cotas

    def adicionar_imovel(self, imovel: Imovel) -> None:
        self.imoveis.append(imovel)

    def calcular_fluxo_aluguel(self) -> float:
        # Similarmente, generator expression é eficiente
        return sum(imovel.gerar_fluxo_aluguel() for imovel in self.imoveis)

    def distribuir_dividendos(self) -> float:
        fluxo_aluguel = self.calcular_fluxo_aluguel()
        dividendos_rate = self.params.get("dividendos_taxa", 0.95)
        caixa_rate = self.params.get("dividendos_caixa_taxa", 0.05)
        # Verifica num_cotas antes de dividir para evitar ZeroDivisionError
        dividendos = fluxo_aluguel * dividendos_rate / self.num_cotas if self.num_cotas > 0 else 0
        self.historico_dividendos.append(dividendos)
        self.caixa += fluxo_aluguel * caixa_rate
        return dividendos

    def atualizar_caixa_para_despesas(self, despesas: float) -> None:
        self.caixa -= despesas
        if self.caixa < 0:
            self.caixa = 0

    def atualizar_imoveis_investir(self, inflacao: float) -> None:
        investimento_fracao = self.params.get("investimento_fracao", 0.50)
        valor_investir = investimento_fracao * self.caixa
        self.caixa -= valor_investir

        # Se houver muitos imóveis, podemos extrair os valores para um array NumPy temporariamente.
        # Mas para o número de imóveis no exemplo, o loop é ok.
        num_imoveis = len(self.imoveis)
        if num_imoveis > 0:
            investimento_por_imovel = valor_investir / num_imoveis
        else:
            investimento_por_imovel = 0

        for imovel in self.imoveis:
            imovel.valor *= (1 + inflacao)
            imovel.valor += investimento_por_imovel # Adiciona o valor por imóvel
            imovel.aluguel = imovel.valor * 0.005
            # Formatação de string otimizada e f-string no Python 3.6+ já é eficiente
            # Não é necessário o replace complexo.
            print(f"Imovel atualizado: Valor: R${imovel.valor:,.2f}")

        print(f"[FII] Imóveis atualizados; reinvestido: R${valor_investir:,.2f}")

    def realizar_investimento(self, valor: float) -> None:
        if self.caixa >= valor:
            self.caixa -= valor
        else:
            raise ValueError("Caixa insuficiente para realizar o investimento.")

    def calcular_retorno_diario(self, novo_preco_cota: float) -> None:
        if self.preco_cota > 0:
            retorno = (novo_preco_cota - self.preco_cota) / self.preco_cota
            self.retornos_diarios.append(retorno)
        self.preco_cota = novo_preco_cota

    def obter_estatisticas_retornos(self) -> None or dict:
        if not self.retornos_diarios:
            return None
        media_retorno = np.mean(self.retornos_diarios)
        volatilidade = np.std(self.retornos_diarios)
        return {"media_retorno": media_retorno, "volatilidade": volatilidade}

    #SEM MEMÓRIA
    # def inicializar_historico(self, dias: int=30) -> list:
    #     self.historico_precos = []
    #     for _ in range(dias):
    #         self.historico_precos.append(self.valor_patrimonial_por_acao()*0.6)
    #     return self.historico_precos

    #COM MEMÓRIA

    def inicializar_historico(self, dias: int=252, memoria= False) -> list:
        P0 = self.valor_patrimonial_por_acao() * 0.65
        if memoria:
          precos_ifix = carregar_ifix()['IFIX'][-dias:].values
          self.historico_precos = list(reconstruir_precos(precos_ifix, P0, dias-1))
          # for _ in range(dias):
          #     self.historico_precos.append(self.valor_patrimonial_por_acao()*0.6)
          return self.historico_precos

        else:
          preco_inicial = P0
          self.historico_precos.append(preco_inicial)
          return self.historico_precos



