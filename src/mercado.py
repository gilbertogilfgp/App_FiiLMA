import os
import warnings
import itertools
import datetime as dt
import pandas as pd
import numpy as np
import random # Usado para random.sample e random.shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from scipy.stats import norm, skew, kurtosis, truncnorm # Importar diretamente as funções
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from scipy.optimize import curve_fit
from statsmodels.graphics.gofplots import qqplot
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import random
import traceback # Importar para printar o traceback em caso de erro

warnings.filterwarnings("ignore")

from src.ativos import Imovel, FII
from src.agentes import Agente, gerar_literacia_financeira
from src.microestrutura import OrderBook


class BancoCentral:
    def __init__(self, params: dict = None) -> None:
        params = params if params is not None else {}
        self.taxa_selic = params.get("taxa_selic", 0.15)
        self.expectativa_inflacao = params.get("expectativa_inflacao", 0.07)
        self.premio_risco = params.get("premio_risco", 0.08)





class Midia:
    def __init__(self, dias, valor_inicial, sigma, valores_fixos):
        self.dias = dias
        self.valor_atual = valor_inicial
        self.sigma = sigma
        self.valores_fixos = valores_fixos
        self.t = 0  # contador de tempo
        self.historico = [valor_inicial]

    def gerar_noticia(self):
        #----------------------- Com triangular ---------------------

        #self.valor_atual = np.random.normal(0, self.sigma) # ou random.normal(-2, 2, 0)

        #----------------------- Com passeio aleatório ----------------
        if self.t >= self.dias + 2:
            raise StopIteration("Fim da simulação de notícias.")

        self.t += 1

        if self.t in self.valores_fixos:
            self.valor_atual = self.valores_fixos[self.t]
        else:
            passo = np.random.normal(0, self.sigma)
            #self.valor_atual = np.clip(self.valor_atual + passo, -3, 3)
            self.valor_atual = np.clip(passo, -3, 3)
        #-------------------------------------------------------------------

        self.historico.append(self.valor_atual)
        return self.valor_atual

    def get_historico(self):
        return self.historico





def _processar_agente_para_pool(agente_data):
    """
    Versão paralela alinhada com a lógica atual da classe Agente.
    Garante consistência entre execução standalone (Pool.map) e métodos originais.
    """

    try:
        (
            agente_id,
            agente_state,
            vizinhos_sentiments_snapshot,
            mercado_snapshot,
            banco_central_snapshot,
            parametros_sentimento_snapshot,
            agente_params_snapshot
        ) = agente_data

        # =======================================================
        # 0. Verifica se o agente vai negociar no dia
        # =======================================================
        prob_negociar = agente_state.get("prob_negociar", 1.0)
        if random.random() >= prob_negociar:
            return {
                "id": agente_id,
                "vai_negociar": False,
                "sentimento": agente_state["sentimento"],
                "historico_sentimentos_novo": agente_state["sentimento"],
            }

        # =======================================================
        # 1. Reconstrução mínima do estado do agente
        # =======================================================
        LF = agente_state["literacia_financeira"]
        hist_precos = np.array(agente_state["historico_precos"])
        hist_riqueza = np.array(agente_state["historico_riqueza"])
        sent_ant = agente_state["sentimento"]

        ruido_std = agente_params_snapshot.get("ruido_std", 0.1)
        peso_retorno = agente_params_snapshot.get("peso_retorno", 0.8)
        peso_riqueza = agente_params_snapshot.get("peso_riqueza", 0.4)

        # =======================================================
        # 2. Expectativas (seguindo a lógica atual do Agente)
        # =======================================================
        peso_sent_infl = parametros_sentimento_snapshot.get("peso_sentimento_inflacao", 0.4)
        peso_sent_exp = parametros_sentimento_snapshot.get("peso_sentimento_expectativa", 0.4)

        exp_infl_bc = banco_central_snapshot.get("expectativa_inflacao", 0.07)
        premio_bc = banco_central_snapshot.get("premio_risco", 0.08)

        expectativa_inflacao_agente = exp_infl_bc * (1 - sent_ant * peso_sent_infl)
        expectativa_premio_agente = premio_bc * (1 - sent_ant * peso_sent_exp)

        # =======================================================
        # 3. Cálculo de I_privado (com SMA e ruído)
        # =======================================================
        def calcular_sma(precos, LF):
            omega = max(2, int(LF * 252))
            janela_curta = max(2, int(omega / 4))

            if len(precos) < 2:
                return precos[-1] if len(precos) else 0.0, precos[-1] if len(precos) else 0.0

            sma_short = np.mean(precos[-janela_curta:]) if len(precos) >= janela_curta else precos[-1]
            sma_long = np.mean(precos[-omega:]) if len(precos) >= omega else precos[-1]
            return sma_short, sma_long

        dividendos = mercado_snapshot.get("fii_historico_dividendos_ultimo", 0.0)
        beta = parametros_sentimento_snapshot["beta"]

        # Preço esperado (fundamental + especulativo + ruído)
        x = LF / (np.exp(1) ** beta)
        z = (1 - beta) * (1 - LF)
        y = 1 - x - z

        preco_fund = dividendos * 12 * (1 + expectativa_inflacao_agente) / expectativa_premio_agente
        retorno_fund = np.log(preco_fund / hist_precos[-1]) if (len(hist_precos) and hist_precos[-1] > 0 and preco_fund > 0) else 0.0

        sma_short, sma_long = calcular_sma(hist_precos, LF)
        retorno_chart = np.log(sma_short / sma_long) if sma_long > 0 else 0.0
        retorno_ruido = np.random.normal(0, ruido_std)

        retorno_expect = x * retorno_fund + y * retorno_chart + z * retorno_ruido
        preco_esperado = hist_precos[-1] * np.exp(retorno_expect) if len(hist_precos) and hist_precos[-1] > 0 else 0.0

        preco_atual = hist_precos[-1] if len(hist_precos) > 0 else 0.0
        componente_retorno = np.log(preco_esperado / preco_atual) if preco_atual > 0 and preco_esperado > 0 else 0.0

        n = int(LF * 252)
        if len(hist_riqueza) >= n and hist_riqueza[-n] != 0:
            var_riqueza = (hist_riqueza[-1] - hist_riqueza[-n]) / hist_riqueza[-n]
        else:
            var_riqueza = 0.0

        I_privado = (peso_retorno * componente_retorno + peso_riqueza * var_riqueza + np.random.normal(0, 0.01))* LF / 2

        # =======================================================
        # 4. Cálculo de I_social (média dos vizinhos)
        # =======================================================
        I_social = np.mean(np.nan_to_num(vizinhos_sentiments_snapshot)) if len(vizinhos_sentiments_snapshot) else 0.0

        # =======================================================
        # 5. Cálculo do Sentimento, RD e Alocação
        # =======================================================
        a_i = parametros_sentimento_snapshot["a0"] * LF
        b_i = parametros_sentimento_snapshot["b0"] * (1 - LF)
        c_i = parametros_sentimento_snapshot["c0"] * (1 - LF)
        news = mercado_snapshot["news"]

        S_bruto = round(a_i * I_privado + b_i * I_social + c_i * news, 6)
        sentimento_agente = np.clip(S_bruto, -1, 1)

        vol = mercado_snapshot["volatilidade_historica"]
        RD = (sentimento_agente + 1) / 2 * vol
        aloc = RD / vol if vol > 0 else 0

        # =======================================================
        # 6. Retorno (dados coerentes com o Agente real)
        # =======================================================
        return {
            "id": agente_id,
            "vai_negociar": True,
            "sentimento": sentimento_agente,
            "historico_sentimentos_novo": sentimento_agente,
            "expectativa_inflacao": expectativa_inflacao_agente,
            "expectativa_premio": expectativa_premio_agente,
            "RD": RD,
            "percentual_alocacao": aloc,
        }

    except Exception as e:
        print(f"⚠️ Erro processando agente {agente_data[0]} no pool: {e}")
        traceback.print_exc()
        return None



class Mercado:
    def __init__(self, agentes: list, imoveis: list, fii: "FII", banco_central: "BancoCentral", midia: "Midia", params: dict = None) -> None:
        self.agentes = agentes
        self.imoveis = imoveis # Referência aos imóveis do FII
        self.fii = fii # Referência ao FII
        self.order_book = OrderBook()
        self.banco_central = banco_central
        self.midia = midia
        params = params if params is not None else {}
        self.volatilidade_historica = params.get("volatilidade_inicial", 0.1)
        self.dividendos_frequencia = params.get("dividendos_frequencia", 21)
        self.atualizacao_imoveis_frequencia = params.get("atualizacao_imoveis_frequencia", 126)
        self.news = 0
        self.historico_news = []
        self.dia_atual = 0
        # Configura o Pool de processos. O número de processos pode ser ajustado.
        

    def executar_dia(self, parametros_sentimento: dict) -> None:

        if not hasattr(self, "executor"):
            self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
            
        self.dia_atual += 1
        #print(f"\n--- Dia {self.dia_atual} ---")
        # ======================================================
        # 0. APLICAR CHOQUE ECONÔMICO (caso seja o dia do evento)
        # ======================================================

        choque_cfg = parametros_sentimento  # acessa o dicionário dentro de sim_params

        if choque_cfg:
            dia_choque = choque_cfg.get("dia", None)
            tipo_choque = choque_cfg.get("tipo", "negativo")
            intensidade_choque = choque_cfg.get("intensidade", 0.2)
            duracao_choque = choque_cfg.get("duracao", 5)
            delta_choque = choque_cfg.get("delta", 0.6)
            prob_choque_diario = choque_cfg.get("prob_choque_diario", 0.05)

            # Aplica o choque apenas no dia especificado
            if self.dia_atual == dia_choque:
                print(f"\n💥 Choque {tipo_choque.upper()} aplicado no dia {self.dia_atual}")
                for agente in self.agentes:
                    agente.aplicar_choque(
                        tipo_choque=tipo_choque,
                        intensidade=intensidade_choque,
                        duracao=duracao_choque,
                        delta=delta_choque
                    )

                # 🔥 Ajusta a notícia conforme o tipo do choque
                if tipo_choque == "negativo":
                    self._choque_news = -(2 + intensidade_choque)
                elif tipo_choque == "positivo":
                    self._choque_news = +(2 + intensidade_choque)

            #  choques aleatórios leves em qualquer dia

            if random.random() < prob_choque_diario:
                tipo_random = random.choice(["negativo", "negativo","negativo","negativo","positivo"])
                intensidade_random = np.random.uniform(0.3, 0.7)
                duracao_random = np.random.randint(5, 9)
                delta_random = np.random.uniform(0.6, 0.9)

                print(f"⚡ Choque aleatório {tipo_random} no dia {self.dia_atual} (intensidade {intensidade_random:.2f})")

                if tipo_random == "negativo":
                    self._choque_news = -(3 + intensidade_random)
                elif tipo_random == "positivo":
                    self._choque_news = +(3 + intensidade_random)

                for agente in self.agentes:
                    agente.aplicar_choque(
                        tipo_choque=tipo_random,
                        intensidade=intensidade_random,
                        duracao=duracao_random,
                        delta=delta_random
                    )

        # Atualiza dissipação dos choques ativos em todos os agentes
        for agente in self.agentes:
            if hasattr(agente, "atualizar_choque"):
              agente.atualizar_choque()



        # 1. Gerar Notícias
        try:
            self.news = self.midia.gerar_noticia()
            # 💥 Se houve choque, ele altera a notícia
            if hasattr(self, "_choque_news") and self._choque_news is not None:
                self.news = self._choque_news
                self._choque_news = None  # reseta após o uso

            self.historico_news.append(self.news)
            #print(f"[Mercado] Notícia do dia: {self.news:.2f}")
        except StopIteration:
             #print(f"[Mercado] Fim da simulação de notícias no dia {self.dia_atual}.")
             # Decida o que fazer quando as notícias acabarem, talvez manter a última ou definir 0.
             self.news = self.historico_news[-1] if self.historico_news else 0
             pass # Continua a simulação sem novas notícias

        # 2. Distribuir Dividendos (se for dia de dividendo)
        if self.dia_atual % self.dividendos_frequencia == 0:
            dividendos_por_cota = self.fii.distribuir_dividendos()
            #print(f"[Mercado] Dividendos distribuídos: R${dividendos_por_cota:,.4f} por cota.")

            # Agentes recebem dividendos
            for agente in self.agentes:
                # Adiciona dividendos ao caixa
                # Certifica-se de que a quantidade de cotas é numérica antes da multiplicação
                cotas_do_agente = agente.carteira.get("FII", 0)
                if not isinstance(cotas_do_agente, (int, float)):
                     # Log ou handle unexpectedly typed data if necessary
                     print(f"Warning: Agente {agente.id} tem quantidade de cotas não numérica: {cotas_do_agente}. Treating as 0 for dividend calculation.")
                     cotas_do_agente = 0

                agente.caixa += cotas_do_agente * dividendos_por_cota
                agente.saldo = agente.caixa # Sincroniza saldo com caixa

        # 3. Atualizar Imóveis e Investir (se for dia de atualização)
        if self.dia_atual % self.atualizacao_imoveis_frequencia == 0:
            # Assumindo que você quer usar a expectativa de inflação média ou do BC para atualizar
            # Usando a expectativa do Banco Central para esta atualização do FII
            inflacao_para_atualizacao = self.banco_central.expectativa_inflacao
            self.fii.atualizar_imoveis_investir(inflacao_para_atualizacao)

        # 4. Preparar dados para paralelização (Agentes)
        agentes_data_for_pool = []
        agentes_dict = {agente.id: agente for agente in self.agentes} # Para acesso rápido no processo principal

        # Criar snapshots dos dados que os processos paralelos precisam
        mercado_snapshot = {
            'volatilidade_historica': self.volatilidade_historica,
            'news': self.news,
            'fii_preco_cota': self.fii.preco_cota, # Preço do dia anterior
            'fii_num_cotas': self.fii.num_cotas,
            'fii_caixa': self.fii.caixa, # Caixa do FII no início do dia
            'fii_historico_dividendos_ultimo': self.fii.historico_dividendos[-1] if self.fii.historico_dividendos else 0.0,
            # Adicionar expectativa de inflação e prêmio *médios* ou do BC para _calcular_preco_esperado_standalone
            # Como a expectativa do agente muda com o sentimento, usar a média OU a do BC
            # Usaremos a do BC ou a média do dia anterior para evitar ciclos
             'agente_expectativa_inflacao': np.mean([a.expectativa_inflacao for a in self.agentes]) if self.agentes else self.banco_central.expectativa_inflacao,
             'agente_expectativa_premio': np.mean([a.expectativa_premio for a in self.agentes]) if self.agentes else self.banco_central.premio_risco,
             'fii_fluxo_aluguel': self.fii.calcular_fluxo_aluguel() # Calcular fluxo para o dividendo
        }

        banco_central_snapshot = {
            'taxa_selic': self.banco_central.taxa_selic,
            'expectativa_inflacao': self.banco_central.expectativa_inflacao,
            'premio_risco': self.banco_central.premio_risco,
        }

        parametros_sentimento_snapshot = parametros_sentimento # Parâmetros de sentimento são estáticos por dia

        # Coletar dados de cada agente para passar ao pool
        for agente in self.agentes:
            agente_state = {
                'id': agente.id,
                'literacia_financeira': agente.LF,
                'caixa': agente.caixa, # Caixa no início do dia
                'cotas': agente.cotas, # Cotas no início do dia
                'saldo': agente.saldo, # Saldo no início do dia
                'carteira': agente.carteira.copy(), # Copia da carteira
                'sentimento': agente.sentimento, # Sentimento do dia anterior
                'historico_precos': agente.historico_precos.tolist(), # Converter np array para list para serialização
                'historico_riqueza': agente.historico_riqueza.tolist(), # Converter np array para list para serialização
                'expectativa_inflacao': agente.expectativa_inflacao, # Expectativa do dia anterior
                'expectativa_premio': agente.expectativa_premio,   # Expectativa do dia anterior
                'prob_negociar'      : agente.prob_negociar,  # 🔧 MODIFICAÇÃO --------------------------------------------------------------
            }

            # Coletar sentimentos dos vizinhos (snapshot)
            # Usar o sentimento do dia anterior dos vizinhos
            vizinhos_sentiments_snapshot = [v.sentimento for v in agente.vizinhos]

            # Passar os parâmetros internos do agente também
            agente_params_snapshot = agente.params

            agentes_data_for_pool.append((agente.id, agente_state, vizinhos_sentiments_snapshot, mercado_snapshot, banco_central_snapshot, parametros_sentimento_snapshot, agente_params_snapshot))

        # 5. Executar cálculos dos agentes em paralelo (Sentimento, Expectativas, Alocação)
        # Usa Pool.map para distribuir o trabalho
        # O timeout pode ser ajustado conforme necessário
        list(self.executor.map(
            lambda ag: ag.calcular_sentimento_risco_alocacao(
                self, ag.vizinhos, parametros_sentimento),
            self.agentes
            ))
        # # 6. Sincronizar dados atualizados de volta para os objetos Agente originais
        # # Iterar sobre os resultados do pool_map
        # for dados_atualizados in agentes_dados_atualizados:
        #      # Adicionado verificação para 'None' retornado por _processar_agente_para_pool em caso de erro
        #     if dados_atualizados is not None:
        #         agente_id = dados_atualizados['id']
        #         agente_original = agentes_dict[agente_id]

        #         # Atualizar os atributos do agente original
        #         agente_original.sentimento = dados_atualizados['sentimento']
        #         agente_original.historico_sentimentos.append(dados_atualizados['historico_sentimentos_novo'])
        #         agente_original.expectativa_inflacao = dados_atualizados['expectativa_inflacao']
        #         agente_original.expectativa_premio = dados_atualizados['expectativa_premio']
        #         agente_original.RD = dados_atualizados['RD']
        #         agente_original.percentual_alocacao = dados_atualizados['percentual_alocacao']

        #         # Outros atributos podem ser atualizados aqui conforme necessário
        #         # Por exemplo, se a criação da intenção de ordem fosse feita no pool,
        #         # você processaria essa intenção aqui para criar a Ordem real e adicioná-la ao OrderBook.
        #     else:
        #         # Lidar com agentes que falharam no processamento paralelo, se necessário
        #         # Por enquanto, apenas printamos o erro dentro da função do pool.
        #         print(f"Warning: Dados para um agente eram None. Possível erro durante o processamento paralelo.")


        # 6. Sincronizar retorno ----------------------------------------------------------------------------------------------



        # 7. Agentes criam ordens (SEQUENCIALMENTE após sentimentos serem atualizados)
        # Esta parte é sequencial pois a ordem criada depende do estado final do agente
        # (incluindo o sentimento e alocação recém-atualizados) e interage com o OrderBook
        self.order_book = OrderBook() # Limpa o OrderBook do dia anterior

        self.order_book = OrderBook()  # limpa livro
        for agente in self.agentes:
            if random.random() < agente.prob_negociar:
                ordem = agente.criar_ordem(self, parametros_sentimento)
                if ordem:
                    self.order_book.adicionar_ordem(ordem)


        # for agente in self.agentes:
        #     # Verificar se o agente tem probabilidade de negociar
        #     if random.random() < agente.prob_negociar:
        #         ordem = agente.criar_ordem(self, parametros_sentimento)
        #         if ordem:
        #             self.order_book.adicionar_ordem(ordem)

        # 8. Executar Ordens no OrderBook (SEQUENCIAL)
        # O casamento de ordens é inerentemente sequencial
        #print("[Mercado] Executando ordens...")
        self.order_book.executar_ordens("FII", self) # Passa o mercado para que a transação possa atualizar o preço do FII

        # 9. Atualizar histórico de preços do FII (já feito na execução de ordens)
        # O preço do FII é atualizado dentro de Transacao.executar()
        # Adiciona o novo preço ao histórico do FII e dos Agentes
        #self.fii.historico_precos.append(self.fii.preco_cota)

        for agente in self.agentes:
             # Atualizar histórico de preços e riqueza do agente
             # Ensure historico_precos is a numpy array before appending
             if not isinstance(agente.historico_precos, np.ndarray):
                 agente.historico_precos = np.array(agente.historico_precos)
             agente.historico_precos = np.append(agente.historico_precos, self.fii.preco_cota)

             # Atualiza a riqueza após o fechamento do mercado (incluindo mudanças no preço da cota)
             agente.atualizar_historico(self.fii.preco_cota)

        # 10. Calcular Volatilidade Histórica para o PRÓXIMO dia
        # Usa uma janela configurável dos retornos históricos
        # Retornos diários são calculados APÓS a execução das ordens
        if len(self.fii.historico_precos) > 1:
            # Calcular retornos para a janela de volatilidade
            window_vol = parametros_sentimento.get("window_volatilidade", 200) # Usar um parâmetro para janela
            if len(self.fii.historico_precos) >= window_vol + 1:
                precos_janela = np.array(self.fii.historico_precos[-window_vol -1 :]) # Preços para calcular window_vol retornos
                retornos_janela = np.diff(np.log(precos_janela))
                self.volatilidade_historica = np.std(retornos_janela) * (252 ** 0.5) # Anualizada
            elif len(self.fii.historico_precos) > 1:
                 # Se não há dados suficientes para a janela, usar o que tem
                 precos_janela = np.array(self.fii.historico_precos)
                 retornos_janela = np.diff(np.log(precos_janela))
                 self.volatilidade_historica = np.std(retornos_janela) * (252 ** 0.5) if len(retornos_janela) > 0 else 0.0
            else:
                 self.volatilidade_historica = 0.1 # Valor inicial ou default

        #print(f"[Mercado] Preço de Fechamento: R${self.fii.preco_cota:,.2f}, Vol. Histórica (Prox Dia): {self.volatilidade_historica:.4f}")

    def fechar_pool(self):
        self.pool.close()
        self.pool.join()
        print("Pool de processos fechado.")


# Funções auxiliares (já existiam no seu código ou foram refatoradas/adicionadas)
def calcular_sentimento_medio(agentes: list) -> float:
    # Vetorização da média dos sentimentos dos agentes
    if not agentes:
        return 0.0
    sentimentos = np.array([agente.sentimento for agente in agentes])
    return np.mean(sentimentos)



