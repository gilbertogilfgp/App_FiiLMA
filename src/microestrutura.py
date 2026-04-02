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
from scipy.optimize import curve_fit
from statsmodels.graphics.gofplots import qqplot
from multiprocessing import Pool
import random
import traceback # Importar para printar o traceback em caso de erro

# Importar nas classes quando usar função standalone

warnings.filterwarnings("ignore")





@dataclass
class Ordem:
    tipo: str            # "compra" ou "venda"
    agente: "Agente"     # Agente que criou a ordem
    ativo: str           # Nome do ativo (ex.: "FII")
    preco_limite: float  # Preço máximo (para compra) ou mínimo (para venda)
    quantidade: int      # Quantidade a negociar


@dataclass
class Transacao:
    comprador: "Agente"
    vendedor: "Agente"
    ativo: str
    quantidade: int
    preco_execucao: float

    def executar(self) -> None:
        valor_total = self.quantidade * self.preco_execucao
        self.comprador.saldo -= valor_total
        self.comprador.caixa -= valor_total
        self.vendedor.saldo += valor_total
        self.vendedor.caixa += valor_total
        # Usar .get com valor padrão é bom para evitar KeyError
        self.comprador.carteira[self.ativo] = (
            self.comprador.carteira.get(self.ativo, 0) + self.quantidade
        )
        if self.ativo in self.vendedor.carteira:
            self.vendedor.carteira[self.ativo] -= self.quantidade
            if self.vendedor.carteira[self.ativo] <= 0:
                del self.vendedor.carteira[self.ativo]




class OrderBook:
    def __init__(self, params: dict = None) -> None:
        self.ordens_compra = {}
        self.ordens_venda = {}
        self.params = params if params is not None else {}

    def adicionar_ordem(self, ordem: Ordem) -> None:
        if ordem.tipo == "compra":
            self.ordens_compra.setdefault(ordem.ativo, []).append(ordem)
        elif ordem.tipo == "venda":
            self.ordens_venda.setdefault(ordem.ativo, []).append(ordem)

    def executar_ordens(self, ativo: str, mercado: "Mercado") -> None:
        if ativo in self.ordens_compra and ativo in self.ordens_venda:
            # Ordena as ordens: compras por preço limite decrescente e vendas por crescente.
            # Esta ordenação é eficiente e necessária para o casamento de ordens.
            self.ordens_compra[ativo].sort(key=lambda x: x.preco_limite, reverse=True)
            self.ordens_venda[ativo].sort(key=lambda x: x.preco_limite)
            while self.ordens_compra[ativo] and self.ordens_venda[ativo]:
                ordem_compra = self.ordens_compra[ativo][0]
                ordem_venda = self.ordens_venda[ativo][0]
                if ordem_compra.preco_limite >= ordem_venda.preco_limite:
                    # Possibilidade de parametrizar o método de cálculo do preço de execução.
                    preco_execucao = (ordem_compra.preco_limite + ordem_venda.preco_limite) / 2
                    quantidade_exec = min(ordem_compra.quantidade, ordem_venda.quantidade)
                    transacao = Transacao(
                        comprador=ordem_compra.agente,
                        vendedor=ordem_venda.agente,
                        ativo=ativo,
                        quantidade=quantidade_exec,
                        preco_execucao=preco_execucao,
                    )
                    transacao.executar()
                    # Atualiza o preço do ativo no FII.
                    mercado.fii.preco_cota = preco_execucao
                    ordem_compra.quantidade -= quantidade_exec
                    ordem_venda.quantidade -= quantidade_exec
                    if ordem_compra.quantidade == 0:
                        self.ordens_compra[ativo].pop(0)
                    if ordem_venda.quantidade == 0:
                        self.ordens_venda[ativo].pop(0)
                else:
                    break

    def imprimir(self) -> None:
        print("== Order Book ==")
        print("Ordens de COMPRA:")
        for ativo, ordens in self.ordens_compra.items():
            print(f" Ativo: {ativo}")
            for ordem in ordens:
                print(f"  -> Agente {ordem.agente.id}: Preço Limite: R${ordem.preco_limite:.2f}, Quantidade: {ordem.quantidade}")
        print("Ordens de VENDA:")
        for ativo, ordens in self.ordens_venda.items():
            print(f" Ativo: {ativo}")
            for ordem in ordens:
                print(f"  -> Agente {ordem.agente.id}: Preço Limite: R${ordem.preco_limite:.2f}, Quantidade: {ordem.quantidade}")






