# -*- coding: utf-8 -*-
"""
fatos_estilizados.py
====================
Biblioteca de funções para cálculo e visualização dos fatos estilizados
de séries de retornos financeiros.

Fatos implementados:
    01 — Intermitência
    02 — Autocorrelação dos retornos
    03 — Decaimento em lei de potência
    04 — Gaussianidade agregacional
    05 — Caudas pesadas condicionais (resíduos GARCH)

Uso:
    from fatos_estilizados import (
        plot_intermitencia,
        plot_acf_retornos,
        plot_power_law,
        plot_gaussianidade_agregacional,
        calcular_residuos_garch,
    )
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.stats import norm, kurtosis
from statsmodels.graphics.gofplots import qqplot


try:
    from arch import arch_model
except ImportError:
    raise ImportError("Instale a biblioteca arch: pip install arch")


# ─────────────────────────────────────────────────────────────────────────────
# Fato estilizado 01 — Intermitência
# ─────────────────────────────────────────────────────────────────────────────

def plot_intermitencia(log_returns: np.ndarray,
                       titulo: str = "Retorno diário simulado",
                       cor: str = "tab:orange",
                       ax: plt.Axes = None) -> plt.Figure:
    """
    Fato estilizado 01 — Intermitência.

    Plota a série de retornos logarítmicos diários.

    Parâmetros
    ----------
    log_returns : np.ndarray
        Sequência de retornos logarítmicos.
    titulo : str
        Título do gráfico.
    cor : str
        Cor da linha.
    ax : matplotlib.axes.Axes, opcional
        Eixo externo. Se None, cria figura própria.

    Retorna
    -------
    matplotlib.figure.Figure

    Exemplo
    -------
    >>> fig = plot_intermitencia(log_returns[-60:])
    >>> display(fig)

    >>> # Lado a lado
    >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    >>> plt.close(fig)
    >>> plot_intermitencia(retorno_bench, cor="tab:blue", ax=ax1)
    >>> plot_intermitencia(log_returns,   cor="tab:orange", ax=ax2)
    >>> display(fig)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.get_figure()

    ax.plot(log_returns, color=cor, linewidth=0.9)
    ax.axhline(0, color="#adb5bd", linewidth=0.7, linestyle=":")
    ax.set_title(titulo, fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel("Log-retorno")
    ax.set_xlabel("Dias")
    ax.tick_params(axis="x", rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Fato estilizado 02 — Autocorrelação dos retornos
# ─────────────────────────────────────────────────────────────────────────────

def plot_acf_retornos(retornos: np.ndarray,
                      titulo: str = "ACF — Retornos",
                      lags: int = 20,
                      alpha: float = 0.05,
                      cor: str = "tab:blue",
                      ax: plt.Axes = None) -> plt.Figure:
    """
    Fato estilizado 02 — Autocorrelação dos retornos.

    Plota o correlograma (ACF) da série de retornos com bandas de confiança.

    Parâmetros
    ----------
    retornos : np.ndarray
        Série de retornos logarítmicos.
    titulo : str
        Título do gráfico.
    lags : int
        Número de lags.
    alpha : float
        Nível de significância para bandas de confiança.
    cor : str
        Cor das barras.
    ax : matplotlib.axes.Axes, opcional
        Eixo externo. Se None, cria figura própria.

    Retorna
    -------
    matplotlib.figure.Figure

    Exemplo
    -------
    >>> fig = plot_acf_retornos(log_returns, lags=20)
    >>> display(fig)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    sm.graphics.tsa.plot_acf(retornos, lags=lags, ax=ax, alpha=alpha, color=cor)

    ax.title.set_text("")
    ax.set_title(titulo, fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("Lags")
    ax.set_ylabel("ACF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Fato estilizado 03 — Decaimento em lei de potência
# ─────────────────────────────────────────────────────────────────────────────

def _power(x, a, b):
    """Função power law: a / x^b."""
    return a / np.power(x, b)


def plot_power_law(retornos: np.ndarray,
                   titulo: str = "Power law — ACF retornos absolutos",
                   lags: int = 40,
                   alpha: float = 0.05,
                   cor: str = "tab:blue",
                   ax: plt.Axes = None) -> tuple[plt.Figure, np.ndarray]:
    """
    Fato estilizado 03 — Decaimento em lei de potência.

    Plota a ACF dos retornos absolutos e ajusta uma curva power law
    ancorada em (0, 1.0).

    Parâmetros
    ----------
    retornos : np.ndarray
        Série de retornos logarítmicos.
    titulo : str
        Título do gráfico.
    lags : int
        Número de lags.
    alpha : float
        Nível de significância para bandas de confiança.
    cor : str
        Cor das barras da ACF.
    ax : matplotlib.axes.Axes, opcional
        Eixo externo. Se None, cria figura própria.

    Retorna
    -------
    fig : matplotlib.figure.Figure
    popt : np.ndarray
        Parâmetros ajustados [a, b]   da power law.

    Exemplo
    -------
    >>> fig, popt = plot_power_law(log_returns, lags=40)
    >>> display(fig)
    >>> print(f"a={popt[0]:.3f}, b={popt[1]:.3f}")
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    nlags = min(lags, len(retornos) - 2)
    acf_vals = sm.tsa.stattools.acf(np.abs(retornos), nlags=nlags)
    lags_arr = np.arange(1, nlags + 1)

    try:
        popt, _ = curve_fit(_power, lags_arr, acf_vals[1:],
                            p0=[1, 1], maxfev=5000)
    except RuntimeError:
        popt = np.array([np.nan, np.nan])

    sm.graphics.tsa.plot_acf(np.abs(retornos), lags=nlags,
                              ax=ax, alpha=alpha, color=cor)

    ax.title.set_text("")
    ax.set_title(titulo, fontsize=12, fontweight="bold", loc="left")

    if not np.isnan(popt).any():
        x_plot = np.linspace(0.01, nlags, 300)
        curva = np.concatenate([[1.0], _power(x_plot[1:], *popt)])
        ax.plot(np.concatenate([[0], x_plot[1:]]), curva,
                color="red", linewidth=1.5, linestyle="--",
                label=f"Power law: a={popt[0]:.3f}, b={popt[1]:.3f}")
        ax.legend(fontsize=9, framealpha=0.5)

    ax.set_xlabel("Lags")
    ax.set_ylabel("ACF retornos absolutos")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    return fig, popt


# ─────────────────────────────────────────────────────────────────────────────
# Fato estilizado 04 — Gaussianidade agregacional
# ─────────────────────────────────────────────────────────────────────────────

def plot_gaussianidade_agregacional(retornos: np.ndarray,
                                    escalas: list = [1, 5, 21],
                                    labels_escalas: list = ["Diário", "Semanal", "Mensal"],
                                    titulo: str = "Gaussianidade agregacional",
                                    cor: str = "tab:blue") -> tuple[plt.Figure, list]:
    """
    Fato estilizado 04 — Gaussianidade agregacional.

    Para cada escala de agregação, plota o histograma dos retornos com
    curva gaussiana de referência e o QQ-plot correspondente.
    Imprime a curtose de cada escala.

    Parâmetros
    ----------
    retornos : np.ndarray
        Série de retornos logarítmicos diários.
    escalas : list
        Janelas de agregação em dias (ex: [1, 5, 21]).
    labels_escalas : list
        Rótulos para cada escala (ex: ["Diário", "Semanal", "Mensal"]).
    titulo : str
        Título geral da figura.
    cor : str
        Cor dos histogramas e pontos do QQ-plot.

    Retorna
    -------
    fig : matplotlib.figure.Figure
    curtoses : list of float
        Curtose (não-fisher) para cada escala.

    Exemplo
    -------
    >>> fig, curtoses = plot_gaussianidade_agregacional(
    ...     log_returns,
    ...     escalas=[1, 5, 21],
    ...     labels_escalas=["Diário", "Semanal", "Mensal"],
    ...     titulo="Gaussianidade agregacional — IFIX simulado",
    ...     cor="tab:blue"
    ... )
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        n = len(escalas)
        fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
        curtoses = []

        for col, (lag, label) in enumerate(zip(escalas, labels_escalas)):
            # Agregação
            n_blocos = len(retornos) // lag
            agregados = np.array([
                np.sum(retornos[i * lag:(i + 1) * lag])
                for i in range(n_blocos)
            ])

            k = float(kurtosis(agregados, bias=True, fisher=False))
            curtoses.append(k)

            # — Histograma + gaussiana —
            ax_hist = axes[0, col]
            sns.histplot(agregados, bins=50, color=cor, stat="density", ax=ax_hist)
            x = np.linspace(agregados.min(), agregados.max(), 200)
            p = norm.pdf(x, np.mean(agregados), np.std(agregados))
            ax_hist.plot(x, p, color="red", linestyle="--", linewidth=2, label="Gaussiana")
            ax_hist.set_title(f"{label}\nCurtose: {k:.2f}", fontsize=11,
                              fontweight="bold", loc="left")
            ax_hist.set_xlabel(f"Retornos {label.lower()}")
            ax_hist.set_ylabel("Densidade")
            ax_hist.legend(fontsize=9)
            ax_hist.spines["top"].set_visible(False)
            ax_hist.spines["right"].set_visible(False)

            # — QQ-plot —
            ax_qq = axes[1, col]
            qqplot(np.sort(agregados, axis=0), line="s", ax=ax_qq,
                   markerfacecolor=cor, fit=False, alpha=0.6)
            plt.close()
            ax_qq.set_title("QQ-plot", fontsize=10, loc="left")
            ax_qq.set_xlabel("Quantis teóricos")
            ax_qq.set_ylabel("Quantis da amostra")
            ax_qq.set_xlim([-4, 4])
            ax_qq.spines["top"].set_visible(False)
            ax_qq.spines["right"].set_visible(False)
            ax_qq.grid(True, alpha=0.15)

        fig.suptitle(titulo, fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        

    return fig, curtoses


# ─────────────────────────────────────────────────────────────────────────────
# Fato estilizado 05 — Caudas pesadas condicionais (resíduos GARCH)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_residuos_garch(retornos: np.ndarray,
                             escalas: list = [1, 5, 21],
                             p: int = 1,
                             q: int = 1,
                             dist: str = "gaussian") -> list:
    """
    Fato estilizado 05 — Caudas pesadas condicionais.

    Agrega os retornos em cada escala e ajusta um modelo GARCH(p,q).
    Retorna os resíduos padronizados para uso com
    plot_gaussianidade_agregacional.

    Parâmetros
    ----------
    retornos : np.ndarray
        Série de retornos logarítmicos diários.
    escalas : list
        Janelas de agregação em dias (ex: [1, 5, 21]).
    p : int
        Ordem p do GARCH. Default: 1.
    q : int
        Ordem q do GARCH. Default: 1.
    dist : str
        Distribuição dos erros ('gaussian', 't', 'skewt'). Default: 'gaussian'.

    Retorna
    -------
    list of np.ndarray
        Resíduos padronizados para cada escala, na mesma ordem de escalas.

    Exemplo
    -------
    >>> residuos = calcular_residuos_garch(log_returns, escalas=[1, 5, 21])
    >>>
    >>> # Visualizar gaussianidade dos resíduos
    >>> fig, curtoses = plot_gaussianidade_agregacional(
    ...     residuos[0],
    ...     escalas=[1, 5, 21],
    ...     titulo="Caudas pesadas condicionais — resíduos GARCH",
    ... )
    """
    residuos = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for lag in escalas:
            n_blocos = len(retornos) // lag
            agregados = np.array([
                np.sum(retornos[i * lag:(i + 1) * lag])
                for i in range(n_blocos)
            ])

            modelo = arch_model(100 * agregados, vol="GARCH",
                                p=p, q=q, dist=dist)
            fit = modelo.fit(update_freq=0, disp="off")
            resid_pad = pd.Series(
                fit.resid / fit.conditional_volatility
            ).dropna().values
            residuos.append(resid_pad)

    return residuos
