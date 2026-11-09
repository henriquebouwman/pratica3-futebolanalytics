#!/usr/bin/env python3
"""
Q5.2 — Sensibilidade da posse e do nº de passes ao raio de posse (1, 2, 3 m).

Reaproveita as funções de metricas.py sem sobrescrever as saídas principais.
Gera: <saida>/sensibilidade_raio.csv e figuras/sensibilidade_raio.png

Uso: python sensibilidade_raio.py --saida ../saidas/jogo1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from metricas import carregar_bola, carregar_jogadores, detectar_passes, posse_por_quadro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    ap.add_argument("--raios", default="1,1.5,2,2.5,3")
    ap.add_argument("--max-lacuna-posse", type=int, default=None)
    ap.add_argument("--pixels", action="store_true", help="Usa pixels (raios em px; dist. mín. de passe 40 px)")
    a = ap.parse_args()
    pasta = Path(a.saida)
    fps = json.loads((pasta / "meta.json").read_text())["fps"]
    jog, bola = carregar_jogadores(pasta, a.pixels), carregar_bola(pasta, a.pixels)
    dmin = 40.0 if a.pixels else 3.0
    un = "px" if a.pixels else "m"

    linhas = []
    for r in [float(x) for x in a.raios.split(",")]:
        p = posse_por_quadro(jog, bola, r, 3, a.max_lacuna_posse)
        ps = detectar_passes(p, jog, fps, dmin, 4.0)
        pos = p["time_posse"].value_counts(normalize=True)
        linha = {f"raio_{un}": r, "quadros_com_dono_pct": round(100 * p["dono"].notna().mean(), 1)}
        for t in (0, 1):
            pt = ps[ps["time"] == t] if len(ps) else ps
            linha[f"posse_t{t}_pct"] = round(100 * float(pos.get(float(t), 0)), 1)
            linha[f"passes_t{t}"] = len(pt)
            linha[f"precisao_t{t}_pct"] = round(100 * pt["sucesso"].mean(), 1) if len(pt) else None
        linha["passes_total"] = len(ps)
        linhas.append(linha)
    df = pd.DataFrame(linhas)
    df.to_csv(pasta / "sensibilidade_raio.csv", index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(df.iloc[:, 0], df.posse_t0_pct, "o-", color="#d62728", label="Time 0")
    ax[0].plot(df.iloc[:, 0], df.posse_t1_pct, "o-", color="#3b3b3b", label="Time 1")
    ax[0].set(xlabel=f"Raio de posse ({un})", ylabel="Posse (%)", title="Posse × raio")
    ax[1].plot(df.iloc[:, 0], df.passes_t0, "o-", color="#d62728", label="Time 0")
    ax[1].plot(df.iloc[:, 0], df.passes_t1, "o-", color="#3b3b3b", label="Time 1")
    ax[1].plot(df.iloc[:, 0], df.passes_total, "s--", color="gray", label="Total")
    ax[1].set(xlabel=f"Raio de posse ({un})", ylabel="Nº de passes detectados", title="Passes × raio")
    for x in ax:
        x.grid(alpha=.3)
        x.legend()
    fig.tight_layout()
    d = pasta / "figuras"
    d.mkdir(exist_ok=True)
    fig.savefig(d / "sensibilidade_raio.png", dpi=150)


if __name__ == "__main__":
    main()
