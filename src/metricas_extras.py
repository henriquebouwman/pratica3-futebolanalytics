#!/usr/bin/env python3
"""
Métricas extras (além de posse/passes/eXP), calculadas a partir de tracks_campo.csv
e posse_por_quadro.csv (gerados por homografia.py e metricas.py).

Por quadro e por time (amostrando a cada --passo quadros):
  - n_visiveis      : jogadores do time visíveis na câmera
  - largura_m       : amplitude lateral (P90 − P10 de y)
  - profundidade_m  : amplitude longitudinal (P90 − P10 de x)
  - area_hull_m2    : área do fecho convexo (compactação do bloco)
  - centroide_x     : altura média do time no campo
  - linha_def_x     : x médio dos 3 jogadores mais recuados (pela direção de ataque)
  - controle_pct    : % da região visível controlada (diagrama de Voronoi em grade de 1 m)
Por quadro com dono da bola:
  - pressao_n5m     : adversários a ≤ 5 m do portador

LIMITAÇÃO: a câmera de transmissão mostra só parte do campo; tudo é medido sobre
os jogadores visíveis. A "região visível" do Voronoi é o fecho convexo de todos os
jogadores visíveis no quadro, não o campo inteiro.

Uso: python metricas_extras.py --saida ../saidas/jogo1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath
from scipy.spatial import ConvexHull, cKDTree

from campo import desenhar_campo
from metricas import carregar_jogadores

CORES = {0: "#d62728", 1: "#3b3b3b"}


def area_hull(xy: np.ndarray) -> float:
    if len(xy) < 3:
        return np.nan
    try:
        return float(ConvexHull(xy).volume)  # em 2D, volume = área
    except Exception:
        return np.nan


def controle_voronoi(q: pd.DataFrame, passo_grade: float = 1.0):
    """% da região visível (fecho convexo de todos) mais próxima de cada time."""
    xy = q[["x", "y"]].to_numpy()
    if len(xy) < 4 or q["time"].nunique() < 2:
        return {}, None
    try:
        hull = ConvexHull(xy)
    except Exception:
        return {}, None
    poli = MPath(xy[hull.vertices])
    x0, y0 = xy.min(0)
    x1, y1 = xy.max(0)
    gx, gy = np.meshgrid(np.arange(x0, x1, passo_grade), np.arange(y0, y1, passo_grade))
    g = np.c_[gx.ravel(), gy.ravel()]
    g = g[poli.contains_points(g)]
    if len(g) == 0:
        return {}, None
    _, idx = cKDTree(xy).query(g)
    times = q["time"].to_numpy()[idx]
    return {t: float((times == t).mean() * 100) for t in (0, 1)}, (g, times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    ap.add_argument("--passo", type=int, default=5, help="Amostra 1 a cada N quadros")
    a = ap.parse_args()
    pasta = Path(a.saida)
    fig_dir = pasta / "figuras"
    fig_dir.mkdir(exist_ok=True)

    meta = json.loads((pasta / "meta.json").read_text())
    fps = meta["fps"]
    resumo_t = json.loads((pasta / "metricas_times.json").read_text())
    direcao = {int(k): v for k, v in resumo_t["direcao_ataque"].items()}

    jog = carregar_jogadores(pasta)
    jog = jog[jog["time"] >= 0]
    frames = sorted(jog["frame"].unique())[:: a.passo]

    linhas, snap = [], None
    for f in frames:
        q = jog[jog["frame"] == f]
        ctrl, grade = controle_voronoi(q)
        for t in (0, 1):
            qt = q[q["time"] == t]
            if len(qt) == 0:
                continue
            xs = qt["x"].to_numpy()
            # linha defensiva: 3 mais recuados (ataca para a direita → defende o x pequeno)
            rec = np.sort(xs) if direcao[t] == "direita" else -np.sort(-xs)
            linhas.append({
                "frame": int(f), "t_s": f / fps, "time": t, "n_visiveis": len(qt),
                "largura_m": float(np.subtract(*np.percentile(qt["y"], [90, 10]))) if len(qt) >= 3 else np.nan,
                "profundidade_m": float(np.subtract(*np.percentile(xs, [90, 10]))) if len(qt) >= 3 else np.nan,
                "area_hull_m2": area_hull(qt[["x", "y"]].to_numpy()),
                "centroide_x": float(xs.mean()),
                "linha_def_x": float(rec[:3].mean()) if len(rec) >= 3 else np.nan,
                "controle_pct": ctrl.get(t, np.nan),
            })
        if grade is not None and snap is None and len(q) >= 14:
            snap = (f, q, grade)
    ext = pd.DataFrame(linhas)
    ext.to_csv(pasta / "metricas_extras.csv", index=False)

    # Pressão no portador da bola
    p = pd.read_csv(pasta / "posse_por_quadro.csv").dropna(subset=["dono"])
    pres = []
    jg = jog.set_index("frame")
    for r in p.itertuples():
        if r.frame not in jg.index:
            continue
        q = jg.loc[[r.frame]]
        port = q[q["track_id"] == r.dono]
        if port.empty:
            continue
        adv = q[q["time"] != r.time_dono]
        d = np.hypot(adv["x"] - port["x"].iloc[0], adv["y"] - port["y"].iloc[0])
        pres.append({"frame": r.frame, "time_portador": int(r.time_dono), "pressao_n5m": int((d <= 5).sum())})
    pres = pd.DataFrame(pres)
    pres.to_csv(pasta / "pressao_portador.csv", index=False)

    resumo = {"passo_amostragem": a.passo, "quadros_amostrados": len(frames)}
    for t in (0, 1):
        e = ext[ext["time"] == t]
        pr = pres[pres["time_portador"] == t] if len(pres) else pres
        resumo[f"time{t}"] = {
            "n_visiveis_medio": round(float(e["n_visiveis"].mean()), 1),
            "largura_media_m": round(float(e["largura_m"].mean()), 1),
            "profundidade_media_m": round(float(e["profundidade_m"].mean()), 1),
            "area_hull_media_m2": round(float(e["area_hull_m2"].mean()), 0),
            "linha_def_media_x": round(float(e["linha_def_x"].mean()), 1),
            "controle_voronoi_medio_pct": round(float(e["controle_pct"].mean()), 1),
            "pressao_sofrida_media_n5m": round(float(pr["pressao_n5m"].mean()), 2) if len(pr) else None,
        }
    (pasta / "metricas_extras.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(json.dumps(resumo, indent=2, ensure_ascii=False))

    # Figuras
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for t in (0, 1):
        e = ext[ext["time"] == t].set_index("t_s")
        lab = f"Time {t}"
        axs[0].plot(e["area_hull_m2"].rolling(6, min_periods=1).median(), color=CORES[t], label=lab)
        axs[1].plot(e["linha_def_x"].rolling(6, min_periods=1).median(), color=CORES[t], label=lab)
        axs[2].plot(e["controle_pct"].rolling(6, min_periods=1).median(), color=CORES[t], label=lab)
    axs[0].set_ylabel("Área do bloco (m²)")
    axs[1].set_ylabel("Linha defensiva x (m)")
    axs[2].set_ylabel("Controle Voronoi (%)")
    axs[2].set_xlabel("Tempo no clipe (s)")
    axs[0].legend()
    for ax in axs:
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "extras_series.png", dpi=150)
    plt.close(fig)

    if snap is not None:
        f, q, (g, times) = snap
        ax = desenhar_campo()
        ax.scatter(g[:, 0], g[:, 1], c=[CORES[int(t)] for t in times], s=4, alpha=.25, marker="s")
        for t in (0, 1):
            qt = q[q["time"] == t]
            ax.scatter(qt["x"], qt["y"], c=CORES[t], s=60, edgecolors="white", zorder=5, label=f"Time {t}")
        ax.set_title(f"Controle de espaço (Voronoi) — quadro {f} ({f/fps:.1f} s)")
        ax.legend(loc="upper left")
        plt.savefig(fig_dir / "voronoi_exemplo.png", dpi=150, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
