#!/usr/bin/env python3
"""
Figuras de apoio às Partes 2–4:
  - times_cores.png        : cor dos centróides do K-Means (meta.json → times_bgr) + recortes de exemplo por grupo
  - bola_falsos_positivos.png : candidatos a bola REJEITADOS pelo gating do Kalman (prováveis falsos positivos)
  - bola_cobertura.png     : linha do tempo — bola detectada (aceita) × preenchida pelo Kalman × ausente

Uso: python figuras_deteccao.py --saida ../saidas/jogo1
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    pasta = Path(a.saida)
    fig_dir = pasta / "figuras"
    fig_dir.mkdir(exist_ok=True)
    meta = json.loads((pasta / "meta.json").read_text())
    tr = pd.read_csv(pasta / "tracks.csv")
    cap = cv2.VideoCapture(meta["video"])

    def quadro(f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        return cap.read()[1]

    # 1. cores dos times + recortes
    nomes = {"0": "Time 0", "1": "Time 1", "-2": "Outros (árbitro/goleiros)"}
    p = tr[(tr["classe"] != "bola") & (tr["time"].isin([0, 1, -2]))]
    rng = np.random.default_rng(1)
    fig, axs = plt.subplots(3, 7, figsize=(12, 6))
    for i, t in enumerate(["0", "1", "-2"]):
        bgr = meta["times_bgr"][t]
        axs[i, 0].imshow(np.full((10, 10, 3), bgr[::-1], np.uint8))
        axs[i, 0].set_title(nomes[t], fontsize=8)
        amostra = p[(p["time"] == int(t)) & ((p["y2"] - p["y1"]) > 45)]
        amostra = amostra.iloc[rng.choice(len(amostra), min(6, len(amostra)), replace=False)] if len(amostra) else amostra
        for j, r in enumerate(amostra.itertuples()):
            fr = quadro(r.frame)
            rec = fr[max(r.y1, 0):r.y2, max(r.x1, 0):r.x2]
            axs[i, j + 1].imshow(cv2.cvtColor(rec, cv2.COLOR_BGR2RGB))
            axs[i, j + 1].set_title(f"#{r.track_id} f{r.frame}", fontsize=7)
    for ax in axs.ravel():
        ax.axis("off")
    fig.suptitle("K-Means (k=3) em (S·cosH, S·sinH, V): centróide e recortes por grupo")
    fig.tight_layout()
    fig.savefig(fig_dir / "times_cores.png", dpi=130)
    plt.close(fig)

    # 2. falsos positivos de bola: candidatos longe (> 40 px) da bola aceita no quadro
    bola = pd.read_csv(pasta / "bola.csv").set_index("frame")
    cand = tr[tr["classe"] == "bola"].copy()
    cand = cand[cand["frame"].isin(bola.index)]
    cand["d"] = np.hypot(cand["px"].to_numpy() - bola.loc[cand["frame"], "px"].to_numpy(),
                         cand["py"].to_numpy() - bola.loc[cand["frame"], "py"].to_numpy())
    fp = cand[cand["d"] > 40].sort_values("conf", ascending=False)
    fp = fp.drop_duplicates("frame").head(40)
    fp = fp.iloc[np.linspace(0, len(fp) - 1, min(12, len(fp))).astype(int)] if len(fp) else fp
    fig, axs = plt.subplots(2, 6, figsize=(12, 4.6))
    for ax, r in zip(axs.ravel(), fp.itertuples()):
        fr = quadro(r.frame)
        cx, cy = int(r.px), int(r.py)
        rec = fr[max(cy - 40, 0):cy + 40, max(cx - 40, 0):cx + 40].copy()
        ax.imshow(cv2.cvtColor(rec, cv2.COLOR_BGR2RGB))
        ax.set_title(f"f{r.frame} conf={r.conf:.2f}", fontsize=8)
    for ax in axs.ravel():
        ax.axis("off")
    pct = 100 * (cand["d"] > 40).mean() if len(cand) else 0
    fig.suptitle(f"Candidatos a bola NÃO usados pelo Kalman (a >40 px da bola final; {pct:.0f}% dos candidatos) — mais confiáveis primeiro")
    fig.tight_layout()
    fig.savefig(fig_dir / "bola_falsos_positivos.png", dpi=130)
    plt.close(fig)

    # 3. cobertura da bola no tempo
    n = meta["n_frames"]
    est = np.zeros(n)
    est[bola.index[bola["fonte"] == "detectada"]] = 2
    est[bola.index[bola["fonte"] == "kalman"]] = 1
    fig, ax = plt.subplots(figsize=(11, 1.8))
    ax.imshow(est[None, :], aspect="auto", cmap=matplotlib.colors.ListedColormap(["#dddddd", "#f2a541", "#2a9d8f"]),
              extent=[0, n / meta["fps"], 0, 1])
    ax.set_yticks([])
    ax.set_xlabel("Tempo no clipe (s)")
    ax.set_title("Bola: verde = detecção aceita · laranja = predição do Kalman · cinza = sem bola")
    fig.tight_layout()
    fig.savefig(fig_dir / "bola_cobertura.png", dpi=130)
    plt.close(fig)
    # 4. Kalman: um trecho em que a predição acerta e um em que "inventa"
    b2 = bola.reset_index()
    b2["run"] = (b2["fonte"] != b2["fonte"].shift()).cumsum()
    casos = []
    for _, g in b2.groupby("run"):
        if g["fonte"].iloc[0] != "kalman" or len(g) < 4:
            continue
        prox = b2[(b2["frame"] > g["frame"].max()) & (b2["fonte"] == "detectada")].head(1)
        if prox.empty or prox["frame"].iloc[0] - g["frame"].max() > 1:
            continue
        ult = g.iloc[-1]
        erro = float(np.hypot(ult["px"] - prox["px"].iloc[0], ult["py"] - prox["py"].iloc[0]))
        casos.append((erro, g, prox.iloc[0]))
    if casos:
        casos.sort(key=lambda c: c[0])
        fig, axs = plt.subplots(2, 2, figsize=(13, 7.6))
        for lin, (rot, (erro, g, prox)) in enumerate([("ACERTA", casos[0]), ("INVENTA", casos[-1])]):
            for col, (f, pts, titulo) in enumerate([
                    (int(g["frame"].iloc[len(g) // 2]), g, f"predição do Kalman ({len(g)} quadros sem bola aceita)"),
                    (int(prox["frame"]), None, "quadro seguinte: bola detectada")]):
                fr = cv2.cvtColor(quadro(f), cv2.COLOR_BGR2RGB)
                ax = axs[lin, col]
                ax.imshow(fr)
                if pts is not None:
                    ax.plot(pts["px"], pts["py"], "o-", color="#f2a541", ms=4, lw=2, label="trajetória prevista")
                    ax.legend(loc="lower right", fontsize=8)
                else:
                    ax.plot(g["px"].iloc[-1], g["py"].iloc[-1], "x", color="#f2a541", ms=14, mew=3, label="última predição")
                    ax.plot(prox["px"], prox["py"], "o", mfc="none", mec="#2a9d8f", ms=18, mew=3, label="bola detectada")
                    ax.legend(loc="lower right", fontsize=8)
                ax.set_title(f"{rot} · quadro {f} ({f / meta['fps']:.1f} s) · {titulo}" + (f" · erro {erro:.0f} px" if pts is None else ""), fontsize=9)
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / "kalman_acerta_inventa.png", dpi=110)
        plt.close(fig)
    print("figuras ok")


if __name__ == "__main__":
    main()
