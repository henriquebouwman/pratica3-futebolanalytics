#!/usr/bin/env python3
"""
Conferência visual da homografia ao longo do clipe (Q4.2/Q4.3): desenha as linhas
oficiais do campo reprojetadas com H_t = H0·G_t em quadros espaçados e mede o erro
(distância mediana, em px, entre pixels de linha detectados e as linhas reprojetadas).

Gera: figuras/camera_reprojecao.jpg (mosaico) e <saida>/camera_erro_px.csv

Uso: python checar_camera.py --saida ../saidas/jogo1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from camera_linhas import LINHAS, CIRCULO, pixels_linha, proj


def desenhar(fr, Hinv):
    dens_all = []
    for eixo, v, (a, b) in LINHAS:
        t = np.linspace(a, b, 300)
        M = np.c_[np.full_like(t, v), t] if eixo == "x" else np.c_[t, np.full_like(t, v)]
        p = proj(Hinv, M)
        ok = (np.abs(p) < 5000).all(1)
        if ok.sum() > 2:
            cv2.polylines(fr, [p[ok].astype(np.int32)], False, (255, 0, 255), 2, cv2.LINE_AA)
            dens_all.append(p[ok])
    th = np.linspace(0, 2 * np.pi, 400)
    pc = proj(Hinv, np.c_[CIRCULO[0] + CIRCULO[2] * np.cos(th), CIRCULO[1] + CIRCULO[2] * np.sin(th)])
    cv2.polylines(fr, [pc.astype(np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)
    dens_all.append(pc)
    return np.vstack(dens_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--passo-check", type=int, default=15, help="Mede o erro a cada N quadros (1 = todos)")
    a = ap.parse_args()
    pasta = Path(a.saida)
    meta = json.loads((pasta / "meta.json").read_text())
    H0 = np.array(json.loads((pasta / "calibracao.json").read_text())["H0"])
    G = np.load(pasta / "camera.npy")
    tr = pd.read_csv(pasta / "tracks.csv")
    tr = tr[tr["classe"] != "bola"]
    cap = cv2.VideoCapture(meta["video"])
    fps = meta["fps"]

    # erro em todos os quadros múltiplos de 15 (1 s)
    linhas, mosaico = [], []
    alvo = set(np.linspace(0, len(G) - 1, a.n).astype(int))
    for f in range(0, len(G)):
        ok, fr = cap.read()
        if not ok:
            break
        if a.passo_check and f % a.passo_check and f not in alvo:
            continue
        Hinv = np.linalg.inv(H0 @ G[f])
        cx = tr[tr["frame"] == f][["x1", "y1", "x2", "y2"]].to_numpy(float)
        P = pixels_linha(fr, cx)
        vis = fr.copy()
        dens = desenhar(vis, Hinv)
        dens = dens[(dens[:, 0] > 0) & (dens[:, 0] < 1280) & (dens[:, 1] > 0) & (dens[:, 1] < 720)]
        if len(dens) and len(P):
            d, _ = cKDTree(P).query(dens)  # para cada ponto da linha projetada, pixel de linha mais próximo
            erro = float(np.median(d))
        else:
            erro = np.nan
        linhas.append({"frame": f, "t_s": round(f / fps, 2), "erro_mediano_px": round(erro, 2)})
        if f in alvo:
            cv2.putText(vis, f"t={f/fps:5.1f}s  erro mediano={erro:.1f}px", (12, 700),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            mosaico.append(cv2.resize(vis, (640, 360)))
    df = pd.DataFrame(linhas)
    df.to_csv(pasta / "camera_erro_px.csv", index=False)
    print(df["erro_mediano_px"].describe().round(2).to_string())
    while len(mosaico) % 3:
        mosaico.append(np.zeros_like(mosaico[0]))
    grade = np.vstack([np.hstack(mosaico[i:i + 3]) for i in range(0, len(mosaico), 3)])
    d = pasta / "figuras"
    d.mkdir(exist_ok=True)
    cv2.imwrite(str(d / "camera_reprojecao.jpg"), grade)


if __name__ == "__main__":
    main()
