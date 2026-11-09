#!/usr/bin/env python3
"""
Calibrações-semente (H px→m) em quadros com vistas diferentes da câmera, usadas por
camera_linhas.py para RELOCALIZAR quando o rastreamento de câmera se perde
(a transmissão varre o campo inteiro: área direita → meio → área esquerda → ...).

Mesmo método de calibrar_linhas.py (ajuste robusto às linhas), com:
  - quadro 300: grande área ESQUERDA (linha de fundo, lateral de cima, áreas);
  - quadro 450: meio-campo (linha do meio + círculo central).

Gera <saida>/sementes_camera.json  {quadro: H}
Uso: python sementes_camera.py --video ../dados/clip15.mp4 --saida ../saidas/jogo1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from calibrar_linhas import faixa, pixels_de_linha, projetar

SEMENTES = {
    300: {
        "linhas": [("x", 0.0, [((20, 277), (470, 166))]), ("y", 0.0, [((500, 163), (1270, 210))]),
                   ("y", 13.84, [((300, 214), (700, 256))]), ("x", 16.5, [((20, 515), (700, 262))]),
                   ("x", 5.5, [((15, 337), (200, 289))]), ("y", 24.84, [((75, 270), (200, 284))])],
        "pontos": {(490, 162): (0, 0), (285, 213): (0, 13.84), (715, 257): (16.5, 13.84), (210, 285): (5.5, 24.84)},
        "circulo": None,
    },
    450: {
        "linhas": [("x", 52.5, [((764, 160), (749, 715))])],
        "pontos": {(752, 331): (52.5, 24.85), (748, 577): (52.5, 43.15), (322, 452): (43.35, 34.0), (1195, 455): (61.65, 34.0)},
        "circulo": ((52.5, 34.0, 9.15), [(322, 330, 1200, 585)]),  # caixa onde está a elipse
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    out = {}
    for f, cfg in SEMENTES.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        P = pixels_de_linha(fr)
        rng = np.random.default_rng(0)
        dados = []
        for e, v, segs in cfg["linhas"]:
            Q = np.vstack([faixa(P, s0, s1) for s0, s1 in segs])
            dados.append((e, v, Q[rng.choice(len(Q), min(300, len(Q)), replace=False)]))
        if cfg["circulo"]:
            (cx, cy, r), [(x0, y0, x1, y1)] = cfg["circulo"]
            Q = P[(P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1)]
            Q = Q[np.abs(Q[:, 0] - 755) > 12]   # tira a linha do meio
            dados.append(("c", (cx, cy, r), Q[rng.choice(len(Q), min(400, len(Q)), replace=False)]))
        src = np.float64(list(cfg["pontos"].keys()))
        dst = np.float64(list(cfg["pontos"].values()))
        Hi, _ = cv2.findHomography(src, dst, 0)

        def res(h):
            r = []
            for e, v, Q in dados:
                p = projetar(h, Q)
                if e == "x":
                    r.append(p[:, 0] - v)
                elif e == "y":
                    r.append(p[:, 1] - v)
                else:
                    r.append(np.hypot(p[:, 0] - v[0], p[:, 1] - v[1]) - v[2])
            return np.concatenate(r)

        h = (Hi / Hi[2, 2]).ravel()[:8]
        for escala, corte in [(3.0, 3.0), (1.0, 1.0), (0.3, 0.5)]:
            h = least_squares(res, h, loss="soft_l1", f_scale=escala).x
            novos = []
            for e, v, Q in dados:
                p = projetar(h, Q)
                d = p[:, 0] - v if e == "x" else p[:, 1] - v if e == "y" else np.hypot(p[:, 0] - v[0], p[:, 1] - v[1]) - v[2]
                novos.append((e, v, Q[np.abs(d) < corte]))
            dados = novos
        h = least_squares(res, h, loss="soft_l1", f_scale=0.3).x
        r = res(h)
        print(f"semente {f}: {len(r)} px, resíduo mediano {np.median(np.abs(r)):.3f} m")
        out[str(f)] = np.append(h, 1).reshape(3, 3).tolist()
    (Path(a.saida) / "sementes_camera.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
