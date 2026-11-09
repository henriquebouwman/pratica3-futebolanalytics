#!/usr/bin/env python3
"""
Remove candidatos a bola fora do gramado, ANTES do Kalman.

Problema observado: no início do clipe o Kalman "travou" em falsos positivos na
torcida/placas (y ≈ 50–100 px) e chegou a prever a bola fora da imagem (x < 0).
Com a bola errada, ninguém fica a ≤ 35 px dela → quase não há dono da bola e a
posse acumulada fica 100% de um time por ~30 s.

Regra: um candidato só fica se (1) não estiver na faixa do placar (12% de cima) e
(2) houver gramado em volta dele (fração de pixels verdes num anel de 25×25 px ≥ --min-grama).

Uso: python filtrar_bola_gramado.py --video ../dados/clip15.mp4 --saida ../saidas/jogo1
     (depois: python bola_kalman.py --saida ../saidas/jogo1, e --limpar-bola para tirar predições fora da imagem)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--min-grama", type=float, default=0.35)
    ap.add_argument("--limpar-bola", action="store_true", help="Só limpa bola.csv (predições fora da imagem)")
    a = ap.parse_args()
    pasta = Path(a.saida)
    meta = json.loads((pasta / "meta.json").read_text())
    W, H = meta["largura"], meta["altura"]

    if a.limpar_bola:
        b = pd.read_csv(pasta / "bola.csv")
        ok = b["px"].between(0, W - 1) & b["py"].between(0.12 * H, H - 1)
        b[ok].to_csv(pasta / "bola.csv", index=False)
        print(f"bola.csv: removidas {int((~ok).sum())} posições fora da imagem/placar")
        return

    tr = pd.read_csv(pasta / "tracks.csv")
    bola = tr[tr["classe"] == "bola"]
    por_quadro = {f: g.index.to_numpy() for f, g in bola.groupby("frame")}
    manter = np.ones(len(tr), bool)
    cap = cv2.VideoCapture(a.video)
    f = -1
    while True:
        ok, fr = cap.read()
        f += 1
        if not ok:
            break
        if f not in por_quadro:
            continue
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        verde = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] > 40) & (hsv[..., 2] > 40)
        for i in por_quadro[f]:
            x, y = int(tr.at[i, "px"]), int(tr.at[i, "py"])
            if y < 0.12 * H:
                manter[tr.index.get_loc(i)] = False
                continue
            reg = verde[max(y - 12, 0):y + 13, max(x - 12, 0):x + 13]
            if reg.size == 0 or reg.mean() < a.min_grama:
                manter[tr.index.get_loc(i)] = False
    removidos = int((~manter).sum())
    tr[manter].to_csv(pasta / "tracks.csv", index=False)
    print(f"Candidatos a bola: {len(bola)} → removidos {removidos} fora do gramado/placar "
          f"({100 * removidos / max(len(bola), 1):.0f}%)")


if __name__ == "__main__":
    main()
