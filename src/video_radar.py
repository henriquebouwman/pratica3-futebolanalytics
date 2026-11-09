#!/usr/bin/env python3
"""
Vídeo de apresentação: rastreado.mp4 + minimapa (radar) do campo em metros + barra de posse.

Lê: rastreado.mp4, tracks_campo.csv, bola_campo.csv, posse_por_quadro.csv, meta.json
Gera: <saida>/apresentacao.mp4 (NÃO versionar; enviar pelo Drive)

Uso: python video_radar.py --saida ../saidas/jogo1 --inicio 0 --fim 90
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from campo import COMPRIMENTO, LARGURA
from metricas import carregar_jogadores

ESC = 4            # px por metro no radar
MARG = 10
COR = {0: (40, 40, 220), 1: (70, 70, 70), -2: (0, 215, 255)}  # BGR


def campo_base():
    w, h = int(COMPRIMENTO * ESC) + 2 * MARG, int(LARGURA * ESC) + 2 * MARG
    img = np.full((h, w, 3), (60, 125, 60), np.uint8)
    P = lambda x, y: (int(MARG + x * ESC), int(MARG + y * ESC))
    b = (235, 235, 235)
    cv2.rectangle(img, P(0, 0), P(COMPRIMENTO, LARGURA), b, 1)
    cv2.line(img, P(COMPRIMENTO / 2, 0), P(COMPRIMENTO / 2, LARGURA), b, 1)
    cv2.circle(img, P(COMPRIMENTO / 2, LARGURA / 2), int(9.15 * ESC), b, 1)
    for x0, s in ((0, 1), (COMPRIMENTO, -1)):
        cv2.rectangle(img, P(x0, LARGURA / 2 - 20.16), P(x0 + s * 16.5, LARGURA / 2 + 20.16), b, 1)
        cv2.rectangle(img, P(x0, LARGURA / 2 - 9.16), P(x0 + s * 5.5, LARGURA / 2 + 9.16), b, 1)
    return img, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    ap.add_argument("--inicio", type=float, default=0)
    ap.add_argument("--fim", type=float, default=90)
    ap.add_argument("--posse", default=None, help="posse_por_quadro.csv alternativo (ex.: versão em pixels)")
    a = ap.parse_args()
    pasta = Path(a.saida)
    meta = json.loads((pasta / "meta.json").read_text())
    fps = meta["fps"]

    jog = carregar_jogadores(pasta)
    bola = pd.read_csv(pasta / "bola_campo.csv").dropna(subset=["x_m"]).set_index("frame")
    posse = pd.read_csv(a.posse or (pasta / "posse_por_quadro.csv")).set_index("frame")
    jf = {f: g for f, g in jog.groupby("frame")}

    cap = cv2.VideoCapture(str(pasta / "rastreado.mp4"))
    W, H = int(cap.get(3)), int(cap.get(4))
    base, P = campo_base()
    rh, rw = base.shape[:2]
    f0, f1 = int(a.inicio * fps), int(a.fim * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    out = cv2.VideoWriter(str(pasta / "apresentacao.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    cont = {0: 0, 1: 0, -1: 0}
    bola_px = pd.read_csv(pasta / "bola.csv").drop_duplicates("frame").set_index("frame")

    for f in range(f0, f1):
        ok, fr = cap.read()
        if not ok:
            break
        radar = base.copy()
        if f in jf:
            for r in jf[f].itertuples():
                cv2.circle(radar, P(r.x, r.y), 5, COR.get(int(r.time), (255, 255, 255)), -1)
                cv2.circle(radar, P(r.x, r.y), 5, (255, 255, 255), 1)
        if f in bola.index:
            b = bola.loc[f]
            b = b.iloc[0] if isinstance(b, pd.DataFrame) else b
            cv2.circle(radar, P(b["x_m"], b["y_m"]), 3, (255, 255, 255), -1)
        tp = posse.loc[f, "time_posse"] if f in posse.index else np.nan
        cont[int(tp) if tp == tp else -1] = cont.get(int(tp) if tp == tp else -1, 0) + 1
        # bola final (Kalman) na imagem: círculo cheio = detectada, vazado = prevista
        if f in bola_px.index:
            bp = bola_px.loc[f]
            c = (int(bp["px"]), int(bp["py"]))
            cv2.circle(fr, c, 9, (0, 255, 255), 2 if bp["fonte"] == "detectada" else 1, cv2.LINE_AA)
        # cola o radar no canto inferior esquerdo, semitransparente
        y0, x0 = H - rh - 10, 10
        roi = fr[y0:y0 + rh, x0:x0 + rw]
        fr[y0:y0 + rh, x0:x0 + rw] = cv2.addWeighted(roi, 0.25, radar, 0.75, 0)
        # barra de posse acumulada (desde o início do trecho exibido)
        tot = max(1, cont[0] + cont[1] + cont[-1])
        p0, p1 = cont[0] / tot, cont[1] / tot
        bx, by, bw = x0, y0 - 28, rw
        cv2.rectangle(fr, (bx, by), (bx + int(bw * p0), by + 18), COR[0], -1)
        cv2.rectangle(fr, (bx + int(bw * p0), by), (bx + int(bw * (p0 + p1)), by + 18), COR[1], -1)
        cv2.rectangle(fr, (bx + int(bw * (p0 + p1)), by), (bx + bw, by + 18), (150, 150, 150), -1)
        cv2.putText(fr, f"Posse  United {100*p0:.0f}% | Roma {100*p1:.0f}% | indef. {100*(1-p0-p1):.0f}%",
                    (bx + 6, by + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out.write(fr)
    out.release()
    print(f"Vídeo salvo em {pasta/'apresentacao.mp4'}")


if __name__ == "__main__":
    main()
