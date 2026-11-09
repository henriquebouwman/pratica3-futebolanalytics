#!/usr/bin/env python3
"""
Corrige goleiros classificados no time errado (Q3.2).

Problema observado no rastreado.mp4: o goleiro do United usa verde-limão. O
classificador remove os pixels de "grama" (H ∈ [35, 85]) antes de calcular a cor
da camisa, então quase toda a camisa do goleiro some e sobram calção/meião
escuros → ele cai no cluster cinza (Roma).

Correção: para cada track, mede em até --amostras recortes a fração de pixels
"verde-limão" na camisa (H ∈ [25, 50], S > 90, V > 170 — bem mais claro e
saturado que o gramado noturno). Tracks acima de --limiar viram OUTROS (−2),
o mesmo grupo de árbitro/goleiros usado com k=3, e saem da posse/passes.

Uso: python corrigir_goleiros.py --video ../dados/clip15.mp4 --saida ../saidas/jogo1
Gera tracks.csv corrigido (original salvo em tracks_antes_goleiro.csv) e goleiros.csv.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def frac_limao(rec):
    """(fração de pixels verde-limão/amarelo vivo na camisa, matiz mediano desses pixels)."""
    h = rec.shape[0]
    cam = rec[int(0.10 * h):int(0.55 * h)]
    if cam.size == 0:
        return np.nan, np.nan
    hsv = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 0] >= 25) & (hsv[..., 0] <= 50) & (hsv[..., 1] > 90) & (hsv[..., 2] > 170)
    return float(m.mean()), (float(np.median(hsv[..., 0][m])) if m.any() else np.nan)


def grama_no_pe(fr, x1, x2, y2):
    """Fração de gramado numa faixa logo abaixo/ao lado do pé: ~1 em campo, ~0 na pista/placas."""
    H, W = fr.shape[:2]
    w = max(x2 - x1, 10)
    reg = fr[min(y2, H - 1):min(y2 + 12, H), max(x1 - w, 0):min(x2 + w, W)]
    if reg.size == 0:
        return np.nan
    hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] > 40)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--amostras", type=int, default=5)
    ap.add_argument("--limiar", type=float, default=0.15)
    a = ap.parse_args()
    pasta = Path(a.saida)
    orig = pasta / "tracks_antes_goleiro.csv"
    tr = pd.read_csv(orig if orig.exists() else pasta / "tracks.csv")
    if not orig.exists():
        tr.to_csv(orig, index=False)
    p = tr[(tr["classe"] != "bola") & ((tr["y2"] - tr["y1"]) > 30)]
    rng = np.random.default_rng(0)
    pedidos = {}
    for tid, g in p.groupby("track_id"):
        for r in g.iloc[rng.choice(len(g), min(a.amostras, len(g)), replace=False)].itertuples():
            pedidos.setdefault(r.frame, []).append((tid, r.x1, r.y1, r.x2, r.y2))
    cap = cv2.VideoCapture(a.video)
    med = {}
    f = -1
    while True:
        ok, fr = cap.read()
        f += 1
        if not ok or f > max(pedidos):
            break
        if f not in pedidos:
            continue
        for tid, x1, y1, x2, y2 in pedidos[f]:
            fl, hue = frac_limao(fr[max(y1, 0):y2, max(x1, 0):x2])
            med.setdefault(tid, []).append((fl, hue, grama_no_pe(fr, x1, x2, y2)))
    res = pd.DataFrame([{"track_id": t, "frac_limao": float(np.nanmedian([x[0] for x in v])),
                         "matiz": float(np.nanmedian([x[1] for x in v])) if any(x[1] == x[1] for x in v) else np.nan,
                         "grama_pe": float(np.nanmedian([x[2] for x in v]))} for t, v in med.items()])
    times = tr[tr["classe"] != "bola"].groupby("track_id")["time"].agg(lambda s: s.value_counts().idxmax())
    res["time_antes"] = res["track_id"].map(times)
    gk = res[res["frac_limao"] > a.limiar].sort_values("frac_limao", ascending=False).copy()
    # Papel: fora do gramado (pista/placas) = staff; no gramado = goleiro.
    # Time do goleiro pelo matiz: verde-limão (H ≥ 36) = United (0); amarelo (H < 36) = Roma (1).
    gk["papel"] = np.where(gk["grama_pe"] < 0.5, "staff", "goleiro")
    gk["time_goleiro"] = np.where(gk["papel"] == "goleiro", np.where(gk["matiz"] >= 36, 0, 1), -2)
    gk.to_csv(pasta / "goleiros.csv", index=False)
    for r in gk.itertuples():
        tr.loc[(tr["track_id"] == r.track_id) & (tr["classe"] != "bola"), "time"] = r.time_goleiro
    tr.to_csv(pasta / "tracks.csv", index=False)
    print(gk.round(3).to_string(index=False))
    print(res["frac_limao"].describe().round(3).to_string())


if __name__ == "__main__":
    main()
