#!/usr/bin/env python3
"""
Movimento de câmera (pan/zoom) robusto: ECC quadro a quadro + RE-ANCORAGEM pelas linhas do campo.

Por que: o homografia.py camera (ORB + quadros-chave) divergiu neste clipe em ~7 s
(o gramado tem poucos pontos ORB e as placas de LED são animadas). Aqui:

  1. Movimento relativo t→t−1 por ECC (cv2.findTransformECC, MOTION_HOMOGRAPHY) em
     imagem reduzida (0,5×), usando só o gramado (máscara verde) sem jogadores e sem placar.
     Encadeia: G_t = G_{t−1} · M_t  (G_t leva pixels do quadro t para o quadro 0).
  2. A cada --passo quadros, RE-ANCORA: projeta TODAS as linhas oficiais do campo com a
     estimativa atual H_t = H0·G_t, coleta pixels de linha (realce "cinza − mediana")
     a até ±6 px das linhas projetadas e reajusta H_t por mínimos quadrados robustos
     (mesmo método de calibrar_linhas.py). Aceita o reajuste só se houver linhas nas
     duas direções e resíduo baixo; aí zera a deriva acumulada.

Saídas (mesmo formato do homografia.py): <saida>/camera.npy e camera_cortes.json,
mais camera_ancoragens.csv (quadro, nº de pixels, resíduo, aceito).

Uso: python camera_linhas.py --saida ../saidas/jogo1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from campo import COMPRIMENTO as L, LARGURA as W

# Linhas do campo: (eixo, valor, intervalo no outro eixo)
LINHAS = [("y", 0, (0, L)), ("y", W, (0, L)), ("x", 0, (0, W)), ("x", L, (0, W)), ("x", L / 2, (0, W))]
for x0, s in ((0, 1), (L, -1)):
    LINHAS += [("x", x0 + s * 16.5, (13.84, 54.16)), ("y", 13.84, tuple(sorted((x0, x0 + s * 16.5)))),
               ("y", 54.16, tuple(sorted((x0, x0 + s * 16.5)))),
               ("x", x0 + s * 5.5, (24.84, 43.16)), ("y", 24.84, tuple(sorted((x0, x0 + s * 5.5)))),
               ("y", 43.16, tuple(sorted((x0, x0 + s * 5.5))))]
CIRCULO = (L / 2, W / 2, 9.15)


def pixels_linha(fr, caixas):
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    m = ((g - cv2.medianBlur(g.astype(np.uint8), 15).astype(np.int16)) > 12).astype(np.uint8)
    m[: int(0.12 * m.shape[0])] = 0
    for x1, y1, x2, y2 in caixas.astype(int):
        m[max(y1 - 3, 0):y2 + 3, max(x1 - 3, 0):x2 + 3] = 0
    # mantém só pixels dentro do gramado (fecha as linhas brancas dentro da máscara verde
    # e erode a borda): elimina placas, torcida e o placar
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    gram = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    gram = cv2.morphologyEx(gram, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    gram = cv2.erode(gram, np.ones((9, 9), np.uint8))
    m &= (gram > 0).astype(np.uint8)
    ys, xs = np.nonzero(m)
    return np.stack([xs, ys], 1).astype(float)


def mascara_gramado(fr, caixas, esc):
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    m[: int(0.12 * m.shape[0])] = 0
    for x1, y1, x2, y2 in (caixas * esc).astype(int):
        m[max(y1 - 2, 0):y2 + 2, max(x1 - 2, 0):x2 + 2] = 0
    return m


def proj(H, P):
    q = np.c_[P, np.ones(len(P))] @ H.T
    return q[:, :2] / q[:, 2:3]


def ancorar(H, P, banda=6.0, min_px=150):
    """Reajusta H (px→m) usando pixels de linha perto das linhas projetadas."""
    Hinv = np.linalg.inv(H)
    tree_pts = P
    grupos = []
    for eixo, v, (a, b) in LINHAS:
        t = np.linspace(a, b, 200)
        M = np.c_[np.full_like(t, v), t] if eixo == "x" else np.c_[t, np.full_like(t, v)]
        pm = proj(Hinv, M)
        ok = (pm[:, 0] > -50) & (pm[:, 0] < 1330) & (pm[:, 1] > 60) & (pm[:, 1] < 770)
        if ok.sum() < 10:
            continue
        pm = pm[ok]
        # distância de cada pixel de linha à polilinha projetada (amostrada densamente)
        dens = np.vstack([np.linspace(p, q, 8, endpoint=False) for p, q in zip(pm[:-1], pm[1:])])
        d, _ = cKDTree(dens).query(tree_pts, distance_upper_bound=banda)
        sel = tree_pts[d < banda]
        if len(sel) >= 25:
            grupos.append((eixo, v, sel[np.random.default_rng(0).choice(len(sel), min(250, len(sel)), replace=False)]))
    # círculo central
    th = np.linspace(0, 2 * np.pi, 720)
    pc = proj(Hinv, np.c_[CIRCULO[0] + CIRCULO[2] * np.cos(th), CIRCULO[1] + CIRCULO[2] * np.sin(th)])
    okc = (pc[:, 0] > 0) & (pc[:, 0] < 1280) & (pc[:, 1] > 60) & (pc[:, 1] < 720)
    if okc.sum() > 30:
        d, _ = cKDTree(pc[okc]).query(P, distance_upper_bound=banda)
        sel = P[d < banda]
        if len(sel) >= 25:
            grupos.append(("c", 0, sel[np.random.default_rng(0).choice(len(sel), min(250, len(sel)), replace=False)]))

    eixos = {g[0] for g in grupos}
    n = sum(len(g[2]) for g in grupos)
    if n < min_px or not ({"x", "c"} & eixos) or not ({"y", "c"} & eixos) or len(grupos) < 3:
        return None, n, np.nan

    def res(h):
        Hh = np.append(h, 1).reshape(3, 3)
        r = []
        for e, v, Q in grupos:
            p = proj(Hh, Q)
            if e == "x":
                r.append(p[:, 0] - v)
            elif e == "y":
                r.append(p[:, 1] - v)
            else:
                r.append(np.hypot(p[:, 0] - CIRCULO[0], p[:, 1] - CIRCULO[1]) - CIRCULO[2])
        return np.concatenate(r)

    h0 = (H / H[2, 2]).ravel()[:8]
    sol = least_squares(res, h0, loss="soft_l1", f_scale=0.3)
    r = res(sol.x)
    med = float(np.median(np.abs(r)))
    return np.append(sol.x, 1).reshape(3, 3), n, med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    ap.add_argument("--passo", type=int, default=5, help="Re-ancora a cada N quadros")
    ap.add_argument("--esc", type=float, default=0.5)
    ap.add_argument("--max-res", type=float, default=0.35, help="Mediana do resíduo (m) para aceitar")
    a = ap.parse_args()
    pasta = Path(a.saida)
    meta = json.loads((pasta / "meta.json").read_text())
    H0 = np.array(json.loads((pasta / "calibracao.json").read_text())["H0"])
    tr = pd.read_csv(pasta / "tracks.csv")
    tr = tr[tr["classe"] != "bola"]
    caixas = {f: g[["x1", "y1", "x2", "y2"]].to_numpy(float) for f, g in tr.groupby("frame")}

    cap = cv2.VideoCapture(meta["video"])
    n = meta["n_frames"]
    G = np.zeros((n, 3, 3))
    S = np.diag([a.esc, a.esc, 1.0])
    Si = np.linalg.inv(S)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    ant = ant_m = None
    Wp = np.eye(3, dtype=np.float32)
    log = []
    cortes, falhas = [], 0
    perdidos, nreloc = 0, 0
    biblioteca = []
    arq_s = pasta / 'sementes_camera.json'
    sementes = {int(k): np.array(v) for k, v in json.loads(arq_s.read_text()).items()} if arq_s.exists() else {}
    for f in range(n):
        ok, fr = cap.read()
        if not ok:
            G = G[:f]
            break
        cx = caixas.get(f, np.empty((0, 4)))
        peq = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), None, fx=a.esc, fy=a.esc)
        peq = cv2.GaussianBlur(peq, (3, 3), 0)
        m = cv2.resize(mascara_gramado(fr, cx, 1.0), (peq.shape[1], peq.shape[0]), interpolation=cv2.INTER_NEAREST)
        if f == 0:
            G[0] = np.eye(3)
        else:
            try:
                # inicialização por correlação de fase (aguenta pans rápidos, onde o ECC
                # sozinho a partir da identidade perde o alinhamento)
                (dx, dy), resp = cv2.phaseCorrelate(np.float32(ant), np.float32(peq))
                Wp = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float32) if resp > 0.05 else np.eye(3, dtype=np.float32)
                _, Wp = cv2.findTransformECC(ant, peq, Wp, cv2.MOTION_HOMOGRAPHY, crit, ant_m, 5)
                M = Si @ np.linalg.inv(Wp.astype(float)) @ S   # px(t) → px(t−1), escala cheia
            except cv2.error:
                falhas += 1
                Wp = np.eye(3, dtype=np.float32)
                M = np.eye(3)
            G[f] = G[f - 1] @ M
        # re-ancoragem pelas linhas
        if f % a.passo == 0:
            H = H0 @ G[f]
            P = pixels_linha(fr, cx)
            Hn, npx, med, aceito = None, 0, np.nan, False
            # bandas crescentes: se a deriva cresceu (sem ancorar há tempo), procura mais longe
            for banda in (6.0, 14.0, 30.0):
                Hc, npx, med = ancorar(H, P, banda=banda)
                if Hc is not None and med <= a.max_res * (1 if banda == 6.0 else 2):
                    Hc2, npx2, med2 = ancorar(Hc, P, banda=5.0)   # refinamento fino
                    if Hc2 is not None and med2 <= a.max_res:
                        Hn, npx, med, aceito = Hc2, npx2, med2, True
                        break
            perdidos = 0 if aceito else perdidos + 1
            # RELOCALIZAÇÃO: perdido há ≥ 4 tentativas → busca deslocamentos/zoom da imagem
            if not aceito and perdidos >= 3 and perdidos % 4 == 3:
                melhor = None
                Ps = P[np.random.default_rng(f).choice(len(P), min(4000, len(P)), replace=False)] if len(P) else P
                # hipóteses: sementes + poses já aceitas (a câmera revisita as mesmas vistas)
                # 1º: poses já vistas sem deslocamento; 2º (se nada servir): com deslocamentos
                hip = [(Hc0, 0) for Hc0 in list(sementes.values()) + biblioteca[-30:]]
                hip += [(Hc0, dx) for Hc0 in list(sementes.values()) + biblioteca[-4:]
                        for dx in (-150, 150, -300, 300, -450, 450)]
                for Hc0, dx in hip:
                    if melhor is not None and dx != 0:
                        break
                    A = np.array([[1.0, 0, dx], [0, 1, 0], [0, 0, 1]])
                    Hc, n1, m1 = ancorar(Hc0 @ A, Ps, banda=25.0, min_px=60)
                    if Hc is not None and m1 <= 0.5 and (melhor is None or m1 < melhor[1]):
                        melhor = (Hc, m1)
                if melhor is not None:
                    Hc2, npx2, med2 = ancorar(melhor[0], P, banda=5.0)
                    if Hc2 is not None and med2 <= a.max_res and npx2 >= 300:
                        Hn, npx, med, aceito, perdidos = Hc2, npx2, med2, True, 0
                        nreloc += 1
            if f in sementes:      # calibração manual por linhas neste quadro
                Hn, aceito, perdidos = sementes[f], True, 0
            if aceito:
                G[f] = np.linalg.inv(H0) @ Hn
                if f % 15 == 0:
                    biblioteca.append(Hn)
            log.append([f, npx, round(med, 3) if med == med else "", int(aceito)])
        G[f] /= G[f][2, 2]
        ant, ant_m = peq, m
        if f % 100 == 0:
            ac = sum(r[3] for r in log)
            print(f"quadro {f}/{n}  ancoragens aceitas {ac}/{len(log)}  relocalizacoes {nreloc}", flush=True)

    np.save(pasta / "camera.npy", G)
    json.dump({"cortes_frames": cortes, "metodo": "ECC + re-ancoragem por linhas (camera_linhas.py)"},
              open(pasta / "camera_cortes.json", "w"))
    with open(pasta / "camera_ancoragens.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "n_pixels_linha", "residuo_mediano_m", "aceito"])
        w.writerows(log)
    ac = sum(r[3] for r in log)
    print(f"camera.npy salvo ({len(G)} quadros). ECC falhas: {falhas}. Ancoragens aceitas: {ac}/{len(log)}. Relocalizacoes: {nreloc}")


if __name__ == "__main__":
    main()
