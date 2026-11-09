#!/usr/bin/env python3
"""
Calibração da homografia H0 por AJUSTE DE LINHAS (alternativa ao clique de 4 marcos).

Por que: no frame0 deste clipe a câmera está fechada na grande área da direita;
os marcos "clicáveis" (quinas) ficam todos numa faixa estreita da imagem e a H0
obtida só com eles extrapola mal (a lateral do campo projetada saía ~15 m errada).
As LINHAS, por outro lado, aparecem longas no quadro (lateral, linha de fundo,
frente e lateral da grande área, frente da pequena área).

Método
  1. Realça pixels de linha: cinza − mediana 15×15 > 12 (linha branca mais clara que a grama ao redor).
  2. Para cada linha do campo, coleta os pixels realçados numa faixa de ±3 px em torno
     de um segmento aproximado (definido à mão em LINHAS, abaixo).
  3. Estima H (8 parâmetros) por mínimos quadrados robustos (soft-L1): cada pixel de
     uma linha "x = c" deve projetar com x = c (idem para "y = c").
  4. Inicialização: H por 4 marcos clicados (CLIQUES).
  5. Validação independente: a meia-lua da grande área (NÃO usada no ajuste) é
     reprojetada e desenhada sobre a imagem.

Saídas: <saida>/calibracao.json (mesmo formato de homografia.py calibrar, + campo "metodo")
        figuras/calibracao_grade.jpg e figuras/calibracao_linhas_campo.jpg

Uso: python calibrar_linhas.py --saida ../saidas/jogo1
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from campo import COMPRIMENTO, LARGURA, MARCOS

# (eixo, valor em metros, [segmentos aproximados em pixels no frame0])
LINHAS = [
    ("x", 105.0, [((900, 203), (1270, 288))]),   # linha de fundo (direita)
    ("x", 88.5, [((345, 163), (850, 325))]),     # frente da grande área
    ("y", 54.16, [((870, 325), (1110, 258))]),   # lateral (inferior) da grande área
    ("y", 68.0, [((640, 715), (1270, 425))]),    # linha lateral (perto da câmera)
    ("x", 99.5, [((610, 165), (840, 218))]),     # frente da pequena área
]
# Marcos clicados no frame0 (usados para inicializar e para reportar o erro de reprojeção)
CLIQUES = {"ga_dir_sup": (337, 160), "ga_dir_inf": (860, 328),
           "ga_dir_inf_linha": (1120, 255), "pa_dir_inf": (852, 221)}


def pixels_de_linha(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    fundo = cv2.medianBlur(g.astype(np.uint8), 15).astype(np.int16)
    ys, xs = np.nonzero((g - fundo) > 12)
    return np.stack([xs, ys], 1).astype(float)


def faixa(P, a, b, largura=3):
    a, b = np.array(a, float), np.array(b, float)
    L = np.linalg.norm(b - a)
    d = (b - a) / L
    n = np.array([-d[1], d[0]])
    t, s = (P - a) @ d, (P - a) @ n
    return P[(t > 0) & (t < L) & (np.abs(s) < largura)]


def projetar(h, Q):
    H = np.append(h, 1).reshape(3, 3)
    q = np.c_[Q, np.ones(len(Q))] @ H.T
    return q[:, :2] / q[:, 2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    pasta = Path(a.saida)
    fig_dir = pasta / "figuras"
    fig_dir.mkdir(exist_ok=True)
    img = cv2.imread(str(pasta / "frame0.jpg"))

    P = pixels_de_linha(img)
    rng = np.random.default_rng(0)
    dados = []
    for eixo, valor, segs in LINHAS:
        Q = np.vstack([faixa(P, s0, s1) for s0, s1 in segs])
        Q = Q[rng.choice(len(Q), min(300, len(Q)), replace=False)]
        dados.append((eixo, valor, Q))

    # Inicialização robusta: ajusta cada linha NA IMAGEM (PCA com descarte de outliers)
    # e usa as interseções entre elas como correspondências — inclusive interseções
    # fora da imagem (ex.: frente da grande área × lateral = (88,5; 68)), que espalham
    # os pontos pelo campo e evitam a extrapolação ruim dos 4 cliques.
    def reta(Q):
        for _ in range(3):
            c = Q.mean(0)
            d = np.linalg.svd(Q - c)[2][0]
            n = np.array([-d[1], d[0]])
            r = np.abs((Q - c) @ n)
            Q = Q[r < max(1.5, np.percentile(r, 80))]
        return np.array([n[0], n[1], -n @ c])          # a·u + b·v + c = 0

    retas = {(e, v): reta(Q) for e, v, Q in dados}
    pares_px, pares_m = [], []
    for (e1, v1), r1 in retas.items():
        for (e2, v2), r2 in retas.items():
            if e1 == "x" and e2 == "y":
                p = np.cross(r1, r2)
                pares_px.append(p[:2] / p[2])
                pares_m.append((v1, v2))
    H_ini, _ = cv2.findHomography(np.float64(pares_px), np.float64(pares_m), 0)
    src = np.float32(list(CLIQUES.values()))
    dst = np.float32([MARCOS[k] for k in CLIQUES])

    def residuos(h):
        return np.concatenate([projetar(h, Q)[:, 0 if e == "x" else 1] - v for e, v, Q in dados])

    # Ajuste em etapas: escala robusta larga -> estreita, descartando pixels que não
    # pertencem à linha (jogadores, ruído) a cada etapa.
    h = (H_ini / H_ini[2, 2]).ravel()[:8]
    for escala, corte in [(3.0, 3.0), (1.0, 1.0), (0.3, 0.5)]:
        sol = least_squares(residuos, h, loss="soft_l1", f_scale=escala)
        h = sol.x
        dados = [(e, v, Q[np.abs(projetar(h, Q)[:, 0 if e == "x" else 1] - v) < corte]) for e, v, Q in dados]
    sol = least_squares(residuos, h, loss="soft_l1", f_scale=0.3)
    H = np.append(sol.x, 1).reshape(3, 3)

    print("Resíduo por linha (m):")
    rms_linhas = {}
    for e, v, Q in dados:
        r = projetar(sol.x, Q)[:, 0 if e == "x" else 1] - v
        rms_linhas[f"{e}={v}"] = round(float(np.sqrt((r ** 2).mean())), 3)
        print(f"  {e} = {v:6.2f}  rms = {rms_linhas[f'{e}={v}']:.2f}  ({len(Q)} px)")

    erro = np.linalg.norm(projetar(sol.x, src.astype(float)) - dst, axis=1)
    print("Erro de reprojeção dos marcos clicados (m):")
    for k, e in zip(CLIQUES, erro):
        print(f"  {k:18s} {e:.2f}")

    json.dump({"metodo": "ajuste robusto por linhas (calibrar_linhas.py)",
               "pixels": [list(map(float, v)) for v in CLIQUES.values()], "marcos": list(CLIQUES),
               "H0": H.tolist(), "erro_reprojecao_m": erro.tolist(), "rms_linhas_m": rms_linhas},
              open(pasta / "calibracao.json", "w"), indent=2)

    # Figura 1: grade de 5 m
    Hinv = np.linalg.inv(H)
    ap_h = lambda M, s: cv2.perspectiveTransform(np.float64(s)[None], M)[0]
    grade = img.copy()
    for x in np.arange(0, COMPRIMENTO + 0.1, 5):
        p0, p1 = np.clip(ap_h(Hinv, [[x, 0], [x, LARGURA]]), -1e4, 1e4).astype(int)
        cv2.line(grade, tuple(map(int, p0)), tuple(map(int, p1)), (0, 255, 255), 1)
    for y in np.arange(0, LARGURA + 0.1, 5):
        p0, p1 = np.clip(ap_h(Hinv, [[0, y], [COMPRIMENTO, y]]), -1e4, 1e4).astype(int)
        cv2.line(grade, tuple(map(int, p0)), tuple(map(int, p1)), (0, 255, 255), 1)
    for v in CLIQUES.values():
        cv2.circle(grade, v, 5, (0, 0, 255), 2)
    cv2.imwrite(str(fig_dir / "calibracao_grade.jpg"), grade)

    # Figura 2: linhas oficiais do campo + meia-lua (validação independente)
    lin = img.copy()

    def seg(p, q, cor=(255, 0, 255)):
        pts = np.float32([[p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t] for t in np.linspace(0, 1, 60)])
        cv2.polylines(lin, [ap_h(Hinv, pts).astype(np.int32)], False, cor, 2, cv2.LINE_AA)

    L, W = COMPRIMENTO, LARGURA
    for p, q in [((0, W), (L, W)), ((L, 0), (L, W)), ((0, 0), (L, 0)),
                 ((L - 16.5, 13.84), (L - 16.5, 54.16)), ((L - 16.5, 54.16), (L, 54.16)), ((L - 16.5, 13.84), (L, 13.84)),
                 ((L - 5.5, 24.84), (L - 5.5, 43.16)), ((L - 5.5, 43.16), (L, 43.16)), ((L - 5.5, 24.84), (L, 24.84))]:
        seg(p, q)
    arco = np.float32([[L - 11 + 9.15 * math.cos(t), W / 2 + 9.15 * math.sin(t)]
                       for t in np.linspace(math.radians(127), math.radians(233), 40)])
    cv2.polylines(lin, [ap_h(Hinv, arco).astype(np.int32)], False, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(lin, "magenta: linhas ajustadas | vermelho: meia-lua (validacao, fora do ajuste)",
                (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(fig_dir / "calibracao_linhas_campo.jpg"), lin)
    print(f"H0 salva em {pasta/'calibracao.json'}")


if __name__ == "__main__":
    main()
