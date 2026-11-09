#!/usr/bin/env python3
"""
Etapa 1 do pipeline em duas fases (detecção em cache → rastreamento offline).

Por que existe: sem GPU, a detecção YOLO a 1280 px leva ~1 s/quadro. Separar a
detecção (cara, paralelizável, retomável em blocos) do rastreamento (barato,
sequencial) permitiu dividir o clipe entre duas máquinas e sobreviver a
reinícios do ambiente. O resultado final (tracks.csv, meta.json) segue o mesmo
formato de detectar_rastrear.py.

Para cada quadro do intervalo [--ini, --fim) grava, em blocos de --bloco quadros,
<saida>/cache/det_<ini>.csv com:
  frame, classe (pessoa|bola), conf, x1, y1, x2, y2, c1, c2, c3
onde (c1, c2, c3) = cor da camisa em (S·cos H, S·sin H, V) — classificador_times.cor_camisa.

Uso: python detectar_cache.py --video clip15.mp4 --saida ../saidas/jogo1 --ini 0 --fim 2553
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from classificador_times import cor_camisa

PESSOA, BOLA = 0, 32  # ids COCO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--modelo", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.10, help="Limiar mínimo salvo (filtra depois)")
    ap.add_argument("--ini", type=int, default=0)
    ap.add_argument("--fim", type=int, default=10**9)
    ap.add_argument("--bloco", type=int, default=50)
    ap.add_argument("--max-seg", type=float, default=1e9, help="Para após N segundos (retomável)")
    a = ap.parse_args()

    cache = Path(a.saida) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    modelo = YOLO(a.modelo)
    cap = cv2.VideoCapture(a.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fim = min(a.fim, total)
    t0 = time.time()

    for b0 in range(a.ini, fim, a.bloco):
        arq = cache / f"det_{b0:05d}.csv"
        if arq.exists():
            continue
        if time.time() - t0 > a.max_seg:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, b0)
        linhas = []
        for f in range(b0, min(b0 + a.bloco, fim)):
            ok, fr = cap.read()
            if not ok:
                break
            r = modelo.predict(fr, classes=[PESSOA, BOLA], conf=a.conf, imgsz=a.imgsz,
                               device="cpu", verbose=False)[0]
            for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.numpy().astype(int),
                                              r.boxes.cls.numpy().astype(int), r.boxes.conf.numpy()):
                cor = ["", "", ""]
                if c == PESSOA:
                    rec = fr[max(y1, 0):y2, max(x1, 0):x2]
                    v = cor_camisa(rec)
                    if v is not None:
                        cor = [f"{x:.2f}" for x in v]
                linhas.append([f, "pessoa" if c == PESSOA else "bola", f"{s:.3f}", x1, y1, x2, y2, *cor])
        tmp = arq.with_suffix(".tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "classe", "conf", "x1", "y1", "x2", "y2", "c1", "c2", "c3"])
            w.writerows(linhas)
        tmp.rename(arq)  # gravação atômica: bloco só "existe" quando completo
        print(f"bloco {b0} ok ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
