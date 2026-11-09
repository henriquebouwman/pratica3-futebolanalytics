#!/usr/bin/env python3
"""
Etapa 2 do pipeline em duas fases: rastreamento (ByteTrack) + times (K-Means) sobre
as detecções em cache (detectar_cache.py). Gera os MESMOS arquivos de
detectar_rastrear.py: tracks.csv, meta.json, frame0.jpg e rastreado.mp4.

Diferenças em relação a detectar_rastrear.py (documentadas no README):
  - o ByteTrack recebe as pessoas com conf ≥ 0,10 (usa a 2ª etapa de associação do
    ByteTrack com detecções de baixa confiança); tracks novos só nascem com conf ≥ 0,25;
  - bola: até 5 candidatos/quadro; se houver algum com conf ≥ --conf (0,30) só esses,
    senão os de conf ≥ --conf-bola (0,10) — mesma regra do script original.

Uso: python rastrear_cache.py --video ../dados/clip15.mp4 --saida ../saidas/jogo1 --k 3
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from classificador_times import DESCONHECIDO, OUTROS, para_bgr

COR_TIME = {0: (40, 40, 220), 1: (60, 60, 60), OUTROS: (0, 215, 255), DESCONHECIDO: (160, 160, 160)}
COR_BOLA = (0, 255, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--modelo", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--conf-bola", type=float, default=0.10)
    ap.add_argument("--calib-frames", type=int, default=60)
    ap.add_argument("--votos", type=int, default=10)
    ap.add_argument("--sem-video", action="store_true")
    ap.add_argument("--goleiros", default=None,
                    help="goleiros.csv de corrigir_goleiros.py: esses track_ids viram OUTROS (−2)")
    a = ap.parse_args()
    pasta = Path(a.saida)

    det = pd.concat([pd.read_csv(f) for f in sorted((pasta / "cache").glob("det_*.csv"))], ignore_index=True)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(3)), int(cap.get(4))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    faltando = sorted(set(range(n_frames)) - set(det["frame"]))
    if faltando:
        print(f"AVISO: {len(faltando)} quadros sem detecção em cache (ex.: {faltando[:5]})")

    pes = det[det["classe"] == "pessoa"]
    bol = det[det["classe"] == "bola"]

    # ── Times: K-Means nas cores das pessoas dos primeiros quadros ──────────
    cal = pes[(pes["frame"] < a.calib_frames) & (pes["conf"] >= a.conf)].dropna(subset=["c1"])
    X = cal[["c1", "c2", "c3"]].to_numpy()
    km = KMeans(n_clusters=a.k, n_init=10, random_state=42).fit(X)
    ordem = np.argsort(-np.bincount(km.labels_, minlength=a.k))
    clu2time = {int(c): (i if i < 2 else OUTROS) for i, c in enumerate(ordem)}
    times_bgr = {str(t): para_bgr(km.cluster_centers_[c]) for c, t in clu2time.items()}
    print("Centróides (BGR):", times_bgr, "| amostras:", len(X))

    # ── ByteTrack ───────────────────────────────────────────────────────────
    args = SimpleNamespace(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
                           track_buffer=30, match_thresh=0.8, fuse_score=True)
    tracker = BYTETracker(args)  # track_buffer em quadros (30 a 15 fps = 2 s)
    votos: dict[int, Counter] = defaultdict(Counter)
    fixo: dict[int, int] = {}

    # goleiros.csv: track_id → time do goleiro (0/1) ou −2 (staff fora do campo)
    gkdf = pd.read_csv(a.goleiros) if a.goleiros else pd.DataFrame(columns=["track_id", "papel", "time_goleiro"])
    forcados = dict(zip(gkdf["track_id"], gkdf.get("time_goleiro", pd.Series([OUTROS] * len(gkdf)))))
    papel = dict(zip(gkdf["track_id"], gkdf.get("papel", pd.Series(["goleiro"] * len(gkdf)))))

    def time_de(tid, cor):
        if tid in forcados:
            return int(forcados[tid])
        if tid in fixo:
            return fixo[tid]
        if cor is not None:
            c = int(km.predict(np.asarray(cor, float).reshape(1, -1))[0])
            votos[tid][clu2time[c]] += 1
            if sum(votos[tid].values()) >= a.votos:
                fixo[tid] = votos[tid].most_common(1)[0][0]
        return votos[tid].most_common(1)[0][0] if votos[tid] else DESCONHECIDO

    pes_f = {f: g for f, g in pes.groupby("frame")}
    bol_f = {f: g for f, g in bol.groupby("frame")}
    escritor = None if a.sem_video else cv2.VideoWriter(
        str(pasta / "rastreado.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    arq = open(pasta / "tracks.csv", "w", newline="")
    w = csv.writer(arq)
    w.writerow(["frame", "t_s", "track_id", "classe", "time", "conf", "x1", "y1", "x2", "y2", "px", "py"])
    ativos: set[int] = set()
    for f in range(n_frames):
        ok, fr = cap.read() if escritor or f == 0 else (True, None)
        if f == 0:
            cv2.imwrite(str(pasta / "frame0.jpg"), fr)
        t_s = f / fps
        g = pes_f.get(f)
        atuais = set()
        vis = fr.copy() if escritor else None
        if g is not None and len(g):
            arr = np.c_[g[["x1", "y1", "x2", "y2", "conf"]].to_numpy(float), np.zeros(len(g))]
            cores = g[["c1", "c2", "c3"]].to_numpy(float)
            out = tracker.update(Boxes(arr, (H, W)))
        else:
            out = tracker.update(Boxes(np.zeros((0, 6)), (H, W)))
            cores = None
        for row in out:
            x1, y1, x2, y2, tid, s, _, idx = row[:8]
            tid, idx = int(tid), int(idx)
            cor = cores[idx] if cores is not None and not np.isnan(cores[idx]).any() else None
            tm = int(forcados[tid]) if tid in forcados else (DESCONHECIDO if f < a.calib_frames else time_de(tid, cor))
            atuais.add(tid)
            x1, y1, x2, y2 = (int(v) for v in (x1, y1, x2, y2))
            w.writerow([f, f"{t_s:.3f}", tid, "person", tm, f"{s:.3f}", x1, y1, x2, y2,
                        f"{(x1+x2)/2:.1f}", f"{y2:.1f}"])
            if vis is not None:
                cv2.rectangle(vis, (x1, y1), (x2, y2), COR_TIME[tm], 2)
                if papel.get(tid) == "goleiro":
                    rot = f"GK{tm}#{tid}"
                elif papel.get(tid) == "staff":
                    rot = f"Staff#{tid}"
                else:
                    rot = f"T{tm}#{tid}" if tm >= 0 else f"{'Arb' if tm == OUTROS else '?'}#{tid}"
                cv2.putText(vis, rot, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COR_TIME[tm], 1, cv2.LINE_AA)
        # bola
        gb = bol_f.get(f)
        if gb is not None and len(gb):
            alta = gb[gb["conf"] >= a.conf]
            cand = (alta if len(alta) else gb[gb["conf"] >= a.conf_bola]).sort_values("conf", ascending=False).head(5)
            for k, r in enumerate(cand.itertuples()):
                w.writerow([f, f"{t_s:.3f}", -1, "bola", "", f"{r.conf:.3f}", r.x1, r.y1, r.x2, r.y2,
                            f"{(r.x1+r.x2)/2:.1f}", f"{(r.y1+r.y2)/2:.1f}"])
                # (a bola NÃO é desenhada aqui: os candidatos brutos incluem falsos positivos;
                #  video_radar.py desenha a bola final, depois do filtro do gramado e do Kalman)
        for sumiu in ativos - atuais:   # mesmo comportamento de clf.esquecer()
            votos.pop(sumiu, None)
            fixo.pop(sumiu, None)
        ativos = atuais
        if vis is not None:
            cv2.putText(vis, f"{t_s:6.1f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            escritor.write(vis)
    arq.close()
    if escritor:
        escritor.release()

    meta = {"video": a.video, "modelo": a.modelo, "device": "cpu", "fps": fps, "largura": W, "altura": H,
            "n_frames": n_frames, "inicio_s": 0.0, "imgsz": a.imgsz, "conf": a.conf, "conf_bola": a.conf_bola,
            "k": a.k, "pipeline": "detectar_cache.py + rastrear_cache.py", "times_bgr": times_bgr}
    (pasta / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"{n_frames} quadros → {pasta/'tracks.csv'}")


if __name__ == "__main__":
    main()
