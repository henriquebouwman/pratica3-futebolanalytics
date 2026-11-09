#!/usr/bin/env python3
"""
Rastreamento v2 (offline) sobre as detecções em cache — substitui rastrear_cache.py.

Melhorias (motivadas pelo que vimos no rastreado.mp4 da v1):
  1. Times desde o 1º quadro: o K-Means é ajustado com amostras do CLIPE INTEIRO
     (não só dos 60 primeiros quadros), então não há período "sem time" no início.
  2. Time por track com SUAVIZAÇÃO TEMPORAL: cada detecção recebe o cluster da sua cor;
     ao longo do track aplicamos a moda numa janela móvel. Um track que "pula" de um
     jogador do United para um da Roma (troca de ID) muda de time no ponto certo, em
     vez de ficar preso ao primeiro voto.
  3. BoT-SORT com compensação de movimento de câmera (GMC, fluxo óptico esparso):
     com os pans rápidos da transmissão, o ByteTrack puro perdia a associação por IoU.
  4. Bola: candidatos filtrados ANTES do Kalman — tamanho (4–30 px, quase quadrado),
     gramado em volta, fora do placar e fora do tronco/cabeça de um jogador
     (cabeças e meiões brancos eram falsos positivos frequentes).
  5. Goleiros/staff (goleiros.csv de corrigir_goleiros.py) com time e rótulo próprios.

Saídas: tracks.csv, meta.json, frame0.jpg, rastreado.mp4 (mesmo formato do pipeline).
Uso: python rastrear_v2.py --video ../dados/clip15.mp4 --saida ../saidas/jogo1 --goleiros ../saidas/jogo1/goleiros.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker

from classificador_times import DESCONHECIDO, OUTROS, para_bgr

COR = {0: (40, 40, 220), 1: (60, 60, 60), OUTROS: (0, 215, 255), DESCONHECIDO: (160, 160, 160)}


def moda_movel(v: np.ndarray, janela: int) -> np.ndarray:
    """Moda numa janela centrada (ignora −99 = sem cor)."""
    out = np.empty_like(v)
    h = janela // 2
    for i in range(len(v)):
        w = v[max(0, i - h):i + h + 1]
        w = w[w != -99]
        if len(w):
            vals, cnt = np.unique(w, return_counts=True)
            out[i] = vals[np.argmax(cnt)]
        else:
            out[i] = -99
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--modelo", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tracker", choices=["botsort", "bytetrack"], default="botsort")
    ap.add_argument("--buffer", type=int, default=30, help="Quadros que um track perdido sobrevive")
    ap.add_argument("--janela", type=int, default=15, help="Janela (detecções) da moda de time")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--conf-bola", type=float, default=0.10)
    ap.add_argument("--goleiros", default=None)
    ap.add_argument("--sem-video", action="store_true")
    a = ap.parse_args()
    pasta = Path(a.saida)

    det = pd.concat([pd.read_csv(f) for f in sorted((pasta / "cache").glob("det_*.csv"))], ignore_index=True)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(3)), int(cap.get(4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pes = det[det["classe"] == "pessoa"].reset_index(drop=True)
    bol = det[det["classe"] == "bola"].reset_index(drop=True)

    # ── 1. K-Means com amostras do clipe inteiro ─────────────────────────────
    am = pes[(pes["conf"] >= a.conf) & ((pes["y2"] - pes["y1"]) >= 30)].dropna(subset=["c1"])
    am = am.sample(min(6000, len(am)), random_state=0)
    km = KMeans(n_clusters=a.k, n_init=10, random_state=42).fit(am[["c1", "c2", "c3"]].to_numpy())
    ordem = np.argsort(-np.bincount(km.labels_, minlength=a.k))
    clu2time = {int(c): (i if i < 2 else OUTROS) for i, c in enumerate(ordem)}
    times_bgr = {str(t): para_bgr(km.cluster_centers_[c]) for c, t in clu2time.items()}
    print("Centróides (BGR):", times_bgr, "| amostras:", len(am))
    tem_cor = pes["c1"].notna().to_numpy()
    rot = np.full(len(pes), -99)
    rot[tem_cor] = [clu2time[int(c)] for c in km.predict(pes.loc[tem_cor, ["c1", "c2", "c3"]].to_numpy())]
    pes["rot_det"] = rot

    # ── 2. Rastreamento (+ filtro de bola, que também precisa do quadro) ─────
    base = dict(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
                track_buffer=a.buffer, match_thresh=0.8, fuse_score=True)
    if a.tracker == "botsort":
        tracker = BOTSORT(SimpleNamespace(**base, gmc_method="sparseOptFlow", proximity_thresh=0.5,
                                          appearance_thresh=0.8, with_reid=False, model="auto"))
    else:
        tracker = BYTETracker(SimpleNamespace(**base))
    pes_f = {f: g for f, g in pes.groupby("frame")}
    bol_f = {f: g for f, g in bol.groupby("frame")}
    linhas, bolas = [], []
    for f in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        if f == 0:
            cv2.imwrite(str(pasta / "frame0.jpg"), fr)
        g = pes_f.get(f)
        arr = np.c_[g[["x1", "y1", "x2", "y2", "conf"]].to_numpy(float), np.zeros(len(g))] if g is not None else np.zeros((0, 6))
        out = tracker.update(Boxes(arr, (H, W)), fr)
        for row in out:
            x1, y1, x2, y2, tid, s, _, idx = row[:8]
            r = g.iloc[int(idx)]
            linhas.append((f, int(tid), int(x1), int(y1), int(x2), int(y2), float(s), int(r["rot_det"])))
        # bola: tamanho, placar, gramado em volta, fora do tronco/cabeça de jogadores
        gb = bol_f.get(f)
        if gb is not None and len(gb):
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            verde = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] > 40) & (hsv[..., 2] > 40)
            corpos = g[["x1", "y1", "x2", "y2"]].to_numpy(float) if g is not None else np.zeros((0, 4))
            cand = []
            for r in gb.itertuples():
                w, h = r.x2 - r.x1, r.y2 - r.y1
                cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
                if not (4 <= w <= 30 and 4 <= h <= 30 and 0.5 <= w / max(h, 1) <= 2.0):
                    continue
                if cy < 0.12 * H:
                    continue
                reg = verde[max(int(cy) - 12, 0):int(cy) + 13, max(int(cx) - 12, 0):int(cx) + 13]
                if reg.size == 0 or reg.mean() < 0.35:
                    continue
                dentro = ((corpos[:, 0] <= cx) & (cx <= corpos[:, 2]) &
                          (corpos[:, 1] <= cy) & (cy <= corpos[:, 1] + 0.65 * (corpos[:, 3] - corpos[:, 1])))
                if dentro.any():
                    continue
                cand.append(r)
            alta = [r for r in cand if r.conf >= a.conf]
            sel = sorted(alta if alta else [r for r in cand if r.conf >= a.conf_bola], key=lambda r: -r.conf)[:5]
            for r in sel:
                bolas.append((f, int(r.x1), int(r.y1), int(r.x2), int(r.y2), float(r.conf)))

    tr = pd.DataFrame(linhas, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "rot_det"])

    # ── 3. Time por track: moda móvel ao longo do track ──────────────────────
    tr = tr.sort_values(["track_id", "frame"]).reset_index(drop=True)
    tr["time"] = DESCONHECIDO
    for tid, idx in tr.groupby("track_id").indices.items():
        v = tr.loc[idx, "rot_det"].to_numpy()
        m = moda_movel(v, a.janela)
        if (m == -99).all():
            continue
        s = pd.Series(np.where(m == -99, np.nan, m)).ffill().bfill().astype(int).to_numpy()
        tr.loc[idx, "time"] = s
    papel = {}
    if a.goleiros:
        gk = pd.read_csv(a.goleiros)
        for r in gk.itertuples():
            tr.loc[tr["track_id"] == r.track_id, "time"] = int(r.time_goleiro)
            papel[int(r.track_id)] = r.papel
    print(f"Tracks: {tr['track_id'].nunique()} | duração mediana {tr.groupby('track_id').size().median():.0f} quadros")

    # ── 4. Escreve tracks.csv ─────────────────────────────────────────────────
    with open(pasta / "tracks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_s", "track_id", "classe", "time", "conf", "x1", "y1", "x2", "y2", "px", "py"])
        for r in tr.sort_values(["frame", "track_id"]).itertuples():
            w.writerow([r.frame, f"{r.frame / fps:.3f}", r.track_id, "person", int(r.time), f"{r.conf:.3f}",
                        r.x1, r.y1, r.x2, r.y2, f"{(r.x1 + r.x2) / 2:.1f}", f"{r.y2:.1f}"])
        for f, x1, y1, x2, y2, s in bolas:
            w.writerow([f, f"{f / fps:.3f}", -1, "bola", "", f"{s:.3f}", x1, y1, x2, y2,
                        f"{(x1 + x2) / 2:.1f}", f"{(y1 + y2) / 2:.1f}"])

    meta = {"video": a.video, "modelo": a.modelo, "device": "cpu", "fps": fps, "largura": W, "altura": H,
            "n_frames": n, "inicio_s": 0.0, "imgsz": a.imgsz, "conf": a.conf, "conf_bola": a.conf_bola,
            "k": a.k, "tracker": a.tracker, "track_buffer": a.buffer, "janela_moda_time": a.janela,
            "pipeline": "detectar_cache.py + rastrear_v2.py", "times_bgr": times_bgr}
    (pasta / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # ── 5. Vídeo rastreado ────────────────────────────────────────────────────
    if a.sem_video:
        return
    cap = cv2.VideoCapture(a.video)
    vw = cv2.VideoWriter(str(pasta / "rastreado.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    por_f = {f: g for f, g in tr.groupby("frame")}
    for f in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        g = por_f.get(f)
        if g is not None:
            for r in g.itertuples():
                t = int(r.time)
                p = papel.get(r.track_id)
                txt = (f"GK{t}#{r.track_id}" if p == "goleiro" else f"Staff#{r.track_id}" if p == "staff"
                       else f"T{t}#{r.track_id}" if t >= 0 else f"Arb#{r.track_id}" if t == OUTROS else f"?#{r.track_id}")
                cv2.rectangle(fr, (r.x1, r.y1), (r.x2, r.y2), COR.get(t, COR[DESCONHECIDO]), 2)
                cv2.putText(fr, txt, (r.x1, r.y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COR.get(t, COR[DESCONHECIDO]), 1, cv2.LINE_AA)
        cv2.putText(fr, f"{f / fps:6.1f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(fr)
    vw.release()
    print("rastreado.mp4 ok")


if __name__ == "__main__":
    main()
