#!/usr/bin/env python3
"""
Refaz e REGISTRA os experimentos citados no README (para que todo número venha de um arquivo):
  - rastreadores: ByteTrack × BoT-SORT+GMC (buffer 2 s e 4 s) → nº de IDs          (Q3.1)
  - k=2 × k=3 no K-Means de cor                                                     (Q3.3)
  - efeito do limiar de confiança da bola (conf-bola) nas detecções brutas          (Q2.3)
  - parâmetros do Kalman (max-lacuna, gate, reinício por confiança) → F1 dos passes,
    no clipe inteiro e em cada metade                                              (Q4.1/Q5.3)
  - raio de posse → F1 dos passes                                                   (Q5.2)
Saída: saidas/experimentos.json

Uso (de dentro de src/): python experimentos.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from metricas import carregar_bola, carregar_jogadores, detectar_passes, posse_por_quadro
from validar_passes import casar

R = Path(__file__).resolve().parent.parent
S = R / "saidas" / "jogo1"
VIDEO = R / "dados" / "clip15.mp4"
GT = pd.read_csv(S / "passes_anotados.csv")
PY = sys.executable


def f1(det, gt):
    vp = len(casar(det.reset_index(drop=True), gt.reset_index(drop=True), 1.0)) if len(det) else 0
    p = vp / len(det) if len(det) else 0
    r = vp / len(gt) if len(gt) else 0
    return round(2 * p * r / (p + r), 3) if vp else 0.0


def f1_metades(passes):
    meio = 85.0
    return {"inteiro": f1(passes, GT),
            "1a_metade": f1(passes[passes["t_ini_s"] < meio], GT[GT["t_s"] < meio]),
            "2a_metade": f1(passes[passes["t_ini_s"] >= meio], GT[GT["t_s"] >= meio])}


def run(*args):
    subprocess.run([PY, *map(str, args)], check=True, capture_output=True, cwd=R / "src")


def main():
    out = {}
    tmp = Path(tempfile.mkdtemp())

    # 1. rastreadores
    trk = []
    for nome, extra in [("ByteTrack (buffer 2 s)", ["--tracker", "bytetrack"]),
                        ("BoT-SORT + GMC (buffer 2 s)", []),
                        ("BoT-SORT + GMC (buffer 4 s)", ["--buffer", "60"])]:
        d = tmp / f"trk{len(trk)}"
        d.mkdir()
        (d / "cache").symlink_to(S / "cache")
        run("rastrear_v2.py", "--video", VIDEO, "--saida", d, "--sem-video", *extra)
        t = pd.read_csv(d / "tracks.csv")
        t = t[t["classe"] == "person"]
        trk.append({"rastreador": nome, "ids": int(t["track_id"].nunique()),
                    "duracao_mediana_quadros": float(t.groupby("track_id").size().median())})
    out["rastreadores"] = trk

    # 2. k=2 × k=3 (mesmas amostras de cor do rastrear_v2)
    det = pd.concat([pd.read_csv(f) for f in sorted((S / "cache").glob("det_*.csv"))])
    pes = det[(det["classe"] == "pessoa") & (det["conf"] >= 0.3) & ((det["y2"] - det["y1"]) >= 30)].dropna(subset=["c1"])
    am = pes.sample(min(6000, len(pes)), random_state=0)[["c1", "c2", "c3"]].to_numpy()
    k3 = KMeans(3, n_init=10, random_state=42).fit(am)
    k2 = KMeans(2, n_init=10, random_state=42).fit(am)
    menor3 = np.argmin(np.bincount(k3.labels_))
    arb = k3.labels_ == menor3                      # amostras do 3º cluster (árbitro/goleiros)
    tam2 = np.bincount(k2.labels_)
    out["k2_vs_k3"] = {
        "k3_tamanhos_clusters": np.bincount(k3.labels_).tolist(),
        "k2_tamanhos_clusters": tam2.tolist(),
        "amostras_do_3o_cluster_k3": int(arb.sum()),
        "com_k2_essas_amostras_vao_para_cluster": {int(c): int((k2.labels_[arb] == c).sum()) for c in range(2)},
    }

    # 3. conf-bola nas detecções brutas (cache: tudo com conf >= 0,10)
    bola = det[det["classe"] == "bola"]
    n = json.loads((S / "meta.json").read_text())["n_frames"]
    out["conf_bola"] = [{"conf_min": th, "pct_quadros_com_bola": round(100 * bola[bola["conf"] >= th]["frame"].nunique() / n, 1),
                         "candidatos_por_quadro": round(len(bola[bola["conf"] >= th]) / n, 2)}
                        for th in (0.1, 0.2, 0.3, 0.5)]

    # 4. Kalman (sobre o tracks.csv final) → F1 em pixels
    kal = []
    d = tmp / "kal"
    d.mkdir()
    for f_ in ("tracks.csv", "meta.json"):
        shutil.copy(S / f_, d / f_)
    for lac, gate, rc in [(15, 6, None), (8, 6, None), (15, 9, None), (8, 9, None),
                          (8, 9, 0.5), (8, 9, 0.7)]:
        args = ["bola_kalman.py", "--saida", d, "--max-lacuna", lac, "--gate", gate]
        if rc:
            args += ["--reinicio-conf", rc]
        run(*args)
        run("filtrar_bola_gramado.py", "--video", VIDEO, "--saida", d, "--limpar-bola")
        jog, b = carregar_jogadores(d, True), carregar_bola(d, True)
        p = posse_por_quadro(jog, b, 35.0, 3, 30)
        ps = detectar_passes(p, jog, 15.0, 40.0, 4.0)
        kal.append({"max_lacuna": lac, "gate": gate, "reinicio_conf": rc, "passes": len(ps), **{f"f1_{k}": v for k, v in f1_metades(ps).items()}})
    out["kalman"] = kal

    # 5. raio de posse (pixels, configuração final)
    P = R / "saidas" / "jogo1_px"
    for f_ in ("tracks.csv", "bola.csv", "meta.json"):
        shutil.copy(S / f_, d / f_)
    jog, b = carregar_jogadores(d, True), carregar_bola(d, True)
    rr = []
    for r in (20, 35, 50, 65, 80):
        ps = detectar_passes(posse_por_quadro(jog, b, r, 3, 30), jog, 15.0, 40.0, 4.0)
        rr.append({"raio_px": r, "passes": len(ps), "f1": f1(ps, GT)})
    out["raio_posse_f1"] = rr

    (R / "saidas" / "experimentos.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
