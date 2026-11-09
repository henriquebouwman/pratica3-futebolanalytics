#!/usr/bin/env python3
"""
Invalida (NaN) as posições em metros dos quadros em que a homografia não é confiável:
quadros a mais de --max-gap quadros da última re-ancoragem aceita (camera_ancoragens.csv).
Assim a deriva do ECC entre ancoragens não contamina distância, posse, passes e eXP.

Uso: python filtrar_projecao.py --saida ../saidas/jogo1 --max-gap 30
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    ap.add_argument("--max-gap", type=int, default=30, help="Quadros desde a última ancoragem (30 = 2 s a 15 fps)")
    ap.add_argument("--excluir", default="339-375",
                    help="Intervalos de quadros sem câmera principal (ex.: close-up), 'a-b,c-d'")
    ap.add_argument("--erro-max", type=float, default=3.0,
                    help="Erro mediano máx. (px) entre linhas reprojetadas e detectadas (camera_erro_px.csv)")
    a = ap.parse_args()
    pasta = Path(a.saida)
    n = json.loads((pasta / "meta.json").read_text())["n_frames"]
    anc = pd.read_csv(pasta / "camera_ancoragens.csv")
    ok_f = np.sort(anc.loc[anc["aceito"] == 1, "frame"].to_numpy())
    frames = np.arange(n)
    idx = np.searchsorted(ok_f, frames, side="right") - 1
    gap = np.where(idx >= 0, frames - ok_f[np.clip(idx, 0, None)], 10**6)
    valido = gap <= a.max_gap
    # Critério principal: verificação quadro a quadro das linhas reprojetadas (checar_camera.py --passo-check 1).
    # A aceitação da re-ancoragem sozinha não basta: há "travamentos" em linhas erradas com resíduo baixo.
    arq_e = pasta / "camera_erro_px.csv"
    if arq_e.exists():
        e = pd.read_csv(arq_e).set_index("frame")["erro_mediano_px"]
        if len(e) >= 0.9 * n:
            valido = np.array([bool(e.get(f, np.inf) <= a.erro_max) for f in frames])
            gap = np.where(valido, 0, gap)
    for iv in filter(None, a.excluir.split(",")):
        i0, i1 = map(int, iv.split("-"))
        valido[i0:i1 + 1] = False
    pd.DataFrame({"frame": frames, "quadros_desde_ancoragem": gap, "homografia_valida": valido.astype(int)}) \
        .to_csv(pasta / "homografia_validade.csv", index=False)
    for arq in ("tracks_campo.csv", "bola_campo.csv"):
        df = pd.read_csv(pasta / arq)
        ruim = ~df["frame"].map(dict(zip(frames, valido))).fillna(False).astype(bool)
        df.loc[ruim, ["x_m", "y_m"]] = np.nan
        df.to_csv(pasta / arq, index=False)
        print(f"{arq}: {100*df['x_m'].notna().mean():.1f}% posições válidas após o filtro")
    print(f"Quadros com homografia válida: {100*valido.mean():.1f}%")


if __name__ == "__main__":
    main()
