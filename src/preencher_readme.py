#!/usr/bin/env python3
"""
Gera README.md a partir de README.template.md, preenchendo os números com os
arquivos de saidas/ (evita divergência README × arquivos gerados).

Uso: python preencher_readme.py   (rodar de dentro de src/)
"""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent.parent
S = R / "saidas" / "jogo1"


def j(nome):
    return json.loads((S / nome).read_text())


def md(df):
    cols = list(df.columns)
    linhas = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        linhas.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(linhas)


def fmt(x, n=1):
    return f"{x:.{n}f}".replace(".", ",") if isinstance(x, (int, float)) and x == x else str(x)


def main():
    t = (R / "README.template.md").read_text()
    meta, mt, ex = j("meta.json"), j("metricas_times.json"), j("metricas_extras.json")
    cal, val = j("calibracao.json"), j("validacao_passes.json")
    tr = pd.read_csv(S / "tracks.csv")
    p = tr[tr["classe"] != "bola"]
    b = tr[tr["classe"] == "bola"]
    bola = pd.read_csv(S / "bola.csv")
    hv = pd.read_csv(S / "homografia_validade.csv")
    anc = pd.read_csv(S / "camera_ancoragens.csv")
    cam = pd.read_csv(S / "camera_erro_px.csv")
    fis = pd.read_csv(S / "metricas_jogadores.csv")
    n = meta["n_frames"]
    novos = (p.groupby("track_id")["frame"].min() > 0).sum() / (n / meta["fps"] / 60)

    v = {
        "PESSOAS_Q": fmt(p.groupby("frame").size().mean()),
        "BOLA_DET": fmt(100 * b["frame"].nunique() / n),
        "BOLA_CAND": len(b),
        "BOLA_ACEITA": int((bola["fonte"] == "detectada").sum()),
        "BOLA_KALMAN": int((bola["fonte"] == "kalman").sum()),
        "BOLA_FINAL": fmt(100 * len(bola) / n),
        "HOMOG_VALIDA": fmt(100 * hv["homografia_valida"].mean()),
        "ANC_ACEITAS": int(anc["aceito"].sum()), "ANC_TOTAL": len(anc),
        "CAM_ERRO_MED": fmt(cam["erro_mediano_px"].median()),
        "N_TRACKS": p["track_id"].nunique(), "IDS_MIN": int(round(novos)),
        "COR_T0": tuple(meta["times_bgr"]["0"]), "COR_T1": tuple(meta["times_bgr"]["1"]),
        "COR_OUT": tuple(meta["times_bgr"].get("-2", [])),
        "VMAX": fmt(fis["vel_max"].max()),
        "M_POR_MIN": fmt((fis["distancia"].sum() / fis["tempo_s"].sum()) * 60 if fis["tempo_s"].sum() else float("nan")),
    }
    for k, nome in [("x=105.0", "RMS_X105"), ("x=88.5", "RMS_X885"), ("y=54.16", "RMS_Y5416"),
                    ("y=68.0", "RMS_Y68"), ("x=99.5", "RMS_X995")]:
        v[nome] = fmt(cal["rms_linhas_m"][k], 2)
    for nome, e in zip(["E_GA_SUP", "E_GA_INF", "E_GA_INF_L", "E_PA_INF"], cal["erro_reprojecao_m"]):
        v[nome] = fmt(e, 2)
    for t_ in (0, 1):
        d = mt[f"time{t_}"]
        v[f"POSSE_T{t_}"] = fmt(d["posse_pct"])
        v[f"PASSES_T{t_}"] = d["passes"]
        v[f"CERTOS_T{t_}"] = d["passes_certos"]
        v[f"PREC_T{t_}"] = fmt(d["precisao_pct"]) if d["precisao_pct"] is not None else "—"
        v[f"DIST_T{t_}"] = fmt(d["distancia_total"], 0)
        e = ex[f"time{t_}"]
        v[f"EX_N_T{t_}"] = fmt(e["n_visiveis_medio"])
        v[f"EX_LARG_T{t_}"] = fmt(e["largura_media_m"])
        v[f"EX_PROF_T{t_}"] = fmt(e["profundidade_media_m"])
        v[f"EX_AREA_T{t_}"] = fmt(e["area_hull_media_m2"], 0)
        v[f"EX_LINHA_T{t_}"] = fmt(e["linha_def_media_x"])
        v[f"EX_VOR_T{t_}"] = fmt(e["controle_voronoi_medio_pct"])
        v[f"EX_PRES_T{t_}"] = fmt(e["pressao_sofrida_media_n5m"], 2)
    v["PASSES_DET"] = mt["time0"]["passes"] + mt["time1"]["passes"]
    v1 = val["tol_1s"]
    v["VAL_P"], v["VAL_R"], v["VAL_F1"] = fmt(v1["precisao"], 2), fmt(v1["revocacao"], 2), fmt(v1["f1"], 2)
    PX = R / "saidas" / "jogo1_px"
    valpx = json.loads((PX / "validacao_passes.json").read_text())
    mpx = json.loads((PX / "metricas_times.json").read_text())
    linhas_v = []
    for nome, vv in (("pixels (clipe inteiro)", valpx), ("metros (quadros verificados)", val)):
        for k, r in vv.items():
            linhas_v.append(f"| {nome} | {k.replace('tol_', '').replace('s', ' s')} | {r['detectados']} | {r['anotados']} | "
                            f"{r['casados']} | {fmt(r['precisao'], 2)} | {fmt(r['revocacao'], 2)} | {fmt(r['f1'], 2)} | "
                            f"{fmt(r['acordo_sucesso'], 2)} |")
    v["TABELA_VALIDACAO"] = "\n".join(linhas_v)
    p1 = valpx["tol_1s"]
    v["PX_CASADOS"] = p1["casados"]
    v["PX_VAL_F1_2"] = fmt(valpx["tol_2s"]["f1"], 2)
    v["PX_ACORDO"] = f"{fmt(100 * p1['acordo_sucesso'], 0)}%"
    v["PX_VAL_P"], v["PX_VAL_R"], v["PX_VAL_F1"] = fmt(p1["precisao"], 2), fmt(p1["revocacao"], 2), fmt(p1["f1"], 2)
    for t_ in (0, 1):
        d = mpx[f"time{t_}"]
        v[f"PX_POSSE_T{t_}"] = fmt(d["posse_pct"])
        v[f"PX_PASSES_T{t_}"] = d["passes"]
        v[f"PX_CERTOS_T{t_}"] = d["passes_certos"]
        v[f"PX_PREC_T{t_}"] = fmt(d["precisao_pct"]) if d["precisao_pct"] is not None else "—"
    v["PX_INDEF"] = fmt(mpx.get("posse_indefinida_pct_dos_quadros", float("nan")))
    v["PX_PASSES_DET"] = mpx["time0"]["passes"] + mpx["time1"]["passes"]
    v["M_QUADROS_BOLA"] = mt["quadros_com_bola"]
    v["TABELA_RAIO_PX"] = md(pd.read_csv(PX / "sensibilidade_raio.csv"))

    cm = pd.read_csv(R / "saidas" / "comparacao_modelos" / "comparacao_modelos.csv")
    cm.columns = ["modelo", "imgsz", "quadros", "fps (CPU)", "jogadores/quadro", "% quadros c/ bola",
                  "detecções de bola/quadro", "conf. média bola"]
    v["TABELA_MODELOS"] = md(cm)
    gk = pd.read_csv(S / "goleiros.csv")
    v["N_GK"] = len(gk)
    v["N_GK_ROMA"] = int((gk["time_antes"] == 1).sum())
    cache = pd.concat([pd.read_csv(f) for f in sorted((S / "cache").glob("det_*.csv"))])
    bruto = cache[cache["classe"] == "bola"]
    v["BOLA_REMOV"] = f"{len(bruto) - len(b)} de {len(bruto)} ({fmt(100 * (len(bruto) - len(b)) / len(bruto), 0)}%)"
    v["BOLA_DET_FILT"] = fmt(100 * b["frame"].nunique() / n)
    v["BOLA_DET"] = fmt(100 * bruto["frame"].nunique() / n)
    v["BOLA_CAND"] = len(bruto)
    v["N_PASSES_M"] = len(pd.read_csv(S / "passes_features.csv"))
    v["EX_QUADROS"] = ex["quadros_amostrados"]
    # % candidatos de bola a >40 px da bola aceita
    bi = bola.set_index("frame")
    c = b[b["frame"].isin(bi.index)]
    import numpy as np
    dd = np.hypot(c["px"].to_numpy() - bi.loc[c["frame"], "px"].to_numpy(),
                  c["py"].to_numpy() - bi.loc[c["frame"], "py"].to_numpy())
    v["BOLA_FP_PCT"] = fmt(100 * (dd > 40).mean(), 0)

    for k, val_ in v.items():
        t = t.replace(f"⟪{k}⟫", str(val_))
    (R / "README.md").write_text(t)
    faltando = sorted(set(__import__("re").findall(r"⟪([A-Z0-9_]+)⟫", t)))
    print("README.md gerado. Placeholders ainda abertos:", faltando)


if __name__ == "__main__":
    main()
