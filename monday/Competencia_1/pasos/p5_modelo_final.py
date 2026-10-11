"""Paso 5: modelo final y probabilidades de 202108.

- Dataset fe_final, entrenamiento con todos los meses con clase (202103-202106).
- min_data_in_leaf se escala por filas_final / filas_ventana (~x2): los hiperparametros se
  eligieron con 2 meses y el final usa 4.
- 20 semillas (semillerio): una probabilidad por semilla para cada cliente de 202108.
  Se guarda en salida/modelo_final/probabilidades_202108.parquet; si se corta, retoma.

Uso:  python pasos/p5_modelo_final.py [--semillas 20]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from lib.datos import cargar, features, ids_validos, log, mejores_params, semillas  # noqa: E402
from lib.modelo import entrenar_predecir, escalar_minimos  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=config.N_SEMILLAS_FINAL)
    args = ap.parse_args()

    df = cargar([*config.MESES_FINAL, config.MES_FUTURO], config.DATASET_FINAL)
    feats = features(df)
    tr = df[df["foto_mes"].isin(config.MESES_FINAL)].reset_index(drop=True)
    fu = df[df["foto_mes"] == config.MES_FUTURO].reset_index(drop=True)
    filas_ventana = int(df["foto_mes"].isin(config.MESES_VAL_TRAIN).sum())
    del df
    params = escalar_minimos(mejores_params(), filas_ventana, len(tr))
    ids = fu["numero_de_cliente"].to_numpy()
    assert np.array_equal(ids, ids_validos(config.MES_FUTURO, config.DATASET_FINAL)), "ids de 202108 desalineados"
    sems = semillas(args.semillas)

    config.DIR_FINAL.mkdir(parents=True, exist_ok=True)
    (config.DIR_FINAL / "config.json").write_text(json.dumps(
        {"params": params, "meses_train": config.MESES_FINAL, "filas_train": len(tr),
         "filas_ventana": filas_ventana, "semillas": sems, "features": feats}, indent=2))
    log(f"modelo final: {params}, train {config.MESES_FINAL} {tr.shape} -> {config.MES_FUTURO}, {len(sems)} semillas")

    arch = config.DIR_FINAL / "probabilidades_202108.parquet"
    probs = pd.read_parquet(arch) if arch.exists() else pd.DataFrame({"numero_de_cliente": ids})
    assert np.array_equal(probs["numero_de_cliente"].to_numpy(), ids), "probabilidades guardadas desalineadas"
    for s in sems:
        if f"s{s}" in probs.columns:
            continue
        t0 = time.time()
        probs[f"s{s}"] = entrenar_predecir(tr, fu, feats, params, s)
        probs.to_parquet(arch, index=False)
        log(f"  semilla {s} ({time.time() - t0:.0f}s)")
    log(f"listo -> {arch.relative_to(config.RAIZ)}")


if __name__ == "__main__":
    main()
