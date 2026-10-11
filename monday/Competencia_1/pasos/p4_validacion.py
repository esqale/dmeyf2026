"""Paso 4: validacion del modelo en la ventana de desarrollo.

Dataset fe_final, hiperparametros del paso 3, 10 semillas, train 202103+202104 -> 202106.
Por semilla: curva de ganancia por cantidad de envios N. Resumen: N de la curva media
suavizada, desvio, minimo y meseta (cortes a menos de 2M del maximo). Chequea que
config.N_ENVIOS caiga dentro de la meseta.

No es necesario para generar la entrega: documenta cuanto rinde el modelo y de donde sale N.

Uso:  python pasos/p4_validacion.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from lib.datos import cargar, features, log, mejores_params, semillas  # noqa: E402
from lib.ganancia import curva, resumen  # noqa: E402
from lib.modelo import entrenar_predecir  # noqa: E402


def main():
    params = mejores_params()
    df = cargar([*config.MESES_VAL_TRAIN, config.MES_VAL_TEST], config.DATASET_FINAL)
    feats = features(df)
    tr = df[df["foto_mes"].isin(config.MESES_VAL_TRAIN)].reset_index(drop=True)
    te = df[df["foto_mes"] == config.MES_VAL_TEST].reset_index(drop=True)
    del df
    sems = semillas(config.N_SEMILLAS_VALIDACION)
    log(f"validacion: train {config.MESES_VAL_TRAIN} -> {config.MES_VAL_TEST}, {len(feats)} features, {len(sems)} semillas")

    filas = []
    for s in sems:
        c = curva(entrenar_predecir(tr, te, feats, params, s), te["clase_ternaria"])
        filas.append({"semilla": s, **{str(n): v for n, v in zip(config.CORTES, c)}})
        log(f"  semilla {s}: mejor N={config.CORTES[int(c.argmax())]} -> {c.max() / 1e6:6.1f}M")

    tb = pd.DataFrame(filas)
    config.DIR_VALIDACION.mkdir(parents=True, exist_ok=True)
    tb.to_csv(config.DIR_VALIDACION / "por_semilla.csv", index=False)
    r = resumen(tb)
    (config.DIR_VALIDACION / "resumen.json").write_text(json.dumps(
        {k: v for k, v in r.items() if k != "curva_suave"} | {"curva_suave": {str(k): v for k, v in r["curva_suave"].items()}},
        indent=2, default=float))
    log(f"202106: N={r['N']}  media {r['media'] / 1e6:.1f}M  desvio {r['desvio'] / 1e6:.1f}M  "
        f"min {r['minimo'] / 1e6:.1f}M  meseta {r['meseta'][0]}-{r['meseta'][1]}")
    log("  curva media suavizada (M): " + "  ".join(f"{n}:{v / 1e6:.0f}" for n, v in r["curva_suave"].items()))
    dentro = r["meseta"][0] <= config.N_ENVIOS <= r["meseta"][1]
    log(f"N de la entrega ({config.N_ENVIOS}) dentro de la meseta: {'SI' if dentro else 'NO'}")


if __name__ == "__main__":
    main()
