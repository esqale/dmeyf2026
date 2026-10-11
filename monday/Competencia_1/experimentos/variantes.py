"""Experimentos que se midieron y NO entraron al modelo (ver experimentos/RESULTADOS.md).

Cada variante se compara contra el modelo entregado con el mismo protocolo: ventana
202103+202104 -> 202106, mismos hiperparametros (params/mejores_params.json), 10 semillas
pareadas. Regla de adopcion: mejora media >= 5M y gana en >= 7 de 10 semillas.

  referencia     fe_final (el dataset del modelo entregado)
  aguinaldo      fe_final + cobra_aguinaldo (junio/diciembre con mpayroll > 1,3 x promedio de
                 los 3 meses previos) y mpayroll_sin_sac (= mpayroll / 1,5 en ese caso) con su
                 lag_1, delta_1 y ma_3
  rank_cero      fe_final + ranking dentro del mes con cero fijo de todas las variables de monto:
                 0 -> 0, positivos en (0,1], negativos en [-1,0)
  undersampling  fe_final, cada semilla entrena con todas las bajas y su propio 25% de CONTINUA
                 (min_data_in_leaf escalado por las filas que quedan)
  busqueda_final hiperparametros re-buscados con Optuna sobre fe_final (mismo protocolo que p3)
  solo_baja2     target: positivo = solo BAJA+2 (el modelo entregado usa BAJA+1 + BAJA+2)
  peso_baja1     target: BAJA+1 + BAJA+2, con las filas BAJA+1 pesando 0,25
  sin_202103     se entrena sin 202103 (el primer mes no tiene lags ni deltas): ventana 202104 -> 202106,
                 min_data_in_leaf escalado por filas
  ma3_completo   ma_3 y slope_3 en NULL cuando el cliente tiene menos de 3 meses de historia

Para cada variante tambien se guarda cuantos BAJA+1 y BAJA+2 caen en el top config.N_ENVIOS.

Con --final NOMBRE se entrena el modelo final de una variante de dataset (202103-202106, 20 semillas,
min_data_in_leaf escalado como en p5) y se exportan un CSV por semilla y el promedio en
config.N_ENVIOS, para medirla en el leaderboard publico:
  python experimentos/variantes.py --final ma3_completo

Requiere haber corrido p1 y p2. Uso:
  python experimentos/variantes.py --variantes referencia aguinaldo rank_cero undersampling
  python experimentos/variantes.py --variantes busqueda_final
"""
import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from lib.datos import cargar, features, log, semillas  # noqa: E402
from lib.ganancia import comparar, curva, resumen, suave  # noqa: E402
from lib.modelo import entrenar_predecir, escalar_minimos  # noqa: E402
from pasos.p3_busqueda import sugerir  # noqa: E402

DIR = config.SALIDA / "experimentos"

SQL_AGUINALDO = """
CREATE OR REPLACE TABLE t AS
SELECT f.*,
    CASE WHEN foto_mes % 100 IN (6, 12)
              AND avg(mpayroll) OVER previos > 0
              AND mpayroll > 1.3 * avg(mpayroll) OVER previos THEN 1 ELSE 0 END AS cobra_aguinaldo
FROM t f
WINDOW previos AS (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING);
CREATE OR REPLACE TABLE t AS
SELECT f.*, CASE WHEN cobra_aguinaldo = 1 THEN mpayroll / 1.5 ELSE mpayroll END AS mpayroll_sin_sac FROM t f;
CREATE OR REPLACE TABLE t AS
SELECT f.*
    , lag(mpayroll_sin_sac, 1) OVER ventana_cliente AS lag_1_mpayroll_sin_sac
    , mpayroll_sin_sac - lag(mpayroll_sin_sac, 1) OVER ventana_cliente AS delta_1_mpayroll_sin_sac
    , avg(mpayroll_sin_sac) OVER ventana_3m AS ma_3_mpayroll_sin_sac
FROM t f
WINDOW
    ventana_cliente AS (PARTITION BY numero_de_cliente ORDER BY foto_mes),
    ventana_3m AS (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
"""


def sql_rank_cero(montos):
    cols = [f"CASE WHEN {c} IS NULL THEN NULL WHEN {c} = 0 THEN 0 "
            f"WHEN {c} > 0 THEN cume_dist() OVER (PARTITION BY foto_mes, sign({c}) ORDER BY {c}) "
            f"ELSE -cume_dist() OVER (PARTITION BY foto_mes, sign({c}) ORDER BY {c} DESC) END AS rkc_{c}"
            for c in montos]
    return "CREATE OR REPLACE TABLE t AS SELECT f.*, " + ", ".join(cols) + " FROM t f"


def dataset_variante(nombre):
    """Parquet de la variante (se genera una vez)."""
    destino = DIR / f"dataset_{nombre}.parquet"
    if destino.exists():
        return destino
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t AS SELECT * FROM read_parquet('{config.DATASET_FINAL.as_posix()}')")
    if nombre == "aguinaldo":
        con.execute(SQL_AGUINALDO)
    elif nombre == "rank_cero":
        crudas = [c for c, *_ in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{config.DATASET_CLASE.as_posix()}')").fetchall()]
        montos = [c for c in crudas if c.startswith("m") or c.startswith(("Visa_m", "Master_m"))]
        con.execute(sql_rank_cero(montos))
    elif nombre == "ma3_completo":
        cols = [c for c, *_ in con.execute("DESCRIBE t").fetchall() if c.startswith(("ma_3_", "slope_3_"))]
        reemplazos = ", ".join(f"CASE WHEN n_hist < 3 THEN NULL ELSE {c} END AS {c}" for c in cols)
        con.execute(f"""CREATE OR REPLACE TABLE t AS
            SELECT * EXCLUDE (n_hist) REPLACE ({reemplazos})
            FROM (SELECT *, count(*) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes
                                          ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS n_hist FROM t)""")
    log(f"generando dataset {nombre}")
    con.execute(f"COPY t TO '{destino.as_posix()}' (FORMAT PARQUET)")
    return destino


def ventana(dataset, meses_train=config.MESES_VAL_TRAIN):
    df = cargar([*meses_train, config.MES_VAL_TEST], dataset)
    feats = features(df)
    return (df[df["foto_mes"].isin(meses_train)].reset_index(drop=True),
            df[df["foto_mes"] == config.MES_VAL_TEST].reset_index(drop=True), feats)


def validar(nombre, dataset, params, meses_train=config.MESES_VAL_TRAIN):
    arch = DIR / f"validacion_{nombre}.csv"
    if arch.exists():
        return pd.read_csv(arch)
    tr, te, feats = ventana(dataset, meses_train)
    if meses_train != config.MESES_VAL_TRAIN:  # misma regularizacion relativa que con la ventana original
        filas_ref = duckdb.connect().execute(
            f"SELECT count(*) FROM read_parquet('{Path(dataset).as_posix()}') "
            f"WHERE foto_mes IN {tuple(config.MESES_VAL_TRAIN)}").fetchone()[0]
        params = escalar_minimos(params, filas_ref, len(tr))
    log(f"{nombre}: train {meses_train} {tr.shape}, {params}")
    clases = te["clase_ternaria"].to_numpy()
    filas = []
    for s in semillas(config.N_SEMILLAS_VALIDACION):
        prob = entrenar_predecir(tr, te, feats, params, s)
        c = curva(prob, clases)
        top = clases[np.argsort(-prob, kind="stable")[:config.N_ENVIOS]]
        filas.append({"semilla": s, **{str(n): v for n, v in zip(config.CORTES, c)},
                      "top_baja1": int((top == "BAJA+1").sum()), "top_baja2": int((top == "BAJA+2").sum())})
        log(f"  {nombre} semilla {s}: {c.max() / 1e6:6.1f}M | top {config.N_ENVIOS}: "
            f"{filas[-1]['top_baja2']} BAJA+2, {filas[-1]['top_baja1']} BAJA+1")
    tb = pd.DataFrame(filas)
    tb.to_csv(arch, index=False)
    return tb


def busqueda_final():
    """Optuna con el mismo protocolo que p3, pero sobre fe_final."""
    tr, te, feats = ventana(config.DATASET_FINAL)
    sems = semillas(config.N_SEMILLAS_TRIAL)

    def objective(trial):
        p = sugerir(trial)
        prob = np.mean([entrenar_predecir(tr, te, feats, p, s) for s in sems], axis=0)
        return float(suave(curva(prob, te["clase_ternaria"])).max())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="busqueda_final", load_if_exists=True,
                                storage=f"sqlite:///{(DIR / 'optuna_busqueda_final.db').as_posix()}",
                                sampler=optuna.samplers.TPESampler(seed=semillas(1)[0]))
    faltan = config.N_TRIALS - len([t for t in study.trials if t.state.name == "COMPLETE"])
    if faltan > 0:
        study.optimize(objective, n_trials=faltan,
                       callbacks=[lambda st, t: log(f"  trial {t.number}: {t.value / 1e6:6.1f}M")])
    return dict(study.best_params)


def final_variante(nombre):
    """Final de una variante de dataset, igual que p5 + p6 pero en salida/experimentos/final_<nombre>/."""
    import time

    from lib.datos import ids_validos
    from pasos.p6_entrega import exportar

    dataset = dataset_variante(nombre)
    destino = DIR / f"final_{nombre}"
    destino.mkdir(parents=True, exist_ok=True)
    df = cargar([*config.MESES_FINAL, config.MES_FUTURO], dataset)
    feats = features(df)
    tr = df[df["foto_mes"].isin(config.MESES_FINAL)].reset_index(drop=True)
    fu = df[df["foto_mes"] == config.MES_FUTURO].reset_index(drop=True)
    filas_ventana = int(df["foto_mes"].isin(config.MESES_VAL_TRAIN).sum())
    del df
    params = escalar_minimos(json.loads(config.PARAMS_ENTREGA.read_text()), filas_ventana, len(tr))
    ids = fu["numero_de_cliente"].to_numpy()
    validos = ids_validos(config.MES_FUTURO, dataset)
    assert np.array_equal(ids, validos), "ids de 202108 desalineados"
    log(f"final {nombre}: {params}, train {config.MESES_FINAL} {tr.shape}, {config.N_SEMILLAS_FINAL} semillas")

    arch = destino / "probabilidades_202108.parquet"
    probs = pd.read_parquet(arch) if arch.exists() else pd.DataFrame({"numero_de_cliente": ids})
    for s in semillas(config.N_SEMILLAS_FINAL):
        if f"s{s}" in probs.columns:
            continue
        t0 = time.time()
        probs[f"s{s}"] = entrenar_predecir(tr, fu, feats, params, s)
        probs.to_parquet(arch, index=False)
        exportar(ids, probs[f"s{s}"], config.N_ENVIOS, destino / f"{nombre}_s{s}_n{config.N_ENVIOS}.csv", validos)
        log(f"  semilla {s} ({time.time() - t0:.0f}s)")
    cols = [c for c in probs.columns if c.startswith("s") and c[1:].isdigit()]
    exportar(ids, probs[cols].mean(axis=1), config.N_ENVIOS,
             destino / f"{nombre}_promedio{len(cols)}_n{config.N_ENVIOS}.csv", validos)
    log(f"RESULTADO final {nombre} listo -> {destino}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", choices=["aguinaldo", "rank_cero", "ma3_completo"], default=None)
    ap.add_argument("--variantes", nargs="+", default=["referencia", "aguinaldo", "rank_cero", "undersampling"],
                    choices=["referencia", "aguinaldo", "rank_cero", "undersampling", "busqueda_final",
                             "solo_baja2", "peso_baja1", "sin_202103", "ma3_completo"])
    args = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    if args.final:
        final_variante(args.final)
        return
    params = json.loads(config.PARAMS_ENTREGA.read_text())

    ref = validar("referencia", config.DATASET_FINAL, params)
    for v in args.variantes:
        if v == "referencia":
            continue
        if v == "undersampling":
            tb = validar(v, config.DATASET_FINAL, {**params, "undersampling": 0.25})
        elif v == "busqueda_final":
            tb = validar(v, config.DATASET_FINAL, busqueda_final())
        elif v == "solo_baja2":
            tb = validar(v, config.DATASET_FINAL, {**params, "positivos": ("BAJA+2",)})
        elif v == "peso_baja1":
            tb = validar(v, config.DATASET_FINAL, {**params, "peso_baja1": 0.25})
        elif v == "sin_202103":
            tb = validar(v, config.DATASET_FINAL, params, meses_train=(202104,))
        else:
            tb = validar(v, dataset_variante(v), params)
        r, c = resumen(tb), comparar(ref, tb)
        comp = (f" | top {config.N_ENVIOS}: {tb['top_baja2'].mean():.0f} BAJA+2, {tb['top_baja1'].mean():.0f} BAJA+1"
                if "top_baja1" in tb else "")
        log(f"RESULTADO {v}: media {r['media'] / 1e6:.1f}M (N={r['N']}, desvio {r['desvio'] / 1e6:.1f}M) | "
            f"vs referencia {c['diferencia'] / 1e6:+.1f}M, gana {c['gana']}/{c['semillas']} -> "
            f"{'SE ADOPTA' if c['adopta'] else 'no se adopta'}{comp}")
    r = resumen(ref)
    log(f"RESULTADO referencia: media {r['media'] / 1e6:.1f}M (N={r['N']}) | top {config.N_ENVIOS}: "
        f"{ref['top_baja2'].mean():.0f} BAJA+2, {ref['top_baja1'].mean():.0f} BAJA+1")


if __name__ == "__main__":
    main()
