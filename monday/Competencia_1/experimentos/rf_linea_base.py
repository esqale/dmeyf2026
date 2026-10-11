"""Random Forest como linea de base (no entra en la entrega).

Sigue modelo.ipynb / z401: imputacion por mediana, clase ternaria y Optuna sobre el OOB del mes
de entrenamiento (proxy rapido). Diferencias con modelo.ipynb: la ganancia se mide cortando por
cantidad de envios N (regla de la entrega) y se excluyen las columnas con drift.

  buscar   Optuna (25 trials) sobre el OOB, entrenando en 202104 con fe_base
  validar  mejores hiperparametros, train 202104 -> 202106 (2 meses de distancia), 5 semillas

Es lento (~4 min por modelo de 300 arboles). Uso:
  python experimentos/rf_linea_base.py [--etapas buscar validar] [--trials 25] [--semillas 5]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from lib.datos import cargar, features, log, semillas  # noqa: E402
from lib.ganancia import curva, resumen, suave  # noqa: E402

MES_TRAIN, MES_VALID = 202104, 202106
DIR = config.SALIDA / "experimentos" / "rf"


def rf(params, seed, n_estimators, oob=False):
    return RandomForestClassifier(n_estimators=n_estimators, max_samples=0.7, n_jobs=-1,
                                  random_state=seed, oob_score=oob, **params)


def buscar(trials):
    df = cargar([MES_TRAIN], config.DATASET_BASE)
    feats = features(df)
    X = SimpleImputer(strategy="median").fit_transform(df[feats])
    y = df["clase_ternaria"].to_numpy()
    sem = semillas(1)[0]

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 2000, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 400, log=True),
            "max_features": trial.suggest_float("max_features", 0.05, 0.7),
        }
        m = rf(params, sem, n_estimators=100, oob=True)
        m.fit(X, y)
        oob = np.nan_to_num(m.oob_decision_function_[:, list(m.classes_).index("BAJA+2")], nan=0.0)
        return float(suave(curva(oob, y)).max())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="rf_oob", load_if_exists=True,
                                storage=f"sqlite:///{(DIR / 'optuna.db').as_posix()}",
                                sampler=optuna.samplers.TPESampler(seed=sem))
    faltan = trials - len([t for t in study.trials if t.state.name == "COMPLETE"])
    if faltan > 0:
        study.optimize(objective, n_trials=faltan,
                       callbacks=[lambda st, t: log(f"  trial {t.number}: {t.value / 1e6:6.1f}M (OOB)")])
    (DIR / "mejores_params.json").write_text(json.dumps(study.best_params, indent=2))
    log(f"mejores: {study.best_params}")


def validar(n_semillas):
    params = json.loads((DIR / "mejores_params.json").read_text())
    df = cargar([MES_TRAIN, MES_VALID], config.DATASET_BASE)
    feats = features(df)
    tr, te = df[df["foto_mes"] == MES_TRAIN], df[df["foto_mes"] == MES_VALID]
    imp = SimpleImputer(strategy="median")
    Xtr, Xte = imp.fit_transform(tr[feats]), imp.transform(te[feats])
    filas = []
    for s in semillas(n_semillas):
        m = rf(params, s, n_estimators=300)
        m.fit(Xtr, tr["clase_ternaria"].to_numpy())
        prob = m.predict_proba(Xte)[:, list(m.classes_).index("BAJA+2")]
        c = curva(prob, te["clase_ternaria"].to_numpy())
        filas.append({"semilla": s, **{str(n): v for n, v in zip(config.CORTES, c)}})
        log(f"  semilla {s}: {c.max() / 1e6:6.1f}M")
    tb = pd.DataFrame(filas)
    tb.to_csv(DIR / "validacion.csv", index=False)
    r = resumen(tb)
    log(f"RF 202106: N={r['N']}  media {r['media'] / 1e6:.1f}M  desvio {r['desvio'] / 1e6:.1f}M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapas", nargs="+", default=["buscar", "validar"], choices=["buscar", "validar"])
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--semillas", type=int, default=5)
    args = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    if "buscar" in args.etapas:
        buscar(args.trials)
    if "validar" in args.etapas:
        validar(args.semillas)


if __name__ == "__main__":
    main()
