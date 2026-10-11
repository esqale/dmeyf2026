"""Paso 3: busqueda de hiperparametros de LightGBM con Optuna.

- Dataset: fe_base. Ventana: train 202103+202104 -> medir 202106 (la misma distancia de 2 meses
  entre la ultima baja aprendida y la baja a predecir que en la entrega, 202106 -> 202108).
- Cada trial entrena 3 semillas, promedia las probabilidades y mide la curva de ganancia
  suavizada sobre ese promedio: el valor del trial depende menos de la suerte de una semilla.
- 30 trials, TPE con semilla fija. Resultado: salida/busqueda/mejores_params.json, que se
  compara con params/mejores_params.json (los de la entrega).

Uso:  python pasos/p3_busqueda.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from lib.datos import cargar, features, log, semillas  # noqa: E402
from lib.ganancia import curva, suave  # noqa: E402
from lib.modelo import entrenar_predecir  # noqa: E402


def sugerir(trial):
    p = {}
    for nombre, (tipo, desde, hasta, es_log) in config.ESPACIO_OPTUNA.items():
        sug = trial.suggest_int if tipo == "int" else trial.suggest_float
        p[nombre] = sug(nombre, desde, hasta, log=es_log)
    return p


def main():
    df = cargar([*config.MESES_VAL_TRAIN, config.MES_VAL_TEST], config.DATASET_BASE)
    feats = features(df)
    tr = df[df["foto_mes"].isin(config.MESES_VAL_TRAIN)].reset_index(drop=True)
    te = df[df["foto_mes"] == config.MES_VAL_TEST].reset_index(drop=True)
    del df
    sems = semillas(config.N_SEMILLAS_TRIAL)
    log(f"busqueda: train {config.MESES_VAL_TRAIN} {tr.shape} -> {config.MES_VAL_TEST}, {len(feats)} features, "
        f"{config.N_TRIALS} trials x {len(sems)} semillas")

    def objective(trial):
        p = sugerir(trial)
        prob = np.mean([entrenar_predecir(tr, te, feats, p, s) for s in sems], axis=0)
        c = suave(curva(prob, te["clase_ternaria"]))
        i = int(np.argmax(c))
        trial.set_user_attr("N", config.CORTES[i])
        return float(c[i])

    config.DIR_BUSQUEDA.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="busqueda", load_if_exists=True,
                                storage=f"sqlite:///{(config.DIR_BUSQUEDA / 'optuna.db').as_posix()}",
                                sampler=optuna.samplers.TPESampler(seed=semillas(1)[0]))

    def cb(st, t):
        log(f"  trial {t.number}: {t.value / 1e6:6.1f}M N={t.user_attrs['N']} | mejor {st.best_value / 1e6:.1f}M "
            f"(trial {st.best_trial.number})")

    faltan = config.N_TRIALS - len([t for t in study.trials if t.state.name == "COMPLETE"])
    if faltan > 0:  # si se corto, retoma
        study.optimize(objective, n_trials=faltan, callbacks=[cb])

    study.trials_dataframe().sort_values("value", ascending=False).to_csv(
        config.DIR_BUSQUEDA / "trials.csv", index=False)
    mejores = dict(study.best_params)
    (config.DIR_BUSQUEDA / "mejores_params.json").write_text(json.dumps(mejores, indent=2))
    log(f"mejor trial {study.best_trial.number} ({study.best_value / 1e6:.1f}M): {mejores}")

    entregados = json.loads(config.PARAMS_ENTREGA.read_text())
    log("coinciden con params/mejores_params.json (los de la entrega): "
        + ("SI" if mejores == entregados else f"NO -> entregados: {entregados}"))


if __name__ == "__main__":
    main()
