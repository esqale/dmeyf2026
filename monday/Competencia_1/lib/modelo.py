"""LightGBM: parametros, entrenamiento por semilla y escalado de minimos."""
import lightgbm as lgb
import numpy as np

import config


def escalar_minimos(params, filas_referencia, filas_nuevas):
    """Los minimos de filas por hoja son absolutos: se escalan con la cantidad de filas para mantener
    la misma regularizacion relativa (misma idea que param_normalizado de z494)."""
    factor = filas_nuevas / filas_referencia
    return {k: (max(1, int(round(v * factor))) if k in config.PARAMS_MINIMOS else v) for k, v in params.items()}


def params_lgb(p, seed):
    return {**config.LGBM_FIJOS,
            "num_leaves": p["num_leaves"],
            "learning_rate": p["learning_rate"],
            "min_data_in_leaf": p["min_data_in_leaf"],
            "feature_fraction": p["feature_fraction"],
            "bagging_fraction": p["bagging_fraction"],
            "lambda_l2": p["lambda_l2"],
            "seed": seed}


def entrenar_predecir(df_train, df_apply, feats, p, seed):
    """Entrena con positivo = BAJA+1 o BAJA+2 y devuelve la probabilidad para df_apply.
    Opciones de p que solo usan los experimentos:
      'undersampling'  fraccion de CONTINUA por semilla
      'positivos'      clases positivas (por defecto config.CLASES_POSITIVAS)
      'peso_baja1'     peso de las filas BAJA+1 (por defecto 1)"""
    clases = df_train["clase_ternaria"]
    y = clases.isin(p.get("positivos", config.CLASES_POSITIVAS)).to_numpy().astype(int)
    pesos = None
    if p.get("peso_baja1", 1.0) != 1.0:
        pesos = np.where(clases.to_numpy() == "BAJA+1", p["peso_baja1"], 1.0)
    filas = np.ones(len(df_train), dtype=bool)
    us = p.get("undersampling", 1.0)
    if us < 1.0:
        filas = (clases != "CONTINUA").to_numpy() | (np.random.default_rng(seed).random(len(df_train)) < us)
        p = escalar_minimos(p, len(df_train), int(filas.sum()))
    # un Dataset nuevo por semilla: LightGBM arma los bins muestreando filas con la semilla
    dtrain = lgb.Dataset(df_train.loc[filas, feats], label=y[filas],
                         weight=None if pesos is None else pesos[filas],
                         params={"seed": seed, "max_bin": config.LGBM_FIJOS["max_bin"]})
    m = lgb.train(params_lgb(p, seed), dtrain, num_boost_round=p["num_boost_round"])
    return m.predict(df_apply[feats], num_threads=config.HILOS)
