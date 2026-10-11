"""Carga de datos, lista de features, semillas y log."""
import json
import time
from pathlib import Path

import duckdb
import numpy as np

import config

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def semillas(n):
    """Pool fijo: la semilla i es la misma en todos los pasos y experimentos."""
    rng = np.random.RandomState(config.SEMILLA_POOL)
    return rng.choice(1_000_000, size=50, replace=False).tolist()[:n]


def cargar(meses, dataset):
    """Filas de los meses pedidos, ordenadas por (foto_mes, numero_de_cliente): los modelos dependen
    del orden de filas, asi el resultado es reproducible. El id se lee como entero y el resto de
    las columnas DOUBLE se bajan a float32 (float32 redondearia ids de mas de 16,7 millones)."""
    lista = ", ".join(str(m) for m in meses)
    df = duckdb.connect().execute(
        f"SELECT * REPLACE (CAST(numero_de_cliente AS BIGINT) AS numero_de_cliente) "
        f"FROM read_parquet('{Path(dataset).as_posix()}') WHERE foto_mes IN ({lista}) "
        f"ORDER BY foto_mes, numero_de_cliente"
    ).df()
    num = df.select_dtypes("float64").columns
    df[num] = df[num].astype("float32")
    return df


def ids_validos(mes, dataset):
    """Ids reales del mes, en el mismo orden en que cargar() devuelve las filas."""
    return duckdb.connect().execute(
        f"SELECT CAST(numero_de_cliente AS BIGINT) FROM read_parquet('{Path(dataset).as_posix()}') "
        f"WHERE foto_mes = {mes} ORDER BY foto_mes, numero_de_cliente"
    ).df().iloc[:, 0].to_numpy()


def features(df):
    return [c for c in df.columns if c not in config.FUERA and not any(d in c for d in config.DRIFT)]


def mejores_params():
    """Los de la busqueda de esta corrida si existen; si no, los versionados en el repo."""
    propios = config.DIR_BUSQUEDA / "mejores_params.json"
    origen = propios if propios.exists() else config.PARAMS_ENTREGA
    log(f"hiperparametros desde {origen.relative_to(config.RAIZ)}")
    return json.loads(origen.read_text())
