"""Paso 6: archivo de entrega para el bot.

- Probabilidad final de cada cliente = promedio de las 20 semillas del paso 5.
- Se envia a los config.N_ENVIOS clientes de mayor probabilidad.
- Formato del bot: solo numero_de_cliente, sin encabezado, enteros (sin notacion cientifica).
- Verifica: cantidad de filas, ids unicos, todos existen en 202108, formato. Calcula el SHA-256
  y lo compara con el del archivo entregado (config.SHA256_ENTREGADO).

Con --por-semilla tambien escribe un archivo por semilla (para leer media y desvio en el bot).

Uso:  python pasos/p6_entrega.py [--n 12000] [--por-semilla]
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from lib.datos import ids_validos, log  # noqa: E402


def exportar(ids, prob, n, destino, validos):
    orden = np.argsort(-np.asarray(prob), kind="stable")[:n]
    sel = pd.Series(np.asarray(ids)[orden]).astype("int64")
    sel.to_csv(destino, header=False, index=False)
    txt = destino.read_text()
    assert "e+" not in txt and "." not in txt, f"formato invalido en {destino}"
    assert len(txt.splitlines()) == n, f"cantidad de filas invalida en {destino}"
    assert sel.is_unique, f"ids repetidos en {destino}"
    malos = int((~sel.isin(set(validos))).sum())
    assert malos == 0, f"{malos} ids de {destino} no existen en {config.MES_FUTURO}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=config.N_ENVIOS)
    ap.add_argument("--por-semilla", action="store_true")
    args = ap.parse_args()

    probs = pd.read_parquet(config.DIR_FINAL / "probabilidades_202108.parquet")
    validos = ids_validos(config.MES_FUTURO, config.DATASET_FINAL)
    ids = probs["numero_de_cliente"].to_numpy()
    assert np.array_equal(ids, validos), "ids de las probabilidades desalineados con 202108"
    cols = [c for c in probs.columns if c.startswith("s") and c[1:].isdigit()]
    log(f"promedio de {len(cols)} semillas, N={args.n}")

    config.DIR_ENTREGA.mkdir(parents=True, exist_ok=True)
    destino = config.DIR_ENTREGA / f"competencia_01_entrega_n{args.n}.csv"
    exportar(ids, probs[cols].mean(axis=1), args.n, destino, validos)
    sha = hashlib.sha256(destino.read_bytes()).hexdigest()
    log(f"listo -> {destino.relative_to(config.RAIZ)}  sha256 {sha}")
    if args.n == config.N_ENVIOS and len(cols) == config.N_SEMILLAS_FINAL:
        log("identico al archivo entregado: " + ("SI" if sha == config.SHA256_ENTREGADO else "NO"))
    else:
        log(f"no se compara con el archivo entregado (es N={config.N_ENVIOS} con {config.N_SEMILLAS_FINAL} semillas)")

    if args.por_semilla:
        carpeta = config.DIR_ENTREGA / f"por_semilla_n{args.n}"
        carpeta.mkdir(parents=True, exist_ok=True)
        for c in cols:
            exportar(ids, probs[c], args.n, carpeta / f"semilla_{c[1:]}_n{args.n}.csv", validos)
        log(f"{len(cols)} archivos por semilla -> {carpeta.relative_to(config.RAIZ)}")


if __name__ == "__main__":
    main()
