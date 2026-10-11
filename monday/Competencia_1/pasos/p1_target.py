"""Paso 1: crudo -> clase_ternaria (logica de z101_target_sql_local).

Para cada cliente y foto t se mira si esta en t+1 y en t+2:
  BAJA+1    no esta en t+1 (o reaparece despues de un hueco)        -> requiere conocer t+1
  BAJA+2    esta en t+1 pero no en t+2                                -> requiere conocer t+2
  CONTINUA  esta en t+1 y en t+2
Los dos ultimos meses quedan con clase NULL (202107 solo puede ser BAJA+1): no son CONTINUA.

Uso:  python pasos/p1_target.py
"""
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from lib.datos import log  # noqa: E402

SQL_TARGET = """
CREATE OR REPLACE TABLE competencia_01 AS
WITH base AS (
    SELECT * REPLACE (CAST(numero_de_cliente AS BIGINT) AS numero_de_cliente),
        (foto_mes // 100) * 12 + foto_mes % 100 AS periodo0
    FROM read_csv('{crudo}', sample_size = -1)
),
topes AS (SELECT max(periodo0) AS ultimo FROM base),
leads AS (
    SELECT b.*,
        lead(periodo0, 1) OVER (PARTITION BY numero_de_cliente ORDER BY periodo0) AS periodo1,
        lead(periodo0, 2) OVER (PARTITION BY numero_de_cliente ORDER BY periodo0) AS periodo2
    FROM base b
)
SELECT * EXCLUDE (periodo0, periodo1, periodo2, ultimo),
    CASE
        WHEN periodo0 < ultimo AND (periodo1 IS NULL OR periodo1 > periodo0 + 1) THEN 'BAJA+1'
        WHEN periodo0 < ultimo - 1 AND periodo1 = periodo0 + 1
             AND (periodo2 IS NULL OR periodo2 > periodo0 + 2) THEN 'BAJA+2'
        WHEN periodo0 < ultimo - 1 THEN 'CONTINUA'
        ELSE NULL
    END AS clase_ternaria
FROM leads, topes
"""


def main():
    if not config.CRUDO.exists():
        raise SystemExit(f"falta {config.CRUDO}: correr pasos/p0_descargar.py")
    con = duckdb.connect()
    log(f"leyendo {config.CRUDO.name} y armando clase_ternaria")
    con.execute(SQL_TARGET.format(crudo=config.CRUDO.as_posix()))
    for r in con.execute("""
            SELECT foto_mes, count(*) FILTER (WHERE clase_ternaria = 'BAJA+1'),
                   count(*) FILTER (WHERE clase_ternaria = 'BAJA+2'),
                   count(*) FILTER (WHERE clase_ternaria = 'CONTINUA'),
                   count(*) FILTER (WHERE clase_ternaria IS NULL)
            FROM competencia_01 GROUP BY 1 ORDER BY 1""").fetchall():
        log(f"  {r[0]}: BAJA+1 {r[1]:>5}  BAJA+2 {r[2]:>5}  CONTINUA {r[3]:>6}  sin clase {r[4]:>6}")
    config.DATASET_CLASE.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY competencia_01 TO '{config.DATASET_CLASE.as_posix()}' (FORMAT PARQUET)")
    log(f"listo -> {config.DATASET_CLASE.relative_to(config.RAIZ)}")


if __name__ == "__main__":
    main()
