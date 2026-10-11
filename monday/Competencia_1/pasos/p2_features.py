"""Paso 2: limpieza y feature engineering -> dos datasets.

  fe_base   features de clase4 (z402): tarjetas combinadas Visa+Master, ratios con division
            segura, actividad digital, cantidad de productos, ranking dentro del mes (contra la
            inflacion) y tendencias por cliente (lag_1, delta_1, ma_3, slope_3) de 7 variables,
            nominales y sobre el ranking. Es el dataset de la busqueda de hiperparametros.
  fe_final  fe_base + lag_1 y delta_1 de TODAS las variables numericas del crudo.
            Si config.MA3_COMPLETO: ma_3 y slope_3 en NULL cuando el cliente tiene menos de 3
            meses de historia (en 202103-04 promediaban 1 o 2 meses).
            Es el dataset del modelo entregado (docs/DECISIONES.md, D10).

Limpieza: ccajas_depositos vale 0 para todos los clientes en 202105 (en el resto de los meses
hay entre 800 y 6.500 con valor): es un error de datos -> NULL.

Uso:  python pasos/p2_features.py
"""
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from lib.datos import log  # noqa: E402


def sql_fe_base():
    rank = "".join(f"\n        , percent_rank() OVER (PARTITION BY foto_mes ORDER BY {c}) AS rank_{c}"
                   for c in config.CAMPOS_RANK)
    tend = "".join(f"""
    , lag({c}, 1) OVER ventana_cliente AS lag_1_{c}
    , {c} - lag({c}, 1) OVER ventana_cliente AS delta_1_{c}
    , avg({c}) OVER ventana_3m AS ma_3_{c}
    , regr_slope({c}, cliente_antiguedad) OVER ventana_3m AS slope_3_{c}""" for c in config.CAMPOS_TENDENCIA)
    tend_rank = "".join(f"""
    , lag(rank_{c}, 1) OVER ventana_cliente AS lag_1_rank_{c}
    , rank_{c} - lag(rank_{c}, 1) OVER ventana_cliente AS delta_1_rank_{c}
    , avg(rank_{c}) OVER ventana_3m AS ma_3_rank_{c}
    , regr_slope(rank_{c}, cliente_antiguedad) OVER ventana_3m AS slope_3_rank_{c}""" for c in config.CAMPOS_RANK)
    return f"""
CREATE OR REPLACE MACRO suma_sin_null(a, b) AS ifnull(a, 0) + ifnull(b, 0);
CREATE OR REPLACE MACRO ratio_seguro(a, b) AS a / nullif(b, 0);

CREATE OR REPLACE TABLE competencia_01_fe AS
WITH base_rank AS (
    SELECT c.* {rank}
    FROM competencia_01 c
)
SELECT
    br.*
    -- tarjetas combinadas (Visa + Master)
    , suma_sin_null(Visa_msaldototal, Master_msaldototal) AS tc_msaldototal
    , suma_sin_null(Visa_mconsumototal, Master_mconsumototal) AS tc_mconsumototal
    , suma_sin_null(Visa_mlimitecompra, Master_mlimitecompra) AS tc_mlimitecompra
    , suma_sin_null(Visa_mpagado, Master_mpagado) AS tc_mpagado
    , suma_sin_null(Visa_cconsumos, Master_cconsumos) AS tc_cconsumos
    , suma_sin_null(Visa_cadelantosefectivo, Master_cadelantosefectivo) AS tc_cadelantosefectivo
    , ifnull(Visa_delinquency, 0) + ifnull(Master_delinquency, 0) AS tc_cantidad_mora
    , least(Visa_Fvencimiento, Master_Fvencimiento) AS tc_fvencimiento_menor
    -- ratios de riesgo / uso
    , ratio_seguro(suma_sin_null(Visa_mconsumototal, Master_mconsumototal),
                   suma_sin_null(Visa_mlimitecompra, Master_mlimitecompra)) AS r_uso_limite_tc
    , ratio_seguro(ifnull(mprestamos_personales, 0) + ifnull(mprestamos_prendarios, 0)
                   + ifnull(mprestamos_hipotecarios, 0), mcuentas_saldo) AS r_endeudamiento_saldo
    , ratio_seguro(mrentabilidad, mrentabilidad_annual / 12.0) AS r_rentabilidad_vs_anual
    , ratio_seguro(cliente_antiguedad, cliente_edad * 12) AS r_antiguedad_vida
    , ratio_seguro(mcomisiones, mrentabilidad) AS r_comisiones_vs_rentabilidad
    , ratio_seguro(mcuentas_saldo, mpayroll) AS r_saldo_vs_sueldo
    -- actividad digital
    , ifnull(internet, 0) + ifnull(thomebanking, 0) + ifnull(tmobile_app, 0) + ifnull(tcallcenter, 0)
        AS dig_canales_activos
    , ifnull(chomebanking_transacciones, 0) + ifnull(cmobile_app_trx, 0) + ifnull(ccallcenter_transacciones, 0)
        AS dig_trx_total
    , ratio_seguro(ifnull(chomebanking_transacciones, 0) + ifnull(cmobile_app_trx, 0)
                   + ifnull(ccallcenter_transacciones, 0), ctrx_quarter) AS dig_ratio_trx
    -- cantidad de productos
    , ifnull(cseguro_vida, 0) + ifnull(cseguro_auto, 0) + ifnull(cseguro_vivienda, 0)
      + ifnull(cseguro_accidentes_personales, 0) AS prod_total_seguros
    , ifnull(cinversion1, 0) + ifnull(cinversion2, 0) + ifnull(cplazo_fijo, 0) AS prod_total_inversiones
    , ifnull(cprestamos_personales, 0) + ifnull(cprestamos_prendarios, 0) + ifnull(cprestamos_hipotecarios, 0)
        AS prod_total_prestamos
    -- tendencia temporal por cliente (valor nominal)
    {tend}
    -- tendencia temporal por cliente (sobre el ranking, sin efecto inflacion)
    {tend_rank}
FROM base_rank br
WINDOW
    ventana_cliente AS (PARTITION BY numero_de_cliente ORDER BY foto_mes),
    ventana_3m AS (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
"""


def columnas_numericas(con):
    tipos = ("BIGINT", "INTEGER", "DOUBLE", "FLOAT", "DECIMAL", "HUGEINT", "SMALLINT", "TINYINT")
    return [nombre for nombre, tipo, *_ in con.execute("DESCRIBE competencia_01").fetchall()
            if nombre not in ("numero_de_cliente", "foto_mes") and tipo.startswith(tipos)]


def sql_lags_todas(numericas):
    """lag_1 y delta_1 de todas las variables numericas del crudo (las 7 de CAMPOS_TENDENCIA ya estan)."""
    cols = []
    for c in numericas:
        if c in config.CAMPOS_TENDENCIA:
            continue
        cols += [f"lag({c}, 1) OVER ventana_cliente AS lag_1_{c}",
                 f"{c} - lag({c}, 1) OVER ventana_cliente AS delta_1_{c}"]
    lista = "\n    , ".join(cols)
    return f"""
CREATE OR REPLACE TABLE competencia_01_fe AS
SELECT f.*
    , {lista}
FROM competencia_01_fe f
WINDOW ventana_cliente AS (PARTITION BY numero_de_cliente ORDER BY foto_mes)
"""


def sql_ma3_completo(con):
    """ma_3_* y slope_3_* en NULL si la ventana de 3 meses no esta completa."""
    cols = [c for c, *_ in con.execute("DESCRIBE competencia_01_fe").fetchall() if c.startswith(("ma_3_", "slope_3_"))]
    reemplazos = ", ".join(f"CASE WHEN n_hist < 3 THEN NULL ELSE {c} END AS {c}" for c in cols)
    return f"""
CREATE OR REPLACE TABLE competencia_01_fe AS
SELECT * EXCLUDE (n_hist) REPLACE ({reemplazos})
FROM (SELECT *, count(*) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes
                              ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS n_hist
      FROM competencia_01_fe)
"""


def guardar(con, destino):
    destino.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY competencia_01_fe TO '{destino.as_posix()}' (FORMAT PARQUET)")
    ncol = len(con.execute("SELECT * FROM competencia_01_fe LIMIT 0").description)
    log(f"listo -> {destino.relative_to(config.RAIZ)} ({ncol} columnas)")


def main():
    if not config.DATASET_CLASE.exists():
        raise SystemExit(f"falta {config.DATASET_CLASE}: correr pasos/p1_target.py")
    con = duckdb.connect()
    con.execute(f"CREATE TABLE competencia_01 AS SELECT * FROM read_parquet('{config.DATASET_CLASE.as_posix()}')")
    con.execute("UPDATE competencia_01 SET ccajas_depositos = NULL WHERE foto_mes = 202105")

    log("feature engineering base (clase4)")
    con.execute(sql_fe_base())
    guardar(con, config.DATASET_BASE)

    log("lag_1 y delta_1 de todas las variables")
    con.execute(sql_lags_todas(columnas_numericas(con)))
    if config.MA3_COMPLETO:
        log("ma_3 y slope_3 en NULL con menos de 3 meses de historia")
        con.execute(sql_ma3_completo(con))
    guardar(con, config.DATASET_FINAL)


if __name__ == "__main__":
    main()
