"""Unica fuente de parametros del pipeline. Ningun script tiene numeros sueltos."""
from pathlib import Path

# ----------------------------------------------------------------------------- rutas
RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"                     # fuera del repo (.gitignore)
SALIDA = RAIZ / "salida"                   # fuera del repo (.gitignore)
PARAMS = RAIZ / "params"                   # en el repo: hiperparametros de la entrega

CRUDO = DATOS / "competencia_01_crudo.csv"
URL_CRUDO = "https://storage.googleapis.com/open-courses/dmeyf2026-9c6f/competencia_01_crudo.csv"

DATASET_CLASE = SALIDA / "datasets" / "competencia_01_clase.parquet"
DATASET_BASE = SALIDA / "datasets" / "competencia_01_fe_base.parquet"    # FE de clase4
DATASET_FINAL = SALIDA / "datasets" / "competencia_01_fe_final.parquet"  # base + lag/delta de todas

DIR_BUSQUEDA = SALIDA / "busqueda"
DIR_VALIDACION = SALIDA / "validacion"
DIR_FINAL = SALIDA / "modelo_final"
DIR_ENTREGA = SALIDA / "entrega"

PARAMS_ENTREGA = PARAMS / "mejores_params.json"

# ----------------------------------------------------------------------------- negocio
GANANCIA_ACIERTO = 1_072_500
COSTO_ESTIMULO = 27_500
# la entrega pide entre 8.000 y 15.000 clientes: se elige cantidad de envios, no un umbral
CORTES = list(range(8000, 15001, 500))
N_ENVIOS = 12_000  # elegido en docs/DECISIONES.md (D12)

# ----------------------------------------------------------------------------- meses
# ventana de desarrollo: misma distancia de 2 meses que la entrega real (202106 -> 202108)
MESES_VAL_TRAIN = (202103, 202104)
MES_VAL_TEST = 202106
# modelo final: todos los meses con clase completa
MESES_FINAL = (202103, 202104, 202105, 202106)
MES_FUTURO = 202108

# ----------------------------------------------------------------------------- features
FUERA = ("clase_ternaria", "foto_mes", "numero_de_cliente")
# drift de datos (z701): se excluyen junto con todo lo derivado de ellas
DRIFT = ("fultimo_cierre", "Finiciomora")

# clase4: ranking dentro del mes y tendencias por cliente
CAMPOS_RANK = ["mrentabilidad", "mcuentas_saldo", "mtarjeta_visa_consumo", "mtarjeta_master_consumo",
               "mpayroll", "ctrx_quarter"]
CAMPOS_TENDENCIA = ["ctrx_quarter", "mcuentas_saldo", "mtarjeta_visa_consumo", "mtarjeta_master_consumo",
                    "mpayroll", "mrentabilidad", "cpayroll_trx"]
# ma_3 y slope_3 en NULL cuando el cliente tiene menos de 3 meses de historia (solo en fe_final)
MA3_COMPLETO = False

# ----------------------------------------------------------------------------- semillas
SEMILLA_POOL = 17          # pool de semillas: np.random.RandomState(17), 50 primos sin reemplazo
N_SEMILLAS_TRIAL = 3       # cada trial de Optuna promedia 3 semillas
N_SEMILLAS_VALIDACION = 10
N_SEMILLAS_FINAL = 20

# ----------------------------------------------------------------------------- LightGBM
HILOS = 16  # fijo: con otra cantidad de hilos pueden cambiar los ultimos decimales
LGBM_FIJOS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "boost_from_average": True,
    "feature_pre_filter": False,
    "max_bin": 31,
    "bagging_freq": 1,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": HILOS,
    "verbose": -1,
}
# positivos del entrenamiento (z601 "(anti)Discriminacion"); la ganancia cuenta solo BAJA+2
CLASES_POSITIVAS = ("BAJA+1", "BAJA+2")
# parametros que son minimos absolutos de filas: se escalan con la cantidad de filas (z494)
PARAMS_MINIMOS = ("min_data_in_leaf",)

# ----------------------------------------------------------------------------- Optuna
N_TRIALS = 30
ESPACIO_OPTUNA = {
    # nombre: (tipo, desde, hasta, log)
    "num_leaves": ("int", 8, 512, True),
    "learning_rate": ("float", 0.005, 0.1, True),
    "num_boost_round": ("int", 100, 1500, True),
    "min_data_in_leaf": ("int", 20, 4000, True),
    "feature_fraction": ("float", 0.1, 0.9, False),
    "bagging_fraction": ("float", 0.5, 1.0, False),
    "lambda_l2": ("float", 1e-3, 100, True),
}
SUAVIZADO = 3  # media movil de 3 cortes sobre la curva de ganancia

# ----------------------------------------------------------------------------- verificacion
# SHA-256 del archivo entregado al bot el 11/10/2026 (publico 107,47)
SHA256_ENTREGADO = "7dede71a1cf143dcb05937a3a0657fcc17d32f2d473e8f0ece75d529210421fd"
