# Decisiones

Qué se decidió, por qué y qué se descartó. Los números de cada experimento están en `experimentos/RESULTADOS.md`.

---

## D1 — Qué se predice y en qué formato
- Clientes de la foto **202108** que van a ser **BAJA+2**, es decir, que se dan de baja **dentro de 2 meses**: siguen siendo clientes en 202109 (no son BAJA+1) y ya no están en 202110. A los BAJA+1 se llega tarde para la campaña de retención.
- Ganancia: +1.072.500 por cada BAJA+2 estimulado y −27.500 por cualquier otro estimulado.
- Entrega: CSV con solo `numero_de_cliente`, sin encabezado, enteros, entre 8.000 y 15.000 filas. Por eso se elige una **cantidad de envíos N** (los N de mayor probabilidad), no un umbral.

## D2 — Meses utilizables
- La clase de la foto *t* necesita ver *t+1* y *t+2*: solo **202103–202106** tienen clase completa.
- 202107 (solo puede ser BAJA+1) y 202108 (sin clase) se usan únicamente para calcular features. Sus filas sin clase **no** son CONTINUA (z101).

## D3 — Todo desde el crudo, en código
- `p1_target.py` arma la clase con la lógica de z101 y `p2_features.py` las features. No hay ningún dataset editado a mano.
- Respecto de clase4, una corrección: la ventana de `ma_3` / `slope_3` es de 3 meses (`2 PRECEDING` + el actual; `3 PRECEDING` son 4 filas).

## D4 — LightGBM como modelo; Random Forest como línea de base
- LightGBM es el modelo de los notebooks de la materia (z601) y es varias veces más rápido por modelo, lo que permite semillerío y búsqueda en el tiempo disponible.
- El RF queda en `experimentos/rf_linea_base.py`. Su validación (394,0M) **no es comparable** con la de LightGBM (otra ventana, otra versión de features, otra búsqueda): es solo una referencia.

## D5 — Ventana de desarrollo: entrenar 202103 + 202104, medir 202106
- En la entrega, la baja más reciente que se puede aprender es la de agosto (BAJA+2 de 202106) y la que se predice es la de octubre (202108): **2 meses de distancia**.
- Entrenando con 202103–04 y midiendo en 202106 se reproduce exactamente esa situación (se aprende hasta las bajas de junio y se predicen las de agosto). Es el esquema de los notebooks de la materia (z301/z401/z601: 202103 → 202105).
- **Descartado:** entrenar con 202103–05. Da más datos pero incluye bajas de julio, un mes "del futuro" si uno está parado en junio, así que la medición sale optimista.

## D6 — Target binario: BAJA+1 + BAJA+2
- Positivo = BAJA+1 o BAJA+2 (z601, "(anti)Discriminación"): hay más ejemplos para aprender el patrón de baja. La ganancia se sigue midiendo contando solo BAJA+2.
- **Medido:** solo BAJA+2 pierde 31,7M (0/10) y BAJA+1 con peso 0,25 pierde 7,8M (0/10). En el top 12.000 hay ~681 BAJA+2 y ~639 BAJA+1: cada BAJA+1 enviado cuesta, pero entrenar con ellos hace que el modelo encuentre más BAJA+2.

## D7 — Limpieza de datos
- Fuera del modelo: `*_fultimo_cierre` y `*_Finiciomora` con todo lo derivado. Son drift de datos, no de comportamiento: `fultimo_cierre` tiene PSI ~30–39 (z701) y `Finiciomora` es casi todo nulo. También quedan fuera `numero_de_cliente` y `foto_mes`.
- `ccajas_depositos` vale 0 para los 163.768 clientes de 202105, contra 800–6.500 con valor en los demás meses: error de datos → NULL en ese mes.

## D8 — Búsqueda de hiperparámetros
- Optuna (TPE con semilla fija), 30 trials, sobre el dataset `fe_base`.
- **Cada trial promedia 3 semillas** y se evalúa con la curva de ganancia suavizada (media móvil de 3 cortes): con una sola semilla el valor del trial tiene demasiado ruido (z301).
- Espacio: `learning_rate` 0,005–0,1 · `num_leaves` 8–512 · `num_boost_round` 100–1.500 · `min_data_in_leaf` 20–4.000 · `feature_fraction` 0,1–0,9 · `bagging_fraction` 0,5–1 · `lambda_l2` 0,001–100.
- Resultado (trial 24): `num_leaves` 108, `learning_rate` 0,0109, 1.180 árboles, `min_data_in_leaf` 230, `feature_fraction` 0,34, `bagging_fraction` 0,72, `lambda_l2` 0,17.

## D9 — Regla para adoptar una variante (fijada antes de medir)
- Misma ventana, mismos hiperparámetros y 10 semillas pareadas, cada variante en su N.
- Se adopta solo si la mejora media es **≥ 5M** y gana en **≥ 7 de 10** semillas.
- Sin una regla previa se termina eligiendo lo que salió mejor por suerte (la maldición del ganador, z301).

## D10 — Features del modelo entregado: FE base + lag_1 y delta_1 de todas las variables
- +7,5M sobre el FE base, gana en 9/10 semillas, y además baja el desvío (6,8M → 5,5M).
- En el leaderboard público, con 20 archivos por modelo: 95,87 → 101,93.
- Se probaron y **no** se adoptaron: rankings con cero fijo, undersampling, target solo BAJA+2 o con BAJA+1 de menor peso, y entrenar sin 202103 (ver RESULTADOS.md).
- **El aguinaldo no se pudo evaluar:** la marca solo vale 1 en junio y diciembre, y el entrenamiento de la ventana (202103–04) no tiene ninguno de los dos meses.

## D11 — Se mantienen los hiperparámetros de la búsqueda sobre el FE base
- Re-buscar con Optuna sobre el dataset final dio −0,2M y ganó en 4/10: no se encontró mejora, lo que no prueba que sean equivalentes. Se mantienen los que ya estaban validados y confirmados en el público.

## D12 — Corte: N = 12.000
- Validación (10 semillas): meseta de 11.000 a 13.000, máximo en 11.500–12.000.
- Público (20 archivos por punto): 10.500 → 101,07 · 11.500 → 101,93 · 12.500 → 102,19 · 13.500 → 99,56. 11.500 y 12.500 empatan (la diferencia es menor que el error de la media).
- Ante empate, el centro de la meseta de validación. No se toma el punto más alto del público, porque esa diferencia es ruido.

## D13 — Modelo final: 202103–202106, 20 semillas promediadas
- Se entrena con los 4 meses con clase. `min_data_in_leaf` se escala por la cantidad de filas (230 → 461): los mínimos de filas por hoja son absolutos, y sin escalar el modelo final quedaría menos regularizado que el validado (`param_normalizado`, z494).
- 20 modelos idénticos salvo la semilla. La probabilidad final de cada cliente es el promedio, que baja la varianza respecto de depender de una sola semilla. En el público, el promedio dio 107,47 contra ~102 de la media de los archivos por semilla. Es un solo archivo, así que no prueba cuánto suma el promedio.

## D14 — Reproducibilidad
- Filas ordenadas por (`foto_mes`, `numero_de_cliente`), un `lgb.Dataset` nuevo por semilla (LightGBM arma los bins muestreando con la semilla), `deterministic=True`, 16 hilos fijos, pool fijo de semillas (`RandomState(17)`).
- `numero_de_cliente` viene del crudo como DOUBLE: se castea a entero **antes** de bajar el resto de las columnas a float32. Sin esto float32 redondea los ids de más de 16,7 millones (pasó en un primer envío y el bot lo rechazó).
- `p6_entrega.py` verifica filas, ids únicos y existentes en 202108 y formato, y compara el SHA-256 con el del archivo entregado.
- **Verificado el 11/10/2026:** la corrida de punta a punta (`correr_todo.py --sin-busqueda`, desde el crudo) reprodujo el archivo entregado con el mismo SHA-256, y `p3` desde cero encontró los mismos hiperparámetros.

## D15 — Limitaciones conocidas
- **Una sola ventana de validación (202106).** Las comparaciones pareadas reducen el ruido entre variantes, pero no protegen contra decisiones que solo valen para junio (un mes con aguinaldo). La ganancia absoluta en 202106 tiene una incertidumbre de ~±30M solo por qué clientes caen en ese mes (~680 BAJA+2 en el top). El público de 202108 se usó como segundo control independiente.
- **202103 no tiene lags ni deltas** (es el primer mes del dataset) y `ma_3`/`slope_3` promedian 1 o 2 meses en 202103–04. En la validación, la mitad del entrenamiento tiene esas columnas vacías o incompletas y el test no: lo más probable es que la validación subestime el aporte de las tendencias.
- Las 3 semillas de cada trial de Optuna son las 3 primeras de las 10 de validación: hay un sesgo leve a favor de los hiperparámetros elegidos.
- Optuna optimiza el máximo de la curva suavizada (un objetivo optimista) con 30 trials para 7 hiperparámetros: la búsqueda tiene poca cobertura.
- `comparar` evalúa cada variante en su propio N, lo que le da un margen chico de selección a la variante.
- `lag` y las ventanas cuentan filas, no meses: para los 23 clientes con algún mes faltante, "el mes anterior" no es el mes calendario anterior. Impacto despreciable.

## D16 — No se adopta `ma_3`/`slope_3` con historia completa
- Validación: +4,2M y gana en 9/10, debajo del umbral de D9 (5M).
- Desempate en el público con la regla fijada antes de mirar: adoptar solo si la media de 20 archivos en N = 12.000 es ≥ 103,1. Dio 102,43 ± 2,30.
- Queda disponible en el código, apagado (`config.MA3_COMPLETO = False`).
