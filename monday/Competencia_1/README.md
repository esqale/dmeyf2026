# Primera Competencia — DMEyF 2026 (comisión lunes)

Alejandro Esquivel

Predicción de los clientes **BAJA+2** del Paquete Premium en la foto **202108**: los que se van a dar de baja **dentro de 2 meses** (siguen siendo clientes en 202109, por lo que no son BAJA+1, y ya no están en 202110). La campaña de retención estimula a los **12.000** clientes con mayor probabilidad de ser BAJA+2.

| | |
|---|---|
| Archivo entregado | `competencia_01_entrega_n12000.csv`: 12.000 `numero_de_cliente`, sin encabezado |
| SHA-256 | `7dede71a1cf143dcb05937a3a0657fcc17d32f2d473e8f0ece75d529210421fd` |
| Leaderboard público | 107,47 |
| Validación (202106, 10 semillas) | 421,0M de media, desvío 5,5M |

## Cómo correrlo

```bash
pip install -r requirements.txt
python correr_todo.py                  # todo, desde el crudo (~2 h con 16 núcleos)
python correr_todo.py --sin-busqueda   # usa params/mejores_params.json en vez de re-buscar (~1 h 20)
```

El crudo se descarga solo del bucket de la cátedra a `datos/` (o se puede copiar ahí a mano). Todo lo generado queda en `salida/`. El archivo final es `salida/entrega/competencia_01_entrega_n12000.csv`, y el último paso imprime si es idéntico (SHA-256) al entregado.

Cada paso se puede correr suelto, en orden:

| Paso | Qué hace | Tiempo |
|---|---|---|
| `pasos/p0_descargar.py` | baja `competencia_01_crudo.csv` | — |
| `pasos/p1_target.py` | `clase_ternaria` (CONTINUA / BAJA+1 / BAJA+2) | 30 s |
| `pasos/p2_features.py` | limpieza + features → `fe_base` y `fe_final` | 1 min |
| `pasos/p3_busqueda.py` | Optuna sobre `fe_base` → hiperparámetros | 45 min |
| `pasos/p4_validacion.py` | 10 semillas en 202103–04 → 202106: ganancia por N y meseta | 15 min |
| `pasos/p5_modelo_final.py` | 202103–06 → 202108, 20 semillas | 60 min |
| `pasos/p6_entrega.py` | promedio, top 12.000, verificación y SHA-256 | segundos |

`p6_entrega.py --por-semilla` además escribe un archivo por semilla, para leer media y desvío en el bot.

## Estructura

```
config.py          todos los parámetros: rutas, meses, semillas, N, espacio de Optuna, hilos
correr_todo.py     corre los pasos en orden
pasos/             un script por etapa (p0 a p6)
lib/               datos.py (carga, features), ganancia.py (curva por N, meseta, comparación),
                   modelo.py (LightGBM por semilla, escalado de mínimos)
params/            hiperparámetros del modelo entregado
experimentos/      lo que se midió y no entró: variantes de LightGBM y Random Forest
docs/DECISIONES.md cada decisión con su porqué
```

## El modelo, en una tabla

| Pieza | Elección | Por qué (detalle en `docs/DECISIONES.md`) |
|---|---|---|
| Clase | positivo = BAJA+1 + BAJA+2; la ganancia cuenta solo BAJA+2 | D6 |
| Features | clase4 (tarjetas combinadas, ratios, actividad digital, productos, rankings dentro del mes, tendencias) + **lag_1 y delta_1 de todas las variables** | D3, D10 |
| Limpieza | fuera `*_fultimo_cierre` y `*_Finiciomora` (drift de datos); `ccajas_depositos` → NULL en 202105 | D7 |
| Validación | entrenar 202103–04, medir 202106: misma distancia de 2 meses que la entrega | D5 |
| Hiperparámetros | Optuna, 30 trials × 3 semillas promediadas por trial | D8, D11 |
| Modelo final | LightGBM, 202103–06, 20 semillas promediadas, `min_data_in_leaf` escalado por filas | D13 |
| Corte | N = 12.000, centro de la meseta de validación | D12 |

## Qué se probó y no entró

Con la regla de adopción fijada antes de medir (≥ 5M y gana en ≥ 7 de 10 semillas):

| Variante | vs modelo entregado |
|---|---|
| Target solo BAJA+2 | −31,7M, 0/10 |
| Target BAJA+1 + BAJA+2 con BAJA+1 de peso 0,25 | −7,8M, 0/10 |
| Undersampling de CONTINUA (25% por semilla) | −7,5M, 2/10 |
| Entrenar sin 202103 (primer mes, sin lags) | −4,4M, 3/10 |
| `ma_3`/`slope_3` en NULL con menos de 3 meses de historia | +4,2M, 9/10 (debajo de 5M); en el público 102,43, debajo del umbral fijado de 103,1 |
| Rankings dentro del mes con cero fijo | −2,1M, 4/10 |
| Re-búsqueda de hiperparámetros sobre el dataset final | −0,2M, 4/10 |
| Marca de aguinaldo | no evaluable con la ventana de validación (sin junio en el entrenamiento) |
| Random Forest | 394,0M en una validación no comparable (referencia) |

Las limitaciones conocidas de la metodología (una sola ventana de validación, meses sin historia en el entrenamiento, sesgos de la búsqueda) están en `docs/DECISIONES.md` (D15).

Detalle en `experimentos/RESULTADOS.md`. Para reproducirlos (requieren haber corrido `p1` y `p2`):

```bash
python experimentos/variantes.py --variantes referencia solo_baja2 peso_baja1 sin_202103 ma3_completo
python experimentos/variantes.py --variantes aguinaldo rank_cero undersampling busqueda_final
python experimentos/variantes.py --final ma3_completo     # modelo final de una variante, para el público
python experimentos/rf_linea_base.py                      # Random Forest (lento)
```

## Reproducibilidad

**Verificado el 11/10/2026:** `python correr_todo.py --sin-busqueda`, desde el crudo y con `salida/` vacía, generó un archivo idéntico al entregado (mismo SHA-256) y la misma validación (421,0M, meseta 11.000–13.000). La búsqueda de Optuna (`p3`) se verificó por separado ese mismo día: desde cero encontró exactamente los hiperparámetros de `params/mejores_params.json`.

Versiones usadas: las de `requirements.txt` (Python 3.13, LightGBM 4.7.0), con 16 hilos. Con otra cantidad de hilos o de versión, LightGBM puede cambiar los últimos decimales de las probabilidades. En el corte, la distancia entre el cliente 12.000 y el 12.001 es de 1,2e-5, así que no debería cambiar la lista. Detalle en `docs/DECISIONES.md` (D14).
