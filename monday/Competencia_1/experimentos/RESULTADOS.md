# Experimentos

Todas las variantes de LightGBM se midieron con el mismo protocolo que el modelo entregado:
- entrenamiento 202103 + 202104, medición en 202106;
- mismos hiperparámetros (`params/mejores_params.json`), salvo que la variante sea justamente cambiarlos;
- 10 semillas pareadas, cada variante en el N de su curva media suavizada.

**Regla de adopción, fijada antes de medir:** mejora media ≥ 5M y gana en ≥ 7 de 10 semillas. El ruido entre semillas ronda los 5–7M.

Script: `experimentos/variantes.py`.

## LightGBM

| Variante | Qué cambia | Ganancia media 202106 | vs referencia | Gana en | Decisión |
|---|---|---|---|---|---|
| FE base | features de clase4 (226) | 413,5M | — | — | punto de partida |
| **lags de todas** | + lag_1 y delta_1 de todas las variables (508) | **421,0M** | **+7,5M vs FE base** | **9/10** | ✅ **adoptada: es el modelo entregado** |
| aguinaldo | + `cobra_aguinaldo`, `mpayroll_sin_sac` y sus tendencias | 418,1M | −2,9M | 4/10 | ⚠️ no evaluable (ver abajo) |
| rankings con cero fijo | + ranking dentro del mes con 0 fijo de todos los montos | 418,9M | −2,1M | 4/10 | ❌ |
| undersampling 25% | cada semilla: todas las bajas + su propio 25% de CONTINUA | 413,5M | −7,5M | 2/10 | ❌ |
| búsqueda sobre el dataset final | Optuna (30×3) sobre las 508 features | 420,8M | −0,2M | 4/10 | ❌ (no se encontró mejora) |
| target solo BAJA+2 | positivo = solo BAJA+2 | 389,3M | −31,7M | 0/10 | ❌ |
| BAJA+1 con peso 0,25 | positivo = BAJA+1 + BAJA+2, filas BAJA+1 con peso 0,25 | 413,2M | −7,8M | 0/10 | ❌ |
| sin 202103 | entrenar solo con 202104 (202103 no tiene lags) | 416,6M | −4,4M | 3/10 | ❌ |
| `ma_3`/`slope_3` completos | NULL si el cliente tiene menos de 3 meses de historia | 425,1M | +4,2M | 9/10 | ❌ debajo del umbral de 5M; en el público 102,43 (umbral 103,1) |

Lectura:
- Lo que más aportó fue darle al modelo **la tendencia de cada variable** (lag y delta de todas).
- **El target BAJA+1 + BAJA+2 está medido:** cuanto más peso tiene BAJA+1, mejor (solo BAJA+2: −31,7M; peso 0,25: −7,8M). En el top 12.000 del modelo hay ~681 BAJA+2 y ~639 BAJA+1. Aprender el patrón de baja con los dos le sirve para encontrar más BAJA+2.
- **Sacar 202103** (que no tiene lags) pierde: sus filas aportan más de lo que ensucian.
- **El experimento de aguinaldo no es válido con esta ventana.** `cobra_aguinaldo` solo puede valer 1 en junio y diciembre, y el entrenamiento (202103–04) no tiene ninguno de los dos: el modelo no puede aprender la variable. El −2,9M es ruido y no dice nada sobre su utilidad en el modelo final, que sí entrena con junio. Para evaluarlo hace falta una ventana con junio en el entrenamiento (o el leaderboard público).
- El undersampling, además de perder ganancia, sube el desvío entre semillas (de 5,5M a 7,1M).
- Re-buscar hiperparámetros sobre el dataset final no encontró mejora (−0,2M, 4/10). Eso no prueba que los dos conjuntos sean equivalentes: solo que la diferencia no se distingue del ruido.

## Random Forest (línea de base)

Script: `experimentos/rf_linea_base.py`. Clase ternaria, imputación por mediana, Optuna sobre el OOB de 202104 (25 trials) y validación 202104 → 202106 con 5 semillas.

| Modelo | Ganancia media 202106 | Desvío | N |
|---|---|---|---|
| RF (300 árboles, `max_depth` 27, `min_samples_leaf` 97, `max_features` 0,61) | 394,0M | 6,3M | 14.000 |

**No es comparable con LightGBM:** se midió con una versión previa del FE base (sin la limpieza de `ccajas_depositos`), con 1 mes de entrenamiento en vez de 2, eligiendo hiperparámetros por OOB en ese mismo mes y con 5 semillas. Es solo una referencia de orden de magnitud. No siguió en el pipeline por costo (~4 minutos por modelo, contra ~1,5 de LightGBM).

## Corte (cantidad de envíos) en el leaderboard público

Mismo modelo, 20 archivos por punto (uno por semilla), ganancia pública media:

| N | 10.500 | 11.500 | 12.000 (promedio de semillas, 1 archivo) | 12.500 | 13.500 |
|---|---|---|---|---|---|
| Público | 101,07 ± 2,30 | 101,93 ± 2,79 | **107,47** | 102,19 ± 2,55 | 99,56 ± 1,98 |

11.500 y 12.500 empatan (la diferencia es menor que el error de la media). Se tomó 12.000, el centro de la meseta de validación (11.000–13.000), y se entrega el promedio de las 20 semillas.

El 107,47 es **un solo archivo** sobre ~25% de 202108. Es compatible con que el promedio de semillas rinda más que cada semilla suelta, pero no lo prueba: puede incluir ruido de la muestra pública.

## `ma_3` / `slope_3` con historia completa: desempate en el público

En validación dio +4,2M y ganó en 9/10, debajo del umbral de 5M. Se desempató en el público con una regla fijada antes de mirar: entrenar el final con el arreglo (`python experimentos/variantes.py --final ma3_completo`) y adoptarlo solo si la media de los 20 archivos en N = 12.000 era ≥ 103,1 (+1,0 sobre ~102,1, el promedio del modelo actual en 11.500 y 12.500).

| Modelo (20 archivos, uno por semilla) | N | Público |
|---|---|---|
| Modelo entregado | 11.500 / 12.500 | 101,93 ± 2,79 / 102,19 ± 2,55 |
| Con `ma_3`/`slope_3` completos | 12.000 | 102,43 ± 2,30 |

**No se adopta.** En el público, con 20 archivos en N = 12.000, dio **102,43 ± 2,30**, debajo del umbral fijado antes de mirar (≥ 103,1, es decir +1,0 sobre ~102,1 del modelo actual en ese corte). La mejora de validación (+4,2M, 9/10) no se trasladó a 202108.
