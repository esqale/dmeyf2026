"""Curva de ganancia por cantidad de envios, suavizado, meseta y comparacion pareada."""
import numpy as np
import pandas as pd

import config


def curva(prob, clases):
    """Ganancia acumulada enviando a los N de mayor probabilidad, para cada N de config.CORTES.
    Solo BAJA+2 suma; cualquier otro estimulado resta."""
    orden = np.argsort(-np.asarray(prob), kind="stable")
    gan = np.where(np.asarray(clases)[orden] == "BAJA+2", config.GANANCIA_ACIERTO, -config.COSTO_ESTIMULO)
    acum = np.cumsum(gan)
    return np.array([acum[n - 1] for n in config.CORTES], dtype=float)


def suave(c):
    """Media movil centrada: se elige la zona alta de la curva, no un pico aislado."""
    return pd.Series(c).rolling(config.SUAVIZADO, center=True, min_periods=1).mean().to_numpy()


def resumen(tb):
    """tb: una fila por semilla con columnas str(N). N elegido sobre la curva media suavizada."""
    cortes = [str(n) for n in config.CORTES]
    media = suave(tb[cortes].mean().to_numpy())
    i = int(np.argmax(media))
    col = cortes[i]
    meseta = [n for n, v in zip(config.CORTES, media) if v >= media.max() - 2_000_000]
    return {"N": config.CORTES[i], "media": tb[col].mean(), "desvio": tb[col].std(), "minimo": tb[col].min(),
            "meseta": (meseta[0], meseta[-1]), "curva_suave": dict(zip(config.CORTES, media))}


def comparar(tb_base, tb_nueva, mejora_min=5_000_000, gana_min=7):
    """Comparacion pareada (misma semilla), cada variante en su N. Regla de adopcion:
    mejora media >= 5M y gana en >= 7 de 10 semillas."""
    nb, nn = resumen(tb_base)["N"], resumen(tb_nueva)["N"]
    d = (tb_nueva.set_index("semilla")[str(nn)] - tb_base.set_index("semilla")[str(nb)]).dropna()
    return {"diferencia": d.mean(), "desvio": d.std(), "gana": int((d > 0).sum()), "semillas": len(d),
            "adopta": bool(d.mean() >= mejora_min and (d > 0).sum() >= gana_min)}
