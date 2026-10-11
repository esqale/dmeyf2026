"""Paso 0: descarga el dataset crudo de la catedra (si no esta ya en datos/).

Uso:  python pasos/p0_descargar.py
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from lib.datos import log  # noqa: E402


def main():
    if config.CRUDO.exists():
        log(f"ya existe {config.CRUDO.relative_to(config.RAIZ)}, no se descarga")
        return
    config.DATOS.mkdir(parents=True, exist_ok=True)
    log(f"descargando {config.URL_CRUDO}")
    urllib.request.urlretrieve(config.URL_CRUDO, config.CRUDO)
    log(f"listo -> {config.CRUDO.relative_to(config.RAIZ)} ({config.CRUDO.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
