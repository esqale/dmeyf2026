"""Corre el pipeline completo, un paso por vez (sin paralelizar: LightGBM usa todos los nucleos).

  p0_descargar     crudo de la catedra -> datos/
  p1_target        clase_ternaria
  p2_features      fe_base y fe_final
  p3_busqueda      Optuna sobre fe_base (~45 min)
  p4_validacion    10 semillas en 202103-04 -> 202106 (~15 min)
  p5_modelo_final  202103-06 -> 202108, 20 semillas (~60 min)
  p6_entrega       promedio, top N, verificacion y SHA-256

Uso:
  python correr_todo.py                    todo (~2 h)
  python correr_todo.py --sin-busqueda     sin p3: usa params/mejores_params.json (~1 h 20)
  python correr_todo.py --desde p5         desde un paso en adelante
"""
import argparse
import subprocess
import sys
from pathlib import Path

PASOS = ["p0_descargar", "p1_target", "p2_features", "p3_busqueda", "p4_validacion", "p5_modelo_final", "p6_entrega"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", choices=PASOS, default=PASOS[0])
    ap.add_argument("--sin-busqueda", action="store_true", help="saltea p3 y usa los hiperparametros versionados")
    ap.add_argument("--sin-validacion", action="store_true", help="saltea p4")
    args = ap.parse_args()

    raiz = Path(__file__).resolve().parent
    pasos = PASOS[PASOS.index(args.desde):]
    if args.sin_busqueda:
        pasos = [p for p in pasos if p != "p3_busqueda"]
    if args.sin_validacion:
        pasos = [p for p in pasos if p != "p4_validacion"]
    for paso in pasos:
        print(f"\n===== {paso} =====", flush=True)
        subprocess.run([sys.executable, "-u", str(raiz / "pasos" / f"{paso}.py")], cwd=raiz, check=True)


if __name__ == "__main__":
    main()
