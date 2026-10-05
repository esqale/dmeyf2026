"""Perfiles de clientes BAJA+2 del Paquete Premium (versión de reglas de negocio).

Uso:  python perfiles_baja2.py --csv data/competencia_01.csv --out perfiles_baja2
Requiere: pip install duckdb pandas numpy matplotlib

Qué hace:
1. Toma TODOS los clientes BAJA+2 (202103-202106) con su historia y una muestra de CONTINUA.
2. Asigna cada BAJA+2 a uno de 5 perfiles con reglas simples (en orden de prioridad).
   Las reglas se derivaron de una segmentación exploratoria (k-means sobre 17 variables
   de negocio) y coinciden con ella en el 87% de los clientes; se usan reglas porque se
   pueden explicar a negocio y cada cliente cae en un solo perfil sin ambigüedad.
     S  El sueldo que se muda     : cobró sueldo en el banco en algún mes observado
     E  El endeudado sin red      : tiene préstamo vigente o tarjeta en mora
     F  El fantasma que paga      : no usa ni tarjeta de débito ni de crédito en el mes
     A  El ahorrista digital      : usa la tarjeta de débito (opera el día a día)
     T  El cliente de una tarjeta : el resto (usa sólo la tarjeta de crédito)
3. Exporta tablas (CSV) y gráficos ejecutivos (PNG 1920x1080).
"""
import argparse
from pathlib import Path
import duckdb, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PERFILES = {  # orden fijo = color fijo
    "F": ("El fantasma que paga", "#2a78d6"),
    "T": ("El cliente de una sola tarjeta", "#eb6834"),
    "E": ("El endeudado sin red", "#1baf7a"),
    "A": ("El ahorrista digital castigado", "#eda100"),
    "S": ("El sueldo que se muda", "#e87ba4"),
}
FIEL = "#9a9893"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e4e3df"; BG = "#fcfcfb"
ETQ_REL = {-2: "4 meses\nantes", -1: "3 meses\nantes", 0: "2 meses\nantes", 1: "Último mes\ncomo cliente"}


def feats(d):
    f = pd.DataFrame({"id": d.numero_de_cliente, "mes": d.foto_mes})
    f["tx_trim"] = d.ctrx_quarter
    f["saldo"] = d.mcuentas_saldo
    f["caja_ahorro"] = d.mcaja_ahorro
    f["deuda_cc"] = (-d.mcuenta_corriente).clip(lower=0)
    f["sueldo"] = d.mpayroll + d.mpayroll2
    f["tc_consumo"] = d.mtarjeta_visa_consumo.fillna(0) + d.mtarjeta_master_consumo.fillna(0)
    f["debito_tx"] = d.ctarjeta_debito_transacciones
    f["prest_deuda"] = d.mprestamos_personales + d.mprestamos_prendarios + d.mprestamos_hipotecarios
    f["prest_cant"] = d.cprestamos_personales
    f["mora"] = ((d.Visa_delinquency.fillna(0) > 0) | (d.Master_delinquency.fillna(0) > 0)
                 | (d.Visa_status.fillna(0) > 0) | (d.Master_status.fillna(0) > 0)).astype(int)
    f["paga_mant"] = (d.ccomisiones_mantenimiento > 0).astype(int)
    f["com_mant"] = d.mcomisiones_mantenimiento
    f["digital_tx"] = d.chomebanking_transacciones + d.cmobile_app_trx
    f["inversiones"] = (d.mplazo_fijo_dolares + d.mplazo_fijo_pesos + d.minversion1_pesos
                        + d.minversion1_dolares + d.minversion2 + d.mcaja_ahorro_dolares)
    f["antig"] = d.cliente_antiguedad; f["edad"] = d.cliente_edad
    f["rent_anual"] = d.mrentabilidad_annual; f["rent_mes"] = d.mrentabilidad
    f["ntc"] = d.ctarjeta_visa + d.ctarjeta_master
    return f


def regla(r):
    if r.sueldo_hist > 0: return "S"
    if r.mora == 1 or r.prest_deuda > 0: return "E"
    if r.debito_tx == 0 and r.tc_consumo <= 0: return "F"
    if r.debito_tx > 0: return "A"
    return "T"


def cargar(csv):
    con = duckdb.connect()
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_csv('{csv}', sample_size=-1)")
    b2 = con.execute("""WITH c AS (SELECT numero_de_cliente id, foto_mes mb FROM t WHERE clase_ternaria='BAJA+2')
        SELECT t.*, (t.foto_mes//100*12+t.foto_mes%100)-(c.mb//100*12+c.mb%100) AS rel
        FROM t JOIN c ON t.numero_de_cliente=c.id""").df()
    co = con.execute("SELECT * FROM t WHERE clase_ternaria='CONTINUA'").df()
    mensual = con.execute("""SELECT foto_mes, COUNT(*) n, SUM(mrentabilidad_annual) rent
        FROM t WHERE clase_ternaria='BAJA+2' GROUP BY 1 ORDER BY 1""").df()
    return b2[b2.rel.between(-3, 1)], co, mensual


# ------------------------------------------------------------------ gráficos
def lienzo(titulo, subtitulo):
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    fig.text(0.05, 0.93, titulo, fontsize=30, weight="bold", color=INK, va="top")
    if subtitulo: fig.text(0.05, 0.855, subtitulo, fontsize=18, color=INK2, va="top")
    return fig


def estilo(ax):
    ax.set_facecolor(BG)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=15, length=0)
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)


def barras_vs_fiel(fig, rect, valor_b2, valor_fiel, color, titulo, fmt):
    ax = fig.add_axes(rect); estilo(ax)
    b = ax.bar([0, 1], [valor_b2, valor_fiel], color=[color, FIEL], width=0.6, edgecolor=BG, linewidth=2)
    for bar, v in zip(b, [valor_b2, valor_fiel]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt(v), ha="center", va="bottom",
                fontsize=24, weight="bold", color=INK)
    ax.set_xticks([0, 1], ["Se fueron", "Cliente fiel"], fontsize=16, color=INK)
    ax.set_yticks([]); ax.grid(False)
    ax.set_ylim(0, max(valor_b2, valor_fiel) * 1.25)
    ax.set_title(titulo, fontsize=18, color=INK, loc="left", pad=14)
    return ax


def linea_trayectoria(fig, rect, serie_b2, serie_fiel, color, nombre_b2, nombre_fiel, fmt, ylabel=None):
    """Línea del perfil (4 puntos relativos a la baja) contra el cliente fiel como referencia plana
    (promedio de los mismos 4 meses calendario: el fiel no tiene 'fecha de baja')."""
    ax = fig.add_axes(rect); estilo(ax)
    x = np.arange(4); ref = float(np.mean(serie_fiel))
    ax.axhline(ref, color=FIEL, lw=2.5, ls=(0, (6, 4)))
    ax.plot(x, serie_b2, color=color, lw=3.5, marker="o", ms=12, mec=BG, mew=2)
    ax.text(-0.15, serie_b2[0], fmt(serie_b2[0]), fontsize=20, weight="bold", color=INK, ha="right", va="center")
    ax.text(3.15, serie_b2[3], fmt(serie_b2[3]) + "  " + nombre_b2, fontsize=20, weight="bold", color=INK, ha="left", va="center")
    ax.text(3.15, ref, fmt(ref) + "  " + nombre_fiel, fontsize=16, color=INK2, ha="left", va="center",
            bbox=dict(facecolor=BG, edgecolor="none", pad=4))
    ax.set_xticks(x, [ETQ_REL[r] for r in [-2, -1, 0, 1]], fontsize=15)
    ax.set_xlim(-1.1, 5.2); ax.set_yticks([])
    lo = min(min(serie_b2), ref); hi = max(max(serie_b2), ref)
    ax.set_ylim(min(0, lo), hi * 1.15)
    return ax


def pct(v): return f"{v:.0%}"
def pesos(v): return f"${v/1000:,.1f} mil".replace(",", "X").replace(".", ",").replace("X", ".")
def pesos_m(v): return f"${v/1e6:,.1f} M".replace(".", ",")
def num(v): return f"{v:,.0f}".replace(",", ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/competencia_01.csv")
    ap.add_argument("--out", default="perfiles_baja2")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True)

    b2, co, mensual = cargar(a.csv)
    F = feats(b2); F["rel"] = b2.rel.values
    hist = F[F.rel <= 0].groupby("id").agg(sueldo_hist=("sueldo", "max"))
    snap = F[F.rel == 0].set_index("id").join(hist)
    snap["perfil"] = snap.apply(regla, axis=1)
    F = F.merge(snap[["perfil"]], left_on="id", right_index=True)
    C = feats(co); C["sueldo_hist"] = C.sueldo
    C_snap = C[C.mes.between(202103, 202106)]

    # ---------------- tabla de perfiles (foto 2 meses antes de la baja)
    def resumen(d):
        return pd.Series({
            "clientes": len(d), "edad_mediana": d.edad.median(), "antiguedad_meses_mediana": d.antig.median(),
            "movimientos_trimestre_mediana": d.tx_trim.median(), "pct_menos_10_movimientos": (d.tx_trim < 10).mean(),
            "pct_cobra_sueldo": (d.sueldo_hist > 0).mean(), "pct_usa_tarjeta_credito": (d.tc_consumo > 0).mean(),
            "consumo_tarjeta_mediana": d.tc_consumo.median(), "pct_usa_debito": (d.debito_tx > 0).mean(),
            "pct_con_prestamo": (d.prest_deuda > 0).mean(), "pct_en_mora": d.mora.mean(),
            "pct_cuenta_en_rojo": (d.deuda_cc > 0).mean(), "deuda_cuenta_mediana": d.deuda_cc.median(),
            "pct_paga_comision_mant": d.paga_mant.mean(), "pct_usa_canal_digital": (d.digital_tx > 0).mean(),
            "operaciones_digitales_mediana": d.digital_tx.median(), "pct_con_inversiones": (d.inversiones > 0).mean(),
            "caja_ahorro_mediana": d.caja_ahorro.median(), "rentabilidad_anual_mediana": d.rent_anual.median(),
            "rentabilidad_anual_total": d.rent_anual.sum()})
    tab = snap.groupby("perfil").apply(resumen).T
    tab["TODOS_BAJA2"] = resumen(snap); tab["CLIENTE_FIEL"] = resumen(C_snap)
    tab.loc["pct_de_las_bajas"] = tab.loc["clientes"] / len(snap)
    tab.loc["pct_de_la_rentabilidad_perdida"] = tab.loc["rentabilidad_anual_total"] / snap.rent_anual.sum()
    tab.loc[["pct_de_las_bajas", "pct_de_la_rentabilidad_perdida"], "CLIENTE_FIEL"] = np.nan
    tab = tab[list(PERFILES) + ["TODOS_BAJA2", "CLIENTE_FIEL"]]
    tab.columns = [PERFILES[c][0] if c in PERFILES else c for c in tab.columns]
    tab.round(3).to_csv(out / "tabla_perfiles.csv")
    snap[["mes", "perfil"]].assign(perfil_nombre=snap.perfil.map(lambda p: PERFILES[p][0])).to_csv(out / "cliente_perfil.csv")

    # ---------------- trayectorias (panel balanceado: clientes con 4 meses de historia)
    ids_bal = F[F.rel == -2].id.unique()
    P = F[F.id.isin(ids_bal) & F.rel.between(-2, 1)].copy()
    P["cobra_sueldo"] = (P.sueldo > 0).astype(int)
    tray = P.groupby(["perfil", "rel"]).agg(
        clientes=("id", "nunique"), pct_paga_comision=("paga_mant", "mean"), deuda_cuenta_prom=("deuda_cc", "mean"),
        pct_cobra_sueldo=("cobra_sueldo", "mean"), sueldo_prom=("sueldo", "mean"), consumo_tarjeta_prom=("tc_consumo", "mean"),
        movimientos_trim_prom=("tx_trim", "mean"), deuda_prestamos_prom=("prest_deuda", "mean"),
        prestamos_cant_prom=("prest_cant", "mean"), inversiones_prom=("inversiones", "mean"), pct_mora=("mora", "mean"))
    tray.to_csv(out / "trayectorias_perfiles.csv")

    # fieles: mismos 4 meses calendario (202103-202106), clientes presentes los 4 meses
    Cf = C[C.mes.between(202103, 202106)]
    presentes = Cf.groupby("id").size(); Cf = Cf[Cf.id.isin(presentes[presentes == 4].index)]
    fiel_mant = Cf.groupby("mes").paga_mant.mean().values
    con_prest = Cf[(Cf.mes == 202103) & (Cf.prest_deuda > 0)].id
    fiel_deuda_cc_prest = Cf[Cf.id.isin(con_prest)].groupby("mes").deuda_cc.mean().values
    con_sueldo = Cf[(Cf.mes == 202103) & (Cf.sueldo > 0)].id
    fiel_sueldo = Cf[Cf.id.isin(con_sueldo)].assign(s=lambda d: (d.sueldo > 0)).groupby("mes").s.mean().values
    fiel_consumo = Cf.groupby("mes").tc_consumo.mean().values
    pd.DataFrame({"mes": sorted(Cf.mes.unique()), "pct_paga_comision": fiel_mant,
                  "deuda_cuenta_prom_fieles_con_prestamo": fiel_deuda_cc_prest,
                  "pct_cobra_sueldo_fieles_con_sueldo": fiel_sueldo}).to_csv(out / "trayectorias_fieles.csv", index=False)
    t = lambda p, c: tray.loc[p][c].reindex([-2, -1, 0, 1]).values

    # ---------------- 00 gancho
    n_mes = mensual.n.mean(); rent_mes = mensual.rent.mean()
    fig = lienzo("Cada mes perdemos ~1.000 clientes Premium", "Clientes que se dieron de baja, por mes (marzo a junio 2021)")
    ax = fig.add_axes([0.07, 0.12, 0.5, 0.62]); estilo(ax)
    etq = ["Marzo", "Abril", "Mayo", "Junio"]
    b = ax.bar(etq, mensual.n, color=PERFILES["F"][1], width=0.6, edgecolor=BG, lw=2)
    for bar, v in zip(b, mensual.n):
        ax.text(bar.get_x() + bar.get_width() / 2, v, num(v), ha="center", va="bottom", fontsize=22, weight="bold", color=INK)
    ax.set_yticks([]); ax.grid(False); ax.set_ylim(0, mensual.n.max() * 1.2); ax.tick_params(labelsize=18)
    fig.text(0.63, 0.66, pesos_m(rent_mes), fontsize=64, weight="bold", color=INK)
    fig.text(0.63, 0.58, "de ganancia anual que se va\ncon cada camada mensual de bajas", fontsize=20, color=INK2, va="top")
    fig.text(0.63, 0.36, pesos_m(rent_mes * 12), fontsize=44, weight="bold", color=INK)
    fig.text(0.63, 0.30, "al año, si el ritmo se mantiene", fontsize=20, color=INK2, va="top")
    fig.savefig(out / "00_gancho_perdida.png", facecolor=BG); plt.close(fig)

    # ---------------- 01 mapa de perfiles
    orden = list(PERFILES)
    pc = [tab.loc["pct_de_las_bajas", PERFILES[p][0]] for p in orden]
    pr = [tab.loc["pct_de_la_rentabilidad_perdida", PERFILES[p][0]] for p in orden]
    fig = lienzo("No se van todos por lo mismo: son 5 historias distintas",
                 "Peso de cada perfil en las bajas y en la ganancia que se perdió")
    ax = fig.add_axes([0.30, 0.08, 0.62, 0.70]); estilo(ax); ax.grid(False)
    y = np.arange(len(orden))[::-1]
    for i, p in enumerate(orden):
        col = PERFILES[p][1]
        ax.barh(y[i] + 0.2, pc[i], height=0.36, color=col, edgecolor=BG, lw=2)
        ax.barh(y[i] - 0.2, pr[i], height=0.36, color=col, alpha=0.45, edgecolor=BG, lw=2, hatch="//")
        ax.text(pc[i] + 0.005, y[i] + 0.2, f"{pct(pc[i])} de las bajas", va="center", fontsize=15, color=INK)
        ax.text(pr[i] + 0.005, y[i] - 0.2, f"{pct(pr[i])} de la ganancia perdida", va="center", fontsize=15, color=INK2)
    ax.set_yticks(y, [PERFILES[p][0] for p in orden], fontsize=19, color=INK)
    ax.set_xticks([]); ax.set_xlim(0, max(pc + pr) * 1.55); ax.spines["bottom"].set_visible(False)
    fig.savefig(out / "01_mapa_perfiles.png", facecolor=BG); plt.close(fig)

    # ---------------- 02 lo que tienen en común
    T_ = tab["TODOS_BAJA2"]; Fl = tab["CLIENTE_FIEL"]
    fig = lienzo("Lo que tienen en común los que se fueron",
                 "Foto dos meses antes de la baja, comparada con un cliente fiel")
    c0 = "#2a78d6"
    barras_vs_fiel(fig, [0.06, 0.1, 0.25, 0.6], T_.pct_cobra_sueldo, Fl.pct_cobra_sueldo, c0, "Cobran el sueldo en el banco", pct)
    barras_vs_fiel(fig, [0.38, 0.1, 0.25, 0.6], T_.pct_paga_comision_mant, Fl.pct_paga_comision_mant, c0, "Pagan comisión de mantenimiento", pct)
    barras_vs_fiel(fig, [0.70, 0.1, 0.25, 0.6], T_.movimientos_trimestre_mediana, Fl.movimientos_trimestre_mediana, c0,
                   "Movimientos en 3 meses", num)
    fig.savefig(out / "02_patron_comun.png", facecolor=BG); plt.close(fig)

    # ---------------- 03 fantasma
    nom, col = PERFILES["F"]; d = tab[nom]
    fig = lienzo(f"{nom}  ·  {pct(d.pct_de_las_bajas)} de las bajas",
                 "No usa la cuenta, está en rojo… y le seguimos cobrando el mantenimiento")
    barras_vs_fiel(fig, [0.06, 0.1, 0.25, 0.6], d.movimientos_trimestre_mediana, Fl.movimientos_trimestre_mediana, col, "Movimientos en 3 meses", num)
    barras_vs_fiel(fig, [0.38, 0.1, 0.25, 0.6], d.pct_cuenta_en_rojo, Fl.pct_cuenta_en_rojo, col, "Cuenta corriente en rojo", pct)
    barras_vs_fiel(fig, [0.70, 0.1, 0.25, 0.6], d.pct_paga_comision_mant, Fl.pct_paga_comision_mant, col, "Paga comisión de mantenimiento", pct)
    fig.savefig(out / "03_perfil_fantasma.png", facecolor=BG); plt.close(fig)

    # ---------------- 04 una sola tarjeta
    nom, col = PERFILES["T"]; d = tab[nom]
    fig = lienzo(f"{nom}  ·  {pct(d.pct_de_las_bajas)} de las bajas",
                 "Sigue comprando con la tarjeta hasta el final, pero cada vez más le cobramos mantenimiento")
    linea_trayectoria(fig, [0.03, 0.13, 0.42, 0.58], t("T", "consumo_tarjeta_prom"), fiel_consumo, col,
                      "Se fueron", "Fiel", pesos).set_title("Consumo mensual con tarjeta de crédito", fontsize=18, loc="left", color=INK, pad=14)
    linea_trayectoria(fig, [0.53, 0.13, 0.42, 0.58], t("T", "pct_paga_comision"), fiel_mant, col,
                      "Se fueron", "Fiel", pct).set_title("Clientes a los que les cobramos mantenimiento", fontsize=18, loc="left", color=INK, pad=14)
    fig.savefig(out / "04_perfil_una_tarjeta.png", facecolor=BG); plt.close(fig)

    # ---------------- 05 endeudado
    nom, col = PERFILES["E"]; d = tab[nom]
    fig = lienzo(f"{nom}  ·  {pct(d.pct_de_las_bajas)} de las bajas · {pct(d.pct_de_la_rentabilidad_perdida)} de la ganancia perdida",
                 "No se va porque terminó de pagar: se va con la deuda abierta y el rojo en la cuenta creciendo")
    ax = linea_trayectoria(fig, [0.04, 0.13, 0.58, 0.58], t("E", "deuda_cuenta_prom"), fiel_deuda_cc_prest, col,
                           "Se fueron", "Fiel con préstamo", pesos)
    ax.set_title("Deuda en cuenta corriente (rojo), promedio por cliente", fontsize=18, loc="left", color=INK, pad=14)
    ids_p = P[(P.perfil == "E") & (P.rel == -2) & (P.prest_deuda > 0)].id
    cant_b2 = (snap[(snap.perfil == "E") & (snap.prest_deuda > 0)].prest_cant >= 5).mean()
    cant_fiel = (Cf[(Cf.mes == 202106) & (Cf.prest_deuda > 0)].prest_cant >= 5).mean()
    pct_pagado = (P[P.id.isin(ids_p) & (P.rel == 0)].prest_deuda == 0).mean()
    fig.text(0.70, 0.70, pct(cant_b2), fontsize=52, weight="bold", color=INK)
    fig.text(0.70, 0.64, f"de los que tienen préstamo, tiene 5 o más\n(fiel con préstamo: {pct(cant_fiel)})", fontsize=17, color=INK2, va="top")
    con_p = snap[snap.prest_deuda > 0]
    sueldo_b2_prest = (con_p.sueldo_hist > 0).mean()
    sueldo_fiel_prest = (Cf[(Cf.mes == 202106) & (Cf.prest_deuda > 0)].sueldo > 0).mean()
    fig.text(0.70, 0.47, pct(sueldo_b2_prest), fontsize=52, weight="bold", color=INK)
    fig.text(0.70, 0.41, f"de los que se fueron con préstamo\ncobraba el sueldo acá (fiel: {pct(sueldo_fiel_prest)})", fontsize=17, color=INK2, va="top")
    fig.text(0.70, 0.24, pct(pct_pagado), fontsize=52, weight="bold", color=INK)
    fig.text(0.70, 0.18, "había terminado de pagar\nsus préstamos", fontsize=17, color=INK2, va="top")
    fig.savefig(out / "05_perfil_endeudado.png", facecolor=BG); plt.close(fig)

    # ---------------- 06 ahorrista digital
    nom, col = PERFILES["A"]; d = tab[nom]
    fig = lienzo(f"{nom}  ·  {pct(d.pct_de_las_bajas)} de las bajas",
                 "El cliente más activo y digital de los que se fueron: empezamos a cobrarle y se llevó la plata")
    linea_trayectoria(fig, [0.03, 0.13, 0.42, 0.58], t("A", "pct_paga_comision"), fiel_mant, col,
                      "Se fueron", "Fiel", pct).set_title("Clientes a los que les cobramos mantenimiento", fontsize=18, loc="left", color=INK, pad=14)
    linea_trayectoria(fig, [0.53, 0.13, 0.42, 0.58], t("A", "inversiones_prom"), Cf.groupby("mes").inversiones.mean().values, col,
                      "Se fueron", "Fiel", pesos).set_title("Plazos fijos, inversiones y dólares, promedio", fontsize=18, loc="left", color=INK, pad=14)
    fig.savefig(out / "06_perfil_ahorrista_digital.png", facecolor=BG); plt.close(fig)

    # ---------------- 07 sueldo
    nom, col = PERFILES["S"]; d = tab[nom]
    fig = lienzo(f"{nom}  ·  {pct(d.pct_de_las_bajas)} de las bajas",
                 "Primero se va el sueldo; la cuenta se cierra después")
    linea_trayectoria(fig, [0.03, 0.13, 0.42, 0.58], t("S", "pct_cobra_sueldo"), fiel_sueldo, col,
                      "Se fueron", "Fiel", pct).set_title("Sigue cobrando el sueldo en el banco", fontsize=18, loc="left", color=INK, pad=14)
    linea_trayectoria(fig, [0.53, 0.13, 0.42, 0.58], t("S", "movimientos_trim_prom"),
                      Cf[Cf.id.isin(con_sueldo)].groupby("mes").tx_trim.mean().values, col,
                      "Se fueron", "Fiel", num).set_title("Movimientos en los últimos 3 meses", fontsize=18, loc="left", color=INK, pad=14)
    fig.savefig(out / "07_perfil_sueldo.png", facecolor=BG); plt.close(fig)

    print(tab.round(3).to_string()); print(tray.round(3).to_string())
    print("bajas/mes", n_mes, "rent anual por camada", rent_mes, "prestamos E", cant_b2, "sueldo prest", sueldo_b2_prest, sueldo_fiel_prest, "fiel", cant_fiel, "pagado", pct_pagado)


if __name__ == "__main__":
    main()
