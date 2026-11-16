#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
ANÁLISIS BIOESTADÍSTICO DE LA BRECHA DE GÉNERO EN MATEMÁTICAS - SABER 11 (2025-2)
===============================================================================
Autores:
    - Elías Arrieta (2251161)
    - Wilmar Herley Patiño Castaño (225171)
    - Keinner Pulido Cuadros (2231213)
    - María Castañeda Villamizar (2232337)
Profesor: Said Pertuz | Universidad Industrial de Santander (UIS)

Descripción:
    Pipeline reproducible para evaluar la existencia, magnitud y variación
    contextual de la brecha de género en el módulo de matemáticas del examen
    nacional Saber 11 (periodo 2025-2, n = 576.209 registros válidos).

Estructura del análisis:
    1. Carga, depuración y validación de calidad de datos.
    2. Análisis descriptivo, verificación de supuestos (Levene, asimetría)
       y contraste global (t de Welch, U de Mann-Whitney, d de Cohen).
    3. Estratificación por perfiles socioeconómicos comparables (A, B y C).
    4. Regresión lineal múltiple con errores estándar agrupados por colegio,
       diagnóstico de multicolinealidad (VIF) y análisis de sensibilidad.
    5. Modelos de interacción (género x estrato, zona y naturaleza del colegio).
    6. Modelos lineales mixtos (multinivel: estudiantes anidados en colegios)
       con intercepto y pendiente aleatoria de género.
    7. Regresión cuantílica (percentiles P10, P25, P50, P75 y P90).
    8. Exportación automática de visualizaciones (versión informe y versión
       con fondo transparente para divulgación audiovisual).

Uso desde terminal:
    python analisis_brecha_genero_nacional.py Examen_Saber_11_20252_nacional.txt [carpeta_salida]

Dependencias:
    pandas, numpy, scipy, statsmodels, matplotlib
===============================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# Configuración de rutas de entrada y salida
RUTA_DATOS = sys.argv[1] if len(sys.argv) > 1 else "Examen_Saber_11_20252_nacional.txt"
DIR_SALIDA = sys.argv[2] if len(sys.argv) > 2 else "resultados_brecha"
os.makedirs(DIR_SALIDA, exist_ok=True)

T_INICIO = time.time()
ARCHIVO_LOG = open(os.path.join(DIR_SALIDA, "resultados.txt"), "w", encoding="utf-8")


def registrar(mensaje: str) -> None:
    """Imprime un mensaje en consola y lo guarda simultáneamente en resultados.txt."""
    print(mensaje, flush=True)
    ARCHIVO_LOG.write(mensaje + "\n")
    ARCHIVO_LOG.flush()


def analizar_brecha_grupo(df_sub: pd.DataFrame, etiqueta: str, col_y: str = "punt_matematicas") -> dict:
    """
    Calcula estadísticas descriptivas, contraste paramétrico y no paramétrico,
    intervalo de confianza al 95% y tamaño del efecto (d de Cohen) entre sexos.

    Justificación estadística:
        - Prueba de Levene: evalúa homocedasticidad. Al rechazarse (p < 0.001),
          se utiliza la prueba t de Welch en lugar de t de Student clásica.
        - U de Mann-Whitney: respaldo no paramétrico robusto ante desviaciones
          de normalidad.
        - d de Cohen: cuantifica la relevancia práctica independientemente del
          tamaño muestral masivo (donde cualquier diferencia produce p -> 0).
    """
    hombres = df_sub.loc[df_sub["hombre"] == 1, col_y]
    mujeres = df_sub.loc[df_sub["hombre"] == 0, col_y]

    diferencia = hombres.mean() - mujeres.mean()
    error_estandar = np.sqrt(hombres.var() / len(hombres) + mujeres.var() / len(mujeres))
    ic_inf = diferencia - 1.96 * error_estandar
    ic_sup = diferencia + 1.96 * error_estandar

    # Desviación estándar agrupada (pooled SD) para d de Cohen
    sd_agrupada = np.sqrt(
        ((len(hombres) - 1) * hombres.var() + (len(mujeres) - 1) * mujeres.var())
        / (len(hombres) + len(mujeres) - 2)
    )
    d_cohen = diferencia / sd_agrupada
    solapamiento = 2 * stats.norm.cdf(-abs(d_cohen) / 2) * 100

    # Contrastes de hipótesis
    test_welch = stats.ttest_ind(hombres, mujeres, equal_var=False)
    test_levene = stats.levene(hombres, mujeres)
    test_mw = stats.mannwhitneyu(hombres, mujeres, alternative="two-sided") if len(df_sub) < 700000 else None

    resumen = (
        f"[{etiqueta}] n={len(df_sub):,} (F={len(mujeres):,}, M={len(hombres):,}) | "
        f"media F={mujeres.mean():.2f}, M={hombres.mean():.2f} | "
        f"DE F={mujeres.std():.2f}, M={hombres.std():.2f} | "
        f"brecha={diferencia:.2f} IC95%=({ic_inf:.2f}, {ic_sup:.2f}) | "
        f"d de Cohen={d_cohen:.3f} | Welch t={test_welch.statistic:.1f}, p={test_welch.pvalue:.2g} | "
        f"Levene p={test_levene.pvalue:.2g}"
        + (f" | Mann-Whitney p={test_mw.pvalue:.2g}" if test_mw else "")
    )
    registrar(resumen)

    return {
        "etiqueta": etiqueta,
        "n": len(df_sub),
        "media_f": mujeres.mean(),
        "media_m": hombres.mean(),
        "brecha": diferencia,
        "ic_inf": ic_inf,
        "ic_sup": ic_sup,
        "d_cohen": d_cohen,
    }


# ==============================================================================
# 1. CARGA Y DEPURACIÓN DE DATOS (DECISIONES METODOLÓGICAS D1 - D5)
# ==============================================================================
COLUMNAS_REQUERIDAS = [
    "punt_matematicas",
    "estu_genero",
    "estu_grado",
    "fami_estratovivienda",
    "fami_educacionmadre",
    "fami_educacionpadre",
    "fami_tienecomputador",
    "fami_tieneinternet",
    "cole_naturaleza",
    "cole_area_ubicacion",
    "cole_codigo_icfes",
]

raw = pd.read_csv(RUTA_DATOS, sep=";", usecols=COLUMNAS_REQUERIDAS, low_memory=False)
registrar(f"Registros leídos: {len(raw):,}")

# D1: Conservar registros con sexo biológico/género reportado válido (F o M)
df = raw[raw["estu_genero"].isin(["F", "M"])].copy()
registrar(f"Con sexo válido (F/M): {len(df):,}")

# Codificación y homologación de categorías
df["hombre"] = (df["estu_genero"] == "M").astype(int)
df["area"] = df["cole_area_ubicacion"].replace({"URBANO": "URBANA"})  # D4: Unificación de etiquetas
df["naturaleza"] = df["cole_naturaleza"]
df["estrato"] = df["fami_estratovivienda"].replace({"Sin Estrato": np.nan})  # D5: Valores faltantes
df["madre"] = df["fami_educacionmadre"].replace({"No sabe": np.nan, "No Aplica": np.nan})
df["padre"] = df["fami_educacionpadre"].replace({"No sabe": np.nan, "No Aplica": np.nan})
df["pc"] = df["fami_tienecomputador"]
df["net"] = df["fami_tieneinternet"]
Y_COL = "punt_matematicas"


# ==============================================================================
# 2. ANÁLISIS DESCRIPTIVO GLOBAL Y CONTRASTE DE WELCH (D2)
# ==============================================================================
registrar("\n=== 1. DESCRIPTIVO Y PRUEBA DE WELCH ===")
analizar_brecha_grupo(df, "Toda la base")
analizar_brecha_grupo(df[df["estu_grado"] == 11], "Solo grado 11")
analizar_brecha_grupo(df[df["estu_grado"] == 26], "Solo grado 26 (validantes)")

registrar(f"Sesgo (skewness) por sexo: {df.groupby('hombre')[Y_COL].skew().round(3).to_dict()}")
for umbral in (60, 70, 80):
    prop_m = (df[df["hombre"] == 1][Y_COL] >= umbral).mean()
    prop_f = (df[df["hombre"] == 0][Y_COL] >= umbral).mean()
    registrar(f"  % con puntaje >= {umbral}: hombres {prop_m:.1%}, mujeres {prop_f:.1%}")


# ==============================================================================
# 3. ESTRATIFICACIÓN POR PERFILES SOCIOECONÓMICOS COMPARABLES (D7)
# ==============================================================================
registrar("\n=== 2. PERFILES A/B/C (estratificación) ===")
est = df["fami_estratovivienda"]
mad = df["fami_educacionmadre"]

perfil_A = (
    (df["naturaleza"] == "OFICIAL")
    & (df["area"] == "RURAL")
    & est.isin(["Estrato 1", "Estrato 2"])
    & mad.isin(["Ninguno", "Primaria incompleta", "Primaria completa", "Secundaria (Bachillerato) incompleta"])
)
perfil_B = (
    (df["naturaleza"] == "OFICIAL")
    & (df["area"] == "URBANA")
    & (est == "Estrato 3")
    & mad.isin(["Secundaria (Bachillerato) completa", "Técnica o tecnológica incompleta", "Técnica o tecnológica completa"])
)
perfil_C = (
    (df["naturaleza"] == "NO OFICIAL")
    & (df["area"] == "URBANA")
    & est.isin(["Estrato 4", "Estrato 5", "Estrato 6"])
    & mad.isin(["Educación profesional completa", "Educación profesional incompleta", "Postgrado"])
)

res_perfiles = []
for nombre_perfil, mascara in (("Perfil A", perfil_A), ("Perfil B", perfil_B), ("Perfil C", perfil_C)):
    res_perfiles.append(analizar_brecha_grupo(df[mascara], nombre_perfil))

df_perfiles = pd.DataFrame(res_perfiles)
df_perfiles.round(3).to_csv(os.path.join(DIR_SALIDA, "perfiles_comparables.csv"), index=False)

# ==============================================================================
# 4. CONSTRUCCIÓN DE MUESTRA ANALÍTICA PARA MODELADO (D3, D6)
# ==============================================================================
registrar("\n=== 3. MUESTRA PARA LOS MODELOS (grado 11, casos completos) ===")
# D3: Se restringe a grado 11 regular (excluye validantes grado 26 sin estructura escolar)
g11 = df[df["estu_grado"] == 11]
vars_modelo = [Y_COL, "estrato", "madre", "padre", "area", "naturaleza", "pc", "net", "cole_codigo_icfes"]
d_mod = g11.dropna(subset=vars_modelo).copy()  # D6: Análisis de casos completos
d_mod["cole"] = d_mod["cole_codigo_icfes"].astype(int).astype(str)

registrar(f"n modelos = {len(d_mod):,} ({len(d_mod)/len(g11):.1%} de grado 11) | colegios = {d_mod['cole'].nunique():,}")
analizar_brecha_grupo(d_mod, "Muestra de modelos")


# ==============================================================================
# 5. REGRESIÓN LINEAL MÚLTIPLE CON ERRORES AGRUPADOS Y DIAGNÓSTICOS
# ==============================================================================
registrar("\n=== 4. REGRESIÓN LINEAL MÚLTIPLE (errores estándar agrupados por colegio) ===")
formula_nula = f"{Y_COL} ~ hombre"
formula_ajustada = f"{Y_COL} ~ hombre + C(estrato) + C(madre) + C(padre) + C(area) + C(naturaleza) + C(pc) + C(net)"

# Errores estándar robustos agrupados (clusterizados) por código ICFES del colegio
cluster_cfg = dict(cov_type="cluster", cov_kwds={"groups": d_mod["cole"]})
mod_sin_ajustar = smf.ols(formula_nula, d_mod).fit(**cluster_cfg)
mod_ajustado = smf.ols(formula_ajustada, d_mod).fit(**cluster_cfg)

for etiqueta_m, modelo in (("Sin ajustar", mod_sin_ajustar), ("Ajustado", mod_ajustado)):
    ci = modelo.conf_int().loc["hombre"].values
    registrar(
        f"{etiqueta_m}: brecha={modelo.params['hombre']:.2f} "
        f"IC95%=({ci[0]:.2f}, {ci[1]:.2f}) R2={modelo.rsquared:.3f}"
    )

pd.DataFrame({
    "coef": mod_ajustado.params,
    "ee": mod_ajustado.bse,
    "p": mod_ajustado.pvalues,
}).round(4).to_csv(os.path.join(DIR_SALIDA, "regresion_coeficientes.csv"))

# Diagnóstico de multicolinealidad mediante Factor de Inflación de la Varianza (VIF)
X_mat = mod_ajustado.model.exog
nombres_exog = mod_ajustado.model.exog_names
idx_muestra = np.random.RandomState(1).choice(len(X_mat), 30000, replace=False)
vifs = [(nombres_exog[i], variance_inflation_factor(X_mat[idx_muestra], i)) for i in range(1, X_mat.shape[1])]
registrar(f"VIF más altos: {sorted([(n, round(v, 1)) for n, v in vifs], key=lambda t: -t[1])[:5]}")

# Análisis de sensibilidad: excluyendo educación de los padres (recupera casos faltantes)
d_sens = g11.dropna(subset=[Y_COL, "estrato", "area", "naturaleza", "pc", "net", "cole_codigo_icfes"]).copy()
d_sens["cole"] = d_sens["cole_codigo_icfes"].astype(int).astype(str)
mod_sens = smf.ols(
    f"{Y_COL} ~ hombre + C(estrato) + C(area) + C(naturaleza) + C(pc) + C(net)", d_sens
).fit(cov_type="cluster", cov_kwds={"groups": d_sens["cole"]})
registrar(f"Sensibilidad (sin educ. de padres, n={len(d_sens):,}): brecha ajustada={mod_sens.params['hombre']:.2f}")


# ==============================================================================
# 6. MODELOS DE INTERACCIÓN (HETEROGENEIDAD DE LA BRECHA POR CONTEXTO)
# ==============================================================================
registrar("\n=== 5. INTERACCIONES (brecha ajustada por nivel) ===")
resultados_interaccion = []

for var_ctx in ("estrato", "area", "naturaleza"):
    mod_int = smf.ols(formula_ajustada + f" + hombre:C({var_ctx})", d_mod).fit()
    p_interaccion = anova_lm(smf.ols(formula_ajustada, d_mod).fit(), mod_int)["Pr(>F)"].iloc[1]
    brechas_nivel = {}

    for nivel in sorted(d_mod[var_ctx].unique()):
        sub_nivel = d_mod[d_mod[var_ctx] == nivel]
        f_sub = formula_ajustada.replace(f" + C({var_ctx})", "")
        fit_sub = smf.ols(f_sub, sub_nivel).fit(**dict(cov_type="cluster", cov_kwds={"groups": sub_nivel["cole"]}))
        b_val = fit_sub.params["hombre"]
        ci_val = fit_sub.conf_int().loc["hombre"].values
        brechas_nivel[nivel] = round(b_val, 2)
        resultados_interaccion.append((var_ctx, nivel, b_val, ci_val[0], ci_val[1]))

    registrar(f"hombre x {var_ctx}: p(interacción)={p_interaccion:.2g} | brecha por nivel: {brechas_nivel}")


# ==============================================================================
# 7. MODELOS MULTINIVEL (ESTUDIANTES ANIDADOS EN COLEGIOS)
# ==============================================================================
registrar("\n=== 6. MODELOS MULTINIVEL (estudiantes dentro de colegios) ===")
# Modelo nulo para calcular el Coeficiente de Correlación Intraclase (ICC)
m_nulo = smf.mixedlm(f"{Y_COL} ~ 1", d_mod, groups=d_mod["cole"]).fit(reml=True)
var_colegios, var_residual = float(m_nulo.cov_re.iloc[0, 0]), m_nulo.scale
registrar(f"Modelo nulo: var. entre colegios={var_colegios:.1f}, var. residual={var_residual:.1f}, ICC={var_colegios/(var_colegios+var_residual):.3f}")

# Modelo con intercepto aleatorio + covariables
m_ri = smf.mixedlm(formula_ajustada, d_mod, groups=d_mod["cole"]).fit(reml=True)
var_col_cond = float(m_ri.cov_re.iloc[0, 0])
registrar(
    f"Intercepto aleatorio + covariables: brecha={m_ri.params['hombre']:.2f} "
    f"(EE {m_ri.bse['hombre']:.3f}), ICC condicional={var_col_cond/(var_col_cond+m_ri.scale):.3f}"
)

# Modelo con pendiente aleatoria de género (evalúa cuánto cambia la brecha entre colegios)
registrar("Ajustando pendiente aleatoria de género (lento)...")
m_rs = smf.mixedlm(formula_ajustada, d_mod, groups=d_mod["cole"], re_formula="~hombre").fit(reml=True, method=["lbfgs"])
registrar(f"Pendiente aleatoria: brecha media={m_rs.params['hombre']:.2f} (EE {m_rs.bse['hombre']:.3f})")
registrar("Matriz de covarianza de efectos aleatorios (intercepto, hombre):\n" + str(m_rs.cov_re.round(2)))
sd_pendiente = np.sqrt(float(m_rs.cov_re.iloc[1, 1]))
registrar(f"DE de la brecha entre colegios ≈ {sd_pendiente:.2f} puntos")


# ==============================================================================
# 8. REGRESIÓN CUANTÍLICA (BRECHA A LO LARGO DE LA DISTRIBUCIÓN)
# ==============================================================================
registrar("\n=== 7. REGRESIÓN CUANTÍLICA (submuestra de 150.000, semilla 1) ===")
sub_cuantil = d_mod.sample(150000, random_state=1)
filas_cuantil = []

for q in (0.10, 0.25, 0.50, 0.75, 0.90):
    rq = smf.quantreg(formula_ajustada, sub_cuantil).fit(q=q, max_iter=3000, p_tol=1e-4)
    ci_q = rq.conf_int().loc["hombre"].values
    filas_cuantil.append((q, rq.params["hombre"], ci_q[0], ci_q[1]))
    registrar(f"  cuantil {q:.2f}: brecha ajustada={rq.params['hombre']:.2f} IC95%=({ci_q[0]:.2f}, {ci_q[1]:.2f})")

df_cuantil = pd.DataFrame(filas_cuantil, columns=["cuantil", "brecha", "ic_inf", "ic_sup"])
df_cuantil.round(3).to_csv(os.path.join(DIR_SALIDA, "cuantilica.csv"), index=False)


# ==============================================================================
# 9. GENERACIÓN AUTOMÁTICA DE GRÁFICAS (ESTÁNDAR Y FONDO TRANSPARENTE)
# ==============================================================================
registrar("\n=== 8. GENERANDO GRÁFICAS PARA INFORME Y VIDEO ===")


def guardar_figura(fig, nombre_base: str) -> None:
    """Guarda cada figura en PNG estándar (fondo blanco) y PNG transparente (video)."""
    fig.savefig(os.path.join(DIR_SALIDA, f"{nombre_base}.png"), dpi=300, bbox_inches="tight", transparent=False)
    fig.savefig(os.path.join(DIR_SALIDA, f"{nombre_base}_transparente.png"), dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


# Gráfica 1: Distribución de puntajes por sexo (solapamiento)
fig1, ax1 = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(0, 100, 51)
ax1.hist(df.loc[df["hombre"] == 0, Y_COL], bins=bins, density=True, alpha=0.45, color="#D95F02", label="Mujeres (Media: 50,4)")
ax1.hist(df.loc[df["hombre"] == 1, Y_COL], bins=bins, density=True, alpha=0.45, color="#1B9E77", label="Hombres (Media: 53,9)")
ax1.axvline(df.loc[df["hombre"] == 0, Y_COL].mean(), color="#D95F02", linestyle="--", linewidth=2)
ax1.axvline(df.loc[df["hombre"] == 1, Y_COL].mean(), color="#1B9E77", linestyle="--", linewidth=2)
ax1.set_title("Distribución del puntaje en matemáticas por sexo (n = 576.209)", fontweight="bold")
ax1.set_xlabel("Puntaje en matemáticas (0–100)")
ax1.set_ylabel("Densidad")
ax1.legend(frameon=False)
ax1.spines[["top", "right"]].set_visible(False)
guardar_figura(fig1, "fig1_distribucion_sexo")

# Gráfica 2: Forest plot de brecha ajustada por contexto
df_int = pd.DataFrame(resultados_interaccion, columns=["variable", "nivel", "brecha", "ic_inf", "ic_sup"])
orden_niveles = ["Estrato 1", "Estrato 2", "Estrato 3", "Estrato 4", "Estrato 5", "Estrato 6", "RURAL", "URBANA", "OFICIAL", "NO OFICIAL"]
df_int = df_int.set_index("nivel").loc[orden_niveles].reset_index()

fig2, ax2 = plt.subplots(figsize=(8, 5))
y_pos = np.arange(len(df_int))
ax2.axvline(mod_ajustado.params["hombre"], color="gray", linestyle=":", linewidth=1.5, label="Media ajustada (3,33)")
ax2.errorbar(
    df_int["brecha"], y_pos,
    xerr=[df_int["brecha"] - df_int["ic_inf"], df_int["ic_sup"] - df_int["brecha"]],
    fmt="o", color="#1B9E77", ecolor="#1B9E77", elinewidth=2, capsize=3
)
for i, row in df_int.iterrows():
    ax2.text(row["brecha"], i - 0.25, f"{row['brecha']:.2f}", ha="center", fontsize=9, fontweight="bold")
ax2.set_yticks(y_pos)
ax2.set_yticklabels(df_int["nivel"])
ax2.invert_yaxis()
ax2.set_xlabel("Brecha ajustada (Hombres − Mujeres, puntos) e IC 95%")
ax2.set_title("La brecha ajustada varía según el contexto socioeconómico y escolar", fontweight="bold")
ax2.spines[["top", "right"]].set_visible(False)
guardar_figura(fig2, "fig2_brecha_contexto")

# Gráfica 3: Regresión cuantílica (P10 a P90)
fig3, ax3 = plt.subplots(figsize=(7, 4.5))
etiquetas_q = [f"P{int(q*100)}" for q in df_cuantil["cuantil"]]
ax3.plot(etiquetas_q, df_cuantil["brecha"], marker="o", color="#1B9E77", linewidth=2.5)
ax3.fill_between(etiquetas_q, df_cuantil["ic_inf"], df_cuantil["ic_sup"], color="#1B9E77", alpha=0.2)
for i, row in df_cuantil.iterrows():
    ax3.text(i, row["brecha"] + 0.12, f"{row['brecha']:.2f}", ha="center", fontweight="bold")
ax3.set_ylim(1.5, 4.8)
ax3.set_ylabel("Brecha ajustada (puntos)")
ax3.set_xlabel("Percentil del puntaje")
ax3.set_title("Brecha ajustada por percentil: mayor en el centro (P50)", fontweight="bold")
ax3.spines[["top", "right"]].set_visible(False)
guardar_figura(fig3, "fig3_regresion_cuantilica")

# Gráfica 4: Histograma de brecha por colegio (>= 15 estudiantes de cada sexo)
resumen_cole = d_mod.groupby(["cole", "hombre"])[Y_COL].agg(["count", "mean"]).unstack()
validos_cole = resumen_cole[(resumen_cole[("count", 0)] >= 15) & (resumen_cole[("count", 1)] >= 15)]
brechas_cole = validos_cole[("mean", 1)] - validos_cole[("mean", 0)]
media_cole = brechas_cole.mean()
pct_positiva = (brechas_cole > 0).mean() * 100

registrar(
    f"Colegios con >=15 estudiantes por sexo: {len(brechas_cole):,} | "
    f"brecha media={media_cole:.2f} | {pct_positiva:.1f}% con brecha positiva"
)

fig4, ax4 = plt.subplots(figsize=(7.5, 4.5))
bins_cole = np.arange(-10, 22, 2)
ax4.hist(brechas_cole, bins=bins_cole, color="#1B9E77", edgecolor="white", alpha=0.9)
ax4.axvline(0, color="black", linewidth=1.2)
ax4.axvline(media_cole, color="#D95F02", linestyle="--", linewidth=2)
ax4.text(
    media_cole + 0.6,
    ax4.get_ylim()[1] * 0.90,
    f"Media {media_cole:.1f}".replace(".", ","),
    color="#D95F02",
    fontweight="bold",
    fontsize=11,
)
ax4.set_title(f"Brecha por colegio ({len(brechas_cole):,} colegios)".replace(",", "."), fontweight="bold")
ax4.set_xlabel("Brecha del colegio (hombres − mujeres)")
ax4.set_ylabel("Colegios")
ax4.spines[["top", "right"]].set_visible(False)
guardar_figura(fig4, "fig4_brecha_colegios")

# Gráfica 5: Brecha en perfiles socioeconómicos comparables (A, B y C)
fig5, ax5 = plt.subplots(figsize=(7, 4.2))
etiquetas_perf = [
    "Perfil A\n(Oficial, Rural,\nEstrato 1-2)",
    "Perfil B\n(Oficial, Urbano,\nEstrato 3)",
    "Perfil C\n(Privado, Urbano,\nEstrato 4-6)",
]
colores_perf = ["#D95F02", "#E6AB02", "#1B9E77"]
barras = ax5.bar(etiquetas_perf, df_perfiles["brecha"], color=colores_perf, width=0.52, alpha=0.9)
ax5.errorbar(
    range(len(df_perfiles)),
    df_perfiles["brecha"],
    yerr=[df_perfiles["brecha"] - df_perfiles["ic_inf"], df_perfiles["ic_sup"] - df_perfiles["brecha"]],
    fmt="none",
    ecolor="#222222",
    elinewidth=1.5,
    capsize=4,
)
for i, fila in df_perfiles.iterrows():
    ax5.text(
        i,
        fila["ic_sup"] + 0.15,
        f"{fila['brecha']:.2f} pts\n(d = {fila['d_cohen']:.2f})".replace(".", ","),
        ha="center",
        fontsize=9.5,
        fontweight="bold",
    )
ax5.set_ylim(0, 4.3)
ax5.set_ylabel("Brecha (Hombres − Mujeres, puntos)")
ax5.set_title("Brecha en perfiles comparables: se reduce en el Perfil C", fontweight="bold")
ax5.spines[["top", "right"]].set_visible(False)
guardar_figura(fig5, "fig5_perfiles_comparables")

registrar(f"\nListo en {(time.time() - T_INICIO)/60:.1f} min. Resultados y figuras en: {DIR_SALIDA}/")
ARCHIVO_LOG.close()