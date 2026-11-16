# Análisis Bioestadístico de la Brecha de Género en Matemáticas — ICFES Saber 11 (2025-2)

Repositorio de evidencia técnica y código reproducible del proyecto de análisis bioestadístico sobre la brecha de género en el módulo de matemáticas del examen nacional **Saber 11 (periodo 2025-2)** en Colombia ($N = 576.209$ registros válidos).

* **Institución:** Universidad Industrial de Santander (UIS)
* **Docente:** Prof. Said Pertuz
* **Integrantes del equipo:**
  * Elías Arrieta (`2251161`)
  * Wilmar Herley Patiño Castaño (`225171`)
  * Keinner Pulido Cuadros (`2231213`)
  * María Castañeda Villamizar (`2232337`)
* **Video de divulgación:** `[Inserta aquí el enlace de YouTube del video final]`

---

## 1. Resumen del proyecto

¿Existe realmente una brecha de género en el desempeño en matemáticas en Colombia, o responde a factores socioeconómicos y escolares? Este proyecto construye un **radar de diagnóstico reproducible** sobre el censo completo de evaluados del Saber 11 (2025-2) para medir la magnitud real de la diferencia, evaluar su persistencia en igualdad de condiciones y detectar en qué colegios y niveles de rendimiento se concentra.

### Hallazgos principales
* **Brecha global pequeña pero significativa:** Los hombres promedian $53,9$ puntos frente a $50,4$ de las mujeres ($\Delta = 3,43$ puntos; $\text{IC}_{95\%}: [3,36, 3,50]$; $p < 0,001$). Sin embargo, el tamaño del efecto es pequeño ($d = 0,259$), con un solapamiento cercano al $90\%$ entre ambas distribuciones.
* **Persistencia tras ajuste socioeconómico:** Al controlar por estrato, educación de madre y padre, zona, naturaleza del colegio y acceso a computador/internet en grado 11 ($n = 348.209$ estudiantes en $11.725$ colegios), la brecha se mantiene en $3,33$ puntos ($\text{IC}_{95\%}: [3,22, 3,43]$), aunque se reduce a $1,17$ puntos en contextos socioeconómicos altamente favorecidos (Perfil C).
* **Peso institucional (efecto colegio):** El modelo multinivel muestra un coeficiente de correlación intraclase $\text{ICC} = 0,338$ (el $34\%$ de la variación del puntaje ocurre entre colegios) y una desviación estándar de la brecha entre instituciones de $\approx 3,23$ puntos.
* **Comportamiento por percentiles:** La regresión cuantílica revela que la brecha ajustada alcanza su máximo en el centro de la distribución ($4,00$ puntos en P50), siendo mayor que en los extremos (P10: $2,67$ puntos; P90: $3,42$ puntos).

---

## 2. Estructura del repositorio

```text
├── analisis_brecha_genero_nacional.py   # Script principal documentado y reproducible
├── resultados_brecha/                   # Salidas generadas automáticamente por el script
│   ├── resultados.txt                   # Log completo de pruebas estadísticas y modelos
│   ├── regresion_coeficientes.csv       # Coeficientes, errores estándar y valores p (OLS)
│   ├── cuantilica.csv                   # Estimaciones de brecha ajustada por percentil (P10-P90)
│   ├── fig1_distribucion_sexo.png       # Distribución de puntajes por sexo (estándar y transparente)
│   ├── fig2_brecha_contexto.png         # Forest plot por estrato, zona y naturaleza
│   └── fig3_regresion_cuantilica.png    # Curva de brecha ajustada a lo largo de los percentiles
├── .gitignore                           # Exclusión de archivos pesados y temporales
└── README.md                            # Documentación del proyecto
```

> **Nota sobre los datos crudos:** Por buenas prácticas de control de versiones y restricciones de tamaño de archivo en GitHub (> 100 MB), la base de datos original (`Examen_Saber_11_20252_nacional.txt`) no está alojada en este repositorio. Se trata de una base secundaria, pública y anonimizada que puede descargarse directamente desde el portal de **Datos Abiertos del ICFES** (periodo `2025-2`).

---

## 3. Metodología implementada en el script

El pipeline en Python ejecuta de forma secuencial:
1. **Depuración y validación:** Filtrado de registros con sexo válido (`F`/`M`), homologación de etiquetas geográficas (`URBANO` $\rightarrow$ `URBANA`) y recodificación de valores sin dato (`Sin Estrato`, `No sabe`, `No Aplica`).
2. **Contraste descriptivo y verificación de supuestos:** Prueba de Levene para homocedasticidad ($p < 0,001$), prueba $t$ de Welch para varianzas desiguales, respaldo no paramétrico $U$ de Mann-Whitney, tamaño del efecto $d$ de Cohen y estratificación en perfiles comparables (`A`, `B` y `C`).
3. **Regresión lineal múltiple e interacciones:** Estimación con errores estándar robustos agrupados por colegio (`cole_codigo_icfes`), verificación de ausencia de multicolinealidad ($\text{VIF}_{\text{máx}} \approx 3,0$), análisis de sensibilidad por datos faltantes e interacciones de género con estrato, zona y naturaleza.
4. **Modelos multinivel y regresión cuantílica:** Modelos lineales mixtos (REML) con intercepto y pendiente aleatoria por colegio, y regresión cuantílica en los percentiles 10, 25, 50, 75 y 90.

---

## 4. Requisitos y ejecución

### Instalación de dependencias
Asegúrate de tener Python 3.9 o superior e instala las librerías requeridas:

```bash
pip install pandas numpy scipy statsmodels matplotlib
```

### Ejecución del análisis
Ubica el archivo `Examen_Saber_11_20252_nacional.txt` en la raíz del proyecto (o indica su ruta como primer argumento) y ejecuta:

```bash
python analisis_brecha_genero_nacional.py Examen_Saber_11_20252_nacional.txt resultados_brecha
```

El tiempo estimado de ejecución completa es de **6 a 10 minutos** debido al ajuste del modelo multinivel con pendiente aleatoria sobre los $348.209$ registros de grado 11.

---

## 5. Referencias bibliográficas

* Abadía, L. K., & Bernal, G. L. (2017). A widening gap? A gender-based analysis of performance on the Colombian high school exit examination. *Revista de Economía del Rosario*, 20(1), 5–31.
* Carrillo-Sierra, S.-M., González-Gutiérrez, O., Arévalo-Batista, A.-E., Pereira Mora, A. F., & Arenas Tarazona, D. Y. (2023). *Determinantes del desempeño en el examen Saber 11 del año 2020 en Cúcuta: un análisis multinivel* (Documento de Trabajo Saber Investigar N.º 10). ICFES.
* Else-Quest, N. M., Hyde, J. S., & Linn, M. C. (2010). Cross-national patterns of gender differences in mathematics: A meta-analysis. *Psychological Bulletin*, 136(1), 103–127.
* ICFES. (2025). *Datos abiertos Saber 11, periodo 2025-2* [Base de datos]. Instituto Colombiano para la Evaluación de la Educación.
