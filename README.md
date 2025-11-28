# Sistema de Detección de Bots

Sistema de programación probabilística para detectar bots en sistemas de recomendación.

Basado en el **Capítulo 18** de "Artificial Intelligence: A Modern Approach" (Russell & Norvig).

---

## Planteamiento del Problema

### Contexto

Los **sistemas de recomendación** (Amazon, Netflix, Spotify) dependen de las calificaciones de usuarios para sugerir productos. Sin embargo, estos sistemas son vulnerables a **ataques de bots** que manipulan las calificaciones para:

- Inflar artificialmente la puntuación de productos de baja calidad
- Sabotear productos de la competencia con calificaciones negativas
- Crear múltiples cuentas falsas (Sybil attacks) para amplificar su impacto

### Problema

> ¿Cómo distinguir automáticamente entre usuarios legítimos y bots maliciosos en un sistema de recomendación, considerando la incertidumbre inherente en el comportamiento de los usuarios?

### Solución Propuesta

Este proyecto implementa **modelos probabilísticos relacionales** (RPM y OUPM) que:

1. **Modelan la incertidumbre:** Usan redes bayesianas para representar la probabilidad de que un usuario sea bot
2. **Capturan relaciones:** Consideran la relación entre usuarios, productos y calificaciones
3. **Detectan patrones anómalos:** Identifican comportamientos típicos de bots (ratings extremos, baja varianza, múltiples cuentas)
4. **Realizan inferencia aproximada:** Usan algoritmos MCMC (Gibbs Sampling, Metropolis-Hastings) para calcular probabilidades posteriores

---

## Hipótesis y Resultados

### Hipótesis 1: Los bots pueden distinguirse de usuarios reales por su patrón de calificaciones

> **H1:** Los bots tienden a dar calificaciones extremas (1 o 5 estrellas) con baja varianza, mientras que los usuarios reales muestran mayor diversidad en sus ratings.

**Resultado:** ✅ **CONFIRMADA**

El Escenario 1 demuestra que el detector logra Precision=1.0 y Recall=1.0, separando perfectamente bots de usuarios basándose en sus patrones de calificación.

### Hipótesis 2: Gibbs Sampling converge más rápido que Metropolis-Hastings para este dominio

> **H2:** Para inferencia en redes bayesianas con variables discretas, Gibbs Sampling convergerá a la distribución correcta con menos muestras que Metropolis-Hastings.

**Resultado:** ✅ **CONFIRMADA**

El Escenario 2 muestra que con 500 muestras, Gibbs infiere correctamente la calidad del libro (Q=3), mientras que MH queda sesgado hacia Q=4. Gibbs muestrea directamente de las condicionales, siendo más eficiente para este problema.

### Hipótesis 3: Los bots crean más cuentas que los usuarios legítimos

> **H3:** Los actores maliciosos (bots) tienden a crear múltiples cuentas (Sybil attacks) para amplificar su impacto, resultando en un promedio de cuentas por bot mayor que por usuario.

**Resultado:** ✅ **CONFIRMADA**

El Escenario 3 muestra que los bots tienen en promedio 3.4 cuentas vs 1.8 de los usuarios reales. Los peores ofensores (Bot_2, Bot_3) llegan a 6 cuentas cada uno.

---

## Características

- **Detección de bots** mediante modelos probabilísticos
- **Detección de Sybil attacks** (múltiples cuentas fraudulentas)
- **Inferencia MCMC** con Gibbs Sampling y Metropolis-Hastings
- **Visualizaciones** en PNG (curvas ROC, matrices de confusión, etc.)

## Instalación

```powershell
git clone https://github.com/dsolisav/ProyectoEstocasticos.git
cd ProyectoEstocasticos
pip install -r requirements.txt
```

## Uso Rápido

```powershell
python main.py
```

## Ejemplos

```powershell
python examples/escenario_1_deteccion_bots.py   # Detectar bots
python examples/escenario_2_comparacion_mcmc.py # Comparar Gibbs vs MH
python examples/escenario_3_sybil_attacks.py    # Detectar múltiples cuentas
```

## Documentación

📖 **[MANUAL_USUARIO.md](MANUAL_USUARIO.md)** - Guía completa con:
- Instalación detallada
- Ejemplos de uso
- 3 escenarios experimentales reproducibles
- Interpretación de resultados
- Verificación del sistema
- Preguntas frecuentes

## Estructura

```
ProyectoEstocasticos/
├── main.py              # Demo principal
├── MANUAL_USUARIO.md    # Documentación completa
├── requirements.txt     # Dependencias
├── src/                 # Código fuente
│   ├── models.py        # Entidades (Customer, Book, etc.)
│   ├── rpm_model.py     # Relational Probability Model
│   ├── oupm_model.py    # Open Universe PM
│   ├── bot_detection.py # Detector de bots
│   ├── inference/       # Algoritmos MCMC
│   └── visualization_plots.py  # Gráficos PNG
└── examples/            # Escenarios experimentales
    └── output/          # Gráficos generados
```

## Conceptos Implementados

| Concepto | Descripción |
|----------|-------------|
| RPM | Relational Probability Model |
| OUPM | Open Universe Probability Model |
| Gibbs Sampling | MCMC con muestreo condicional |
| Metropolis-Hastings | MCMC con accept/reject |
| Sybil Detection | Detección de múltiples cuentas |

## Métricas del Sistema

- **Precision:** 1.000
- **Recall:** 1.000
- **F1-Score:** 1.000

## Referencias

- Russell, S. & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Chapter 18

---

**Versión:** 2.0.0 | **Fecha:** Noviembre 2025
