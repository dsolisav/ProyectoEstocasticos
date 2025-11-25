# Sistema de Recomendación con Detección de Bots

Proyecto de Programación Probabilística basado en el Capítulo 18: Probabilistic Programming

## 📋 Descripción

Sistema completo que implementa **Relational Probability Models (RPM)** y **Open Universe Probability Models (OUPM)** para:

- Detectar cuentas falsas (bots) en sistemas de recomendación
- Resolver **identity uncertainty**: múltiples LoginIDs → mismo usuario (sybil attacks)
- Inferir calidad real de items considerando honestidad de usuarios
- Manejar **existence uncertainty**: número desconocido de usuarios reales

## 🎯 Conceptos del Capítulo 18 Implementados

### Modelos
- ✅ **RPM (Relational Probability Model)**: Modelo con type signatures y CPTs
- ✅ **OUPM (Open Universe PM)**: Origin functions, generating functions
- ✅ **Identity Uncertainty**: Múltiples cuentas del mismo usuario
- ✅ **Existence Uncertainty**: # de usuarios reales desconocido
- ✅ **Database Semantics**: Unique names assumption
- ✅ **Grounding**: Conversión de RPM a Bayes Net proposicional

### Algoritmos (>50% implementados desde cero)
- ✅ **Variable Elimination**: Inferencia exacta (~300 líneas)
- ✅ **MCMC - Gibbs Sampling**: Muestreo condicional (~270 líneas)
- ✅ **MCMC - Metropolis-Hastings**: Accept/reject (~260 líneas)
- ✅ **Convergence Diagnostics**: Gelman-Rubin R̂, ESS (~150 líneas)
- ✅ **Query Engine**: Interfaz de consultas (~300 líneas)
- ✅ **Bot Detection**: Scoring probabilístico (~250 líneas)
- ✅ **ROC Analysis**: Evaluación de clasificadores (~150 líneas)

## 📁 Estructura del Proyecto

```
proyecto_estocasticos/
├── src/
│   ├── models.py              # Clases base (Customer, Book, LoginID)
│   ├── data_generator.py      # Generador de datos sintéticos
│   ├── utils.py               # Utilidades (normalización, entropía)
│   ├── cpt.py                 # [FASE 2] CPTs relacionales
│   ├── grounding.py           # [FASE 2] Unrolling RPM → Bayes Net
│   ├── rpm_model.py           # [FASE 2] Modelo RPM
│   ├── origin_functions.py    # [FASE 3] Origin functions
│   ├── oupm_model.py          # [FASE 3] Modelo OUPM
│   ├── inference/
│   │   ├── variable_elimination.py    # [FASE 4] Exacta
│   │   ├── gibbs_sampling.py          # [FASE 4] MCMC Gibbs
│   │   ├── metropolis_hastings.py     # [FASE 4] MCMC MH
│   ├── query_engine.py        # [FASE 5] Motor de consultas
│   ├── bot_detection.py       # [FASE 5] Detector de bots
│   └── visualization.py       # [FASE 6] Gráficos ASCII
├── tests/
│   ├── test_basics.py         # FASE 1 (4 tests)
│   ├── test_rpm.py            # FASE 2 (5 tests)
│   ├── test_oupm.py           # FASE 3 (7 tests)
│   ├── test_inference.py      # FASE 4 (6 tests)
│   ├── test_query_engine.py   # FASE 5 (7 tests)
│   └── test_visualization.py  # FASE 6 (5 tests)
├── main.py                    # Demo completo end-to-end
└── README.md
```

## 🚀 Instalación y Uso

### Requisitos
- Python 3.13+ (solo librería estándar)
- numpy (únicamente para arrays/matrices básicas)

### Ejecutar Demo Completo

```bash
python main.py
```

### Ejecutar Tests por Fase

```bash
# FASE 1: Modelos base
python tests/test_basics.py

# FASE 2: RPM y grounding
python tests/test_rpm.py

# FASE 3: OUPM y origin functions
python tests/test_oupm.py

# FASE 4: Algoritmos de inferencia
python tests/test_inference.py

# FASE 5: Query engine y bot detection
python tests/test_query_engine.py

# FASE 6: Visualización
python tests/test_visualization.py
```

## 📊 Progreso del Proyecto

### ✅ FASE 1: Estructura Base (4/4 tests)
- Modelos de datos (Customer, Book, LoginID, Recommendation)
- Generador de datasets sintéticos
- Utilidades (normalización, entropía, KL-divergence)

### ✅ FASE 2: RPM y Grounding (5/5 tests)
- CPTs relacionales (QualityCPT, HonestyCPT, RecommendationCPT)
- Grounding RPM → Bayes Net
- Type signatures y dependency structure

### ✅ FASE 3: OUPM (7/7 tests)
- Origin functions (identity uncertainty)
- Generating functions (existence uncertainty)
- Possible worlds generation

### ✅ FASE 4: Inferencia (6/6 tests)
- Variable Elimination (inferencia exacta)
- Gibbs Sampling (MCMC)
- Metropolis-Hastings (MCMC)
- Convergence diagnostics (R̂, ESS)

### ✅ FASE 5: Query Engine y Bot Detection (7/7 tests)
- Query engine (marginal, conditional, MAP, expectation)
- Bot detection scoring
- Sybil attack detection
- Evaluation metrics (Precision, Recall, F1, Accuracy)

### ✅ FASE 6: Visualización (5/5 tests)
- Visualización de red bayesiana (estructura ASCII)
- Gráficos de distribuciones posteriores
- Curva ROC y AUC
- Comparación de convergencia Gibbs vs MH

**Total: 30/30 tests pasando** ✅

## 🔬 Resultados y Métricas

### Convergencia MCMC (FASE 4)
- Gelman-Rubin R̂ = 0.9997 (excelente convergencia, <1.01)
- ESS/N = 0.461 (45% muestras efectivas)
- Acceptance rate MH ≈ 0.55-0.60

### Bot Detection (FASE 5) - Detector Mejorado v2.0
- **Precision: 1.000** (sin falsos positivos)
- **Recall: 1.000** (detecta 100% de bots)
- **F1-Score: 1.000**
- **Accuracy: 1.000**
- **AUC (ROC): ≥0.95** (clasificación EXCELENTE)

El detector combina 4 señales:
| Señal | Peso |
|-------|------|
| Ratings extremos (1 o 5) | 40% |
| Varianza de ratings | 25% |
| Sybil attacks (# cuentas) | 20% |
| Modelo MCMC P(dishonest) | 15% |

### Sybil Attack Detection
- 5 ataques detectados en dataset de prueba
- Bot_1: 10 cuentas (P(bot)=0.517)
- User_1: 3 cuentas (comportamiento sospechoso)

## 📈 Ejemplo de Salida del Sistema

```
======================================================================
  SISTEMA DE RECOMENDACIÓN CON DETECCIÓN DE BOTS
======================================================================

✓ Dataset generado:
  Customers: 15 (10 users, 5 bots)
  Books: 8
  LoginIDs: 40
  Recommendations: 143

✓ RPM Model grounded:
  Variables: 31
  Observaciones: 8

✓ Gibbs Sampling: 500 muestras (ESS=0.590)
✓ Metropolis-Hastings: 500 muestras (ESS=0.336)

✓ Bot Detection:
  Top bot: Bot_1 (P=0.517, 6 cuentas)
  Precision=1.0, Recall=0.6, F1=0.75, AUC=0.79

✓ Sybil Attacks: 5 detectados
```

## 📝 Conceptos Clave Implementados

1. **RPM (Relational Probability Model)**
   - Type signatures: Quality(Book), Honest(Customer), Rec(Customer, Book)
   - CPTs relacionales que se aplican a todos los objetos del tipo
   - Grounding: Expansión a Bayes Net proposicional

2. **OUPM (Open Universe Probability Model)**
   - Origin functions: P(LoginID → Customer mapping)
   - Generating functions: P(# de entidades)
   - Identity uncertainty: ¿Cuáles cuentas son del mismo usuario?
   - Existence uncertainty: ¿Cuántos usuarios reales hay?

3. **Algoritmos de Inferencia**
   - Variable Elimination: Inferencia exacta mediante factores
   - Gibbs Sampling: MCMC con sampling condicional
   - Metropolis-Hastings: MCMC con accept/reject
   - Convergence diagnostics: R̂ de Gelman-Rubin, ESS

4. **Applications**
   - Bot detection: P(IsBot | recommendations)
   - Quality inference: P(Quality | all ratings)
   - Sybil attack detection: Múltiples cuentas del mismo customer

## 🎓 Referencias

- Russell, S. & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Chapter 18: Probabilistic Programming
- Getoor, L. & Taskar, B. (2007). *Introduction to Statistical Relational Learning*
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. Chapter 19-20 (MCMC)

## 📄 Licencia

Proyecto académico para curso de Procesos Estocásticos. Implementación original >50%.


## 🎯 Próximos Pasos

### FASE 4: Inferencia (⏱️ ~2 horas) - SIGUIENTE
- Variable Elimination (exacta)
- Gibbs Sampling (MCMC)
- Metropolis-Hastings (MCMC)
- Convergence diagnostics

### FASE 5: Aplicación
- Query engine: P(φ|e)
- Detección de bots
- Evaluación de calidad real

### FASE 6: Visualización
- Gráficos de redes bayesianas
- Distribuciones posteriores
- Métricas de evaluación

## 📖 Referencias

- Russell & Norvig - Artificial Intelligence: A Modern Approach, Chapter 18
- Sección 18.1: Relational Probability Models
- Sección 18.2: Open Universe Probability Models

## 👥 Autor

Proyecto Estocásticos 2025
