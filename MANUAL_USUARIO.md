# Manual de Usuario - Sistema de Detección de Bots

## Tabla de Contenidos
1. [Introducción](#introducción)
2. [Instalación](#instalación)
3. [Inicio Rápido](#inicio-rápido)
4. [Escenarios Experimentales](#escenarios-experimentales)
5. [Interpretación de Resultados](#interpretación-de-resultados)
6. [Uso Avanzado](#uso-avanzado)
7. [Preguntas Frecuentes](#preguntas-frecuentes)

---

## Introducción

### ¿Qué es este sistema?

Sistema de **programación probabilística** para detectar bots y ataques sybil en sistemas de recomendación. Implementa modelos del Capítulo 18 de "Artificial Intelligence: A Modern Approach" (Russell & Norvig).

### Conceptos implementados

| Concepto | Descripción |
|----------|-------------|
| **RPM** | Relational Probability Model |
| **OUPM** | Open Universe Probability Model |
| **Gibbs Sampling** | MCMC con muestreo condicional |
| **Metropolis-Hastings** | MCMC con accept/reject |
| **Sybil Attack** | Detección de múltiples cuentas |

---

## Instalación

```powershell
# 1. Clonar repositorio
git clone https://github.com/dsolisav/ProyectoEstocasticos.git
cd ProyectoEstocasticos

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Verificar
python -c "import numpy, matplotlib; print('OK')"
```

---

## Inicio Rápido

### Demo Principal

```powershell
python main.py
```

Ejecuta el pipeline completo en ~2-3 minutos.

---

## Escenarios Experimentales

El proyecto incluye **3 escenarios reproducibles** en la carpeta `examples/`:

| # | Escenario | Archivo | Descripción |
|---|-----------|---------|-------------|
| 1 | Detección de Bots | `escenario_1_deteccion_bots.py` | Distingue bots de usuarios reales |
| 2 | Comparación MCMC | `escenario_2_comparacion_mcmc.py` | Compara Gibbs vs Metropolis-Hastings |
| 3 | Sybil Attacks | `escenario_3_sybil_attacks.py` | Detecta usuarios con múltiples cuentas |

### Escenario 1: Detección de Bots

**Objetivo:** Demostrar que el sistema distingue bots de usuarios reales.

**Ejecutar:**
```powershell
python examples/escenario_1_deteccion_bots.py
```

**Gráficos generados:**
- `esc1_bot_scores.png` - Ranking de probabilidades de ser bot
- `esc1_roc_curve.png` - Curva ROC con AUC
- `esc1_confusion_matrix.png` - Matriz de confusión

**Resultados esperados:**
- Precision >= 0.8
- Recall >= 0.8
- AUC >= 0.9

---

### Escenario 2: Comparación de Algoritmos MCMC

**Objetivo:** Comparar Gibbs Sampling vs Metropolis-Hastings.

**Ejecutar:**
```powershell
python examples/escenario_2_comparacion_mcmc.py
```

**Gráficos generados:**
- `esc2_distribucion_gibbs.png` - Distribución inferida por Gibbs
- `esc2_distribucion_mh.png` - Distribución inferida por MH

**Resultados esperados:**
- Ambos algoritmos convergen a distribuciones similares
- Gibbs generalmente más estable
- Diferencia L1 < 0.15 indica convergencia

---

### Escenario 3: Detección de Sybil Attacks

**Objetivo:** Detectar usuarios con múltiples cuentas fraudulentas.

**Ejecutar:**
```powershell
python examples/escenario_3_sybil_attacks.py
```

**Gráficos generados:**
- `esc3_sybil_attacks.png` - Usuarios con múltiples cuentas
- `esc3_cuentas_por_tipo.png` - Distribución de cuentas (bots vs usuarios)

**Resultados esperados:**
- Bots tienen más cuentas que usuarios reales
- Promedio bots > promedio usuarios

---

## Interpretación de Resultados

### Métricas de Clasificación

| Métrica | Definición | Bueno si |
|---------|------------|----------|
| **Precision** | TP / (TP + FP) | > 0.8 |
| **Recall** | TP / (TP + FN) | > 0.8 |
| **F1-Score** | Media armónica | > 0.8 |
| **AUC** | Área bajo curva ROC | > 0.9 |

### Matriz de Confusión

```
              Predicción
              BOT    USER
Real  BOT  [  TP  |  FN  ]
      USER [  FP  |  TN  ]
```

- **TP:** Bots correctamente detectados
- **FN:** Bots que escaparon
- **FP:** Usuarios marcados incorrectamente como bots
- **TN:** Usuarios correctamente identificados

---

## Uso Avanzado

### Usar tus propios datos

```python
from src.models import Customer, Book, LoginID, Recommendation, EntityType
from src.rpm_model import RPMModel
from src.bot_detection import BotDetector

# Crear datos
customers = [Customer("user1", EntityType.REAL_USER)]
books = [Book("book1", true_quality=4)]
login_ids = [LoginID("login1", "user1")]
recommendations = [Recommendation("login1", "book1", rating=4)]

# Ejecutar
rpm = RPMModel()
grounded = rpm.ground_model(customers, books, recommendations)
detector = BotDetector(rpm)
scores = detector.score_customers(customers, books, recommendations, login_ids)
```

### Ajustar parámetros

```python
# Más muestras = mejor precisión, más tiempo
detector.score_customers(..., num_samples=1000)

# Threshold de detección
detector = BotDetector(rpm, detection_threshold=0.6)  # Menos falsos positivos
detector = BotDetector(rpm, detection_threshold=0.4)  # Detecta más bots
```

---

## Preguntas Frecuentes

### ¿Cuánto tiempo tarda?

| Comando | Tiempo |
|---------|--------|
| `main.py` | 2-3 min |
| Escenario 1 | ~1 min |
| Escenario 2 | ~1 min |
| Escenario 3 | ~1 min |

### ¿Por qué Gibbs y MH dan resultados diferentes?

Con pocas muestras, MH puede no converger completamente. Aumentar `num_samples` mejora la convergencia.

### ¿Qué significa "grounding"?

Convertir un modelo relacional (RPM) a una red bayesiana concreta con variables específicas.

---

## Comandos de Referencia

```powershell
# Demo principal
python main.py

# Escenarios experimentales
python examples/escenario_1_deteccion_bots.py
python examples/escenario_2_comparacion_mcmc.py
python examples/escenario_3_sybil_attacks.py

# Ver gráficos
explorer examples\output
```

---

**Versión:** 2.0.0 | **Fecha:** Noviembre 2025
