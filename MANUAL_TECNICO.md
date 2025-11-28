# Manual Técnico - Sistema de Detección de Bots

## Tabla de Contenidos
1. [Arquitectura del Sistema](#1-arquitectura-del-sistema)
2. [Modelo de Datos](#2-modelo-de-datos)
3. [Modelos Probabilísticos](#3-modelos-probabilísticos)
4. [Algoritmos de Inferencia](#4-algoritmos-de-inferencia)
5. [Estructura del Código](#5-estructura-del-código)
6. [Flujo de Ejecución](#6-flujo-de-ejecución)
7. [Extensibilidad](#7-extensibilidad)

---

## 1. Arquitectura del Sistema

### 1.1 Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                                  │
│                    (Punto de entrada)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      src/ (Módulos Core)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   models.py  │  │data_generator│  │  visualization*.py   │   │
│  │  (Entidades) │  │   .py        │  │    (Gráficos)        │   │
│  └──────┬───────┘  └──────────────┘  └──────────────────────┘   │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  rpm_model.py│◄─│   cpt.py     │  │   oupm_model.py      │   │
│  │    (RPM)     │  │   (CPTs)     │  │      (OUPM)          │   │
│  └──────┬───────┘  └──────────────┘  └──────────────────────┘   │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                                │
│  │ grounding.py │ ──► GroundedBayesNet                          │
│  └──────┬───────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐  ┌────────────────────────────────────────┐   │
│  │query_engine  │◄─│         inference/                      │   │
│  │    .py       │  │  ┌─────────────┐  ┌──────────────────┐ │   │
│  └──────┬───────┘  │  │gibbs_sampling│  │metropolis_hastings│ │   │
│         │          │  │    .py       │  │      .py         │ │   │
│         ▼          │  └─────────────┘  └──────────────────┘ │   │
│  ┌──────────────┐  └────────────────────────────────────────┘   │
│  │bot_detection │                                                │
│  │    .py       │                                                │
│  └──────────────┘                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Dependencias entre Módulos

| Módulo | Depende de |
|--------|------------|
| `models.py` | (ninguno) |
| `cpt.py` | (ninguno) |
| `origin_functions.py` | `models.py` |
| `grounding.py` | `models.py`, `cpt.py` |
| `rpm_model.py` | `models.py`, `cpt.py`, `grounding.py` |
| `oupm_model.py` | `models.py`, `cpt.py`, `origin_functions.py` |
| `query_engine.py` | `grounding.py`, `inference/*` |
| `bot_detection.py` | `models.py`, `rpm_model.py`, `query_engine.py` |
| `data_generator.py` | `models.py` |

---

## 2. Modelo de Datos

### 2.1 Entidades Principales

```
┌─────────────────┐       ┌─────────────────┐
│    Customer     │       │      Book       │
├─────────────────┤       ├─────────────────┤
│ customer_id: str│       │ book_id: str    │
│ entity_type:    │       │ true_quality:int│
│   EntityType    │       │ title: str?     │
│ honesty: float  │       │ genre: str?     │
└────────┬────────┘       └────────┬────────┘
         │                         │
         │ 1:N                     │
         ▼                         │
┌─────────────────┐                │
│    LoginID      │                │
├─────────────────┤                │
│ login_id: str   │                │
│ origin: Customer│                │
│ recommendations │                │
└────────┬────────┘                │
         │                         │
         │ 1:N                     │ N:1
         ▼                         ▼
    ┌─────────────────────────────────┐
    │        Recommendation           │
    ├─────────────────────────────────┤
    │ login_id: str                   │
    │ book_id: str                    │
    │ rating: int (1-5)               │
    │ timestamp: float?               │
    └─────────────────────────────────┘
```

### 2.2 Enumeraciones

```python
class EntityType(Enum):
    REAL_USER = "real_user"  # Usuario legítimo
    BOT = "bot"              # Bot malicioso
    UNKNOWN = "unknown"      # Tipo desconocido
```

### 2.3 Relaciones Clave

| Relación | Cardinalidad | Descripción |
|----------|--------------|-------------|
| Customer → LoginID | 1:N | Un customer puede tener múltiples cuentas (Sybil) |
| LoginID → Recommendation | 1:N | Una cuenta hace múltiples recomendaciones |
| Book → Recommendation | 1:N | Un libro recibe múltiples ratings |

---

## 3. Modelos Probabilísticos

### 3.1 RPM (Relational Probability Model)

El RPM define la estructura probabilística usando **predicados relacionales**:

#### Type Signatures

| Predicado | Tipo | Dominio |
|-----------|------|---------|
| `Quality(Book)` | Book → {1,2,3,4,5} | Calidad real del libro |
| `Honest(Customer)` | Customer → {True, False} | Si el customer es honesto |
| `Recommendation(Customer, Book)` | (Customer, Book) → {1,2,3,4,5} | Rating observado |

#### Estructura de Dependencias

```
Quality(b) ────────┐
                   ├───► Recommendation(c, b)
Honest(c) ─────────┘
```

**Interpretación:** El rating que un customer `c` da a un libro `b` depende de:
1. La calidad real del libro (`Quality(b)`)
2. Si el customer es honesto (`Honest(c)`)

### 3.2 CPTs (Conditional Probability Tables)

#### CPT de Quality (Prior)

```
P(Quality = q) = 1/5 para q ∈ {1,2,3,4,5}  (Uniforme)
```

#### CPT de Honest (Prior)

```
P(Honest = True | EntityType = REAL_USER) = 0.9
P(Honest = True | EntityType = BOT) = 0.1
```

#### CPT de Recommendation

```
P(Rec = r | Quality = q, Honest = True):
  - Alta probabilidad cerca de q (usuario honesto refleja calidad real)
  
P(Rec = r | Quality = q, Honest = False):
  - Distribución sesgada hacia extremos (1 o 5)
  - Bot da ratings manipulados
```

**Tabla simplificada:**

| Honest | Quality | Rating más probable |
|--------|---------|---------------------|
| True | q | r ≈ q (con varianza pequeña) |
| False | cualquier | r ∈ {1, 5} (extremos) |

### 3.3 OUPM (Open Universe Probability Model)

Extiende RPM para manejar:

1. **Identity Uncertainty:** No sabemos qué LoginIDs corresponden al mismo Customer
2. **Existence Uncertainty:** No sabemos cuántos Customers reales existen

#### Origin Functions

```python
O_LoginID: LoginID → Customer
```

Mapea cada cuenta a su customer real. En OUPM, esta función es una variable aleatoria.

### 3.4 Proceso de Grounding

El **grounding** convierte el RPM relacional en una red bayesiana proposicional:

```
RPM (compacto)                    Bayes Net (expandida)
─────────────────                 ────────────────────────
Quality(Book)          ──►        Quality_Book_1
                                  Quality_Book_2
                                  ...
                                  
Honest(Customer)       ──►        Honest_User_1
                                  Honest_Bot_1
                                  ...
                                  
Rec(Customer, Book)    ──►        Rec_LoginID_1_Book_1
                                  Rec_LoginID_1_Book_2
                                  ...
```

**Número de variables grounded:**
- `n_books` variables Quality
- `n_customers` variables Honest
- `n_recommendations` variables Rec

---

## 4. Algoritmos de Inferencia

### 4.1 Gibbs Sampling

**Objetivo:** Aproximar P(X | evidence) mediante muestreo.

#### Pseudocódigo

```
función GIBBS_SAMPLING(red, evidence, n_samples, burn_in):
    # Inicialización
    estado ← asignación_aleatoria(red, evidence)
    muestras ← []
    
    para i de 1 a n_samples + burn_in:
        para cada variable X no en evidence:
            # Calcular distribución condicional
            P(X | MB(X)) ← calcular_condicional(X, estado)
            
            # Muestrear nuevo valor
            estado[X] ← sample(P(X | MB(X)))
        
        si i > burn_in:
            muestras.agregar(copia(estado))
    
    retornar muestras
```

#### Cálculo de Distribución Condicional

Para cada variable X:

$$P(X = x | X_{-i}) \propto P(X = x | Parents(X)) \cdot \prod_{Y \in Children(X)} P(Y | Parents(Y))$$

Donde $X_{-i}$ son todas las variables excepto X.

### 4.2 Metropolis-Hastings

**Objetivo:** MCMC con proposal distribution arbitraria.

#### Pseudocódigo

```
función METROPOLIS_HASTINGS(red, evidence, n_samples, burn_in):
    estado ← asignación_aleatoria(red, evidence)
    muestras ← []
    
    para i de 1 a n_samples + burn_in:
        # Proponer nuevo estado
        propuesta ← flip_variable_aleatoria(estado, evidence)
        
        # Calcular probabilidad de aceptación
        α ← min(1, P(propuesta) / P(estado))
        
        # Accept/Reject
        si random() < α:
            estado ← propuesta
        
        si i > burn_in:
            muestras.agregar(copia(estado))
    
    retornar muestras
```

#### Probabilidad de Estado

$$P(estado) = \prod_{X \in Variables} P(X | Parents(X))$$

### 4.3 Comparación de Algoritmos

| Aspecto | Gibbs Sampling | Metropolis-Hastings |
|---------|----------------|---------------------|
| Proposal | P(Xi \| MB(Xi)) exacta | Flip aleatorio |
| Acceptance | Siempre 1.0 | α = min(1, P(x')/P(x)) |
| Convergencia | Generalmente más rápida | Puede ser lenta |
| Complejidad | O(n × dominios) por muestra | O(1) por propuesta |
| Mejor para | Variables discretas pequeñas | Espacios continuos |

---

## 5. Estructura del Código

### 5.1 Módulos Core

#### `models.py`
Define las entidades de datos:
- `Customer`: Usuario (real o bot)
- `Book`: Producto a calificar
- `LoginID`: Cuenta de usuario
- `Recommendation`: Rating observado
- `PossibleWorld`: Mundo posible en OUPM

#### `cpt.py`
Implementa Conditional Probability Tables:
- `CPT`: Clase base
- `QualityCPT`: Prior sobre calidad de libros
- `HonestyCPT`: Prior sobre honestidad
- `RecommendationCPT`: P(Rec | Quality, Honest)

#### `grounding.py`
Convierte RPM a Bayes Net:
- `BayesNetVariable`: Variable proposicional
- `GroundedBayesNet`: Red completa
- `RPMGrounder`: Realiza el grounding

#### `rpm_model.py`
Modelo RPM completo:
- `TypeSignature`: Tipos de predicados
- `RPMModel`: Modelo relacional

#### `oupm_model.py`
Extensión para universo abierto:
- `OUPMModel`: Maneja identity/existence uncertainty

#### `query_engine.py`
Motor de consultas:
- `QueryResult`: Resultado de inferencia
- `QueryEngine`: Ejecuta queries MCMC

#### `bot_detection.py`
Detector de bots:
- `BotScore`: Score por customer
- `DetectionMetrics`: Precision/Recall/F1
- `BotDetector`: Clasificador principal

### 5.2 Módulos de Inferencia

#### `inference/gibbs_sampling.py`
- `GibbsSample`: Una muestra
- `GibbsSampling`: Algoritmo completo

#### `inference/metropolis_hastings.py`
- `MHSample`: Una muestra con accept/reject
- `MetropolisHastings`: Algoritmo completo

### 5.3 Módulos de Soporte

#### `data_generator.py`
Genera datos sintéticos:
- `DatasetConfig`: Configuración
- `DataGenerator`: Generador

#### `visualization.py` / `visualization_plots.py`
Visualizaciones:
- Curvas ROC
- Matrices de confusión
- Distribuciones

---

## 6. Flujo de Ejecución

### 6.1 Pipeline Principal (`main.py`)

```
1. GENERACIÓN DE DATOS
   DataGenerator.generate()
   └─► customers[], books[], login_ids[], recommendations[]

2. CONSTRUCCIÓN DE MODELO RPM
   RPMModel.ground_model(customers, books, recommendations)
   └─► GroundedBayesNet con ~31 variables

3. CONSTRUCCIÓN DE MODELO OUPM
   OUPMModel(customers, books, recommendations, login_ids)
   └─► Modelo con origin functions

4. INFERENCIA MCMC
   QueryEngine.query_marginal("Quality_Book_1", evidence)
   └─► GibbsSampling.sample() ─► distribución P(Quality | evidence)

5. DETECCIÓN DE BOTS
   BotDetector.score_customers(...)
   └─► Para cada customer:
       a. Contar cuentas (sybil detection)
       b. Analizar patrones de ratings
       c. Calcular P(bot | evidence)
       d. Clasificar según threshold

6. EVALUACIÓN
   BotDetector.evaluate_predictions()
   └─► Precision, Recall, F1, Confusion Matrix

7. VISUALIZACIÓN
   BotDetectionPlotter.save_*()
   └─► Archivos PNG
```

### 6.2 Flujo de Inferencia Detallado

```
query_marginal("Quality_Book_1", evidence={...})
│
├─► Inicializar estado aleatorio
│   └─► estado = {Quality_Book_1: 3, Honest_User_1: True, ...}
│
├─► Para cada iteración (1 a n_samples + burn_in):
│   │
│   ├─► Para cada variable no-evidencia:
│   │   ├─► Calcular P(Xi | resto)
│   │   └─► Muestrear nuevo valor
│   │
│   └─► Si pasó burn_in: guardar muestra
│
└─► Estimar distribución marginal
    └─► Contar frecuencias de Quality_Book_1 en muestras
```

---

## 7. Extensibilidad

### 7.1 Agregar Nuevo Tipo de Entidad

1. **Definir en `models.py`:**
```python
@dataclass
class NewEntity:
    entity_id: str
    attribute: Any
```

2. **Agregar predicado en `rpm_model.py`:**
```python
self.type_signatures["NewPredicate"] = TypeSignature("NewPredicate", ["NewEntity"])
```

3. **Crear CPT en `cpt.py`:**
```python
class NewPredicateCPT(CPT):
    def _build_table(self):
        # Definir probabilidades
```

4. **Actualizar grounding en `grounding.py`**

### 7.2 Modificar CPTs

Las CPTs están en `src/cpt.py`. Para cambiar el comportamiento:

```python
class RecommendationCPT(CPT):
    def _build_table(self):
        # Modificar P(Rec | Quality, Honest)
        for quality in range(1, 6):
            for honest in [True, False]:
                # Ajustar distribución aquí
```

### 7.3 Agregar Nuevo Algoritmo de Inferencia

1. **Crear módulo en `inference/`:**
```python
# inference/nuevo_algoritmo.py
class NuevoAlgoritmo:
    def __init__(self, grounded_network):
        self.network = grounded_network
    
    def sample(self, evidence, num_samples, burn_in):
        # Implementar
        return samples
    
    def estimate_marginal(self, variable, samples):
        # Implementar
        return distribution
```

2. **Integrar en `query_engine.py`:**
```python
from .inference.nuevo_algoritmo import NuevoAlgoritmo

class QueryEngine:
    def __init__(self, grounded_network):
        self.nuevo = NuevoAlgoritmo(grounded_network)
    
    def query_marginal(self, ..., method='nuevo'):
        if method == 'nuevo':
            samples = self.nuevo.sample(...)
```

### 7.4 Agregar Nuevas Métricas de Detección

En `bot_detection.py`:

```python
@dataclass
class DetectionMetrics:
    # Agregar nuevos campos
    nueva_metrica: float

class BotDetector:
    def evaluate_predictions(self, scores):
        # Calcular nueva métrica
        metrics.nueva_metrica = calcular(...)
```

---

## Apéndice A: Fórmulas Matemáticas

### Probabilidad Conjunta de la Red

$$P(X_1, ..., X_n) = \prod_{i=1}^{n} P(X_i | Parents(X_i))$$

### Inferencia Bayesiana

$$P(Query | Evidence) = \frac{P(Query, Evidence)}{P(Evidence)}$$

### Estimación por Muestreo

$$\hat{P}(X = x | E) \approx \frac{\#\{muestras\ donde\ X = x\}}{n_{muestras}}$$

### Convergencia MCMC

El error de estimación decrece como $O(1/\sqrt{n})$ donde $n$ es el número de muestras.

---

## Apéndice B: Configuración por Defecto

| Parámetro | Valor | Ubicación |
|-----------|-------|-----------|
| MCMC samples | 500 | `query_engine.py` |
| Burn-in | 100 | `query_engine.py` |
| Detection threshold | 0.5 | `bot_detection.py` |
| Honesty prior (user) | 0.9 | `cpt.py` |
| Honesty prior (bot) | 0.1 | `cpt.py` |
| Quality prior | Uniforme | `cpt.py` |

---

**Versión:** 2.0.0 | **Fecha:** Noviembre 2025
