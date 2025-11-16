# Manual de Usuario - Sistema de Detección de Bots

## Tabla de Contenidos
1. [Introducción](#introducción)
2. [Instalación](#instalación)
3. [Inicio Rápido](#inicio-rápido)
4. [Uso Básico](#uso-básico)
5. [Uso Avanzado](#uso-avanzado)
6. [Interpretación de Resultados](#interpretación-de-resultados)
7. [Ejemplos de Uso](#ejemplos-de-uso)
8. [Preguntas Frecuentes](#preguntas-frecuentes)

---

## Introducción

### ¿Qué es este sistema?

Este sistema utiliza **programación probabilística** para detectar bots y ataques sybil en sistemas de recomendación. Implementa modelos del Capítulo 18 de "Artificial Intelligence: A Modern Approach" (Russell & Norvig).

### ¿Para qué sirve?

- **Detectar bots** en plataformas de recomendación (como Amazon, Yelp, etc.)
- **Identificar ataques sybil** (usuarios con múltiples cuentas fraudulentas)
- **Inferir calidad de productos** basándose en recomendaciones
- **Evaluar honestidad** de usuarios

### ¿Qué incluye?

- Generación de datos sintéticos para experimentación
- Modelos probabilísticos (RPM y OUPM)
- Algoritmos de inferencia MCMC (Gibbs Sampling y Metropolis-Hastings)
- Detección automática de bots con métricas de evaluación
- Visualizaciones profesionales en PNG

---

## Instalación

### Requisitos Previos

- **Python 3.13** o superior
- **pip** (gestor de paquetes de Python)
- **Git** (opcional, para clonar el repositorio)

### Paso 1: Descargar el Proyecto

**Opción A: Clonar desde GitHub**
```powershell
git clone https://github.com/dsolisav/ProyectoEstocasticos.git
cd ProyectoEstocasticos
```

**Opción B: Descargar ZIP**
1. Ve a https://github.com/dsolisav/ProyectoEstocasticos
2. Click en "Code" → "Download ZIP"
3. Extrae el archivo ZIP
4. Abre terminal en la carpeta extraída

### Paso 2: Instalar Dependencias

```powershell
pip install -r requirements.txt
```

Esto instalará:
- `numpy` - Cálculos numéricos
- `matplotlib` - Generación de gráficos
- `pytest` - Ejecución de tests (opcional)

### Paso 3: Verificar Instalación

```powershell
python -c "import numpy, matplotlib; print('✓ Instalación correcta')"
```

---

## Inicio Rápido

### Opción 1: Demo Rápido (30 segundos)

```powershell
python main.py
```

Demo completo con visualización en consola.  
**Tiempo:** 2-3 minutos

### Opción 2: Demo Interactivo con Gráficos (RECOMENDADO)

```powershell
python demo_paso_a_paso.py
```

Demo paso a paso que genera 8 gráficos PNG profesionales.  
**Tiempo:** 5 minutos

### Opción 3: Ejemplos Específicos (PARA APRENDER)

La carpeta `examples/` contiene **4 ejemplos ejecutables** para aprender paso a paso:

```powershell
# Ejemplo 1: Uso básico completo (~1 min)
python examples/ejemplo_1_basico.py

# Ejemplo 2: Comparar algoritmos de inferencia (~1 min)
python examples/ejemplo_2_inferencia.py

# Ejemplo 3: Generar todas las visualizaciones (~2 min)
python examples/ejemplo_3_visualizacion.py

# Ejemplo 4: Experimentar con configuraciones (~4 min)
python examples/ejemplo_4_experimento.py
```

Ver **Sección 7** para descripción detallada de cada ejemplo.

---

## Uso Básico

> **💡 TIP:** Si es tu primera vez, ejecuta `python examples/ejemplo_1_basico.py` para ver el pipeline completo en acción.

### 1. Generar Datos Sintéticos

```python
from src.data_generator import DataGenerator, DatasetConfig

# Configurar parámetros
config = DatasetConfig(
    num_real_users=10,      # Usuarios legítimos
    num_bots=5,             # Bots
    num_books=6,            # Productos/libros
    prob_bot_multiple_accounts=0.7,  # Probabilidad de bot con múltiples cuentas
    random_seed=42          # Semilla para reproducibilidad
)

# Generar dataset
generator = DataGenerator(config)
customers, books, login_ids, recommendations = generator.generate_dataset()

print(f"Generados: {len(customers)} customers, {len(recommendations)} ratings")
```

**Resultado esperado:**
```
Generados: 15 customers, 85 ratings
```

**📝 Ejemplo ejecutable:** `examples/ejemplo_1_basico.py` (Pasos 1-2)

### 2. Construir Modelo Probabilístico

```python
from src.rpm_model import RPMModel

# Crear modelo RPM
rpm = RPMModel()

# Convertir a red bayesiana (grounding)
grounded_network = rpm.ground_model(customers, books, recommendations)

print(f"Red bayesiana con {len(grounded_network.variables)} variables")
```

**📝 Ejemplo ejecutable:** `examples/ejemplo_1_basico.py` (Paso 2)

### 3. Ejecutar Inferencia

```python
from src.inference.gibbs_sampling import GibbsSampling

# Crear algoritmo de inferencia
gibbs = GibbsSampling(grounded_network)

# Generar muestras
samples = gibbs.sample(
    evidence={},           # Sin evidencia adicional
    num_samples=500,       # Número de muestras
    burn_in=100           # Muestras iniciales a descartar
)

print(f"✓ {len(samples)} muestras generadas")
```

**📝 Ejemplo ejecutable:** `examples/ejemplo_2_inferencia.py` (comparación de algoritmos)

### 4. Detectar Bots

```python
from src.bot_detection import BotDetector

# Crear detector
detector = BotDetector(rpm, detection_threshold=0.5)

# Calcular scores de bot para cada customer
bot_scores = detector.score_customers(
    customers,
    books,
    recommendations,
    login_ids,
    num_samples=300
)

# Ver top 5 posibles bots
sorted_scores = sorted(bot_scores, key=lambda x: x.bot_probability, reverse=True)
for score in sorted_scores[:5]:
    print(f"{score.customer_id}: P(bot)={score.bot_probability:.3f}")
```

### 5. Evaluar Resultados

```python
# Calcular métricas de evaluación
metrics = detector.evaluate(bot_scores)

print(f"Precision: {metrics.precision:.3f}")
print(f"Recall: {metrics.recall:.3f}")
print(f"F1-Score: {metrics.f1_score:.3f}")
```

**📝 Ejemplo ejecutable:** `examples/ejemplo_1_basico.py` (Pasos 3-5)

---

## Uso Avanzado

> **💡 TIP:** Ver `examples/ejemplo_2_inferencia.py` para comparación de algoritmos y `examples/ejemplo_3_visualizacion.py` para gráficos personalizados.

### Personalizar Configuración del Dataset

```python
config = DatasetConfig(
    num_real_users=20,               # Más usuarios reales
    num_bots=10,                     # Más bots
    num_books=15,                    # Más productos
    prob_bot_multiple_accounts=0.9,  # Bots más agresivos
    max_accounts_per_bot=10,         # Hasta 10 cuentas por bot
    max_recommendations_per_account=8,  # Más recomendaciones
    random_seed=123                  # Diferente semilla
)
```

### Usar Diferentes Algoritmos de Inferencia

#### Metropolis-Hastings

```python
from src.inference.metropolis_hastings import MetropolisHastings

mh = MetropolisHastings(grounded_network)
samples = mh.sample(
    evidence={},
    num_samples=500,
    burn_in=100,
    proposal='gibbs_style'  # Tipo de propuesta
)
```

#### Variable Elimination (Inferencia Exacta)

```python
from src.inference.variable_elimination import VariableElimination

ve = VariableElimination(grounded_network)

# Consulta específica
result = ve.query(
    query_var="Quality_Book_1",
    evidence={"Honest_Customer_1": True}
)
```

### Query Engine - Consultas Flexibles

```python
from src.query_engine import QueryEngine

query_engine = QueryEngine(grounded_network)

# Consultar calidad de un libro
quality_dist = query_engine.query_marginal(
    variable="Quality_Book_1",
    evidence={},
    method='gibbs',  # o 'mh' para Metropolis-Hastings
    num_samples=500
)

print("Distribución de calidad:")
for value, prob in quality_dist.distribution.items():
    print(f"  Quality={value}: {prob:.3f}")
```

### Detectar Sybil Attacks

```python
# Detectar customers con múltiples cuentas
sybil_attacks = detector.detect_sybil_attacks(
    login_ids,
    min_accounts=2  # Mínimo de cuentas para considerar sybil
)

print(f"Detectados {len(sybil_attacks)} posibles sybil attacks:")
for customer_id, accounts in sybil_attacks.items():
    print(f"  {customer_id}: {len(accounts)} cuentas")
```

### Generar Visualizaciones Personalizadas

```python
from src.visualization_plots import (
    NetworkPlotter,
    DistributionPlotter,
    ROCPlotter,
    BotDetectionPlotter
)

# Gráfico de estructura de red
network_plotter = NetworkPlotter()
network_plotter.plot_network_structure(
    grounded_network,
    "mi_grafico_red.png"
)

# Gráfico de distribución
dist_plotter = DistributionPlotter()
dist_plotter.plot_distribution(
    quality_dist.distribution,
    title="Calidad Inferida",
    xlabel="Nivel de Calidad",
    output_path="mi_distribucion.png"
)

# Gráfico de bot scores
bot_plotter = BotDetectionPlotter()
bot_plotter.plot_bot_scores(bot_scores, "mi_bot_scores.png")
```

---

## Interpretación de Resultados

### Métricas de Clasificación

#### **Precision (Precisión)**
- **Definición:** De todos los clasificados como bots, ¿cuántos realmente lo son?
- **Fórmula:** TP / (TP + FP)
- **Interpretación:**
  - `>0.8`: Excelente - Pocas falsas alarmas
  - `0.6-0.8`: Bueno - Algunas falsas alarmas
  - `<0.6`: Pobre - Muchos falsos positivos

#### **Recall (Sensibilidad)**
- **Definición:** De todos los bots reales, ¿cuántos detectamos?
- **Fórmula:** TP / (TP + FN)
- **Interpretación:**
  - `>0.8`: Excelente - Detecta casi todos los bots
  - `0.5-0.8`: Bueno - Detecta la mayoría
  - `<0.5`: Pobre - Muchos bots escapan

#### **F1-Score**
- **Definición:** Media armónica de Precision y Recall
- **Fórmula:** 2 × (Precision × Recall) / (Precision + Recall)
- **Interpretación:**
  - `>0.8`: Excelente balance
  - `0.6-0.8`: Buen balance
  - `<0.6`: Balance pobre

#### **AUC (Area Under Curve)**
- **Definición:** Área bajo la curva ROC
- **Interpretación:**
  - `>0.9`: **EXCELENTE** - Discriminación casi perfecta
  - `0.8-0.9`: **BUENA** - Discriminación confiable
  - `0.7-0.8`: **ACEPTABLE** - Discriminación moderada
  - `<0.7`: **POBRE** - Discriminación débil
  - `0.5`: Aleatorio (sin capacidad de discriminación)

### Gráficos Generados

#### 1. **Estructura de Red** (`01_network_structure.png`)
- Muestra cuántas variables de cada tipo hay
- Útil para entender la complejidad del modelo

#### 2. **Convergencia MCMC** (`02_mcmc_convergence.png`)
- **Trace plots:** Evolución de las muestras
- **Histogramas:** Distribución posterior inferida
- **Running mean:** Convergencia de la media

**¿Cómo saber si convergió?**
- Running mean se estabiliza (horizontal)
- Trace plot "explora" todo el espacio sin tendencias
- Ambos algoritmos (Gibbs/MH) coinciden

#### 3. **Distribución de Calidad** (`03_quality_distribution.png`)
- Probabilidad inferida de cada nivel de calidad
- Barras más altas = más probable
- Útil para entender qué calidad tiene cada producto

#### 4. **Comparación de Libros** (`04_quality_comparison.png`)
- Compara distribuciones de calidad de múltiples productos
- Permite identificar productos de mejor/peor calidad

#### 5. **Bot Scores** (`05_bot_scores.png`)
- **Top panel:** Ranking de todos los customers
  - Barra roja = Bot real
  - Barra verde = Usuario real
  - Línea naranja = Threshold de decisión
- **Bottom panel:** Distribución de scores por tipo
  - Separación clara = Buen modelo

#### 6. **Sybil Attacks** (`06_sybil_attacks.png`)
- Muestra customers con múltiples cuentas
- Barras más largas = Más cuentas
- Rojo = Bot, Naranja = Usuario real sospechoso

#### 7. **Curva ROC** (`07_roc_curve.png`)
- Curva azul = Performance del clasificador
- Línea roja punteada = Clasificador aleatorio
- AUC mostrado en el gráfico

#### 8. **Matriz de Confusión** (`08_confusion_matrix.png`)
- **Izquierda:** Matriz 2×2 con conteos
  - TP (arriba-izquierda): Bots correctamente detectados
  - TN (abajo-derecha): Usuarios correctamente identificados
  - FP (arriba-derecha): Usuarios marcados como bots
  - FN (abajo-izquierda): Bots que escaparon
- **Derecha:** Métricas visuales con barras de color

---

## Ejemplos de Uso

Esta sección describe los **4 ejemplos ejecutables** incluidos en la carpeta `examples/`.  
Estos scripts demuestran diferentes aspectos del sistema de forma práctica.

---

### 📌 Ejemplo 1: Uso Básico - Pipeline Completo

**Archivo:** `examples/ejemplo_1_basico.py`  
**Tiempo:** ~1 minuto  
**Nivel:** Principiante

#### ¿Qué hace?

Ejecuta el pipeline completo paso a paso:
1. Genera datos sintéticos (15 customers, ~85 recommendations)
2. Construye modelo probabilístico RPM
3. Detecta bots usando MCMC
4. Evalúa métricas (Precision, Recall, F1, Accuracy)
5. Detecta sybil attacks (múltiples cuentas)

#### Cómo ejecutar:

```powershell
python examples/ejemplo_1_basico.py
```

#### Salida esperada:

```
[PASO 1] Generando datos sintéticos...
✓ Generados:
  - 15 customers
  - 6 libros
  - 23 cuentas (loginIDs)
  - 87 recomendaciones

[PASO 2] Construyendo modelo probabilístico (RPM)...
✓ Red bayesiana construida:
  - 29 variables
  - 6 observaciones

[PASO 3] Detectando bots...
✓ Scores calculados para 15 customers

[PASO 4] Top 5 posibles bots:
  1. Bot_4          | P(bot)=0.547 | Real=🤖 BOT    | Predicción=BOT  | Cuentas=5
  2. Bot_1          | P(bot)=0.532 | Real=🤖 BOT    | Predicción=BOT  | Cuentas=4
  3. User_7         | P(bot)=0.489 | Real=👤 USER   | Predicción=USER | Cuentas=3
  ...

[PASO 5] Métricas de evaluación:
  Precision:  0.750
  Recall:     0.600
  F1-Score:   0.667
  Accuracy:   0.800

[PASO 6] Detectando sybil attacks...
✓ Detectados 5 customers con múltiples cuentas
```

#### Para qué sirve:

- ✅ Primera toma de contacto con el sistema
- ✅ Verificar que todo funciona correctamente
- ✅ Entender el flujo básico de detección

---

### 📌 Ejemplo 2: Comparación de Algoritmos de Inferencia

**Archivo:** `examples/ejemplo_2_inferencia.py`  
**Tiempo:** ~1 minuto  
**Nivel:** Intermedio

#### ¿Qué hace?

Compara Gibbs Sampling vs Metropolis-Hastings:
- Tiempo de ejecución de cada algoritmo
- Calidad de las distribuciones inferidas
- Convergencia y similitud de resultados
- Inferencia para múltiples libros

#### Cómo ejecutar:

```powershell
python examples/ejemplo_2_inferencia.py
```

#### Salida esperada:

```
[PASO 3] Consultando: P(Quality_Book_1 | Evidence)
  Calidad real: 4

  → Ejecutando Gibbs Sampling...
    Tiempo: 2.35s
    Distribución inferida:
      Quality=1:  0.067
      Quality=2: █ 0.113
      Quality=3: ████████████████ 0.320 ← TRUE
      Quality=4: ████████████████ 0.327
      Quality=5: ████████ 0.173

  → Ejecutando Metropolis-Hastings...
    Tiempo: 2.89s
    Distribución inferida:
      Quality=1:  0.080
      Quality=2: █ 0.107
      Quality=3: ███████████████ 0.307
      Quality=4: ████████████████ 0.333 ← TRUE
      Quality=5: ████████ 0.173

[PASO 4] Comparación de resultados:
  Tiempo Gibbs: 2.35s
  Tiempo MH:    2.89s
  Speedup:      1.23x

  Diferencia entre distribuciones: 0.0267
  ✓ Ambos algoritmos CONVERGEN a resultados similares
```

#### Para qué sirve:

- ✅ Entender diferencias entre algoritmos MCMC
- ✅ Decidir qué algoritmo usar según el caso
- ✅ Validar convergencia del modelo

---

### 📌 Ejemplo 3: Generación de Visualizaciones

**Archivo:** `examples/ejemplo_3_visualizacion.py`  
**Tiempo:** ~2 minutos  
**Nivel:** Intermedio

#### ¿Qué hace?

Genera todos los gráficos profesionales del sistema:
1. Estructura de red bayesiana
2. Distribución de calidad de un libro
3. Comparación de calidades (múltiples libros)
4. Bot scores con ranking
5. Sybil attacks detectados
6. Curva ROC con AUC
7. Matriz de confusión con métricas

#### Cómo ejecutar:

```powershell
python examples/ejemplo_3_visualizacion.py
```

#### Salida esperada:

```
[PASO 3] Generando gráfico de estructura de red...
✓ Guardado: examples/output/01_network_structure.png

[PASO 4] Generando gráfico de distribución de calidad...
✓ Guardado: examples/output/02_quality_distribution.png

[PASO 5] Generando gráfico de comparación de libros...
✓ Guardado: examples/output/03_quality_comparison.png

[PASO 6] Detectando bots y generando gráfico...
✓ Guardado: examples/output/04_bot_scores.png

[PASO 7] Generando gráfico de sybil attacks...
✓ Guardado: examples/output/05_sybil_attacks.png

[PASO 8] Generando curva ROC...
✓ Guardado: examples/output/06_roc_curve.png (AUC=0.8932)

[PASO 9] Generando matriz de confusión...
✓ Guardado: examples/output/07_confusion_matrix.png

✓ TODOS LOS GRÁFICOS GENERADOS

Archivos creados en: examples/output

Métricas del modelo:
  Precision: 0.714
  Recall:    0.833
  F1-Score:  0.769
  AUC:       0.893
```

#### Archivos generados:

```
examples/output/
├── 01_network_structure.png      (Estructura de la red)
├── 02_quality_distribution.png   (Distribución de probabilidad)
├── 03_quality_comparison.png     (Comparación de libros)
├── 04_bot_scores.png             (Ranking de bot scores)
├── 05_sybil_attacks.png          (Sybil attacks detectados)
├── 06_roc_curve.png              (Curva ROC con AUC)
└── 07_confusion_matrix.png       (Matriz de confusión)
```

#### Para qué sirve:

- ✅ Generar gráficos para reportes/presentaciones
- ✅ Analizar visualmente el comportamiento del modelo
- ✅ Compartir resultados de forma profesional

---

### 📌 Ejemplo 4: Experimentación - Efecto del Número de Muestras

**Archivo:** `examples/ejemplo_4_experimento.py`  
**Tiempo:** ~3-4 minutos  
**Nivel:** Avanzado

#### ¿Qué hace?

Experimenta con diferentes configuraciones:
- Prueba con 50, 100, 200 y 400 muestras
- Mide tiempo de ejecución y métricas para cada caso
- Calcula el trade-off tiempo vs precisión
- Recomienda configuración óptima

#### Cómo ejecutar:

```powershell
python examples/ejemplo_4_experimento.py
```

#### Salida esperada:

```
[PASO 3] Experimentando con diferentes números de muestras:

  Samples | Tiempo | Precision | Recall | F1-Score | Accuracy
  -----------------------------------------------------------------
    50    |   8.2s |   0.667   | 0.667  |  0.667   |  0.800
   100    |  15.7s |   0.714   | 0.833  |  0.769   |  0.867
   200    |  31.2s |   0.750   | 0.750  |  0.750   |  0.867
   400    |  62.5s |   0.800   | 0.667  |  0.727   |  0.867

[PASO 4] Análisis de resultados:

  Mejor F1-Score: 0.769 con 100 muestras
  Mejor eficiencia: 100 muestras (F1=0.769 en 15.7s)

  De 50 a 400 muestras:
    F1-Score mejora:    +9.0%
    Tiempo incrementa:  +662.2%

[PASO 5] Recomendación:
  ✓ Para uso rápido: 100 muestras
    (F1=0.769, tiempo=15.7s)
  ✓ Para mejor precisión: 200 muestras
    (F1=0.750, tiempo=31.2s)

Conclusión:
  - Más muestras generalmente mejoran la precisión
  - Existe un punto de rendimientos decrecientes
  - Para producción, considera el trade-off tiempo/precisión
```

#### Para qué sirve:

- ✅ Optimizar configuración para tu caso de uso
- ✅ Entender trade-offs del sistema
- ✅ Decidir parámetros para producción

---

### 🚀 Ejecutar Todos los Ejemplos

Para ejecutar todos los ejemplos en secuencia:

```powershell
cd "c:\Users\dsoli\OneDrive\Desktop\proyecto estocasticos"
python examples/ejemplo_1_basico.py
python examples/ejemplo_2_inferencia.py
python examples/ejemplo_3_visualizacion.py
python examples/ejemplo_4_experimento.py
```

**Tiempo total:** ~8 minutos

### 💡 Snippets de Código Útiles

Estos son fragmentos de código que puedes copiar y adaptar:

#### Snippet 1: Usar tus propios datos

```python
"""
Usar el sistema con tus propios datos
"""
from src.models import Customer, Book, LoginID, Recommendation, EntityType

# Crear tus propios datos
customers = [
    Customer("user1", EntityType.REAL_USER),
    Customer("user2", EntityType.REAL_USER),
    Customer("bot1", EntityType.BOT),
]

books = [
    Book("book1", true_quality=4),
    Book("book2", true_quality=2),
]

login_ids = [
    LoginID("login1", "user1"),
    LoginID("login2", "user2"),
    LoginID("login3", "bot1"),
]

recommendations = [
    Recommendation("login1", "book1", rating=5),
    Recommendation("login2", "book1", rating=3),
    Recommendation("login3", "book1", rating=5),
]

# Continuar con el pipeline normal
from src.rpm_model import RPMModel
rpm = RPMModel()
grounded = rpm.ground_model(customers, books, recommendations)
# ... resto del código
```

#### Snippet 2: Batch Processing

```python
"""
Procesar múltiples datasets y guardar resultados
"""
import json
from src.data_generator import DataGenerator, DatasetConfig
from src.rpm_model import RPMModel
from src.bot_detection import BotDetector

results = []

for seed in range(10):  # 10 experimentos
    config = DatasetConfig(
        num_real_users=15, 
        num_bots=8, 
        num_books=10,
        random_seed=seed
    )
    
    # Ejecutar pipeline completo
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    rpm = RPMModel()
    grounded = rpm.ground_model(customers, books, recommendations)
    
    detector = BotDetector(rpm)
    bot_scores = detector.score_customers(
        customers, books, recommendations, login_ids, num_samples=200
    )
    
    metrics = detector.evaluate(bot_scores)
    
    results.append({
        'seed': seed,
        'precision': metrics.precision,
        'recall': metrics.recall,
        'f1': metrics.f1_score
    })

# Guardar resultados
with open('experiment_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Calcular promedio
avg_precision = sum(r['precision'] for r in results) / len(results)
print(f"Precision promedio: {avg_precision:.3f}")
```

---

## Preguntas Frecuentes

### ¿Cuánto tiempo tarda el sistema?

- **Demo básico (`main.py`):** 2-3 minutos
- **Demo con gráficos:** 5 minutos
- **Detección en dataset pequeño (20 users):** ~30 segundos
- **Detección en dataset grande (100 users):** ~5 minutos

El tiempo depende principalmente del número de muestras MCMC.

### ¿Cómo mejorar la precisión?

1. **Aumentar número de muestras:** `num_samples=1000` en vez de 500
2. **Ajustar threshold:** `detection_threshold=0.6` si tienes muchos falsos positivos
3. **Más datos de entrenamiento:** Más recommendations por customer
4. **Ajustar priors:** Modificar probabilidades en CPTs

### ¿Qué hacer si MCMC no converge?

Señales de no convergencia:
- Running mean no se estabiliza
- Trace plot muestra tendencias
- Resultados muy diferentes entre ejecuciones

Soluciones:
1. **Aumentar burn_in:** `burn_in=200` en vez de 100
2. **Más muestras:** `num_samples=1000`
3. **Verificar modelo:** Asegúrate de que las CPTs sumen 1.0

### ¿Cómo usar mis propios datos?

Debes convertir tus datos al formato del sistema:

```python
# Formato requerido:
customers = [Customer(id, entity_type), ...]
books = [Book(id, true_quality), ...]  # true_quality opcional
login_ids = [LoginID(id, customer_id), ...]
recommendations = [Recommendation(login_id, book_id, rating), ...]
```

Ver **Ejemplo 3** arriba para más detalles.

### ¿Qué significa "grounding"?

**Grounding** es convertir un modelo relacional (RPM) en una red bayesiana concreta:
- RPM: "Honest(customer)" (genérico)
- Grounded: "Honest_Customer_1", "Honest_Customer_2", ... (específico)

Es necesario para ejecutar inferencia.

### ¿Por qué usar dos algoritmos MCMC?

- **Gibbs Sampling:** Más rápido, generalmente converge mejor
- **Metropolis-Hastings:** Más flexible, útil para distribuciones complejas

El demo los compara para fines educativos.

### ¿Cómo cambiar el threshold de detección?

```python
detector = BotDetector(rpm, detection_threshold=0.6)  # Default: 0.5
```

- **Threshold alto (0.7):** Menos falsos positivos, más bots escapan
- **Threshold bajo (0.3):** Detecta más bots, más falsas alarmas

### ¿Los gráficos se regeneran cada vez?

Sí, cada ejecución de `demo_paso_a_paso.py` sobrescribe los gráficos en `output/`. 
Si quieres conservarlos, cópialos a otra carpeta o cambia el nombre del archivo de salida:

```python
network_plotter.plot_network_structure(grounded, "output/red_v2.png")
```

### ¿Puedo usar esto en producción?

Este es un **sistema educativo/experimental**. Para producción necesitarías:
- Optimización de performance
- Validación con datos reales
- Manejo de errores robusto
- API/interfaz de usuario
- Logging y monitoreo

### ¿Qué hacer si encuentro un error?

1. **Verificar instalación:**
   ```powershell
   python -c "import numpy, matplotlib; print('OK')"
   ```

2. **Ejecutar tests:**
   ```powershell
   python -m pytest tests/ -v
   ```

3. **Revisar logs:** El sistema imprime información de debug

4. **Reportar issue:** https://github.com/dsolisav/ProyectoEstocasticos/issues

---

## Comandos Útiles de Referencia

```powershell
# Instalación
pip install -r requirements.txt

# Demos principales
python main.py                      # Demo rápido (2-3 min)
python demo_paso_a_paso.py          # Demo interactivo con gráficos (5 min)

# Ejemplos paso a paso (RECOMENDADO para aprender)
python examples/ejemplo_1_basico.py          # Uso básico (~1 min)
python examples/ejemplo_2_inferencia.py      # Comparar algoritmos (~1 min)
python examples/ejemplo_3_visualizacion.py   # Generar gráficos (~2 min)
python examples/ejemplo_4_experimento.py     # Experimentación (~4 min)

# Tests
python -m pytest tests/ -v                   # Todos los tests
python -m pytest tests/test_inference.py -v  # Test específico

# Limpieza
Remove-Item -Recurse -Force __pycache__, .pytest_cache

# Ver gráficos generados
explorer output              # Demo principal
explorer examples/output     # Ejemplos

# Verificar estructura del proyecto
tree /F /A
```

---

## Recursos Adicionales

### Documentación del Proyecto

- **README.md** - Documentación técnica completa del proyecto
- **MANUAL_USUARIO.md** - Esta guía de usuario (estás aquí)
- **GUIA_VERIFICACION.md** - Comandos paso a paso para verificar funcionamiento
- **examples/README.md** - Descripción detallada de cada ejemplo

### Referencias Teóricas

- **capitulo18.pdf** - Capítulo 18 de "AI: A Modern Approach" (Russell & Norvig)
  - Sección 18.1: Relational Probability Models (RPM)
  - Sección 18.2: Open Universe Probability Models (OUPM)
  - Sección 18.3: Inference en modelos probabilísticos

### Repositorio GitHub

- **URL:** https://github.com/dsolisav/ProyectoEstocasticos
- **Issues:** Para reportar bugs o solicitar features
- **Tests:** Carpeta `tests/` con 30 tests unitarios

### Código de Ejemplo

- **examples/** - 4 ejemplos ejecutables listos para usar
- **main.py** - Demo completo con visualización ASCII
- **demo_paso_a_paso.py** - Demo interactivo con 8 gráficos PNG

---

**Fecha:** Noviembre 2025  
**Versión:** 1.0.0  
**Autor:** ProyectoEstocasticos

---

## Índice de Ejemplos Ejecutables

| Ejemplo | Archivo | Tiempo | Descripción |
|---------|---------|--------|-------------|
| 1 | `ejemplo_1_basico.py` | ~1 min | Pipeline completo paso a paso |
| 2 | `ejemplo_2_inferencia.py` | ~1 min | Comparación Gibbs vs MH |
| 3 | `ejemplo_3_visualizacion.py` | ~2 min | Generación de 7 gráficos PNG |
| 4 | `ejemplo_4_experimento.py` | ~4 min | Experimentación con configuraciones |

**Total:** ~8 minutos para todos los ejemplos
