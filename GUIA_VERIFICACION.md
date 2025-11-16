# Guía de Verificación - Sistema de Detección de Bots

## Flujo de Uso Paso a Paso con Comandos

Esta guía te permite verificar que todo el sistema funciona correctamente.

---

## ✅ PASO 0: Verificación del Entorno

### Verificar instalación de Python
```powershell
python --version
```
**Resultado esperado:** `Python 3.13.x` o superior

### Verificar librerías requeridas
```powershell
python -c "import numpy; print(f'numpy {numpy.__version__}')"
python -c "import matplotlib; print(f'matplotlib {matplotlib.__version__}')"
```
**Resultado esperado:**
- `numpy 2.x.x`
- `matplotlib 3.10.x`

### Verificar estructura del proyecto
```powershell
Get-ChildItem -Recurse -Filter "*.py" | Select-Object Name, Directory | Format-Table
```
**Resultado esperado:** Lista de todos los archivos `.py` del proyecto

---

## ✅ PASO 1: Ejecutar Tests Unitarios

### Ejecutar todos los tests
```powershell
cd "c:\Users\dsoli\OneDrive\Desktop\proyecto estocasticos"
python -m pytest tests/ -v
```

**Resultado esperado:**
```
tests/test_basics.py::TestPhase1DataGeneration PASSED
tests/test_basics.py::TestPhase2Models PASSED
tests/test_inference.py::TestPhase3VariableElimination PASSED
tests/test_inference.py::TestPhase3GibbsSampling PASSED
tests/test_inference.py::TestPhase3MetropolisHastings PASSED
tests/test_query_engine.py::TestPhase4QueryEngine PASSED
tests/test_oupm.py::TestPhase5OUPMModel PASSED
tests/test_rpm.py::TestPhase6RPMModel PASSED
tests/test_visualization.py::TestVisualization PASSED

======================== 30 passed in X.XXs ========================
```

### Ejecutar tests por fase (opcional)
```powershell
# Solo FASE 1: Generación de datos
python -m pytest tests/test_basics.py::TestPhase1DataGeneration -v

# Solo FASE 3: Inferencia
python -m pytest tests/test_inference.py -v

# Solo FASE 6: Bot Detection
python -m pytest tests/test_rpm.py -v
```

---

## ✅ PASO 2: Demo Interactivo Paso a Paso

### Ejecutar demo completo con visualizaciones
```powershell
python demo_paso_a_paso.py
```

Este comando ejecuta un demo interactivo que:
1. **Genera datos sintéticos** (18 customers, 131 ratings)
2. **Construye modelos** (RPM con 34 variables, OUPM)
3. **Ejecuta inferencia** (Gibbs Sampling + Metropolis-Hastings, 500 muestras cada uno)
4. **Consultas probabilísticas** (Query Engine)
5. **Detección de bots** (Bot scores, métricas de evaluación)
6. **Análisis de clasificación** (Curva ROC, matriz de confusión)

**Pausa interactiva:** El demo se pausa después de cada paso para que puedas revisar los resultados. Presiona **ENTER** para continuar.

**Resultado esperado:**
- 8 gráficos PNG generados en `output/`
- Métricas impresas en consola:
  - Precision: ~0.75
  - Recall: ~0.50
  - F1-Score: ~0.60
  - AUC: ~0.90 (EXCELENTE)

---

## ✅ PASO 3: Demo Rápido Sin Pausa

### Ejecutar demo principal (sin interacción)
```powershell
python main.py
```

Este es el demo original que ejecuta todo de forma continua (sin pausas).

**Resultado esperado:**
- Impresión de todas las fases en consola
- Visualizaciones ASCII en terminal
- Tiempo de ejecución: ~2-3 minutos

---

## ✅ PASO 4: Verificar Archivos Generados

### Listar gráficos generados
```powershell
Get-ChildItem output/*.png | Select-Object Name, Length, LastWriteTime | Format-Table
```

**Resultado esperado:** 8 archivos PNG

### Descripción de los gráficos

| Archivo | Descripción |
|---------|-------------|
| `01_network_structure.png` | Estructura de la red bayesiana (variables por tipo) |
| `02_mcmc_convergence.png` | Análisis de convergencia MCMC (trace plots, distribuciones, running mean) |
| `03_quality_distribution.png` | Distribución de probabilidad de calidad de un libro |
| `04_quality_comparison.png` | Comparación de distribuciones de calidad entre múltiples libros |
| `05_bot_scores.png` | Ranking de bot scores para todos los customers |
| `06_sybil_attacks.png` | Gráfico de barras de sybil attacks (customers con múltiples cuentas) |
| `07_roc_curve.png` | Curva ROC con AUC para evaluación del clasificador |
| `08_confusion_matrix.png` | Matriz de confusión con métricas de clasificación |

### Abrir un gráfico (ejemplo)
```powershell
Invoke-Item output/07_roc_curve.png
```

### Ver tamaño de todos los gráficos
```powershell
Get-ChildItem output/*.png | Measure-Object -Property Length -Sum | Select-Object Count, @{Name="TotalMB";Expression={[math]::Round($_.Sum/1MB,2)}}
```

---

## ✅ PASO 5: Análisis de Resultados

### Ver contenido del dataset generado
```powershell
python -c "import json; data = json.load(open('data/dataset.json')); print(f'Customers: {len(data[\"customers\"])}\nBooks: {len(data[\"books\"])}\nLoginIDs: {len(data[\"login_ids\"])}\nRecommendations: {len(data[\"recommendations\"])}')"
```

### Ejecutar solo inferencia MCMC
```powershell
python -c "from tests.test_inference import *; import pytest; pytest.main(['-v', 'tests/test_inference.py'])"
```

### Verificar cobertura de código (opcional)
```powershell
python -m pytest --cov=src --cov-report=term-missing tests/
```

---

## ✅ PASO 6: Comandos de Desarrollo

### Ejecutar script personalizado de Python
```powershell
python -c "from src.data_generator import *; config = DatasetConfig(num_real_users=5, num_bots=3, num_books=4); gen = DataGenerator(config); customers, books, lids, recs = gen.generate_dataset(); print(f'Generado: {len(customers)} customers, {len(recs)} ratings')"
```

### Ver estructura de un módulo
```powershell
python -c "from src import bot_detection; import inspect; print('\n'.join([f'{name}' for name, obj in inspect.getmembers(bot_detection, inspect.isclass)]))"
```

### Verificar imports de todos los módulos
```powershell
python -c "import src.data_generator, src.models, src.rpm_model, src.oupm_model, src.bot_detection, src.query_engine, src.visualization, src.visualization_plots, src.inference.gibbs_sampling, src.inference.metropolis_hastings, src.inference.variable_elimination; print('✓ Todos los imports exitosos')"
```

---

## 📊 Interpretación de Resultados

### Métricas de Clasificación (Bot Detection)

- **Precision (0.75)**: De los customers clasificados como bots, el 75% realmente son bots
- **Recall (0.50)**: El sistema detecta el 50% de todos los bots reales
- **F1-Score (0.60)**: Balance entre precision y recall
- **AUC (0.90)**: EXCELENTE capacidad de discriminación (>0.9 es excelente)

### Valores típicos:
- AUC ≥ 0.9: EXCELENTE
- AUC ≥ 0.8: BUENA
- AUC ≥ 0.7: ACEPTABLE
- AUC < 0.7: POBRE

### Análisis MCMC (Convergencia)

Los gráficos de convergencia muestran:
1. **Trace plots**: Cómo evoluciona cada muestra (debe ser "estable" después del burn-in)
2. **Histogramas**: Distribución posterior inferida
3. **Running mean**: Convergencia de la media (debe estabilizarse)

**Buena convergencia:** Running mean se estabiliza, trace plot no muestra tendencias, ambos algoritmos (Gibbs/MH) coinciden.

---

## 🔧 Troubleshooting

### Error: ModuleNotFoundError
```powershell
# Verificar que estás en el directorio correcto
cd "c:\Users\dsoli\OneDrive\Desktop\proyecto estocasticos"

# Verificar PYTHONPATH
python -c "import sys; print('\n'.join(sys.path))"
```

### Error: matplotlib no encontrado
```powershell
pip install matplotlib
```

### Los gráficos no se generan
```powershell
# Crear carpeta output manualmente
New-Item -ItemType Directory -Force -Path output

# Verificar permisos de escritura
Test-Path -Path output -PathType Container
```

### Tests fallan
```powershell
# Limpiar cache de pytest
Remove-Item -Recurse -Force __pycache__, .pytest_cache, src/__pycache__, tests/__pycache__, src/inference/__pycache__

# Re-ejecutar tests
python -m pytest tests/ -v --tb=short
```

---

## 📝 Comandos de Verificación Rápida

```powershell
# Todo en uno: verificar entorno + tests + demo
python --version ; python -m pytest tests/ -v ; python demo_paso_a_paso.py
```

```powershell
# Verificación mínima (solo tests)
python -m pytest tests/ -v --tb=line
```

```powershell
# Generar gráficos sin interacción
echo "" | python demo_paso_a_paso.py
```

---

## 🎯 Checklist de Verificación

- [ ] Python 3.13+ instalado
- [ ] numpy y matplotlib instalados
- [ ] 30/30 tests pasando
- [ ] `demo_paso_a_paso.py` ejecuta sin errores
- [ ] 8 gráficos PNG generados en `output/`
- [ ] AUC ≥ 0.80 (clasificación BUENA o mejor)
- [ ] Sybil attacks detectados (≥ 5 customers con múltiples cuentas)
- [ ] MCMC converge (running mean se estabiliza)

---

## 📁 Archivos Clave del Proyecto

### Ejecución
- `main.py` - Demo principal (ASCII)
- `demo_paso_a_paso.py` - Demo interactivo con gráficos (PNG)

### Código fuente
- `src/data_generator.py` - Generación de datos sintéticos
- `src/models.py` - Modelos base y CPTs
- `src/rpm_model.py` - Relational Probability Model
- `src/oupm_model.py` - Open Universe Probability Model
- `src/bot_detection.py` - Detección de bots
- `src/query_engine.py` - Motor de consultas
- `src/visualization.py` - Visualización ASCII
- `src/visualization_plots.py` - Visualización PNG (matplotlib)
- `src/inference/variable_elimination.py` - Inferencia exacta
- `src/inference/gibbs_sampling.py` - MCMC Gibbs
- `src/inference/metropolis_hastings.py` - MCMC Metropolis-Hastings

### Tests
- `tests/test_basics.py` - FASE 1-2
- `tests/test_inference.py` - FASE 3
- `tests/test_query_engine.py` - FASE 4
- `tests/test_oupm.py` - FASE 5
- `tests/test_rpm.py` - FASE 6
- `tests/test_visualization.py` - Visualización

### Documentación
- `README.md` - Documentación completa del proyecto
- `FASE2_RESUMEN.md` - Resumen de FASE 2
- `GUIA_VERIFICACION.md` - Esta guía
- `probabilistic_programming.md` - Teoría de programación probabilística

---

## 🚀 Workflow Recomendado

1. **Desarrollo:** Ejecuta tests específicos mientras desarrollas
   ```powershell
   python -m pytest tests/test_<module>.py -v
   ```

2. **Verificación:** Ejecuta todos los tests antes de commit
   ```powershell
   python -m pytest tests/ -v
   ```

3. **Demo:** Ejecuta demo paso a paso para generar gráficos
   ```powershell
   python demo_paso_a_paso.py
   ```

4. **Análisis:** Revisa los gráficos PNG en `output/`
   ```powershell
   explorer output
   ```

5. **Presentación:** Usa `main.py` para demo rápido sin pausas
   ```powershell
   python main.py
   ```

---

**Fecha de última actualización:** 2024
**Versión del proyecto:** 1.0.0 (Todas las fases completadas)
