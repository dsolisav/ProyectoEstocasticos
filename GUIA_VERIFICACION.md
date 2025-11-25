# Guía de Verificación - Sistema de Detección de Bots

## Flujo de Uso Paso a Paso con Comandos

Esta guía te permite verificar que todo el sistema funciona correctamente.

**Última actualización:** Noviembre 2025  
**Versión del proyecto:** 2.0.0 (Detector de bots mejorado)

---

## ✅ PASO 0: Verificación del Entorno

### Verificar instalación de Python
```powershell
cd "c:\Users\Vile\Desktop\proyecto estocasticos\ProyectoEstocasticos"
python --version
```
**Resultado esperado:** `Python 3.13.x` o superior

### Instalar dependencias
```powershell
pip install -r requirements.txt
```

### Verificar librerías requeridas
```powershell
python -c "import numpy; print(f'numpy {numpy.__version__}')"
python -c "import matplotlib; print(f'matplotlib {matplotlib.__version__}')"
```
**Resultado esperado:**
- `numpy 2.x.x`
- `matplotlib 3.x.x`

### Verificar imports del proyecto
```powershell
python -c "import src.data_generator, src.models, src.rpm_model, src.oupm_model, src.bot_detection, src.query_engine, src.visualization, src.visualization_plots; print('Todos los imports exitosos')"
```

---

## ✅ PASO 1: Ejecutar Ejemplos Básicos

### Ejemplo 1: Pipeline Completo (Recomendado para empezar)
```powershell
python examples/ejemplo_1_basico.py
```

**Resultado esperado:**
- Genera dataset con ~18 customers (12 usuarios + 6 bots)
- Ejecuta detección de bots
- **Precision: 1.000** (todos los detectados son bots reales)
- **Recall: 1.000** (detecta todos los bots)
- **F1-Score: 1.000**

### Ejemplo 2: Comparación de Algoritmos MCMC
```powershell
python examples/ejemplo_2_inferencia.py
```

**Resultado esperado:**
- Compara Gibbs Sampling vs Metropolis-Hastings
- Muestra distribuciones inferidas
- ESS (Effective Sample Size) > 0.3

### Ejemplo 3: Visualizaciones (Genera gráficos PNG)
```powershell
python examples/ejemplo_3_visualizacion.py
```

**Resultado esperado:**
- Genera 7 gráficos PNG en `examples/output/`
- Incluye: estructura de red, convergencia MCMC, distribuciones, ROC, confusion matrix

### Ejemplo 4: Experimento con Muestras
```powershell
python examples/ejemplo_4_experimento.py
```

**Resultado esperado:**
- Compara precisión con diferentes números de muestras (50, 100, 200, 400)
- Recomienda configuración óptima

---

## ✅ PASO 2: Demo Interactivo Completo

### Ejecutar demo con pausas interactivas
```powershell
python demo_paso_a_paso.py
```

Este comando ejecuta un demo interactivo que:
1. **Genera datos sintéticos** (~18 customers, ~130 ratings)
2. **Construye modelos** (RPM con ~34 variables, OUPM)
3. **Ejecuta inferencia** (Gibbs Sampling + Metropolis-Hastings)
4. **Consultas probabilísticas** (Query Engine)
5. **Detección de bots** (Bot scores con nuevo algoritmo mejorado)
6. **Análisis de clasificación** (Curva ROC, matriz de confusión)

**Pausa interactiva:** El demo se pausa después de cada paso. Presiona **ENTER** para continuar.

**Gráficos generados en `output/`:**
- `01_network_structure.png`
- `02_mcmc_convergence.png`
- `03_quality_distribution.png`
- `04_quality_comparison.png`
- `05_bot_scores.png`
- `06_sybil_attacks.png`
- `07_roc_curve.png`
- `08_confusion_matrix.png`

---

## ✅ PASO 3: Diagnóstico de Detección de Bots

### Ejecutar análisis detallado del detector
```powershell
python diagnostico_bots.py
```

**Este script analiza:**
- Comportamiento de cada customer (ratings, varianza, extremos)
- Bots detectados vs no detectados
- Comparación de características entre grupos
- Métricas finales (Precision, Recall, F1)

**Resultado esperado (con detector mejorado v2.0):**
```
Bots detectados: 6/6 (100.0%)
Bots no detectados: 0/6 (0.0%)
Falsos positivos: 0/12

Precision: 1.000
Recall: 1.000
F1-Score: 1.000
```

---

## ✅ PASO 4: Demo Rápido Sin Pausas

### Ejecutar demo principal
```powershell
python main.py
```

Demo continuo sin pausas interactivas.

**Tiempo de ejecución:** ~2-3 minutos

---

## ✅ PASO 5: Verificar Gráficos Generados

### Listar gráficos
```powershell
Get-ChildItem output/*.png | Select-Object Name, Length, LastWriteTime | Format-Table
```

### Abrir carpeta de gráficos
```powershell
explorer output
```

### Abrir gráfico específico
```powershell
Invoke-Item output/07_roc_curve.png
```

---

## 📊 Interpretación de Resultados

### Métricas de Clasificación (Detector Mejorado v2.0)

| Métrica | Valor Esperado | Descripción |
|---------|----------------|-------------|
| **Precision** | 1.000 | 100% de los detectados son bots reales |
| **Recall** | 1.000 | Detecta 100% de todos los bots |
| **F1-Score** | 1.000 | Balance perfecto |
| **AUC** | ≥0.95 | EXCELENTE discriminación |

### Señales del Detector de Bots

El detector mejorado combina 4 señales:

| Señal | Peso | Descripción |
|-------|------|-------------|
| **Ratings Extremos** | 40% | % de ratings que son 1 o 5 |
| **Varianza** | 25% | Varianza de los ratings |
| **Sybil Attacks** | 20% | Número de cuentas por customer |
| **Modelo MCMC** | 15% | P(dishonest) inferida |

### Valores de Referencia - AUC
- AUC ≥ 0.95: **EXCELENTE**
- AUC ≥ 0.90: **MUY BUENA**
- AUC ≥ 0.80: **BUENA**
- AUC ≥ 0.70: **ACEPTABLE**
- AUC < 0.70: **POBRE**

---

## 🔧 Troubleshooting

### Error: ModuleNotFoundError
```powershell
# Verificar directorio correcto
cd "c:\Users\Vile\Desktop\proyecto estocasticos\ProyectoEstocasticos"

# Instalar dependencias
pip install -r requirements.txt
```

### Error: matplotlib no encontrado
```powershell
pip install matplotlib numpy
```

### Los gráficos no se generan
```powershell
# Crear carpeta output
New-Item -ItemType Directory -Force -Path output
New-Item -ItemType Directory -Force -Path examples/output
```

### Limpiar cache de Python
```powershell
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Remove-Item -Recurse -Force .pytest_cache -ErrorAction SilentlyContinue
```

---

## 📝 Comandos de Verificación Rápida

### Verificación mínima (1 minuto)
```powershell
cd "c:\Users\Vile\Desktop\proyecto estocasticos\ProyectoEstocasticos"
python examples/ejemplo_1_basico.py
```

### Verificación completa con gráficos (5 minutos)
```powershell
python demo_paso_a_paso.py
```

### Diagnóstico detallado del detector
```powershell
python diagnostico_bots.py
```

---

## 🎯 Checklist de Verificación

- [ ] Python 3.13+ instalado
- [ ] numpy y matplotlib instalados
- [ ] `ejemplo_1_basico.py` ejecuta sin errores
- [ ] Precision = 1.000 en detección de bots
- [ ] Recall = 1.000 en detección de bots
- [ ] `demo_paso_a_paso.py` genera 8 gráficos PNG
- [ ] AUC ≥ 0.90 (clasificación EXCELENTE)
- [ ] Sybil attacks detectados correctamente
- [ ] `diagnostico_bots.py` muestra análisis completo

---

## 📁 Estructura del Proyecto

```
ProyectoEstocasticos/
├── main.py                      # Demo rápido
├── demo_paso_a_paso.py          # Demo interactivo con gráficos
├── diagnostico_bots.py          # Análisis del detector
├── requirements.txt             # Dependencias
├── MANUAL_USUARIO.md            # Manual completo
├── GUIA_VERIFICACION.md         # Esta guía
│
├── examples/                    # Ejemplos ejecutables
│   ├── ejemplo_1_basico.py      # Pipeline completo
│   ├── ejemplo_2_inferencia.py  # Comparación MCMC
│   ├── ejemplo_3_visualizacion.py # Genera gráficos
│   ├── ejemplo_4_experimento.py # Experimentos
│   └── output/                  # Gráficos de ejemplos
│
├── src/                         # Código fuente
│   ├── data_generator.py        # Generación de datos
│   ├── models.py                # Modelos base
│   ├── cpt.py                   # Conditional Probability Tables
│   ├── grounding.py             # Grounding RPM → Bayes Net
│   ├── rpm_model.py             # Relational Probability Model
│   ├── oupm_model.py            # Open Universe Model
│   ├── origin_functions.py      # Origin functions OUPM
│   ├── bot_detection.py         # Detector de bots (v2.0 mejorado)
│   ├── query_engine.py          # Motor de consultas
│   ├── visualization.py         # Visualización ASCII
│   ├── visualization_plots.py   # Visualización PNG
│   └── inference/               # Algoritmos de inferencia
│       ├── variable_elimination.py
│       ├── gibbs_sampling.py
│       └── metropolis_hastings.py
│
├── tests/                       # Tests unitarios
│   ├── test_basics.py
│   ├── test_inference.py
│   ├── test_oupm.py
│   ├── test_query_engine.py
│   ├── test_rpm.py
│   └── test_visualization.py
│
├── output/                      # Gráficos generados
└── data/                        # Datasets generados
```

---

## 🚀 Workflow Recomendado

1. **Primera vez:** Ejecutar ejemplo básico
   ```powershell
   python examples/ejemplo_1_basico.py
   ```

2. **Ver gráficos:** Ejecutar demo con visualizaciones
   ```powershell
   python demo_paso_a_paso.py
   explorer output
   ```

3. **Análisis profundo:** Ejecutar diagnóstico
   ```powershell
   python diagnostico_bots.py
   ```

4. **Experimentar:** Modificar parámetros en ejemplos
   ```powershell
   python examples/ejemplo_4_experimento.py
   ```

---

**Repositorio:** https://github.com/dsolisav/ProyectoEstocasticos
