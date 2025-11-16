# Ejemplos del Sistema de Detección de Bots

Esta carpeta contiene ejemplos ejecutables que demuestran el uso del sistema.

## 📋 Lista de Ejemplos

### 1. **ejemplo_1_basico.py** - Uso Básico
**Tiempo:** ~1 minuto

Pipeline completo paso a paso:
- Generar datos sintéticos
- Construir modelo probabilístico
- Detectar bots
- Evaluar resultados
- Detectar sybil attacks

```powershell
python examples/ejemplo_1_basico.py
```

### 2. **ejemplo_2_inferencia.py** - Comparación de Algoritmos
**Tiempo:** ~1 minuto

Compara Gibbs Sampling vs Metropolis-Hastings:
- Tiempo de ejecución
- Calidad de convergencia
- Distribuciones inferidas

```powershell
python examples/ejemplo_2_inferencia.py
```

### 3. **ejemplo_3_visualizacion.py** - Generación de Gráficos
**Tiempo:** ~2 minutos

Genera todos los gráficos del sistema:
- Estructura de red
- Distribuciones de probabilidad
- Bot scores
- Curva ROC
- Matriz de confusión

```powershell
python examples/ejemplo_3_visualizacion.py
```

Los gráficos se guardan en `examples/output/`

### 4. **ejemplo_4_experimento.py** - Experimentación
**Tiempo:** ~3-4 minutos

Experimenta con diferentes configuraciones:
- Efecto del número de muestras en la precisión
- Trade-off tiempo vs precisión
- Recomendaciones de configuración

```powershell
python examples/ejemplo_4_experimento.py
```

## 🚀 Ejecutar Todos los Ejemplos

```powershell
# Desde la raíz del proyecto
python examples/ejemplo_1_basico.py
python examples/ejemplo_2_inferencia.py
python examples/ejemplo_3_visualizacion.py
python examples/ejemplo_4_experimento.py
```

## 📁 Salidas Generadas

Los gráficos del Ejemplo 3 se guardan en:
```
examples/output/
├── 01_network_structure.png
├── 02_quality_distribution.png
├── 03_quality_comparison.png
├── 04_bot_scores.png
├── 05_sybil_attacks.png
├── 06_roc_curve.png
└── 07_confusion_matrix.png
```

## 💡 Uso de los Ejemplos

Estos ejemplos son **código ejecutable** que puedes:
- ✅ Ejecutar directamente para ver resultados
- ✅ Modificar parámetros para experimentar
- ✅ Usar como plantillas para tus propios scripts
- ✅ Copiar snippets en tu código

## 🔧 Personalización

Puedes modificar los parámetros en cada ejemplo:

```python
# Cambiar configuración del dataset
config = DatasetConfig(
    num_real_users=20,     # Más usuarios
    num_bots=10,           # Más bots
    num_books=15,          # Más libros
    random_seed=123        # Diferente semilla
)

# Cambiar número de muestras
num_samples=500  # Más muestras = mejor precisión (más lento)

# Cambiar threshold de detección
detection_threshold=0.6  # Más conservador
```

## 📚 Más Información

- **README.md** - Documentación técnica completa
- **MANUAL_USUARIO.md** - Guía de usuario detallada
- **GUIA_VERIFICACION.md** - Comandos de verificación
