# Sistema de Detección de Bots

Sistema de programación probabilística para detectar bots en sistemas de recomendación.

Basado en el **Capítulo 18** de "Artificial Intelligence: A Modern Approach" (Russell & Norvig).

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
├── tests/               # Tests unitarios (30 tests)
└── examples/            # Ejemplos ejecutables
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
- **Recall:** 0.833+
- **F1-Score:** 0.909+
- **30/30 tests** pasando

## Tests

```powershell
python -m pytest tests/ -v
```

## Referencias

- Russell, S. & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Chapter 18

---

**Versión:** 2.0.0 | **Fecha:** Noviembre 2025
