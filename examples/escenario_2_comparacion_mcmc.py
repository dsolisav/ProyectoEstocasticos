"""
ESCENARIO 2: Comparación de Algoritmos MCMC
============================================

Objetivo: Comparar el rendimiento de dos algoritmos de inferencia 
MCMC: Gibbs Sampling vs Metropolis-Hastings.

Hipótesis: Ambos algoritmos deben converger a distribuciones similares,
pero con diferentes características de eficiencia.

Salida: 2 gráficos PNG en examples/output/
- esc2_distribucion_gibbs.png
- esc2_distribucion_mh.png

Tiempo de ejecución: ~1 minuto
"""

import sys
import os
import random
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

random.seed(42)

from src.data_generator import DataGenerator, DatasetConfig
from src.rpm_model import RPMModel
from src.query_engine import QueryEngine
from src.visualization_plots import DistributionPlotter


def main():
    print("="*70)
    print("  ESCENARIO 2: COMPARACIÓN DE ALGORITMOS MCMC")
    print("="*70)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================
    # CONFIGURACIÓN DEL EXPERIMENTO
    # =========================================
    print("\n[CONFIGURACIÓN]")
    print("  - Algoritmos: Gibbs Sampling vs Metropolis-Hastings")
    print("  - Muestras: 500 por algoritmo")
    print("  - Burn-in: 100 muestras descartadas")
    print("  - Variable: Calidad de un libro (Quality)")
    
    config = DatasetConfig(
        num_real_users=8,
        num_bots=4,
        num_books=5,
        random_seed=42
    )
    
    # =========================================
    # PASO 1: Preparar datos y modelo
    # =========================================
    print("\n[PASO 1] Preparando datos y modelo...")
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    rpm = RPMModel()
    grounded = rpm.ground_model(customers, books, recommendations)
    query_engine = QueryEngine(grounded)
    
    book = books[0]
    var_name = f"Quality_{book.book_id}"
    true_quality = book.true_quality
    
    print(f"  Dataset: {len(customers)} customers, {len(recommendations)} ratings")
    print(f"  Variable a inferir: {var_name}")
    print(f"  Calidad real: {true_quality}")
    
    # =========================================
    # PASO 2: Ejecutar Gibbs Sampling
    # =========================================
    print("\n[PASO 2] Ejecutando Gibbs Sampling...")
    start = time.time()
    gibbs_result = query_engine.query_marginal(
        variable=var_name,
        evidence={},
        method='gibbs',
        num_samples=500
    )
    gibbs_time = time.time() - start
    
    print(f"  Tiempo: {gibbs_time:.2f}s")
    print(f"  Distribución inferida:")
    gibbs_max_val = max(gibbs_result.distribution.items(), key=lambda x: x[1])
    for value in sorted(gibbs_result.distribution.keys()):
        prob = gibbs_result.distribution[value]
        bar = "█" * int(prob * 40)
        marker = " <-- TRUE" if value == true_quality else ""
        marker += " (max)" if value == gibbs_max_val[0] else ""
        print(f"    Q={value}: {bar} {prob:.3f}{marker}")
    
    # =========================================
    # PASO 3: Ejecutar Metropolis-Hastings
    # =========================================
    print("\n[PASO 3] Ejecutando Metropolis-Hastings...")
    start = time.time()
    mh_result = query_engine.query_marginal(
        variable=var_name,
        evidence={},
        method='mh',
        num_samples=500
    )
    mh_time = time.time() - start
    
    print(f"  Tiempo: {mh_time:.2f}s")
    print(f"  Distribución inferida:")
    mh_max_val = max(mh_result.distribution.items(), key=lambda x: x[1])
    for value in sorted(mh_result.distribution.keys()):
        prob = mh_result.distribution[value]
        bar = "█" * int(prob * 40)
        marker = " <-- TRUE" if value == true_quality else ""
        marker += " (max)" if value == mh_max_val[0] else ""
        print(f"    Q={value}: {bar} {prob:.3f}{marker}")
    
    # =========================================
    # PASO 4: Comparación
    # =========================================
    print("\n[COMPARACIÓN]")
    print(f"  Tiempo Gibbs:    {gibbs_time:.2f}s")
    print(f"  Tiempo MH:       {mh_time:.2f}s")
    
    # Calcular diferencia entre distribuciones
    common_keys = set(gibbs_result.distribution.keys()) & set(mh_result.distribution.keys())
    diff = sum(abs(gibbs_result.distribution.get(k, 0) - mh_result.distribution.get(k, 0)) 
               for k in common_keys)
    
    print(f"  Diferencia (L1): {diff:.4f}")
    
    if diff < 0.15:
        convergence = "CONVERGEN - Resultados muy similares"
    elif diff < 0.3:
        convergence = "CONVERGEN PARCIALMENTE - Diferencias moderadas"
    else:
        convergence = "NO CONVERGEN - Considerar más muestras"
    
    print(f"  Estado: {convergence}")
    
    # =========================================
    # GRÁFICOS
    # =========================================
    print("\n[GRÁFICOS]")
    
    dist_plotter = DistributionPlotter()
    
    # Gráfico 1: Distribución Gibbs
    path1 = os.path.join(output_dir, 'esc2_distribucion_gibbs.png')
    dist_plotter.plot_distribution(
        gibbs_result.distribution,
        title=f"Gibbs Sampling: P({var_name})",
        xlabel="Nivel de Calidad",
        output_path=path1
    )
    print(f"  Guardado: {path1}")
    
    # Gráfico 2: Distribución MH
    path2 = os.path.join(output_dir, 'esc2_distribucion_mh.png')
    dist_plotter.plot_distribution(
        mh_result.distribution,
        title=f"Metropolis-Hastings: P({var_name})",
        xlabel="Nivel de Calidad",
        output_path=path2
    )
    print(f"  Guardado: {path2}")
    
    # =========================================
    # CONCLUSIÓN
    # =========================================
    print("\n" + "="*70)
    print("  CONCLUSIÓN")
    print("="*70)
    
    gibbs_correct = gibbs_max_val[0] == true_quality
    mh_correct = mh_max_val[0] == true_quality
    
    print(f"""
  Resultados de inferencia:
  
  Gibbs Sampling:
    - Valor más probable: {gibbs_max_val[0]} (P={gibbs_max_val[1]:.3f})
    - Valor real: {true_quality}
    - Correcto: {"Sí" if gibbs_correct else "No"}
    
  Metropolis-Hastings:
    - Valor más probable: {mh_max_val[0]} (P={mh_max_val[1]:.3f})
    - Valor real: {true_quality}
    - Correcto: {"Sí" if mh_correct else "No"}
  
  Análisis:
  - Gibbs Sampling muestrea directamente de distribuciones condicionales
  - Metropolis-Hastings usa mecanismo accept/reject
  - Ambos son métodos MCMC válidos para inferencia aproximada
  
  Gráficos generados en: examples/output/
""")


if __name__ == "__main__":
    main()
