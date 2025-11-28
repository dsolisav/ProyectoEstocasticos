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
    print("\n" + "="*60)
    print("   ESCENARIO 2: Comparacion de algoritmos MCMC")
    print("="*60)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuracion
    print("\nConfiguracion:")
    print("  - Gibbs Sampling vs Metropolis-Hastings")
    print("  - 500 muestras, 100 burn-in")
    print("  - Variable objetivo: Quality de un libro")
    
    config = DatasetConfig(
        num_real_users=8,
        num_bots=4,
        num_books=5,
        random_seed=42
    )
    
    # Preparar datos
    print("\nPreparando modelo...")
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    rpm = RPMModel()
    grounded = rpm.ground_model(customers, books, recommendations)
    query_engine = QueryEngine(grounded)
    
    book = books[0]
    var_name = f"Quality_{book.book_id}"
    true_quality = book.true_quality
    
    print(f"  Dataset: {len(customers)} customers, {len(recommendations)} ratings")
    print(f"  Inferir: {var_name}")
    print(f"  Valor real: {true_quality}")
    
    # Gibbs Sampling
    print("\nEjecutando Gibbs Sampling...")
    start = time.time()
    gibbs_result = query_engine.query_marginal(
        variable=var_name,
        evidence={},
        method='gibbs',
        num_samples=500
    )
    gibbs_time = time.time() - start
    
    print(f"  Tiempo: {gibbs_time:.2f}s")
    print(f"  Distribucion:")
    gibbs_max_val = max(gibbs_result.distribution.items(), key=lambda x: x[1])
    for value in sorted(gibbs_result.distribution.keys()):
        prob = gibbs_result.distribution[value]
        bar = "#" * int(prob * 30)
        marker = " <- real" if value == true_quality else ""
        marker += " (max)" if value == gibbs_max_val[0] else ""
        print(f"    Q={value}: {bar} {prob:.3f}{marker}")
    
    # Metropolis-Hastings
    print("\nEjecutando Metropolis-Hastings...")
    start = time.time()
    mh_result = query_engine.query_marginal(
        variable=var_name,
        evidence={},
        method='mh',
        num_samples=500
    )
    mh_time = time.time() - start
    
    print(f"  Tiempo: {mh_time:.2f}s")
    print(f"  Distribucion:")
    mh_max_val = max(mh_result.distribution.items(), key=lambda x: x[1])
    for value in sorted(mh_result.distribution.keys()):
        prob = mh_result.distribution[value]
        bar = "#" * int(prob * 30)
        marker = " <- real" if value == true_quality else ""
        marker += " (max)" if value == mh_max_val[0] else ""
        print(f"    Q={value}: {bar} {prob:.3f}{marker}")
    
    # Comparacion
    print("\n--- COMPARACION ---")
    print(f"  Gibbs:    {gibbs_time:.2f}s")
    print(f"  MH:       {mh_time:.2f}s")
    
    # Calcular diferencia entre distribuciones
    common_keys = set(gibbs_result.distribution.keys()) & set(mh_result.distribution.keys())
    diff = sum(abs(gibbs_result.distribution.get(k, 0) - mh_result.distribution.get(k, 0)) 
               for k in common_keys)
    
    print(f"  Diferencia L1: {diff:.4f}")
    
    if diff < 0.15:
        print("  -> Convergen bien")
    elif diff < 0.3:
        print("  -> Convergen parcialmente")
    else:
        print("  -> No convergen, necesitan mas muestras")
    
    # Graficos
    print("\nGenerando graficos...")
    
    dist_plotter = DistributionPlotter()
    
    path1 = os.path.join(output_dir, 'esc2_distribucion_gibbs.png')
    dist_plotter.plot_distribution(
        gibbs_result.distribution,
        title=f"Gibbs Sampling: P({var_name})",
        xlabel="Nivel de Calidad",
        output_path=path1
    )
    print(f"  -> {path1}")
    
    path2 = os.path.join(output_dir, 'esc2_distribucion_mh.png')
    dist_plotter.plot_distribution(
        mh_result.distribution,
        title=f"Metropolis-Hastings: P({var_name})",
        xlabel="Nivel de Calidad",
        output_path=path2
    )
    print(f"  -> {path2}")
    
    # Conclusion
    print("\n" + "="*60)
    
    gibbs_correct = gibbs_max_val[0] == true_quality
    mh_correct = mh_max_val[0] == true_quality
    
    print(f"""Resultados:
    
  Gibbs Sampling:
    Valor inferido: {gibbs_max_val[0]} (P={gibbs_max_val[1]:.3f})
    Correcto: {"Si" if gibbs_correct else "No"}
    
  Metropolis-Hastings:
    Valor inferido: {mh_max_val[0]} (P={mh_max_val[1]:.3f})
    Correcto: {"Si" if mh_correct else "No"}
  
Gibbs muestrea de las condicionales directamente.
MH usa accept/reject, puede necesitar mas muestras.
  
Graficos guardados en examples/output/
""")


if __name__ == "__main__":
    main()
