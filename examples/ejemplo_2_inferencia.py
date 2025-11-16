"""
EJEMPLO 2: Comparación de Algoritmos de Inferencia
===================================================

Este script compara Gibbs Sampling vs Metropolis-Hastings:
- Velocidad de convergencia
- Calidad de resultados
- Tiempo de ejecución

Tiempo de ejecución: ~1 minuto
"""

import sys
import os
# Agregar directorio raíz al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import time
from src.data_generator import DataGenerator, DatasetConfig
from src.rpm_model import RPMModel
from src.query_engine import QueryEngine


def main():
    print("="*70)
    print("  EJEMPLO 2: COMPARACIÓN DE ALGORITMOS DE INFERENCIA")
    print("="*70)
    
    # Generar datos
    print("\n[PASO 1] Generando datos...")
    config = DatasetConfig(
        num_real_users=8,
        num_bots=4,
        num_books=5,
        random_seed=42
    )
    
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    # Construir modelo
    print("\n[PASO 2] Construyendo modelo...")
    rpm = RPMModel()
    grounded = rpm.ground_model(customers, books, recommendations)
    
    query_engine = QueryEngine(grounded)
    
    # Variable a consultar
    book_id = books[0].book_id
    var_name = f"Quality_{book_id}"
    true_quality = books[0].true_quality
    
    print(f"\n[PASO 3] Consultando: P({var_name} | Evidence)")
    print(f"  Calidad real: {true_quality}")
    
    # Gibbs Sampling
    print("\n  → Ejecutando Gibbs Sampling...")
    start = time.time()
    gibbs_result = query_engine.query_marginal(
        variable=var_name,
        evidence={},
        method='gibbs',
        num_samples=500
    )
    gibbs_time = time.time() - start
    
    print(f"    Tiempo: {gibbs_time:.2f}s")
    print(f"    Distribución inferida:")
    for value in sorted(gibbs_result.distribution.keys()):
        prob = gibbs_result.distribution[value]
        bar = "█" * int(prob * 50)
        marker = " ← TRUE" if value == true_quality else ""
        print(f"      Quality={value}: {bar} {prob:.3f}{marker}")
    
    # Metropolis-Hastings
    print("\n  → Ejecutando Metropolis-Hastings...")
    start = time.time()
    mh_result = query_engine.query_marginal(
        variable=var_name,
        evidence={},
        method='mh',
        num_samples=500
    )
    mh_time = time.time() - start
    
    print(f"    Tiempo: {mh_time:.2f}s")
    print(f"    Distribución inferida:")
    for value in sorted(mh_result.distribution.keys()):
        prob = mh_result.distribution[value]
        bar = "█" * int(prob * 50)
        marker = " ← TRUE" if value == true_quality else ""
        print(f"      Quality={value}: {bar} {prob:.3f}{marker}")
    
    # Comparación
    print("\n[PASO 4] Comparación de resultados:")
    print(f"  Tiempo Gibbs: {gibbs_time:.2f}s")
    print(f"  Tiempo MH:    {mh_time:.2f}s")
    print(f"  Speedup:      {mh_time/gibbs_time:.2f}x")
    
    # Calcular similitud (KL divergence simplificado)
    common_keys = set(gibbs_result.distribution.keys()) & set(mh_result.distribution.keys())
    diff = sum(abs(gibbs_result.distribution.get(k, 0) - mh_result.distribution.get(k, 0)) 
               for k in common_keys)
    
    print(f"\n  Diferencia entre distribuciones: {diff:.4f}")
    if diff < 0.1:
        print("  ✓ Ambos algoritmos CONVERGEN a resultados similares")
    elif diff < 0.3:
        print("  ⚠ Algoritmos tienen diferencias moderadas")
    else:
        print("  ✗ Algoritmos NO convergen (considerar más muestras)")
    
    # Comparar múltiples libros
    print("\n[PASO 5] Comparando inferencia para múltiples libros:")
    for book in books[:3]:
        var_name = f"Quality_{book.book_id}"
        result = query_engine.query_marginal(
            variable=var_name,
            evidence={},
            method='gibbs',
            num_samples=300
        )
        
        # Obtener valor más probable
        max_prob_value = max(result.distribution.items(), key=lambda x: x[1])[0]
        max_prob = result.distribution[max_prob_value]
        
        correct = "✓" if max_prob_value == book.true_quality else "✗"
        print(f"  {book.book_id:15s} | True={book.true_quality} | "
              f"Inferred={max_prob_value} (P={max_prob:.3f}) {correct}")
    
    print("\n" + "="*70)
    print("  ✓ EJEMPLO COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    main()
