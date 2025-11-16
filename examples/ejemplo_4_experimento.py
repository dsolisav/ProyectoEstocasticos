"""
EJEMPLO 4: Experimentación - Efecto del Número de Muestras
===========================================================

Este script experimenta con diferentes configuraciones:
- ¿Cómo afecta el número de muestras a la precisión?
- ¿Cuál es el trade-off tiempo vs precisión?

Tiempo de ejecución: ~3-4 minutos
"""

import sys
import os
# Agregar directorio raíz al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import time
from src.data_generator import DataGenerator, DatasetConfig
from src.rpm_model import RPMModel
from src.bot_detection import BotDetector


def main():
    print("="*70)
    print("  EJEMPLO 4: EXPERIMENTACIÓN - EFECTO DEL NÚMERO DE MUESTRAS")
    print("="*70)
    
    # Generar datos
    print("\n[PASO 1] Generando dataset de prueba...")
    config = DatasetConfig(
        num_real_users=10,
        num_bots=5,
        num_books=6,
        random_seed=42
    )
    
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    print(f"✓ Dataset generado: {len(customers)} customers")
    
    # Construir modelo
    print("\n[PASO 2] Construyendo modelo...")
    rpm = RPMModel()
    grounded = rpm.ground_model(customers, books, recommendations)
    print(f"✓ Modelo construido")
    
    # Experimentar con diferentes números de muestras
    print("\n[PASO 3] Experimentando con diferentes números de muestras:")
    print("\n  Samples | Tiempo | Precision | Recall | F1-Score | Accuracy")
    print("  " + "-"*65)
    
    detector = BotDetector(rpm)
    sample_counts = [50, 100, 200, 400]
    
    results = []
    for num_samples in sample_counts:
        start_time = time.time()
        
        bot_scores = detector.score_customers(
            customers,
            books,
            recommendations,
            login_ids,
            num_samples=num_samples
        )
        
        elapsed = time.time() - start_time
        metrics = detector.evaluate(bot_scores)
        
        results.append({
            'samples': num_samples,
            'time': elapsed,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1': metrics.f1_score,
            'accuracy': metrics.accuracy
        })
        
        print(f"  {num_samples:4d}    | {elapsed:5.1f}s | "
              f"{metrics.precision:7.3f}   | {metrics.recall:6.3f} | "
              f"{metrics.f1_score:8.3f} | {metrics.accuracy:8.3f}")
    
    # Análisis
    print("\n[PASO 4] Análisis de resultados:")
    
    # Encontrar mejor F1
    best_f1 = max(results, key=lambda x: x['f1'])
    print(f"\n  Mejor F1-Score: {best_f1['f1']:.3f} con {best_f1['samples']} muestras")
    
    # Encontrar mejor balance tiempo/precisión
    # Score = F1 / tiempo (más alto es mejor)
    for r in results:
        r['efficiency'] = r['f1'] / r['time']
    
    best_efficiency = max(results, key=lambda x: x['efficiency'])
    print(f"  Mejor eficiencia: {best_efficiency['samples']} muestras "
          f"(F1={best_efficiency['f1']:.3f} en {best_efficiency['time']:.1f}s)")
    
    # Comparar primero vs último
    first = results[0]
    last = results[-1]
    
    f1_improvement = ((last['f1'] - first['f1']) / first['f1']) * 100
    time_increase = ((last['time'] - first['time']) / first['time']) * 100
    
    print(f"\n  De {first['samples']} a {last['samples']} muestras:")
    print(f"    F1-Score mejora:    {f1_improvement:+.1f}%")
    print(f"    Tiempo incrementa:  {time_increase:+.1f}%")
    
    # Recomendación
    print("\n[PASO 5] Recomendación:")
    if best_efficiency['samples'] < 200:
        print(f"  ✓ Para uso rápido: {best_efficiency['samples']} muestras")
        print(f"    (F1={best_efficiency['f1']:.3f}, tiempo={best_efficiency['time']:.1f}s)")
    if best_f1['samples'] >= 200:
        print(f"  ✓ Para mejor precisión: {best_f1['samples']} muestras")
        print(f"    (F1={best_f1['f1']:.3f}, tiempo={best_f1['time']:.1f}s)")
    
    print("\n" + "="*70)
    print("  ✓ EXPERIMENTO COMPLETADO")
    print("="*70)
    
    print("\nConclusión:")
    print("  - Más muestras generalmente mejoran la precisión")
    print("  - Existe un punto de rendimientos decrecientes")
    print("  - Para producción, considera el trade-off tiempo/precisión")


if __name__ == "__main__":
    main()
