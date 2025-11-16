"""
EJEMPLO 1: Uso Básico - Pipeline Completo
==========================================

Este script demuestra el uso básico del sistema:
1. Generar datos sintéticos
2. Construir modelo probabilístico
3. Ejecutar inferencia
4. Detectar bots
5. Evaluar resultados

Tiempo de ejecución: ~1 minuto
"""

import sys
import os
# Agregar directorio raíz al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_generator import DataGenerator, DatasetConfig
from src.models import EntityType
from src.rpm_model import RPMModel
from src.bot_detection import BotDetector


def main():
    print("="*70)
    print("  EJEMPLO 1: USO BÁSICO - PIPELINE COMPLETO")
    print("="*70)
    
    # PASO 1: Generar datos sintéticos
    print("\n[PASO 1] Generando datos sintéticos...")
    config = DatasetConfig(
        num_real_users=10,
        num_bots=5,
        num_books=6,
        prob_bot_multiple_accounts=0.7,
        random_seed=42
    )
    
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    print(f"✓ Generados:")
    print(f"  - {len(customers)} customers")
    print(f"  - {len(books)} libros")
    print(f"  - {len(login_ids)} cuentas (loginIDs)")
    print(f"  - {len(recommendations)} recomendaciones")
    
    # PASO 2: Construir modelo
    print("\n[PASO 2] Construyendo modelo probabilístico (RPM)...")
    rpm = RPMModel()
    grounded_network = rpm.ground_model(customers, books, recommendations)
    
    print(f"✓ Red bayesiana construida:")
    print(f"  - {len(grounded_network.variables)} variables")
    print(f"  - {len(grounded_network.observations)} observaciones")
    
    # PASO 3: Detectar bots
    print("\n[PASO 3] Detectando bots (esto puede tomar ~30 segundos)...")
    detector = BotDetector(rpm, detection_threshold=0.5)
    
    bot_scores = detector.score_customers(
        customers,
        books,
        recommendations,
        login_ids,
        num_samples=200  # Reducido para velocidad
    )
    
    print(f"✓ Scores calculados para {len(bot_scores)} customers")
    
    # PASO 4: Ver resultados
    print("\n[PASO 4] Top 5 posibles bots:")
    sorted_scores = sorted(bot_scores, key=lambda x: x.bot_probability, reverse=True)
    
    for i, score in enumerate(sorted_scores[:5], 1):
        tipo = "🤖 BOT" if score.entity_type == EntityType.BOT else "👤 USER"
        pred = "BOT" if score.prediction == EntityType.BOT else "USER"
        print(f"  {i}. {score.customer_id:15s} | P(bot)={score.bot_probability:.3f} | "
              f"Real={tipo:8s} | Predicción={pred:4s} | Cuentas={score.num_accounts}")
    
    # PASO 5: Evaluar métricas
    print("\n[PASO 5] Métricas de evaluación:")
    metrics = detector.evaluate(bot_scores)
    
    print(f"  Precision:  {metrics.precision:.3f}")
    print(f"  Recall:     {metrics.recall:.3f}")
    print(f"  F1-Score:   {metrics.f1_score:.3f}")
    print(f"  Accuracy:   {metrics.accuracy:.3f}")
    
    print(f"\n  Confusion Matrix:")
    print(f"    True Positives (TP):  {metrics.true_positives}")
    print(f"    False Positives (FP): {metrics.false_positives}")
    print(f"    True Negatives (TN):  {metrics.true_negatives}")
    print(f"    False Negatives (FN): {metrics.false_negatives}")
    
    # PASO 6: Detectar sybil attacks
    print("\n[PASO 6] Detectando sybil attacks (múltiples cuentas)...")
    sybil_attacks = detector.detect_sybil_attacks(login_ids, min_accounts=2)
    
    print(f"✓ Detectados {len(sybil_attacks)} customers con múltiples cuentas:")
    for customer_id, accounts in sorted(sybil_attacks.items(), 
                                        key=lambda x: len(x[1]), 
                                        reverse=True)[:5]:
        customer = next((c for c in customers if c.customer_id == customer_id), None)
        tipo = "🤖 BOT" if customer.entity_type == EntityType.BOT else "👤 USER"
        print(f"  - {customer_id:15s} ({tipo}): {len(accounts)} cuentas")
    
    print("\n" + "="*70)
    print("  ✓ EJEMPLO COMPLETADO")
    print("="*70)
    print("\nPróximos pasos:")
    print("  - Ejecuta 'python examples/ejemplo_2_inferencia.py' para ver inferencia")
    print("  - Ejecuta 'python examples/ejemplo_3_visualizacion.py' para gráficos")


if __name__ == "__main__":
    main()
