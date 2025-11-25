"""
Demo Principal del Sistema de Recomendacion con Deteccion de Bots

Este script ejecuta un pipeline completo que demuestra:
1. Generacion de datos sinteticos
2. Construccion de modelos probabilisticos (RPM y OUPM)
3. Inferencia con MCMC (Gibbs Sampling y Metropolis-Hastings)
4. Deteccion de bots y sybil attacks
5. Evaluacion de metricas
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import DataGenerator, DatasetConfig
from src.models import EntityType
from src.rpm_model import RPMModel
from src.oupm_model import OUPMModel
from src.bot_detection import BotDetector
from src.query_engine import QueryEngine
from src.visualization import BayesianNetworkVisualizer


def main():
    print("=" * 70)
    print("  SISTEMA DE RECOMENDACION CON DETECCION DE BOTS")
    print("  Programacion Probabilistica - Capitulo 18")
    print("=" * 70)
    
    # ========================================================================
    # FASE 1: Generacion de Datos Sinteticos
    # ========================================================================
    print("\n" + "=" * 70)
    print("FASE 1: GENERACION DE DATOS SINTETICOS")
    print("=" * 70)
    
    config = DatasetConfig(
        num_real_users=10,
        num_bots=5,
        num_books=8,
        prob_bot_multiple_accounts=0.8,
        max_accounts_per_bot=8,
        max_recommendations_per_account=6,
        random_seed=42
    )
    
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    num_users = sum(1 for c in customers if c.entity_type == EntityType.REAL_USER)
    num_bots = sum(1 for c in customers if c.entity_type == EntityType.BOT)
    
    print(f"\n[OK] Dataset generado:")
    print(f"  Customers: {len(customers)} ({num_users} users, {num_bots} bots)")
    print(f"  Books: {len(books)}")
    print(f"  LoginIDs: {len(login_ids)}")
    print(f"  Recommendations: {len(recommendations)}")
    
    # ========================================================================
    # FASE 2-3: Modelos Probabilisticos (RPM y OUPM)
    # ========================================================================
    print("\n" + "=" * 70)
    print("FASE 2-3: CONSTRUCCION DE MODELOS PROBABILISTICOS")
    print("=" * 70)
    
    # RPM Model
    rpm = RPMModel()
    grounded_rpm = rpm.ground_model(customers, books, recommendations)
    print(f"\n[OK] RPM Model grounded:")
    print(f"  Variables: {len(grounded_rpm.variables)}")
    
    # OUPM Model
    oupm = OUPMModel()
    print(f"\n[OK] OUPM Model creado")
    print(f"  Soporta: Identity Uncertainty, Existence Uncertainty")
    
    # ========================================================================
    # FASE 4: Inferencia MCMC
    # ========================================================================
    print("\n" + "=" * 70)
    print("FASE 4: INFERENCIA MCMC")
    print("=" * 70)
    
    query_engine = QueryEngine(grounded_rpm)
    
    # Consultar calidad de un libro
    book_var = f"Quality_{books[0].book_id}"
    if book_var in grounded_rpm.variables:
        print(f"\nConsultando: P({book_var} | Evidence)")
        print(f"  Calidad real: {books[0].true_quality}")
        
        result = query_engine.query_marginal(
            variable=book_var,
            evidence={},
            method='gibbs',
            num_samples=500,
            burn_in=100
        )
        
        print(f"\n  Distribucion inferida (Gibbs):")
        for val, prob in sorted(result.distribution.items()):
            bar = "#" * int(prob * 30)
            marker = " <-- TRUE" if val == books[0].true_quality else ""
            print(f"    Quality={val}: {bar} {prob:.3f}{marker}")
    
    # ========================================================================
    # FASE 5: Deteccion de Bots
    # ========================================================================
    print("\n" + "=" * 70)
    print("FASE 5: DETECCION DE BOTS")
    print("=" * 70)
    
    detector = BotDetector(rpm, detection_threshold=0.5)
    bot_scores = detector.score_customers(
        customers=customers,
        books=books,
        recommendations=recommendations,
        login_ids=login_ids,
        num_samples=200
    )
    
    print(f"\n[OK] Scores calculados para {len(bot_scores)} customers")
    
    # Top 5 posibles bots
    print("\nTop 5 posibles bots:")
    for i, score in enumerate(bot_scores[:5]):
        is_bot = score.entity_type == EntityType.BOT
        real_type = "BOT" if is_bot else "USER"
        pred_type = "BOT" if score.prediction == EntityType.BOT else "USER"
        correct = "[OK]" if (is_bot and score.prediction == EntityType.BOT) or \
                           (not is_bot and score.prediction != EntityType.BOT) else "[X]"
        print(f"  {i+1}. {score.customer_id:15} | P(bot)={score.bot_probability:.3f} | "
              f"Real={real_type:4} | Pred={pred_type:4} | Cuentas={score.num_accounts} {correct}")
    
    # Metricas
    metrics = detector.evaluate(bot_scores)
    
    print(f"\nMetricas de evaluacion:")
    print(f"  Precision:  {metrics.precision:.3f}")
    print(f"  Recall:     {metrics.recall:.3f}")
    print(f"  F1-Score:   {metrics.f1_score:.3f}")
    print(f"  Accuracy:   {metrics.accuracy:.3f}")
    
    print(f"\n  Confusion Matrix:")
    print(f"    True Positives (TP):  {metrics.true_positives}")
    print(f"    False Positives (FP): {metrics.false_positives}")
    print(f"    True Negatives (TN):  {metrics.true_negatives}")
    print(f"    False Negatives (FN): {metrics.false_negatives}")
    
    # ========================================================================
    # FASE 6: Sybil Attacks
    # ========================================================================
    print("\n" + "=" * 70)
    print("FASE 6: DETECCION DE SYBIL ATTACKS")
    print("=" * 70)
    
    sybil_attacks = detector.detect_sybil_attacks(login_ids, min_accounts=2)
    
    print(f"\n[OK] Detectados {len(sybil_attacks)} customers con multiples cuentas:")
    for cust_id, accounts in sorted(sybil_attacks.items(), 
                                     key=lambda x: len(x[1]), reverse=True)[:5]:
        customer = next((c for c in customers if c.customer_id == cust_id), None)
        if customer:
            ctype = "BOT" if customer.entity_type == EntityType.BOT else "USER"
            print(f"  - {cust_id:15} ({ctype}): {len(accounts)} cuentas")
    
    # ========================================================================
    # Resumen Final
    # ========================================================================
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    
    print(f"""
Sistema de Recomendacion con Deteccion de Bots
----------------------------------------------
Dataset:
  - {len(customers)} customers ({num_users} usuarios, {num_bots} bots)
  - {len(books)} libros
  - {len(recommendations)} recomendaciones
  - {len(login_ids)} cuentas (LoginIDs)

Modelo RPM:
  - {len(grounded_rpm.variables)} variables
  
Deteccion de Bots:
  - Precision: {metrics.precision:.3f}
  - Recall: {metrics.recall:.3f}
  - F1-Score: {metrics.f1_score:.3f}
  - Sybil attacks detectados: {len(sybil_attacks)}

[OK] Demo completado exitosamente!
""")


if __name__ == "__main__":
    main()


