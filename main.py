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
    print("\n" + "=" * 60)
    print("   Sistema de Recomendacion - Deteccion de Bots")
    print("   Basado en el Capitulo 18 (Russell & Norvig)")
    print("=" * 60)
    
    # Fase 1: Datos
    print("\n>> Generando datos sinteticos...")
    
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
    
    print(f"   {len(customers)} customers ({num_users} usuarios, {num_bots} bots)")
    print(f"   {len(books)} libros, {len(login_ids)} cuentas")
    print(f"   {len(recommendations)} ratings")
    
    # Fase 2: Modelos
    print("\n>> Construyendo modelos probabilisticos...")
    
    # RPM
    rpm = RPMModel()
    grounded_rpm = rpm.ground_model(customers, books, recommendations)
    print(f"   RPM: {len(grounded_rpm.variables)} variables")
    
    # OUPM
    oupm = OUPMModel()
    print(f"   OUPM: soporta identity/existence uncertainty")
    
    # Fase 3: Inferencia
    print("\n>> Ejecutando inferencia MCMC...")
    
    query_engine = QueryEngine(grounded_rpm)
    
    book_var = f"Quality_{books[0].book_id}"
    if book_var in grounded_rpm.variables:
        print(f"   Consultando P({book_var})")
        print(f"   Calidad real: {books[0].true_quality}")
        
        result = query_engine.query_marginal(
            variable=book_var,
            evidence={},
            method='gibbs',
            num_samples=500,
            burn_in=100
        )
        
        print(f"\n   Distribucion inferida (Gibbs):")
        for val, prob in sorted(result.distribution.items()):
            bar = "#" * int(prob * 25)
            marker = " <- real" if val == books[0].true_quality else ""
            print(f"     Q={val}: {bar} {prob:.3f}{marker}")
    
    # Fase 4: Deteccion de bots
    print("\n>> Detectando bots...")
    
    detector = BotDetector(rpm, detection_threshold=0.5)
    bot_scores = detector.score_customers(
        customers=customers,
        books=books,
        recommendations=recommendations,
        login_ids=login_ids,
        num_samples=200
    )
    
    print(f"   Scores para {len(bot_scores)} customers")
    
    print("\n   Top 5 sospechosos:")
    for i, score in enumerate(bot_scores[:5]):
        is_bot = score.entity_type == EntityType.BOT
        real_type = "BOT" if is_bot else "USER"
        pred_type = "BOT" if score.prediction == EntityType.BOT else "USER"
        ok = "ok" if (is_bot == (score.prediction == EntityType.BOT)) else "x"
        print(f"     {i+1}. {score.customer_id:14s} P={score.bot_probability:.3f} "
              f"({real_type}->{pred_type}) [{ok}]")
    
    # Metricas
    metrics = detector.evaluate(bot_scores)
    
    print(f"\n   Metricas:")
    print(f"     Precision={metrics.precision:.3f}, Recall={metrics.recall:.3f}")
    print(f"     F1={metrics.f1_score:.3f}, Accuracy={metrics.accuracy:.3f}")
    
    print(f"\n   Confusion matrix:")
    print(f"     TP={metrics.true_positives}, FP={metrics.false_positives}")
    print(f"     FN={metrics.false_negatives}, TN={metrics.true_negatives}")
    
    # Fase 5: Sybil
    print("\n>> Buscando sybil attacks...")
    
    sybil_attacks = detector.detect_sybil_attacks(login_ids, min_accounts=2)
    
    print(f"   {len(sybil_attacks)} con multiples cuentas:")
    for cust_id, accounts in sorted(sybil_attacks.items(), 
                                     key=lambda x: len(x[1]), reverse=True)[:5]:
        customer = next((c for c in customers if c.customer_id == cust_id), None)
        if customer:
            ctype = "BOT" if customer.entity_type == EntityType.BOT else "USER"
            print(f"     {cust_id:14s} ({ctype}): {len(accounts)} cuentas")
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    
    print(f"""
Dataset: {len(customers)} customers, {len(books)} libros, {len(recommendations)} ratings
Modelo: {len(grounded_rpm.variables)} variables

Deteccion:
  Precision={metrics.precision:.3f}, Recall={metrics.recall:.3f}, F1={metrics.f1_score:.3f}
  Sybil attacks: {len(sybil_attacks)}

Terminado.
""")


if __name__ == "__main__":
    main()


