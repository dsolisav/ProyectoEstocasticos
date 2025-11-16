"""
Sistema de Recomendación con Detección de Bots
Demo completo mostrando todas las fases del proyecto
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generator import DataGenerator, DatasetConfig
from src.models import Customer, Book, LoginID, Recommendation, EntityType
from src.rpm_model import RPMModel
from src.oupm_model import OUPMModel
from src.bot_detection import BotDetector
from src.query_engine import QueryEngine
from src.visualization import (
    BayesianNetworkVisualizer,
    DistributionPlotter,
    ROCCurveAnalyzer,
    ConvergenceAnalyzer
)
from src.inference.gibbs_sampling import GibbsSampling
from src.inference.metropolis_hastings import MetropolisHastings


def main():
    """Demo completo del sistema."""
    print("\n" + "="*70)
    print("  SISTEMA DE RECOMENDACIÓN CON DETECCIÓN DE BOTS")
    print("  Programación Probabilística - Capítulo 18")
    print("="*70)
    
    # ========================================================================
    # FASE 1: Generación de Datos
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 1: GENERACIÓN DE DATOS SINTÉTICOS")
    print("="*70)
    
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
    
    print(f"\n✓ Dataset generado:")
    print(f"  Customers: {len(customers)} ({sum(1 for c in customers if c.entity_type == EntityType.REAL_USER)} users, {sum(1 for c in customers if c.entity_type == EntityType.BOT)} bots)")
    print(f"  Books: {len(books)}")
    print(f"  LoginIDs: {len(login_ids)}")
    print(f"  Recommendations: {len(recommendations)}")
    
    # ========================================================================
    # FASE 2-3: Modelos Probabilísticos (RPM y OUPM)
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 2-3: CONSTRUCCIÓN DE MODELOS PROBABILÍSTICOS")
    print("="*70)
    
    # RPM Model
    rpm = RPMModel()
    grounded_rpm = rpm.ground_model(customers, books, recommendations)
    print(f"\n✓ RPM Model grounded:")
    print(f"  Variables: {len(grounded_rpm.variables)}")
    print(f"  Observaciones: {len(grounded_rpm.observations)}")
    
    # Visualizar estructura
    visualizer = BayesianNetworkVisualizer()
    network_viz = visualizer.visualize_network(grounded_rpm)
    print("\n" + network_viz)
    
    # OUPM Model
    oupm = OUPMModel(lambda_customers=len(customers), lambda_bots=5.0)
    print(f"\n✓ OUPM Model creado:")
    print(f"  Origin functions: {oupm.origin_function}")
    print(f"  Variables: Quality, Honest, Recommendation")
    
    # ========================================================================
    # FASE 4: Algoritmos de Inferencia
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 4: INFERENCIA PROBABILÍSTICA")
    print("="*70)
    
    print("\n[1] Gibbs Sampling...")
    gibbs = GibbsSampling(grounded_rpm)
    gibbs_samples = gibbs.sample(
        evidence={},
        num_samples=500,
        burn_in=100
    )
    print(f"✓ {len(gibbs_samples)} muestras generadas")
    
    print("\n[2] Metropolis-Hastings...")
    mh = MetropolisHastings(grounded_rpm)
    mh_samples = mh.sample(
        evidence={},
        num_samples=500,
        burn_in=100,
        proposal='gibbs_style'
    )
    print(f"✓ {len(mh_samples)} muestras generadas")
    
    # Comparar convergencia
    var_name = f"Quality_{books[0].book_id}"
    print(f"\n[3] Análisis de convergencia para {var_name}:")
    
    analyzer = ConvergenceAnalyzer()
    gibbs_assignments = [s.assignment for s in gibbs_samples]
    mh_assignments = [s.assignment for s in mh_samples]
    convergence_report = analyzer.compare_algorithms(
        gibbs_assignments,
        mh_assignments,
        var_name
    )
    print(convergence_report)
    
    # ========================================================================
    # FASE 5: Query Engine y Bot Detection
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 5: DETECCIÓN DE BOTS Y CONSULTAS")
    print("="*70)
    
    # Query Engine
    print("\n[1] Query Engine - Inferencia de calidades...")
    query_engine = QueryEngine(grounded_rpm)
    
    # Consultar calidad de primer libro
    quality_result = query_engine.query_marginal(
        variable=var_name,
        evidence={},
        method='gibbs',
        num_samples=300
    )
    
    plotter = DistributionPlotter()
    quality_plot = plotter.plot_distribution(
        quality_result.distribution,
        title=f"P({var_name} | Evidence)"
    )
    print(quality_plot)
    
    # Bot Detection
    print("\n[2] Bot Detection - Scoring customers...")
    detector = BotDetector(rpm, detection_threshold=0.5)
    bot_scores = detector.score_customers(
        customers,
        books,
        recommendations,
        login_ids,
        num_samples=300
    )
    
    print(f"\n✓ Scores calculados para {len(bot_scores)} customers")
    print("\nTop 5 posibles bots:")
    sorted_scores = sorted(bot_scores, key=lambda x: x.bot_probability, reverse=True)
    for i, score in enumerate(sorted_scores[:5], 1):
        print(f"  {i}. {score.customer_id}: P(bot)={score.bot_probability:.3f}, "
              f"Ground truth={score.entity_type.value}, "
              f"Predicción={score.prediction.value}, "
              f"Cuentas={score.num_accounts}")
    
    # Evaluation Metrics
    print("\n[3] Métricas de evaluación...")
    metrics = detector.evaluate(bot_scores)
    
    print(f"\n✓ Métricas:")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall: {metrics.recall:.3f}")
    print(f"  F1-Score: {metrics.f1_score:.3f}")
    print(f"  Accuracy: {metrics.accuracy:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={metrics.true_positives}, FP={metrics.false_positives}")
    print(f"    FN={metrics.false_negatives}, TN={metrics.true_negatives}")
    
    # ========================================================================
    # FASE 6: Visualización y Análisis
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 6: VISUALIZACIÓN Y ANÁLISIS")
    print("="*70)
    
    # ROC Curve
    print("\n[1] Curva ROC...")
    roc_analyzer = ROCCurveAnalyzer()
    
    score_list = [s.bot_probability for s in bot_scores]
    label_list = [s.entity_type == EntityType.BOT for s in bot_scores]
    
    thresholds, tpr_list, fpr_list = roc_analyzer.compute_roc_curve(score_list, label_list)
    auc = roc_analyzer.compute_auc(tpr_list, fpr_list)
    
    roc_plot = roc_analyzer.plot_roc_curve(tpr_list, fpr_list, auc)
    print(roc_plot)
    
    # Sybil Attack Detection
    print("\n[2] Detección de Sybil Attacks...")
    sybil_dict = detector.detect_sybil_attacks(login_ids, min_accounts=2)
    
    print(f"\n✓ {len(sybil_dict)} customers con múltiples cuentas:")
    sorted_sybils = sorted(sybil_dict.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (cust_id, accounts) in enumerate(sorted_sybils[:5], 1):
        customer = next((c for c in customers if c.customer_id == cust_id), None)
        entity_type = customer.entity_type.value if customer else "unknown"
        print(f"  {i}. {cust_id} ({entity_type}): {len(accounts)} cuentas")
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    
    print(f"\n✓ Todas las fases completadas exitosamente:")
    print(f"  FASE 1: Dataset generado ({len(customers)} customers, {len(recommendations)} ratings)")
    print(f"  FASE 2: RPM Model ({len(grounded_rpm.variables)} variables)")
    print(f"  FASE 3: OUPM Model (origin functions implementadas)")
    print(f"  FASE 4: Inferencia (Gibbs, MH, convergencia analizada)")
    print(f"  FASE 5: Bot Detection (Precision={metrics.precision:.3f}, Recall={metrics.recall:.3f})")
    print(f"  FASE 6: Visualización (AUC={auc:.3f}, {len(sybil_dict)} customers con múltiples cuentas)")
    
    print(f"\n✓ Sistema de recomendación con detección de bots operacional!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

