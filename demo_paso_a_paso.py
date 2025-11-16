"""
DEMO PASO A PASO - Verificación completa del sistema
Este script ejecuta cada fase del proyecto y genera gráficos de análisis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generator import DataGenerator, DatasetConfig
from src.models import EntityType
from src.rpm_model import RPMModel
from src.oupm_model import OUPMModel
from src.bot_detection import BotDetector
from src.query_engine import QueryEngine
from src.visualization_plots import (
    NetworkPlotter,
    DistributionPlotter,
    ROCPlotter,
    ConvergencePlotter,
    BotDetectionPlotter
)
from src.inference.gibbs_sampling import GibbsSampling
from src.inference.metropolis_hastings import MetropolisHastings


def print_header(text):
    """Imprime encabezado formateado."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def pause_for_user():
    """Pausa opcional para revisar resultados."""
    input("\n[Presiona ENTER para continuar...]")


def paso_1_generar_datos():
    """PASO 1: Generación de datos sintéticos."""
    print_header("PASO 1: GENERACIÓN DE DATOS")
    
    print("\n📊 Configurando generador de datos...")
    config = DatasetConfig(
        num_real_users=12,
        num_bots=6,
        num_books=8,
        prob_bot_multiple_accounts=0.8,
        max_accounts_per_bot=8,
        max_recommendations_per_account=6,
        random_seed=42
    )
    
    print(f"  - Usuarios reales: {config.num_real_users}")
    print(f"  - Bots: {config.num_bots}")
    print(f"  - Libros: {config.num_books}")
    print(f"  - Prob. bot con múltiples cuentas: {config.prob_bot_multiple_accounts}")
    
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    print(f"\n✓ Dataset generado:")
    print(f"  Total customers: {len(customers)}")
    print(f"  - Real users: {sum(1 for c in customers if c.entity_type == EntityType.REAL_USER)}")
    print(f"  - Bots: {sum(1 for c in customers if c.entity_type == EntityType.BOT)}")
    print(f"  LoginIDs (cuentas): {len(login_ids)}")
    print(f"  Recommendations (ratings): {len(recommendations)}")
    
    # Estadísticas adicionales
    accounts_per_customer = {}
    for lid in login_ids:
        if lid.origin:
            cid = lid.origin.customer_id
            accounts_per_customer[cid] = accounts_per_customer.get(cid, 0) + 1
    
    multi_account = [cid for cid, count in accounts_per_customer.items() if count > 1]
    print(f"\n  Customers con múltiples cuentas: {len(multi_account)}")
    print(f"  Max cuentas por customer: {max(accounts_per_customer.values())}")
    
    pause_for_user()
    return customers, books, login_ids, recommendations


def paso_2_construir_modelos(customers, books, recommendations):
    """PASO 2: Construcción de modelos probabilísticos."""
    print_header("PASO 2: MODELOS PROBABILÍSTICOS (RPM & OUPM)")
    
    # RPM Model
    print("\n🔹 Construyendo RPM (Relational Probability Model)...")
    rpm = RPMModel()
    grounded_rpm = rpm.ground_model(customers, books, recommendations)
    
    print(f"✓ RPM grounded:")
    print(f"  Variables totales: {len(grounded_rpm.variables)}")
    print(f"  Observaciones: {len(grounded_rpm.observations)}")
    
    # Contar por tipo
    quality_vars = [v for v in grounded_rpm.variables.values() if v.var_type == 'quality']
    honest_vars = [v for v in grounded_rpm.variables.values() if v.var_type == 'honesty']
    rec_vars = [v for v in grounded_rpm.variables.values() if v.var_type == 'recommendation']
    
    print(f"  - Quality variables: {len(quality_vars)}")
    print(f"  - Honesty variables: {len(honest_vars)}")
    print(f"  - Recommendation variables: {len(rec_vars)}")
    
    # OUPM Model
    print("\n🔹 Construyendo OUPM (Open Universe Probability Model)...")
    oupm = OUPMModel(lambda_customers=15.0, lambda_bots=3.0)
    print(f"✓ OUPM creado:")
    print(f"  Generating functions: Poisson")
    print(f"  Origin functions: Implementadas")
    
    # Visualización
    print("\n📊 Generando gráfico de estructura de red...")
    os.makedirs("output", exist_ok=True)
    
    network_plotter = NetworkPlotter()
    network_plotter.plot_network_structure(grounded_rpm, "output/01_network_structure.png")
    
    pause_for_user()
    return rpm, grounded_rpm, oupm


def paso_3_inferencia(grounded_rpm, books):
    """PASO 3: Algoritmos de inferencia MCMC."""
    print_header("PASO 3: INFERENCIA PROBABILÍSTICA (MCMC)")
    
    print("\n🔹 Ejecutando Gibbs Sampling...")
    gibbs = GibbsSampling(grounded_rpm)
    gibbs_samples = gibbs.sample(
        evidence={},
        num_samples=500,
        burn_in=100
    )
    print(f"✓ {len(gibbs_samples)} muestras generadas (Gibbs)")
    
    print("\n🔹 Ejecutando Metropolis-Hastings...")
    mh = MetropolisHastings(grounded_rpm)
    mh_samples = mh.sample(
        evidence={},
        num_samples=500,
        burn_in=100,
        proposal='gibbs_style'
    )
    print(f"✓ {len(mh_samples)} muestras generadas (MH)")
    
    # Análisis de convergencia
    var_name = f"Quality_{books[0].book_id}"
    print(f"\n📊 Analizando convergencia para: {var_name}")
    
    convergence_plotter = ConvergencePlotter()
    gibbs_assignments = [s.assignment for s in gibbs_samples]
    mh_assignments = [s.assignment for s in mh_samples]
    
    convergence_plotter.plot_mcmc_comparison(
        gibbs_assignments,
        mh_assignments,
        var_name,
        "output/02_mcmc_convergence.png"
    )
    
    pause_for_user()
    return gibbs_samples, mh_samples


def paso_4_query_engine(grounded_rpm, books):
    """PASO 4: Query Engine - Consultas sobre el modelo."""
    print_header("PASO 4: QUERY ENGINE - CONSULTAS PROBABILÍSTICAS")
    
    print("\n🔹 Inicializando Query Engine...")
    query_engine = QueryEngine(grounded_rpm)
    
    var_name = f"Quality_{books[0].book_id}"
    print(f"\n📊 Consultando: P({var_name} | Evidence)")
    
    quality_result = query_engine.query_marginal(
        variable=var_name,
        evidence={},
        method='gibbs',
        num_samples=300
    )
    
    print(f"\n✓ Distribución inferida:")
    for value, prob in sorted(quality_result.distribution.items()):
        bar = "█" * int(prob * 50)
        print(f"  {value}: {bar} {prob:.3f}")
    
    # Visualización
    print("\n📊 Generando gráfico de distribución...")
    dist_plotter = DistributionPlotter()
    dist_plotter.plot_distribution(
        quality_result.distribution,
        title=f"P({var_name} | Evidence)",
        xlabel="Quality Level",
        output_path="output/03_quality_distribution.png"
    )
    
    # Comparar varios libros
    if len(books) >= 3:
        print("\n📊 Comparando calidades de múltiples libros...")
        distributions = {}
        for book in books[:3]:
            var_name_book = f"Quality_{book.book_id}"
            result = query_engine.query_marginal(
                variable=var_name_book,
                evidence={},
                method='gibbs',
                num_samples=200
            )
            distributions[f"{book.book_id} (True={book.true_quality})"] = result.distribution
        
        dist_plotter.plot_comparison(
            distributions,
            title="Comparación de Calidades Inferidas",
            output_path="output/04_quality_comparison.png"
        )
    
    pause_for_user()
    return query_engine


def paso_5_bot_detection(rpm, customers, books, recommendations, login_ids):
    """PASO 5: Detección de bots y sybil attacks."""
    print_header("PASO 5: DETECCIÓN DE BOTS Y SYBIL ATTACKS")
    
    print("\n🔹 Inicializando Bot Detector...")
    detector = BotDetector(rpm, detection_threshold=0.5)
    
    print("🔹 Calculando bot scores (esto puede tomar 1-2 minutos)...")
    bot_scores = detector.score_customers(
        customers,
        books,
        recommendations,
        login_ids,
        num_samples=300
    )
    
    print(f"\n✓ Scores calculados para {len(bot_scores)} customers")
    
    # Top bots
    print("\n🚨 Top 5 posibles bots:")
    sorted_scores = sorted(bot_scores, key=lambda x: x.bot_probability, reverse=True)
    for i, score in enumerate(sorted_scores[:5], 1):
        print(f"  {i}. {score.customer_id}: P(bot)={score.bot_probability:.3f}, "
              f"Truth={score.entity_type.value}, Pred={score.prediction.value}, "
              f"Cuentas={score.num_accounts}")
    
    # Métricas de evaluación
    print("\n📊 Evaluando performance del detector...")
    metrics = detector.evaluate(bot_scores)
    
    print(f"\n✓ Métricas de evaluación:")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall: {metrics.recall:.3f}")
    print(f"  F1-Score: {metrics.f1_score:.3f}")
    print(f"  Accuracy: {metrics.accuracy:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={metrics.true_positives}, FP={metrics.false_positives}")
    print(f"    FN={metrics.false_negatives}, TN={metrics.true_negatives}")
    
    # Visualizaciones
    print("\n📊 Generando gráficos de bot detection...")
    bot_plotter = BotDetectionPlotter()
    bot_plotter.plot_bot_scores(bot_scores, "output/05_bot_scores.png")
    
    # Sybil attacks
    print("\n🔹 Detectando sybil attacks...")
    sybil_dict = detector.detect_sybil_attacks(login_ids, min_accounts=2)
    
    print(f"\n✓ {len(sybil_dict)} customers con múltiples cuentas detectados:")
    for cid, accounts in sorted(sybil_dict.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        customer = next((c for c in customers if c.customer_id == cid), None)
        entity_type = customer.entity_type.value if customer else "unknown"
        print(f"  {cid} ({entity_type}): {len(accounts)} cuentas")
    
    if len(sybil_dict) > 0:
        print("\n📊 Generando gráfico de sybil attacks...")
        bot_plotter.plot_sybil_attacks(sybil_dict, customers, "output/06_sybil_attacks.png")
    
    pause_for_user()
    return bot_scores, metrics, sybil_dict


def paso_6_curva_roc(bot_scores):
    """PASO 6: Curva ROC y análisis de clasificación."""
    print_header("PASO 6: CURVA ROC Y ANÁLISIS DE CLASIFICACIÓN")
    
    print("\n🔹 Calculando curva ROC...")
    from src.visualization import ROCCurveAnalyzer
    
    roc_analyzer = ROCCurveAnalyzer()
    
    score_list = [s.bot_probability for s in bot_scores]
    label_list = [s.entity_type == EntityType.BOT for s in bot_scores]
    
    thresholds, tpr_list, fpr_list = roc_analyzer.compute_roc_curve(score_list, label_list)
    auc = roc_analyzer.compute_auc(tpr_list, fpr_list)
    
    print(f"\n✓ Curva ROC calculada:")
    print(f"  Puntos: {len(thresholds)}")
    print(f"  AUC: {auc:.4f}")
    
    if auc >= 0.9:
        classification = "EXCELENTE"
    elif auc >= 0.8:
        classification = "BUENA"
    elif auc >= 0.7:
        classification = "ACEPTABLE"
    else:
        classification = "POBRE"
    
    print(f"  Clasificación: {classification}")
    
    # Visualizaciones
    print("\n📊 Generando gráficos...")
    roc_plotter = ROCPlotter()
    roc_plotter.plot_roc_curve(tpr_list, fpr_list, auc, "output/07_roc_curve.png")
    
    # Matriz de confusión
    from src.bot_detection import BotDetector
    detector = BotDetector(None)  # Solo para evaluate
    metrics = detector.evaluate(bot_scores)
    
    roc_plotter.plot_confusion_matrix(metrics, "output/08_confusion_matrix.png")
    
    pause_for_user()
    return auc


def resumen_final(customers, recommendations, metrics, auc, sybil_dict):
    """Resumen final con todos los resultados."""
    print_header("RESUMEN FINAL - RESULTADOS DEL SISTEMA")
    
    print("\n✅ Todas las fases completadas exitosamente:\n")
    
    print("📊 DATOS:")
    print(f"  - Customers: {len(customers)}")
    print(f"  - Recommendations: {len(recommendations)}")
    
    print("\n🎯 DETECCIÓN DE BOTS:")
    print(f"  - Precision: {metrics.precision:.3f}")
    print(f"  - Recall: {metrics.recall:.3f}")
    print(f"  - F1-Score: {metrics.f1_score:.3f}")
    print(f"  - AUC: {auc:.4f}")
    
    print("\n🚨 SYBIL ATTACKS:")
    print(f"  - Detectados: {len(sybil_dict)} customers con múltiples cuentas")
    
    print("\n📁 ARCHIVOS GENERADOS:")
    output_files = [
        "01_network_structure.png",
        "02_mcmc_convergence.png",
        "03_quality_distribution.png",
        "04_quality_comparison.png",
        "05_bot_scores.png",
        "06_sybil_attacks.png",
        "07_roc_curve.png",
        "08_confusion_matrix.png"
    ]
    
    for i, filename in enumerate(output_files, 1):
        filepath = os.path.join("output", filename)
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  ✓ {filename} ({size_kb:.1f} KB)")
    
    print("\n" + "="*70)
    print("  🎉 DEMO COMPLETADO - Revisa los gráficos en la carpeta 'output/'")
    print("="*70)


def main():
    """Ejecuta el demo paso a paso completo."""
    print("\n" + "="*70)
    print("  DEMO PASO A PASO - SISTEMA DE DETECCIÓN DE BOTS")
    print("  Verificación completa con visualizaciones")
    print("="*70)
    
    print("\nEste demo ejecutará 6 pasos:")
    print("  1. Generación de datos sintéticos")
    print("  2. Construcción de modelos (RPM & OUPM)")
    print("  3. Inferencia con MCMC (Gibbs & Metropolis-Hastings)")
    print("  4. Query Engine - Consultas probabilísticas")
    print("  5. Detección de bots y sybil attacks")
    print("  6. Curva ROC y evaluación final")
    
    print("\nTiempo estimado: 3-5 minutos")
    print("\nPresiona ENTER para comenzar, o Ctrl+C para cancelar...")
    input()
    
    try:
        # Ejecutar pasos
        customers, books, login_ids, recommendations = paso_1_generar_datos()
        rpm, grounded_rpm, oupm = paso_2_construir_modelos(customers, books, recommendations)
        gibbs_samples, mh_samples = paso_3_inferencia(grounded_rpm, books)
        query_engine = paso_4_query_engine(grounded_rpm, books)
        bot_scores, metrics, sybil_dict = paso_5_bot_detection(
            rpm, customers, books, recommendations, login_ids
        )
        auc = paso_6_curva_roc(bot_scores)
        
        # Resumen
        resumen_final(customers, recommendations, metrics, auc, sybil_dict)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo cancelado por el usuario")
    except Exception as e:
        print(f"\n\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
