"""
EJEMPLO 3: Generación de Visualizaciones
=========================================

Este script genera todos los gráficos del sistema:
- Estructura de red
- Distribuciones de probabilidad
- Bot scores
- Curva ROC
- Matriz de confusión

Tiempo de ejecución: ~2 minutos
Salida: Gráficos PNG en examples/output/
"""

import sys
import os
import random

# Agregar directorio raíz al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Fijar semilla global para reproducibilidad
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

from src.data_generator import DataGenerator, DatasetConfig
from src.rpm_model import RPMModel
from src.bot_detection import BotDetector
from src.query_engine import QueryEngine
from src.visualization_plots import (
    NetworkPlotter,
    DistributionPlotter,
    ROCPlotter,
    BotDetectionPlotter
)


def main():
    print("="*70)
    print("  EJEMPLO 3: GENERACIÓN DE VISUALIZACIONES")
    print("="*70)
    
    # Crear carpeta de salida
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar datos
    print("\n[PASO 1] Generando datos...")
    config = DatasetConfig(
        num_real_users=12,
        num_bots=6,
        num_books=8,
        random_seed=42
    )
    
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    print(f"✓ {len(customers)} customers, {len(recommendations)} recommendations")
    
    # Construir modelo
    print("\n[PASO 2] Construyendo modelo...")
    rpm = RPMModel()
    grounded = rpm.ground_model(customers, books, recommendations)
    print(f"✓ {len(grounded.variables)} variables")
    
    # GRÁFICO 1: Estructura de red
    print("\n[PASO 3] Generando gráfico de estructura de red...")
    network_plotter = NetworkPlotter()
    output_path = os.path.join(output_dir, '01_network_structure.png')
    network_plotter.plot_network_structure(grounded, output_path)
    print(f"✓ Guardado: {output_path}")
    
    # GRÁFICO 2: Distribución de calidad
    print("\n[PASO 4] Generando gráfico de distribución de calidad...")
    query_engine = QueryEngine(grounded)
    var_name = f"Quality_{books[0].book_id}"
    
    result = query_engine.query_marginal(
        variable=var_name,
        evidence={},
        method='gibbs',
        num_samples=300
    )
    
    dist_plotter = DistributionPlotter()
    output_path = os.path.join(output_dir, '02_quality_distribution.png')
    dist_plotter.plot_distribution(
        result.distribution,
        title=f"P({var_name} | Evidence)",
        xlabel="Quality Level",
        output_path=output_path
    )
    print(f"✓ Guardado: {output_path}")
    
    # GRÁFICO 3: Comparación de libros
    print("\n[PASO 5] Generando gráfico de comparación de libros...")
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
    
    output_path = os.path.join(output_dir, '03_quality_comparison.png')
    dist_plotter.plot_comparison(
        distributions,
        title="Comparación de Calidades Inferidas",
        output_path=output_path
    )
    print(f"✓ Guardado: {output_path}")
    
    # GRÁFICO 4: Bot scores
    print("\n[PASO 6] Detectando bots y generando gráfico...")
    detector = BotDetector(rpm, detection_threshold=0.5)
    bot_scores = detector.score_customers(
        customers,
        books,
        recommendations,
        login_ids,
        num_samples=300
    )
    
    bot_plotter = BotDetectionPlotter()
    output_path = os.path.join(output_dir, '04_bot_scores.png')
    bot_plotter.plot_bot_scores(bot_scores, output_path)
    print(f"✓ Guardado: {output_path}")
    
    # GRÁFICO 5: Sybil attacks
    print("\n[PASO 7] Generando gráfico de sybil attacks...")
    sybil_attacks = detector.detect_sybil_attacks(login_ids, min_accounts=2)
    
    if len(sybil_attacks) > 0:
        output_path = os.path.join(output_dir, '05_sybil_attacks.png')
        bot_plotter.plot_sybil_attacks(sybil_attacks, customers, output_path)
        print(f"✓ Guardado: {output_path}")
    else:
        print("  (No hay sybil attacks para graficar)")
    
    # GRÁFICO 6: Curva ROC
    print("\n[PASO 8] Generando curva ROC...")
    from src.visualization import ROCCurveAnalyzer
    
    roc_analyzer = ROCCurveAnalyzer()
    score_list = [s.bot_probability for s in bot_scores]
    label_list = [s.entity_type.value == 'bot' for s in bot_scores]
    
    thresholds, tpr_list, fpr_list = roc_analyzer.compute_roc_curve(score_list, label_list)
    auc = roc_analyzer.compute_auc(tpr_list, fpr_list)
    
    roc_plotter = ROCPlotter()
    output_path = os.path.join(output_dir, '06_roc_curve.png')
    roc_plotter.plot_roc_curve(tpr_list, fpr_list, auc, output_path)
    print(f"✓ Guardado: {output_path} (AUC={auc:.4f})")
    
    # GRÁFICO 7: Matriz de confusión
    print("\n[PASO 9] Generando matriz de confusión...")
    metrics = detector.evaluate(bot_scores)
    
    output_path = os.path.join(output_dir, '07_confusion_matrix.png')
    roc_plotter.plot_confusion_matrix(metrics, output_path)
    print(f"✓ Guardado: {output_path}")
    
    # Resumen
    print("\n" + "="*70)
    print("  ✓ TODOS LOS GRÁFICOS GENERADOS")
    print("="*70)
    print(f"\nArchivos creados en: {output_dir}")
    print("\nPara visualizar los gráficos:")
    print(f"  Windows: explorer {output_dir}")
    print(f"  macOS:   open {output_dir}")
    print(f"  Linux:   xdg-open {output_dir}")
    
    print("\nMétricas del modelo:")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall:    {metrics.recall:.3f}")
    print(f"  F1-Score:  {metrics.f1_score:.3f}")
    print(f"  AUC:       {auc:.3f}")


if __name__ == "__main__":
    main()
