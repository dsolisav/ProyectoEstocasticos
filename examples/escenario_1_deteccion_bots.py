"""
ESCENARIO 1: Detección de Bots
==============================

Objetivo: Demostrar que el sistema distingue correctamente entre 
usuarios reales y bots en un sistema de recomendación.

Hipótesis: Los bots tienen patrones de comportamiento distintos:
- Califican con valores extremos (1 o 5 estrellas)
- Tienen baja variabilidad en sus ratings
- Pueden usar múltiples cuentas (sybil attacks)

Salida: 3 gráficos PNG en examples/output/
- esc1_bot_scores.png
- esc1_roc_curve.png  
- esc1_confusion_matrix.png

Tiempo de ejecución: ~1 minuto
"""

import sys
import os
import random

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

random.seed(42)

from src.data_generator import DataGenerator, DatasetConfig
from src.models import EntityType
from src.rpm_model import RPMModel
from src.bot_detection import BotDetector
from src.visualization_plots import ROCPlotter, BotDetectionPlotter
from src.visualization import ROCCurveAnalyzer


def main():
    print("="*70)
    print("  ESCENARIO 1: DETECCIÓN DE BOTS")
    print("="*70)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================
    # CONFIGURACIÓN DEL EXPERIMENTO
    # =========================================
    print("\n[CONFIGURACIÓN]")
    print("  - 10 usuarios reales (honestos)")
    print("  - 5 bots (deshonestos)")
    print("  - 6 libros para calificar")
    print("  - Semilla: 42 (reproducible)")
    
    config = DatasetConfig(
        num_real_users=10,
        num_bots=5,
        num_books=6,
        prob_bot_multiple_accounts=0.7,
        random_seed=42
    )
    
    # =========================================
    # PASO 1: Generar datos sintéticos
    # =========================================
    print("\n[PASO 1] Generando datos sintéticos...")
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    num_bots = sum(1 for c in customers if c.entity_type == EntityType.BOT)
    num_users = sum(1 for c in customers if c.entity_type == EntityType.REAL_USER)
    
    print(f"  Generados: {len(customers)} customers ({num_users} usuarios, {num_bots} bots)")
    print(f"  Ratings: {len(recommendations)}")
    
    # =========================================
    # PASO 2: Construir modelo y detectar bots
    # =========================================
    print("\n[PASO 2] Construyendo modelo probabilístico...")
    rpm = RPMModel()
    grounded = rpm.ground_model(customers, books, recommendations)
    print(f"  Red bayesiana: {len(grounded.variables)} variables")
    
    print("\n[PASO 3] Ejecutando detección de bots...")
    detector = BotDetector(rpm, detection_threshold=0.5)
    bot_scores = detector.score_customers(
        customers, books, recommendations, login_ids,
        num_samples=200
    )
    print(f"  Scores calculados para {len(bot_scores)} customers")
    
    # =========================================
    # RESULTADOS
    # =========================================
    print("\n[RESULTADOS]")
    print("\n  Top 5 sospechosos de ser bots:")
    sorted_scores = sorted(bot_scores, key=lambda x: x.bot_probability, reverse=True)
    
    for i, score in enumerate(sorted_scores[:5], 1):
        real = "BOT" if score.entity_type == EntityType.BOT else "USER"
        pred = "BOT" if score.prediction == EntityType.BOT else "USER"
        check = "OK" if real == pred else "X"
        print(f"    {i}. {score.customer_id:12s} | Score={score.bot_probability:.3f} | "
              f"Real={real:4s} | Pred={pred:4s} [{check}]")
    
    # Métricas
    metrics = detector.evaluate(bot_scores)
    
    print("\n  Métricas de evaluación:")
    print(f"    Precision:  {metrics.precision:.3f}")
    print(f"    Recall:     {metrics.recall:.3f}")
    print(f"    F1-Score:   {metrics.f1_score:.3f}")
    print(f"    Accuracy:   {metrics.accuracy:.3f}")
    
    print(f"\n  Matriz de Confusión:")
    print(f"    TP={metrics.true_positives} (bots detectados correctamente)")
    print(f"    FN={metrics.false_negatives} (bots no detectados)")
    print(f"    FP={metrics.false_positives} (usuarios marcados como bots)")
    print(f"    TN={metrics.true_negatives} (usuarios identificados correctamente)")
    
    # =========================================
    # GRÁFICOS
    # =========================================
    print("\n[GRÁFICOS]")
    
    # Gráfico 1: Bot Scores
    bot_plotter = BotDetectionPlotter()
    path1 = os.path.join(output_dir, 'esc1_bot_scores.png')
    bot_plotter.plot_bot_scores(bot_scores, path1)
    print(f"  Guardado: {path1}")
    
    # Gráfico 2: Curva ROC
    roc_analyzer = ROCCurveAnalyzer()
    score_list = [s.bot_probability for s in bot_scores]
    label_list = [s.entity_type == EntityType.BOT for s in bot_scores]
    thresholds, tpr_list, fpr_list = roc_analyzer.compute_roc_curve(score_list, label_list)
    auc = roc_analyzer.compute_auc(tpr_list, fpr_list)
    
    roc_plotter = ROCPlotter()
    path2 = os.path.join(output_dir, 'esc1_roc_curve.png')
    roc_plotter.plot_roc_curve(tpr_list, fpr_list, auc, path2)
    print(f"  Guardado: {path2} (AUC={auc:.3f})")
    
    # Gráfico 3: Matriz de Confusión
    path3 = os.path.join(output_dir, 'esc1_confusion_matrix.png')
    roc_plotter.plot_confusion_matrix(metrics, path3)
    print(f"  Guardado: {path3}")
    
    # =========================================
    # CONCLUSIÓN
    # =========================================
    print("\n" + "="*70)
    print("  CONCLUSIÓN")
    print("="*70)
    
    if metrics.precision >= 0.8 and metrics.recall >= 0.8:
        conclusion = "EXITOSO"
    else:
        conclusion = "PARCIAL"
    
    print(f"""
  Resultado: {conclusion}
  
  El detector de bots logra:
  - Precision: {metrics.precision:.1%} de los detectados son realmente bots
  - Recall: {metrics.recall:.1%} de los bots fueron detectados
  - AUC: {auc:.3f} ({"Excelente" if auc > 0.9 else "Buena" if auc > 0.8 else "Aceptable"})
  
  Los bots son identificados por:
  1. Ratings extremos (solo 1 o 5 estrellas)
  2. Baja varianza en sus calificaciones
  3. Múltiples cuentas (sybil attacks)

  Gráficos generados en: examples/output/
""")


if __name__ == "__main__":
    main()
