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
    print("\n" + "="*60)
    print("   ESCENARIO 1: Deteccion de Bots")
    print("="*60)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuracion
    print("\nConfiguracion del experimento:")
    print("  - 10 usuarios reales")
    print("  - 5 bots")
    print("  - 6 libros")
    print("  - Seed=42 para reproducibilidad")
    
    config = DatasetConfig(
        num_real_users=10,
        num_bots=5,
        num_books=6,
        prob_bot_multiple_accounts=0.7,
        random_seed=42
    )
    
    # Paso 1: Generar datos
    print("\nGenerando datos sinteticos...")
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    num_bots = sum(1 for c in customers if c.entity_type == EntityType.BOT)
    num_users = sum(1 for c in customers if c.entity_type == EntityType.REAL_USER)
    
    print(f"  {len(customers)} customers ({num_users} usuarios, {num_bots} bots)")
    print(f"  {len(recommendations)} ratings generados")
    
    # Paso 2: Construir modelo
    print("\nConstruyendo modelo RPM...")
    rpm = RPMModel()
    grounded = rpm.ground_model(customers, books, recommendations)
    print(f"  Red bayesiana con {len(grounded.variables)} variables")
    
    # Paso 3: Detectar bots
    print("\nEjecutando deteccion...")
    detector = BotDetector(rpm, detection_threshold=0.5)
    bot_scores = detector.score_customers(
        customers, books, recommendations, login_ids,
        num_samples=200
    )
    print(f"  Scores calculados para {len(bot_scores)} customers")
    
    # Resultados
    print("\n--- RESULTADOS ---")
    print("\nTop 5 sospechosos:")
    sorted_scores = sorted(bot_scores, key=lambda x: x.bot_probability, reverse=True)
    
    for i, score in enumerate(sorted_scores[:5], 1):
        real = "BOT" if score.entity_type == EntityType.BOT else "USER"
        pred = "BOT" if score.prediction == EntityType.BOT else "USER"
        check = "OK" if real == pred else "X"
        print(f"    {i}. {score.customer_id:12s} | Score={score.bot_probability:.3f} | "
              f"Real={real:4s} | Pred={pred:4s} [{check}]")
    
    # Metricas
    metrics = detector.evaluate(bot_scores)
    
    print(f"\nMetricas:")
    print(f"  Precision:  {metrics.precision:.3f}")
    print(f"  Recall:     {metrics.recall:.3f}")
    print(f"  F1-Score:   {metrics.f1_score:.3f}")
    print(f"  Accuracy:   {metrics.accuracy:.3f}")
    
    print(f"\nMatriz de confusion:")
    print(f"  TP={metrics.true_positives}, FN={metrics.false_negatives}")
    print(f"  FP={metrics.false_positives}, TN={metrics.true_negatives}")
    
    # Guardar graficos
    print("\nGenerando graficos...")
    
    bot_plotter = BotDetectionPlotter()
    path1 = os.path.join(output_dir, 'esc1_bot_scores.png')
    bot_plotter.plot_bot_scores(bot_scores, path1)
    print(f"  -> {path1}")
    
    roc_analyzer = ROCCurveAnalyzer()
    score_list = [s.bot_probability for s in bot_scores]
    label_list = [s.entity_type == EntityType.BOT for s in bot_scores]
    thresholds, tpr_list, fpr_list = roc_analyzer.compute_roc_curve(score_list, label_list)
    auc = roc_analyzer.compute_auc(tpr_list, fpr_list)
    
    roc_plotter = ROCPlotter()
    path2 = os.path.join(output_dir, 'esc1_roc_curve.png')
    roc_plotter.plot_roc_curve(tpr_list, fpr_list, auc, path2)
    print(f"  -> {path2} (AUC={auc:.3f})")
    
    path3 = os.path.join(output_dir, 'esc1_confusion_matrix.png')
    roc_plotter.plot_confusion_matrix(metrics, path3)
    print(f"  -> {path3}")
    
    # Conclusion
    print("\n" + "="*60)
    if metrics.precision >= 0.8 and metrics.recall >= 0.8:
        print("Resultado: El detector funciona correctamente")
    else:
        print("Resultado: Deteccion parcial")
    
    print(f"""  
El modelo logra distinguir bots de usuarios reales:
  - Precision={metrics.precision:.1%}, Recall={metrics.recall:.1%}
  - AUC={auc:.3f}
  
Patrones detectados en bots:
  - Ratings extremos (1 o 5)
  - Poca varianza
  - Multiples cuentas

Graficos guardados en examples/output/
""")


if __name__ == "__main__":
    main()
