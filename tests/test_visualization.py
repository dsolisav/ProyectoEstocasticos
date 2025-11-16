"""
Tests para FASE 6: Visualización y Análisis
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import Customer, Book, Recommendation, LoginID, EntityType
from src.data_generator import DataGenerator
from src.rpm_model import RPMModel
from src.bot_detection import BotDetector
from src.visualization import (
    BayesianNetworkVisualizer,
    DistributionPlotter,
    ROCCurveAnalyzer,
    ConvergenceAnalyzer
)
from src.inference.gibbs_sampling import GibbsSampling
from src.inference.metropolis_hastings import MetropolisHastings


def test_network_visualization():
    """Test visualización de red bayesiana."""
    print("\n" + "="*60)
    print("TEST: Visualización de Red Bayesiana")
    print("="*60)
    
    # Crear dataset pequeño
    from src.data_generator import DatasetConfig
    config = DatasetConfig(
        num_real_users=3,
        num_bots=2,
        num_books=3,
        max_recommendations_per_account=5,
        random_seed=42
    )
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    # Crear RPM
    rpm = RPMModel()
    grounded_net = rpm.ground_model(customers, books, recommendations)
    
    print(f"✓ Red construida: {len(grounded_net.variables)} variables")
    
    # Visualizar
    visualizer = BayesianNetworkVisualizer()
    output = visualizer.visualize_network(grounded_net)
    
    print(output)
    print("✓ Test de visualización de red: PASSED\n")


def test_distribution_plots():
    """Test gráficos de distribuciones."""
    print("\n" + "="*60)
    print("TEST: Gráficos de Distribuciones")
    print("="*60)
    
    plotter = DistributionPlotter(width=60, height=10)
    
    # Distribución de calidad
    quality_dist = {
        1: 0.10,
        2: 0.15,
        3: 0.35,
        4: 0.25,
        5: 0.15
    }
    
    print(plotter.plot_distribution(quality_dist, "P(Quality | Evidence)"))
    
    # Distribución binaria
    honest_dist = {
        True: 0.72,
        False: 0.28
    }
    
    print(plotter.plot_distribution(honest_dist, "P(Honest | Evidence)"))
    
    print("✓ Test de gráficos de distribuciones: PASSED\n")


def test_distribution_comparison():
    """Test comparación de distribuciones."""
    print("\n" + "="*60)
    print("TEST: Comparación de Distribuciones")
    print("="*60)
    
    plotter = DistributionPlotter()
    
    # Comparar distribuciones de dos libros
    distributions = {
        "Book_1 (True Quality=4)": {1: 0.05, 2: 0.10, 3: 0.25, 4: 0.40, 5: 0.20},
        "Book_2 (True Quality=2)": {1: 0.20, 2: 0.35, 3: 0.25, 4: 0.15, 5: 0.05}
    }
    
    output = plotter.plot_comparison(distributions, "Comparación de Calidades Inferidas")
    print(output)
    
    print("✓ Test de comparación: PASSED\n")


def test_roc_curve():
    """Test curva ROC y AUC."""
    print("\n" + "="*60)
    print("TEST: Curva ROC y AUC")
    print("="*60)
    
    # Generar dataset para bot detection
    from src.data_generator import DatasetConfig
    config = DatasetConfig(
        num_real_users=8,
        num_bots=8,
        num_books=6,
        max_recommendations_per_account=6,
        random_seed=42
    )
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    print(f"✓ Dataset: {len(customers)} customers (8 users, 8 bots)")
    
    # Construir modelo y detector
    rpm = RPMModel()
    
    detector = BotDetector(rpm)
    scores = detector.score_customers(
        customers,
        books,
        recommendations,
        login_ids,
        num_samples=200
    )
    
    print(f"✓ Scores calculados para {len(scores)} customers")
    
    # Preparar datos para ROC
    score_list = []
    label_list = []
    
    for bot_score in scores:
        score_list.append(bot_score.bot_probability)
        label_list.append(bot_score.entity_type == EntityType.BOT)
    
    # Calcular ROC
    analyzer = ROCCurveAnalyzer()
    thresholds, tpr_list, fpr_list = analyzer.compute_roc_curve(score_list, label_list)
    auc = analyzer.compute_auc(tpr_list, fpr_list)
    
    print(f"✓ Curva ROC calculada: {len(thresholds)} puntos")
    print(f"✓ AUC = {auc:.4f}")
    
    # Visualizar
    output = analyzer.plot_roc_curve(tpr_list, fpr_list, auc)
    print(output)
    
    print("✓ Test de ROC curve: PASSED\n")


def test_convergence_analysis():
    """Test análisis de convergencia MCMC."""
    print("\n" + "="*60)
    print("TEST: Análisis de Convergencia MCMC")
    print("="*60)
    
    # Crear dataset pequeño
    from src.data_generator import DatasetConfig
    config = DatasetConfig(
        num_real_users=3,
        num_bots=2,
        num_books=2,
        max_recommendations_per_account=4,
        random_seed=42
    )
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    # Construir red
    rpm = RPMModel()
    grounded_net = rpm.ground_model(customers, books, recommendations)
    
    print(f"✓ Red con {len(grounded_net.variables)} variables")
    
    # Variable a analizar
    var_name = f"Quality_{books[0].book_id}"
    print(f"✓ Analizando convergencia de: {var_name}")
    
    # Gibbs sampling
    gibbs = GibbsSampling(grounded_net)
    gibbs_samples = gibbs.sample(
        num_samples=400,
        burn_in=100,
        evidence={}
    )
    
    print(f"✓ Gibbs: {len(gibbs_samples)} muestras")
    
    # Metropolis-Hastings
    mh = MetropolisHastings(grounded_net)
    mh_samples = mh.sample(
        evidence={},
        num_samples=400,
        burn_in=100,
        proposal='gibbs_style'
    )
    
    print(f"✓ MH: {len(mh_samples)} muestras")
    
    # Extraer assignments (ambos retornan Sample objects con atributo assignment)
    gibbs_assignments = [s.assignment for s in gibbs_samples]
    mh_assignments = [s.assignment for s in mh_samples]
    
    # Comparar
    analyzer = ConvergenceAnalyzer()
    output = analyzer.compare_algorithms(gibbs_assignments, mh_assignments, var_name)
    print(output)
    
    print("✓ Test de convergencia: PASSED\n")


def main():
    print("\n" + "="*60)
    print("  EJECUTANDO TESTS - FASE 6: VISUALIZACIÓN")
    print("="*60)
    
    test_network_visualization()
    test_distribution_plots()
    test_distribution_comparison()
    test_roc_curve()
    test_convergence_analysis()
    
    print("\n" + "="*60)
    print("  ✓ TODOS LOS TESTS DE FASE 6 PASARON")
    print("="*60)


if __name__ == "__main__":
    main()
