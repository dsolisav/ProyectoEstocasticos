"""
Tests para FASE 5: Query Engine y Bot Detection

Validación de:
- Query engine para consultas probabilísticas
- Bot detection y scoring
- Quality inference
- Métricas de evaluación
"""

import sys
import os
import io

# Configurar encoding UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import Customer, Book, LoginID, Recommendation, EntityType
from src.data_generator import DataGenerator, DatasetConfig
from src.rpm_model import RPMModel
from src.query_engine import QueryEngine
from src.bot_detection import BotDetector, QualityEstimator


def print_header(title: str):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def print_test(test_name: str):
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print('='*60)


def test_query_engine_marginal():
    """Test 1: Query Engine - Consultas marginales."""
    print_test("Query Engine - Marginal Queries")
    
    # Crear dataset pequeño
    config = DatasetConfig(
        num_real_users=3,
        num_bots=2,
        num_books=2,
        prob_user_multiple_accounts=0.0,
        prob_bot_multiple_accounts=0.0
    )
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    # Ground modelo
    rpm = RPMModel()
    grounded = rpm.ground_model(customers[:3], books[:2], recommendations[:10])
    
    print(f"✓ Red con {len(grounded.variables)} variables")
    
    # Crear query engine
    query_engine = QueryEngine(grounded)
    
    # Query: P(Quality(book))
    quality_vars = [v for v in grounded.variables.keys() if 'Quality' in v]
    if quality_vars:
        quality_var = quality_vars[0]
        result = query_engine.query_marginal(
            variable=quality_var,
            evidence={},
            method='gibbs',
            num_samples=500
        )
        
        print(f"✓ Query: P({quality_var})")
        for value, prob in sorted(result.distribution.items()):
            print(f"    {value}: {prob:.4f}")
        
        total = sum(result.distribution.values())
        print(f"✓ Suma de probabilidades: {total:.6f}")
        assert abs(total - 1.0) < 1e-5
        
        # MAP query
        map_value = query_engine.query_map(quality_var, evidence={}, num_samples=500)
        print(f"✓ MAP: {quality_var} = {map_value}")
        
        # Expectation
        exp_value = query_engine.query_expectation(quality_var, evidence={}, num_samples=500)
        print(f"✓ E[{quality_var}] = {exp_value:.3f}")
    
    print("✓ Test de Query Engine marginal: PASSED\n")


def test_query_engine_conditional():
    """Test 2: Query Engine - Consultas condicionales."""
    print_test("Query Engine - Conditional Queries")
    
    # Crear dataset
    config = DatasetConfig(
        num_real_users=2,
        num_bots=1,
        num_books=2,
        prob_user_multiple_accounts=0.0,
        prob_bot_multiple_accounts=0.0
    )
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    # Ground modelo
    rpm = RPMModel()
    grounded = rpm.ground_model(customers[:2], books[:1], recommendations[:5])
    
    # Query engine
    query_engine = QueryEngine(grounded)
    
    # Query con evidencia
    honest_vars = [v for v in grounded.variables.keys() if 'Honest' in v]
    if honest_vars:
        honest_var = honest_vars[0]
        
        # P(Honest = True)
        prob_honest_true = query_engine.query_conditional_probability(
            variable=honest_var,
            value=True,
            evidence={},
            num_samples=500
        )
        
        print(f"✓ P({honest_var} = True) = {prob_honest_true:.4f}")
        
        # P(Honest = False)
        prob_honest_false = query_engine.query_conditional_probability(
            variable=honest_var,
            value=False,
            evidence={},
            num_samples=500
        )
        
        print(f"✓ P({honest_var} = False) = {prob_honest_false:.4f}")
        
        # Debe sumar ~1
        total = prob_honest_true + prob_honest_false
        print(f"✓ Suma: {total:.4f}")
        assert abs(total - 1.0) < 0.1  # Margen más amplio para MCMC
    
    print("✓ Test de Query Engine condicional: PASSED\n")


def test_batch_queries():
    """Test 3: Batch queries."""
    print_test("Query Engine - Batch Queries")
    
    # Crear dataset
    config = DatasetConfig(
        num_real_users=2,
        num_bots=1,
        num_books=2,
        prob_user_multiple_accounts=0.0,
        prob_bot_multiple_accounts=0.0
    )
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    # Ground modelo
    rpm = RPMModel()
    grounded = rpm.ground_model(customers[:2], books[:2], recommendations[:8])
    
    # Query engine
    query_engine = QueryEngine(grounded)
    
    # Batch query para todas las variables Quality
    quality_vars = [v for v in grounded.variables.keys() if 'Quality' in v]
    
    if quality_vars:
        results = query_engine.batch_query(
            variables=quality_vars[:2],  # Solo las primeras 2
            evidence={},
            method='gibbs',
            num_samples=500
        )
        
        print(f"✓ Batch query ejecutado para {len(results)} variables")
        
        for var, result in results.items():
            map_val = max(result.distribution.items(), key=lambda x: x[1])[0]
            print(f"  {var}: MAP = {map_val}")
    
    print("✓ Test de batch queries: PASSED\n")


def test_bot_detection_scoring():
    """Test 4: Bot detection scoring."""
    print_test("Bot Detection - Scoring")
    
    # Crear dataset con bots conocidos
    config = DatasetConfig(
        num_real_users=5,
        num_bots=3,
        num_books=3,
        prob_user_multiple_accounts=0.2,
        prob_bot_multiple_accounts=0.8
    )
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    print(f"✓ Dataset: {len(customers)} customers ({sum(1 for c in customers if c.entity_type == EntityType.REAL_USER)} users, {sum(1 for c in customers if c.entity_type == EntityType.BOT)} bots)")
    
    # Bot detector
    rpm = RPMModel()
    detector = BotDetector(rpm, detection_threshold=0.5)
    
    # Score customers (usar menos samples para tests rápidos)
    scores = detector.score_customers(
        customers=customers,
        books=books,
        recommendations=recommendations,
        login_ids=login_ids,
        num_samples=500
    )
    
    print(f"✓ Scores calculados para {len(scores)} customers")
    
    # Mostrar top 3 más probables de ser bots
    print("\n  Top 3 posibles bots:")
    for i, score in enumerate(scores[:3]):
        print(f"    {i+1}. {score.customer_id}: P(bot)={score.bot_probability:.3f}, "
              f"Ground truth={score.entity_type.name}, "
              f"Predicción={score.prediction.name}")
    
    # Verificar que todos tienen scores entre 0 y 1
    for score in scores:
        assert 0.0 <= score.bot_probability <= 1.0
    
    print("\n✓ Test de bot scoring: PASSED\n")


def test_sybil_attack_detection():
    """Test 5: Sybil attack detection."""
    print_test("Sybil Attack Detection")
    
    # Crear dataset con sybil attacks
    config = DatasetConfig(
        num_real_users=4,
        num_bots=2,
        num_books=2,
        prob_user_multiple_accounts=0.5,
        prob_bot_multiple_accounts=1.0  # Todos los bots tienen múltiples cuentas
    )
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    print(f"✓ Dataset: {len(login_ids)} LoginIDs para {len(customers)} customers")
    
    # Detector
    rpm = RPMModel()
    detector = BotDetector(rpm)
    
    # Detectar sybil attacks
    sybil_attacks = detector.detect_sybil_attacks(login_ids, min_accounts=2)
    
    print(f"✓ Sybil attacks detectados: {len(sybil_attacks)}")
    
    for customer_id, accounts in sybil_attacks.items():
        customer = next((c for c in customers if c.customer_id == customer_id), None)
        if customer:
            print(f"  {customer_id} ({customer.entity_type.name}): {len(accounts)} cuentas")
    
    # Verificar que detectamos al menos un sybil attack
    assert len(sybil_attacks) > 0, "Debe detectar al menos un sybil attack"
    
    print("✓ Test de sybil detection: PASSED\n")


def test_evaluation_metrics():
    """Test 6: Métricas de evaluación."""
    print_test("Evaluation Metrics")
    
    # Crear dataset balanceado
    config = DatasetConfig(
        num_real_users=10,
        num_bots=10,
        num_books=5,
        prob_user_multiple_accounts=0.1,
        prob_bot_multiple_accounts=0.9
    )
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    print(f"✓ Dataset: {len(customers)} customers (balanceado)")
    
    # Bot detector
    rpm = RPMModel()
    detector = BotDetector(rpm, detection_threshold=0.5)
    
    # Score customers
    scores = detector.score_customers(
        customers=customers,
        books=books,
        recommendations=recommendations,
        login_ids=login_ids,
        num_samples=500
    )
    
    # Evaluar
    metrics = detector.evaluate(scores)
    
    print(f"\n✓ Métricas de evaluación:")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall: {metrics.recall:.3f}")
    print(f"  F1-Score: {metrics.f1_score:.3f}")
    print(f"  Accuracy: {metrics.accuracy:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={metrics.true_positives}, FP={metrics.false_positives}")
    print(f"    FN={metrics.false_negatives}, TN={metrics.true_negatives}")
    
    # Verificar que las métricas están en rango válido
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1_score <= 1.0
    assert 0.0 <= metrics.accuracy <= 1.0
    
    print("\n✓ Test de evaluation metrics: PASSED\n")


def test_quality_estimation():
    """Test 7: Estimación de calidad de libros."""
    print_test("Quality Estimation")
    
    # Crear dataset
    config = DatasetConfig(
        num_real_users=5,
        num_bots=2,
        num_books=3,
        prob_user_multiple_accounts=0.0,
        prob_bot_multiple_accounts=0.5
    )
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    print(f"✓ Dataset: {len(books)} books, {len(recommendations)} recommendations")
    
    # Quality estimator
    rpm = RPMModel()
    estimator = QualityEstimator(rpm)
    
    # Estimar calidades
    quality_dists = estimator.estimate_book_qualities(
        customers=customers,
        books=books,
        recommendations=recommendations,
        num_samples=500
    )
    
    print(f"\n✓ Calidades estimadas para {len(quality_dists)} libros:")
    
    for book_id, dist in quality_dists.items():
        book = next((b for b in books if b.book_id == book_id), None)
        map_quality = estimator.get_map_quality(dist)
        exp_quality = estimator.get_expected_quality(dist)
        
        if book:
            print(f"  {book_id}: MAP={map_quality}, E[Q]={exp_quality:.2f}, "
                  f"True={book.true_quality}")
    
    # Verificar que todas las distribuciones suman ~1
    for book_id, dist in quality_dists.items():
        total = sum(dist.values())
        assert abs(total - 1.0) < 0.01, f"Distribution for {book_id} must sum to 1"
    
    print("\n✓ Test de quality estimation: PASSED\n")


def main():
    print_header("EJECUTANDO TESTS - FASE 5: QUERY ENGINE & BOT DETECTION")
    
    test_query_engine_marginal()
    test_query_engine_conditional()
    test_batch_queries()
    test_bot_detection_scoring()
    test_sybil_attack_detection()
    test_evaluation_metrics()
    test_quality_estimation()
    
    print_header("✓ TODOS LOS TESTS DE FASE 5 PASARON")


if __name__ == "__main__":
    main()
