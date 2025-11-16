"""
Tests para FASE 2: Modelo RPM (Relational Probability Model)
"""

import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import Customer, Book, Recommendation, EntityType
from cpt import (
    CPT, RecommendationCPT, QualityCPT, HonestyCPT, 
    EntityTypeCPT, ConditionalHonestyCPT, create_default_cpts
)
from grounding import RPMGrounder, GroundedBayesNet, BayesNetVariable
from rpm_model import RPMModel, TypeSignature


def test_cpt_creation():
    """Test de creación y validación de CPTs"""
    print("\n" + "="*60)
    print("TEST 1: Creación de CPTs")
    print("="*60)
    
    # Test RecommendationCPT
    rec_cpt = RecommendationCPT()
    assert rec_cpt.validate(), "RecommendationCPT debe ser válida"
    
    # Verificar que honesto da ratings cercanos a calidad
    prob_exact = rec_cpt.get_rating_probability(4, True, 4)
    prob_far = rec_cpt.get_rating_probability(4, True, 1)
    assert prob_exact > prob_far, "Usuario honesto debe dar ratings cercanos a calidad"
    print(f"✓ P(Rating=4 | Quality=4, Honest=True) = {prob_exact:.3f}")
    print(f"✓ P(Rating=1 | Quality=4, Honest=True) = {prob_far:.3f}")
    print(f"✓ {prob_exact:.3f} > {prob_far:.3f} ✓")
    
    # Verificar que deshonesto da ratings uniformes
    probs = [rec_cpt.get_rating_probability(3, False, r) for r in [1,2,3,4,5]]
    assert all(abs(p - 0.2) < 0.01 for p in probs), "Deshonesto debe dar ratings uniformes"
    print(f"✓ P(Rating | Quality=3, Honest=False) es uniforme: {probs}")
    
    # Test QualityCPT
    quality_cpt = QualityCPT(prior_type="uniform")
    assert quality_cpt.validate()
    assert abs(quality_cpt.get_quality_probability(3) - 0.2) < 0.01
    print(f"✓ QualityCPT uniforme: P(Quality=3) = {quality_cpt.get_quality_probability(3):.3f}")
    
    # Test HonestyCPT
    honesty_cpt = HonestyCPT(honest_prob=0.7)
    assert honesty_cpt.validate()
    assert abs(honesty_cpt.get_honesty_probability(True) - 0.7) < 0.01
    print(f"✓ HonestyCPT: P(Honest=True) = {honesty_cpt.get_honesty_probability(True):.3f}")
    
    print("✓ Test de CPTs: PASSED")


def test_grounding():
    """Test de grounding RPM → Bayes Net"""
    print("\n" + "="*60)
    print("TEST 2: Grounding RPM → Bayes Net")
    print("="*60)
    
    # Crear objetos
    customers = [
        Customer("User_1", EntityType.REAL_USER, 0.9),
        Customer("User_2", EntityType.REAL_USER, 0.7)
    ]
    
    books = [
        Book("Book_1", true_quality=4),
        Book("Book_2", true_quality=2)
    ]
    
    recommendations = [
        Recommendation("LoginID_1", "Book_1", 4),
        Recommendation("LoginID_2", "Book_2", 2)
    ]
    
    # Ground
    cpts = create_default_cpts()
    grounder = RPMGrounder(cpts)
    net = grounder.ground_rpm(customers, books, recommendations)
    
    # Verificar estructura
    print(f"✓ Variables creadas: {len(net.variables)}")
    assert len(net.variables) > 0
    
    # Verificar que existen variables Quality
    quality_vars = [v for v in net.variables.values() if v.var_type == "quality"]
    assert len(quality_vars) == 2, f"Debe haber 2 Quality vars, hay {len(quality_vars)}"
    print(f"✓ Quality variables: {len(quality_vars)}")
    
    # Verificar que existen variables Honest
    honesty_vars = [v for v in net.variables.values() if v.var_type == "honesty"]
    assert len(honesty_vars) == 2, f"Debe haber 2 Honesty vars, hay {len(honesty_vars)}"
    print(f"✓ Honesty variables: {len(honesty_vars)}")
    
    # Verificar que existen variables Recommendation con padres
    rec_vars = [v for v in net.variables.values() if v.var_type == "recommendation"]
    assert len(rec_vars) > 0
    for rec_var in rec_vars:
        assert len(rec_var.parents) == 2, "Rec debe tener 2 padres (Quality, Honest)"
    print(f"✓ Recommendation variables: {len(rec_vars)}, cada una con 2 padres")
    
    # Verificar evidencia
    assert len(net.observations) == len(recommendations)
    print(f"✓ Evidencia establecida: {len(net.observations)} observaciones")
    
    print("✓ Test de grounding: PASSED")


def test_probability_computation():
    """Test de cálculo de probabilidades"""
    print("\n" + "="*60)
    print("TEST 3: Cálculo de Probabilidades")
    print("="*60)
    
    # Crear red simple
    customers = [Customer("User_1", EntityType.REAL_USER, 0.9)]
    books = [Book("Book_1", true_quality=5)]
    
    cpts = create_default_cpts()
    grounder = RPMGrounder(cpts)
    net = grounder.ground_rpm(customers, books)
    
    # Assignment que da alta probabilidad
    assignment_good = {
        "Quality_Book_1": 5,
        "Honest_User_1": True,
        "Rec_User_1_Book_1": 5
    }
    
    # Assignment que da baja probabilidad
    assignment_bad = {
        "Quality_Book_1": 1,
        "Honest_User_1": True,
        "Rec_User_1_Book_1": 5
    }
    
    prob_good = net.compute_probability(assignment_good)
    prob_bad = net.compute_probability(assignment_bad)
    
    print(f"P(Quality=5, Honest=True, Rec=5) = {prob_good:.6f}")
    print(f"P(Quality=1, Honest=True, Rec=5) = {prob_bad:.6f}")
    
    # Usuario honesto dando rating 5 a libro calidad 5 debe ser más probable
    # que usuario honesto dando rating 5 a libro calidad 1
    assert prob_good > prob_bad, "Assignment consistente debe tener mayor probabilidad"
    print(f"✓ {prob_good:.6f} > {prob_bad:.6f} ✓")
    
    print("✓ Test de cálculo de probabilidades: PASSED")


def test_rpm_model():
    """Test del modelo RPM completo"""
    print("\n" + "="*60)
    print("TEST 4: Modelo RPM Completo")
    print("="*60)
    
    # Crear modelo
    model = RPMModel()
    
    # Verificar type signatures
    assert "Quality" in model.type_signatures
    assert "Honest" in model.type_signatures
    assert "Recommendation" in model.type_signatures
    print(f"✓ Type signatures definidos: {len(model.type_signatures)}")
    
    # Verificar dependencies
    rec_parents = model.get_predicate_parents("Recommendation")
    assert "Quality" in rec_parents
    assert "Honest" in rec_parents
    print(f"✓ Recommendation depende de: {rec_parents}")
    
    quality_parents = model.get_predicate_parents("Quality")
    assert len(quality_parents) == 0, "Quality no debe tener padres"
    print(f"✓ Quality no tiene padres (es un prior)")
    
    # Verificar CPTs
    assert "recommendation" in model.cpts
    assert "quality" in model.cpts
    assert "honesty" in model.cpts
    print(f"✓ CPTs cargadas: {len(model.cpts)}")
    
    # Ground el modelo
    customers = [Customer("User_1", EntityType.REAL_USER)]
    books = [Book("Book_1", true_quality=4)]
    
    net = model.ground_model(customers, books)
    assert len(net.variables) > 0
    print(f"✓ Modelo grounded exitosamente: {len(net.variables)} variables")
    
    print("✓ Test de modelo RPM: PASSED")


def test_markov_blanket():
    """Test de Markov Blanket"""
    print("\n" + "="*60)
    print("TEST 5: Markov Blanket")
    print("="*60)
    
    # Crear red
    customers = [Customer("User_1", EntityType.REAL_USER)]
    books = [Book("Book_1", true_quality=4)]
    recommendations = [Recommendation("LoginID_1", "Book_1", 4)]
    
    cpts = create_default_cpts()
    grounder = RPMGrounder(cpts)
    net = grounder.ground_rpm(customers, books, recommendations)
    
    # Markov blanket de Quality_Book_1
    blanket = net.get_markov_blanket("Quality_Book_1")
    print(f"Markov Blanket de Quality_Book_1: {blanket}")
    
    # Debe incluir Rec_User_1_Book_1 (hijo) y Honest_User_1 (co-padre)
    assert "Rec_User_1_Book_1" in blanket, "Debe incluir hijo Rec"
    assert "Honest_User_1" in blanket, "Debe incluir co-padre Honest"
    print(f"✓ Markov Blanket correcto: {len(blanket)} variables")
    
    print("✓ Test de Markov Blanket: PASSED")


def run_all_tests():
    """Ejecuta todos los tests de FASE 2"""
    print("\n" + "="*70)
    print("  EJECUTANDO TESTS - FASE 2: RPM")
    print("="*70)
    
    try:
        test_cpt_creation()
        test_grounding()
        test_probability_computation()
        test_rpm_model()
        test_markov_blanket()
        
        print("\n" + "="*70)
        print("  ✓ TODOS LOS TESTS DE FASE 2 PASARON")
        print("="*70)
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FALLÓ: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
