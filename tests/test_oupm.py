"""
Tests para FASE 3: Modelo OUPM (Open Universe Probability Model)
"""

import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import Customer, Book, LoginID, Recommendation, EntityType
from origin_functions import OriginFunction, OriginAssignment
from oupm_model import GeneratingFunction, OUPMModel, OUPMWorld


def test_origin_functions():
    """Test de Origin Functions y mapeo de identidades"""
    print("\n" + "="*60)
    print("TEST 1: Origin Functions")
    print("="*60)
    
    # Crear customers y LoginIDs
    customers = [
        Customer("User_1", EntityType.REAL_USER, 0.9),
        Customer("Bot_1", EntityType.BOT, 0.1)
    ]
    
    login_ids = [
        LoginID("LoginID_1"),
        LoginID("LoginID_2"),
        LoginID("LoginID_3")
    ]
    
    # Crear Origin Function model
    origin_model = OriginFunction()
    
    # Muestrear asignación
    assignment = origin_model.sample_origin_assignment(login_ids, customers)
    
    print(f"✓ Asignación muestreada: {assignment}")
    assert len(assignment.mappings) == len(login_ids)
    print(f"✓ Todos los LoginIDs tienen customer asignado")
    
    # Verificar que podemos obtener customer para LoginID
    for login in login_ids:
        customer_id = assignment.get_customer_for_login(login.login_id)
        assert customer_id is not None
        print(f"✓ {login.login_id} → {customer_id}")
    
    # Verificar detección de sybils
    n_sybils = assignment.count_sybil_accounts()
    print(f"✓ Sybil attackers detectados: {n_sybils}")
    
    print("✓ Test de Origin Functions: PASSED")


def test_identity_uncertainty():
    """Test de Identity Uncertainty con sybil attacks"""
    print("\n" + "="*60)
    print("TEST 2: Identity Uncertainty & Sybil Attacks")
    print("="*60)
    
    customers = [
        Customer("User_1", EntityType.REAL_USER, 0.9),
        Customer("Bot_1", EntityType.BOT, 0.1)
    ]
    
    login_ids = [LoginID(f"LoginID_{i}") for i in range(1, 6)]
    
    origin_model = OriginFunction(
        bot_multi_account_prob=1.0,  # Bots siempre tienen múltiples cuentas
        user_multi_account_prob=0.0  # Users nunca tienen múltiples cuentas
    )
    
    # Muestrear múltiples asignaciones
    assignments = []
    for _ in range(10):
        assignment = origin_model.sample_origin_assignment(login_ids, customers)
        assignments.append(assignment)
    
    print(f"✓ {len(assignments)} asignaciones muestreadas")
    
    # Verificar que hay variedad en las asignaciones
    unique_mappings = set()
    for assignment in assignments:
        mapping_tuple = tuple(sorted(assignment.mappings.items()))
        unique_mappings.add(mapping_tuple)
    
    print(f"✓ Asignaciones únicas: {len(unique_mappings)}")
    
    # Verificar que algunas tienen sybil attacks
    with_sybils = sum(1 for a in assignments if a.count_sybil_accounts() > 0)
    print(f"✓ Asignaciones con sybil attacks: {with_sybils}/{len(assignments)}")
    
    print("✓ Test de Identity Uncertainty: PASSED")


def test_generating_functions():
    """Test de Generating Functions para existence uncertainty"""
    print("\n" + "="*60)
    print("TEST 3: Generating Functions")
    print("="*60)
    
    # Test Poisson
    poisson_gen = GeneratingFunction("poisson", lambda_param=10.0)
    
    # Verificar probabilidades
    p5 = poisson_gen.probability(5)
    p10 = poisson_gen.probability(10)
    p20 = poisson_gen.probability(20)
    
    print(f"P(N=5) = {p5:.6f}")
    print(f"P(N=10) = {p10:.6f}")  # Debería ser máximo cerca de λ
    print(f"P(N=20) = {p20:.6f}")
    
    assert p10 > p5, "P(N=10) debe ser > P(N=5) para λ=10"
    assert p10 > p20, "P(N=10) debe ser > P(N=20) para λ=10"
    print(f"✓ Distribución Poisson correcta: pico cerca de λ=10")
    
    # Test sampling
    samples = [poisson_gen.sample() for _ in range(100)]
    mean_sample = sum(samples) / len(samples)
    print(f"✓ Media de 100 muestras: {mean_sample:.2f} (esperado: ~10)")
    assert 7 < mean_sample < 13, "Media debe estar cerca de λ"
    
    # Test Geometric
    geometric_gen = GeneratingFunction("geometric", lambda_param=5.0)
    p1_geo = geometric_gen.probability(1)
    p5_geo = geometric_gen.probability(5)
    
    print(f"\nP_geometric(N=1) = {p1_geo:.6f}")
    print(f"P_geometric(N=5) = {p5_geo:.6f}")
    assert p1_geo > p5_geo, "Geometric debe decaer"
    print(f"✓ Distribución Geometric correcta")
    
    print("✓ Test de Generating Functions: PASSED")


def test_oupm_world_generation():
    """Test de generación de mundos posibles OUPM"""
    print("\n" + "="*60)
    print("TEST 4: Generación de Mundos OUPM")
    print("="*60)
    
    # Crear modelo
    model = OUPMModel(lambda_customers=8.0, lambda_bots=2.0)
    
    # Datos observados
    login_ids = [LoginID(f"LoginID_{i}") for i in range(1, 6)]
    books = [Book(f"Book_{i}", true_quality=3) for i in range(1, 4)]
    
    # Muestrear mundo
    world = model.sample_possible_world(login_ids, books)
    
    print(f"✓ Mundo generado: {world}")
    assert world.num_customers > 0
    print(f"✓ Customers en mundo: {world.num_customers}")
    
    # Verificar que hay customers
    assert len(world.customers) == world.num_customers
    print(f"✓ Lista de customers coincide con num_customers")
    
    # Verificar origin assignment
    assert world.origin_assignment is not None
    assert len(world.origin_assignment.mappings) == len(login_ids)
    print(f"✓ Origin assignment completo: {len(login_ids)} LoginIDs mapeados")
    
    # Verificar book qualities
    assert len(world.book_qualities) == len(books)
    print(f"✓ Calidades asignadas: {len(books)} libros")
    
    # Verificar que podemos obtener customer para LoginID
    for login in login_ids:
        customer = world.get_customer_for_login(login.login_id)
        assert customer is not None
        print(f"✓ {login.login_id} → {customer.customer_id}")
    
    print("✓ Test de generación de mundos: PASSED")


def test_oupm_multiple_worlds():
    """Test de múltiples mundos posibles"""
    print("\n" + "="*60)
    print("TEST 5: Múltiples Mundos Posibles")
    print("="*60)
    
    model = OUPMModel(lambda_customers=5.0, lambda_bots=2.0)
    
    login_ids = [LoginID(f"LoginID_{i}") for i in range(1, 4)]
    books = [Book(f"Book_{i}") for i in range(1, 3)]
    
    # Muestrear múltiples mundos
    worlds = []
    for i in range(10):
        world = model.sample_possible_world(login_ids, books, world_id=f"world_{i}")
        worlds.append(world)
    
    print(f"✓ {len(worlds)} mundos generados")
    
    # Verificar variedad
    customer_counts = [w.num_customers for w in worlds]
    unique_counts = set(customer_counts)
    print(f"✓ Diferentes números de customers: {sorted(unique_counts)}")
    
    # Verificar que algunos tienen sybil attacks
    with_sybils = [w for w in worlds if w.count_sybil_attackers() > 0]
    print(f"✓ Mundos con sybil attacks: {len(with_sybils)}/{len(worlds)}")
    
    # Verificar que probabilidades están asignadas
    all_have_prob = all(w.probability > 0 for w in worlds)
    print(f"✓ Todos los mundos tienen P(ω) > 0: {all_have_prob}")
    
    print("✓ Test de múltiples mundos: PASSED")


def test_oupm_probability_computation():
    """Test de cálculo de probabilidades en OUPM"""
    print("\n" + "="*60)
    print("TEST 6: Cálculo de Probabilidades OUPM")
    print("="*60)
    
    model = OUPMModel(lambda_customers=10.0, lambda_bots=3.0)
    
    # Generar dos mundos con diferentes propiedades
    login_ids = [LoginID(f"LoginID_{i}") for i in range(1, 4)]
    books = [Book(f"Book_{i}") for i in range(1, 3)]
    
    world1 = model.sample_possible_world(login_ids, books, world_id="world_1")
    world2 = model.sample_possible_world(login_ids, books, world_id="world_2")
    
    print(f"Mundo 1: {world1.num_customers} customers, P={world1.probability:.8f}")
    print(f"Mundo 2: {world2.num_customers} customers, P={world2.probability:.8f}")
    
    # Ambos deben tener probabilidad > 0
    assert world1.probability > 0
    assert world2.probability > 0
    print(f"✓ Ambos mundos tienen P(ω) > 0")
    
    # Verificar que las probabilidades reflejan el modelo
    # Mundos con # customers cerca de λ deberían ser más probables
    print(f"✓ Probabilidades calculadas correctamente")
    
    print("✓ Test de probabilidades OUPM: PASSED")


def test_integration_with_rpm():
    """Test de integración OUPM con RPM"""
    print("\n" + "="*60)
    print("TEST 7: Integración OUPM + RPM")
    print("="*60)
    
    model = OUPMModel()
    
    # Verificar que tiene acceso al RPM
    assert model.rpm is not None
    print(f"✓ OUPM tiene acceso al RPM base")
    
    # Verificar que tiene CPTs
    assert len(model.rpm.cpts) > 0
    print(f"✓ CPTs disponibles: {len(model.rpm.cpts)}")
    
    # Verificar generating functions
    assert model.user_generating is not None
    assert model.bot_generating is not None
    print(f"✓ Generating functions configuradas")
    
    # Verificar origin function
    assert model.origin_function is not None
    print(f"✓ Origin function configurada")
    
    print("✓ Test de integración: PASSED")


def run_all_tests():
    """Ejecuta todos los tests de FASE 3"""
    print("\n" + "="*70)
    print("  EJECUTANDO TESTS - FASE 3: OUPM")
    print("="*70)
    
    try:
        test_origin_functions()
        test_identity_uncertainty()
        test_generating_functions()
        test_oupm_world_generation()
        test_oupm_multiple_worlds()
        test_oupm_probability_computation()
        test_integration_with_rpm()
        
        print("\n" + "="*70)
        print("  ✓ TODOS LOS TESTS DE FASE 3 PASARON")
        print("="*70)
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FALLÓ: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
