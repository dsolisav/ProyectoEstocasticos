"""
Tests básicos para verificar el funcionamiento de los modelos.
"""

import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import Customer, Book, LoginID, Recommendation, EntityType, PossibleWorld
from data_generator import DataGenerator, DatasetConfig
from utils import normalize_distribution, entropy, compute_statistics


def test_models():
    """Test de creación de modelos básicos"""
    print("\n" + "="*60)
    print("TEST 1: Modelos Básicos")
    print("="*60)
    
    # Crear customer
    customer = Customer(
        customer_id="User_1",
        entity_type=EntityType.REAL_USER,
        honesty=0.8
    )
    print(f"✓ Customer creado: {customer}")
    
    # Crear book
    book = Book(
        book_id="Book_1",
        true_quality=4,
        title="Probabilistic Programming"
    )
    print(f"✓ Book creado: {book}")
    
    # Crear LoginID
    login = LoginID(login_id="LoginID_1", origin=customer)
    print(f"✓ LoginID creado: {login}")
    
    # Crear recommendation
    rec = login.add_recommendation(book_id="Book_1", rating=5)
    print(f"✓ Recommendation creada: {rec}")
    
    assert len(login.recommendations) == 1
    print("✓ Test de modelos básicos: PASSED")


def test_data_generation():
    """Test de generación de datos"""
    print("\n" + "="*60)
    print("TEST 2: Generación de Datos")
    print("="*60)
    
    config = DatasetConfig(
        num_real_users=10,
        num_bots=3,
        num_books=8,
        random_seed=42
    )
    
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    print(f"✓ Customers generados: {len(customers)}")
    print(f"✓ Books generados: {len(books)}")
    print(f"✓ LoginIDs generados: {len(login_ids)}")
    print(f"✓ Recommendations generadas: {len(recommendations)}")
    
    # Verificar que hay usuarios reales y bots
    num_real = sum(1 for c in customers if c.entity_type == EntityType.REAL_USER)
    num_bots = sum(1 for c in customers if c.entity_type == EntityType.BOT)
    
    assert num_real == 10
    assert num_bots == 3
    print(f"✓ Usuarios reales: {num_real}, Bots: {num_bots}")
    
    # Verificar sybil attacks (algunos customers tienen múltiples LoginIDs)
    customers_with_multiple = sum(
        1 for c in customers 
        if sum(1 for l in login_ids if l.origin == c) > 1
    )
    print(f"✓ Customers con múltiples cuentas: {customers_with_multiple}")
    
    print("✓ Test de generación de datos: PASSED")


def test_utils():
    """Test de utilidades"""
    print("\n" + "="*60)
    print("TEST 3: Utilidades")
    print("="*60)
    
    # Test normalize
    dist = {"a": 2, "b": 3, "c": 5}
    normalized = normalize_distribution(dist)
    total = sum(normalized.values())
    print(f"✓ Normalización: suma = {total:.6f}")
    assert abs(total - 1.0) < 1e-6
    
    # Test entropy
    uniform = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    h = entropy(uniform)
    print(f"✓ Entropía uniforme (4 valores): {h:.3f} bits (esperado: 2.0)")
    assert abs(h - 2.0) < 0.01
    
    # Test statistics
    values = [1, 2, 3, 4, 5]
    stats = compute_statistics(values)
    print(f"✓ Estadísticas: mean={stats['mean']}, std={stats['std']:.2f}")
    assert stats['mean'] == 3.0
    
    print("✓ Test de utilidades: PASSED")


def test_possible_world():
    """Test de mundos posibles"""
    print("\n" + "="*60)
    print("TEST 4: Mundos Posibles")
    print("="*60)
    
    # Crear customers
    user1 = Customer("User_1", EntityType.REAL_USER, 0.9)
    bot1 = Customer("Bot_1", EntityType.BOT, 0.1)
    
    # Crear mundo posible
    world = PossibleWorld(
        world_id="world_1",
        customers=[user1, bot1],
        login_mappings={
            "LoginID_1": "User_1",
            "LoginID_2": "User_1",  # Sybil: misma persona
            "LoginID_3": "Bot_1"
        },
        book_qualities={"Book_1": 5, "Book_2": 3},
        customer_honesties={"User_1": 0.9, "Bot_1": 0.1},
        probability=0.15
    )
    
    print(f"✓ Mundo posible creado: {world}")
    print(f"✓ Usuarios reales: {world.count_real_users()}")
    print(f"✓ Bots: {world.count_bots()}")
    
    # Test get_customer_for_login
    customer = world.get_customer_for_login("LoginID_1")
    print(f"✓ LoginID_1 → {customer.customer_id}")
    assert customer.customer_id == "User_1"
    
    customer2 = world.get_customer_for_login("LoginID_2")
    assert customer2.customer_id == "User_1"  # Mismo customer
    print(f"✓ LoginID_2 → {customer2.customer_id} (sybil attack detectado)")
    
    print("✓ Test de mundos posibles: PASSED")


def run_all_tests():
    """Ejecuta todos los tests"""
    print("\n" + "="*70)
    print("  EJECUTANDO TESTS - FASE 1")
    print("="*70)
    
    try:
        test_models()
        test_data_generation()
        test_utils()
        test_possible_world()
        
        print("\n" + "="*70)
        print("  ✓ TODOS LOS TESTS PASARON")
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
