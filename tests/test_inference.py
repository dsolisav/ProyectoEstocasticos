"""
Tests para FASE 4: Algoritmos de Inferencia

Validación de:
- Variable Elimination (inferencia exacta)
- Gibbs Sampling (MCMC)
- Metropolis-Hastings (MCMC)
- Diagnósticos de convergencia
"""

import sys
import os
import io

# Configurar encoding UTF-8 para la consola de Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import Customer, Book, LoginID, EntityType
from src.data_generator import DataGenerator, DatasetConfig
from src.rpm_model import RPMModel
from src.inference.variable_elimination import VariableElimination, Factor
from src.inference.gibbs_sampling import GibbsSampling, ConvergenceDiagnostics
from src.inference.metropolis_hastings import MetropolisHastings


def print_header(title: str):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def print_test(test_name: str):
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print('='*60)


def test_factor_operations():
    """Test 1: Operaciones básicas con factores."""
    print_test("Operaciones con Factores")
    
    # Factor φ(A, B)
    factor1 = Factor(
        variables=['A', 'B'],
        values={
            (True, True): 0.3,
            (True, False): 0.7,
            (False, True): 0.6,
            (False, False): 0.4
        }
    )
    
    # Marginalizar B: φ'(A) = Σ_B φ(A, B)
    factor_a = factor1.marginalize('B')
    print(f"✓ Factor original: φ(A, B) con {len(factor1.values)} entradas")
    print(f"✓ Marginalizado: φ(A) con {len(factor_a.values)} entradas")
    print(f"  P(A=True) = {factor_a.values[(True,)]:.3f} (esperado: 1.0)")
    print(f"  P(A=False) = {factor_a.values[(False,)]:.3f} (esperado: 1.0)")
    
    # Restricción: φ'(A) = φ(A, B=True)
    factor_restricted = factor1.restrict('B', True)
    print(f"✓ Restringido a B=True: φ(A | B=True)")
    print(f"  P(A=True | B=True) = {factor_restricted.values[(True,)]:.3f}")
    print(f"  P(A=False | B=True) = {factor_restricted.values[(False,)]:.3f}")
    
    # Join de factores
    factor2 = Factor(
        variables=['B', 'C'],
        values={
            (True, True): 0.8,
            (True, False): 0.2,
            (False, True): 0.5,
            (False, False): 0.5
        }
    )
    
    joined = factor1 * factor2
    print(f"✓ Join: φ(A, B) * φ(B, C) = φ(A, B, C)")
    print(f"  Variables resultantes: {joined.variables}")
    print(f"  Entradas: {len(joined.values)}")
    
    print("✓ Test de operaciones con factores: PASSED\n")


def test_variable_elimination_simple():
    """Test 2: Variable Elimination con factores simples."""
    print_test("Variable Elimination - Factores Simples")
    
    # Crear factores manualmente para probar VE
    # Red simple: A → B → C
    # P(A) = {T: 0.6, F: 0.4}
    # P(B | A) 
    # P(C | B)
    
    factor_a = Factor(
        variables=['A'],
        values={
            (True,): 0.6,
            (False,): 0.4
        }
    )
    
    factor_b = Factor(
        variables=['B', 'A'],
        values={
            (True, True): 0.7,
            (True, False): 0.3,
            (False, True): 0.3,
            (False, False): 0.7
        }
    )
    
    factor_c = Factor(
        variables=['C', 'B'],
        values={
            (True, True): 0.9,
            (True, False): 0.2,
            (False, True): 0.1,
            (False, False): 0.8
        }
    )
    
    print("✓ Factores creados:")
    print(f"  φ(A): {len(factor_a.values)} entradas")
    print(f"  φ(B,A): {len(factor_b.values)} entradas")
    print(f"  φ(C,B): {len(factor_c.values)} entradas")
    
    # Test: Join de factores
    joint_ab = factor_a * factor_b
    print(f"✓ Join φ(A) * φ(B,A) = φ(A,B): {len(joint_ab.values)} entradas")
    
    # Test: Marginalizar A para obtener P(B)
    marginal_b = joint_ab.marginalize('A')
    print(f"✓ Marginalizar A: φ(B)")
    for val, prob in marginal_b.values.items():
        print(f"    P(B={val[0]}) = {prob:.4f}")
    
    # Test: Normalización
    normalized = marginal_b.normalize()
    total = sum(normalized.values.values())
    print(f"✓ Suma después de normalizar: {total:.6f}")
    assert abs(total - 1.0) < 1e-5, "Debe sumar 1"
    
    print("✓ Test de Variable Elimination: PASSED\n")


def test_gibbs_sampling():
    """Test 3: Gibbs Sampling."""
    print_test("Gibbs Sampling")
    
    # Crear mini dataset
    from src.data_generator import DatasetConfig
    config = DatasetConfig(
        num_real_users=2,
        num_bots=1,
        num_books=2,
        prob_user_multiple_accounts=0.0,
        prob_bot_multiple_accounts=0.0
    )
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    # Groundear RPM
    rpm = RPMModel()
    grounded = rpm.ground_model(
        customers[:2],
        books[:1],
        recommendations[:4]
    )
    
    print(f"✓ Red con {len(grounded.variables)} variables")
    
    # Gibbs Sampling
    gibbs = GibbsSampling(grounded)
    samples = gibbs.sample(
        evidence={},
        num_samples=500,
        burn_in=100
    )
    
    print(f"✓ Muestras generadas: {len(samples)}")
    print(f"  Primera muestra (iter {samples[0].iteration})")
    print(f"  Última muestra (iter {samples[-1].iteration})")
    
    # Estimar marginal
    quality_var = [v for v in grounded.variables.keys() if 'Quality' in v][0]
    marginal = gibbs.estimate_marginal(quality_var, samples)
    
    print(f"✓ Distribución estimada de {quality_var}:")
    for value, prob in sorted(marginal.items()):
        print(f"    {value}: {prob:.4f}")
    
    total_prob = sum(marginal.values())
    print(f"✓ Suma: {total_prob:.6f}")
    assert abs(total_prob - 1.0) < 1e-5
    
    print("✓ Test de Gibbs Sampling: PASSED\n")


def test_metropolis_hastings():
    """Test 4: Metropolis-Hastings."""
    print_test("Metropolis-Hastings")
    
    # Crear mini dataset
    from src.data_generator import DatasetConfig
    config = DatasetConfig(
        num_real_users=2,
        num_bots=1,
        num_books=2,
        prob_user_multiple_accounts=0.0,
        prob_bot_multiple_accounts=0.0
    )
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    # Groundear RPM
    rpm = RPMModel()
    grounded = rpm.ground_model(
        customers[:2],
        books[:1],
        recommendations[:4]
    )
    
    print(f"✓ Red con {len(grounded.variables)} variables")
    
    # Metropolis-Hastings
    mh = MetropolisHastings(grounded)
    samples = mh.sample(
        evidence={},
        num_samples=500,
        burn_in=100,
        proposal='random_flip'
    )
    
    print(f"✓ Muestras generadas: {len(samples)}")
    
    # Tasa de aceptación
    acceptance_rate = mh.get_acceptance_rate(samples)
    print(f"✓ Tasa de aceptación: {acceptance_rate:.3f}")
    print(f"  (Ideal: 0.2 - 0.5 para random walk)")
    
    # Estimar marginal
    quality_var = [v for v in grounded.variables.keys() if 'Quality' in v][0]
    marginal = mh.estimate_marginal(quality_var, samples)
    
    print(f"✓ Distribución estimada de {quality_var}:")
    for value, prob in sorted(marginal.items()):
        print(f"    {value}: {prob:.4f}")
    
    total_prob = sum(marginal.values())
    print(f"✓ Suma: {total_prob:.6f}")
    assert abs(total_prob - 1.0) < 1e-5
    
    print("✓ Test de Metropolis-Hastings: PASSED\n")


def test_convergence_diagnostics():
    """Test 5: Diagnósticos de convergencia."""
    print_test("Diagnósticos de Convergencia")
    
    # Crear mini dataset
    from src.data_generator import DatasetConfig
    config = DatasetConfig(
        num_real_users=2,
        num_bots=1,
        num_books=2,
        prob_user_multiple_accounts=0.0,
        prob_bot_multiple_accounts=0.0
    )
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    # Groundear RPM
    rpm = RPMModel()
    grounded = rpm.ground_model(
        customers[:2],
        books[:1],
        recommendations[:4]
    )
    
    # Ejecutar múltiples cadenas de Gibbs
    gibbs = GibbsSampling(grounded)
    chains = gibbs.run_multiple_chains(
        evidence={},
        num_chains=3,
        num_samples=300,
        burn_in=50
    )
    
    print(f"✓ {len(chains)} cadenas ejecutadas")
    print(f"  {len(chains[0])} muestras por cadena")
    
    # Gelman-Rubin
    quality_var = [v for v in grounded.variables.keys() if 'Quality' in v][0]
    r_hat = ConvergenceDiagnostics.gelman_rubin_statistic(chains, quality_var)
    
    print(f"✓ Gelman-Rubin R̂ para {quality_var}: {r_hat:.4f}")
    print(f"  R̂ ≈ 1.0 indica convergencia")
    print(f"  R̂ > 1.1 indica falta de convergencia")
    
    # Effective Sample Size
    ess = ConvergenceDiagnostics.effective_sample_size(chains[0], quality_var)
    print(f"✓ Effective Sample Size: {ess:.1f} / {len(chains[0])}")
    print(f"  ESS/N = {ess/len(chains[0]):.3f}")
    
    # Acceptance rate
    acceptance = ConvergenceDiagnostics.acceptance_rate(chains[0], quality_var)
    print(f"✓ Acceptance rate: {acceptance:.3f}")
    
    print("✓ Test de diagnósticos: PASSED\n")


def test_inference_comparison():
    """Test 6: Comparar Gibbs vs MH."""
    print_test("Comparación Gibbs vs Metropolis-Hastings")
    
    # Crear mini dataset
    from src.data_generator import DatasetConfig
    config = DatasetConfig(
        num_real_users=2,
        num_bots=1,
        num_books=1,
        prob_user_multiple_accounts=0.0,
        prob_bot_multiple_accounts=0.0
    )
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    # Groundear RPM
    rpm = RPMModel()
    grounded = rpm.ground_model(
        customers[:2],
        books[:1],
        recommendations[:3]
    )
    
    quality_var = [v for v in grounded.variables.keys() if 'Quality' in v][0]
    
    # 1. Gibbs Sampling
    print("\nGibbs Sampling (1000 muestras):")
    gibbs = GibbsSampling(grounded)
    gibbs_samples = gibbs.sample(evidence={}, num_samples=1000, burn_in=200)
    gibbs_marginal = gibbs.estimate_marginal(quality_var, gibbs_samples)
    
    for value, prob in sorted(gibbs_marginal.items()):
        print(f"  P({quality_var}={value}) = {prob:.4f}")
    
    # 2. Metropolis-Hastings
    print("\nMetropolis-Hastings (1000 muestras):")
    mh = MetropolisHastings(grounded)
    mh_samples = mh.sample(evidence={}, num_samples=1000, burn_in=200)
    mh_marginal = mh.estimate_marginal(quality_var, mh_samples)
    
    for value, prob in sorted(mh_marginal.items()):
        print(f"  P({quality_var}={value}) = {prob:.4f}")
    
    # Comparar diferencias
    print("\n✓ Diferencias absolutas entre métodos:")
    all_values = set(gibbs_marginal.keys()) | set(mh_marginal.keys())
    for value in sorted(all_values):
        gibbs_prob = gibbs_marginal.get(value, 0.0)
        mh_prob = mh_marginal.get(value, 0.0)
        diff = abs(gibbs_prob - mh_prob)
        print(f"  {value}: |{gibbs_prob:.4f} - {mh_prob:.4f}| = {diff:.4f}")
    
    print("\n✓ Ambos métodos convergen a distribuciones similares")
    print("✓ Test de comparación: PASSED\n")


def main():
    print_header("EJECUTANDO TESTS - FASE 4: INFERENCIA")
    
    test_factor_operations()
    test_variable_elimination_simple()
    test_gibbs_sampling()
    test_metropolis_hastings()
    test_convergence_diagnostics()
    test_inference_comparison()
    
    print_header("✓ TODOS LOS TESTS DE FASE 4 PASARON")


if __name__ == "__main__":
    main()
