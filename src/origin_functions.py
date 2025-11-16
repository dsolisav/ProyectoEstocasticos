"""
Origin Functions para OUPM (Open Universe Probability Model).

Las Origin Functions mapean objetos observables (LoginIDs) a objetos
latentes (Customers reales), con uncertainty sobre estas correspondencias.

Conceptos del Capítulo 18.2:
- O_LoginID: Función que mapea cada LoginID a su Customer real
- Identity Uncertainty: Múltiples LoginIDs pueden ser el mismo Customer
- Sybil Attacks: Múltiples cuentas controladas por la misma entidad
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
import random
import math

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models import Customer, LoginID, EntityType


@dataclass
class OriginAssignment:
    """
    Una asignación de Origin Functions para un conjunto de LoginIDs.
    
    Especifica qué Customer real está detrás de cada LoginID.
    Esta asignación puede ser incierta - múltiples asignaciones son posibles.
    
    Attributes:
        mappings: {login_id: customer_id} - Mapeo de LoginID → Customer
        probability: P(asignación) según el modelo
    """
    mappings: Dict[str, str] = field(default_factory=dict)
    probability: float = 1.0
    
    def get_customer_for_login(self, login_id: str) -> Optional[str]:
        """
        Obtiene el Customer asociado a un LoginID.
        
        Args:
            login_id: ID de la cuenta
        
        Returns:
            customer_id o None si no hay mapeo
        """
        return self.mappings.get(login_id)
    
    def get_logins_for_customer(self, customer_id: str) -> List[str]:
        """
        Obtiene todos los LoginIDs asociados a un Customer.
        
        Útil para detectar sybil attacks.
        
        Args:
            customer_id: ID del customer
        
        Returns:
            Lista de login_ids controlados por este customer
        """
        return [login_id for login_id, cust_id in self.mappings.items() 
                if cust_id == customer_id]
    
    def count_sybil_accounts(self) -> int:
        """
        Cuenta cuántos customers tienen múltiples cuentas.
        
        Returns:
            Número de customers con >1 LoginID (sybil attackers)
        """
        customer_counts = {}
        for customer_id in self.mappings.values():
            customer_counts[customer_id] = customer_counts.get(customer_id, 0) + 1
        
        return sum(1 for count in customer_counts.values() if count > 1)
    
    def is_sybil_attacker(self, customer_id: str) -> bool:
        """
        Determina si un customer tiene múltiples cuentas (sybil attack).
        
        Args:
            customer_id: ID del customer
        
        Returns:
            True si tiene >1 LoginID
        """
        return len(self.get_logins_for_customer(customer_id)) > 1
    
    def __repr__(self):
        n_logins = len(self.mappings)
        n_customers = len(set(self.mappings.values()))
        sybils = self.count_sybil_accounts()
        return f"OriginAssignment({n_logins} logins → {n_customers} customers, {sybils} sybils, P={self.probability:.4f})"


class OriginFunction:
    """
    Modelo probabilístico de Origin Functions.
    
    Define la distribución sobre posibles asignaciones de LoginIDs → Customers.
    Incorpora prior knowledge sobre sybil attacks y comportamiento de bots.
    """
    
    def __init__(
        self,
        sybil_probability: float = 0.3,
        bot_multi_account_prob: float = 0.8,
        user_multi_account_prob: float = 0.2,
        max_accounts_per_user: int = 3,
        max_accounts_per_bot: int = 10
    ):
        """
        Args:
            sybil_probability: Prior P(customer tiene múltiples cuentas)
            bot_multi_account_prob: P(bot tiene múltiples cuentas)
            user_multi_account_prob: P(usuario real tiene múltiples cuentas)
            max_accounts_per_user: Máximo de cuentas para usuario real
            max_accounts_per_bot: Máximo de cuentas para bot
        """
        self.sybil_probability = sybil_probability
        self.bot_multi_account_prob = bot_multi_account_prob
        self.user_multi_account_prob = user_multi_account_prob
        self.max_accounts_per_user = max_accounts_per_user
        self.max_accounts_per_bot = max_accounts_per_bot
    
    def sample_origin_assignment(
        self,
        login_ids: List[LoginID],
        customers: List[Customer]
    ) -> OriginAssignment:
        """
        Muestrea una asignación de origin functions.
        
        Args:
            login_ids: Lista de LoginIDs observados
            customers: Lista de customers reales posibles
        
        Returns:
            Una asignación muestreada con su probabilidad
        """
        mappings = {}
        
        # Para cada LoginID, asignar un Customer
        for login in login_ids:
            # Si ya conocemos el origin (ground truth), usar ese
            if login.origin is not None:
                mappings[login.login_id] = login.origin.customer_id
            else:
                # Muestrear uniformemente por ahora
                # En un modelo más sofisticado, usaríamos features del LoginID
                customer = random.choice(customers)
                mappings[login.login_id] = customer.customer_id
        
        # Calcular probabilidad de esta asignación
        prob = self._compute_assignment_probability(mappings, login_ids, customers)
        
        return OriginAssignment(mappings=mappings, probability=prob)
    
    def enumerate_possible_assignments(
        self,
        login_ids: List[LoginID],
        customers: List[Customer],
        max_assignments: int = 1000
    ) -> List[OriginAssignment]:
        """
        Enumera posibles asignaciones de origin functions.
        
        Para N LoginIDs y K Customers, hay K^N posibles asignaciones.
        Este método limita la enumeración para eficiencia.
        
        Args:
            login_ids: Lista de LoginIDs
            customers: Lista de customers posibles
            max_assignments: Máximo número de asignaciones a generar
        
        Returns:
            Lista de asignaciones posibles con probabilidades
        """
        assignments = []
        
        # Si hay pocas combinaciones, enumerar todas
        n_logins = len(login_ids)
        n_customers = len(customers)
        total_combinations = n_customers ** n_logins
        
        if total_combinations <= max_assignments:
            # Enumerar todas las combinaciones
            assignments = self._enumerate_all_assignments(login_ids, customers)
        else:
            # Muestrear las más probables
            assignments = self._sample_top_assignments(
                login_ids, customers, max_assignments
            )
        
        # Normalizar probabilidades
        total_prob = sum(a.probability for a in assignments)
        for assignment in assignments:
            assignment.probability /= total_prob
        
        return assignments
    
    def _enumerate_all_assignments(
        self,
        login_ids: List[LoginID],
        customers: List[Customer]
    ) -> List[OriginAssignment]:
        """Enumera todas las asignaciones posibles (para casos pequeños)"""
        import itertools
        
        assignments = []
        customer_ids = [c.customer_id for c in customers]
        
        # Generar todas las combinaciones
        for combo in itertools.product(customer_ids, repeat=len(login_ids)):
            mappings = {
                login.login_id: customer_id 
                for login, customer_id in zip(login_ids, combo)
            }
            
            prob = self._compute_assignment_probability(mappings, login_ids, customers)
            assignments.append(OriginAssignment(mappings=mappings, probability=prob))
        
        return assignments
    
    def _sample_top_assignments(
        self,
        login_ids: List[LoginID],
        customers: List[Customer],
        n_samples: int
    ) -> List[OriginAssignment]:
        """Muestrea asignaciones probables usando MCMC o importancia"""
        assignments = []
        
        for _ in range(n_samples):
            assignment = self.sample_origin_assignment(login_ids, customers)
            assignments.append(assignment)
        
        # Ordenar por probabilidad y tomar las top
        assignments.sort(key=lambda a: a.probability, reverse=True)
        
        # Remover duplicados
        seen = set()
        unique_assignments = []
        for assignment in assignments:
            key = tuple(sorted(assignment.mappings.items()))
            if key not in seen:
                seen.add(key)
                unique_assignments.append(assignment)
        
        return unique_assignments[:n_samples]
    
    def _compute_assignment_probability(
        self,
        mappings: Dict[str, str],
        login_ids: List[LoginID],
        customers: List[Customer]
    ) -> float:
        """
        Calcula P(asignación) según el modelo.
        
        Factores considerados:
        - Prior sobre número de cuentas por customer
        - Tipo de customer (bot vs. usuario real)
        - Patrones de comportamiento observados
        
        Args:
            mappings: Mapeo login_id → customer_id
            login_ids: LoginIDs observados
            customers: Customers posibles
        
        Returns:
            Probabilidad de la asignación
        """
        prob = 1.0
        
        # Crear lookup de customers por ID
        customer_lookup = {c.customer_id: c for c in customers}
        
        # Contar cuántas cuentas tiene cada customer en esta asignación
        accounts_per_customer = {}
        for customer_id in mappings.values():
            accounts_per_customer[customer_id] = accounts_per_customer.get(customer_id, 0) + 1
        
        # Para cada customer, calcular probabilidad de tener N cuentas
        for customer_id, n_accounts in accounts_per_customer.items():
            customer = customer_lookup.get(customer_id)
            if customer is None:
                continue
            
            # P(N cuentas | tipo de customer)
            if customer.entity_type == EntityType.BOT:
                # Bots tienden a tener múltiples cuentas
                prob *= self._prob_n_accounts_bot(n_accounts)
            else:
                # Usuarios reales raramente tienen múltiples cuentas
                prob *= self._prob_n_accounts_user(n_accounts)
        
        return prob
    
    def _prob_n_accounts_bot(self, n: int) -> float:
        """
        P(bot tiene N cuentas).
        
        Bots típicamente tienen múltiples cuentas (sybil attacks).
        Usamos distribución geométrica truncada.
        """
        if n == 0:
            return 0.0
        
        if n == 1:
            # Bot con solo 1 cuenta es menos probable
            return 1.0 - self.bot_multi_account_prob
        
        # Múltiples cuentas (geometric decay)
        if n > self.max_accounts_per_bot:
            return 1e-6  # Muy improbable
        
        p_multi = self.bot_multi_account_prob
        decay = 0.5  # Decay factor
        prob = p_multi * (decay ** (n - 2))
        
        return prob
    
    def _prob_n_accounts_user(self, n: int) -> float:
        """
        P(usuario real tiene N cuentas).
        
        Usuarios reales típicamente tienen 1 cuenta.
        Algunos pueden tener 2-3 (cuenta personal, trabajo, etc.)
        """
        if n == 0:
            return 0.0
        
        if n == 1:
            # Una cuenta es lo más común
            return 1.0 - self.user_multi_account_prob
        
        # Múltiples cuentas es raro
        if n > self.max_accounts_per_user:
            return 1e-6
        
        p_multi = self.user_multi_account_prob
        decay = 0.3  # Decay más rápido que bots
        prob = p_multi * (decay ** (n - 2))
        
        return prob
    
    def compute_identity_likelihood(
        self,
        login_id1: str,
        login_id2: str,
        assignment: OriginAssignment
    ) -> float:
        """
        Calcula P(LoginID1 y LoginID2 son el mismo customer).
        
        Args:
            login_id1: Primer LoginID
            login_id2: Segundo LoginID
            assignment: Asignación de origins
        
        Returns:
            Probabilidad de que sean el mismo customer
        """
        customer1 = assignment.get_customer_for_login(login_id1)
        customer2 = assignment.get_customer_for_login(login_id2)
        
        if customer1 is None or customer2 is None:
            return 0.0
        
        return 1.0 if customer1 == customer2 else 0.0


def demo_origin_functions():
    """Demo de Origin Functions con sybil attacks"""
    print("="*70)
    print("  DEMO: ORIGIN FUNCTIONS & IDENTITY UNCERTAINTY")
    print("="*70)
    
    # Crear customers
    customers = [
        Customer("User_1", EntityType.REAL_USER, 0.9),
        Customer("User_2", EntityType.REAL_USER, 0.8),
        Customer("Bot_1", EntityType.BOT, 0.1),
    ]
    
    # Crear LoginIDs (algunos son sybils)
    login_ids = [
        LoginID("LoginID_1"),
        LoginID("LoginID_2"),
        LoginID("LoginID_3"),
        LoginID("LoginID_4"),
        LoginID("LoginID_5"),
    ]
    
    print(f"\n📊 SETUP:")
    print(f"  Customers: {len(customers)} (2 users, 1 bot)")
    print(f"  LoginIDs: {len(login_ids)}")
    print(f"  Posibles asignaciones: {len(customers)**len(login_ids):,}")
    
    # Crear Origin Function model
    origin_model = OriginFunction(
        sybil_probability=0.3,
        bot_multi_account_prob=0.8,
        user_multi_account_prob=0.2
    )
    
    # Muestrear algunas asignaciones
    print(f"\n🎲 MUESTREANDO ASIGNACIONES:")
    for i in range(5):
        assignment = origin_model.sample_origin_assignment(login_ids, customers)
        print(f"\n  Asignación {i+1}: {assignment}")
        
        # Mostrar detalles
        for customer in customers:
            logins = assignment.get_logins_for_customer(customer.customer_id)
            if logins:
                is_sybil = "⚠️ SYBIL" if len(logins) > 1 else ""
                print(f"    {customer.customer_id}: {len(logins)} cuentas {is_sybil}")
                for login_id in logins:
                    print(f"      - {login_id}")
    
    # Enumerar top asignaciones
    print(f"\n🔝 TOP ASIGNACIONES MÁS PROBABLES:")
    top_assignments = origin_model.enumerate_possible_assignments(
        login_ids, customers, max_assignments=10
    )
    
    for i, assignment in enumerate(top_assignments[:5], 1):
        print(f"\n  #{i}: P = {assignment.probability:.4f}")
        print(f"      {assignment}")


if __name__ == "__main__":
    demo_origin_functions()
