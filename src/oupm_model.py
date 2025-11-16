"""
OUPM Model - Open Universe Probability Model.

Extiende el RPM básico para manejar:
- Identity Uncertainty: Múltiples LoginIDs → mismo Customer
- Existence Uncertainty: Número desconocido de Customers reales
- Generating Functions: Distribuciones sobre # de entidades

Conceptos del Capítulo 18.2:
- Origin Functions O_LoginID
- Generating Functions para números de entidades
- Possible Worlds con diferentes interpretaciones de identidades
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
import random
import math

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models import Customer, Book, LoginID, Recommendation, EntityType, PossibleWorld
from origin_functions import OriginFunction, OriginAssignment
from rpm_model import RPMModel
from cpt import create_default_cpts


class GeneratingFunction:
    """
    Generating Function para el número de entidades en el universo.
    
    Define P(N) donde N es el número de customers reales que existen.
    Típicamente usamos distribución Poisson o Geometric.
    """
    
    def __init__(self, distribution: str = "poisson", lambda_param: float = 10.0):
        """
        Args:
            distribution: Tipo de distribución ("poisson", "geometric", "uniform")
            lambda_param: Parámetro de la distribución
        """
        self.distribution = distribution
        self.lambda_param = lambda_param
    
    def probability(self, n: int) -> float:
        """
        P(N = n) - Probabilidad de que existan n customers.
        
        Args:
            n: Número de customers
        
        Returns:
            Probabilidad
        """
        if n < 0:
            return 0.0
        
        if self.distribution == "poisson":
            return self._poisson(n)
        elif self.distribution == "geometric":
            return self._geometric(n)
        elif self.distribution == "uniform":
            return self._uniform(n)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
    
    def _poisson(self, n: int) -> float:
        """Distribución Poisson: P(N=n) = (λ^n * e^(-λ)) / n!"""
        lam = self.lambda_param
        return (lam ** n) * math.exp(-lam) / math.factorial(n)
    
    def _geometric(self, n: int) -> float:
        """Distribución Geométrica: P(N=n) = p * (1-p)^n"""
        if n == 0:
            return 0.0
        p = 1.0 / self.lambda_param  # lambda como media
        return p * ((1 - p) ** (n - 1))
    
    def _uniform(self, n: int, max_n: int = 50) -> float:
        """Distribución Uniforme hasta max_n"""
        if 0 <= n <= max_n:
            return 1.0 / (max_n + 1)
        return 0.0
    
    def sample(self) -> int:
        """
        Muestrea el número de customers según la distribución.
        
        Returns:
            Número de customers muestreado
        """
        if self.distribution == "poisson":
            # Muestrear de Poisson usando algoritmo de Knuth
            return self._sample_poisson()
        elif self.distribution == "geometric":
            # Muestrear de Geometric
            p = 1.0 / self.lambda_param
            return self._sample_geometric(p)
        else:
            # Uniform
            return random.randint(1, int(self.lambda_param))
    
    def _sample_poisson(self) -> int:
        """Muestrea de distribución Poisson usando algoritmo de Knuth"""
        L = math.exp(-self.lambda_param)
        k = 0
        p = 1.0
        
        while p > L:
            k += 1
            p *= random.random()
        
        return k - 1
    
    def _sample_geometric(self, p: float) -> int:
        """Muestrea de distribución Geométrica"""
        if p <= 0 or p >= 1:
            return 1
        return int(math.log(random.random()) / math.log(1 - p)) + 1
    
    def __repr__(self):
        return f"GeneratingFunction({self.distribution}, λ={self.lambda_param})"


@dataclass
class OUPMWorld:
    """
    Un mundo posible en el modelo OUPM.
    
    Extiende PossibleWorld con información sobre origin functions
    y existence uncertainty.
    
    Attributes:
        world_id: ID único del mundo
        num_customers: Número de customers que existen en este mundo
        customers: Lista de customers reales
        origin_assignment: Asignación de LoginIDs → Customers
        book_qualities: Calidades reales de los libros
        probability: P(ω) - probabilidad de este mundo
    """
    world_id: str
    num_customers: int
    customers: List[Customer] = field(default_factory=list)
    origin_assignment: Optional[OriginAssignment] = None
    book_qualities: Dict[str, int] = field(default_factory=dict)
    probability: float = 0.0
    
    def get_customer_for_login(self, login_id: str) -> Optional[Customer]:
        """
        Obtiene el customer real detrás de un LoginID.
        
        Args:
            login_id: ID de la cuenta
        
        Returns:
            Customer o None
        """
        if self.origin_assignment is None:
            return None
        
        customer_id = self.origin_assignment.get_customer_for_login(login_id)
        if customer_id is None:
            return None
        
        for customer in self.customers:
            if customer.customer_id == customer_id:
                return customer
        return None
    
    def count_sybil_attackers(self) -> int:
        """Cuenta customers con múltiples cuentas"""
        if self.origin_assignment is None:
            return 0
        return self.origin_assignment.count_sybil_accounts()
    
    def __repr__(self):
        sybils = self.count_sybil_attackers()
        return f"OUPMWorld({self.world_id}: {self.num_customers} customers, {sybils} sybils, P={self.probability:.6f})"


class OUPMModel:
    """
    Open Universe Probability Model completo.
    
    Combina:
    - RPM básico (Quality, Honest, Recommendation)
    - Origin Functions (LoginID → Customer con uncertainty)
    - Generating Functions (número de customers con uncertainty)
    - Possible Worlds con diferentes interpretaciones
    """
    
    def __init__(
        self,
        name: str = "RecommendationOUPM",
        lambda_customers: float = 15.0,
        lambda_bots: float = 3.0
    ):
        """
        Args:
            name: Nombre del modelo
            lambda_customers: Parámetro λ para # de usuarios reales (Poisson)
            lambda_bots: Parámetro λ para # de bots (Poisson)
        """
        self.name = name
        
        # RPM base
        self.rpm = RPMModel()
        
        # Generating Functions
        self.user_generating = GeneratingFunction("poisson", lambda_customers)
        self.bot_generating = GeneratingFunction("poisson", lambda_bots)
        
        # Origin Function model
        self.origin_function = OriginFunction(
            sybil_probability=0.3,
            bot_multi_account_prob=0.8,
            user_multi_account_prob=0.2
        )
    
    def sample_possible_world(
        self,
        login_ids: List[LoginID],
        books: List[Book],
        world_id: Optional[str] = None
    ) -> OUPMWorld:
        """
        Muestrea un mundo posible del modelo OUPM.
        
        Pasos:
        1. Muestrear # de usuarios reales y bots (generating functions)
        2. Crear customers con propiedades
        3. Asignar LoginIDs → Customers (origin functions)
        4. Asignar calidades a libros
        5. Calcular P(mundo)
        
        Args:
            login_ids: LoginIDs observados
            books: Books en el sistema
            world_id: ID opcional para el mundo
        
        Returns:
            Mundo posible muestreado
        """
        if world_id is None:
            world_id = f"world_{random.randint(1000, 9999)}"
        
        # 1. Muestrear número de customers
        n_users = max(1, self.user_generating.sample())
        n_bots = max(0, self.bot_generating.sample())
        n_total = n_users + n_bots
        
        # 2. Crear customers
        customers = []
        
        # Usuarios reales
        for i in range(n_users):
            honest_prob = random.uniform(0.6, 0.95)
            customer = Customer(
                customer_id=f"RealUser_{i+1}",
                entity_type=EntityType.REAL_USER,
                honesty=honest_prob
            )
            customers.append(customer)
        
        # Bots
        for i in range(n_bots):
            honest_prob = random.uniform(0.0, 0.2)
            customer = Customer(
                customer_id=f"Bot_{i+1}",
                entity_type=EntityType.BOT,
                honesty=honest_prob
            )
            customers.append(customer)
        
        # 3. Asignar LoginIDs → Customers (origin assignment)
        origin_assignment = self.origin_function.sample_origin_assignment(
            login_ids, customers
        )
        
        # 4. Asignar calidades a libros
        book_qualities = {}
        for book in books:
            # Muestrear calidad del prior
            quality = random.choices(
                [1, 2, 3, 4, 5],
                weights=[0.1, 0.15, 0.3, 0.3, 0.15]
            )[0]
            book_qualities[book.book_id] = quality
        
        # 5. Calcular P(mundo)
        prob = self._compute_world_probability(
            n_users, n_bots, origin_assignment, book_qualities
        )
        
        # Crear mundo
        world = OUPMWorld(
            world_id=world_id,
            num_customers=n_total,
            customers=customers,
            origin_assignment=origin_assignment,
            book_qualities=book_qualities,
            probability=prob
        )
        
        return world
    
    def enumerate_possible_worlds(
        self,
        login_ids: List[LoginID],
        books: List[Book],
        max_worlds: int = 100,
        max_customers: int = 30
    ) -> List[OUPMWorld]:
        """
        Enumera mundos posibles hasta un límite.
        
        Args:
            login_ids: LoginIDs observados
            books: Books en el sistema
            max_worlds: Máximo número de mundos a generar
            max_customers: Máximo número de customers a considerar
        
        Returns:
            Lista de mundos posibles
        """
        worlds = []
        
        # Iterar sobre posibles números de customers
        for n_total in range(len(login_ids), min(max_customers, len(login_ids) * 3)):
            # Para cada distribución de users/bots
            for n_bots in range(0, min(n_total, 10)):
                n_users = n_total - n_bots
                
                if n_users < 1:
                    continue
                
                # Crear customers
                customers = self._create_customers(n_users, n_bots)
                
                # Muestrear algunas asignaciones de origins
                assignments = self.origin_function.enumerate_possible_assignments(
                    login_ids, customers, max_assignments=10
                )
                
                for assignment in assignments[:5]:  # Top 5 por eficiencia
                    # Muestrear calidades de libros
                    book_qualities = {
                        book.book_id: random.choice([3, 4])  # Simplificado
                        for book in books
                    }
                    
                    # Calcular probabilidad
                    prob = self._compute_world_probability(
                        n_users, n_bots, assignment, book_qualities
                    )
                    
                    world = OUPMWorld(
                        world_id=f"world_{len(worlds)}",
                        num_customers=n_total,
                        customers=customers,
                        origin_assignment=assignment,
                        book_qualities=book_qualities,
                        probability=prob
                    )
                    
                    worlds.append(world)
                    
                    if len(worlds) >= max_worlds:
                        break
                
                if len(worlds) >= max_worlds:
                    break
            
            if len(worlds) >= max_worlds:
                break
        
        # Normalizar probabilidades
        total_prob = sum(w.probability for w in worlds)
        if total_prob > 0:
            for world in worlds:
                world.probability /= total_prob
        
        return worlds
    
    def _create_customers(self, n_users: int, n_bots: int) -> List[Customer]:
        """Crea lista de customers con propiedades"""
        customers = []
        
        for i in range(n_users):
            customer = Customer(
                customer_id=f"RealUser_{i+1}",
                entity_type=EntityType.REAL_USER,
                honesty=random.uniform(0.7, 0.95)
            )
            customers.append(customer)
        
        for i in range(n_bots):
            customer = Customer(
                customer_id=f"Bot_{i+1}",
                entity_type=EntityType.BOT,
                honesty=random.uniform(0.0, 0.2)
            )
            customers.append(customer)
        
        return customers
    
    def _compute_world_probability(
        self,
        n_users: int,
        n_bots: int,
        origin_assignment: OriginAssignment,
        book_qualities: Dict[str, int]
    ) -> float:
        """
        Calcula P(mundo) según el modelo OUPM.
        
        P(ω) = P(#users) × P(#bots) × P(origin_assignment) × P(qualities)
        
        Args:
            n_users: Número de usuarios reales
            n_bots: Número de bots
            origin_assignment: Asignación de origins
            book_qualities: Calidades de libros
        
        Returns:
            Probabilidad del mundo
        """
        prob = 1.0
        
        # P(#users)
        prob *= self.user_generating.probability(n_users)
        
        # P(#bots)
        prob *= self.bot_generating.probability(n_bots)
        
        # P(origin assignment)
        prob *= origin_assignment.probability
        
        # P(book qualities)
        quality_cpt = self.rpm.cpts["quality"]
        for quality in book_qualities.values():
            prob *= quality_cpt.get_quality_probability(quality)
        
        return prob
    
    def print_model_summary(self):
        """Imprime resumen del modelo OUPM"""
        print(f"\n{'='*70}")
        print(f"OPEN UNIVERSE PROBABILITY MODEL: {self.name}")
        print(f"{'='*70}")
        
        print(f"\n📊 GENERATING FUNCTIONS:")
        print(f"  Users: {self.user_generating}")
        print(f"  Bots: {self.bot_generating}")
        
        print(f"\n🔗 ORIGIN FUNCTIONS:")
        print(f"  Sybil probability: {self.origin_function.sybil_probability}")
        print(f"  Bot multi-account: {self.origin_function.bot_multi_account_prob}")
        print(f"  User multi-account: {self.origin_function.user_multi_account_prob}")
        
        print(f"\n🎲 BASE RPM:")
        print(f"  Type signatures: {len(self.rpm.type_signatures)}")
        print(f"  CPTs: {len(self.rpm.cpts)}")
        
        print(f"\n{'='*70}")


def demo_oupm_model():
    """Demo del modelo OUPM completo"""
    print("="*70)
    print("  DEMO: OPEN UNIVERSE PROBABILITY MODEL")
    print("="*70)
    
    # Crear modelo
    model = OUPMModel(lambda_customers=10.0, lambda_bots=3.0)
    model.print_model_summary()
    
    # Crear datos observados
    login_ids = [LoginID(f"LoginID_{i}") for i in range(1, 8)]
    books = [Book(f"Book_{i}", true_quality=3) for i in range(1, 5)]
    
    print(f"\n📊 DATOS OBSERVADOS:")
    print(f"  LoginIDs: {len(login_ids)}")
    print(f"  Books: {len(books)}")
    
    # Muestrear mundos posibles
    print(f"\n🌍 MUESTREANDO MUNDOS POSIBLES:")
    
    for i in range(5):
        world = model.sample_possible_world(login_ids, books)
        print(f"\n  Mundo {i+1}: {world}")
        print(f"    Usuarios reales: {sum(1 for c in world.customers if c.entity_type == EntityType.REAL_USER)}")
        print(f"    Bots: {sum(1 for c in world.customers if c.entity_type == EntityType.BOT)}")
        print(f"    Sybil attackers: {world.count_sybil_attackers()}")
    
    print(f"\n✓ DEMO completado!")


if __name__ == "__main__":
    demo_oupm_model()
