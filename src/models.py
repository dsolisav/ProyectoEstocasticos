"""
Modelos de datos base para el sistema de recomendación.

Define las entidades principales:
- Customer: Cliente/usuario (puede ser real o bot)
- Book: Libro/producto a recomendar
- Recommendation: Recomendación de un cliente sobre un libro
- LoginID: Identificador de cuenta (puede mapear a múltiples usuarios)
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class EntityType(Enum):
    """Tipos de entidades en el sistema"""
    REAL_USER = "real_user"
    BOT = "bot"
    UNKNOWN = "unknown"


@dataclass
class Customer:
    """
    Representa un cliente/usuario real del sistema.
    
    En el modelo OUPM, pueden existir usuarios reales que no conocemos,
    y múltiples LoginIDs pueden mapear al mismo Customer (sybil attacks).
    
    Attributes:
        customer_id: Identificador único del usuario real
        entity_type: Si es usuario real o bot
        honesty: Probabilidad de dar recomendaciones honestas [0,1]
        true_preferences: Gustos reales del usuario (escala 1-5)
    """
    customer_id: str
    entity_type: EntityType = EntityType.UNKNOWN
    honesty: float = 0.5  # Prior: 50% honesto
    true_preferences: dict = field(default_factory=dict)  # {book_id: rating}
    
    def __repr__(self):
        return f"Customer({self.customer_id}, type={self.entity_type.value})"
    
    def __hash__(self):
        return hash(self.customer_id)
    
    def __eq__(self, other):
        if not isinstance(other, Customer):
            return False
        return self.customer_id == other.customer_id


@dataclass
class Book:
    """
    Representa un libro/producto en el sistema.
    
    Attributes:
        book_id: Identificador único del libro
        true_quality: Calidad real del libro (escala 1-5)
        title: Título descriptivo (opcional)
        genre: Género/categoría (opcional)
    """
    book_id: str
    true_quality: int = 3  # Calidad desconocida inicialmente
    title: Optional[str] = None
    genre: Optional[str] = None
    
    def __repr__(self):
        title_str = f", '{self.title}'" if self.title else ""
        return f"Book({self.book_id}{title_str}, quality={self.true_quality})"
    
    def __hash__(self):
        return hash(self.book_id)
    
    def __eq__(self, other):
        if not isinstance(other, Book):
            return False
        return self.book_id == other.book_id


@dataclass
class Recommendation:
    """
    Representa una recomendación (rating) de un cliente sobre un libro.
    
    Esta es la evidencia observable en nuestro sistema.
    
    Attributes:
        login_id: ID de la cuenta que hizo la recomendación
        book_id: ID del libro recomendado
        rating: Rating dado (escala 1-5)
        timestamp: Momento de la recomendación (opcional)
    """
    login_id: str
    book_id: str
    rating: int  # 1-5
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        """Validar que el rating esté en rango válido"""
        if not (1 <= self.rating <= 5):
            raise ValueError(f"Rating debe estar entre 1 y 5, recibido: {self.rating}")
    
    def __repr__(self):
        return f"Rec({self.login_id} → {self.book_id}: {self.rating}★)"


@dataclass
class LoginID:
    """
    Representa una cuenta/identidad en el sistema.
    
    En el modelo OUPM, hay identity uncertainty:
    múltiples LoginIDs pueden corresponder al mismo Customer real.
    
    Attributes:
        login_id: Identificador de la cuenta
        origin: Customer real detrás de esta cuenta (puede ser desconocido)
        recommendations: Lista de recomendaciones hechas por esta cuenta
    """
    login_id: str
    origin: Optional[Customer] = None  # O_LoginID en notación del capítulo
    recommendations: List[Recommendation] = field(default_factory=list)
    
    def add_recommendation(self, book_id: str, rating: int, timestamp: Optional[float] = None):
        """Agregar una recomendación a esta cuenta"""
        rec = Recommendation(
            login_id=self.login_id,
            book_id=book_id,
            rating=rating,
            timestamp=timestamp
        )
        self.recommendations.append(rec)
        return rec
    
    def get_recommendations_count(self) -> int:
        """Número de recomendaciones hechas por esta cuenta"""
        return len(self.recommendations)
    
    def __repr__(self):
        origin_str = f" → {self.origin.customer_id}" if self.origin else ""
        return f"LoginID({self.login_id}{origin_str}, {len(self.recommendations)} recs)"
    
    def __hash__(self):
        return hash(self.login_id)
    
    def __eq__(self, other):
        if not isinstance(other, LoginID):
            return False
        return self.login_id == other.login_id


@dataclass
class PossibleWorld:
    """
    Representa un mundo posible en el modelo OUPM.
    
    Un mundo posible especifica:
    - Qué customers reales existen
    - Qué LoginIDs mapean a qué customers (origin functions)
    - Calidad real de cada libro
    - Honestidad de cada customer
    
    Attributes:
        world_id: Identificador único del mundo
        customers: Lista de customers reales que existen en este mundo
        login_mappings: Mapeo de LoginID → Customer (origin functions)
        book_qualities: Mapeo de book_id → calidad real
        customer_honesties: Mapeo de customer_id → honestidad
        probability: P(ω) - probabilidad de este mundo
    """
    world_id: str
    customers: List[Customer] = field(default_factory=list)
    login_mappings: dict = field(default_factory=dict)  # {login_id: customer_id}
    book_qualities: dict = field(default_factory=dict)  # {book_id: quality}
    customer_honesties: dict = field(default_factory=dict)  # {customer_id: honesty}
    probability: float = 0.0
    
    def get_customer_for_login(self, login_id: str) -> Optional[Customer]:
        """Obtener el customer real detrás de un LoginID en este mundo"""
        customer_id = self.login_mappings.get(login_id)
        if customer_id is None:
            return None
        
        for customer in self.customers:
            if customer.customer_id == customer_id:
                return customer
        return None
    
    def count_bots(self) -> int:
        """Contar número de bots en este mundo"""
        return sum(1 for c in self.customers if c.entity_type == EntityType.BOT)
    
    def count_real_users(self) -> int:
        """Contar número de usuarios reales en este mundo"""
        return sum(1 for c in self.customers if c.entity_type == EntityType.REAL_USER)
    
    def __repr__(self):
        n_customers = len(self.customers)
        n_bots = self.count_bots()
        n_real = self.count_real_users()
        return f"World({self.world_id}: {n_real} users, {n_bots} bots, P={self.probability:.4f})"


# Constantes del modelo
RATING_VALUES = [1, 2, 3, 4, 5]  # Posibles valores de rating
MIN_RATING = 1
MAX_RATING = 5

# Prior por defecto sobre honestidad
DEFAULT_HONESTY_PRIOR = 0.7  # Asumimos que la mayoría son honestos

# Prior por defecto sobre calidad de libros (distribución uniforme)
DEFAULT_QUALITY_PRIOR = {q: 1.0/5.0 for q in RATING_VALUES}
