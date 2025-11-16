"""
Generador de datos sintéticos para el sistema de recomendación.

Genera datasets con:
- Usuarios reales y bots
- Libros con diferentes calidades
- Recomendaciones realistas
- Sybil attacks (múltiples cuentas del mismo usuario/bot)
"""

import random
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
import json
import time

from .models import (
    Customer, Book, Recommendation, LoginID, EntityType,
    RATING_VALUES, MIN_RATING, MAX_RATING
)


@dataclass
class DatasetConfig:
    """Configuración para generación de datasets"""
    # Usuarios
    num_real_users: int = 20
    num_bots: int = 5
    
    # Libros
    num_books: int = 15
    
    # Sybil attacks
    prob_user_multiple_accounts: float = 0.2  # 20% usuarios tienen >1 cuenta
    max_accounts_per_user: int = 3
    
    prob_bot_multiple_accounts: float = 0.8   # 80% bots tienen múltiples cuentas
    max_accounts_per_bot: int = 10
    
    # Recomendaciones
    min_recommendations_per_account: int = 1
    max_recommendations_per_account: int = 10
    
    # Comportamiento
    honest_user_probability: float = 0.8  # 80% usuarios son honestos
    bot_honesty: float = 0.1  # Bots casi nunca son honestos
    
    # Semilla para reproducibilidad
    random_seed: int = 42


class DataGenerator:
    """
    Generador de datos sintéticos para el sistema.
    """
    
    def __init__(self, config: DatasetConfig = None):
        """
        Args:
            config: Configuración del dataset
        """
        self.config = config or DatasetConfig()
        random.seed(self.config.random_seed)
        
    def generate_dataset(self) -> Tuple[List[Customer], List[Book], List[LoginID], List[Recommendation]]:
        """
        Genera un dataset completo.
        
        Returns:
            (customers, books, login_ids, recommendations)
        """
        # 1. Generar usuarios reales y bots
        customers = self._generate_customers()
        
        # 2. Generar libros
        books = self._generate_books()
        
        # 3. Generar cuentas (LoginIDs) con sybil attacks
        login_ids = self._generate_login_ids(customers)
        
        # 4. Generar recomendaciones
        recommendations = self._generate_recommendations(customers, books, login_ids)
        
        return customers, books, login_ids, recommendations
    
    def _generate_customers(self) -> List[Customer]:
        """Genera usuarios reales y bots"""
        customers = []
        
        # Usuarios reales
        for i in range(self.config.num_real_users):
            honesty = 1.0 if random.random() < self.config.honest_user_probability else 0.0
            
            customer = Customer(
                customer_id=f"User_{i+1}",
                entity_type=EntityType.REAL_USER,
                honesty=honesty,
                true_preferences={}
            )
            customers.append(customer)
        
        # Bots
        for i in range(self.config.num_bots):
            customer = Customer(
                customer_id=f"Bot_{i+1}",
                entity_type=EntityType.BOT,
                honesty=self.config.bot_honesty,
                true_preferences={}
            )
            customers.append(customer)
        
        return customers
    
    def _generate_books(self) -> List[Book]:
        """Genera libros con calidades variadas"""
        books = []
        
        genres = ["Fiction", "Science", "History", "Fantasy", "Mystery", "Romance"]
        
        for i in range(self.config.num_books):
            # Calidad real del libro (distribución más realista: sesgada hacia calidad media-alta)
            quality = random.choices(
                RATING_VALUES,
                weights=[0.1, 0.15, 0.25, 0.3, 0.2],  # Más libros de calidad 3-4
                k=1
            )[0]
            
            book = Book(
                book_id=f"Book_{i+1}",
                true_quality=quality,
                title=f"Book Title {i+1}",
                genre=random.choice(genres)
            )
            books.append(book)
        
        return books
    
    def _generate_login_ids(self, customers: List[Customer]) -> List[LoginID]:
        """
        Genera LoginIDs con sybil attacks.
        
        Algunos usuarios/bots tendrán múltiples cuentas.
        """
        login_ids = []
        login_counter = 1
        
        for customer in customers:
            # Determinar cuántas cuentas tiene este customer
            if customer.entity_type == EntityType.BOT:
                # Bots tienden a tener múltiples cuentas
                if random.random() < self.config.prob_bot_multiple_accounts:
                    num_accounts = random.randint(2, self.config.max_accounts_per_bot)
                else:
                    num_accounts = 1
            else:
                # Usuarios reales raramente tienen múltiples cuentas
                if random.random() < self.config.prob_user_multiple_accounts:
                    num_accounts = random.randint(2, self.config.max_accounts_per_user)
                else:
                    num_accounts = 1
            
            # Crear las cuentas
            for _ in range(num_accounts):
                login_id = LoginID(
                    login_id=f"LoginID_{login_counter}",
                    origin=customer
                )
                login_ids.append(login_id)
                login_counter += 1
        
        return login_ids
    
    def _generate_recommendations(
        self,
        customers: List[Customer],
        books: List[Book],
        login_ids: List[LoginID]
    ) -> List[Recommendation]:
        """
        Genera recomendaciones basadas en:
        - Calidad real del libro
        - Honestidad del usuario
        - Comportamiento de bots (ratings extremos, patrones)
        """
        recommendations = []
        timestamp = time.time()
        
        for login in login_ids:
            customer = login.origin
            
            # Número de recomendaciones que hace esta cuenta
            num_recs = random.randint(
                self.config.min_recommendations_per_account,
                self.config.max_recommendations_per_account
            )
            
            # Seleccionar libros aleatorios (sin repetir)
            selected_books = random.sample(books, min(num_recs, len(books)))
            
            for book in selected_books:
                # Generar rating basado en honestidad y tipo de usuario
                rating = self._generate_rating(customer, book)
                
                rec = Recommendation(
                    login_id=login.login_id,
                    book_id=book.book_id,
                    rating=rating,
                    timestamp=timestamp
                )
                
                login.recommendations.append(rec)
                recommendations.append(rec)
                timestamp += random.uniform(1, 3600)  # Espaciar en el tiempo
        
        return recommendations
    
    def _generate_rating(self, customer: Customer, book: Book) -> int:
        """
        Genera un rating basado en el modelo del capítulo:
        
        - Si honesto: rating cercano a la calidad real
        - Si deshonesto: rating aleatorio o con patrón
        - Bots: comportamiento específico (todo 5★ o todo 1★)
        """
        true_quality = book.true_quality
        
        if customer.entity_type == EntityType.BOT:
            # Bots tienen comportamiento extremo y predecible
            if random.random() < 0.5:
                # Bots que inflan ratings
                return 5
            else:
                # Bots que destruyen reputación
                return 1
        
        # Usuario real
        if random.random() < customer.honesty:
            # Usuario honesto: rating cercano a calidad real con ruido
            noise = random.randint(-1, 1)
            rating = true_quality + noise
        else:
            # Usuario deshonesto: rating aleatorio
            rating = random.choice(RATING_VALUES)
        
        # Asegurar rango válido
        rating = max(MIN_RATING, min(MAX_RATING, rating))
        
        return rating
    
    def save_to_json(
        self,
        customers: List[Customer],
        books: List[Book],
        login_ids: List[LoginID],
        recommendations: List[Recommendation],
        filepath: str = "data/dataset.json"
    ):
        """Guardar dataset a archivo JSON"""
        data = {
            "config": {
                "num_real_users": self.config.num_real_users,
                "num_bots": self.config.num_bots,
                "num_books": self.config.num_books,
                "num_login_ids": len(login_ids),
                "num_recommendations": len(recommendations)
            },
            "customers": [
                {
                    "customer_id": c.customer_id,
                    "entity_type": c.entity_type.value,
                    "honesty": c.honesty
                }
                for c in customers
            ],
            "books": [
                {
                    "book_id": b.book_id,
                    "true_quality": b.true_quality,
                    "title": b.title,
                    "genre": b.genre
                }
                for b in books
            ],
            "login_ids": [
                {
                    "login_id": l.login_id,
                    "origin_customer": l.origin.customer_id if l.origin else None,
                    "num_recommendations": len(l.recommendations)
                }
                for l in login_ids
            ],
            "recommendations": [
                {
                    "login_id": r.login_id,
                    "book_id": r.book_id,
                    "rating": r.rating,
                    "timestamp": r.timestamp
                }
                for r in recommendations
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Dataset guardado en: {filepath}")
    
    def print_statistics(
        self,
        customers: List[Customer],
        books: List[Book],
        login_ids: List[LoginID],
        recommendations: List[Recommendation]
    ):
        """Imprime estadísticas del dataset generado"""
        print("\n" + "="*60)
        print("ESTADÍSTICAS DEL DATASET")
        print("="*60)
        
        # Customers
        num_real = sum(1 for c in customers if c.entity_type == EntityType.REAL_USER)
        num_bots = sum(1 for c in customers if c.entity_type == EntityType.BOT)
        num_honest = sum(1 for c in customers if c.honesty >= 0.5)
        
        print(f"\n📊 USUARIOS:")
        print(f"   Total customers: {len(customers)}")
        print(f"   - Usuarios reales: {num_real}")
        print(f"   - Bots: {num_bots}")
        print(f"   - Honestos: {num_honest} ({100*num_honest/len(customers):.1f}%)")
        
        # Books
        avg_quality = sum(b.true_quality for b in books) / len(books)
        print(f"\n📚 LIBROS:")
        print(f"   Total libros: {len(books)}")
        print(f"   Calidad promedio: {avg_quality:.2f}")
        
        # LoginIDs y Sybil Attacks
        customers_with_multiple = sum(
            1 for c in customers 
            if sum(1 for l in login_ids if l.origin == c) > 1
        )
        
        print(f"\n🔐 CUENTAS (LoginIDs):")
        print(f"   Total LoginIDs: {len(login_ids)}")
        print(f"   Customers con >1 cuenta: {customers_with_multiple}")
        print(f"   Ratio LoginID/Customer: {len(login_ids)/len(customers):.2f}")
        
        # Recommendations
        avg_rating = sum(r.rating for r in recommendations) / len(recommendations)
        rating_dist = {i: sum(1 for r in recommendations if r.rating == i) 
                       for i in RATING_VALUES}
        
        print(f"\n⭐ RECOMENDACIONES:")
        print(f"   Total recomendaciones: {len(recommendations)}")
        print(f"   Rating promedio: {avg_rating:.2f}")
        print(f"   Distribución de ratings:")
        for rating, count in rating_dist.items():
            pct = 100 * count / len(recommendations)
            bar = "█" * int(pct / 2)
            print(f"     {rating}★: {count:3d} ({pct:5.1f}%) {bar}")
        
        print("\n" + "="*60)


def generate_sample_dataset():
    """Función auxiliar para generar un dataset de ejemplo"""
    config = DatasetConfig(
        num_real_users=20,
        num_bots=5,
        num_books=15,
        prob_user_multiple_accounts=0.2,
        prob_bot_multiple_accounts=0.8,
        max_accounts_per_bot=10,
        random_seed=42
    )
    
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    # Imprimir estadísticas
    generator.print_statistics(customers, books, login_ids, recommendations)
    
    # Guardar a JSON
    generator.save_to_json(customers, books, login_ids, recommendations)
    
    return customers, books, login_ids, recommendations


if __name__ == "__main__":
    # Generar dataset de ejemplo
    generate_sample_dataset()
