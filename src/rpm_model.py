"""
Relational Probability Model (RPM) - Modelo probabilístico relacional.

Implementa el modelo RPM del Capítulo 18 para el sistema de recomendación.

Conceptos implementados:
- Type signatures (Customer, Book)
- Predicados relacionales: Quality(b), Honest(c), Rec(c,b)
- CPTs relacionales
- Dependency structure
- Database semantics
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from models import Customer, Book, Recommendation, EntityType
from cpt import CPT, create_default_cpts
from grounding import RPMGrounder, GroundedBayesNet


@dataclass
class TypeSignature:
    """
    Especifica el tipo de los argumentos de un predicado.
    
    Por ejemplo:
    - Quality(Book) → TypeSignature(["Book"])
    - Honest(Customer) → TypeSignature(["Customer"])
    - Rec(Customer, Book) → TypeSignature(["Customer", "Book"])
    """
    name: str
    arg_types: List[str]
    
    def __repr__(self):
        args_str = ", ".join(self.arg_types)
        return f"{self.name}({args_str})"


class RPMModel:
    """
    Relational Probability Model completo.
    
    Un RPM especifica:
    1. Type signatures para cada predicado
    2. Dependency structure (qué depende de qué)
    3. CPTs relacionales (probabilidades condicionales)
    
    El RPM es una representación compacta que se "ground"
    para obtener una Bayes Net completa.
    """
    
    def __init__(self, name: str = "RecommendationRPM"):
        """
        Args:
            name: Nombre del modelo
        """
        self.name = name
        
        # Type signatures
        self.types = ["Customer", "Book"]
        self.type_signatures = self._define_type_signatures()
        
        # CPTs relacionales
        self.cpts = create_default_cpts()
        
        # Dependency structure
        self.dependencies = self._define_dependencies()
    
    def _define_type_signatures(self) -> Dict[str, TypeSignature]:
        """
        Define los type signatures de los predicados.
        
        Returns:
            Diccionario {predicate_name: TypeSignature}
        """
        return {
            "Quality": TypeSignature("Quality", ["Book"]),
            "Honest": TypeSignature("Honest", ["Customer"]),
            "Recommendation": TypeSignature("Recommendation", ["Customer", "Book"]),
            "EntityType": TypeSignature("EntityType", ["Customer"])
        }
    
    def _define_dependencies(self) -> Dict[str, List[str]]:
        """
        Define la estructura de dependencias del modelo.
        
        Format: {child: [parent1, parent2, ...]}
        
        En nuestro modelo:
        - Quality(b): sin padres (prior)
        - Honest(c): sin padres (prior)
        - Rec(c,b): depende de Quality(b) y Honest(c)
        
        Returns:
            Diccionario de dependencias
        """
        return {
            "Quality": [],
            "Honest": [],
            "Recommendation": ["Quality", "Honest"],
            "EntityType": []
        }
    
    def get_predicate_parents(self, predicate: str) -> List[str]:
        """
        Obtiene los predicados padre de un predicado.
        
        Args:
            predicate: Nombre del predicado
        
        Returns:
            Lista de predicados padre
        """
        return self.dependencies.get(predicate, [])
    
    def ground_model(
        self,
        customers: List[Customer],
        books: List[Book],
        recommendations: List[Recommendation] = None
    ) -> GroundedBayesNet:
        """
        Ground el modelo RPM para obtener una Bayes Net.
        
        Args:
            customers: Objetos de tipo Customer
            books: Objetos de tipo Book
            recommendations: Evidencia (recomendaciones observadas)
        
        Returns:
            Red bayesiana grounded
        """
        grounder = RPMGrounder(self.cpts)
        return grounder.ground_rpm(customers, books, recommendations)
    
    def compute_world_probability(
        self,
        customers: List[Customer],
        books: List[Book],
        assignment: Dict[str, Any]
    ) -> float:
        """
        Calcula P(ω) para un mundo posible ω.
        
        Un mundo especifica valores para todas las variables:
        - Quality(b) para cada libro b
        - Honest(c) para cada customer c
        - Rec(c,b) para cada par (c,b) que tiene recomendación
        
        Args:
            customers: Customers en el mundo
            books: Books en el mundo
            assignment: Asignación completa de valores
        
        Returns:
            P(ω) según el modelo
        """
        # Ground el modelo
        net = self.ground_model(customers, books)
        
        # Calcular probabilidad usando la red grounded
        return net.compute_probability(assignment)
    
    def query_probability(
        self,
        query_vars: Dict[str, Any],
        evidence: Dict[str, Any],
        customers: List[Customer],
        books: List[Book]
    ) -> float:
        """
        Calcula P(query | evidence) usando el modelo.
        
        Por ejemplo:
        query_vars = {"Quality_Book_1": 5}
        evidence = {"Rec_User_1_Book_1": 5, "Honest_User_1": True}
        
        Calcula: P(Quality_Book_1=5 | Rec=5, Honest=True)
        
        Args:
            query_vars: Variables de query con valores
            evidence: Evidencia observada
            customers: Lista de customers
            books: Lista de books
        
        Returns:
            Probabilidad condicional P(query | evidence)
        """
        # Esta es una versión simplificada
        # En FASE 4 implementaremos inferencia completa
        
        # Por ahora, calcular usando enumeración directa
        # P(Q|E) = P(Q,E) / P(E)
        
        # Ground el modelo
        net = self.ground_model(customers, books)
        
        # Agregar evidencia
        for var_name, value in evidence.items():
            net.set_observation(var_name, value)
        
        # Esto requiere algoritmos de inferencia
        # que implementaremos en FASE 4
        raise NotImplementedError(
            "Query inference será implementado en FASE 4 "
            "(Variable Elimination y MCMC)"
        )
    
    def print_model_summary(self):
        """Imprime un resumen del modelo RPM"""
        print(f"\n{'='*70}")
        print(f"RELATIONAL PROBABILITY MODEL: {self.name}")
        print(f"{'='*70}")
        
        print(f"\n📋 TYPE SIGNATURES:")
        for name, sig in self.type_signatures.items():
            print(f"  {sig}")
        
        print(f"\n🔗 DEPENDENCY STRUCTURE:")
        for child, parents in self.dependencies.items():
            if parents:
                parents_str = ", ".join(parents)
                print(f"  {child} depends on: {parents_str}")
            else:
                print(f"  {child} (no parents - prior)")
        
        print(f"\n📊 CPTs:")
        for name, cpt in self.cpts.items():
            if cpt:
                print(f"  {cpt}")
        
        print(f"\n{'='*70}")


def demo_rpm_model():
    """
    Demo del modelo RPM completo.
    """
    print("="*70)
    print("  DEMO: RELATIONAL PROBABILITY MODEL")
    print("="*70)
    
    # Crear modelo
    model = RPMModel()
    model.print_model_summary()
    
    # Crear objetos de ejemplo
    print("\n" + "="*70)
    print("  CREANDO OBJETOS DE EJEMPLO")
    print("="*70)
    
    customers = [
        Customer(customer_id="User_1", entity_type=EntityType.REAL_USER, honesty=0.9),
        Customer(customer_id="User_2", entity_type=EntityType.REAL_USER, honesty=0.7),
        Customer(customer_id="Bot_1", entity_type=EntityType.BOT, honesty=0.1)
    ]
    
    books = [
        Book(book_id="Book_1", true_quality=5, title="Great Book"),
        Book(book_id="Book_2", true_quality=2, title="Poor Book"),
        Book(book_id="Book_3", true_quality=4, title="Good Book")
    ]
    
    recommendations = [
        Recommendation(login_id="LoginID_1", book_id="Book_1", rating=5),
        Recommendation(login_id="LoginID_1", book_id="Book_2", rating=2),
        Recommendation(login_id="LoginID_2", book_id="Book_1", rating=4),
        Recommendation(login_id="LoginID_3", book_id="Book_1", rating=5),
        Recommendation(login_id="LoginID_3", book_id="Book_2", rating=5),
    ]
    
    print(f"\n✓ {len(customers)} customers creados")
    print(f"✓ {len(books)} books creados")
    print(f"✓ {len(recommendations)} recommendations creadas")
    
    # Ground el modelo
    print("\n" + "="*70)
    print("  GROUNDING RPM → BAYES NET")
    print("="*70)
    
    net = model.ground_model(customers, books, recommendations)
    net.print_structure()
    
    # Mostrar algunas variables
    print("\n" + "="*70)
    print("  VARIABLES GROUNDED (muestra)")
    print("="*70)
    
    for var_name in list(net.variables.keys())[:5]:
        var = net.variables[var_name]
        parents_str = f" | {', '.join(var.parents)}" if var.parents else ""
        print(f"  {var.name}{parents_str}")
    
    print(f"\n  ... y {len(net.variables) - 5} más")
    
    # Calcular probabilidad de un mundo
    print("\n" + "="*70)
    print("  CÁLCULO DE P(ω) PARA UN MUNDO")
    print("="*70)
    
    # Definir un mundo posible
    assignment = {
        "Quality_Book_1": 5,
        "Quality_Book_2": 2,
        "Quality_Book_3": 4,
        "Honest_User_1": True,
        "Honest_User_2": True,
        "Honest_Bot_1": False,
        "Rec_User_1_Book_1": 5,
        "Rec_User_1_Book_2": 2,
        "Rec_User_2_Book_1": 4,
        "Rec_Bot_1_Book_1": 5,
        "Rec_Bot_1_Book_2": 5
    }
    
    prob = model.compute_world_probability(customers, books, assignment)
    print(f"\nP(mundo) = {prob:.8f}")
    print(f"log P(mundo) = {prob:.2e}")
    
    # Mostrar algunas CPT entries usadas
    print("\n" + "="*70)
    print("  FACTORES USADOS EN EL CÁLCULO")
    print("="*70)
    
    rec_cpt = model.cpts["recommendation"]
    print(f"\nP(Rec=5 | Quality=5, Honest=True) = {rec_cpt.get_rating_probability(5, True, 5):.4f}")
    print(f"P(Rec=2 | Quality=2, Honest=True) = {rec_cpt.get_rating_probability(2, True, 2):.4f}")
    print(f"P(Rec=5 | Quality=2, Honest=False) = {rec_cpt.get_rating_probability(2, False, 5):.4f}")
    
    quality_cpt = model.cpts["quality"]
    print(f"\nP(Quality=5) = {quality_cpt.get_quality_probability(5):.4f}")
    print(f"P(Quality=2) = {quality_cpt.get_quality_probability(2):.4f}")
    
    honesty_cpt = model.cpts["honesty"]
    print(f"\nP(Honest=True) = {honesty_cpt.get_honesty_probability(True):.4f}")
    print(f"P(Honest=False) = {honesty_cpt.get_honesty_probability(False):.4f}")
    
    print("\n✓ DEMO completado!")


if __name__ == "__main__":
    demo_rpm_model()
