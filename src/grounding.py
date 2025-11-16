"""
Grounding: Conversión de RPM (Relational Probability Model) a Bayes Net.

El proceso de "grounding" o "unrolling" convierte un modelo relacional
compacto en una red bayesiana completamente instanciada.

Conceptos del Capítulo 18:
- Type signatures (Customer, Book)
- Predicados relacionales se convierten en variables proposicionales
- Database semantics: cada objeto tiene un nombre único
"""

from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models import Customer, Book, LoginID, Recommendation
from cpt import CPT, RecommendationCPT, QualityCPT, HonestyCPT


@dataclass
class BayesNetVariable:
    """
    Una variable en la red bayesiana grounded.
    
    Por ejemplo:
    - Quality(Book_1) → variable proposicional con dominio {1,2,3,4,5}
    - Honest(User_3) → variable proposicional con dominio {True, False}
    - Rec(LoginID_5, Book_2) → variable con dominio {1,2,3,4,5}
    """
    name: str                    # Ej: "Quality_Book_1"
    var_type: str               # "quality", "honesty", "recommendation"
    objects: Tuple[str, ...]    # Objetos involucrados, ej: ("Book_1",)
    domain: List[Any]           # Posibles valores
    parents: List[str] = field(default_factory=list)  # Variables padre
    cpt: CPT = None            # CPT asociada
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if not isinstance(other, BayesNetVariable):
            return False
        return self.name == other.name
    
    def __repr__(self):
        parent_str = f", parents={self.parents}" if self.parents else ""
        return f"Var({self.name}{parent_str})"


@dataclass
class GroundedBayesNet:
    """
    Red bayesiana completamente grounded.
    
    Representa el resultado de aplicar grounding a un RPM
    para un conjunto específico de objetos.
    """
    variables: Dict[str, BayesNetVariable] = field(default_factory=dict)
    observations: Dict[str, Any] = field(default_factory=dict)  # Evidencia
    
    def add_variable(self, var: BayesNetVariable):
        """Agrega una variable a la red"""
        self.variables[var.name] = var
    
    def get_variable(self, name: str) -> BayesNetVariable:
        """Obtiene una variable por nombre"""
        return self.variables.get(name)
    
    def set_observation(self, var_name: str, value: Any):
        """Establece evidencia para una variable"""
        if var_name not in self.variables:
            raise KeyError(f"Variable {var_name} not in network")
        self.observations[var_name] = value
    
    def get_parents(self, var_name: str) -> List[BayesNetVariable]:
        """Obtiene las variables padre de una variable"""
        var = self.variables[var_name]
        return [self.variables[p] for p in var.parents]
    
    def get_children(self, var_name: str) -> List[BayesNetVariable]:
        """Obtiene las variables hijo de una variable"""
        children = []
        for other_var in self.variables.values():
            if var_name in other_var.parents:
                children.append(other_var)
        return children
    
    def compute_probability(self, assignment: Dict[str, Any]) -> float:
        """
        Calcula P(assignment) usando la estructura de la red.
        
        P(X1,...,Xn) = ∏ P(Xi | Parents(Xi))
        
        Args:
            assignment: Asignación completa {var_name: value}
        
        Returns:
            Probabilidad del assignment
        """
        prob = 1.0
        
        for var_name, value in assignment.items():
            # Skip variables que no están en la red
            if var_name not in self.variables:
                continue
            
            var = self.variables[var_name]
            
            if var.cpt is None:
                continue  # Skip variables sin CPT
            
            # Obtener valores de los padres
            parent_values = tuple(assignment[p] for p in var.parents)
            
            # Multiplicar por P(value | parent_values)
            try:
                p_conditional = var.cpt.get_probability(parent_values, value)
                prob *= p_conditional
            except KeyError:
                # Si no hay entrada en la CPT, probabilidad 0
                return 0.0
        
        return prob
    
    def get_markov_blanket(self, var_name: str) -> Set[str]:
        """
        Obtiene el Markov Blanket de una variable.
        
        El Markov Blanket incluye:
        - Padres
        - Hijos
        - Padres de los hijos (co-padres)
        
        Args:
            var_name: Nombre de la variable
        
        Returns:
            Set con nombres de variables en el Markov Blanket
        """
        blanket = set()
        var = self.variables[var_name]
        
        # Padres
        blanket.update(var.parents)
        
        # Hijos y sus otros padres
        for child in self.get_children(var_name):
            blanket.add(child.name)
            blanket.update(child.parents)
        
        # Remover la variable misma
        blanket.discard(var_name)
        
        return blanket
    
    def print_structure(self):
        """Imprime la estructura de la red"""
        print(f"\n{'='*60}")
        print(f"GROUNDED BAYES NET")
        print(f"{'='*60}")
        print(f"Variables: {len(self.variables)}")
        print(f"Observations: {len(self.observations)}")
        
        # Agrupar por tipo
        by_type = {}
        for var in self.variables.values():
            if var.var_type not in by_type:
                by_type[var.var_type] = []
            by_type[var.var_type].append(var)
        
        for var_type, vars_list in by_type.items():
            print(f"\n{var_type.upper()}: {len(vars_list)} variables")
            for var in vars_list[:3]:  # Mostrar primeras 3
                print(f"  {var}")
            if len(vars_list) > 3:
                print(f"  ... ({len(vars_list) - 3} more)")


class RPMGrounder:
    """
    Clase para hacer grounding de un RPM a una Bayes Net.
    
    Dado:
    - Conjunto de customers
    - Conjunto de books
    - CPTs del modelo
    
    Produce:
    - Red bayesiana completamente grounded
    """
    
    def __init__(self, cpts: Dict[str, CPT]):
        """
        Args:
            cpts: Diccionario con las CPTs del modelo
        """
        self.cpts = cpts
    
    def ground_rpm(
        self,
        customers: List[Customer],
        books: List[Book],
        recommendations: List[Recommendation] = None
    ) -> GroundedBayesNet:
        """
        Hace grounding del RPM completo.
        
        Args:
            customers: Lista de customers
            books: Lista de books
            recommendations: Lista de recommendations (evidencia)
        
        Returns:
            Red bayesiana grounded
        """
        net = GroundedBayesNet()
        
        # 1. Crear variables Quality(b) para cada libro
        for book in books:
            self._add_quality_variable(net, book)
        
        # 2. Crear variables Honest(c) para cada customer
        for customer in customers:
            self._add_honesty_variable(net, customer)
        
        # 3. Crear variables Rec(c,b) para cada par que tiene recomendación
        if recommendations:
            for rec in recommendations:
                # Encontrar customer correspondiente al LoginID
                customer = self._find_customer_for_login(customers, rec.login_id)
                if customer:
                    self._add_recommendation_variable(net, customer, rec.book_id)
        
        # 4. Agregar evidencia de las recomendaciones
        if recommendations:
            for rec in recommendations:
                customer = self._find_customer_for_login(customers, rec.login_id)
                if customer:
                    var_name = self._rec_var_name(customer.customer_id, rec.book_id)
                    net.set_observation(var_name, rec.rating)
        
        return net
    
    def _add_quality_variable(self, net: GroundedBayesNet, book: Book):
        """Agrega variable Quality(book) a la red"""
        var = BayesNetVariable(
            name=f"Quality_{book.book_id}",
            var_type="quality",
            objects=(book.book_id,),
            domain=[1, 2, 3, 4, 5],
            parents=[],
            cpt=self.cpts.get("quality")
        )
        net.add_variable(var)
    
    def _add_honesty_variable(self, net: GroundedBayesNet, customer: Customer):
        """Agrega variable Honest(customer) a la red"""
        var = BayesNetVariable(
            name=f"Honest_{customer.customer_id}",
            var_type="honesty",
            objects=(customer.customer_id,),
            domain=[True, False],
            parents=[],
            cpt=self.cpts.get("honesty")
        )
        net.add_variable(var)
    
    def _add_recommendation_variable(
        self, 
        net: GroundedBayesNet, 
        customer: Customer,
        book_id: str
    ):
        """Agrega variable Rec(customer, book) a la red"""
        # Padres: Quality(book) y Honest(customer)
        quality_var = f"Quality_{book_id}"
        honesty_var = f"Honest_{customer.customer_id}"
        
        var = BayesNetVariable(
            name=self._rec_var_name(customer.customer_id, book_id),
            var_type="recommendation",
            objects=(customer.customer_id, book_id),
            domain=[1, 2, 3, 4, 5],
            parents=[quality_var, honesty_var],
            cpt=self.cpts.get("recommendation")
        )
        net.add_variable(var)
    
    def _rec_var_name(self, customer_id: str, book_id: str) -> str:
        """Genera nombre de variable para Rec(customer, book)"""
        return f"Rec_{customer_id}_{book_id}"
    
    def _find_customer_for_login(
        self, 
        customers: List[Customer], 
        login_id: str
    ) -> Customer:
        """
        Encuentra el customer correspondiente a un LoginID.
        
        Nota: En el modelo básico RPM, asumimos que conocemos
        el mapeo LoginID → Customer. En OUPM esto es incierto.
        """
        # Por ahora, extraer customer_id del login_id
        # En el dataset, LoginIDs tienen .origin que es el Customer
        # Aquí simplificamos buscando por customer_id en el login_id
        for customer in customers:
            # Esta es una simplificación; en código real usaríamos
            # el mapeo explícito de LoginID.origin
            pass
        
        # Retornar primer customer por simplicidad en esta versión
        # En FASE 3 (OUPM) manejaremos esto correctamente
        return customers[0] if customers else None


def ground_simple_example():
    """
    Ejemplo simple de grounding.
    
    1 usuario, 2 libros, 2 recomendaciones.
    """
    print("\n" + "="*70)
    print("  EJEMPLO DE GROUNDING: RPM → BAYES NET")
    print("="*70)
    
    # Importar CPTs
    from cpt import create_default_cpts
    
    # Crear objetos
    user1 = Customer(customer_id="User_1", honesty=0.8)
    book1 = Book(book_id="Book_1", true_quality=4)
    book2 = Book(book_id="Book_2", true_quality=2)
    
    rec1 = Recommendation(login_id="LoginID_1", book_id="Book_1", rating=4)
    rec2 = Recommendation(login_id="LoginID_1", book_id="Book_2", rating=2)
    
    # Crear grounder
    cpts = create_default_cpts()
    grounder = RPMGrounder(cpts)
    
    # Ground
    net = grounder.ground_rpm(
        customers=[user1],
        books=[book1, book2],
        recommendations=[rec1, rec2]
    )
    
    # Mostrar estructura
    net.print_structure()
    
    print(f"\n{'='*60}")
    print("VARIABLES EN DETALLE:")
    print(f"{'='*60}")
    
    for var_name, var in sorted(net.variables.items()):
        print(f"\n{var_name}:")
        print(f"  Type: {var.var_type}")
        print(f"  Domain: {var.domain}")
        print(f"  Parents: {var.parents if var.parents else 'None (root)'}")
    
    print(f"\n{'='*60}")
    print("EVIDENCIA:")
    print(f"{'='*60}")
    
    for var_name, value in net.observations.items():
        print(f"  {var_name} = {value}")
    
    # Calcular probabilidad de un assignment
    print(f"\n{'='*60}")
    print("EJEMPLO DE CÁLCULO DE PROBABILIDAD:")
    print(f"{'='*60}")
    
    assignment = {
        "Quality_Book_1": 4,
        "Quality_Book_2": 2,
        "Honest_User_1": True,
        "Rec_User_1_Book_1": 4,
        "Rec_User_1_Book_2": 2
    }
    
    prob = net.compute_probability(assignment)
    print(f"\nP(assignment) = {prob:.6f}")
    print("\nDesglose:")
    print(f"  P(Quality_Book_1=4) = {cpts['quality'].get_quality_probability(4):.4f}")
    print(f"  P(Quality_Book_2=2) = {cpts['quality'].get_quality_probability(2):.4f}")
    print(f"  P(Honest_User_1=True) = {cpts['honesty'].get_honesty_probability(True):.4f}")
    print(f"  P(Rec=4 | Quality=4, Honest=True) = {cpts['recommendation'].get_rating_probability(4, True, 4):.4f}")
    print(f"  P(Rec=2 | Quality=2, Honest=True) = {cpts['recommendation'].get_rating_probability(2, True, 2):.4f}")


if __name__ == "__main__":
    ground_simple_example()
