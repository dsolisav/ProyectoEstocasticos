"""
Conditional Probability Tables (CPTs) para el modelo RPM.

Implementa las distribuciones de probabilidad condicional del modelo
de recomendación con detección de bots basado en el Capítulo 18.
"""

from typing import Dict, Tuple, List
import sys


class CPT:
    """
    Clase base para Conditional Probability Tables.
    
    Una CPT representa P(X | Parents(X)) como una tabla.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.table = {}  # {(parent_values...): {child_value: probability}}
    
    def set_probability(self, parent_values: Tuple, child_value, probability: float):
        """
        Establece P(child_value | parent_values)
        
        Args:
            parent_values: Tupla con valores de los padres
            child_value: Valor del hijo
            probability: Probabilidad condicional
        """
        if parent_values not in self.table:
            self.table[parent_values] = {}
        self.table[parent_values][child_value] = probability
    
    def get_probability(self, parent_values: Tuple, child_value) -> float:
        """
        Obtiene P(child_value | parent_values)
        
        Returns:
            Probabilidad condicional
        """
        if parent_values not in self.table:
            raise KeyError(f"Parent values {parent_values} not found in CPT {self.name}")
        
        if child_value not in self.table[parent_values]:
            raise KeyError(f"Child value {child_value} not found for parents {parent_values}")
        
        return self.table[parent_values][child_value]
    
    def get_distribution(self, parent_values: Tuple) -> Dict:
        """
        Obtiene la distribución completa P(X | parent_values)
        
        Returns:
            Diccionario {child_value: probability}
        """
        if parent_values not in self.table:
            raise KeyError(f"Parent values {parent_values} not found in CPT {self.name}")
        
        return self.table[parent_values].copy()
    
    def validate(self) -> bool:
        """
        Valida que todas las distribuciones sumen 1.0
        
        Returns:
            True si válida, False si no
        """
        for parent_values, dist in self.table.items():
            total = sum(dist.values())
            if abs(total - 1.0) > 1e-6:
                print(f"Warning: CPT {self.name} with parents {parent_values} sums to {total}", 
                      file=sys.stderr)
                return False
        return True
    
    def __repr__(self):
        return f"CPT({self.name}, {len(self.table)} entries)"


class RecommendationCPT(CPT):
    """
    CPT para Recommendation(customer, book).
    
    Basado en la Figura 18.2(a) del capítulo:
    P(Rec(c,b) | Quality(b), Honest(c))
    
    La tabla tiene 50 filas (10 configuraciones × 5 valores de rating).
    """
    
    def __init__(self):
        super().__init__("Recommendation")
        self._build_table()
    
    def _build_table(self):
        """
        Construye la tabla de probabilidades condicionales.
        
        Modelo del capítulo:
        - Si Honest(c) = True: rating cercano a Quality(b)
        - Si Honest(c) = False: rating aleatorio (uniforme)
        """
        # Para cada combinación de (Quality, Honest)
        for quality in [1, 2, 3, 4, 5]:
            # Usuario honesto: rating sigue calidad con ruido gaussiano
            honest_dist = self._honest_distribution(quality)
            self.table[(quality, True)] = honest_dist
            
            # Usuario deshonesto: rating uniforme
            dishonest_dist = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}
            self.table[(quality, False)] = dishonest_dist
    
    def _honest_distribution(self, quality: int) -> Dict[int, float]:
        """
        Distribución para usuario honesto dado calidad del libro.
        
        Modelo: rating cercano a calidad con ruido.
        Por ejemplo, si quality=4:
        - rating=4 más probable
        - rating=3,5 menos probable
        - rating=1,2 muy poco probable
        
        Args:
            quality: Calidad real del libro (1-5)
        
        Returns:
            Distribución sobre ratings {1,2,3,4,5}
        """
        dist = {}
        
        # Distribución centrada en quality con decaimiento
        for rating in [1, 2, 3, 4, 5]:
            distance = abs(rating - quality)
            
            if distance == 0:
                prob = 0.6  # 60% de dar rating exacto
            elif distance == 1:
                prob = 0.15  # 15% de dar ±1
            elif distance == 2:
                prob = 0.05  # 5% de dar ±2
            else:  # distance >= 3
                prob = 0.025  # 2.5% de dar ±3 o más
            
            dist[rating] = prob
        
        # Normalizar para asegurar que sume 1.0
        total = sum(dist.values())
        dist = {k: v/total for k, v in dist.items()}
        
        return dist
    
    def get_rating_probability(self, quality: int, honest: bool, rating: int) -> float:
        """
        Conveniencia: P(Rec=rating | Quality=quality, Honest=honest)
        
        Args:
            quality: Calidad del libro (1-5)
            honest: Si el usuario es honesto
            rating: Rating dado (1-5)
        
        Returns:
            Probabilidad condicional
        """
        return self.get_probability((quality, honest), rating)


class QualityCPT(CPT):
    """
    CPT para Quality(book).
    
    Prior sobre la calidad real de los libros.
    P(Quality(b)) - sin padres, es un prior.
    """
    
    def __init__(self, prior_type: str = "uniform"):
        """
        Args:
            prior_type: Tipo de prior
                - "uniform": Distribución uniforme
                - "optimistic": Sesgada hacia alta calidad
                - "pessimistic": Sesgada hacia baja calidad
                - "realistic": Distribución realista (más libros medianos)
        """
        super().__init__("Quality")
        self.prior_type = prior_type
        self._build_table()
    
    def _build_table(self):
        """Construye la tabla de priors"""
        # Quality no tiene padres, así que usamos () como key
        
        if self.prior_type == "uniform":
            dist = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}
        
        elif self.prior_type == "optimistic":
            # Más libros de alta calidad
            dist = {1: 0.05, 2: 0.10, 3: 0.20, 4: 0.30, 5: 0.35}
        
        elif self.prior_type == "pessimistic":
            # Más libros de baja calidad
            dist = {1: 0.35, 2: 0.30, 3: 0.20, 4: 0.10, 5: 0.05}
        
        elif self.prior_type == "realistic":
            # Distribución realista: mayoría en rango medio-alto
            dist = {1: 0.10, 2: 0.15, 3: 0.30, 4: 0.30, 5: 0.15}
        
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")
        
        self.table[()] = dist
    
    def get_quality_probability(self, quality: int) -> float:
        """
        Conveniencia: P(Quality=quality)
        
        Args:
            quality: Calidad (1-5)
        
        Returns:
            Probabilidad prior
        """
        return self.get_probability((), quality)
    
    def get_quality_distribution(self) -> Dict[int, float]:
        """
        Obtiene la distribución completa P(Quality)
        
        Returns:
            Diccionario {quality: probability}
        """
        return self.get_distribution(())


class HonestyCPT(CPT):
    """
    CPT para Honest(customer).
    
    Prior sobre la honestidad de los usuarios.
    P(Honest(c)) - sin padres, es un prior.
    
    Nota: En el modelo completo OUPM, esto podría depender del
    tipo de usuario (real vs. bot), pero aquí es un prior simple.
    """
    
    def __init__(self, honest_prob: float = 0.7):
        """
        Args:
            honest_prob: Probabilidad prior de que un usuario sea honesto
                        (por defecto 70%)
        """
        super().__init__("Honesty")
        self.honest_prob = honest_prob
        self._build_table()
    
    def _build_table(self):
        """Construye la tabla de priors"""
        # Honesty no tiene padres
        self.table[()] = {
            True: self.honest_prob,
            False: 1.0 - self.honest_prob
        }
    
    def get_honesty_probability(self, honest: bool) -> float:
        """
        Conveniencia: P(Honest=honest)
        
        Args:
            honest: True o False
        
        Returns:
            Probabilidad prior
        """
        return self.get_probability((), honest)
    
    def get_honesty_distribution(self) -> Dict[bool, float]:
        """
        Obtiene la distribución completa P(Honest)
        
        Returns:
            Diccionario {True: prob_honest, False: prob_dishonest}
        """
        return self.get_distribution(())


class EntityTypeCPT(CPT):
    """
    CPT para EntityType(customer).
    
    Prior sobre si un customer es usuario real o bot.
    P(EntityType(c))
    
    Esta es una extensión del modelo básico del capítulo para
    manejar bots explícitamente.
    """
    
    def __init__(self, bot_probability: float = 0.15):
        """
        Args:
            bot_probability: Probabilidad prior de que un customer sea bot
                           (por defecto 15%)
        """
        super().__init__("EntityType")
        self.bot_probability = bot_probability
        self._build_table()
    
    def _build_table(self):
        """Construye la tabla de priors"""
        self.table[()] = {
            "real_user": 1.0 - self.bot_probability,
            "bot": self.bot_probability
        }
    
    def get_type_probability(self, entity_type: str) -> float:
        """
        Conveniencia: P(EntityType=entity_type)
        
        Args:
            entity_type: "real_user" o "bot"
        
        Returns:
            Probabilidad prior
        """
        return self.get_probability((), entity_type)


class ConditionalHonestyCPT(CPT):
    """
    CPT extendida: P(Honest(c) | EntityType(c))
    
    La honestidad depende del tipo de entidad:
    - Usuarios reales: típicamente honestos
    - Bots: típicamente deshonestos
    
    Esta es una mejora del modelo que conecta honestidad con tipo.
    """
    
    def __init__(self, 
                 user_honest_prob: float = 0.8,
                 bot_honest_prob: float = 0.1):
        """
        Args:
            user_honest_prob: P(Honest=True | EntityType=real_user)
            bot_honest_prob: P(Honest=True | EntityType=bot)
        """
        super().__init__("ConditionalHonesty")
        self.user_honest_prob = user_honest_prob
        self.bot_honest_prob = bot_honest_prob
        self._build_table()
    
    def _build_table(self):
        """Construye la tabla condicional"""
        # P(Honest | EntityType=real_user)
        self.table[("real_user",)] = {
            True: self.user_honest_prob,
            False: 1.0 - self.user_honest_prob
        }
        
        # P(Honest | EntityType=bot)
        self.table[("bot",)] = {
            True: self.bot_honest_prob,
            False: 1.0 - self.bot_honest_prob
        }
    
    def get_honesty_given_type(self, entity_type: str, honest: bool) -> float:
        """
        Conveniencia: P(Honest=honest | EntityType=entity_type)
        
        Args:
            entity_type: "real_user" o "bot"
            honest: True o False
        
        Returns:
            Probabilidad condicional
        """
        return self.get_probability((entity_type,), honest)


def create_default_cpts() -> Dict[str, CPT]:
    """
    Crea las CPTs por defecto para el modelo RPM.
    
    Returns:
        Diccionario con todas las CPTs necesarias
    """
    return {
        "recommendation": RecommendationCPT(),
        "quality": QualityCPT(prior_type="realistic"),
        "honesty": HonestyCPT(honest_prob=0.7),
        "entity_type": EntityTypeCPT(bot_probability=0.15),
        "conditional_honesty": ConditionalHonestyCPT(
            user_honest_prob=0.8,
            bot_honest_prob=0.1
        )
    }


def print_cpt_summary(cpt: CPT, max_rows: int = 10):
    """
    Imprime un resumen de una CPT de forma legible.
    
    Args:
        cpt: CPT a mostrar
        max_rows: Máximo número de filas a mostrar
    """
    print(f"\n{'='*60}")
    print(f"CPT: {cpt.name}")
    print(f"{'='*60}")
    
    rows = list(cpt.table.items())
    
    for i, (parent_values, dist) in enumerate(rows[:max_rows]):
        if len(parent_values) == 0:
            print(f"\nP({cpt.name}):")
        else:
            print(f"\nP({cpt.name} | {parent_values}):")
        
        for value, prob in sorted(dist.items()):
            bar_length = int(prob * 40)
            bar = "█" * bar_length
            print(f"  {str(value):10s}: {prob:.4f} {bar}")
    
    if len(rows) > max_rows:
        print(f"\n... ({len(rows) - max_rows} more rows)")
    
    # Validar
    if cpt.validate():
        print(f"\n✓ CPT válida (todas las distribuciones suman 1.0)")
    else:
        print(f"\n✗ CPT inválida (algunas distribuciones no suman 1.0)")


if __name__ == "__main__":
    # Demo: Crear y mostrar CPTs
    print("="*70)
    print("  CONDITIONAL PROBABILITY TABLES (CPTs)")
    print("="*70)
    
    cpts = create_default_cpts()
    
    # Mostrar cada CPT
    print_cpt_summary(cpts["quality"])
    print_cpt_summary(cpts["honesty"])
    print_cpt_summary(cpts["entity_type"])
    print_cpt_summary(cpts["conditional_honesty"])
    print_cpt_summary(cpts["recommendation"], max_rows=5)
    
    # Ejemplos de uso
    print("\n" + "="*70)
    print("  EJEMPLOS DE CONSULTAS")
    print("="*70)
    
    rec_cpt = cpts["recommendation"]
    
    print("\nP(Rating=5 | Quality=5, Honest=True):", 
          rec_cpt.get_rating_probability(5, True, 5))
    print("P(Rating=5 | Quality=1, Honest=True):", 
          rec_cpt.get_rating_probability(1, True, 5))
    print("P(Rating=5 | Quality=3, Honest=False):", 
          rec_cpt.get_rating_probability(3, False, 5))
    
    quality_cpt = cpts["quality"]
    print("\nP(Quality=4):", quality_cpt.get_quality_probability(4))
    
    honesty_cpt = cpts["honesty"]
    print("\nP(Honest=True):", honesty_cpt.get_honesty_probability(True))
