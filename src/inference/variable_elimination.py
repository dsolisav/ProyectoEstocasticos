"""
Variable Elimination Algorithm (Capítulo 18.3)

Inferencia exacta en redes bayesianas mediante eliminación de variables.
Implementación desde cero sin librerías externas.
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import itertools


@dataclass
class Factor:
    """
    Factor en una red bayesiana: φ(X₁, X₂, ..., Xₙ)
    Representa una tabla de probabilidades sobre un conjunto de variables.
    """
    variables: List[str]  # Variables en el factor
    values: Dict[Tuple, float]  # (val1, val2, ...) → probability
    
    def __post_init__(self):
        """Normalizar si las probabilidades no suman 1."""
        total = sum(self.values.values())
        if total > 0 and abs(total - 1.0) > 1e-6:
            # No normalizar automáticamente - puede ser un factor no normalizado
            pass
    
    def get_probability(self, assignment: Dict[str, any]) -> float:
        """
        Obtener P(...) dado un assignment completo de las variables del factor.
        
        Args:
            assignment: Dict con valores para cada variable del factor
            
        Returns:
            Probabilidad del assignment
        """
        key = tuple(assignment[var] for var in self.variables)
        return self.values.get(key, 0.0)
    
    def marginalize(self, variable: str) -> 'Factor':
        """
        Sumar sobre una variable: φ'(X₁, ..., Xₙ) = Σ_v φ(X₁, ..., Xₙ, v)
        
        Args:
            variable: Variable a eliminar
            
        Returns:
            Nuevo factor sin la variable eliminada
        """
        if variable not in self.variables:
            return self
        
        # Índice de la variable a eliminar
        var_idx = self.variables.index(variable)
        
        # Nuevas variables (sin la eliminada)
        new_variables = [v for v in self.variables if v != variable]
        
        # Sumar sobre todos los valores de la variable eliminada
        new_values = {}
        for assignment, prob in self.values.items():
            # Crear key sin la variable eliminada
            new_key = tuple(val for i, val in enumerate(assignment) 
                          if i != var_idx)
            new_values[new_key] = new_values.get(new_key, 0.0) + prob
        
        return Factor(new_variables, new_values)
    
    def restrict(self, variable: str, value: any) -> 'Factor':
        """
        Fijar una variable a un valor específico (evidencia).
        φ'(...) = φ(..., variable=value)
        
        Args:
            variable: Variable a fijar
            value: Valor de la evidencia
            
        Returns:
            Nuevo factor con la variable fijada
        """
        if variable not in self.variables:
            return self
        
        var_idx = self.variables.index(variable)
        new_variables = [v for v in self.variables if v != variable]
        
        new_values = {}
        for assignment, prob in self.values.items():
            if assignment[var_idx] == value:
                new_key = tuple(val for i, val in enumerate(assignment) 
                              if i != var_idx)
                new_values[new_key] = prob
        
        return Factor(new_variables, new_values)
    
    def __mul__(self, other: 'Factor') -> 'Factor':
        """
        Join de factores: φ₁(X, Y) * φ₂(Y, Z) = φ₃(X, Y, Z)
        """
        # Variables del producto (unión sin duplicados, manteniendo orden)
        new_variables = self.variables.copy()
        for var in other.variables:
            if var not in new_variables:
                new_variables.append(var)
        
        # Calcular todas las combinaciones
        new_values = {}
        
        # Obtener todos los valores únicos por variable
        all_values = {}
        for var in new_variables:
            values_set = set()
            if var in self.variables:
                idx = self.variables.index(var)
                for assignment in self.values.keys():
                    values_set.add(assignment[idx])
            if var in other.variables:
                idx = other.variables.index(var)
                for assignment in other.values.keys():
                    values_set.add(assignment[idx])
            all_values[var] = sorted(values_set)
        
        # Generar todas las combinaciones
        for combo in itertools.product(*[all_values[var] for var in new_variables]):
            assignment = {var: val for var, val in zip(new_variables, combo)}
            
            # Probabilidad = producto de factores
            prob1 = self.get_probability(assignment)
            prob2 = other.get_probability(assignment)
            
            if prob1 > 0 and prob2 > 0:  # Solo incluir si ambos > 0
                new_values[combo] = prob1 * prob2
        
        return Factor(new_variables, new_values)
    
    def normalize(self) -> 'Factor':
        """Normalizar el factor para que sume 1."""
        total = sum(self.values.values())
        if total == 0:
            return self
        
        normalized_values = {k: v/total for k, v in self.values.items()}
        return Factor(self.variables, normalized_values)


class VariableElimination:
    """
    Algoritmo de Variable Elimination para inferencia exacta.
    
    Calcula P(Q | E) donde:
    - Q: variables query
    - E: evidencia (variables observadas)
    
    Algoritmo:
    1. Inicializar factores con CPTs de la red
    2. Restringir factores según evidencia
    3. Para cada variable no-query:
       - Join de todos los factores que contienen la variable
       - Marginalizar la variable
    4. Join de factores restantes
    5. Normalizar
    """
    
    def __init__(self, grounded_network):
        """
        Args:
            grounded_network: GroundedBayesNet con la red proposicional
        """
        self.network = grounded_network
        
    def _create_factors_from_network(self) -> List[Factor]:
        """
        Crear factores a partir de las CPTs de la red.
        Cada variable Xi con padres Pa(Xi) genera factor φ(Xi, Pa(Xi)).
        """
        factors = []
        
        for var_name, var in self.network.variables.items():
            # Variables del factor: [variable] + parents
            factor_vars = [var_name] + list(var.parents)
            
            # Obtener todos los valores posibles
            all_values = {}
            for v in factor_vars:
                net_var = self.network.variables[v]
                all_values[v] = net_var.domain
            
            # Generar tabla de probabilidades
            factor_values = {}
            for combo in itertools.product(*[all_values[v] for v in factor_vars]):
                assignment = {v: val for v, val in zip(factor_vars, combo)}
                
                # P(var | parents) - usar el método específico del CPT
                parent_dict = {p: assignment[p] for p in var.parents}
                
                # Intentar obtener la probabilidad del CPT
                try:
                    # Algunos CPTs requieren parent_values como tupla ordenada
                    if len(var.parents) == 0:
                        # Sin padres - prior
                        prob = var.cpt.get_probability((), assignment[var_name])
                    else:
                        # Con padres - convertir parent_dict a la forma que espera el CPT
                        parent_tuple = tuple(parent_dict[p] for p in var.parents)
                        prob = var.cpt.get_probability(parent_tuple, assignment[var_name])
                except (KeyError, AttributeError, TypeError) as e:
                    # Si falla, usar distribución uniforme
                    prob = 1.0 / len(var.domain)
                
                factor_values[combo] = prob
            
            factors.append(Factor(factor_vars, factor_values))
        
        return factors
    
    def _get_elimination_order(self, 
                              query_vars: List[str], 
                              evidence_vars: List[str]) -> List[str]:
        """
        Heurística: eliminar variables en orden de menor a mayor número
        de vecinos (min-neighbors heuristic).
        
        Args:
            query_vars: Variables query (no eliminar)
            evidence_vars: Variables evidencia (no eliminar)
            
        Returns:
            Orden de eliminación
        """
        # Variables a eliminar = todas - query - evidencia
        to_eliminate = set(self.network.variables.keys())
        to_eliminate -= set(query_vars)
        to_eliminate -= set(evidence_vars)
        
        # Por simplicidad, orden arbitrario (puede mejorarse)
        return sorted(to_eliminate)
    
    def infer(self, 
              query_vars: List[str], 
              evidence: Dict[str, any]) -> Factor:
        """
        Calcular P(Q | E) usando Variable Elimination.
        
        Args:
            query_vars: Lista de variables query
            evidence: Dict {variable: observed_value}
            
        Returns:
            Factor con la distribución P(Q | E)
        """
        # 1. Crear factores iniciales
        factors = self._create_factors_from_network()
        
        # 2. Restringir según evidencia
        factors = [f.restrict(var, val) for f in factors 
                  for var, val in evidence.items()]
        
        # 3. Obtener orden de eliminación
        elimination_order = self._get_elimination_order(
            query_vars, 
            list(evidence.keys())
        )
        
        # 4. Eliminar variables una por una
        for var_to_eliminate in elimination_order:
            # Factores que contienen esta variable
            relevant = [f for f in factors if var_to_eliminate in f.variables]
            others = [f for f in factors if var_to_eliminate not in f.variables]
            
            if not relevant:
                continue
            
            # Join de factores relevantes
            joined = relevant[0]
            for factor in relevant[1:]:
                joined = joined * factor
            
            # Marginalizar variable
            marginalized = joined.marginalize(var_to_eliminate)
            
            # Actualizar lista de factores
            factors = others + [marginalized]
        
        # 5. Join de factores restantes
        if not factors:
            # No hay factores (todas las variables fueron eliminadas?)
            return Factor(query_vars, {tuple([True]*len(query_vars)): 1.0})
        
        result = factors[0]
        for factor in factors[1:]:
            result = result * factor
        
        # 6. Normalizar
        return result.normalize()
    
    def compute_marginal(self, variable: str, evidence: Dict[str, any] = None) -> Dict[any, float]:
        """
        Calcular distribución marginal P(variable | evidence).
        
        Args:
            variable: Variable de interés
            evidence: Evidencia opcional
            
        Returns:
            Dict {value: probability}
        """
        if evidence is None:
            evidence = {}
        
        result_factor = self.infer([variable], evidence)
        
        # Convertir factor a dict
        distribution = {}
        for assignment, prob in result_factor.values.items():
            value = assignment[0]  # Solo una variable
            distribution[value] = prob
        
        return distribution
    
    def compute_joint(self, 
                     variables: List[str], 
                     evidence: Dict[str, any] = None) -> Factor:
        """
        Calcular distribución conjunta P(variables | evidence).
        
        Args:
            variables: Lista de variables
            evidence: Evidencia opcional
            
        Returns:
            Factor con la distribución conjunta
        """
        if evidence is None:
            evidence = {}
        
        return self.infer(variables, evidence)
