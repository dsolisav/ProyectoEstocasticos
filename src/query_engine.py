"""
Query Engine para Inferencia Probabilística

Sistema para responder consultas del tipo:
- P(variable | evidence)
- P(φ | e) donde φ es una fórmula lógica
- Queries condicionales usando MCMC
"""

from typing import Dict, List, Callable, Any, Optional, Tuple
from dataclasses import dataclass
import statistics

from .inference.gibbs_sampling import GibbsSampling, GibbsSample
from .inference.metropolis_hastings import MetropolisHastings, MHSample
from .grounding import GroundedBayesNet


@dataclass
class QueryResult:
    """Resultado de una consulta probabilística."""
    query_vars: List[str]
    evidence: Dict[str, Any]
    distribution: Dict[Any, float]
    method: str  # 'gibbs' o 'metropolis-hastings'
    num_samples: int
    confidence_interval: Optional[Tuple[float, float]] = None


class QueryEngine:
    """
    Motor de consultas para inferencia probabilística.
    
    Permite hacer queries del tipo:
    - P(Quality(Book_1) | recommendations)
    - P(IsBot(User_3) | all evidence)
    - P(Honest(User_2) = True | Quality(Book_1) = 5)
    """
    
    def __init__(self, grounded_network: GroundedBayesNet):
        """
        Args:
            grounded_network: Red bayesiana grounded
        """
        self.network = grounded_network
        self.gibbs = GibbsSampling(grounded_network)
        self.mh = MetropolisHastings(grounded_network)
    
    def query_marginal(self,
                      variable: str,
                      evidence: Dict[str, Any] = None,
                      method: str = 'gibbs',
                      num_samples: int = 1000,
                      burn_in: int = 200) -> QueryResult:
        """
        Calcular P(variable | evidence).
        
        Args:
            variable: Variable de interés
            evidence: Evidencia observada
            method: 'gibbs' o 'mh'
            num_samples: Número de muestras MCMC
            burn_in: Muestras a descartar
            
        Returns:
            QueryResult con la distribución
        """
        if evidence is None:
            evidence = {}
        
        # Ejecutar MCMC
        if method == 'gibbs':
            samples = self.gibbs.sample(
                evidence=evidence,
                num_samples=num_samples,
                burn_in=burn_in
            )
            distribution = self.gibbs.estimate_marginal(variable, samples)
        elif method == 'mh':
            samples = self.mh.sample(
                evidence=evidence,
                num_samples=num_samples,
                burn_in=burn_in
            )
            distribution = self.mh.estimate_marginal(variable, samples)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return QueryResult(
            query_vars=[variable],
            evidence=evidence,
            distribution=distribution,
            method=method,
            num_samples=num_samples
        )
    
    def query_joint(self,
                   variables: List[str],
                   evidence: Dict[str, Any] = None,
                   method: str = 'gibbs',
                   num_samples: int = 1000,
                   burn_in: int = 200) -> QueryResult:
        """
        Calcular P(variables | evidence) conjunta.
        
        Args:
            variables: Lista de variables de interés
            evidence: Evidencia observada
            method: 'gibbs' o 'mh'
            num_samples: Número de muestras
            burn_in: Burn-in
            
        Returns:
            QueryResult con distribución conjunta
        """
        if evidence is None:
            evidence = {}
        
        # Ejecutar MCMC
        if method == 'gibbs':
            samples = self.gibbs.sample(
                evidence=evidence,
                num_samples=num_samples,
                burn_in=burn_in
            )
            distribution = self.gibbs.estimate_joint(variables, samples)
        elif method == 'mh':
            samples = self.mh.sample(
                evidence=evidence,
                num_samples=num_samples,
                burn_in=burn_in
            )
            # MH no tiene estimate_joint, usar samples directamente
            counts = {}
            for sample in samples:
                key = tuple(sample.assignment[v] for v in variables)
                counts[key] = counts.get(key, 0) + 1
            distribution = {k: c/len(samples) for k, c in counts.items()}
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return QueryResult(
            query_vars=variables,
            evidence=evidence,
            distribution=distribution,
            method=method,
            num_samples=num_samples
        )
    
    def query_conditional_probability(self,
                                     variable: str,
                                     value: Any,
                                     evidence: Dict[str, Any] = None,
                                     method: str = 'gibbs',
                                     num_samples: int = 1000) -> float:
        """
        Calcular P(variable = value | evidence).
        
        Args:
            variable: Variable de interés
            value: Valor específico
            evidence: Evidencia
            method: Método MCMC
            num_samples: Número de muestras
            
        Returns:
            Probabilidad condicional
        """
        result = self.query_marginal(
            variable=variable,
            evidence=evidence,
            method=method,
            num_samples=num_samples
        )
        
        return result.distribution.get(value, 0.0)
    
    def query_expectation(self,
                         variable: str,
                         evidence: Dict[str, Any] = None,
                         method: str = 'gibbs',
                         num_samples: int = 1000) -> float:
        """
        Calcular E[variable | evidence] para variables numéricas.
        
        Args:
            variable: Variable numérica
            evidence: Evidencia
            method: Método MCMC
            num_samples: Número de muestras
            
        Returns:
            Valor esperado
        """
        result = self.query_marginal(
            variable=variable,
            evidence=evidence,
            method=method,
            num_samples=num_samples
        )
        
        # E[X] = Σ x * P(X=x)
        expectation = sum(value * prob 
                         for value, prob in result.distribution.items()
                         if isinstance(value, (int, float)))
        
        return expectation
    
    def query_map(self,
                 variable: str,
                 evidence: Dict[str, Any] = None,
                 method: str = 'gibbs',
                 num_samples: int = 1000) -> Any:
        """
        Calcular MAP (Maximum A Posteriori): argmax_x P(x | evidence).
        
        Args:
            variable: Variable de interés
            evidence: Evidencia
            method: Método MCMC
            num_samples: Número de muestras
            
        Returns:
            Valor más probable
        """
        result = self.query_marginal(
            variable=variable,
            evidence=evidence,
            method=method,
            num_samples=num_samples
        )
        
        # Encontrar valor con máxima probabilidad
        if not result.distribution:
            return None
        
        return max(result.distribution.items(), key=lambda x: x[1])[0]
    
    def compute_confidence_interval(self,
                                   samples: List[float],
                                   confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calcular intervalo de confianza para una estimación.
        
        Args:
            samples: Lista de valores muestreados
            confidence: Nivel de confianza (default 95%)
            
        Returns:
            (lower_bound, upper_bound)
        """
        if not samples:
            return (0.0, 0.0)
        
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        
        alpha = 1 - confidence
        lower_idx = int(n * alpha / 2)
        upper_idx = int(n * (1 - alpha / 2))
        
        return (sorted_samples[lower_idx], sorted_samples[upper_idx])
    
    def batch_query(self,
                   variables: List[str],
                   evidence: Dict[str, Any] = None,
                   method: str = 'gibbs',
                   num_samples: int = 1000) -> Dict[str, QueryResult]:
        """
        Ejecutar múltiples queries en un solo MCMC run.
        
        Args:
            variables: Lista de variables a consultar
            evidence: Evidencia compartida
            method: Método MCMC
            num_samples: Número de muestras
            
        Returns:
            Dict {variable: QueryResult}
        """
        if evidence is None:
            evidence = {}
        
        # Ejecutar MCMC una vez
        if method == 'gibbs':
            samples = self.gibbs.sample(
                evidence=evidence,
                num_samples=num_samples,
                burn_in=200
            )
            
            results = {}
            for var in variables:
                distribution = self.gibbs.estimate_marginal(var, samples)
                results[var] = QueryResult(
                    query_vars=[var],
                    evidence=evidence,
                    distribution=distribution,
                    method=method,
                    num_samples=num_samples
                )
        elif method == 'mh':
            samples = self.mh.sample(
                evidence=evidence,
                num_samples=num_samples,
                burn_in=200
            )
            
            results = {}
            for var in variables:
                distribution = self.mh.estimate_marginal(var, samples)
                results[var] = QueryResult(
                    query_vars=[var],
                    evidence=evidence,
                    distribution=distribution,
                    method=method,
                    num_samples=num_samples
                )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return results
