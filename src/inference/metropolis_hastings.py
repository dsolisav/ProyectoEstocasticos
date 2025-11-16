"""
Metropolis-Hastings Algorithm (Capítulo 18.4)

MCMC con proposal distribution arbitraria y accept/reject step.
Implementación desde cero sin librerías externas.
"""

from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
import random
import math


@dataclass
class MHSample:
    """Una muestra de Metropolis-Hastings."""
    assignment: Dict[str, any]
    iteration: int
    accepted: bool  # Si esta muestra fue aceptada o rechazada
    acceptance_prob: float  # Probabilidad de aceptación α


class MetropolisHastings:
    """
    Algoritmo de Metropolis-Hastings para inferencia aproximada.
    
    Metropolis-Hastings es un algoritmo MCMC general donde:
    1. Proponemos un nuevo estado x' ~ Q(x' | x)
    2. Calculamos acceptance probability: α = min(1, [P(x')Q(x|x')] / [P(x)Q(x'|x)])
    3. Aceptamos x' con probabilidad α, sino mantenemos x
    
    Para symmetric proposal Q(x'|x) = Q(x|x'), se simplifica a:
    α = min(1, P(x') / P(x))
    """
    
    def __init__(self, grounded_network):
        """
        Args:
            grounded_network: GroundedBayesNet con la red proposicional
        """
        self.network = grounded_network
        self.samples = []
        
    def _initialize_state(self, evidence: Dict[str, any]) -> Dict[str, any]:
        """
        Inicializar estado aleatorio consistente con evidencia.
        
        Args:
            evidence: Variables observadas
            
        Returns:
            Assignment completo aleatorio
        """
        state = evidence.copy()
        
        for var_name, var in self.network.variables.items():
            if var_name not in evidence:
                state[var_name] = random.choice(var.domain)
        
        return state
    
    def _compute_state_probability(self, state: Dict[str, any]) -> float:
        """
        Calcular P(state) = ∏ P(Xi | Parents(Xi)) para todas las variables.
        
        Args:
            state: Assignment completo
            
        Returns:
            Probabilidad del estado (no normalizada)
        """
        prob = 1.0
        
        for var_name, var in self.network.variables.items():
            try:
                if len(var.parents) == 0:
                    # Sin padres - prior
                    var_prob = var.cpt.get_probability((), state[var_name])
                else:
                    # Con padres
                    parent_tuple = tuple(state[p] for p in var.parents)
                    var_prob = var.cpt.get_probability(parent_tuple, state[var_name])
                prob *= var_prob
            except (KeyError, AttributeError, TypeError):
                # Si falla, usar probabilidad uniforme
                prob *= (1.0 / len(var.domain))
            
            if prob == 0:
                break
        
        return prob
    
    def _propose_random_flip(self, 
                           current_state: Dict[str, any],
                           non_evidence_vars: List[str]) -> Dict[str, any]:
        """
        Proposal: flip una variable aleatoria a un valor aleatorio.
        Esta es una symmetric proposal: Q(x'|x) = Q(x|x').
        
        Args:
            current_state: Estado actual
            non_evidence_vars: Variables no-evidencia
            
        Returns:
            Nuevo estado propuesto
        """
        proposed = current_state.copy()
        
        # Elegir variable aleatoria
        var_name = random.choice(non_evidence_vars)
        var = self.network.variables[var_name]
        
        # Elegir valor aleatorio (diferente al actual si es posible)
        possible_values = [v for v in var.domain if v != current_state[var_name]]
        if possible_values:
            proposed[var_name] = random.choice(possible_values)
        else:
            # Si solo hay un valor en el dominio, mantener igual
            proposed[var_name] = current_state[var_name]
        
        return proposed
    
    def _propose_gibbs_style(self,
                           current_state: Dict[str, any],
                           non_evidence_vars: List[str]) -> Dict[str, any]:
        """
        Proposal estilo Gibbs: elegir variable aleatoria y muestrear de P(Xi | X-i).
        Esta NO es symmetric: Q(x'|x) ≠ Q(x|x').
        
        Args:
            current_state: Estado actual
            non_evidence_vars: Variables no-evidencia
            
        Returns:
            Nuevo estado propuesto
        """
        proposed = current_state.copy()
        
        # Elegir variable aleatoria
        var_name = random.choice(non_evidence_vars)
        var = self.network.variables[var_name]
        
        # Calcular P(Xi | X-i) ∝ P(estado completo)
        distribution = {}
        for value in var.domain:
            temp_state = current_state.copy()
            temp_state[var_name] = value
            
            # P(estado completo)
            prob = 1.0
            for v_name, v in self.network.variables.items():
                try:
                    if len(v.parents) == 0:
                        p = v.cpt.get_probability((), temp_state[v_name])
                    else:
                        parent_tuple = tuple(temp_state[p] for p in v.parents)
                        p = v.cpt.get_probability(parent_tuple, temp_state[v_name])
                    prob *= p
                except (KeyError, AttributeError, TypeError):
                    prob *= (1.0 / len(v.domain))
            
            distribution[value] = prob
        
        # Normalizar
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v/total for k, v in distribution.items()}
        
        # Muestrear
        r = random.random()
        cumulative = 0.0
        for value, prob in distribution.items():
            cumulative += prob
            if r <= cumulative:
                proposed[var_name] = value
                break
        
        return proposed
    
    def sample(self,
              evidence: Dict[str, any],
              num_samples: int = 1000,
              burn_in: int = 100,
              proposal: str = 'random_flip',  # 'random_flip' o 'gibbs_style'
              chain_id: int = 0) -> List[MHSample]:
        """
        Ejecutar Metropolis-Hastings sampling.
        
        Args:
            evidence: Variables observadas
            num_samples: Número de muestras (después de burn-in)
            burn_in: Iteraciones a descartar
            proposal: Tipo de proposal distribution
            chain_id: ID de cadena
            
        Returns:
            Lista de muestras
        """
        # Inicializar
        current_state = self._initialize_state(evidence)
        current_prob = self._compute_state_probability(current_state)
        
        non_evidence_vars = [v for v in self.network.variables.keys() 
                           if v not in evidence]
        
        samples = []
        total_iterations = burn_in + num_samples
        
        accepted_count = 0
        
        for iteration in range(total_iterations):
            # 1. Proponer nuevo estado
            if proposal == 'random_flip':
                proposed_state = self._propose_random_flip(
                    current_state, non_evidence_vars
                )
                # Symmetric proposal: Q(x'|x) = Q(x|x')
                proposal_ratio = 1.0
            elif proposal == 'gibbs_style':
                proposed_state = self._propose_gibbs_style(
                    current_state, non_evidence_vars
                )
                # Para Gibbs, Q ratio se cancela en la práctica
                proposal_ratio = 1.0
            else:
                raise ValueError(f"Unknown proposal: {proposal}")
            
            # 2. Calcular probabilidad del estado propuesto
            proposed_prob = self._compute_state_probability(proposed_state)
            
            # 3. Calcular acceptance probability
            if current_prob == 0:
                acceptance_prob = 1.0
            else:
                # α = min(1, [P(x')Q(x|x')] / [P(x)Q(x'|x)])
                # Con symmetric proposal: α = min(1, P(x') / P(x))
                acceptance_prob = min(1.0, (proposed_prob / current_prob) * proposal_ratio)
            
            # 4. Accept/Reject
            accepted = random.random() < acceptance_prob
            
            if accepted:
                current_state = proposed_state
                current_prob = proposed_prob
                accepted_count += 1
            
            # 5. Guardar muestra (después de burn-in)
            if iteration >= burn_in:
                samples.append(MHSample(
                    assignment=current_state.copy(),
                    iteration=iteration,
                    accepted=accepted,
                    acceptance_prob=acceptance_prob
                ))
        
        return samples
    
    def estimate_marginal(self,
                         variable: str,
                         samples: List[MHSample]) -> Dict[any, float]:
        """
        Estimar P(variable) a partir de muestras.
        
        Args:
            variable: Variable de interés
            samples: Muestras de MH
            
        Returns:
            Dict {value: probability}
        """
        counts = {}
        
        for sample in samples:
            value = sample.assignment[variable]
            counts[value] = counts.get(value, 0) + 1
        
        total = len(samples)
        return {v: c/total for v, c in counts.items()}
    
    def get_acceptance_rate(self, samples: List[MHSample]) -> float:
        """
        Calcular tasa de aceptación global.
        
        Args:
            samples: Muestras de MH
            
        Returns:
            Proporción de muestras aceptadas
        """
        if not samples:
            return 0.0
        
        accepted = sum(1 for s in samples if s.accepted)
        return accepted / len(samples)
    
    def run_multiple_chains(self,
                           evidence: Dict[str, any],
                           num_chains: int = 3,
                           num_samples: int = 1000,
                           burn_in: int = 100,
                           proposal: str = 'random_flip') -> List[List[MHSample]]:
        """
        Ejecutar múltiples cadenas de MH.
        
        Args:
            evidence: Variables observadas
            num_chains: Número de cadenas
            num_samples: Muestras por cadena
            burn_in: Burn-in por cadena
            proposal: Tipo de proposal
            
        Returns:
            Lista de listas de muestras
        """
        all_chains = []
        
        for chain_id in range(num_chains):
            chain_samples = self.sample(
                evidence=evidence,
                num_samples=num_samples,
                burn_in=burn_in,
                proposal=proposal,
                chain_id=chain_id
            )
            all_chains.append(chain_samples)
        
        return all_chains
