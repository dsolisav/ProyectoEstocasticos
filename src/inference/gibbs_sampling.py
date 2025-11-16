"""
Gibbs Sampling Algorithm (Capítulo 18.4)

MCMC mediante muestreo condicional: P(Xi | X-i, evidence)
Implementación desde cero sin librerías externas.
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
import random
import math


@dataclass
class GibbsSample:
    """Una muestra de Gibbs Sampling."""
    assignment: Dict[str, any]
    iteration: int
    chain_id: int = 0


class GibbsSampling:
    """
    Algoritmo de Gibbs Sampling para inferencia aproximada.
    
    Gibbs Sampling es un caso especial de MCMC donde la proposal distribution
    es P(Xi | X-i, evidence), es decir, muestreamos cada variable condicionada
    en todas las demás.
    
    Algoritmo:
    1. Inicializar todas las variables no-evidencia aleatoriamente
    2. Repetir:
       - Para cada variable Xi no-evidencia:
         - Calcular P(Xi | markov_blanket(Xi), evidence)
         - Muestrear Xi ~ P(Xi | MB(Xi))
       - Guardar muestra completa
    3. Descartar burn-in
    4. Usar muestras restantes para estimar distribuciones
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
                # Muestrear uniformemente del dominio
                state[var_name] = random.choice(var.domain)
        
        return state
    
    def _compute_conditional_distribution(self, 
                                         variable: str, 
                                         current_state: Dict[str, any]) -> Dict[any, float]:
        """
        Calcular P(Xi | X-i) usando toda la red (no solo Markov Blanket).
        
        P(Xi | X-i) ∝ P(X) donde X es el estado completo.
        
        Args:
            variable: Variable a muestrear
            current_state: Estado actual de todas las variables
            
        Returns:
            Dict {value: probability}
        """
        var = self.network.variables[variable]
        distribution = {}
        
        # Calcular probabilidad para cada valor posible
        for value in var.domain:
            # Crear estado temporal con Xi = value
            temp_state = current_state.copy()
            temp_state[variable] = value
            
            # P(estado completo) = ∏ P(Xj | Parents(Xj))
            prob = 1.0
            for var_name, net_var in self.network.variables.items():
                try:
                    if len(net_var.parents) == 0:
                        p = net_var.cpt.get_probability((), temp_state[var_name])
                    else:
                        parent_tuple = tuple(temp_state[p] for p in net_var.parents)
                        p = net_var.cpt.get_probability(parent_tuple, temp_state[var_name])
                    prob *= p
                except (KeyError, AttributeError, TypeError):
                    prob *= (1.0 / len(net_var.domain))
            
            distribution[value] = prob
        
        # Normalizar
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v/total for k, v in distribution.items()}
        else:
            # Uniforme si todas las probabilidades son 0
            n = len(distribution)
            distribution = {k: 1.0/n for k in distribution}
        
        return distribution
    
    def _sample_from_distribution(self, distribution: Dict[any, float]) -> any:
        """
        Muestrear un valor de una distribución discreta.
        
        Args:
            distribution: Dict {value: probability}
            
        Returns:
            Valor muestreado
        """
        values = list(distribution.keys())
        probs = list(distribution.values())
        
        # Muestreo usando random.random()
        r = random.random()
        cumulative = 0.0
        
        for value, prob in zip(values, probs):
            cumulative += prob
            if r <= cumulative:
                return value
        
        return values[-1]  # Fallback
    
    def sample(self, 
              evidence: Dict[str, any],
              num_samples: int = 1000,
              burn_in: int = 100,
              thinning: int = 1,
              chain_id: int = 0) -> List[GibbsSample]:
        """
        Ejecutar Gibbs Sampling.
        
        Args:
            evidence: Variables observadas
            num_samples: Número de muestras a generar (después de burn-in)
            burn_in: Número de iteraciones a descartar al inicio
            thinning: Tomar 1 muestra cada `thinning` iteraciones
            chain_id: ID de la cadena (para múltiples cadenas)
            
        Returns:
            Lista de muestras
        """
        # Inicializar estado
        current_state = self._initialize_state(evidence)
        
        # Variables a muestrear (no-evidencia)
        non_evidence_vars = [v for v in self.network.variables.keys() 
                           if v not in evidence]
        
        samples = []
        total_iterations = burn_in + (num_samples * thinning)
        
        for iteration in range(total_iterations):
            # Sweep: muestrear cada variable
            for var in non_evidence_vars:
                # Calcular P(Xi | X-i)
                conditional = self._compute_conditional_distribution(
                    var, current_state
                )
                
                # Muestrear nuevo valor
                new_value = self._sample_from_distribution(conditional)
                current_state[var] = new_value
            
            # Guardar muestra (después de burn-in, con thinning)
            if iteration >= burn_in and (iteration - burn_in) % thinning == 0:
                samples.append(GibbsSample(
                    assignment=current_state.copy(),
                    iteration=iteration,
                    chain_id=chain_id
                ))
        
        return samples
    
    def estimate_marginal(self, 
                         variable: str,
                         samples: List[GibbsSample]) -> Dict[any, float]:
        """
        Estimar P(variable) a partir de muestras.
        
        Args:
            variable: Variable de interés
            samples: Muestras de Gibbs
            
        Returns:
            Dict {value: estimated_probability}
        """
        counts = {}
        
        for sample in samples:
            value = sample.assignment[variable]
            counts[value] = counts.get(value, 0) + 1
        
        total = len(samples)
        return {v: c/total for v, c in counts.items()}
    
    def estimate_joint(self,
                      variables: List[str],
                      samples: List[GibbsSample]) -> Dict[Tuple, float]:
        """
        Estimar P(variables) conjunta a partir de muestras.
        
        Args:
            variables: Lista de variables
            samples: Muestras de Gibbs
            
        Returns:
            Dict {(val1, val2, ...): probability}
        """
        counts = {}
        
        for sample in samples:
            key = tuple(sample.assignment[v] for v in variables)
            counts[key] = counts.get(key, 0) + 1
        
        total = len(samples)
        return {k: c/total for k, c in counts.items()}
    
    def run_multiple_chains(self,
                           evidence: Dict[str, any],
                           num_chains: int = 3,
                           num_samples: int = 1000,
                           burn_in: int = 100) -> List[List[GibbsSample]]:
        """
        Ejecutar múltiples cadenas de Gibbs (para diagnóstico de convergencia).
        
        Args:
            evidence: Variables observadas
            num_chains: Número de cadenas independientes
            num_samples: Muestras por cadena
            burn_in: Burn-in por cadena
            
        Returns:
            Lista de listas de muestras (una por cadena)
        """
        all_chains = []
        
        for chain_id in range(num_chains):
            chain_samples = self.sample(
                evidence=evidence,
                num_samples=num_samples,
                burn_in=burn_in,
                thinning=1,
                chain_id=chain_id
            )
            all_chains.append(chain_samples)
        
        return all_chains


class ConvergenceDiagnostics:
    """
    Diagnósticos de convergencia para MCMC.
    """
    
    @staticmethod
    def gelman_rubin_statistic(chains: List[List[GibbsSample]], 
                               variable: str) -> float:
        """
        Calcular estadístico de Gelman-Rubin (R̂) para detectar convergencia.
        
        R̂ ≈ 1 indica convergencia.
        R̂ > 1.1 sugiere falta de convergencia.
        
        Args:
            chains: Múltiples cadenas de muestras
            variable: Variable a analizar
            
        Returns:
            Valor R̂
        """
        m = len(chains)  # Número de cadenas
        n = len(chains[0])  # Longitud de cada cadena
        
        # Extraer valores (convertir a numérico si es bool)
        chain_values = []
        for chain in chains:
            values = [1.0 if sample.assignment[variable] == True 
                     else 0.0 if sample.assignment[variable] == False
                     else float(sample.assignment[variable])
                     for sample in chain]
            chain_values.append(values)
        
        # Media de cada cadena
        chain_means = [sum(vals)/n for vals in chain_values]
        
        # Media global
        global_mean = sum(chain_means) / m
        
        # Varianza within-chain (W)
        chain_variances = []
        for vals in chain_values:
            variance = sum((x - sum(vals)/n)**2 for x in vals) / (n - 1)
            chain_variances.append(variance)
        W = sum(chain_variances) / m
        
        # Varianza between-chain (B)
        B = (n / (m - 1)) * sum((mean - global_mean)**2 for mean in chain_means)
        
        # Varianza estimada
        var_plus = ((n - 1) / n) * W + (1 / n) * B
        
        # R̂
        if W == 0:
            return 1.0
        
        R_hat = math.sqrt(var_plus / W)
        return R_hat
    
    @staticmethod
    def effective_sample_size(samples: List[GibbsSample], 
                             variable: str,
                             max_lag: int = 100) -> float:
        """
        Estimar tamaño efectivo de muestra considerando autocorrelación.
        
        ESS ≈ N / (1 + 2 * Σρ_k) donde ρ_k es autocorrelación en lag k.
        
        Args:
            samples: Muestras de una cadena
            variable: Variable a analizar
            max_lag: Lag máximo para autocorrelación
            
        Returns:
            Tamaño efectivo de muestra
        """
        n = len(samples)
        
        # Extraer valores (convertir a numérico)
        values = [1.0 if sample.assignment[variable] == True 
                 else 0.0 if sample.assignment[variable] == False
                 else float(sample.assignment[variable])
                 for sample in samples]
        
        mean = sum(values) / n
        variance = sum((x - mean)**2 for x in values) / n
        
        if variance == 0:
            return float(n)
        
        # Calcular autocorrelaciones
        autocorrelations = []
        for lag in range(1, min(max_lag, n//2)):
            covariance = sum((values[i] - mean) * (values[i+lag] - mean) 
                           for i in range(n - lag)) / n
            rho = covariance / variance
            autocorrelations.append(rho)
            
            # Truncar cuando la autocorrelación es muy pequeña
            if abs(rho) < 0.05:
                break
        
        # ESS
        sum_rho = sum(autocorrelations)
        ess = n / (1 + 2 * sum_rho)
        
        return max(1.0, ess)
    
    @staticmethod
    def acceptance_rate(samples: List[GibbsSample], variable: str) -> float:
        """
        Calcular tasa de aceptación (proporción de cambios de estado).
        
        Args:
            samples: Muestras de una cadena
            variable: Variable a analizar
            
        Returns:
            Tasa de aceptación [0, 1]
        """
        if len(samples) < 2:
            return 0.0
        
        changes = 0
        for i in range(1, len(samples)):
            if samples[i].assignment[variable] != samples[i-1].assignment[variable]:
                changes += 1
        
        return changes / (len(samples) - 1)
