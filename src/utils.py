"""
Utilidades generales para el proyecto.
"""

import math
from typing import List, Dict, Any
import json


def normalize_distribution(dist: Dict[Any, float]) -> Dict[Any, float]:
    """
    Normaliza una distribución de probabilidad.
    
    Args:
        dist: Diccionario {valor: probabilidad}
    
    Returns:
        Distribución normalizada que suma 1.0
    """
    total = sum(dist.values())
    if total == 0:
        # Distribución uniforme si todas las probabilidades son 0
        n = len(dist)
        return {k: 1.0/n for k in dist.keys()}
    
    return {k: v/total for k, v in dist.items()}


def kl_divergence(p: Dict[Any, float], q: Dict[Any, float]) -> float:
    """
    Calcula la divergencia KL entre dos distribuciones: KL(P||Q)
    
    Args:
        p: Distribución P
        q: Distribución Q
    
    Returns:
        KL divergence (siempre >= 0)
    """
    kl = 0.0
    for key in p.keys():
        if p[key] > 0:
            if q.get(key, 0) == 0:
                return float('inf')  # Q no tiene soporte donde P sí
            kl += p[key] * math.log(p[key] / q[key])
    return kl


def entropy(dist: Dict[Any, float]) -> float:
    """
    Calcula la entropía de Shannon de una distribución.
    
    H(X) = -Σ p(x) log p(x)
    
    Args:
        dist: Distribución de probabilidad
    
    Returns:
        Entropía en bits (base 2) o nats (base e)
    """
    h = 0.0
    for p in dist.values():
        if p > 0:
            h -= p * math.log2(p)
    return h


def load_json_dataset(filepath: str) -> Dict[str, Any]:
    """
    Carga un dataset desde archivo JSON.
    
    Args:
        filepath: Ruta al archivo JSON
    
    Returns:
        Diccionario con datos del dataset
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def print_distribution(dist: Dict[Any, float], title: str = "Distribution", max_items: int = 20):
    """
    Imprime una distribución de probabilidad de forma legible.
    
    Args:
        dist: Distribución {valor: probabilidad}
        title: Título a mostrar
        max_items: Máximo número de items a mostrar
    """
    print(f"\n{title}:")
    print("-" * 50)
    
    # Ordenar por probabilidad descendente
    sorted_items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    
    for i, (key, prob) in enumerate(sorted_items[:max_items]):
        bar_length = int(prob * 40)  # Barra de hasta 40 caracteres
        bar = "█" * bar_length
        print(f"  {str(key):30s}: {prob:6.4f} {bar}")
    
    if len(sorted_items) > max_items:
        print(f"  ... ({len(sorted_items) - max_items} more items)")
    
    print("-" * 50)


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calcula estadísticas básicas de una lista de valores.
    
    Args:
        values: Lista de valores numéricos
    
    Returns:
        Diccionario con media, varianza, std, min, max
    """
    if not values:
        return {
            "mean": 0.0,
            "variance": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0
        }
    
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(variance)
    
    return {
        "mean": mean,
        "variance": variance,
        "std": std,
        "min": min(values),
        "max": max(values),
        "count": n
    }


def format_probability(p: float) -> str:
    """
    Formatea una probabilidad para mostrar.
    
    Args:
        p: Probabilidad entre 0 y 1
    
    Returns:
        String formateado (ej: "45.2%" o "0.03%")
    """
    if p >= 0.01:
        return f"{p*100:.1f}%"
    elif p >= 0.001:
        return f"{p*100:.2f}%"
    else:
        return f"{p:.2e}"


def log_sum_exp(log_probs: List[float]) -> float:
    """
    Calcula log(sum(exp(log_probs))) de forma numéricamente estable.
    
    Útil para trabajar en log-space y evitar underflow.
    
    Args:
        log_probs: Lista de log-probabilidades
    
    Returns:
        log(sum(exp(x) for x in log_probs))
    """
    if not log_probs:
        return float('-inf')
    
    max_log_prob = max(log_probs)
    
    if max_log_prob == float('-inf'):
        return float('-inf')
    
    sum_exp = sum(math.exp(lp - max_log_prob) for lp in log_probs)
    return max_log_prob + math.log(sum_exp)


class ProgressTracker:
    """Clase simple para trackear progreso de operaciones largas"""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.last_percent = -1
    
    def update(self, increment: int = 1):
        """Actualiza el progreso"""
        self.current += increment
        percent = int(100 * self.current / self.total)
        
        if percent != self.last_percent and percent % 10 == 0:
            print(f"{self.description}: {percent}% ({self.current}/{self.total})")
            self.last_percent = percent
    
    def finish(self):
        """Marca como completado"""
        print(f"{self.description}: 100% ({self.total}/{self.total}) ✓")


if __name__ == "__main__":
    # Tests básicos
    
    # Test normalize
    dist = {"a": 2, "b": 3, "c": 5}
    normalized = normalize_distribution(dist)
    print("Normalized:", normalized)
    print("Sum:", sum(normalized.values()))
    
    # Test entropy
    uniform = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    peaked = {"a": 0.9, "b": 0.05, "c": 0.03, "d": 0.02}
    print(f"\nEntropy (uniform): {entropy(uniform):.3f} bits")
    print(f"Entropy (peaked): {entropy(peaked):.3f} bits")
    
    # Test print_distribution
    print_distribution(normalized, "Example Distribution")
