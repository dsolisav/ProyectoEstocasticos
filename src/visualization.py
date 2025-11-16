"""
FASE 6: Visualización y Análisis
Herramientas para visualizar redes bayesianas, distribuciones posteriores,
y métricas de evaluación.
"""

from typing import Dict, List, Tuple, Any, Optional
import math


class BayesianNetworkVisualizer:
    """
    Visualizador para redes bayesianas usando representación ASCII.
    """
    
    def __init__(self):
        pass
    
    def visualize_network(self, network, output_path: Optional[str] = None) -> str:
        """
        Genera visualización ASCII de la estructura de la red.
        
        Args:
            network: Red bayesiana (RPM o OUPM)
            output_path: Ruta opcional para guardar a archivo
            
        Returns:
            String con representación ASCII de la red
        """
        lines = []
        lines.append("=" * 60)
        lines.append("ESTRUCTURA DE LA RED BAYESIANA")
        lines.append("=" * 60)
        lines.append("")
        
        # Obtener variables y sus padres
        var_parents = {}
        for var_name, var_obj in network.variables.items():
            if hasattr(var_obj, 'parents'):
                var_parents[var_name] = var_obj.parents
            else:
                var_parents[var_name] = []
        
        # Agrupar por niveles
        levels = self._compute_levels(var_parents)
        
        # Dibujar por niveles
        for level_idx in sorted(levels.keys()):
            level_vars = levels[level_idx]
            lines.append(f"Nivel {level_idx}:")
            for var in level_vars:
                parents = var_parents[var]
                if parents:
                    parent_str = ", ".join(parents)
                    lines.append(f"  [{var}] ← {parent_str}")
                else:
                    lines.append(f"  [{var}] (raíz)")
            lines.append("")
        
        # Estadísticas
        lines.append("=" * 60)
        lines.append("ESTADÍSTICAS DE LA RED")
        lines.append("=" * 60)
        lines.append(f"Variables totales: {len(network.variables)}")
        
        total_edges = sum(len(parents) for parents in var_parents.values())
        lines.append(f"Aristas totales: {total_edges}")
        
        max_parents = max(len(parents) for parents in var_parents.values())
        lines.append(f"Máximo in-degree: {max_parents}")
        
        result = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
        
        return result
    
    def _compute_levels(self, var_parents: Dict[str, List[str]]) -> Dict[int, List[str]]:
        """
        Asigna cada variable a un nivel basado en su profundidad en el DAG.
        """
        levels = {}
        visited = set()
        
        def get_level(var):
            if var in visited:
                return levels.get(var, 0)
            visited.add(var)
            
            parents = var_parents.get(var, [])
            if not parents:
                level = 0
            else:
                parent_levels = [get_level(p) for p in parents]
                level = max(parent_levels) + 1
            
            levels[var] = level
            return level
        
        for var in var_parents:
            get_level(var)
        
        # Invertir: agrupar por nivel
        level_groups = {}
        for var, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(var)
        
        return level_groups


class DistributionPlotter:
    """
    Generador de gráficos ASCII para distribuciones de probabilidad.
    """
    
    def __init__(self, width: int = 60, height: int = 15):
        self.width = width
        self.height = height
    
    def plot_distribution(self, distribution: Dict[Any, float], 
                         title: str = "Distribution",
                         output_path: Optional[str] = None) -> str:
        """
        Genera gráfico de barras ASCII de una distribución.
        
        Args:
            distribution: Dict mapeando valores a probabilidades
            title: Título del gráfico
            output_path: Ruta opcional para guardar
            
        Returns:
            String con gráfico ASCII
        """
        lines = []
        lines.append("=" * self.width)
        lines.append(title.center(self.width))
        lines.append("=" * self.width)
        lines.append("")
        
        if not distribution:
            lines.append("(distribución vacía)")
            result = "\n".join(lines)
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result)
            return result
        
        # Normalizar
        total = sum(distribution.values())
        if total > 0:
            normalized = {k: v/total for k, v in distribution.items()}
        else:
            normalized = distribution
        
        # Ordenar por clave
        sorted_items = sorted(normalized.items(), key=lambda x: str(x[0]))
        
        max_prob = max(normalized.values()) if normalized else 1.0
        
        # Dibujar barras
        bar_width = self.width - 20  # Espacio para etiquetas
        
        for value, prob in sorted_items:
            bar_len = int((prob / max_prob) * bar_width) if max_prob > 0 else 0
            bar = "█" * bar_len
            label = f"{value:>8}"
            prob_str = f"{prob:.4f}"
            lines.append(f"{label} | {bar} {prob_str}")
        
        lines.append("")
        lines.append(f"Total: {total:.4f}")
        
        result = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
        
        return result
    
    def plot_comparison(self, distributions: Dict[str, Dict[Any, float]],
                       title: str = "Comparison",
                       output_path: Optional[str] = None) -> str:
        """
        Compara múltiples distribuciones lado a lado.
        """
        lines = []
        lines.append("=" * self.width)
        lines.append(title.center(self.width))
        lines.append("=" * self.width)
        lines.append("")
        
        for name, dist in distributions.items():
            lines.append(f"\n{name}:")
            lines.append("-" * 40)
            
            total = sum(dist.values())
            normalized = {k: v/total for k, v in dist.items()} if total > 0 else dist
            
            for value, prob in sorted(normalized.items(), key=lambda x: str(x[0])):
                bar_len = int(prob * 30)
                bar = "█" * bar_len
                lines.append(f"  {value:>6}: {bar} {prob:.3f}")
        
        result = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
        
        return result


class ROCCurveAnalyzer:
    """
    Análisis de curvas ROC para evaluación de clasificadores.
    """
    
    def __init__(self):
        pass
    
    def compute_roc_curve(self, scores: List[float], 
                         labels: List[bool]) -> Tuple[List[float], List[float], List[float]]:
        """
        Calcula curva ROC.
        
        Args:
            scores: Scores de probabilidad para cada ejemplo
            labels: Labels verdaderos (True=positivo, False=negativo)
            
        Returns:
            (thresholds, tpr_list, fpr_list)
        """
        # Combinar y ordenar por score descendente
        pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
        
        # Contar positivos y negativos totales
        total_pos = sum(1 for label in labels if label)
        total_neg = sum(1 for label in labels if not label)
        
        if total_pos == 0 or total_neg == 0:
            return [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]
        
        thresholds = []
        tpr_list = []  # True Positive Rate
        fpr_list = []  # False Positive Rate
        
        # Punto inicial (threshold=infinity, todo clasificado como negativo)
        thresholds.append(1.0)
        tpr_list.append(0.0)
        fpr_list.append(0.0)
        
        # Recorrer en orden de scores decrecientes
        tp = 0
        fp = 0
        prev_score = None
        
        for score, label in pairs:
            if score != prev_score:
                # Nuevo threshold
                thresholds.append(score)
                tpr = tp / total_pos
                fpr = fp / total_neg
                tpr_list.append(tpr)
                fpr_list.append(fpr)
                prev_score = score
            
            # Clasificar este ejemplo como positivo
            if label:
                tp += 1
            else:
                fp += 1
        
        # Punto final (threshold=0, todo clasificado como positivo)
        thresholds.append(0.0)
        tpr_list.append(1.0)
        fpr_list.append(1.0)
        
        return thresholds, tpr_list, fpr_list
    
    def compute_auc(self, tpr_list: List[float], fpr_list: List[float]) -> float:
        """
        Calcula AUC (Area Under Curve) usando regla del trapecio.
        """
        auc = 0.0
        for i in range(len(fpr_list) - 1):
            # Trapecio: (base * (altura1 + altura2)) / 2
            width = fpr_list[i+1] - fpr_list[i]
            height = (tpr_list[i] + tpr_list[i+1]) / 2
            auc += width * height
        
        return auc
    
    def plot_roc_curve(self, tpr_list: List[float], fpr_list: List[float],
                      auc: float, output_path: Optional[str] = None) -> str:
        """
        Genera gráfico ASCII de la curva ROC.
        """
        width = 60
        height = 20
        
        lines = []
        lines.append("=" * width)
        lines.append(f"CURVA ROC (AUC = {auc:.4f})".center(width))
        lines.append("=" * width)
        lines.append("")
        
        # Crear grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Dibujar diagonal de referencia
        for i in range(min(width, height)):
            x = int(i * width / height)
            y = height - 1 - i
            if 0 <= y < height and 0 <= x < width:
                grid[y][x] = '·'
        
        # Dibujar curva ROC
        for i in range(len(fpr_list)):
            x = int(fpr_list[i] * (width - 1))
            y = height - 1 - int(tpr_list[i] * (height - 1))
            if 0 <= y < height and 0 <= x < width:
                grid[y][x] = '█'
        
        # Dibujar ejes
        for row in grid:
            lines.append(''.join(row))
        
        lines.append("─" * width)
        lines.append("0.0 (FPR)".ljust(width//2) + "1.0".rjust(width//2))
        lines.append("")
        lines.append(f"AUC = {auc:.4f}")
        
        if auc > 0.9:
            lines.append("Clasificación: EXCELENTE")
        elif auc > 0.8:
            lines.append("Clasificación: BUENA")
        elif auc > 0.7:
            lines.append("Clasificación: ACEPTABLE")
        else:
            lines.append("Clasificación: POBRE")
        
        result = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
        
        return result


class ConvergenceAnalyzer:
    """
    Análisis de convergencia para algoritmos MCMC.
    """
    
    def __init__(self):
        pass
    
    def compare_algorithms(self, 
                          gibbs_samples: List[Dict[str, Any]],
                          mh_samples: List[Dict[str, Any]],
                          variable_name: str,
                          output_path: Optional[str] = None) -> str:
        """
        Compara convergencia de Gibbs vs Metropolis-Hastings.
        
        Args:
            gibbs_samples: Muestras de Gibbs sampling
            mh_samples: Muestras de Metropolis-Hastings
            variable_name: Variable a analizar
            output_path: Ruta para guardar
            
        Returns:
            String con análisis
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"COMPARACIÓN DE CONVERGENCIA: {variable_name}".center(70))
        lines.append("=" * 70)
        lines.append("")
        
        # Extraer valores de la variable
        gibbs_values = [s[variable_name] for s in gibbs_samples if variable_name in s]
        mh_values = [s[variable_name] for s in mh_samples if variable_name in s]
        
        if not gibbs_values or not mh_values:
            lines.append("No hay suficientes muestras para comparar.")
            result = "\n".join(lines)
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result)
            return result
        
        # Distribuciones finales
        gibbs_dist = self._compute_distribution(gibbs_values)
        mh_dist = self._compute_distribution(mh_values)
        
        lines.append("DISTRIBUCIONES FINALES:")
        lines.append("-" * 70)
        lines.append("")
        
        lines.append("Gibbs Sampling:")
        for val, prob in sorted(gibbs_dist.items()):
            bar = "█" * int(prob * 50)
            lines.append(f"  {val:>6}: {bar} {prob:.3f}")
        
        lines.append("")
        lines.append("Metropolis-Hastings:")
        for val, prob in sorted(mh_dist.items()):
            bar = "█" * int(prob * 50)
            lines.append(f"  {val:>6}: {bar} {prob:.3f}")
        
        lines.append("")
        lines.append("=" * 70)
        lines.append("MÉTRICAS DE CONVERGENCIA")
        lines.append("=" * 70)
        
        # Medias y varianzas
        gibbs_mean = self._compute_mean(gibbs_values)
        mh_mean = self._compute_mean(mh_values)
        gibbs_var = self._compute_variance(gibbs_values, gibbs_mean)
        mh_var = self._compute_variance(mh_values, mh_mean)
        
        lines.append(f"Gibbs - Media: {gibbs_mean:.4f}, Varianza: {gibbs_var:.4f}")
        lines.append(f"MH    - Media: {mh_mean:.4f}, Varianza: {mh_var:.4f}")
        lines.append(f"Diferencia en media: {abs(gibbs_mean - mh_mean):.4f}")
        
        # ESS aproximado (usando autocorrelación simple)
        gibbs_ess = self._estimate_ess(gibbs_values)
        mh_ess = self._estimate_ess(mh_values)
        
        lines.append("")
        lines.append(f"ESS estimado:")
        lines.append(f"  Gibbs: {gibbs_ess:.1f} / {len(gibbs_values)} = {gibbs_ess/len(gibbs_values):.3f}")
        lines.append(f"  MH:    {mh_ess:.1f} / {len(mh_values)} = {mh_ess/len(mh_values):.3f}")
        
        result = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
        
        return result
    
    def _compute_distribution(self, values: List[Any]) -> Dict[Any, float]:
        """Calcula distribución empírica."""
        counts = {}
        for val in values:
            counts[val] = counts.get(val, 0) + 1
        
        total = len(values)
        return {k: v/total for k, v in counts.items()}
    
    def _compute_mean(self, values: List[Any]) -> float:
        """Calcula media (convirtiendo valores a float si es posible)."""
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                # Si no es numérico, usar hash
                numeric_values.append(float(hash(str(v)) % 1000))
        
        return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
    
    def _compute_variance(self, values: List[Any], mean: float) -> float:
        """Calcula varianza."""
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                numeric_values.append(float(hash(str(v)) % 1000))
        
        if not numeric_values:
            return 0.0
        
        squared_diffs = [(x - mean) ** 2 for x in numeric_values]
        return sum(squared_diffs) / len(squared_diffs)
    
    def _estimate_ess(self, values: List[Any]) -> float:
        """
        Estima Effective Sample Size usando autocorrelación lag-1.
        ESS ≈ n / (1 + 2*ρ₁) donde ρ₁ es autocorrelación lag-1.
        """
        n = len(values)
        if n < 10:
            return float(n)
        
        # Convertir a numérico
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                numeric_values.append(float(hash(str(v)) % 1000))
        
        # Calcular autocorrelación lag-1
        mean = sum(numeric_values) / len(numeric_values)
        
        numerator = 0.0
        denominator = 0.0
        
        for i in range(len(numeric_values) - 1):
            numerator += (numeric_values[i] - mean) * (numeric_values[i+1] - mean)
        
        for val in numeric_values:
            denominator += (val - mean) ** 2
        
        if denominator == 0:
            rho1 = 0.0
        else:
            rho1 = numerator / denominator
        
        # ESS
        ess = n / (1 + 2 * abs(rho1))
        return max(1.0, ess)
