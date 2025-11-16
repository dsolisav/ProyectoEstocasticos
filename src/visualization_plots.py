"""
Visualizaciones gráficas usando matplotlib para análisis de resultados.
Complementa visualization.py (ASCII) con gráficos PNG profesionales.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class NetworkPlotter:
    """Gráficos de estructura de redes bayesianas."""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
    
    def plot_network_structure(self, network, output_path: str = "output/network_structure.png"):
        """
        Visualiza la estructura de la red bayesiana.
        
        Args:
            network: GroundedBayesNet
            output_path: Ruta para guardar el gráfico
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Obtener variables por tipo
        quality_vars = [v for v in network.variables.values() if v.var_type == 'quality']
        honest_vars = [v for v in network.variables.values() if v.var_type == 'honesty']
        rec_vars = [v for v in network.variables.values() if v.var_type == 'recommendation']
        
        # Crear gráfico de barras
        types = ['Quality', 'Honest', 'Recommendation']
        counts = [len(quality_vars), len(honest_vars), len(rec_vars)]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        bars = ax.bar(types, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Anotar valores
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Estadísticas
        total_vars = len(network.variables)
        total_edges = sum(len(v.parents) for v in network.variables.values())
        
        ax.set_ylabel('Número de Variables', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tipo de Variable', fontsize=14, fontweight='bold')
        ax.set_title(f'Estructura de la Red Bayesiana\n'
                    f'Total: {total_vars} variables, {total_edges} aristas',
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(counts) * 1.2)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Gráfico guardado: {output_path}")
        return output_path


class DistributionPlotter:
    """Gráficos de distribuciones de probabilidad."""
    
    def __init__(self, figsize=(10, 6)):
        self.figsize = figsize
    
    def plot_distribution(self, distribution: Dict[Any, float], 
                         title: str = "Distribución de Probabilidad",
                         xlabel: str = "Valor",
                         output_path: str = "output/distribution.png"):
        """
        Gráfico de barras de una distribución de probabilidad.
        
        Args:
            distribution: Dict {valor: probabilidad}
            title: Título del gráfico
            xlabel: Etiqueta del eje X
            output_path: Ruta para guardar
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Ordenar por clave
        items = sorted(distribution.items(), key=lambda x: str(x[0]))
        values = [str(k) for k, v in items]
        probs = [v for k, v in items]
        
        # Colores degradados
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(values)))
        
        bars = ax.bar(values, probs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Anotar probabilidades
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Probabilidad', fontsize=13, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(probs) * 1.2)
        
        # Línea horizontal en y=1
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Gráfico guardado: {output_path}")
        return output_path
    
    def plot_comparison(self, distributions: Dict[str, Dict[Any, float]],
                       title: str = "Comparación de Distribuciones",
                       output_path: str = "output/comparison.png"):
        """
        Compara múltiples distribuciones en un gráfico.
        
        Args:
            distributions: Dict {nombre: distribución}
            title: Título del gráfico
            output_path: Ruta para guardar
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Obtener todos los valores posibles
        all_values = set()
        for dist in distributions.values():
            all_values.update(dist.keys())
        all_values = sorted(list(all_values), key=lambda x: str(x))
        
        # Preparar datos
        x = np.arange(len(all_values))
        width = 0.8 / len(distributions)
        
        # Colores
        colors = plt.cm.Set3(np.linspace(0, 1, len(distributions)))
        
        # Graficar cada distribución
        for i, (name, dist) in enumerate(distributions.items()):
            probs = [dist.get(val, 0.0) for val in all_values]
            offset = (i - len(distributions)/2 + 0.5) * width
            bars = ax.bar(x + offset, probs, width, label=name, 
                         color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
            
            # Anotar valores
            for bar, prob in zip(bars, probs):
                if prob > 0.02:  # Solo anotar si es visible
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{prob:.2f}',
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Probabilidad', fontsize=13, fontweight='bold')
        ax.set_xlabel('Valor', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in all_values])
        ax.legend(fontsize=11, loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Gráfico guardado: {output_path}")
        return output_path


class ROCPlotter:
    """Gráficos de curvas ROC y métricas de clasificación."""
    
    def __init__(self, figsize=(10, 8)):
        self.figsize = figsize
    
    def plot_roc_curve(self, tpr_list: List[float], fpr_list: List[float],
                      auc: float, output_path: str = "output/roc_curve.png"):
        """
        Gráfico profesional de curva ROC.
        
        Args:
            tpr_list: True Positive Rates
            fpr_list: False Positive Rates
            auc: Area Under Curve
            output_path: Ruta para guardar
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Curva ROC
        ax.plot(fpr_list, tpr_list, 'b-', linewidth=3, label=f'ROC Curve (AUC = {auc:.3f})')
        
        # Línea diagonal (clasificador aleatorio)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier (AUC = 0.5)')
        
        # Rellenar área bajo la curva
        ax.fill_between(fpr_list, tpr_list, alpha=0.2, color='blue')
        
        # Etiquetas y formato
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold')
        ax.set_title(f'Curva ROC - Bot Detection\nAUC = {auc:.4f}', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Clasificación del AUC
        if auc >= 0.9:
            classification = "EXCELENTE"
            color = 'green'
        elif auc >= 0.8:
            classification = "BUENA"
            color = 'blue'
        elif auc >= 0.7:
            classification = "ACEPTABLE"
            color = 'orange'
        else:
            classification = "POBRE"
            color = 'red'
        
        ax.text(0.6, 0.2, f'Clasificación: {classification}',
               fontsize=14, fontweight='bold', color=color,
               bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=2))
        
        ax.legend(fontsize=12, loc='lower right')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Curva ROC guardada: {output_path}")
        return output_path
    
    def plot_confusion_matrix(self, metrics, output_path: str = "output/confusion_matrix.png"):
        """
        Visualiza matriz de confusión con métricas.
        
        Args:
            metrics: DetectionMetrics object
            output_path: Ruta para guardar
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Matriz de confusión
        cm = np.array([
            [metrics.true_positives, metrics.false_positives],
            [metrics.false_negatives, metrics.true_negatives]
        ])
        
        im = ax1.imshow(cm, cmap='Blues', alpha=0.8)
        
        # Anotar valores
        for i in range(2):
            for j in range(2):
                text = ax1.text(j, i, str(cm[i, j]),
                              ha="center", va="center", color="black",
                              fontsize=20, fontweight='bold')
        
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Predicted Bot', 'Predicted Real'], fontsize=12)
        ax1.set_yticklabels(['Actual Bot', 'Actual Real'], fontsize=12)
        ax1.set_title('Confusion Matrix', fontsize=15, fontweight='bold', pad=15)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Count', fontsize=11)
        
        # Métricas
        metrics_data = {
            'Precision': metrics.precision,
            'Recall': metrics.recall,
            'F1-Score': metrics.f1_score,
            'Accuracy': metrics.accuracy
        }
        
        metric_names = list(metrics_data.keys())
        metric_values = list(metrics_data.values())
        
        # Colores según valor
        colors = ['green' if v >= 0.7 else 'orange' if v >= 0.5 else 'red' 
                 for v in metric_values]
        
        bars = ax2.barh(metric_names, metric_values, color=colors, 
                       alpha=0.7, edgecolor='black', linewidth=2)
        
        # Anotar valores
        for bar, value in zip(bars, metric_values):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{value:.3f}',
                    ha='left', va='center', fontsize=13, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Score', fontsize=13, fontweight='bold')
        ax2.set_title('Performance Metrics', fontsize=15, fontweight='bold', pad=15)
        ax2.set_xlim(0, 1.1)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        ax2.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Good (≥0.7)')
        ax2.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Fair (≥0.5)')
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Matriz de confusión guardada: {output_path}")
        return output_path


class ConvergencePlotter:
    """Gráficos de análisis de convergencia MCMC."""
    
    def __init__(self, figsize=(14, 10)):
        self.figsize = figsize
    
    def plot_mcmc_comparison(self, gibbs_samples: List[Dict], 
                            mh_samples: List[Dict],
                            variable_name: str,
                            output_path: str = "output/mcmc_convergence.png"):
        """
        Compara convergencia de Gibbs vs Metropolis-Hastings.
        
        Args:
            gibbs_samples: Muestras de Gibbs
            mh_samples: Muestras de MH
            variable_name: Variable a analizar
            output_path: Ruta para guardar
        """
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Extraer valores
        gibbs_values = [s[variable_name] for s in gibbs_samples if variable_name in s]
        mh_values = [s[variable_name] for s in mh_samples if variable_name in s]
        
        # 1. Trace plots
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(gibbs_values, 'b-', alpha=0.6, linewidth=0.5)
        ax1.set_title('Gibbs Sampling - Trace Plot', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel(variable_name, fontsize=11)
        ax1.grid(alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(mh_values, 'r-', alpha=0.6, linewidth=0.5)
        ax2.set_title('Metropolis-Hastings - Trace Plot', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel(variable_name, fontsize=11)
        ax2.grid(alpha=0.3)
        
        # 2. Distribuciones
        ax3 = fig.add_subplot(gs[1, :])
        
        # Convertir a numérico si es posible
        try:
            gibbs_numeric = [float(v) for v in gibbs_values]
            mh_numeric = [float(v) for v in mh_values]
            
            ax3.hist(gibbs_numeric, bins=20, alpha=0.5, color='blue', 
                    label='Gibbs', edgecolor='black', density=True)
            ax3.hist(mh_numeric, bins=20, alpha=0.5, color='red', 
                    label='Metropolis-Hastings', edgecolor='black', density=True)
            ax3.set_ylabel('Density', fontsize=11)
        except:
            # Si no es numérico, usar countplot
            from collections import Counter
            gibbs_counts = Counter(gibbs_values)
            mh_counts = Counter(mh_values)
            
            all_vals = sorted(set(gibbs_values + mh_values), key=str)
            x = np.arange(len(all_vals))
            width = 0.35
            
            gibbs_probs = [gibbs_counts.get(v, 0) / len(gibbs_values) for v in all_vals]
            mh_probs = [mh_counts.get(v, 0) / len(mh_values) for v in all_vals]
            
            ax3.bar(x - width/2, gibbs_probs, width, label='Gibbs', 
                   color='blue', alpha=0.7, edgecolor='black')
            ax3.bar(x + width/2, mh_probs, width, label='Metropolis-Hastings',
                   color='red', alpha=0.7, edgecolor='black')
            ax3.set_xticks(x)
            ax3.set_xticklabels([str(v) for v in all_vals])
            ax3.set_ylabel('Probability', fontsize=11)
        
        ax3.set_xlabel(variable_name, fontsize=11)
        ax3.set_title('Distribuciones Posteriores', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(alpha=0.3, axis='y')
        
        # 3. Autocorrelación (aproximada)
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1])
        
        # Running mean para ver convergencia
        def running_mean(values, window=50):
            try:
                numeric = [float(v) for v in values]
                cumsum = np.cumsum(numeric)
                result = (cumsum[window:] - cumsum[:-window]) / window
                return result
            except:
                return []
        
        gibbs_running = running_mean(gibbs_values)
        mh_running = running_mean(mh_values)
        
        if len(gibbs_running) > 0:
            ax4.plot(gibbs_running, 'b-', linewidth=2)
            ax4.set_title('Gibbs - Running Mean', fontsize=13, fontweight='bold')
            ax4.set_xlabel('Iteration', fontsize=11)
            ax4.set_ylabel('Mean', fontsize=11)
            ax4.grid(alpha=0.3)
        
        if len(mh_running) > 0:
            ax5.plot(mh_running, 'r-', linewidth=2)
            ax5.set_title('MH - Running Mean', fontsize=13, fontweight='bold')
            ax5.set_xlabel('Iteration', fontsize=11)
            ax5.set_ylabel('Mean', fontsize=11)
            ax5.grid(alpha=0.3)
        
        fig.suptitle(f'Análisis de Convergencia MCMC - {variable_name}',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Análisis MCMC guardado: {output_path}")
        return output_path


class BotDetectionPlotter:
    """Gráficos específicos para análisis de detección de bots."""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
    
    def plot_bot_scores(self, bot_scores, output_path: str = "output/bot_scores.png"):
        """
        Visualiza scores de detección de bots.
        
        Args:
            bot_scores: Lista de BotScore objects
            output_path: Ruta para guardar
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # Ordenar por probabilidad
        sorted_scores = sorted(bot_scores, key=lambda x: x.bot_probability, reverse=True)
        
        customer_ids = [s.customer_id for s in sorted_scores]
        probabilities = [s.bot_probability for s in sorted_scores]
        true_types = [s.entity_type.value for s in sorted_scores]
        
        # Colores según ground truth
        colors = ['red' if t == 'bot' else 'green' for t in true_types]
        
        # Gráfico 1: Scores por customer
        bars = ax1.barh(customer_ids, probabilities, color=colors, 
                       alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax1.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, 
                   label='Threshold (0.5)', alpha=0.8)
        ax1.set_xlabel('Bot Probability', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Customer ID', fontsize=13, fontweight='bold')
        ax1.set_title('Bot Detection Scores (Sorted)', fontsize=15, fontweight='bold', pad=15)
        ax1.set_xlim(0, 1)
        ax1.grid(axis='x', alpha=0.3)
        
        # Leyenda
        red_patch = mpatches.Patch(color='red', label='Ground Truth: Bot', alpha=0.7)
        green_patch = mpatches.Patch(color='green', label='Ground Truth: Real User', alpha=0.7)
        ax1.legend(handles=[red_patch, green_patch], fontsize=11, loc='lower right')
        
        # Gráfico 2: Distribución de scores por tipo
        bot_probs = [s.bot_probability for s in sorted_scores if s.entity_type.value == 'bot']
        user_probs = [s.bot_probability for s in sorted_scores if s.entity_type.value == 'real_user']
        
        ax2.hist(bot_probs, bins=15, alpha=0.6, color='red', label='Actual Bots', 
                edgecolor='black', density=True)
        ax2.hist(user_probs, bins=15, alpha=0.6, color='green', label='Actual Users',
                edgecolor='black', density=True)
        ax2.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='Threshold')
        ax2.set_xlabel('Bot Probability', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=13, fontweight='bold')
        ax2.set_title('Distribution of Bot Scores by True Type', fontsize=15, fontweight='bold', pad=15)
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Bot scores guardados: {output_path}")
        return output_path
    
    def plot_sybil_attacks(self, sybil_dict: Dict[str, List[str]], 
                          customers,
                          output_path: str = "output/sybil_attacks.png"):
        """
        Visualiza ataques sybil detectados.
        
        Args:
            sybil_dict: Dict {customer_id: [login_ids]}
            customers: Lista de customers
            output_path: Ruta para guardar
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Ordenar por número de cuentas
        sorted_sybils = sorted(sybil_dict.items(), key=lambda x: len(x[1]), reverse=True)
        
        customer_ids = [cid for cid, _ in sorted_sybils]
        num_accounts = [len(accounts) for _, accounts in sorted_sybils]
        
        # Colores según tipo
        colors = []
        for cid in customer_ids:
            customer = next((c for c in customers if c.customer_id == cid), None)
            if customer and customer.entity_type.value == 'bot':
                colors.append('red')
            else:
                colors.append('orange')
        
        bars = ax.barh(customer_ids, num_accounts, color=colors, 
                      alpha=0.7, edgecolor='black', linewidth=2)
        
        # Anotar valores
        for bar, count in zip(bars, num_accounts):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {int(count)}',
                   ha='left', va='center', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Number of Accounts', fontsize=13, fontweight='bold')
        ax.set_ylabel('Customer ID', fontsize=13, fontweight='bold')
        ax.set_title('Sybil Attacks Detected (Multiple Accounts)', 
                    fontsize=15, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3)
        
        # Leyenda
        red_patch = mpatches.Patch(color='red', label='Bot', alpha=0.7)
        orange_patch = mpatches.Patch(color='orange', label='Real User', alpha=0.7)
        ax.legend(handles=[red_patch, orange_patch], fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Sybil attacks guardados: {output_path}")
        return output_path
