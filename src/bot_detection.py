"""
Bot Detection usando Inferencia Probabilística

Sistema para detectar bots y sybil attacks usando:
- P(IsBot(customer) | recommendations)
- Análisis de patrones de comportamiento
- Scoring y clasificación
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import statistics

from .models import Customer, LoginID, Recommendation, EntityType
from .rpm_model import RPMModel
from .query_engine import QueryEngine, QueryResult


@dataclass
class BotScore:
    """Score de probabilidad de ser bot."""
    customer_id: str
    entity_type: EntityType  # Ground truth
    bot_probability: float
    prediction: EntityType  # Predicción basada en threshold
    confidence: float
    num_accounts: int
    recommendations_count: int


@dataclass
class DetectionMetrics:
    """Métricas de evaluación para bot detection."""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    confusion_matrix: Dict[str, int]


class BotDetector:
    """
    Detector de bots usando inferencia probabilística.
    
    Detecta:
    1. Bots basándose en patrones de ratings
    2. Sybil attacks (múltiples cuentas del mismo customer)
    """
    
    def __init__(self, 
                 rpm_model: RPMModel,
                 detection_threshold: float = 0.5):
        """
        Args:
            rpm_model: Modelo RPM base
            detection_threshold: Umbral para clasificar como bot (default 0.5)
        """
        self.rpm = rpm_model
        self.threshold = detection_threshold
    
    def score_customers(self,
                       customers: List[Customer],
                       books: List,
                       recommendations: List[Recommendation],
                       login_ids: List[LoginID],
                       num_samples: int = 2000) -> List[BotScore]:
        """
        Calcular bot scores para todos los customers.
        
        Args:
            customers: Lista de customers
            books: Lista de books
            recommendations: Recomendaciones observadas
            login_ids: Lista de LoginIDs
            num_samples: Muestras MCMC
            
        Returns:
            Lista de BotScores ordenada por probabilidad
        """
        # Ground el modelo
        grounded = self.rpm.ground_model(customers, books, recommendations)
        
        # Crear query engine
        query_engine = QueryEngine(grounded)
        
        # Contar cuentas por customer
        accounts_per_customer = {}
        for login_id in login_ids:
            if login_id.origin:
                cust_id = login_id.origin.customer_id
                accounts_per_customer[cust_id] = accounts_per_customer.get(cust_id, 0) + 1
        
        # Obtener ratings por customer
        ratings_per_customer = {}
        for rec in recommendations:
            login = next((l for l in login_ids if l.login_id == rec.login_id), None)
            if login and login.origin:
                cust_id = login.origin.customer_id
                if cust_id not in ratings_per_customer:
                    ratings_per_customer[cust_id] = []
                ratings_per_customer[cust_id].append(rec.rating)
        
        scores = []
        
        for customer in customers:
            # Encontrar variable de honestidad para este customer
            honest_var = f"Honest_{customer.customer_id}"
            
            if honest_var not in grounded.variables:
                continue
            
            # Obtener datos de este customer
            customer_ratings = ratings_per_customer.get(customer.customer_id, [])
            num_accounts = accounts_per_customer.get(customer.customer_id, 1)
            num_ratings = len(customer_ratings)
            
            # Calcular features de comportamiento
            behavior_features = self._compute_behavior_features(customer_ratings)
            
            # Query: P(Honest = False | recommendations)
            try:
                result = query_engine.query_marginal(
                    variable=honest_var,
                    evidence={},
                    method='gibbs',
                    num_samples=num_samples,
                    burn_in=400
                )
                dishonest_prob = result.distribution.get(False, 0.5)
            except Exception:
                dishonest_prob = 0.5
            
            # Calcular score combinado
            bot_prob = self._compute_combined_score(
                dishonest_prob=dishonest_prob,
                num_accounts=num_accounts,
                behavior_features=behavior_features
            )
            
            prediction = EntityType.BOT if bot_prob >= self.threshold else EntityType.REAL_USER
            confidence = abs(bot_prob - self.threshold)
            
            scores.append(BotScore(
                customer_id=customer.customer_id,
                entity_type=customer.entity_type,
                bot_probability=bot_prob,
                prediction=prediction,
                confidence=confidence,
                num_accounts=num_accounts,
                recommendations_count=num_ratings
            ))
        
        # Ordenar por probabilidad descendente
        scores.sort(key=lambda x: x.bot_probability, reverse=True)
        
        return scores
    
    def _compute_behavior_features(self, ratings: List[int]) -> dict:
        """Calcular features de comportamiento a partir de ratings."""
        if not ratings:
            return {'extreme_ratio': 0.0, 'variance': 0.0, 'mean': 3.0, 'count': 0}
        
        extreme_count = sum(1 for r in ratings if r in [1, 5])
        extreme_ratio = extreme_count / len(ratings)
        mean = sum(ratings) / len(ratings)
        variance = sum((r - mean) ** 2 for r in ratings) / len(ratings)
        
        return {
            'extreme_ratio': extreme_ratio,
            'variance': variance,
            'mean': mean,
            'count': len(ratings)
        }
    
    def _compute_combined_score(self, 
                                dishonest_prob: float,
                                num_accounts: int,
                                behavior_features: dict) -> float:
        """Calcular score combinado de probabilidad de ser bot."""
        # Factor 1: Modelo probabilistico
        model_score = dishonest_prob
        
        # Factor 2: Sybil attacks
        if num_accounts <= 1:
            sybil_score = 0.0
        elif num_accounts <= 2:
            sybil_score = 0.3
        elif num_accounts <= 4:
            sybil_score = 0.6
        else:
            sybil_score = 1.0
        
        # Factor 3: Ratings extremos (1 o 5)
        extreme_ratio = behavior_features.get('extreme_ratio', 0)
        if extreme_ratio <= 0.3:
            extreme_score = 0.0
        elif extreme_ratio <= 0.7:
            extreme_score = (extreme_ratio - 0.3) / 0.4 * 0.5
        else:
            extreme_score = 0.5 + (extreme_ratio - 0.7) / 0.3 * 0.5
        
        # Factor 4: Varianza alta
        variance = behavior_features.get('variance', 0)
        if variance <= 1.5:
            variance_score = 0.0
        elif variance <= 3.0:
            variance_score = (variance - 1.5) / 1.5 * 0.5
        else:
            variance_score = 0.5 + min(0.5, (variance - 3.0) / 2.0 * 0.5)
        
        # Combinar con pesos
        combined = (
            0.15 * model_score +
            0.20 * sybil_score +
            0.40 * extreme_score +
            0.25 * variance_score
        )
        
        # Bonus por multiples senales fuertes
        strong_signals = sum([
            1 if sybil_score > 0.5 else 0,
            1 if extreme_score > 0.5 else 0,
            1 if variance_score > 0.5 else 0
        ])
        if strong_signals >= 2:
            combined = min(1.0, combined + 0.15)
        
        return min(1.0, max(0.0, combined))
        
        # Ordenar por probabilidad descendente
        scores.sort(key=lambda x: x.bot_probability, reverse=True)
        
        return scores
    
    def detect_sybil_attacks(self,
                            login_ids: List[LoginID],
                            min_accounts: int = 2) -> Dict[str, List[str]]:
        """
        Detectar sybil attacks (customers con múltiples cuentas).
        
        Args:
            login_ids: Lista de LoginIDs
            min_accounts: Mínimo de cuentas para considerar sybil attack
            
        Returns:
            Dict {customer_id: [login_ids]}
        """
        accounts_per_customer = {}
        
        for login_id in login_ids:
            if login_id.origin:
                cust_id = login_id.origin.customer_id
                if cust_id not in accounts_per_customer:
                    accounts_per_customer[cust_id] = []
                accounts_per_customer[cust_id].append(login_id.login_id)
        
        # Filtrar solo los que tienen múltiples cuentas
        sybil_attacks = {
            cust_id: accounts 
            for cust_id, accounts in accounts_per_customer.items()
            if len(accounts) >= min_accounts
        }
        
        return sybil_attacks
    
    def evaluate(self, scores: List[BotScore]) -> DetectionMetrics:
        """
        Evaluar performance del detector contra ground truth.
        
        Args:
            scores: Lista de BotScores con predicciones
            
        Returns:
            DetectionMetrics con métricas de evaluación
        """
        tp = sum(1 for s in scores 
                if s.entity_type == EntityType.BOT and s.prediction == EntityType.BOT)
        
        fp = sum(1 for s in scores 
                if s.entity_type == EntityType.REAL_USER and s.prediction == EntityType.BOT)
        
        tn = sum(1 for s in scores 
                if s.entity_type == EntityType.REAL_USER and s.prediction == EntityType.REAL_USER)
        
        fn = sum(1 for s in scores 
                if s.entity_type == EntityType.BOT and s.prediction == EntityType.REAL_USER)
        
        # Métricas
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(scores) if len(scores) > 0 else 0.0
        
        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            confusion_matrix={
                'TP': tp,
                'FP': fp,
                'TN': tn,
                'FN': fn
            }
        )
    
    def find_optimal_threshold(self,
                              scores: List[BotScore],
                              thresholds: List[float] = None) -> Tuple[float, DetectionMetrics]:
        """
        Encontrar threshold óptimo que maximiza F1-score.
        
        Args:
            scores: Lista de scores
            thresholds: Lista de thresholds a probar (default: 0.1 a 0.9)
            
        Returns:
            (optimal_threshold, best_metrics)
        """
        if thresholds is None:
            thresholds = [i/10 for i in range(1, 10)]  # 0.1, 0.2, ..., 0.9
        
        best_f1 = 0.0
        best_threshold = 0.5
        best_metrics = None
        
        for threshold in thresholds:
            # Re-clasificar con este threshold
            temp_scores = []
            for score in scores:
                new_prediction = (EntityType.BOT if score.bot_probability >= threshold 
                                else EntityType.REAL_USER)
                
                temp_score = BotScore(
                    customer_id=score.customer_id,
                    entity_type=score.entity_type,
                    bot_probability=score.bot_probability,
                    prediction=new_prediction,
                    confidence=abs(score.bot_probability - threshold),
                    num_accounts=score.num_accounts,
                    recommendations_count=score.recommendations_count
                )
                temp_scores.append(temp_score)
            
            # Evaluar
            metrics = self.evaluate(temp_scores)
            
            if metrics.f1_score > best_f1:
                best_f1 = metrics.f1_score
                best_threshold = threshold
                best_metrics = metrics
        
        return best_threshold, best_metrics


class QualityEstimator:
    """
    Estimador de calidad real de libros basado en recommendations.
    """
    
    def __init__(self, rpm_model: RPMModel):
        """
        Args:
            rpm_model: Modelo RPM base
        """
        self.rpm = rpm_model
    
    def estimate_book_qualities(self,
                               customers: List[Customer],
                               books: List,
                               recommendations: List[Recommendation],
                               num_samples: int = 2000) -> Dict[str, Dict[int, float]]:
        """
        Estimar P(Quality(book) | all recommendations).
        
        Args:
            customers: Lista de customers
            books: Lista de books
            recommendations: Recomendaciones observadas
            num_samples: Muestras MCMC
            
        Returns:
            Dict {book_id: {quality_value: probability}}
        """
        # Ground el modelo
        grounded = self.rpm.ground_model(customers, books, recommendations)
        
        # Crear query engine
        query_engine = QueryEngine(grounded)
        
        # Consultar calidad para cada libro
        quality_distributions = {}
        
        for book in books:
            quality_var = f"Quality_{book.book_id}"
            
            if quality_var not in grounded.variables:
                continue
            
            try:
                result = query_engine.query_marginal(
                    variable=quality_var,
                    evidence={},  # Evidencia en las recommendations
                    method='gibbs',
                    num_samples=num_samples,
                    burn_in=400
                )
                
                quality_distributions[book.book_id] = result.distribution
                
            except Exception as e:
                # Si falla, distribución uniforme
                quality_distributions[book.book_id] = {i: 0.2 for i in range(1, 6)}
        
        return quality_distributions
    
    def get_map_quality(self, 
                       quality_distribution: Dict[int, float]) -> int:
        """
        Obtener calidad MAP (más probable).
        
        Args:
            quality_distribution: Distribución de calidad
            
        Returns:
            Calidad con máxima probabilidad
        """
        if not quality_distribution:
            return 3  # Default medio
        
        return max(quality_distribution.items(), key=lambda x: x[1])[0]
    
    def get_expected_quality(self, 
                            quality_distribution: Dict[int, float]) -> float:
        """
        Calcular E[Quality].
        
        Args:
            quality_distribution: Distribución de calidad
            
        Returns:
            Valor esperado
        """
        return sum(q * p for q, p in quality_distribution.items())
