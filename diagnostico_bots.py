"""
Diagnóstico: ¿Por qué algunos bots no se detectan?
Analiza el comportamiento de bots detectados vs no detectados.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import DataGenerator, DatasetConfig
from src.rpm_model import RPMModel
from src.bot_detection import BotDetector
from src.models import EntityType
from collections import defaultdict
import random

# Semilla fija para reproducibilidad
random.seed(42)

def analyze_customer_behavior(customer_id, recommendations, login_ids):
    """Analiza el comportamiento de un customer específico."""
    # Crear mapa de login_id string -> LoginID object
    login_map = {lid.login_id: lid for lid in login_ids}
    
    # Encontrar todos los LoginIDs de este customer
    customer_logins = [lid for lid in login_ids if lid.origin.customer_id == customer_id]
    customer_login_ids = {lid.login_id for lid in customer_logins}
    
    # Obtener todas las recomendaciones de este customer
    customer_recs = [r for r in recommendations if r.login_id in customer_login_ids]
    
    if not customer_recs:
        return None
    
    ratings = [r.rating for r in customer_recs]
    
    return {
        'customer_id': customer_id,
        'num_accounts': len(customer_logins),
        'num_ratings': len(customer_recs),
        'ratings': ratings,
        'avg_rating': sum(ratings) / len(ratings),
        'rating_variance': sum((r - sum(ratings)/len(ratings))**2 for r in ratings) / len(ratings),
        'extreme_ratings': sum(1 for r in ratings if r in [1, 5]),
        'extreme_ratio': sum(1 for r in ratings if r in [1, 5]) / len(ratings),
        'unique_books': len(set(r.book_id for r in customer_recs)),
    }

def main():
    print("=" * 70)
    print("DIAGNOSTICO: Por que algunos bots no se detectan?")
    print("=" * 70)
    
    # Generar dataset
    config = DatasetConfig(
        num_real_users=12,
        num_bots=6,
        num_books=8,
        prob_bot_multiple_accounts=0.8,
        max_accounts_per_bot=8,
        min_recommendations_per_account=3,
        max_recommendations_per_account=10
    )
    
    gen = DataGenerator(config)
    customers, books, login_ids, recommendations = gen.generate_dataset()
    
    # Separar bots y usuarios
    bots = [c for c in customers if c.entity_type == EntityType.BOT]
    users = [c for c in customers if c.entity_type != EntityType.BOT]
    
    print(f"\nDataset generado:")
    print(f"   - Bots: {len(bots)}")
    print(f"   - Usuarios reales: {len(users)}")
    print(f"   - Total LoginIDs: {len(login_ids)}")
    print(f"   - Total ratings: {len(recommendations)}")
    
    # Ejecutar deteccion de bots
    print("\nEjecutando deteccion de bots...")
    rpm = RPMModel()
    detector = BotDetector(rpm, detection_threshold=0.5)
    bot_scores = detector.score_customers(
        customers=customers,
        books=books,
        recommendations=recommendations,
        login_ids=login_ids,
        num_samples=200
    )
    
    # Crear diccionario de scores
    score_dict = {s.customer_id: s.bot_probability for s in bot_scores}
    
    # Analizar comportamiento de cada customer
    print("\n" + "=" * 70)
    print("ANALISIS DE COMPORTAMIENTO POR CUSTOMER")
    print("=" * 70)
    
    bot_behaviors = []
    user_behaviors = []
    
    for customer in customers:
        behavior = analyze_customer_behavior(
            customer.customer_id, recommendations, login_ids
        )
        if behavior:
            behavior['is_bot'] = customer.entity_type == EntityType.BOT
            behavior['bot_score'] = score_dict.get(customer.customer_id, 0)
            behavior['detected'] = behavior['bot_score'] > 0.5
            
            if customer.entity_type == EntityType.BOT:
                bot_behaviors.append(behavior)
            else:
                user_behaviors.append(behavior)
    
    # Separar bots detectados vs no detectados
    detected_bots = [b for b in bot_behaviors if b['detected']]
    undetected_bots = [b for b in bot_behaviors if not b['detected']]
    
    print("\n" + "-" * 70)
    print("BOTS DETECTADOS CORRECTAMENTE (score > 0.5)")
    print("-" * 70)
    if detected_bots:
        for b in sorted(detected_bots, key=lambda x: x['bot_score'], reverse=True):
            print(f"\n  {b['customer_id']}:")
            print(f"    Bot Score: {b['bot_score']:.3f} [DETECTADO]")
            print(f"    Cuentas (sybil): {b['num_accounts']}")
            print(f"    Ratings: {b['num_ratings']}")
            print(f"    Rating promedio: {b['avg_rating']:.2f}")
            print(f"    Varianza: {b['rating_variance']:.2f}")
            print(f"    Ratings extremos (1 o 5): {b['extreme_ratings']} ({b['extreme_ratio']*100:.0f}%)")
            print(f"    Distribucion: {b['ratings']}")
    else:
        print("  Ninguno")
    
    print("\n" + "-" * 70)
    print("BOTS NO DETECTADOS (score <= 0.5)")
    print("-" * 70)
    if undetected_bots:
        for b in sorted(undetected_bots, key=lambda x: x['bot_score'], reverse=True):
            print(f"\n  {b['customer_id']}:")
            print(f"    Bot Score: {b['bot_score']:.3f} [NO DETECTADO]")
            print(f"    Cuentas (sybil): {b['num_accounts']}")
            print(f"    Ratings: {b['num_ratings']}")
            print(f"    Rating promedio: {b['avg_rating']:.2f}")
            print(f"    Varianza: {b['rating_variance']:.2f}")
            print(f"    Ratings extremos (1 o 5): {b['extreme_ratings']} ({b['extreme_ratio']*100:.0f}%)")
            print(f"    Distribucion: {b['ratings']}")
    else:
        print("  Ninguno - Todos los bots fueron detectados!")
    
    print("\n" + "-" * 70)
    print("USUARIOS REALES (para comparacion)")
    print("-" * 70)
    # Mostrar algunos usuarios para comparar
    for b in sorted(user_behaviors, key=lambda x: x['bot_score'], reverse=True)[:5]:
        status = "[FALSO POSITIVO]" if b['bot_score'] > 0.5 else "[OK]"
        print(f"\n  {b['customer_id']}:")
        print(f"    Bot Score: {b['bot_score']:.3f} {status}")
        print(f"    Cuentas: {b['num_accounts']}")
        print(f"    Ratings: {b['num_ratings']}")
        print(f"    Rating promedio: {b['avg_rating']:.2f}")
        print(f"    Varianza: {b['rating_variance']:.2f}")
        print(f"    Ratings extremos (1 o 5): {b['extreme_ratings']} ({b['extreme_ratio']*100:.0f}%)")
    
    # Analisis estadistico comparativo
    print("\n" + "=" * 70)
    print("ANALISIS COMPARATIVO")
    print("=" * 70)
    
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0
    
    if detected_bots or undetected_bots:
        print("\n  Caracteristica          | Bots Detectados | Bots No Detectados | Usuarios")
        print("  " + "-" * 75)
        
        det_accounts = avg([b['num_accounts'] for b in detected_bots]) if detected_bots else 0
        undet_accounts = avg([b['num_accounts'] for b in undetected_bots]) if undetected_bots else 0
        user_accounts = avg([b['num_accounts'] for b in user_behaviors])
        print(f"  Cuentas promedio        | {det_accounts:>15.2f} | {undet_accounts:>18.2f} | {user_accounts:>8.2f}")
        
        det_ratings = avg([b['num_ratings'] for b in detected_bots]) if detected_bots else 0
        undet_ratings = avg([b['num_ratings'] for b in undetected_bots]) if undetected_bots else 0
        user_ratings = avg([b['num_ratings'] for b in user_behaviors])
        print(f"  Ratings promedio        | {det_ratings:>15.2f} | {undet_ratings:>18.2f} | {user_ratings:>8.2f}")
        
        det_avg = avg([b['avg_rating'] for b in detected_bots]) if detected_bots else 0
        undet_avg = avg([b['avg_rating'] for b in undetected_bots]) if undetected_bots else 0
        user_avg = avg([b['avg_rating'] for b in user_behaviors])
        print(f"  Rating medio            | {det_avg:>15.2f} | {undet_avg:>18.2f} | {user_avg:>8.2f}")
        
        det_var = avg([b['rating_variance'] for b in detected_bots]) if detected_bots else 0
        undet_var = avg([b['rating_variance'] for b in undetected_bots]) if undetected_bots else 0
        user_var = avg([b['rating_variance'] for b in user_behaviors])
        print(f"  Varianza                | {det_var:>15.2f} | {undet_var:>18.2f} | {user_var:>8.2f}")
        
        det_ext = avg([b['extreme_ratio'] for b in detected_bots])*100 if detected_bots else 0
        undet_ext = avg([b['extreme_ratio'] for b in undetected_bots])*100 if undetected_bots else 0
        user_ext = avg([b['extreme_ratio'] for b in user_behaviors])*100
        print(f"  % Ratings extremos      | {det_ext:>14.1f}% | {undet_ext:>17.1f}% | {user_ext:>7.1f}%")
    
    # Conclusiones
    print("\n" + "=" * 70)
    print("CONCLUSIONES")
    print("=" * 70)
    
    if undetected_bots:
        print("\n  PROBLEMA IDENTIFICADO:")
        
        # Comparar caracteristicas
        avg_accounts_detected = avg([b['num_accounts'] for b in detected_bots]) if detected_bots else 0
        avg_accounts_undetected = avg([b['num_accounts'] for b in undetected_bots])
        avg_extreme_detected = avg([b['extreme_ratio'] for b in detected_bots]) if detected_bots else 0
        avg_extreme_undetected = avg([b['extreme_ratio'] for b in undetected_bots])
        
        if avg_accounts_undetected < avg_accounts_detected:
            print(f"\n  1. SYBIL ATTACKS MENORES:")
            print(f"     - Bots detectados tienen {avg_accounts_detected:.1f} cuentas promedio")
            print(f"     - Bots no detectados tienen {avg_accounts_undetected:.1f} cuentas promedio")
            print(f"     -> Los bots con menos cuentas son mas dificiles de detectar")
        
        if avg_extreme_undetected < avg_extreme_detected:
            print(f"\n  2. COMPORTAMIENTO MENOS EXTREMO:")
            print(f"     - Bots detectados: {avg_extreme_detected*100:.0f}% ratings extremos")
            print(f"     - Bots no detectados: {avg_extreme_undetected*100:.0f}% ratings extremos")
            print(f"     -> Los bots con ratings menos extremos parecen usuarios normales")
        
        print("\n  RECOMENDACIONES:")
        print("  1. Considerar threshold mas bajo (0.35-0.40)")
        print("  2. Agregar mas features: varianza de ratings, patrones temporales")
        print("  3. Aumentar numero de muestras MCMC")
        print("  4. Considerar multiples senales combinadas (sybil + extremos)")
    else:
        print("\n  Todos los bots fueron detectados correctamente!")
        print("     El sistema esta funcionando bien con este dataset.")
    
    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    total_bots = len(bot_behaviors)
    detected = len(detected_bots)
    print(f"\n  Bots detectados: {detected}/{total_bots} ({detected/total_bots*100:.1f}%)")
    print(f"  Bots no detectados: {total_bots - detected}/{total_bots} ({(total_bots-detected)/total_bots*100:.1f}%)")
    
    false_positives = sum(1 for b in user_behaviors if b['bot_score'] > 0.5)
    print(f"  Falsos positivos: {false_positives}/{len(user_behaviors)}")
    
    if total_bots > 0:
        precision = detected / (detected + false_positives) if (detected + false_positives) > 0 else 0
        recall = detected / total_bots
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"\n  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")

if __name__ == "__main__":
    main()
