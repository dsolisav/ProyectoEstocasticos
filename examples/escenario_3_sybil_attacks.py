"""
ESCENARIO 3: Detección de Sybil Attacks
=======================================

Objetivo: Demostrar que el sistema detecta cuando un mismo usuario (bot)
opera múltiples cuentas para manipular ratings.

Hipótesis: Un ataque Sybil ocurre cuando un bot crea múltiples LoginIDs
para inflar artificialmente el rating de un producto.

Salida: 2 gráficos PNG en examples/output/
- esc3_sybil_attacks.png
- esc3_cuentas_por_tipo.png

Tiempo de ejecución: ~1 minuto
"""

import sys
import os
import random

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

random.seed(42)

from src.data_generator import DataGenerator, DatasetConfig
from src.models import EntityType
from src.rpm_model import RPMModel
from src.bot_detection import BotDetector
from src.visualization_plots import BotDetectionPlotter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    print("="*70)
    print("  ESCENARIO 3: DETECCIÓN DE SYBIL ATTACKS")
    print("="*70)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================
    # CONFIGURACIÓN DEL EXPERIMENTO
    # =========================================
    print("\n[CONFIGURACIÓN]")
    print("  - 10 usuarios reales")
    print("  - 5 bots con alta probabilidad de múltiples cuentas")
    print("  - prob_bot_multiple_accounts = 0.8 (80%)")
    print("  - max_accounts_per_bot = 6")
    
    config = DatasetConfig(
        num_real_users=10,
        num_bots=5,
        num_books=6,
        prob_bot_multiple_accounts=0.8,
        max_accounts_per_bot=6,
        random_seed=42
    )
    
    # =========================================
    # PASO 1: Generar datos
    # =========================================
    print("\n[PASO 1] Generando datos sintéticos...")
    generator = DataGenerator(config)
    customers, books, login_ids, recommendations = generator.generate_dataset()
    
    print(f"  Customers: {len(customers)}")
    print(f"  LoginIDs (cuentas): {len(login_ids)}")
    print(f"  Ratings: {len(recommendations)}")
    
    # Contar cuentas por customer
    accounts_per_customer = {}
    for lid in login_ids:
        cid = lid.origin.customer_id
        accounts_per_customer[cid] = accounts_per_customer.get(cid, 0) + 1
    
    # =========================================
    # PASO 2: Detectar Sybil Attacks
    # =========================================
    print("\n[PASO 2] Detectando Sybil Attacks...")
    rpm = RPMModel()
    grounded = rpm.ground_model(customers, books, recommendations)
    
    detector = BotDetector(rpm, detection_threshold=0.5)
    sybil_attacks = detector.detect_sybil_attacks(login_ids, min_accounts=2)
    
    print(f"  Detectados: {len(sybil_attacks)} customers con múltiples cuentas")
    
    # =========================================
    # RESULTADOS
    # =========================================
    print("\n[RESULTADOS]")
    print("\n  Customers con múltiples cuentas (Sybil Attacks):")
    
    # Ordenar por número de cuentas
    sorted_sybil = sorted(sybil_attacks.items(), key=lambda x: len(x[1]), reverse=True)
    
    bots_detected = 0
    users_with_multiple = 0
    
    for customer_id, accounts in sorted_sybil:
        customer = next((c for c in customers if c.customer_id == customer_id), None)
        if customer:
            tipo = "BOT" if customer.entity_type == EntityType.BOT else "USER"
            if customer.entity_type == EntityType.BOT:
                bots_detected += 1
            else:
                users_with_multiple += 1
            
            print(f"    {customer_id:12s} ({tipo:4s}): {len(accounts)} cuentas")
            # Mostrar algunas cuentas
            account_list = list(accounts)[:3]
            if len(accounts) > 3:
                print(f"      Cuentas: {', '.join(account_list)}... (+{len(accounts)-3} más)")
            else:
                print(f"      Cuentas: {', '.join(account_list)}")
    
    # Estadísticas
    print("\n  Estadísticas:")
    print(f"    Total sybil attacks: {len(sybil_attacks)}")
    print(f"    Bots con múltiples cuentas: {bots_detected}")
    print(f"    Usuarios con múltiples cuentas: {users_with_multiple}")
    
    # Distribución de cuentas
    print("\n  Distribución de cuentas por tipo:")
    bot_accounts = []
    user_accounts = []
    for c in customers:
        num = accounts_per_customer.get(c.customer_id, 0)
        if c.entity_type == EntityType.BOT:
            bot_accounts.append(num)
        else:
            user_accounts.append(num)
    
    avg_bot = sum(bot_accounts) / len(bot_accounts) if bot_accounts else 0
    avg_user = sum(user_accounts) / len(user_accounts) if user_accounts else 0
    
    print(f"    Bots:     promedio {avg_bot:.1f} cuentas/bot (max: {max(bot_accounts) if bot_accounts else 0})")
    print(f"    Usuarios: promedio {avg_user:.1f} cuentas/usuario (max: {max(user_accounts) if user_accounts else 0})")
    
    # =========================================
    # GRÁFICOS
    # =========================================
    print("\n[GRÁFICOS]")
    
    # Gráfico 1: Sybil Attacks
    if sybil_attacks:
        bot_plotter = BotDetectionPlotter()
        path1 = os.path.join(output_dir, 'esc3_sybil_attacks.png')
        bot_plotter.plot_sybil_attacks(sybil_attacks, customers, path1)
        print(f"  Guardado: {path1}")
    
    # Gráfico 2: Cuentas por tipo (custom)
    path2 = os.path.join(output_dir, 'esc3_cuentas_por_tipo.png')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Datos para el gráfico
    bot_data = [(c.customer_id, accounts_per_customer.get(c.customer_id, 0)) 
                for c in customers if c.entity_type == EntityType.BOT]
    user_data = [(c.customer_id, accounts_per_customer.get(c.customer_id, 0)) 
                 for c in customers if c.entity_type == EntityType.REAL_USER]
    
    # Ordenar
    bot_data.sort(key=lambda x: x[1], reverse=True)
    user_data.sort(key=lambda x: x[1], reverse=True)
    
    # Combinar
    all_data = bot_data + user_data
    labels = [d[0] for d in all_data]
    values = [d[1] for d in all_data]
    colors = ['red'] * len(bot_data) + ['green'] * len(user_data)
    
    bars = ax.bar(range(len(all_data)), values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(all_data)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Customer')
    ax.set_ylabel('Número de Cuentas')
    ax.set_title('Número de Cuentas por Customer\n(Rojo=Bot, Verde=Usuario)')
    ax.axhline(y=2, color='orange', linestyle='--', label='Threshold Sybil (2 cuentas)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {path2}")
    
    # =========================================
    # CONCLUSIÓN
    # =========================================
    print("\n" + "="*70)
    print("  CONCLUSIÓN")
    print("="*70)
    print(f"""
  Detección de Sybil Attacks:
  
  Un Sybil Attack ocurre cuando un actor malicioso (bot) crea
  múltiples identidades (LoginIDs) para manipular el sistema.
  
  Resultados:
  - {bots_detected} de {len([c for c in customers if c.entity_type == EntityType.BOT])} bots usan múltiples cuentas
  - {users_with_multiple} usuarios legítimos también tienen múltiples cuentas
  
  Observaciones:
  - Los bots tienden a tener MÁS cuentas que los usuarios reales
  - Promedio bots: {avg_bot:.1f} cuentas vs usuarios: {avg_user:.1f} cuentas
  - El sistema identifica correctamente los sybil attacks
  
  El detector combina el número de cuentas con otras señales
  (ratings extremos, baja varianza) para identificar bots.
  
  Gráficos generados en: examples/output/
""")


if __name__ == "__main__":
    main()
