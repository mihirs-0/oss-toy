#!/usr/bin/env python3
"""
Generate additional test data for the supply chain demo
"""

import pandas as pd
import random
from datetime import datetime, timedelta

def generate_extended_data():
    """Generate more comprehensive test data"""
    
    # Extended items
    items = []
    categories = ['Electronics', 'Accessories', 'Furniture', 'Cables', 'Software']
    suppliers = ['TechCorp', 'AccessoryPlus', 'OfficeFurn', 'CableCo', 'SoftwarePro', 'GlobalSupply']
    
    for i in range(50):
        items.append({
            'item_id': f'ITM{i+1:03d}',
            'item_name': f'Product {i+1}',
            'category': random.choice(categories),
            'unit_cost': round(random.uniform(10, 1000), 2),
            'supplier': random.choice(suppliers)
        })
    
    # Extended warehouses
    warehouses = [
        {'warehouse_id': 'WH001', 'location': 'New York', 'capacity': 10000, 'current_utilization': 0.75},
        {'warehouse_id': 'WH002', 'location': 'Los Angeles', 'capacity': 8000, 'current_utilization': 0.82},
        {'warehouse_id': 'WH003', 'location': 'Chicago', 'capacity': 12000, 'current_utilization': 0.68},
        {'warehouse_id': 'WH004', 'location': 'Houston', 'capacity': 9000, 'current_utilization': 0.71},
        {'warehouse_id': 'WH005', 'location': 'Phoenix', 'capacity': 7500, 'current_utilization': 0.89}
    ]
    
    # Generate inventory
    inventory = []
    for warehouse in warehouses:
        for item in random.sample(items, 30):  # Each warehouse has 30 random items
            current_stock = random.randint(0, 200)
            reorder_point = random.randint(20, 80)
            max_stock = reorder_point + random.randint(100, 300)
            
            inventory.append({
                'warehouse_id': warehouse['warehouse_id'],
                'item_id': item['item_id'],
                'current_stock': current_stock,
                'reorder_point': reorder_point,
                'max_stock': max_stock,
                'last_restock_date': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d')
            })
    
    # Generate deliveries
    deliveries = []
    statuses = ['Completed', 'In Transit', 'Scheduled', 'Delayed']
    
    for i in range(100):
        deliveries.append({
            'delivery_id': f'DEL{i+1:03d}',
            'item_id': random.choice(items)['item_id'],
            'warehouse_id': random.choice(warehouses)['warehouse_id'],
            'quantity': random.randint(10, 100),
            'delivery_date': (datetime.now() + timedelta(days=random.randint(-10, 30))).strftime('%Y-%m-%d'),
            'status': random.choice(statuses),
            'supplier': random.choice(suppliers)
        })
    
    # Save to CSV files
    pd.DataFrame(items).to_csv('data/items.csv', index=False)
    pd.DataFrame(warehouses).to_csv('data/warehouses.csv', index=False)
    pd.DataFrame(inventory).to_csv('data/inventory.csv', index=False)
    pd.DataFrame(deliveries).to_csv('data/deliveries.csv', index=False)
    
    print("✅ Extended test data generated in data/ directory")

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    generate_extended_data() 