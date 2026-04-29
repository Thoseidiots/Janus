import sqlite3
import json
import random
import datetime
import uuid
import os

DB_PATH = 'output/coherent_dataset.db'
SCHEMA_PATH = 'schema.sql'
EVENTS_LOG_PATH = 'output/events_log.json'
ANOMALIES_LOG_PATH = 'output/anomalies_log.json'

NUM_USERS = 2000
NUM_PRODUCTS = 500
NUM_ORDERS = 10000

ANOMALY_RATE = 0.005 # 0.5% chance to inject an anomaly

NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy", "Mallory", "Victor", "Peggy", "Trent", "Walter"]
COUNTRIES = ["USA", "Canada", "UK", "Germany", "France", "Japan", "Australia", "Brazil", "India", "China"]
CATEGORIES = ["Electronics", "Books", "Clothing", "Home & Garden", "Toys", "Sports"]
CARRIERS = ["FedEx", "UPS", "USPS", "DHL"]

def setup_db():
    os.makedirs('output', exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    
    with open(SCHEMA_PATH, 'r') as f:
        conn.executescript(f.read())
        
    return conn

def get_random_date(start_year=2022, end_year=2024):
    start = datetime.datetime(start_year, 1, 1)
    end = datetime.datetime(end_year, 12, 31)
    return start + datetime.timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

def main():
    conn = setup_db()
    cursor = conn.cursor()
    
    events_log = []
    anomalies_log = []
    
    print("Generating Users...")
    users = []
    suppliers = []
    customers = []
    for _ in range(NUM_USERS):
        user_id = str(uuid.uuid4())
        role = 'Supplier' if random.random() < 0.1 else 'Customer'
        name = random.choice(NAMES) + " " + str(random.randint(1, 1000))
        country = random.choice(COUNTRIES)
        created_at = get_random_date(2020, 2021) # Users created early
        
        users.append((user_id, role, name, country, created_at.isoformat()))
        if role == 'Supplier':
            suppliers.append(user_id)
        else:
            customers.append(user_id)
            
    cursor.executemany("INSERT INTO Users VALUES (?, ?, ?, ?, ?)", users)
    
    print("Generating Products...")
    products = []
    product_dict = {} # id -> base_price
    for _ in range(NUM_PRODUCTS):
        product_id = str(uuid.uuid4())
        supplier_id = random.choice(suppliers)
        name = "Product " + str(random.randint(1000, 9999))
        category = random.choice(CATEGORIES)
        base_price = round(random.uniform(5.0, 500.0), 2)
        stock = random.randint(50, 1000)
        
        products.append((product_id, supplier_id, name, category, base_price, stock))
        product_dict[product_id] = base_price
        
    cursor.executemany("INSERT INTO Products VALUES (?, ?, ?, ?, ?, ?)", products)
    
    print("Generating Orders, Items, Payments, and Shipments...")
    
    orders = []
    order_items = []
    payments = []
    shipments = []
    
    for i in range(NUM_ORDERS):
        order_id = str(uuid.uuid4())
        customer_id = random.choice(customers)
        # Contextual/Temporal Coherence: order date is strictly after user creation date (which is 2020/2021)
        order_date = get_random_date(2022, 2024)
        
        # Generate Items
        num_items = random.randint(1, 5)
        chosen_products = random.sample(list(product_dict.keys()), num_items)
        
        expected_total = 0.0
        for p_id in chosen_products:
            item_id = str(uuid.uuid4())
            quantity = random.randint(1, 5)
            unit_price = product_dict[p_id]
            expected_total += quantity * unit_price
            order_items.append((item_id, order_id, p_id, quantity, unit_price))
            
        total_amount = round(expected_total, 2)
        
        status_roll = random.random()
        if status_roll < 0.05:
            status = 'Cancelled'
        elif status_roll < 0.10:
            status = 'Pending'
        else:
            status = 'Delivered'
            
        # INCOHERENCE INJECTION: Logical mismatch on total amount
        if random.random() < ANOMALY_RATE:
            anomaly_type = "Logical: Total Amount Mismatch"
            total_amount = round(total_amount * 1.1, 2) # Off by 10%
            anomalies_log.append({
                "entity": "Order",
                "id": order_id,
                "type": anomaly_type,
                "description": f"Total amount ({total_amount}) does not match line items sum ({round(expected_total, 2)})."
            })
            
        orders.append((order_id, customer_id, order_date.isoformat(), total_amount, status))
        events_log.append({"timestamp": order_date.isoformat(), "event": "ORDER_PLACED", "entity_id": order_id})
        
        if status in ('Paid', 'Shipped', 'Delivered'):
            payment_id = str(uuid.uuid4())
            # Temporal Coherence: payment happens 0-2 days after order
            payment_date = order_date + datetime.timedelta(days=random.randint(0, 2))
            payments.append((payment_id, order_id, total_amount, payment_date.isoformat(), 'Success'))
            events_log.append({"timestamp": payment_date.isoformat(), "event": "PAYMENT_SUCCESS", "entity_id": order_id})
            
            if status in ('Shipped', 'Delivered'):
                shipment_id = str(uuid.uuid4())
                carrier = random.choice(CARRIERS)
                # Temporal Coherence: dispatch happens 1-3 days after payment
                dispatched_date = payment_date + datetime.timedelta(days=random.randint(1, 3))
                
                # INCOHERENCE INJECTION: Temporal drift (dispatch before order)
                if random.random() < ANOMALY_RATE:
                    anomaly_type = "Temporal: Dispatch before Order"
                    dispatched_date = order_date - datetime.timedelta(days=2)
                    anomalies_log.append({
                        "entity": "Shipment",
                        "id": shipment_id,
                        "type": anomaly_type,
                        "description": f"Shipment dispatched ({dispatched_date.isoformat()}) before order placed ({order_date.isoformat()})."
                    })
                
                if status == 'Delivered':
                    # Delivery happens 2-10 days after dispatch
                    delivered_date = dispatched_date + datetime.timedelta(days=random.randint(2, 10))
                    shipments.append((shipment_id, order_id, carrier, str(uuid.uuid4())[:8], dispatched_date.isoformat(), delivered_date.isoformat(), 'Delivered'))
                    events_log.append({"timestamp": dispatched_date.isoformat(), "event": "SHIPMENT_DISPATCHED", "entity_id": shipment_id})
                    events_log.append({"timestamp": delivered_date.isoformat(), "event": "SHIPMENT_DELIVERED", "entity_id": shipment_id})
                else:
                    shipments.append((shipment_id, order_id, carrier, str(uuid.uuid4())[:8], dispatched_date.isoformat(), None, 'In_Transit'))
                    events_log.append({"timestamp": dispatched_date.isoformat(), "event": "SHIPMENT_DISPATCHED", "entity_id": shipment_id})
                    
    # Disable foreign keys temporarily for bulk insert of potentially anomalous data if needed, 
    # but our anomalies don't violate FK constraints, so it's safe.
    try:
        cursor.executemany("INSERT INTO Orders VALUES (?, ?, ?, ?, ?)", orders)
        cursor.executemany("INSERT INTO Order_Items VALUES (?, ?, ?, ?, ?)", order_items)
        cursor.executemany("INSERT INTO Payments VALUES (?, ?, ?, ?, ?)", payments)
        cursor.executemany("INSERT INTO Shipments VALUES (?, ?, ?, ?, ?, ?, ?)", shipments)
        conn.commit()
    except sqlite3.IntegrityError as e:
        print(f"Integrity Error: {e}")
        conn.rollback()

    with open(EVENTS_LOG_PATH, 'w') as f:
        # sort events by timestamp
        events_log.sort(key=lambda x: x['timestamp'])
        json.dump(events_log, f, indent=2)
        
    with open(ANOMALIES_LOG_PATH, 'w') as f:
        json.dump(anomalies_log, f, indent=2)

    print("Database generation complete!")
    print(f"Total Users: {len(users)}")
    print(f"Total Products: {len(products)}")
    print(f"Total Orders: {len(orders)}")
    print(f"Total Anomalies Injected: {len(anomalies_log)}")

if __name__ == "__main__":
    main()
