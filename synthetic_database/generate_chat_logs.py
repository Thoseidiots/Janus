import sqlite3
import json
import random
import os

DB_PATH = 'output/coherent_dataset.db'
EVENTS_LOG_PATH = 'output/events_log.json'
ANOMALIES_LOG_PATH = 'output/anomalies_log.json'
CHAT_LOGS_PATH = 'output/conversational_dataset.json'

# Conversational templates
GREETINGS = [
    "Hi, I'm checking on my order.",
    "Hello! Can you tell me the status of my recent purchase?",
    "Hey there, where is my stuff?",
    "Good morning, I need an update on order {order_id}."
]

def generate_chat_logs():
    if not os.path.exists(DB_PATH):
        print("Database not found. Please run generate_dataset.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Load anomalies to identify orders with anomalies
    anomalous_orders = set()
    if os.path.exists(ANOMALIES_LOG_PATH):
        with open(ANOMALIES_LOG_PATH, 'r') as f:
            anomalies = json.load(f)
            for a in anomalies:
                if a['entity'] == 'Order':
                    anomalous_orders.add(a['id'])
                elif a['entity'] == 'Shipment':
                    # Need to find order_id for this shipment
                    cursor.execute("SELECT order_id FROM Shipments WHERE shipment_id = ?", (a['id'],))
                    res = cursor.fetchone()
                    if res:
                        anomalous_orders.add(res['order_id'])

    # Select 500 random orders to generate conversations for, including some anomalous ones if possible
    cursor.execute("SELECT order_id, customer_id, total_amount, status, order_date FROM Orders ORDER BY RANDOM() LIMIT 500")
    orders = cursor.fetchall()

    conversations = []

    for order in orders:
        order_id = order['order_id']
        is_anomalous = order_id in anomalous_orders

        # Get customer details
        cursor.execute("SELECT name FROM Users WHERE user_id = ?", (order['customer_id'],))
        customer_name = cursor.fetchone()['name']

        # Get items
        cursor.execute("SELECT p.name, oi.quantity, oi.unit_price FROM Order_Items oi JOIN Products p ON oi.product_id = p.product_id WHERE oi.order_id = ?", (order_id,))
        items = cursor.fetchall()
        items_str = ", ".join([f"{item['quantity']}x {item['name']} (at ${item['unit_price']})" for item in items])

        # Get shipment details
        cursor.execute("SELECT carrier, tracking_number, dispatched_date, delivered_date, status FROM Shipments WHERE order_id = ?", (order_id,))
        shipment = cursor.fetchone()

        dialogue = []
        
        # Turn 1: User initiates
        user_greet = random.choice(GREETINGS).format(order_id=order_id[:8])
        dialogue.append({"role": "user", "content": user_greet})

        # Turn 2: AI responds with context
        ai_resp = f"Hello {customer_name}! I'd be happy to help you with order {order_id[:8]}. "
        ai_resp += f"I see you ordered {items_str} on {order['order_date'][:10]} for a total of ${order['total_amount']}. "
        
        if order['status'] == 'Cancelled':
            ai_resp += "This order has been cancelled."
        elif order['status'] == 'Pending':
            ai_resp += "The order is currently pending and waiting for payment processing."
        elif shipment:
            if shipment['status'] == 'Delivered':
                ai_resp += f"Good news! This was shipped via {shipment['carrier']} and delivered on {shipment['delivered_date'][:10]}."
            elif shipment['status'] == 'In_Transit':
                ai_resp += f"This order was dispatched on {shipment['dispatched_date'][:10]} via {shipment['carrier']}. It is currently in transit."
        else:
            ai_resp += f"The order is currently marked as {order['status']}."
            
        # Incoherence handling in conversation
        if is_anomalous:
            # The user points out the logical flaw
            if shipment and shipment['dispatched_date'] < order['order_date']:
                dialogue.append({"role": "assistant", "content": ai_resp})
                dialogue.append({"role": "user", "content": f"Wait, you said it was dispatched on {shipment['dispatched_date'][:10]}, but I didn't even place the order until {order['order_date'][:10]}. How is that possible?"})
                dialogue.append({"role": "assistant", "content": "[ANOMALY_FLAGGED] You are absolutely right. I am detecting a temporal incoherence in my database records. A shipment cannot be dispatched before the order is placed."})
            else:
                dialogue.append({"role": "assistant", "content": ai_resp})
                dialogue.append({"role": "user", "content": "Wait, those item prices don't add up to the total amount you just told me."})
                dialogue.append({"role": "assistant", "content": "[ANOMALY_FLAGGED] You are correct. I am detecting a mathematical logical mismatch between the line items and the grand total."})
        else:
            dialogue.append({"role": "assistant", "content": ai_resp})
            dialogue.append({"role": "user", "content": "Perfect, that makes sense. Thank you!"})
            dialogue.append({"role": "assistant", "content": "You're welcome! Let me know if you need anything else."})

        conversations.append({
            "order_id": order_id,
            "is_anomalous": is_anomalous,
            "dialogue": dialogue
        })

    with open(CHAT_LOGS_PATH, 'w') as f:
        json.dump(conversations, f, indent=2)

    print(f"Generated {len(conversations)} conversational logs successfully!")
    print(f"Dataset saved to {CHAT_LOGS_PATH}")

if __name__ == "__main__":
    generate_chat_logs()
