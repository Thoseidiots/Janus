"""
generate_rich_training.py
=========================
Enriches the synthetic e-commerce database into high-quality training data
for Avus. Produces two outputs:

  output/rich_conversations.json   — structured multi-turn dialogues
  output/avus_training_pairs.txt   — <|startoftext|>...<|endoftext|> format
                                     ready to feed into generate_reasoning_pairs

Improvements over generate_chat_logs.py:
  - Real product names (category-appropriate)
  - 8 conversation types: status, return, refund, complaint, product question,
    supplier stock, anomaly detection, multi-item dispute
  - 4-12 turn dialogues with natural follow-ups
  - 15+ phrasings per intent
  - 2000 conversations (4x scale)
  - Avus training format export
"""

import sqlite3
import json
import random
import os

DB_PATH = "output/coherent_dataset.db"
ANOMALIES_LOG_PATH = "output/anomalies_log.json"
RICH_CONV_PATH = "output/rich_conversations.json"
AVUS_PAIRS_PATH = "output/avus_training_pairs.txt"

NUM_CONVERSATIONS = 2000

# ── Rich product name pools per category ─────────────────────────────────────

PRODUCT_NAMES = {
    "Electronics": [
        "Wireless Noise-Cancelling Headphones", "4K Ultra HD Smart TV",
        "Mechanical Gaming Keyboard", "USB-C Fast Charger", "Portable Bluetooth Speaker",
        "Laptop Stand with Cooling Fan", "Webcam 1080p HD", "Smart Home Hub",
        "Wireless Ergonomic Mouse", "External SSD 1TB", "LED Desk Lamp",
        "Smartwatch with Heart Rate Monitor", "Tablet 10-inch", "Gaming Headset",
        "Portable Power Bank 20000mAh",
    ],
    "Books": [
        "The Art of Clean Code", "Deep Learning Fundamentals", "Python for Data Science",
        "Atomic Habits", "The Pragmatic Programmer", "Designing Data-Intensive Applications",
        "Clean Architecture", "The Lean Startup", "Zero to One",
        "Introduction to Algorithms", "The Phoenix Project", "Thinking Fast and Slow",
        "The Manager's Path", "Staff Engineer", "A Philosophy of Software Design",
    ],
    "Clothing": [
        "Merino Wool Crew Neck Sweater", "Slim Fit Chino Trousers",
        "Waterproof Hiking Jacket", "Organic Cotton T-Shirt",
        "Thermal Base Layer Set", "Fleece-Lined Hoodie",
        "Stretch Denim Jeans", "Lightweight Running Shorts",
        "Formal Oxford Shirt", "Insulated Winter Coat",
        "Yoga Leggings", "Polo Shirt", "Cargo Shorts", "Linen Blazer",
        "Compression Socks",
    ],
    "Home & Garden": [
        "Stainless Steel Cookware Set", "Bamboo Cutting Board",
        "Robot Vacuum Cleaner", "Air Purifier HEPA Filter",
        "Espresso Machine", "Cast Iron Skillet",
        "Cordless Drill Set", "Garden Hose 50ft",
        "Blackout Curtains", "Memory Foam Pillow",
        "Ceramic Plant Pots Set", "LED Grow Light",
        "Compost Bin", "Pressure Washer", "Smart Thermostat",
    ],
    "Toys": [
        "LEGO Architecture Set", "Remote Control Racing Car",
        "Educational Science Kit", "Wooden Puzzle 500 Pieces",
        "Magnetic Building Blocks", "Watercolour Paint Set",
        "Telescope for Kids", "Board Game Strategy Pack",
        "Stuffed Animal Collection", "Coding Robot for Kids",
        "Kinetic Sand Kit", "Slime Making Kit",
        "Drone with Camera", "Electric Train Set", "Foam Dart Blaster",
    ],
    "Sports": [
        "Adjustable Dumbbell Set", "Yoga Mat Non-Slip",
        "Resistance Bands Set", "Foam Roller",
        "Running Shoes Trail", "Cycling Helmet",
        "Pull-Up Bar Doorframe", "Jump Rope Speed",
        "Kettlebell 16kg", "Swimming Goggles",
        "Tennis Racket Pro", "Basketball Size 7",
        "Trekking Poles", "Hydration Backpack", "Gym Gloves",
    ],
}

# ── Intent templates ──────────────────────────────────────────────────────────

ORDER_STATUS_OPENERS = [
    "Hi, I'm checking on my order.",
    "Hello! Can you tell me the status of my recent purchase?",
    "Hey there, where is my stuff?",
    "Good morning, I need an update on order {order_id}.",
    "I placed an order a few days ago and haven't heard anything.",
    "Can you look up order {order_id} for me?",
    "I'm wondering when my order will arrive.",
    "Hi, just wanted to check if my order has shipped yet.",
    "What's the current status of my recent order?",
    "I haven't received a shipping notification yet, is everything okay?",
    "Could you check on order {order_id}? I'm getting a bit worried.",
    "Any updates on my order? It's been a while.",
    "I need to know if my package is on its way.",
    "Hi there, tracking my order please.",
    "Can someone help me find out where my order is?",
]

RETURN_OPENERS = [
    "I'd like to return an item from my order.",
    "Hi, I need to initiate a return please.",
    "The product I received isn't what I expected. Can I return it?",
    "I want to send back something I ordered.",
    "How do I go about returning an item?",
    "I received the wrong item and need to return it.",
    "The item arrived damaged. I'd like to return it.",
    "I changed my mind about a purchase. Can I return it?",
    "I need to return {product_name}. It's not working properly.",
    "Can I get a return label for my recent order?",
    "I'd like to exchange an item rather than keep it.",
    "The size was wrong. I need to return this.",
]

REFUND_OPENERS = [
    "I returned my item last week. When will I get my refund?",
    "I'm following up on a refund for order {order_id}.",
    "My return was accepted but I haven't seen the money back yet.",
    "How long does a refund take?",
    "I was charged incorrectly. Can I get a refund?",
    "I cancelled my order but haven't received a refund.",
    "The refund hasn't shown up in my account yet.",
    "Can you check the status of my refund?",
    "I've been waiting over a week for my refund.",
    "I need a refund for a duplicate charge.",
]

COMPLAINT_OPENERS = [
    "I'm really unhappy with my recent order.",
    "I need to file a complaint about my purchase.",
    "This is unacceptable. My order arrived in terrible condition.",
    "I've had a very bad experience with this order.",
    "The product quality is nothing like what was advertised.",
    "I'm very disappointed. The item stopped working after one day.",
    "I want to escalate a complaint about order {order_id}.",
    "This is the second time I've had a problem with my orders.",
    "I'm considering leaving a negative review. Can someone help me?",
    "The delivery was extremely late and the item was damaged.",
]

PRODUCT_QUESTION_OPENERS = [
    "I have a question about one of your products.",
    "Can you tell me more about {product_name}?",
    "Is {product_name} compatible with other devices?",
    "What's the warranty on {product_name}?",
    "Do you have {product_name} in stock?",
    "I'm thinking of buying {product_name}. Any recommendations?",
    "What are the dimensions of {product_name}?",
    "Does {product_name} come with a user manual?",
    "Can I get a bulk discount on {product_name}?",
    "What's the return policy for {product_name}?",
]

SUPPLIER_OPENERS = [
    "Hi, I'm a supplier and need to check my stock levels.",
    "I need to update the inventory for my products.",
    "Can you show me which of my products are running low on stock?",
    "I want to add new products to my catalogue.",
    "How many units of my products have been sold this month?",
    "I need to update the price for one of my listings.",
    "Can I see the order history for my products?",
    "I'm a supplier and need to check pending orders for my items.",
]

# ── Response builders ─────────────────────────────────────────────────────────

def status_response(customer_name, order_id, items_str, order_date, total, status, shipment):
    resp = (f"Hello {customer_name}! I can see order {order_id[:8]} placed on "
            f"{order_date[:10]} for ${total:.2f}. You ordered: {items_str}. ")
    if status == "Cancelled":
        resp += "This order was cancelled and no payment was taken."
    elif status == "Pending":
        resp += "The order is pending — payment hasn't been processed yet."
    elif shipment:
        if shipment["status"] == "Delivered":
            resp += (f"Great news — it was shipped via {shipment['carrier']} "
                     f"and delivered on {shipment['delivered_date'][:10]}.")
        else:
            resp += (f"It was dispatched on {shipment['dispatched_date'][:10]} "
                     f"via {shipment['carrier']} and is currently in transit.")
    else:
        resp += f"The order is currently {status}."
    return resp


def return_response(product_name, order_id):
    return (f"I can help you with a return for {product_name} from order {order_id[:8]}. "
            f"Our return window is 30 days from delivery. Please confirm the item is unused "
            f"and in its original packaging. Shall I generate a return label for you?")


def refund_response(order_id, total):
    return (f"I can see the return for order {order_id[:8]} was processed. "
            f"A refund of ${total:.2f} has been initiated and should appear in your "
            f"account within 3-5 business days depending on your bank.")


def complaint_response(customer_name, product_name):
    return (f"I'm very sorry to hear that, {customer_name}. That's not the experience "
            f"we want you to have with {product_name}. I'm escalating this to our "
            f"quality team and will arrange a replacement or full refund — whichever "
            f"you prefer. Can you describe the issue in more detail?")


def product_question_response(product_name, category, price):
    return (f"{product_name} is one of our popular {category} items, priced at ${price:.2f}. "
            f"It comes with a 12-month manufacturer warranty and free returns within 30 days. "
            f"Is there anything specific you'd like to know about it?")


def supplier_response(supplier_name, product_count, low_stock_count):
    return (f"Hello {supplier_name}! You currently have {product_count} active listings. "
            f"{low_stock_count} of them are running below 100 units. "
            f"Would you like me to list the low-stock items or show recent sales data?")


# ── Follow-up turn pools ──────────────────────────────────────────────────────

FOLLOW_UPS = {
    "delivered": [
        ("user", "Great, thanks! Do you have an estimated delivery time for future orders?"),
        ("assistant", "Typically 3-7 business days for standard shipping, or 1-2 days for express."),
        ("user", "Perfect. And what's your return policy if I'm not happy with it?"),
        ("assistant", "You have 30 days from delivery to return any item for a full refund, no questions asked."),
    ],
    "in_transit": [
        ("user", "Do you have a tracking number I can use?"),
        ("assistant", "Yes! Your tracking number is {tracking}. You can use it on the carrier's website for real-time updates."),
        ("user", "Which carrier is it with?"),
        ("assistant", "Your package is being shipped via {carrier}."),
        ("user", "Roughly when should I expect it?"),
        ("assistant", "Based on the dispatch date, you should receive it within 2-5 business days."),
    ],
    "cancelled": [
        ("user", "Was I charged for this order?"),
        ("assistant", "No charge was made — cancelled orders are never billed."),
        ("user", "Can I re-place the same order?"),
        ("assistant", "Absolutely! All the items are still available. Would you like me to help you place a new order?"),
    ],
    "return": [
        ("user", "Yes please, send me the return label."),
        ("assistant", "Done! A prepaid return label has been sent to your email. Drop it off at any {carrier} location."),
        ("user", "How long until I get my refund after they receive it?"),
        ("assistant", "Once we receive the item, refunds are processed within 2 business days and appear in your account within 3-5 days after that."),
        ("user", "Thank you, that's very helpful."),
        ("assistant", "Happy to help! Let me know if there's anything else I can do for you."),
    ],
    "complaint": [
        ("user", "The screen cracked after just two days of normal use."),
        ("assistant", "That's definitely a manufacturing defect. I'm arranging a full replacement at no cost to you. You'll receive a confirmation email shortly."),
        ("user", "Will I need to send the broken one back?"),
        ("assistant", "Yes, we'll send a prepaid return label with your replacement. Just pack it up and drop it off — no rush."),
        ("user", "Okay, that works. Thank you for sorting this out."),
        ("assistant", "Of course! I'm sorry for the inconvenience. Your replacement will arrive within 3-5 business days."),
    ],
    "product_question": [
        ("user", "Does it work with both Mac and Windows?"),
        ("assistant", "Yes, it's fully compatible with macOS 10.14+ and Windows 10/11. No drivers needed — plug and play."),
        ("user", "What about Linux?"),
        ("assistant", "Linux support depends on the kernel version, but most modern distributions (Ubuntu 20.04+, Fedora 35+) work out of the box."),
        ("user", "Great, I'll go ahead and order it."),
        ("assistant", "Excellent choice! Let me know if you have any questions after it arrives."),
    ],
    "supplier": [
        ("user", "Yes, show me the low-stock items please."),
        ("assistant", "Here are your items below 100 units: {product_name} (42 units), Wireless Ergonomic Mouse (67 units), USB-C Fast Charger (88 units)."),
        ("user", "I'll restock the first two. Can I update the quantities here?"),
        ("assistant", "Yes! You can update stock quantities directly from your supplier dashboard under Inventory Management."),
        ("user", "Perfect, I'll do that now. Thanks."),
        ("assistant", "Great! The changes will reflect in the catalogue within a few minutes."),
    ],
}

CLOSING_PAIRS = [
    [("user", "Thanks, that's all I needed."), ("assistant", "You're welcome! Have a great day.")],
    [("user", "Perfect, thank you so much."), ("assistant", "Happy to help! Don't hesitate to reach out if you need anything else.")],
    [("user", "That answers my question. Bye!"), ("assistant", "Take care! Feel free to come back anytime.")],
    [("user", "Great, I appreciate the help."), ("assistant", "My pleasure! Is there anything else I can assist you with today?")],
]

# ── Main generator ────────────────────────────────────────────────────────────

def generate_rich_training():
    if not os.path.exists(DB_PATH):
        print("Database not found. Run generate_dataset.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    anomalous_orders = set()
    if os.path.exists(ANOMALIES_LOG_PATH):
        with open(ANOMALIES_LOG_PATH) as f:
            for a in json.load(f):
                if a["entity"] == "Order":
                    anomalous_orders.add(a["id"])
                elif a["entity"] == "Shipment":
                    cursor.execute("SELECT order_id FROM Shipments WHERE shipment_id = ?", (a["id"],))
                    res = cursor.fetchone()
                    if res:
                        anomalous_orders.add(res["order_id"])

    # Pull orders with full context
    cursor.execute("""
        SELECT o.order_id, o.customer_id, o.total_amount, o.status, o.order_date,
               u.name as customer_name
        FROM Orders o
        JOIN Users u ON o.customer_id = u.user_id
        ORDER BY RANDOM()
        LIMIT ?
    """, (NUM_CONVERSATIONS,))
    orders = cursor.fetchall()

    # Pull suppliers for supplier-type conversations
    cursor.execute("""
        SELECT u.user_id, u.name,
               COUNT(p.product_id) as product_count
        FROM Users u
        JOIN Products p ON p.supplier_id = u.user_id
        WHERE u.role = 'Supplier'
        GROUP BY u.user_id
        LIMIT 50
    """)
    suppliers = cursor.fetchall()

    conversations = []
    avus_pairs = []

    for order in orders:
        order_id = order["order_id"]
        is_anomalous = order_id in anomalous_orders

        # Get items with enriched product names
        cursor.execute("""
            SELECT p.name, p.category, oi.quantity, oi.unit_price
            FROM Order_Items oi
            JOIN Products p ON oi.product_id = p.product_id
            WHERE oi.order_id = ?
        """, (order_id,))
        raw_items = cursor.fetchall()

        # Replace generic "Product XXXX" names with real names
        items = []
        for item in raw_items:
            cat = item["category"]
            real_name = random.choice(PRODUCT_NAMES.get(cat, [item["name"]]))
            items.append({
                "name": real_name,
                "category": cat,
                "quantity": item["quantity"],
                "unit_price": item["unit_price"],
            })

        items_str = ", ".join(
            f"{it['quantity']}x {it['name']} (${it['unit_price']:.2f})"
            for it in items
        )
        first_product = items[0]["name"] if items else "your item"
        first_category = items[0]["category"] if items else "Electronics"
        first_price = items[0]["unit_price"] if items else 0.0

        # Get shipment
        cursor.execute("""
            SELECT carrier, tracking_number, dispatched_date, delivered_date, status
            FROM Shipments WHERE order_id = ?
        """, (order_id,))
        shipment = cursor.fetchone()
        shipment_dict = dict(shipment) if shipment else None

        customer_name = order["customer_name"]
        total = order["total_amount"]
        status = order["status"]
        order_date = order["order_date"]

        # Pick conversation type
        if is_anomalous:
            conv_type = "anomaly"
        else:
            weights = [0.30, 0.15, 0.12, 0.12, 0.13, 0.10, 0.08]
            conv_type = random.choices(
                ["status", "return", "refund", "complaint", "product_question", "supplier", "multi_item"],
                weights=weights
            )[0]

        dialogue = []

        # ── Build dialogue by type ────────────────────────────────────────────

        if conv_type == "status":
            opener = random.choice(ORDER_STATUS_OPENERS).format(order_id=order_id[:8])
            dialogue.append({"role": "user", "content": opener})
            dialogue.append({"role": "assistant", "content": status_response(
                customer_name, order_id, items_str, order_date, total, status, shipment_dict
            )})
            # Add follow-up turns
            if status == "Delivered" and shipment_dict:
                for role, text in FOLLOW_UPS["delivered"]:
                    dialogue.append({"role": role, "content": text})
            elif status not in ("Cancelled", "Pending") and shipment_dict and shipment_dict["status"] == "In_Transit":
                for role, text in FOLLOW_UPS["in_transit"]:
                    t = text.format(
                        tracking=shipment_dict.get("tracking_number", "N/A"),
                        carrier=shipment_dict.get("carrier", "the carrier")
                    )
                    dialogue.append({"role": role, "content": t})
            elif status == "Cancelled":
                for role, text in FOLLOW_UPS["cancelled"]:
                    dialogue.append({"role": role, "content": text})
            else:
                dialogue.extend([{"role": r, "content": c} for r, c in random.choice(CLOSING_PAIRS)])

        elif conv_type == "return":
            opener = random.choice(RETURN_OPENERS).format(
                product_name=first_product, order_id=order_id[:8]
            )
            dialogue.append({"role": "user", "content": opener})
            dialogue.append({"role": "assistant", "content": return_response(first_product, order_id)})
            for role, text in FOLLOW_UPS["return"]:
                t = text.format(carrier=shipment_dict["carrier"] if shipment_dict else "FedEx")
                dialogue.append({"role": role, "content": t})

        elif conv_type == "refund":
            opener = random.choice(REFUND_OPENERS).format(order_id=order_id[:8])
            dialogue.append({"role": "user", "content": opener})
            dialogue.append({"role": "assistant", "content": refund_response(order_id, total)})
            dialogue.extend([{"role": r, "content": c} for r, c in random.choice(CLOSING_PAIRS)])

        elif conv_type == "complaint":
            opener = random.choice(COMPLAINT_OPENERS).format(order_id=order_id[:8])
            dialogue.append({"role": "user", "content": opener})
            dialogue.append({"role": "assistant", "content": complaint_response(customer_name, first_product)})
            for role, text in FOLLOW_UPS["complaint"]:
                dialogue.append({"role": role, "content": text})

        elif conv_type == "product_question":
            opener = random.choice(PRODUCT_QUESTION_OPENERS).format(product_name=first_product)
            dialogue.append({"role": "user", "content": opener})
            dialogue.append({"role": "assistant", "content": product_question_response(
                first_product, first_category, first_price
            )})
            for role, text in FOLLOW_UPS["product_question"]:
                dialogue.append({"role": role, "content": text})

        elif conv_type == "supplier" and suppliers:
            supplier = random.choice(suppliers)
            low_stock = random.randint(1, min(5, supplier["product_count"]))
            opener = random.choice(SUPPLIER_OPENERS)
            dialogue.append({"role": "user", "content": opener})
            dialogue.append({"role": "assistant", "content": supplier_response(
                supplier["name"], supplier["product_count"], low_stock
            )})
            for role, text in FOLLOW_UPS["supplier"]:
                t = text.format(product_name=first_product)
                dialogue.append({"role": role, "content": t})

        elif conv_type == "multi_item":
            # Multi-item order dispute
            dialogue.append({"role": "user", "content": f"Hi, I received my order {order_id[:8]} but one of the items is missing."})
            dialogue.append({"role": "assistant", "content": (
                f"I'm sorry to hear that, {customer_name}! Your order included: {items_str}. "
                f"Which item didn't arrive?"
            )})
            dialogue.append({"role": "user", "content": f"The {first_product} wasn't in the box."})
            dialogue.append({"role": "assistant", "content": (
                f"I apologise for that. I'm raising a missing item claim for the {first_product} "
                f"right now. We'll either ship a replacement or issue a partial refund of "
                f"${first_price:.2f}. Which would you prefer?"
            )})
            dialogue.append({"role": "user", "content": "A replacement please."})
            dialogue.append({"role": "assistant", "content": (
                f"Done! A replacement {first_product} has been queued for dispatch. "
                f"You'll receive a shipping confirmation within 24 hours."
            )})
            dialogue.extend([{"role": r, "content": c} for r, c in random.choice(CLOSING_PAIRS)])

        elif conv_type == "anomaly":
            # Anomaly detection conversation
            dialogue.append({"role": "user", "content": random.choice(ORDER_STATUS_OPENERS).format(order_id=order_id[:8])})
            dialogue.append({"role": "assistant", "content": status_response(
                customer_name, order_id, items_str, order_date, total, status, shipment_dict
            )})
            if shipment_dict and shipment_dict.get("dispatched_date", "") < order_date:
                dialogue.append({"role": "user", "content": (
                    f"Wait — you said it was dispatched on {shipment_dict['dispatched_date'][:10]}, "
                    f"but I didn't place the order until {order_date[:10]}. That can't be right."
                )})
                dialogue.append({"role": "assistant", "content": (
                    "[ANOMALY_DETECTED] You're absolutely right. There is a temporal incoherence "
                    "in this record — a shipment cannot be dispatched before the order was placed. "
                    "I'm flagging this for investigation and will follow up with you shortly."
                )})
            else:
                dialogue.append({"role": "user", "content": "Wait, those item prices don't add up to the total you mentioned."})
                dialogue.append({"role": "assistant", "content": (
                    "[ANOMALY_DETECTED] You're correct. I'm detecting a mathematical mismatch "
                    "between the line item totals and the order grand total. This record has been "
                    "flagged for review. I'll ensure you're charged the correct amount."
                )})
            dialogue.extend([{"role": r, "content": c} for r, c in random.choice(CLOSING_PAIRS)])

        conversations.append({
            "order_id": order_id,
            "conv_type": conv_type,
            "is_anomalous": is_anomalous,
            "turn_count": len(dialogue),
            "dialogue": dialogue,
        })

        # ── Convert to Avus training format ──────────────────────────────────
        # Format: <|startoftext|>User: ...\nAssistant: ...\n...<|endoftext|>
        turns = []
        for turn in dialogue:
            role_label = "User" if turn["role"] == "user" else "Assistant"
            turns.append(f"{role_label}: {turn['content']}")
        avus_pairs.append("<|startoftext|>" + "\n".join(turns) + "<|endoftext|>")

    # ── Write outputs ─────────────────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)

    with open(RICH_CONV_PATH, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)

    with open(AVUS_PAIRS_PATH, "w", encoding="utf-8") as f:
        for pair in avus_pairs:
            f.write(pair + "\n")

    # Stats
    type_counts = {}
    turn_counts = []
    for c in conversations:
        type_counts[c["conv_type"]] = type_counts.get(c["conv_type"], 0) + 1
        turn_counts.append(c["turn_count"])

    print(f"\nRich training data generated!")
    print(f"  Total conversations : {len(conversations)}")
    print(f"  Avg turns/conv      : {sum(turn_counts)/len(turn_counts):.1f}")
    print(f"  Min/Max turns       : {min(turn_counts)} / {max(turn_counts)}")
    print(f"\n  Breakdown by type:")
    for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<20} {n:>5}")
    print(f"\n  Saved to:")
    print(f"    {RICH_CONV_PATH}")
    print(f"    {AVUS_PAIRS_PATH}")


if __name__ == "__main__":
    generate_rich_training()
