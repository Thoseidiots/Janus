# Coherence Explainability Layer

This document explicitly details how the 10 Coherence Principles are implemented and maintained within the generated synthetic database. The goal is to provide a ground-truth reference for understanding the logical constraints governing the dataset.

## 1. Structural Coherence
**Mechanism**: SQLite `SCHEMA` definition (`schema.sql`).
**Explanation**: All generated records are strictly constrained by SQL types, `PRIMARY KEY` uniqueness, and `FOREIGN KEY` dependencies. Furthermore, `CHECK` constraints prevent negative prices or quantities (e.g., `CHECK(base_price > 0)`). The data cannot structurally deviate from this schema without throwing an exception during insertion.

## 2. Temporal Coherence
**Mechanism**: Python `datetime` sequencing in `generate_dataset.py`.
**Explanation**: A strict timeline is enforced across entities:
* `User.created_at` (2020-2021) strictly precedes `Order.order_date` (2022-2024).
* `Payment.payment_date` is strictly 0 to 2 days *after* `Order.order_date`.
* `Shipment.dispatched_date` is strictly 1 to 3 days *after* `Payment.payment_date`.
* `Shipment.delivered_date` is strictly 2 to 10 days *after* `Shipment.dispatched_date`.
*(Note: Temporal coherence is intentionally violated in ~0.5% of records, logged in `anomalies_log.json`)*.

## 3. Logical Coherence
**Mechanism**: Arithmetic alignment during the `Order_Items` generation phase.
**Explanation**: The `Orders.total_amount` is calculated strictly by summing `(Order_Items.quantity * Order_Items.unit_price)` for every line item associated with that Order ID. If an order has 3 items, the grand total mathematically matches the sum of the parts. 
*(Note: Logical coherence is intentionally violated in ~0.5% of records)*.

## 4. Contextual Coherence
**Mechanism**: `Role` filtering in `generate_dataset.py`.
**Explanation**: Users are strictly partitioned into `Customer` and `Supplier` roles. When generating a `Product`, the generator only assigns `supplier_id` from the subset of users with the `Supplier` role. Conversely, `Orders` only assign `customer_id` from the `Customer` subset. A Customer cannot contextually supply a product in this ecosystem.

## 5. Referential Coherence
**Mechanism**: `PRAGMA foreign_keys = ON` in SQLite.
**Explanation**: It is mathematically impossible for an `Order_Item` to reference a `Product` that does not exist. The generation script first fully populates the `Users` and `Products` tables before generating any `Orders`.

## 6. Pattern Consistency
**Mechanism**: Hardcoded statistical probabilities.
**Explanation**: 
* Roles are assigned on a consistent 90/10 split (90% Customers, 10% Suppliers).
* Payment success flows naturally into shipment dispatch. Cancelled orders never result in a dispatched shipment. This state transition pattern remains consistent across all 10,000+ orders.

## 7. Controlled Incoherence
**Mechanism**: The `ANOMALY_RATE` probability roll.
**Explanation**: We intentionally introduce contrasting negative examples at a 0.5% frequency.
* **Temporal Drift**: A shipment's dispatch date is maliciously rewritten to be *before* the order was placed.
* **Logical Mismatch**: An order's total amount is artificially inflated by 10%, breaking the mathematical link with its `Order_Items`.
Every injected anomaly is recorded in `anomalies_log.json` to act as ground-truth labels for contrastive learning.

## 8. Multi-Layer Representation
**Mechanism**: Cross-format exportation.
**Explanation**: The dataset is represented via:
* **Raw Database**: `coherent_dataset.db`
* **Metadata Schema**: `schema.sql` and `relationships.json`
* **Temporal Events**: `events_log.json` (flattened time-series graph of transitions)

## 9. Explainability Layer
**Mechanism**: This document.
**Explanation**: It maps the abstract generation code back to the underlying coherence principles demanded by the prompt.

## 10. Scalability
**Mechanism**: Parameterized `NUM_USERS`, `NUM_PRODUCTS`, and `NUM_ORDERS` variables.
**Explanation**: The current output directory contains a 10,000-order sample. To scale to 10,000,000 orders, one only needs to alter `NUM_ORDERS = 10000000` in `generate_dataset.py` and run the generator again. The script uses chunked batch execution and `executemany` bindings, making it capable of scaling effortlessly to millions of rows while maintaining perfect coherence.
