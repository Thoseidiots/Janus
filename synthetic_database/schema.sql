-- Schema for Coherent Synthetic Database (E-Commerce & Logistics)

CREATE TABLE Users (
    user_id TEXT PRIMARY KEY,
    role TEXT CHECK(role IN ('Customer', 'Supplier')),
    name TEXT NOT NULL,
    country TEXT NOT NULL,
    created_at DATETIME NOT NULL
);

CREATE TABLE Products (
    product_id TEXT PRIMARY KEY,
    supplier_id TEXT NOT NULL,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    base_price DECIMAL(10, 2) NOT NULL CHECK(base_price > 0),
    stock_quantity INTEGER NOT NULL CHECK(stock_quantity >= 0),
    FOREIGN KEY(supplier_id) REFERENCES Users(user_id)
);

CREATE TABLE Orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    order_date DATETIME NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL CHECK(total_amount >= 0),
    status TEXT CHECK(status IN ('Pending', 'Paid', 'Shipped', 'Delivered', 'Cancelled')),
    FOREIGN KEY(customer_id) REFERENCES Users(user_id)
);

CREATE TABLE Order_Items (
    order_item_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    unit_price DECIMAL(10, 2) NOT NULL CHECK(unit_price > 0),
    FOREIGN KEY(order_id) REFERENCES Orders(order_id),
    FOREIGN KEY(product_id) REFERENCES Products(product_id)
);

CREATE TABLE Payments (
    payment_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    amount DECIMAL(10, 2) NOT NULL CHECK(amount > 0),
    payment_date DATETIME NOT NULL,
    status TEXT CHECK(status IN ('Success', 'Failed')),
    FOREIGN KEY(order_id) REFERENCES Orders(order_id)
);

CREATE TABLE Shipments (
    shipment_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    carrier TEXT NOT NULL,
    tracking_number TEXT,
    dispatched_date DATETIME NOT NULL,
    delivered_date DATETIME,
    status TEXT CHECK(status IN ('In_Transit', 'Delivered', 'Returned')),
    FOREIGN KEY(order_id) REFERENCES Orders(order_id)
);
