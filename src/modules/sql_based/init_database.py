import random
from datetime import date, datetime, timedelta

import psycopg2
from psycopg2.extras import execute_values

from src.settings import settings

random.seed(42)

SCHEMA_SQL = """
DROP TABLE IF EXISTS reviews, payments, order_items, orders, products, suppliers, categories, customers, countries CASCADE;

CREATE TABLE countries (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    region TEXT NOT NULL
);

CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    country_id INTEGER NOT NULL REFERENCES countries(id),
    registered_at DATE NOT NULL
);

CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    parent_id INTEGER REFERENCES categories(id)
);

CREATE TABLE suppliers (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    country_id INTEGER NOT NULL REFERENCES countries(id)
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    category_id INTEGER NOT NULL REFERENCES categories(id),
    supplier_id INTEGER NOT NULL REFERENCES suppliers(id),
    price NUMERIC(10, 2) NOT NULL,
    stock INTEGER NOT NULL
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    status TEXT NOT NULL CHECK (status IN ('new', 'paid', 'shipped', 'delivered', 'cancelled')),
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(10, 2) NOT NULL
);

CREATE TABLE payments (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    method TEXT NOT NULL CHECK (method IN ('card', 'paypal', 'bank_transfer')),
    amount NUMERIC(10, 2) NOT NULL,
    paid_at TIMESTAMP NOT NULL
);

CREATE TABLE reviews (
    id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(id),
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    comment TEXT,
    created_at TIMESTAMP NOT NULL
);

COMMENT ON TABLE orders IS 'Customer orders';
COMMENT ON COLUMN orders.status IS 'new, paid, shipped, delivered or cancelled';
COMMENT ON COLUMN order_items.unit_price IS 'Price of one unit at the moment of purchase';
"""

COUNTRIES = [
    ("Russia", "Europe"), ("Germany", "Europe"), ("France", "Europe"), ("Spain", "Europe"),
    ("USA", "North America"), ("Canada", "North America"), ("Japan", "Asia"), ("China", "Asia"),
    ("Brazil", "South America"), ("Kazakhstan", "Asia"),
]
FIRST_NAMES = ["Ivan", "Anna", "John", "Maria", "Hans", "Claire", "Yuki", "Li", "Carlos", "Aigul", "Olga", "Peter"]
LAST_NAMES = ["Ivanov", "Smith", "Müller", "Dubois", "Tanaka", "Wang", "Silva", "Petrova", "Garcia", "Brown"]
# (категория, родитель, [товары с базовой ценой])
CATEGORIES = {
    "Electronics": (None, []),
    "Smartphones": ("Electronics", [("Phone X", 699), ("Phone Mini", 449), ("Phone Pro", 999)]),
    "Laptops": ("Electronics", [("Laptop Air", 1099), ("Laptop Gamer", 1799), ("Laptop Office", 649)]),
    "Home": (None, []),
    "Kitchen": ("Home", [("Blender", 59), ("Coffee Machine", 189), ("Toaster", 35)]),
    "Furniture": ("Home", [("Office Chair", 220), ("Desk", 310), ("Bookshelf", 140)]),
    "Books": (None, [("Python Handbook", 39), ("SQL Cookbook", 45), ("RAG in Practice", 55)]),
}
SUPPLIER_NAMES = ["TechSource", "HomeWorld", "BookHouse", "GlobalParts", "NordSupply"]
REVIEW_COMMENTS = ["Great product", "Okay for the price", "Not what I expected", "Excellent quality", "Would buy again", None]
STATUSES = ["new", "paid", "shipped", "delivered", "delivered", "delivered", "cancelled"]


def random_datetime(start: datetime, end: datetime) -> datetime:
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))


def populate(cursor) -> dict[str, int]:
    execute_values(cursor, "INSERT INTO countries (name, region) VALUES %s", COUNTRIES)

    customers = [
        (
            random.choice(FIRST_NAMES),
            random.choice(LAST_NAMES),
            f"user{i}@example.com",
            random.randint(1, len(COUNTRIES)),
            date(2023, 1, 1) + timedelta(days=random.randint(0, 700)),
        )
        for i in range(1, 101)
    ]
    execute_values(
        cursor,
        "INSERT INTO customers (first_name, last_name, email, country_id, registered_at) VALUES %s",
        customers,
    )

    category_ids: dict[str, int] = {}
    for name, (parent, _) in CATEGORIES.items():
        cursor.execute(
            "INSERT INTO categories (name, parent_id) VALUES (%s, %s) RETURNING id",
            (name, category_ids[parent] if parent else None),
        )
        category_ids[name] = cursor.fetchone()[0]

    execute_values(
        cursor,
        "INSERT INTO suppliers (name, country_id) VALUES %s",
        [(name, random.randint(1, len(COUNTRIES))) for name in SUPPLIER_NAMES],
    )

    products = [
        (name, category_ids[category], random.randint(1, len(SUPPLIER_NAMES)), price, random.randint(0, 200))
        for category, (_, items) in CATEGORIES.items()
        for name, price in items
    ]
    execute_values(
        cursor,
        "INSERT INTO products (name, category_id, supplier_id, price, stock) VALUES %s",
        products,
    )
    product_prices = {i: float(p[3]) for i, p in enumerate(products, start=1)}

    start, end = datetime(2024, 1, 1), datetime(2025, 12, 31)
    order_items_count = payments_count = 0
    orders_count = 400
    for _ in range(orders_count):
        created_at = random_datetime(start, end)
        status = random.choice(STATUSES)
        cursor.execute(
            "INSERT INTO orders (customer_id, status, created_at) VALUES (%s, %s, %s) RETURNING id",
            (random.randint(1, len(customers)), status, created_at),
        )
        order_id = cursor.fetchone()[0]

        items = [
            (order_id, product_id, random.randint(1, 3), product_prices[product_id])
            for product_id in random.sample(list(product_prices), random.randint(1, 4))
        ]
        execute_values(
            cursor, "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES %s", items
        )
        order_items_count += len(items)

        if status != "new":
            total = round(sum(quantity * price for _, _, quantity, price in items), 2)
            cursor.execute(
                "INSERT INTO payments (order_id, method, amount, paid_at) VALUES (%s, %s, %s, %s)",
                (
                    order_id,
                    random.choice(["card", "paypal", "bank_transfer"]),
                    total,
                    created_at + timedelta(minutes=random.randint(1, 600)),
                ),
            )
            payments_count += 1

    reviews = [
        (
            random.randint(1, len(products)),
            random.randint(1, len(customers)),
            random.choice([1, 2, 3, 4, 4, 5, 5, 5]),
            random.choice(REVIEW_COMMENTS),
            random_datetime(start, end),
        )
        for _ in range(300)
    ]
    execute_values(
        cursor,
        "INSERT INTO reviews (product_id, customer_id, rating, comment, created_at) VALUES %s",
        reviews,
    )

    return {
        "countries": len(COUNTRIES),
        "customers": len(customers),
        "categories": len(category_ids),
        "suppliers": len(SUPPLIER_NAMES),
        "products": len(products),
        "orders": orders_count,
        "order_items": order_items_count,
        "payments": payments_count,
        "reviews": len(reviews),
    }


conn = psycopg2.connect(**settings.postgres.connect_kwargs())

try:
    with conn.cursor() as cursor:
        cursor.execute(SCHEMA_SQL)
        counts = populate(cursor)
    conn.commit()
    for table, count in counts.items():
        print(f"{table}: {count}")
finally:
    conn.close()
