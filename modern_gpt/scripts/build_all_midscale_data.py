"""Assembler and validator for midscale 500-example ModernGPT-S dataset."""

import ast
import hashlib
import json
from pathlib import Path
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from generate_midscale_dataset import build_python_tasks, test_code_snippet

DATA_DIR = Path("data/midscale_500")

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def build_json_tasks():
    families = []

    # 1. user_account (keys: user_id, username, email, role)
    f1 = ("user_account", ["user_id", "username", "email", "role"], [
        ("User alice with email alice@example.com having role admin and id 101.", {"user_id": 101, "username": "alice", "email": "alice@example.com", "role": "admin"}),
        ("User bob with email bob@company.org having role editor and id 102.", {"user_id": 102, "username": "bob", "email": "bob@company.org", "role": "editor"}),
        ("User charlie with email charlie@test.net having role viewer and id 103.", {"user_id": 103, "username": "charlie", "email": "charlie@test.net", "role": "viewer"}),
        ("User diana with email diana@domain.com having role developer and id 104.", {"user_id": 104, "username": "diana", "email": "diana@domain.com", "role": "developer"}),
        ("User ethan with email ethan@service.io having role moderator and id 105.", {"user_id": 105, "username": "ethan", "email": "ethan@service.io", "role": "moderator"}),
        ("User fiona with email fiona@mail.com having role analyst and id 106.", {"user_id": 106, "username": "fiona", "email": "fiona@mail.com", "role": "analyst"}),
        ("User george with email george@corp.org having role auditor and id 107.", {"user_id": 107, "username": "george", "email": "george@corp.org", "role": "auditor"}),
        ("User hannah with email hannah@agency.gov having role manager and id 108.", {"user_id": 108, "username": "hannah", "email": "hannah@agency.gov", "role": "manager"}),
    ])
    families.append(f1)

    # 2. server_node (keys: host, port, status, is_primary)
    f2 = ("server_node", ["host", "port", "status", "is_primary"], [
        ("Primary server web-01 on localhost port 8080 with status online.", {"host": "web-01", "port": 8080, "status": "online", "is_primary": True}),
        ("Secondary server web-02 on 10.0.0.2 port 8080 with status standby.", {"host": "web-02", "port": 8080, "status": "standby", "is_primary": False}),
        ("Backup node db-replica on 192.168.1.15 port 5432 with status syncing.", {"host": "db-replica", "port": 5432, "status": "syncing", "is_primary": False}),
        ("Cache server redis-master on 127.0.0.1 port 6379 with status healthy.", {"host": "redis-master", "port": 6379, "status": "healthy", "is_primary": True}),
        ("Gateway node proxy-01 on gateway.internal port 443 with status active.", {"host": "proxy-01", "port": 443, "status": "active", "is_primary": True}),
        ("Worker node worker-alpha on 10.1.0.10 port 9000 with status busy.", {"host": "worker-alpha", "port": 9000, "status": "busy", "is_primary": False}),
        ("Ingress node ingress-lb on lb.prod port 80 with status running.", {"host": "ingress-lb", "port": 80, "status": "running", "is_primary": True}),
        ("Logging host logstash-01 on logging.lan port 5044 with status idle.", {"host": "logstash-01", "port": 5044, "status": "idle", "is_primary": False}),
    ])
    families.append(f2)

    # 3. database_profile (keys: database_name, engine, port, max_connections)
    f3 = ("database_profile", ["database_name", "engine", "port", "max_connections"], [
        ("Database production_db running postgresql on port 5432 with 200 max connections.", {"database_name": "production_db", "engine": "postgresql", "port": 5432, "max_connections": 200}),
        ("Database analytics_warehouse running snowflake on port 443 with 500 max connections.", {"database_name": "analytics_warehouse", "engine": "snowflake", "port": 443, "max_connections": 500}),
        ("Database sessions_store running redis on port 6379 with 1000 max connections.", {"database_name": "sessions_store", "engine": "redis", "port": 6379, "max_connections": 1000}),
        ("Database customer_crm running mysql on port 3306 with 150 max connections.", {"database_name": "customer_crm", "engine": "mysql", "port": 3306, "max_connections": 150}),
        ("Database document_vault running mongodb on port 27017 with 300 max connections.", {"database_name": "document_vault", "engine": "mongodb", "port": 27017, "max_connections": 300}),
        ("Database staging_test running sqlite on port 0 with 1 max connections.", {"database_name": "staging_test", "engine": "sqlite", "port": 0, "max_connections": 1}),
        ("Database timeseries_metrics running clickhouse on port 8123 with 400 max connections.", {"database_name": "timeseries_metrics", "engine": "clickhouse", "port": 8123, "max_connections": 400}),
        ("Database auth_directory running ldap on port 389 with 50 max connections.", {"database_name": "auth_directory", "engine": "ldap", "port": 389, "max_connections": 50}),
    ])
    families.append(f3)

    # 4. product_inventory (keys: sku, name, price, in_stock)
    f4 = ("product_inventory", ["sku", "name", "price", "in_stock"], [
        ("Product SKU-101 named Wireless Mouse costing 24.99 and currently in stock.", {"sku": "SKU-101", "name": "Wireless Mouse", "price": 24.99, "in_stock": True}),
        ("Product SKU-102 named Mechanical Keyboard costing 89.5 and out of stock.", {"sku": "SKU-102", "name": "Mechanical Keyboard", "price": 89.5, "in_stock": False}),
        ("Product SKU-103 named USB-C Hub costing 35.0 and currently in stock.", {"sku": "SKU-103", "name": "USB-C Hub", "price": 35.0, "in_stock": True}),
        ("Product SKU-104 named Monitor Stand costing 45.99 and currently in stock.", {"sku": "SKU-104", "name": "Monitor Stand", "price": 45.99, "in_stock": True}),
        ("Product SKU-105 named HDMI Cable 2m costing 12.0 and out of stock.", {"sku": "SKU-105", "name": "HDMI Cable 2m", "price": 12.0, "in_stock": False}),
        ("Product SKU-106 named Noise Cancelling Headphones costing 199.99 and in stock.", {"sku": "SKU-106", "name": "Noise Cancelling Headphones", "price": 199.99, "in_stock": True}),
        ("Product SKU-107 named Ergonomic Chair costing 299.0 and out of stock.", {"sku": "SKU-107", "name": "Ergonomic Chair", "price": 299.0, "in_stock": False}),
        ("Product SKU-108 named Desk Lamp LED costing 29.95 and currently in stock.", {"sku": "SKU-108", "name": "Desk Lamp LED", "price": 29.95, "in_stock": True}),
    ])
    families.append(f4)

    # 5. weather_metric (keys: city, temp_c, condition, humidity_pct)
    f5 = ("weather_metric", ["city", "temp_c", "condition", "humidity_pct"], [
        ("Weather in Paris with 18.5 C, clear sky condition, and 60 percent humidity.", {"city": "Paris", "temp_c": 18.5, "condition": "clear sky", "humidity_pct": 60}),
        ("Weather in London with 14.0 C, light rain condition, and 85 percent humidity.", {"city": "London", "temp_c": 14.0, "condition": "light rain", "humidity_pct": 85}),
        ("Weather in Tokyo with 22.0 C, sunny condition, and 50 percent humidity.", {"city": "Tokyo", "temp_c": 22.0, "condition": "sunny", "humidity_pct": 50}),
        ("Weather in New York with 16.2 C, partly cloudy condition, and 65 percent humidity.", {"city": "New York", "temp_c": 16.2, "condition": "partly cloudy", "humidity_pct": 65}),
        ("Weather in Berlin with 12.8 C, overcast condition, and 75 percent humidity.", {"city": "Berlin", "temp_c": 12.8, "condition": "overcast", "humidity_pct": 75}),
        ("Weather in Sydney with 25.4 C, breezy condition, and 45 percent humidity.", {"city": "Sydney", "temp_c": 25.4, "condition": "breezy", "humidity_pct": 45}),
        ("Weather in Toronto with 9.0 C, foggy condition, and 90 percent humidity.", {"city": "Toronto", "temp_c": 9.0, "condition": "foggy", "humidity_pct": 90}),
        ("Weather in Cairo with 31.0 C, hot condition, and 30 percent humidity.", {"city": "Cairo", "temp_c": 31.0, "condition": "hot", "humidity_pct": 30}),
    ])
    families.append(f5)

    # 6. book_catalog (keys: isbn, title, author, year)
    f6 = ("book_catalog", ["isbn", "title", "author", "year"], [
        ("Book 978-0451524935 titled 1984 by George Orwell published in 1949.", {"isbn": "978-0451524935", "title": "1984", "author": "George Orwell", "year": 1949}),
        ("Book 978-0141439518 titled Pride and Prejudice by Jane Austen published in 1813.", {"isbn": "978-0141439518", "title": "Pride and Prejudice", "author": "Jane Austen", "year": 1813}),
        ("Book 978-0061120084 titled To Kill a Mockingbird by Harper Lee published in 1960.", {"isbn": "978-0061120084", "title": "To Kill a Mockingbird", "author": "Harper Lee", "year": 1960}),
        ("Book 978-0743273565 titled The Great Gatsby by F. Scott Fitzgerald published in 1925.", {"isbn": "978-0743273565", "title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "year": 1925}),
        ("Book 978-0385474542 titled Things Fall Apart by Chinua Achebe published in 1958.", {"isbn": "978-0385474542", "title": "Things Fall Apart", "author": "Chinua Achebe", "year": 1958}),
        ("Book 978-0345339683 titled The Hobbit by J.R.R. Tolkien published in 1937.", {"isbn": "978-0345339683", "title": "The Hobbit", "author": "J.R.R. Tolkien", "year": 1937}),
        ("Book 978-0142437230 titled Moby Dick by Herman Melville published in 1851.", {"isbn": "978-0142437230", "title": "Moby Dick", "author": "Herman Melville", "year": 1851}),
        ("Book 978-0679720201 titled The Stranger by Albert Camus published in 1942.", {"isbn": "978-0679720201", "title": "The Stranger", "author": "Albert Camus", "year": 1942}),
    ])
    families.append(f6)

    # 7. point_3d (keys: point_id, x, y, z)
    f7 = ("point_3d", ["point_id", "x", "y", "z"], [
        ("Point origin with coordinates x=0, y=0, z=0.", {"point_id": "origin", "x": 0.0, "y": 0.0, "z": 0.0}),
        ("Point p1 with coordinates x=1.5, y=2.0, z=-3.5.", {"point_id": "p1", "x": 1.5, "y": 2.0, "z": -3.5}),
        ("Point vertex_a with coordinates x=10.0, y=5.0, z=8.0.", {"point_id": "vertex_a", "x": 10.0, "y": 5.0, "z": 8.0}),
        ("Point target with coordinates x=-4.2, y=-1.0, z=9.8.", {"point_id": "target", "x": -4.2, "y": -1.0, "z": 9.8}),
        ("Point node_start with coordinates x=3.0, y=4.0, z=5.0.", {"point_id": "node_start", "x": 3.0, "y": 4.0, "z": 5.0}),
        ("Point sensor_loc with coordinates x=12.1, y=0.5, z=-1.2.", {"point_id": "sensor_loc", "x": 12.1, "y": 0.5, "z": -1.2}),
        ("Point waypoint_1 with coordinates x=100.0, y=200.0, z=50.0.", {"point_id": "waypoint_1", "x": 100.0, "y": 200.0, "z": 50.0}),
        ("Point center_mass with coordinates x=-0.5, y=0.5, z=0.0.", {"point_id": "center_mass", "x": -0.5, "y": 0.5, "z": 0.0}),
    ])
    families.append(f7)

    # 8. order_record (keys: order_id, customer, total_usd, items_count)
    f8 = ("order_record", ["order_id", "customer", "total_usd", "items_count"], [
        ("Order ORD-1001 for customer John Doe totaling 154.5 USD with 3 items.", {"order_id": "ORD-1001", "customer": "John Doe", "total_usd": 154.5, "items_count": 3}),
        ("Order ORD-1002 for customer Jane Smith totaling 89.0 USD with 1 items.", {"order_id": "ORD-1002", "customer": "Jane Smith", "total_usd": 89.0, "items_count": 1}),
        ("Order ORD-1003 for customer Acme Corp totaling 1250.0 USD with 15 items.", {"order_id": "ORD-1003", "customer": "Acme Corp", "total_usd": 1250.0, "items_count": 15}),
        ("Order ORD-1004 for customer Mark Spencer totaling 42.75 USD with 2 items.", {"order_id": "ORD-1004", "customer": "Mark Spencer", "total_usd": 42.75, "items_count": 2}),
        ("Order ORD-1005 for customer Sarah Connor totaling 310.2 USD with 5 items.", {"order_id": "ORD-1005", "customer": "Sarah Connor", "total_usd": 310.2, "items_count": 5}),
        ("Order ORD-1006 for customer TechStart LLC totaling 550.0 USD with 8 items.", {"order_id": "ORD-1006", "customer": "TechStart LLC", "total_usd": 550.0, "items_count": 8}),
        ("Order ORD-1007 for customer David Miller totaling 19.99 USD with 1 items.", {"order_id": "ORD-1007", "customer": "David Miller", "total_usd": 19.99, "items_count": 1}),
        ("Order ORD-1008 for customer Global Media totaling 2400.0 USD with 25 items.", {"order_id": "ORD-1008", "customer": "Global Media", "total_usd": 2400.0, "items_count": 25}),
    ])
    families.append(f8)

    # 9. device_config (keys: device_name, volume, brightness, bluetooth_on)
    f9 = ("device_config", ["device_name", "volume", "brightness", "bluetooth_on"], [
        ("Device LivingRoom-Speaker with volume 65, brightness 80, and bluetooth enabled.", {"device_name": "LivingRoom-Speaker", "volume": 65, "brightness": 80, "bluetooth_on": True}),
        ("Device Bedside-Clock with volume 30, brightness 20, and bluetooth disabled.", {"device_name": "Bedside-Clock", "volume": 30, "brightness": 20, "bluetooth_on": False}),
        ("Device Tablet-Pro with volume 50, brightness 75, and bluetooth enabled.", {"device_name": "Tablet-Pro", "volume": 50, "brightness": 75, "bluetooth_on": True}),
        ("Device Kitchen-Display with volume 80, brightness 90, and bluetooth disabled.", {"device_name": "Kitchen-Display", "volume": 80, "brightness": 90, "bluetooth_on": False}),
        ("Device Office-Monitor with volume 40, brightness 60, and bluetooth disabled.", {"device_name": "Office-Monitor", "volume": 40, "brightness": 60, "bluetooth_on": False}),
        ("Device Car-Audio with volume 70, brightness 85, and bluetooth enabled.", {"device_name": "Car-Audio", "volume": 70, "brightness": 85, "bluetooth_on": True}),
        ("Device Studio-Headset with volume 90, brightness 10, and bluetooth enabled.", {"device_name": "Studio-Headset", "volume": 90, "brightness": 10, "bluetooth_on": True}),
        ("Device Smart-Frame with volume 0, brightness 50, and bluetooth disabled.", {"device_name": "Smart-Frame", "volume": 0, "brightness": 50, "bluetooth_on": False}),
    ])
    families.append(f9)

    # 10. network_route (keys: protocol, domain, port, secure)
    f10 = ("network_route", ["protocol", "domain", "port", "secure"], [
        ("Route using https to api.service.com on port 443 with secure enabled.", {"protocol": "https", "domain": "api.service.com", "port": 443, "secure": True}),
        ("Route using http to internal.lan on port 80 with secure disabled.", {"protocol": "http", "domain": "internal.lan", "port": 80, "secure": False}),
        ("Route using ssh to bastion.corp.net on port 22 with secure enabled.", {"protocol": "ssh", "domain": "bastion.corp.net", "port": 22, "secure": True}),
        ("Route using ftp to files.archive.org on port 21 with secure disabled.", {"protocol": "ftp", "domain": "files.archive.org", "port": 21, "secure": False}),
        ("Route using wss to stream.market.io on port 8443 with secure enabled.", {"protocol": "wss", "domain": "stream.market.io", "port": 8443, "secure": True}),
        ("Route using smtp to mail.provider.net on port 587 with secure enabled.", {"protocol": "smtp", "domain": "mail.provider.net", "port": 587, "secure": True}),
        ("Route using dns to ns1.cloudflare.com on port 53 with secure disabled.", {"protocol": "dns", "domain": "ns1.cloudflare.com", "port": 53, "secure": False}),
        ("Route using grpc to auth.cluster.local on port 50051 with secure enabled.", {"protocol": "grpc", "domain": "auth.cluster.local", "port": 50051, "secure": True}),
    ])
    families.append(f10)

    # 11. student_score (keys: student_id, course, score, passed)
    f11 = ("student_score", ["student_id", "course", "score", "passed"], [
        ("Student ST-801 in CS101 scoring 92.5 and passed.", {"student_id": "ST-801", "course": "CS101", "score": 92.5, "passed": True}),
        ("Student ST-802 in MATH201 scoring 58.0 and not passed.", {"student_id": "ST-802", "course": "MATH201", "score": 58.0, "passed": False}),
        ("Student ST-803 in PHYS102 scoring 74.0 and passed.", {"student_id": "ST-803", "course": "PHYS102", "score": 74.0, "passed": True}),
        ("Student ST-804 in CHEM105 scoring 88.0 and passed.", {"student_id": "ST-804", "course": "CHEM105", "score": 88.0, "passed": True}),
        ("Student ST-805 in ENG101 scoring 61.5 and passed.", {"student_id": "ST-805", "course": "ENG101", "score": 61.5, "passed": True}),
        ("Student ST-806 in BIO110 scoring 45.0 and not passed.", {"student_id": "ST-806", "course": "BIO110", "score": 45.0, "passed": False}),
        ("Student ST-807 in HIST210 scoring 83.0 and passed.", {"student_id": "ST-807", "course": "HIST210", "score": 83.0, "passed": True}),
        ("Student ST-808 in ECON101 scoring 52.0 and not passed.", {"student_id": "ST-808", "course": "ECON101", "score": 52.0, "passed": False}),
    ])
    families.append(f11)

    # 12. vehicle_spec (keys: vin, make, model, electric)
    f12 = ("vehicle_spec", ["vin", "make", "model", "electric"], [
        ("Vehicle 1HGCR2F8 named Honda Accord which is not electric.", {"vin": "1HGCR2F8", "make": "Honda", "model": "Accord", "electric": False}),
        ("Vehicle 5YJSA1E2 named Tesla Model S which is electric.", {"vin": "5YJSA1E2", "make": "Tesla", "model": "Model S", "electric": True}),
        ("Vehicle JT2BF22K named Toyota Camry which is not electric.", {"vin": "JT2BF22K", "make": "Toyota", "model": "Camry", "electric": False}),
        ("Vehicle WAURGAFC named Audi e-tron which is electric.", {"vin": "WAURGAFC", "make": "Audi", "model": "e-tron", "electric": True}),
        ("Vehicle 1FA6P8CF named Ford Mustang which is not electric.", {"vin": "1FA6P8CF", "make": "Ford", "model": "Mustang", "electric": False}),
        ("Vehicle WBY8P2C5 named BMW i4 which is electric.", {"vin": "WBY8P2C5", "make": "BMW", "model": "i4", "electric": True}),
        ("Vehicle KMHD34LF named Hyundai Ioniq 5 which is electric.", {"vin": "KMHD34LF", "make": "Hyundai", "model": "Ioniq 5", "electric": True}),
        ("Vehicle 3VW2K7AJ named Volkswagen Jetta which is not electric.", {"vin": "3VW2K7AJ", "make": "Volkswagen", "model": "Jetta", "electric": False}),
    ])
    families.append(f12)

    # 13. color_palette (keys: name, hex_code, rgb, is_dark)
    f13 = ("color_palette", ["name", "hex_code", "rgb", "is_dark"], [
        ("Color Midnight Blue with hex #191970, rgb [25, 25, 112], and dark flag true.", {"name": "Midnight Blue", "hex_code": "#191970", "rgb": [25, 25, 112], "is_dark": True}),
        ("Color Lemon Chiffon with hex #FFFACD, rgb [255, 250, 205], and dark flag false.", {"name": "Lemon Chiffon", "hex_code": "#FFFACD", "rgb": [255, 250, 205], "is_dark": False}),
        ("Color Crimson with hex #DC143C, rgb [220, 20, 60], and dark flag false.", {"name": "Crimson", "hex_code": "#DC143C", "rgb": [220, 20, 60], "is_dark": False}),
        ("Color Charcoal with hex #36454F, rgb [54, 69, 79], and dark flag true.", {"name": "Charcoal", "hex_code": "#36454F", "rgb": [54, 69, 79], "is_dark": True}),
        ("Color Snow White with hex #FFFAFA, rgb [255, 250, 250], and dark flag false.", {"name": "Snow White", "hex_code": "#FFFAFA", "rgb": [255, 250, 250], "is_dark": False}),
        ("Color Forest Green with hex #228B22, rgb [34, 139, 34], and dark flag true.", {"name": "Forest Green", "hex_code": "#228B22", "rgb": [34, 139, 34], "is_dark": True}),
        ("Color Coral with hex #FF7F50, rgb [255, 127, 80], and dark flag false.", {"name": "Coral", "hex_code": "#FF7F50", "rgb": [255, 127, 80], "is_dark": False}),
        ("Color Jet Black with hex #0A0A0A, rgb [10, 10, 10], and dark flag true.", {"name": "Jet Black", "hex_code": "#0A0A0A", "rgb": [10, 10, 10], "is_dark": True}),
    ])
    families.append(f13)

    # 14. task_entry (keys: task_id, title, priority, completed)
    f14 = ("task_entry", ["task_id", "title", "priority", "completed"], [
        ("Task TSK-1 titled Fix login bug with priority high and completed false.", {"task_id": "TSK-1", "title": "Fix login bug", "priority": "high", "completed": False}),
        ("Task TSK-2 titled Write documentation with priority medium and completed true.", {"task_id": "TSK-2", "title": "Write documentation", "priority": "medium", "completed": True}),
        ("Task TSK-3 titled Update dependencies with priority low and completed false.", {"task_id": "TSK-3", "title": "Update dependencies", "priority": "low", "completed": False}),
        ("Task TSK-4 titled Run security audit with priority critical and completed true.", {"task_id": "TSK-4", "title": "Run security audit", "priority": "critical", "completed": True}),
        ("Task TSK-5 titled Design landing page with priority high and completed false.", {"task_id": "TSK-5", "title": "Design landing page", "priority": "high", "completed": False}),
        ("Task TSK-6 titled Refactor database queries with priority medium and completed true.", {"task_id": "TSK-6", "title": "Refactor database queries", "priority": "medium", "completed": True}),
        ("Task TSK-7 titled Add unit tests with priority high and completed true.", {"task_id": "TSK-7", "title": "Add unit tests", "priority": "high", "completed": True}),
        ("Task TSK-8 titled Deploy release v1.2 with priority critical and completed false.", {"task_id": "TSK-8", "title": "Deploy release v1.2", "priority": "critical", "completed": False}),
    ])
    families.append(f14)

    # 15. sensor_telemetry (keys: sensor_id, metric, value, healthy)
    f15 = ("sensor_telemetry", ["sensor_id", "metric", "value", "healthy"], [
        ("Sensor SNS-1 measuring temperature with value 42.5 and healthy true.", {"sensor_id": "SNS-1", "metric": "temperature", "value": 42.5, "healthy": True}),
        ("Sensor SNS-2 measuring pressure with value 108.2 and healthy false.", {"sensor_id": "SNS-2", "metric": "pressure", "value": 108.2, "healthy": False}),
        ("Sensor SNS-3 measuring vibration with value 0.04 and healthy true.", {"sensor_id": "SNS-3", "metric": "vibration", "value": 0.04, "healthy": True}),
        ("Sensor SNS-4 measuring voltage with value 11.9 and healthy true.", {"sensor_id": "SNS-4", "metric": "voltage", "value": 11.9, "healthy": True}),
        ("Sensor SNS-5 measuring humidity with value 98.0 and healthy false.", {"sensor_id": "SNS-5", "metric": "humidity", "value": 98.0, "healthy": False}),
        ("Sensor SNS-6 measuring co2_ppm with value 415.0 and healthy true.", {"sensor_id": "SNS-6", "metric": "co2_ppm", "value": 415.0, "healthy": True}),
        ("Sensor SNS-7 measuring rpm with value 3600.0 and healthy true.", {"sensor_id": "SNS-7", "metric": "rpm", "value": 3600.0, "healthy": True}),
        ("Sensor SNS-8 measuring oil_temp with value 115.4 and healthy false.", {"sensor_id": "SNS-8", "metric": "oil_temp", "value": 115.4, "healthy": False}),
    ])
    families.append(f15)

    # 16. support_ticket (keys: ticket_id, urgency, category, resolved)
    f16 = ("support_ticket", ["ticket_id", "urgency", "category", "resolved"], [
        ("Ticket TCK-401 with urgency high in category billing and resolved false.", {"ticket_id": "TCK-401", "urgency": "high", "category": "billing", "resolved": False}),
        ("Ticket TCK-402 with urgency low in category feature_request and resolved true.", {"ticket_id": "TCK-402", "urgency": "low", "category": "feature_request", "resolved": True}),
        ("Ticket TCK-403 with urgency critical in category outage and resolved true.", {"ticket_id": "TCK-403", "urgency": "critical", "category": "outage", "resolved": True}),
        ("Ticket TCK-404 with urgency medium in category access_rights and resolved false.", {"ticket_id": "TCK-404", "urgency": "medium", "category": "access_rights", "resolved": False}),
        ("Ticket TCK-405 with urgency high in category data_loss and resolved false.", {"ticket_id": "TCK-405", "urgency": "high", "category": "data_loss", "resolved": False}),
        ("Ticket TCK-406 with urgency low in category general_faq and resolved true.", {"ticket_id": "TCK-406", "urgency": "low", "category": "general_faq", "resolved": True}),
        ("Ticket TCK-407 with urgency medium in category integration and resolved true.", {"ticket_id": "TCK-407", "urgency": "medium", "category": "integration", "resolved": True}),
        ("Ticket TCK-408 with urgency critical in category security_alert and resolved false.", {"ticket_id": "TCK-408", "urgency": "critical", "category": "security_alert", "resolved": False}),
    ])
    families.append(f16)

    # 17. api_rate_limit (keys: tier_name, rate_limit, window_sec, active)
    f17 = ("api_rate_limit", ["tier_name", "rate_limit", "window_sec", "active"], [
        ("Tier free with rate limit 60 in window 60 sec and active true.", {"tier_name": "free", "rate_limit": 60, "window_sec": 60, "active": True}),
        ("Tier pro with rate limit 1000 in window 60 sec and active true.", {"tier_name": "pro", "rate_limit": 1000, "window_sec": 60, "active": True}),
        ("Tier enterprise with rate limit 10000 in window 60 sec and active true.", {"tier_name": "enterprise", "rate_limit": 10000, "window_sec": 60, "active": True}),
        ("Tier legacy with rate limit 100 in window 120 sec and active false.", {"tier_name": "legacy", "rate_limit": 100, "window_sec": 120, "active": False}),
        ("Tier public with rate limit 20 in window 60 sec and active true.", {"tier_name": "public", "rate_limit": 20, "window_sec": 60, "active": True}),
        ("Tier partner with rate limit 5000 in window 60 sec and active true.", {"tier_name": "partner", "rate_limit": 5000, "window_sec": 60, "active": True}),
        ("Tier trial with rate limit 200 in window 3600 sec and active false.", {"tier_name": "trial", "rate_limit": 200, "window_sec": 3600, "active": False}),
        ("Tier internal with rate limit 50000 in window 60 sec and active true.", {"tier_name": "internal", "rate_limit": 50000, "window_sec": 60, "active": True}),
    ])
    families.append(f17)

    # 18. file_meta (keys: filename, size_kb, file_type, read_only)
    f18 = ("file_meta", ["filename", "size_kb", "file_type", "read_only"], [
        ("File report.pdf with size 2048 KB, type pdf, and read-only true.", {"filename": "report.pdf", "size_kb": 2048, "file_type": "pdf", "read_only": True}),
        ("File dataset.csv with size 51200 KB, type csv, and read-only false.", {"filename": "dataset.csv", "size_kb": 51200, "file_type": "csv", "read_only": False}),
        ("File main.py with size 16 KB, type python, and read-only false.", {"filename": "main.py", "size_kb": 16, "file_type": "python", "read_only": False}),
        ("File logo.png with size 512 KB, type image, and read-only true.", {"filename": "logo.png", "size_kb": 512, "file_type": "image", "read_only": True}),
        ("File config.yaml with size 8 KB, type yaml, and read-only false.", {"filename": "config.yaml", "size_kb": 8, "file_type": "yaml", "read_only": False}),
        ("File backup.tar.gz with size 1048576 KB, type archive, and read-only true.", {"filename": "backup.tar.gz", "size_kb": 1048576, "file_type": "archive", "read_only": True}),
        ("File styles.css with size 64 KB, type css, and read-only false.", {"filename": "styles.css", "size_kb": 64, "file_type": "css", "read_only": False}),
        ("File license.txt with size 4 KB, type text, and read-only true.", {"filename": "license.txt", "size_kb": 4, "file_type": "text", "read_only": True}),
    ])
    families.append(f18)

    # 19. payment_log (keys: payment_id, amount, currency, confirmed)
    f19 = ("payment_log", ["payment_id", "amount", "currency", "confirmed"], [
        ("Payment PAY-701 with amount 49.99 USD and confirmed true.", {"payment_id": "PAY-701", "amount": 49.99, "currency": "USD", "confirmed": True}),
        ("Payment PAY-702 with amount 120.0 EUR and confirmed true.", {"payment_id": "PAY-702", "amount": 120.0, "currency": "EUR", "confirmed": True}),
        ("Payment PAY-703 with amount 3500 JPY and confirmed false.", {"payment_id": "PAY-703", "amount": 3500.0, "currency": "JPY", "confirmed": False}),
        ("Payment PAY-704 with amount 15.5 GBP and confirmed true.", {"payment_id": "PAY-704", "amount": 15.5, "currency": "GBP", "confirmed": True}),
        ("Payment PAY-705 with amount 250.0 CAD and confirmed true.", {"payment_id": "PAY-705", "amount": 250.0, "currency": "CAD", "confirmed": True}),
        ("Payment PAY-706 with amount 89.0 AUD and confirmed false.", {"payment_id": "PAY-706", "amount": 89.0, "currency": "AUD", "confirmed": False}),
        ("Payment PAY-707 with amount 500.0 CHF and confirmed true.", {"payment_id": "PAY-707", "amount": 500.0, "currency": "CHF", "confirmed": True}),
        ("Payment PAY-708 with amount 10.0 USD and confirmed false.", {"payment_id": "PAY-708", "amount": 10.0, "currency": "USD", "confirmed": False}),
    ])
    families.append(f19)

    # 20. flight_card (keys: flight_code, origin, destination, delayed)
    f20 = ("flight_card", ["flight_code", "origin", "destination", "delayed"], [
        ("Flight AF102 from CDG to JFK and not delayed.", {"flight_code": "AF102", "origin": "CDG", "destination": "JFK", "delayed": False}),
        ("Flight BA249 from LHR to HND and delayed true.", {"flight_code": "BA249", "origin": "LHR", "destination": "HND", "delayed": True}),
        ("Flight LH400 from FRA to SFO and not delayed.", {"flight_code": "LH400", "origin": "FRA", "destination": "SFO", "delayed": False}),
        ("Flight UA857 from SFO to PVG and delayed true.", {"flight_code": "UA857", "origin": "SFO", "destination": "PVG", "delayed": True}),
        ("Flight DL120 from ATL to AMS and not delayed.", {"flight_code": "DL120", "origin": "ATL", "destination": "AMS", "delayed": False}),
        ("Flight QF11 from SYD to LAX and not delayed.", {"flight_code": "QF11", "origin": "SYD", "destination": "LAX", "delayed": False}),
        ("Flight SQ26 from SIN to FRA and delayed true.", {"flight_code": "SQ26", "origin": "SIN", "destination": "FRA", "delayed": True}),
        ("Flight EK201 from DXB to JFK and not delayed.", {"flight_code": "EK201", "origin": "DXB", "destination": "JFK", "delayed": False}),
    ])
    families.append(f20)

    # 21. movie_entry (VALIDATION) (keys: title, director, runtime_mins, in_theaters)
    f21 = ("movie_entry", ["title", "director", "runtime_mins", "in_theaters"], [
        ("Movie Inception directed by Christopher Nolan, runtime 148 mins, currently not in theaters.", {"title": "Inception", "director": "Christopher Nolan", "runtime_mins": 148, "in_theaters": False}),
        ("Movie Parasite directed by Bong Joon-ho, runtime 132 mins, currently not in theaters.", {"title": "Parasite", "director": "Bong Joon-ho", "runtime_mins": 132, "in_theaters": False}),
        ("Movie Dune 2 directed by Denis Villeneuve, runtime 166 mins, currently in theaters.", {"title": "Dune 2", "director": "Denis Villeneuve", "runtime_mins": 166, "in_theaters": True}),
        ("Movie Pulp Fiction directed by Quentin Tarantino, runtime 154 mins, not in theaters.", {"title": "Pulp Fiction", "director": "Quentin Tarantino", "runtime_mins": 154, "in_theaters": False}),
        ("Movie Spirited Away directed by Hayao Miyazaki, runtime 125 mins, not in theaters.", {"title": "Spirited Away", "director": "Hayao Miyazaki", "runtime_mins": 125, "in_theaters": False}),
        ("Movie Oppenheimer directed by Christopher Nolan, runtime 180 mins, not in theaters.", {"title": "Oppenheimer", "director": "Christopher Nolan", "runtime_mins": 180, "in_theaters": False}),
        ("Movie Gladiator 2 directed by Ridley Scott, runtime 150 mins, currently in theaters.", {"title": "Gladiator 2", "director": "Ridley Scott", "runtime_mins": 150, "in_theaters": True}),
        ("Movie The Matrix directed by Lana Wachowski, runtime 136 mins, not in theaters.", {"title": "The Matrix", "director": "Lana Wachowski", "runtime_mins": 136, "in_theaters": False}),
    ])
    families.append(f21)

    # 22. recipe_card (VALIDATION) (keys: recipe_name, servings, prep_mins, vegetarian)
    f22 = ("recipe_card", ["recipe_name", "servings", "prep_mins", "vegetarian"], [
        ("Recipe Tomato Pasta for 4 servings, prep time 25 mins, vegetarian true.", {"recipe_name": "Tomato Pasta", "servings": 4, "prep_mins": 25, "vegetarian": True}),
        ("Recipe Beef Stew for 6 servings, prep time 90 mins, vegetarian false.", {"recipe_name": "Beef Stew", "servings": 6, "prep_mins": 90, "vegetarian": False}),
        ("Recipe Greek Salad for 2 servings, prep time 15 mins, vegetarian true.", {"recipe_name": "Greek Salad", "servings": 2, "prep_mins": 15, "vegetarian": True}),
        ("Recipe Roast Chicken for 4 servings, prep time 60 mins, vegetarian false.", {"recipe_name": "Roast Chicken", "servings": 4, "prep_mins": 60, "vegetarian": False}),
        ("Recipe Lentil Soup for 5 servings, prep time 40 mins, vegetarian true.", {"recipe_name": "Lentil Soup", "servings": 5, "prep_mins": 40, "vegetarian": True}),
        ("Recipe Salmon Fillet for 2 servings, prep time 20 mins, vegetarian false.", {"recipe_name": "Salmon Fillet", "servings": 2, "prep_mins": 20, "vegetarian": False}),
        ("Recipe Vegetable Stir-Fry for 3 servings, prep time 18 mins, vegetarian true.", {"recipe_name": "Vegetable Stir-Fry", "servings": 3, "prep_mins": 18, "vegetarian": True}),
        ("Recipe Club Sandwich for 1 servings, prep time 10 mins, vegetarian false.", {"recipe_name": "Club Sandwich", "servings": 1, "prep_mins": 10, "vegetarian": False}),
    ])
    families.append(f22)

    # 23. auth_claims (VALIDATION) (keys: subject_id, issuer, expires_epoch, is_admin)
    f23 = ("auth_claims", ["subject_id", "issuer", "expires_epoch", "is_admin"], [
        ("Auth claim sub usr_1001, issuer auth.corp.com, expires 1700000000, admin true.", {"subject_id": "usr_1001", "issuer": "auth.corp.com", "expires_epoch": 1700000000, "is_admin": True}),
        ("Auth claim sub usr_1002, issuer accounts.google.com, expires 1700003600, admin false.", {"subject_id": "usr_1002", "issuer": "accounts.google.com", "expires_epoch": 1700003600, "is_admin": False}),
        ("Auth claim sub srv_token, issuer vault.cluster.local, expires 1700086400, admin true.", {"subject_id": "srv_token", "issuer": "vault.cluster.local", "expires_epoch": 1700086400, "is_admin": True}),
        ("Auth claim sub guest_55, issuer demo.service.io, expires 1700001800, admin false.", {"subject_id": "guest_55", "issuer": "demo.service.io", "expires_epoch": 1700001800, "is_admin": False}),
        ("Auth claim sub mod_99, issuer community.net, expires 1700007200, admin false.", {"subject_id": "mod_99", "issuer": "community.net", "expires_epoch": 1700007200, "is_admin": False}),
        ("Auth claim sub sys_root, issuer k8s.cluster, expires 1700043200, admin true.", {"subject_id": "sys_root", "issuer": "k8s.cluster", "expires_epoch": 1700043200, "is_admin": True}),
        ("Auth claim sub dev_33, issuer gitlab.internal, expires 1700021600, admin false.", {"subject_id": "dev_33", "issuer": "gitlab.internal", "expires_epoch": 1700021600, "is_admin": False}),
        ("Auth claim sub audit_01, issuer security.corp, expires 1700014400, admin true.", {"subject_id": "audit_01", "issuer": "security.corp", "expires_epoch": 1700014400, "is_admin": True}),
    ])
    families.append(f23)

    # 24. hardware_stats (VALIDATION) (keys: cpu_pct, memory_used_mb, fan_rpm, throttling)
    f24 = ("hardware_stats", ["cpu_pct", "memory_used_mb", "fan_rpm", "throttling"], [
        ("Hardware load cpu 45.2 pct, memory 8192 MB, fan 2100 rpm, throttling false.", {"cpu_pct": 45.2, "memory_used_mb": 8192, "fan_rpm": 2100, "throttling": False}),
        ("Hardware load cpu 98.5 pct, memory 15400 MB, fan 4500 rpm, throttling true.", {"cpu_pct": 98.5, "memory_used_mb": 15400, "fan_rpm": 4500, "throttling": True}),
        ("Hardware load cpu 12.0 pct, memory 4096 MB, fan 1200 rpm, throttling false.", {"cpu_pct": 12.0, "memory_used_mb": 4096, "fan_rpm": 1200, "throttling": False}),
        ("Hardware load cpu 76.8 pct, memory 12000 MB, fan 3200 rpm, throttling false.", {"cpu_pct": 76.8, "memory_used_mb": 12000, "fan_rpm": 3200, "throttling": False}),
        ("Hardware load cpu 99.1 pct, memory 16000 MB, fan 4800 rpm, throttling true.", {"cpu_pct": 99.1, "memory_used_mb": 16000, "fan_rpm": 4800, "throttling": True}),
        ("Hardware load cpu 5.5 pct, memory 2048 MB, fan 800 rpm, throttling false.", {"cpu_pct": 5.5, "memory_used_mb": 2048, "fan_rpm": 800, "throttling": False}),
        ("Hardware load cpu 64.0 pct, memory 9500 MB, fan 2600 rpm, throttling false.", {"cpu_pct": 64.0, "memory_used_mb": 9500, "fan_rpm": 2600, "throttling": False}),
        ("Hardware load cpu 88.0 pct, memory 14200 MB, fan 3900 rpm, throttling false.", {"cpu_pct": 88.0, "memory_used_mb": 14200, "fan_rpm": 3900, "throttling": False}),
    ])
    families.append(f24)

    # 25. git_commit (VALIDATION) (keys: commit_hash, author_name, files_modified, is_merge)
    f25 = ("git_commit", ["commit_hash", "author_name", "files_modified", "is_merge"], [
        ("Commit 7a3f81c by John Developer with 3 files modified and merge false.", {"commit_hash": "7a3f81c", "author_name": "John Developer", "files_modified": 3, "is_merge": False}),
        ("Commit e91b24d by Release Bot with 12 files modified and merge true.", {"commit_hash": "e91b24d", "author_name": "Release Bot", "files_modified": 12, "is_merge": True}),
        ("Commit 3c44a02 by Sarah Engineer with 1 files modified and merge false.", {"commit_hash": "3c44a02", "author_name": "Sarah Engineer", "files_modified": 1, "is_merge": False}),
        ("Commit 88ab109 by David Tester with 5 files modified and merge false.", {"commit_hash": "88ab109", "author_name": "David Tester", "files_modified": 5, "is_merge": False}),
        ("Commit fa012cd by Merge Bot with 24 files modified and merge true.", {"commit_hash": "fa012cd", "author_name": "Merge Bot", "files_modified": 24, "is_merge": True}),
        ("Commit 55be77a by Alex Maintainer with 2 files modified and merge false.", {"commit_hash": "55be77a", "author_name": "Alex Maintainer", "files_modified": 2, "is_merge": False}),
        ("Commit 19de33f by Security Lead with 4 files modified and merge false.", {"commit_hash": "19de33f", "author_name": "Security Lead", "files_modified": 4, "is_merge": False}),
        ("Commit 99cc01a by Branch Manager with 8 files modified and merge true.", {"commit_hash": "99cc01a", "author_name": "Branch Manager", "files_modified": 8, "is_merge": True}),
    ])
    families.append(f25)

    return families


def build_qa_and_explanation_tasks():
    # 10 QA domains, 5 questions each = 50 QA tasks
    # Domains 1-8 (40 items) -> Train, Domains 9-10 (10 items) -> Validation
    qa_domains = [
        # Domain 1: Astronomy
        ("astronomy", [
            ("What is the closest star to Earth?", "The closest star to Earth is the Sun."),
            ("What is the largest planet in our solar system?", "The largest planet in our solar system is Jupiter."),
            ("What galaxy contains our solar system?", "The Milky Way galaxy contains our solar system."),
            ("How many moons does Mars have?", "Mars has 2 moons, Phobos and Deimos."),
            ("What planet is known as the Morning Star?", "Venus is known as the Morning Star."),
        ]),
        # Domain 2: Geography
        ("geography", [
            ("What is the capital city of France?", "The capital city of France is Paris."),
            ("What is the longest river in the world?", "The Nile River is traditionally considered the longest river in the world."),
            ("What is the largest hot desert on Earth?", "The Sahara is the largest hot desert on Earth."),
            ("What continent is Egypt located on?", "Egypt is located primarily on the continent of Africa."),
            ("What mountain is the highest peak above sea level?", "Mount Everest is the highest peak above sea level."),
        ]),
        # Domain 3: Chemistry
        ("chemistry", [
            ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
            ("What is the chemical formula for water?", "The chemical formula for water is H2O."),
            ("What element has atomic number 1?", "Hydrogen has atomic number 1."),
            ("What gas makes up approximately 78 percent of Earth's atmosphere?", "Nitrogen makes up approximately 78 percent of Earth's atmosphere."),
            ("What is the common name for sodium chloride?", "The common name for sodium chloride is table salt."),
        ]),
        # Domain 4: Physics
        ("physics", [
            ("What is the boiling point of water in Celsius at standard pressure?", "The boiling point of water at standard pressure is 100 degrees Celsius."),
            ("What is the freezing point of water in Celsius?", "The freezing point of water is 0 degrees Celsius."),
            ("What subatomic particle carries a negative charge?", "The electron carries a negative charge."),
            ("What unit is used to measure electrical resistance?", "The ohm is the unit used to measure electrical resistance."),
            ("What force keeps planets in orbit around the Sun?", "Gravity keeps planets in orbit around the Sun."),
        ]),
        # Domain 5: Biology
        ("biology", [
            ("What organelle is known as the powerhouse of the cell?", "The mitochondrion is known as the powerhouse of the cell."),
            ("What gas do green plants absorb during photosynthesis?", "Plants absorb carbon dioxide during photosynthesis."),
            ("What molecule stores genetic instructions in living organisms?", "DNA stores genetic instructions in living organisms."),
            ("What blood cells help fight infection in the human body?", "White blood cells help fight infection in the human body."),
            ("What is the largest organ of the human body?", "The skin is the largest organ of the human body."),
        ]),
        # Domain 6: History
        ("history", [
            ("In what year did the Apollo 11 moon landing occur?", "The Apollo 11 moon landing occurred in 1969."),
            ("Who was the first President of the United States?", "George Washington was the first President of the United States."),
            ("In what year did World War II end?", "World War II ended in 1945."),
            ("What ancient civilization built the Colosseum in Rome?", "The Roman civilization built the Colosseum."),
            ("Who invented the modern printing press in Europe?", "Johannes Gutenberg invented the European movable type printing press."),
        ]),
        # Domain 7: Mathematics
        ("mathematics", [
            ("What is the only even prime number?", "The only even prime number is 2."),
            ("What is 15 multiplied by 8?", "15 multiplied by 8 is 120."),
            ("How many degrees are in a right angle?", "There are 90 degrees in a right angle."),
            ("What is the square root of 81?", "The square root of 81 is 9."),
            ("What is the mathematical constant representing circumference over diameter?", "Pi (approximately 3.14159) represents circumference over diameter."),
        ]),
        # Domain 8: Earth Science
        ("earth_science", [
            ("How many continents are there on Earth?", "There are 7 continents on Earth."),
            ("What ocean is the largest and deepest on Earth?", "The Pacific Ocean is the largest and deepest ocean on Earth."),
            ("What layer lies directly beneath Earth's crust?", "The mantle lies directly beneath Earth's crust."),
            ("What scale is commonly used to measure earthquake magnitude?", "The Richter scale and moment magnitude scale measure earthquake magnitude."),
            ("What type of rock is formed from cooled magma or lava?", "Igneous rock is formed from cooled magma or lava."),
        ]),
        # Domain 9: Literature & Arts (VALIDATION)
        ("literature_arts", [
            ("Who wrote the play Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
            ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
            ("What author wrote the novel 1984?", "George Orwell wrote the novel 1984."),
            ("What musical instrument has 88 keys?", "The standard piano has 88 keys."),
            ("Who sculpted the statue of David in Florence?", "Michelangelo sculpted the statue of David."),
        ]),
        # Domain 10: Computer Architecture (VALIDATION)
        ("computer_architecture", [
            ("What does CPU stand for in computer systems?", "CPU stands for Central Processing Unit."),
            ("How many bits are in one byte?", "There are 8 bits in one byte."),
            ("What computer component provides volatile primary working memory?", "RAM (Random Access Memory) provides volatile primary working memory."),
            ("What base number system uses digits 0 and 1?", "The binary number system uses digits 0 and 1."),
            ("What protocol standardizes web addresses starting with https?", "HTTPS (Hypertext Transfer Protocol Secure) standardizes secure web addresses."),
        ]),
    ]

    # 10 Explanation domains, 5 questions each = 50 Explanation tasks
    # Domains 1-8 (40 items) -> Train, Domains 9-10 (10 items) -> Validation
    exp_domains = [
        # Domain 1: Algorithms
        ("algorithms", [
            ("Explain how binary search works step by step.", "Binary search repeatedly divides a sorted search interval in half. It compares the target value to the middle element, discarding the left half if the target is larger or the right half if smaller. This achieves logarithmic O(log n) time complexity."),
            ("Describe the concept of recursion in programming.", "Recursion occurs when a function calls itself to solve smaller subproblems of the same type. Every valid recursive function requires a base case to terminate execution and a recursive step that moves toward the base case. Without a base case, recursion results in a stack overflow."),
            ("Explain the difference between a stack and a queue data structure.", "A stack follows the Last-In, First-Out (LIFO) principle, where elements are added and removed from the same end. A queue follows the First-In, First-Out (FIFO) principle, where elements enter at the rear and exit from the front. Both provide constant-time insertion and deletion."),
            ("What is memoization and why is it useful in dynamic programming?", "Memoization is an optimization technique that caches the return values of expensive function calls based on their input parameters. When the function is called again with identical inputs, the cached result is returned directly. This avoids redundant computations and drastically reduces time complexity."),
            ("Explain the purpose of a hash function in computer science.", "A hash function maps arbitrary-sized input data to a fixed-size numeric or byte output. It enables fast average O(1) key-value lookup in hash tables and verifies data integrity in cryptographic checksums. A good hash function minimizes collisions and distributes values uniformly."),
        ]),
        # Domain 2: Operating Systems
        ("operating_systems", [
            ("What is the primary difference between a process and a thread?", "A process is an independent executing program with its own isolated memory address space. A thread is a lightweight unit of execution within a process that shares memory and resources with other threads of the same process. Consequently, threads have lower context-switching overhead."),
            ("Explain what virtual memory is and why operating systems use it.", "Virtual memory abstracts physical RAM by giving each process the illusion of a continuous, private address space. It maps virtual page addresses to physical frames and swaps inactive pages to disk storage when RAM is exhausted. This enables multitasking and protects processes from interfering with each other."),
            ("Describe what a deadlock is in concurrent computing.", "A deadlock is a situation where two or more processes are permanently blocked because each holds a resource that another process needs. Deadlock requires four simultaneous conditions: mutual exclusion, hold and wait, no preemption, and circular wait. Operating systems prevent deadlocks using ordering rules or detection algorithms."),
            ("Explain the function of a device driver in an operating system.", "A device driver is specialized software that acts as a translator between the operating system kernel and hardware peripherals. It presents a standardized interface to the OS while handling hardware-specific registers, interrupts, and communication protocols. This allows software to interact with devices without knowing hardware details."),
            ("What is a race condition in multi-threaded software?", "A race condition occurs when the correctness of a program depends on the relative timing or order of execution of concurrent threads. If multiple threads access and mutate shared state without synchronization, the final output becomes non-deterministic and can corrupt data. Synchronization primitives like mutexes prevent race conditions."),
        ]),
        # Domain 3: Machine Learning
        ("machine_learning", [
            ("Explain the difference between supervised and unsupervised learning.", "Supervised learning trains models on labeled input-output pairs to predict target values on unseen data. Unsupervised learning analyzes unlabeled data to discover underlying structures, clusters, or patterns without explicit targets. Common examples are regression for supervised and k-means clustering for unsupervised learning."),
            ("What is overfitting in machine learning and how can it be mitigated?", "Overfitting occurs when a model learns noise and idiosyncratic details of the training set rather than the underlying general distribution. It results in very low training loss but poor test performance. Mitigations include regularization (L1/L2), dropout, early stopping, and gathering more training data."),
            ("Describe the role of an activation function in neural networks.", "Activation functions introduce non-linearity into neural network layers, enabling the network to approximate complex non-linear functions. Without non-linear activation functions, a multi-layer network collapses mathematically into a single linear transformation regardless of depth. Common activations include ReLU, GELU, and Sigmoid."),
            ("What is cross-entropy loss and how is it used in classification?", "Cross-entropy loss measures the performance of a classification model whose output is a probability value between 0 and 1. It calculates the negative logarithm of the predicted probability assigned to the true class label. Minimizing cross-entropy loss penalizes confident incorrect predictions heavily."),
            ("Explain the purpose of learning rate warmup in deep learning training.", "Learning rate warmup gradually increases the learning rate from near-zero to its peak value over the initial training steps. This stabilizes early parameter updates when network gradients are large and noisy, preventing early divergence. Once warmup finishes, the learning rate typically decays following a schedule."),
        ]),
        # Domain 4: Software Architecture
        ("software_architecture", [
            ("Explain the single responsibility principle in software design.", "The Single Responsibility Principle states that a module, class, or function should have only one reason to change. By concentrating a single coherent responsibility within each unit, code becomes easier to understand, test, and maintain. It reduces cascading bugs when modifications occur."),
            ("What is an API gateway and what problems does it solve?", "An API gateway is a centralized reverse proxy that routes incoming client requests to appropriate backend microservices. It handles common cross-cutting concerns such as authentication, rate limiting, logging, and SSL termination. This prevents individual microservices from having to reimplement shared infrastructure logic."),
            ("Describe the difference between synchronous and asynchronous communication.", "Synchronous communication blocks the caller until the receiving service processes the request and returns a response. Asynchronous communication allows the caller to proceed immediately without waiting, receiving results later via callbacks, polling, or message queues. Asynchronous systems decouple producers from consumers."),
            ("What is the purpose of database indexing?", "A database index is a data structure (such as a B-tree) that improves the speed of data retrieval operations on a table at the cost of additional storage and slower writes. Without an index, the database engine must scan every row in the table sequentially to find matching records."),
            ("Explain the concept of idempotent operations in REST APIs.", "An operation is idempotent if executing it multiple times produces the exact same server state as executing it a single time. In HTTP specifications, GET, PUT, and DELETE methods are designed to be idempotent, while POST is typically non-idempotent. Idempotency guarantees safe retries over unreliable networks."),
        ]),
        # Domain 5: Biology & Ecology
        ("biology_ecology", [
            ("Explain the biological process of photosynthesis in green plants.", "Photosynthesis is the process by which green plants convert light energy, carbon dioxide, and water into glucose and oxygen. Light-dependent reactions in chloroplast thylakoids capture solar energy, which the Calvin cycle uses to synthesize sugar. This provides the primary energy source for most terrestrial food webs."),
            ("Describe how cellular respiration produces ATP in aerobic organisms.", "Cellular respiration breaks down glucose in the presence of oxygen through glycolysis, the citric acid cycle, and oxidative phosphorylation. Electrons transferred across the mitochondrial inner membrane generate an electrochemical gradient that drives ATP synthase. This produces approximately 30 to 32 ATP molecules per glucose molecule."),
            ("What is natural selection and how does it drive evolution?", "Natural selection is the differential survival and reproduction of individuals due to differences in heritable phenotype. Organisms possessing traits better suited to their environmental pressures are more likely to survive and pass those traits to offspring. Over generations, beneficial adaptations increase in frequency."),
            ("Explain the role of enzymes as biological catalysts.", "Enzymes are specialized proteins that accelerate biochemical reactions by lowering their activation energy without being consumed in the process. Each enzyme has a specific active site that binds corresponding substrate molecules to catalyze reactions. Enzyme activity is regulated by temperature, pH, and inhibitors."),
            ("Describe the ecological concept of a trophic cascade.", "A trophic cascade occurs when predators in a food web suppress the abundance or alter the behavior of their prey, releasing the next lower trophic level from predation. For example, reintroducing wolves can reduce deer browsing, allowing riverside vegetation and beaver populations to recover. It demonstrates ecosystem interdependence."),
        ]),
        # Domain 6: Physical Sciences
        ("physical_sciences", [
            ("Explain Newton's third law of motion with a practical example.", "Newton's third law states that whenever one object exerts a force on a second object, the second object exerts an equal and opposite force on the first. For example, when a rocket expels exhaust gases downward at high velocity, the reaction force pushes the rocket upward into space."),
            ("Describe the greenhouse effect and its influence on planetary temperature.", "The greenhouse effect occurs when atmospheric gases like water vapor, carbon dioxide, and methane absorb outgoing infrared radiation emitted by a planet's surface. Instead of escaping directly into space, this thermal radiation is re-radiated in all directions, warming the lower atmosphere and maintaining habitable temperatures."),
            ("Explain why ice floats on liquid water instead of sinking.", "Ice floats because water expands upon freezing, forming a crystalline lattice held by hydrogen bonds that is less dense than liquid water. Liquid water achieves maximum density around 4 degrees Celsius. This property insulates underlying aquatic ecosystems in freezing weather."),
            ("Describe what conservation of energy means in closed physical systems.", "The law of conservation of energy states that the total energy of an isolated system remains constant over time. Energy cannot be created or destroyed; it can only transform from one form to another, such as potential energy converting into kinetic energy. All energy accounting must balance."),
            ("Explain why the sky appears blue during daylight hours on Earth.", "The sky appears blue because Earth's atmosphere scatters sunlight through Rayleigh scattering. Gas molecules in the atmosphere scatter shorter wavelengths of light (blue and violet) much more efficiently than longer wavelengths (red and yellow). Our eyes perceive this scattered light as a blue sky."),
        ]),
        # Domain 7: Chemistry & Materials
        ("chemistry_materials", [
            ("Explain the difference between ionic and covalent chemical bonding.", "Ionic bonding involves the complete transfer of valence electrons from one atom to another, producing oppositely charged ions that attract electrostatically. Covalent bonding involves the sharing of electron pairs between nonmetal atoms to achieve stable electron configurations. Ionic compounds typically form crystalline lattices with high melting points."),
            ("What is the pH scale and what does it measure chemically?", "The pH scale is a logarithmic measure of the hydrogen ion concentration in an aqueous solution. Values below 7 indicate acidic solutions with high hydrogen ion concentrations, while values above 7 indicate alkaline solutions. A neutral solution like pure water has a pH of 7 at 25 degrees Celsius."),
            ("Describe how catalysts increase chemical reaction rates.", "Catalysts increase reaction rates by providing an alternative reaction pathway with a lower activation energy barrier. Because more reactant molecules possess sufficient energy to overcome this lower barrier, the forward and reverse reactions proceed faster. The catalyst is chemically regenerated at the conclusion of the reaction."),
            ("What is an isotope and how do isotopes of an element differ?", "Isotopes are variants of a particular chemical element that share the same number of protons and electrons but differ in their number of neutrons. Consequently, isotopes have identical chemical properties and atomic numbers but different atomic masses. Some isotopes are stable, while radioisotopes undergo decay."),
            ("Explain what an alloy is and why alloys are widely used in engineering.", "An alloy is a metallic substance composed of two or more chemical elements, where at least one is a metal. By combining different elements, alloys disrupt pure crystal lattices to achieve superior strength, corrosion resistance, and thermal tolerance compared to pure metals. Steel (iron and carbon) and bronze are prominent examples."),
        ]),
        # Domain 8: Mathematics & Geometry
        ("math_geometry", [
            ("Explain the Pythagorean theorem and its geometric significance.", "The Pythagorean theorem states that in any right-angled triangle, the area of the square on the hypotenuse is equal to the sum of the areas of the squares on the other two sides (a^2 + b^2 = c^2). It forms the foundation of Euclidean metric distance calculations across geometry, physics, and computer graphics."),
            ("What is a prime number and why are primes fundamental in number theory?", "A prime number is an integer strictly greater than 1 that has no positive divisors other than 1 and itself. Under the Fundamental Theorem of Arithmetic, every integer greater than 1 factors uniquely into a product of primes. Because primes act as multiplicative building blocks, they underpin modern cryptographic algorithms."),
            ("Describe what a derivative represents in calculus.", "A derivative measures the instantaneous rate of change of a mathematical function with respect to one of its independent variables. Geometrically, it corresponds to the slope of the tangent line to the function's curve at a given point. Derivatives are used to calculate velocities, accelerations, and optimize functions."),
            ("Explain the difference between permutations and combinations.", "Permutations count the number of ways to arrange a subset of items where the order of selection matters. Combinations count the number of selections where the order does not matter. For example, selecting a 3-digit passcode is a permutation, while choosing 3 fruits for a salad is a combination."),
            ("What is the central limit theorem in probability and statistics?", "The Central Limit Theorem establishes that the distribution of sample means approximates a normal distribution as sample size grows, regardless of the underlying population's shape. This requires independent, identically distributed variables with finite variance. It justifies using normal probability models across scientific disciplines."),
        ]),
        # Domain 9: Cryptography & Security (VALIDATION)
        ("cryptography_security", [
            ("Explain the difference between symmetric and asymmetric encryption.", "Symmetric encryption uses a single shared secret key for both encrypting and decrypting data, offering high throughput for bulk data. Asymmetric encryption uses a mathematically linked key pair: a public key for encryption and a private key for decryption. Asymmetric cryptography eliminates the need to securely pre-share secret keys."),
            ("What is a cryptographic digital signature and how does it verify integrity?", "A digital signature is created by hashing a message and encrypting that hash with the sender's private key. Anyone with the sender's public key can decrypt the signature and compare it against their own hash of the message. This provides non-repudiation, authentication, and tamper detection."),
            ("Describe what a man-in-the-middle attack is and how TLS prevents it.", "A man-in-the-middle attack occurs when an unauthorized actor intercepts and potentially alters communication between two parties who believe they are talking directly. Transport Layer Security (TLS) prevents this attack using public key certificates signed by trusted certificate authorities to authenticate servers and establish encrypted channels."),
            ("Explain how public-key infrastructure (PKI) establishes trust on the internet.", "Public-Key Infrastructure is a system of digital certificates, certificate authorities (CAs), and cryptographic keys that binds public keys to verified organizational identities. Browsers trust root CAs, which issue intermediate certificates to validate website identities. This enables secure encrypted connections across untrusted networks."),
            ("What is the principle of least privilege in computer security?", "The Principle of Least Privilege dictates that any user, program, or process should possess only the minimum set of permissions necessary to perform its intended function. Limiting authorization reduces the potential blast radius if credentials are leaked or a component is compromised. It forms a core pillar of defense in depth."),
        ]),
        # Domain 10: Compilers & Language Design (VALIDATION)
        ("compilers_languages", [
            ("Explain the difference between a compiled language and an interpreted language.", "A compiled language translates source code directly into native machine instructions ahead of execution time, yielding fast runtime performance. An interpreted language executes instructions directly through an interpreter program or virtual machine at runtime without pre-generating native code. Many modern languages use hybrid bytecode compilation."),
            ("What is an Abstract Syntax Tree (AST) in compiler design?", "An Abstract Syntax Tree is a hierarchical tree representation of source code structure produced during the parsing phase. Each node in the tree corresponds to an operator, expression, or statement without syntax trivia like semicolons or parentheses. Compilers analyze and transform ASTs to generate target machine code."),
            ("Describe the role of a lexer or tokenizer in language processing.", "A lexer reads raw source code characters and groups them into meaningful atomic units called tokens, such as keywords, identifiers, and literals. It strips away comments and whitespace, converting unstructured text into a sequence of categorized tokens. This token stream is then consumed by the parser."),
            ("Explain the concept of garbage collection in managed runtimes.", "Garbage collection is an automatic memory management process that identifies and reclaims heap memory occupied by objects that are no longer accessible by the application. Common approaches include tracing algorithms (mark-and-sweep) and reference counting. It prevents manual memory leaks and dangling pointer errors."),
            ("What is the difference between static typing and dynamic typing?", "Static typing checks and enforces variable types at compile time before the program runs, catching type mismatches early. Dynamic typing associates types with runtime values rather than variable declarations, checking types during execution. Statically typed languages provide stronger compile-time guarantees, while dynamically typed languages offer faster prototyping."),
        ]),
    ]

    return qa_domains, exp_domains


def assemble_and_validate_dataset():
    print("=" * 70)
    print("BUILDING & VALIDATING MIDSCALE 500-EXAMPLE DATASET")
    print("=" * 70)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    py_families = build_python_tasks()
    json_families = build_json_tasks()
    qa_domains, exp_domains = build_qa_and_explanation_tasks()

    print(f"Loaded {len(py_families)} Python task families ({len(py_families)*8} tasks)")
    print(f"Loaded {len(json_families)} JSON task families ({len(json_families)*8} tasks)")
    print(f"Loaded {len(qa_domains)} QA domains ({len(qa_domains)*5} tasks)")
    print(f"Loaded {len(exp_domains)} Explanation domains ({len(exp_domains)*5} tasks)")

    # 1. Validate all Python tasks
    print("\nValidating Python tasks with ast.parse and multi-case functional unit tests...")
    py_records = []
    py_test_specs = {}
    for fam_idx, (fam_name, task_list) in enumerate(py_families):
        for t_idx, (fn_name, inst, code_body, unit_tests) in enumerate(task_list):
            formatted_output = f"```python\n{code_body}\n```"
            # Execute validation
            test_code_snippet(formatted_output, unit_tests)
            rec_id = f"py_{fam_idx+1:02d}_{t_idx+1:02d}_{fn_name}"
            py_records.append({
                "id": rec_id,
                "category": "python",
                "task_family": fam_name,
                "instruction": inst,
                "input": "",
                "output": formatted_output,
                "family_index": fam_idx + 1,
            })
            py_test_specs[rec_id] = {
                "function_name": fn_name,
                "tests": unit_tests,
            }
    print(f"✓ All {len(py_records)} Python tasks passed AST and multi-case unit tests (normal, edge, negative).")

    # 2. Validate all JSON tasks
    print("\nValidating JSON tasks with json.loads and strict key schema validation...")
    json_records = []
    json_schema_specs = {}
    for fam_idx, (fam_name, req_keys, sample_list) in enumerate(json_families):
        for s_idx, (inp_desc, obj_target) in enumerate(sample_list):
            out_str = json.dumps(obj_target)
            # Validate
            parsed = json.loads(out_str)
            assert isinstance(parsed, dict)
            for k in req_keys:
                assert k in parsed, f"Missing key {k}"
            rec_id = f"json_{fam_idx+1:02d}_{s_idx+1:02d}_{fam_name}"
            json_records.append({
                "id": rec_id,
                "category": "json",
                "task_family": fam_name,
                "instruction": f"Generate a JSON object for {fam_name.replace('_', ' ')} with keys {', '.join(repr(k) for k in req_keys)}.",
                "input": inp_desc,
                "output": out_str,
                "family_index": fam_idx + 1,
            })
            json_schema_specs[rec_id] = {
                "required_keys": req_keys,
            }
    print(f"✓ All {len(json_records)} JSON tasks passed json.loads and schema validation.")

    # 3. Assemble QA tasks
    qa_records = []
    for dom_idx, (dom_name, qa_list) in enumerate(qa_domains):
        for q_idx, (q, a) in enumerate(qa_list):
            rec_id = f"qa_{dom_idx+1:02d}_{q_idx+1:02d}_{dom_name}"
            qa_records.append({
                "id": rec_id,
                "category": "qa",
                "task_family": dom_name,
                "instruction": q,
                "input": "",
                "output": a,
                "family_index": dom_idx + 1,
            })
    print(f"✓ Formatted {len(qa_records)} deterministic QA tasks.")

    # 4. Assemble Explanation tasks
    exp_records = []
    for dom_idx, (dom_name, exp_list) in enumerate(exp_domains):
        for e_idx, (p, exp) in enumerate(exp_list):
            rec_id = f"exp_{dom_idx+1:02d}_{e_idx+1:02d}_{dom_name}"
            exp_records.append({
                "id": rec_id,
                "category": "explanation",
                "task_family": dom_name,
                "instruction": p,
                "input": "",
                "output": exp,
                "family_index": dom_idx + 1,
            })
    print(f"✓ Formatted {len(exp_records)} structured conceptual explanation tasks.")

    all_records = py_records + json_records + qa_records + exp_records
    print(f"\nTotal assembled records: {len(all_records)} (Expected: 500)")
    assert len(all_records) == 500, f"Expected 500 records, got {len(all_records)}"

    # 5. Deduplication and leakage check
    print("\nAuditing deduplication and uniqueness...")
    seen_prompts = set()
    for r in all_records:
        key = (r["instruction"], r.get("input", ""))
        assert key not in seen_prompts, f"Duplicate prompt detected: {key}"
        seen_prompts.add(key)
    print(f"✓ 100% unique instructions and inputs ({len(seen_prompts)} distinct records).")

    # 6. Grouped Train/Validation Splitting
    # Train: Families 1-20 for Python (160) & JSON (160), Domains 1-8 for QA (40) & Exp (40) -> 400
    # Val: Families 21-25 for Python (40) & JSON (40), Domains 9-10 for QA (10) & Exp (10) -> 100
    train_records = []
    val_records = []

    for r in py_records:
        if r["family_index"] <= 20:
            train_records.append(r)
        else:
            val_records.append(r)

    for r in json_records:
        if r["family_index"] <= 20:
            train_records.append(r)
        else:
            val_records.append(r)

    for r in qa_records:
        if r["family_index"] <= 8:
            train_records.append(r)
        else:
            val_records.append(r)

    for r in exp_records:
        if r["family_index"] <= 8:
            train_records.append(r)
        else:
            val_records.append(r)

    print(f"\nGrouped Split Summary:")
    print(f"  Train: {len(train_records)} records")
    print(f"  Val:   {len(val_records)} records")
    assert len(train_records) == 400, f"Expected 400 train, got {len(train_records)}"
    assert len(val_records) == 100, f"Expected 100 val, got {len(val_records)}"

    # Verify zero family overlap
    train_families = set(r["task_family"] for r in train_records)
    val_families = set(r["task_family"] for r in val_records)
    leakage = train_families.intersection(val_families)
    assert len(leakage) == 0, f"Task family leakage detected between splits: {leakage}"
    print(f"✓ Zero task-family leakage! Train families: {len(train_families)}, Val families: {len(val_families)}, Overlap: 0.")

    # 7. Write to files
    train_file = DATA_DIR / "train.jsonl"
    val_file = DATA_DIR / "val.jsonl"
    tests_file = DATA_DIR / "unit_tests.json"
    manifest_file = DATA_DIR / "manifest.json"

    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    test_bundle = {
        "python_tests": py_test_specs,
        "json_schemas": json_schema_specs,
    }
    with open(tests_file, "w", encoding="utf-8") as f:
        json.dump(test_bundle, f, indent=2, default=lambda x: list(x) if isinstance(x, (set, tuple)) else str(x))

    train_bytes = train_file.read_bytes()
    val_bytes = val_file.read_bytes()

    manifest = {
        "dataset_name": "ModernGPT-S Midscale Diagnostic Dataset (v1 - 500 examples)",
        "version": "1.0.0",
        "total_records": 500,
        "splits": {
            "train": {
                "record_count": len(train_records),
                "sha256": hashlib.sha256(train_bytes).hexdigest(),
                "file_path": str(train_file),
                "categories": {
                    "python": sum(1 for r in train_records if r["category"] == "python"),
                    "json": sum(1 for r in train_records if r["category"] == "json"),
                    "qa": sum(1 for r in train_records if r["category"] == "qa"),
                    "explanation": sum(1 for r in train_records if r["category"] == "explanation"),
                },
                "task_families": sorted(list(train_families)),
            },
            "validation": {
                "record_count": len(val_records),
                "sha256": hashlib.sha256(val_bytes).hexdigest(),
                "file_path": str(val_file),
                "categories": {
                    "python": sum(1 for r in val_records if r["category"] == "python"),
                    "json": sum(1 for r in val_records if r["category"] == "json"),
                    "qa": sum(1 for r in val_records if r["category"] == "qa"),
                    "explanation": sum(1 for r in val_records if r["category"] == "explanation"),
                },
                "task_families": sorted(list(val_families)),
            },
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print("MANIFEST & HASH SUMMARY:")
    print(f"  train.jsonl SHA-256: {manifest['splits']['train']['sha256']}")
    print(f"  val.jsonl   SHA-256: {manifest['splits']['validation']['sha256']}")
    print(f"  Saved manifest to: {manifest_file}")
    print(f"  Saved unit test specs to: {tests_file}")
    print("=" * 70)

if __name__ == "__main__":
    assemble_and_validate_dataset()


