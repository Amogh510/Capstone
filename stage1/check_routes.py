import os
import json
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", os.getenv("NEO4J_PASSWORD", "admin123"))

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

print("=" * 60)
print("CHECKING ROUTE NODES IN NEO4J")
print("=" * 60)

with driver.session() as session:
    # Check all node types
    print("\n1. All node types in database:")
    result = session.run("MATCH (n:KGNode) RETURN DISTINCT n.type AS type, count(*) AS count ORDER BY count DESC")
    for record in result:
        print(f"   {record['type']}: {record['count']} nodes")
    
    # Check for Route nodes specifically
    print("\n2. Route nodes (all properties):")
    result = session.run("MATCH (n:KGNode {type: 'Route'}) RETURN n LIMIT 10")
    routes = list(result)
    if not routes:
            print("   ❌ NO ROUTE NODES FOUND!")
            
            # Check what other types might be route-like
            print("\n3. Checking for route-like nodes (path, url, etc):")
            result = session.run("""
                MATCH (n:KGNode)
                WHERE n.name CONTAINS '/' OR n.filePath CONTAINS 'route' OR n.type CONTAINS 'Route'
                RETURN n.type AS type, n.name AS name, n.filePath AS filePath
                LIMIT 20
            """)
            for record in result:
                print(f"   Type: {record['type']}, Name: {record['name']}, Path: {record['filePath']}")
    else:
        print(f"   ✅ Found {len(routes)} Route nodes:")
        for i, record in enumerate(routes, 1):
            node = dict(record['n'])
            print(f"\n   Route #{i}:")
            for key, value in node.items():
                if key == 'embedding':
                    print(f"     {key}: <vector of length {len(value) if isinstance(value, list) else 'unknown'}>")
                else:
                    print(f"     {key}: {value}")
        
        # Check sample of all nodes to understand structure
    print("\n4. Sample nodes from database (first 5):")
    result = session.run("MATCH (n:KGNode) RETURN n LIMIT 5")
    for i, record in enumerate(result, 1):
        node = dict(record['n'])
        print(f"\n   Node #{i} (type: {node.get('type', 'unknown')}):")
        for key in ['id', 'type', 'name', 'filePath', 'component', 'exportType']:
            if key in node:
                print(f"     {key}: {node[key]}")

driver.close()