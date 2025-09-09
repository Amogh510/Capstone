package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// KGNode represents a node in the knowledge graph
type KGNode struct {
	ID           string                 `json:"id"`
	Type         string                 `json:"type"`
	Name         string                 `json:"name,omitempty"`
	FilePath     string                 `json:"filePath,omitempty"`
	Component    string                 `json:"component,omitempty"`
	ExportType   string                 `json:"exportType,omitempty"`
	DomID        string                 `json:"domId,omitempty"`
	ComponentID  string                 `json:"componentId,omitempty"`
	Expression   string                 `json:"expression,omitempty"`
	TagName      string                 `json:"tagName,omitempty"`
	Attributes   map[string]interface{} `json:"attributes,omitempty"`
	ClassNames   []string               `json:"classNames,omitempty"`
	Props        []string               `json:"props,omitempty"`
	HooksUsed    []string               `json:"hooksUsed,omitempty"`
	ContextUsed  []string               `json:"contextUsed,omitempty"`
	// Store all other fields as raw data
	Data map[string]interface{} `json:"-"`
}

// KGEdge represents an edge in the knowledge graph
type KGEdge struct {
	From      string                 `json:"from"`
	To        string                 `json:"to"`
	Type      string                 `json:"type"`
	IntraFile bool                   `json:"intraFile,omitempty"`
	FileID    string                 `json:"fileId,omitempty"`
	Data      map[string]interface{} `json:"-"`
}

// KnowledgeGraph represents the complete KG structure
type KnowledgeGraph struct {
	Nodes []json.RawMessage `json:"nodes"`
	Edges []json.RawMessage `json:"edges"`
}

// Neo4jImporter handles the import process
type Neo4jImporter struct {
	driver neo4j.DriverWithContext
	ctx    context.Context
}

// NewNeo4jImporter creates a new importer instance
func NewNeo4jImporter(uri, username, password string) (*Neo4jImporter, error) {
	driver, err := neo4j.NewDriverWithContext(uri, neo4j.BasicAuth(username, password, ""))
	if err != nil {
		return nil, fmt.Errorf("failed to create driver: %w", err)
	}

	ctx := context.Background()
	
	// Test connection
	if err := driver.VerifyConnectivity(ctx); err != nil {
		return nil, fmt.Errorf("failed to connect to Neo4j: %w", err)
	}

	return &Neo4jImporter{
		driver: driver,
		ctx:    ctx,
	}, nil
}

// Close closes the driver connection
func (ni *Neo4jImporter) Close() error {
	return ni.driver.Close(ni.ctx)
}

// SetupConstraintsAndIndexes creates necessary constraints and indexes
func (ni *Neo4jImporter) SetupConstraintsAndIndexes() error {
	session := ni.driver.NewSession(ni.ctx, neo4j.SessionConfig{})
	defer session.Close(ni.ctx)

	queries := []string{
		"CREATE CONSTRAINT kg_node_id IF NOT EXISTS FOR (n:KGNode) REQUIRE n.id IS UNIQUE",
	}

	// Try to create fulltext index (may fail if it already exists)
	fulltextQuery := "CALL db.index.fulltext.createNodeIndex('kg_fulltext', ['KGNode'], ['name', 'filePath', 'type', 'component'])"

	for _, query := range queries {
		_, err := session.Run(ni.ctx, query, nil)
		if err != nil {
			log.Printf("Warning: Failed to execute query '%s': %v", query, err)
		} else {
			log.Printf("Successfully executed: %s", query)
		}
	}

	// Try to create fulltext index separately
	_, err := session.Run(ni.ctx, fulltextQuery, nil)
	if err != nil {
		log.Printf("Warning: Failed to create fulltext index (may already exist): %v", err)
	} else {
		log.Printf("Successfully created fulltext index")
	}

	return nil
}

// ImportNodes imports nodes in batches
func (ni *Neo4jImporter) ImportNodes(nodes []KGNode) error {
	if len(nodes) == 0 {
		return nil
	}

	session := ni.driver.NewSession(ni.ctx, neo4j.SessionConfig{})
	defer session.Close(ni.ctx)

	// Convert nodes to maps for Neo4j
	nodeMaps := make([]map[string]interface{}, len(nodes))
	for i, node := range nodes {
		nodeMap := map[string]interface{}{
			"id":           node.ID,
			"type":         node.Type,
			"name":         node.Name,
			"filePath":     node.FilePath,
			"component":    node.Component,
			"exportType":   node.ExportType,
			"domId":        node.DomID,
			"componentId":  node.ComponentID,
			"expression":   node.Expression,
			"tagName":      node.TagName,
			"attributes":   node.Attributes,
			"classNames":   node.ClassNames,
			"props":        node.Props,
			"hooksUsed":    node.HooksUsed,
			"contextUsed":  node.ContextUsed,
		}

		// Add data fields, converting complex types to JSON strings
		for k, v := range node.Data {
			if _, exists := nodeMap[k]; !exists {
				// Convert complex types to JSON strings for Neo4j compatibility
				if m, ok := v.(map[string]interface{}); ok {
					if len(m) > 0 {
						if jsonBytes, err := json.Marshal(m); err == nil {
							nodeMap[k] = string(jsonBytes)
						}
					}
				} else {
					nodeMap[k] = v
				}
			}
		}
		
		// Convert attributes to JSON string if it's a complex object
		if attrs, ok := nodeMap["attributes"].(map[string]interface{}); ok && len(attrs) > 0 {
			if jsonBytes, err := json.Marshal(attrs); err == nil {
				nodeMap["attributes"] = string(jsonBytes)
			}
		}

		// Remove empty fields and empty maps/slices
		for k, v := range nodeMap {
			if v == nil || v == "" || (fmt.Sprintf("%v", v) == "[]") || (fmt.Sprintf("%v", v) == "map[]") {
				delete(nodeMap, k)
			}
			// Remove empty maps
			if m, ok := v.(map[string]interface{}); ok && len(m) == 0 {
				delete(nodeMap, k)
			}
			// Remove empty slices
			if s, ok := v.([]interface{}); ok && len(s) == 0 {
				delete(nodeMap, k)
			}
			if s, ok := v.([]string); ok && len(s) == 0 {
				delete(nodeMap, k)
			}
		}

		nodeMaps[i] = nodeMap
	}

	query := `
		UNWIND $batch AS node
		MERGE (n:KGNode {id: node.id})
		SET n += node
	`

	_, err := session.Run(ni.ctx, query, map[string]interface{}{
		"batch": nodeMaps,
	})

	if err != nil {
		return fmt.Errorf("failed to import nodes batch: %w", err)
	}

	return nil
}

// ImportEdges imports edges in batches
func (ni *Neo4jImporter) ImportEdges(edges []KGEdge) error {
	if len(edges) == 0 {
		return nil
	}

	session := ni.driver.NewSession(ni.ctx, neo4j.SessionConfig{})
	defer session.Close(ni.ctx)

	// Convert edges to maps
	edgeMaps := make([]map[string]interface{}, len(edges))
	for i, edge := range edges {
		edgeMap := map[string]interface{}{
			"from":      edge.From,
			"to":        edge.To,
			"type":      edge.Type,
			"intraFile": edge.IntraFile,
			"fileId":    edge.FileID,
		}

		// Add data fields, converting complex types to JSON strings
		for k, v := range edge.Data {
			if _, exists := edgeMap[k]; !exists {
				// Convert complex types to JSON strings for Neo4j compatibility
				if m, ok := v.(map[string]interface{}); ok {
					if len(m) > 0 {
						if jsonBytes, err := json.Marshal(m); err == nil {
							edgeMap[k] = string(jsonBytes)
						}
					}
				} else {
					edgeMap[k] = v
				}
			}
		}

		// Remove empty fields and empty maps/slices
		for k, v := range edgeMap {
			if v == nil || v == "" || (fmt.Sprintf("%v", v) == "[]") || (fmt.Sprintf("%v", v) == "map[]") {
				delete(edgeMap, k)
			}
			// Remove empty maps
			if m, ok := v.(map[string]interface{}); ok && len(m) == 0 {
				delete(edgeMap, k)
			}
			// Remove empty slices
			if s, ok := v.([]interface{}); ok && len(s) == 0 {
				delete(edgeMap, k)
			}
			if s, ok := v.([]string); ok && len(s) == 0 {
				delete(edgeMap, k)
			}
		}

		edgeMaps[i] = edgeMap
	}

	query := `
		UNWIND $batch AS rel
		MATCH (from:KGNode {id: rel.from}), (to:KGNode {id: rel.to})
		MERGE (from)-[r:KGR {type: rel.type, from: rel.from, to: rel.to}]->(to)
		SET r += rel
	`

	_, err := session.Run(ni.ctx, query, map[string]interface{}{
		"batch": edgeMaps,
	})

	if err != nil {
		return fmt.Errorf("failed to import edges batch: %w", err)
	}

	return nil
}

// parseNodeWithFallback parses a node with fallback to raw data
func parseNodeWithFallback(rawNode json.RawMessage) (KGNode, error) {
	var node KGNode
	
	// First parse into the struct
	if err := json.Unmarshal(rawNode, &node); err != nil {
		return node, err
	}

	// Then parse into a map to capture any additional fields
	var rawData map[string]interface{}
	if err := json.Unmarshal(rawNode, &rawData); err != nil {
		return node, err
	}

	// Store additional fields in Data
	node.Data = make(map[string]interface{})
	knownFields := map[string]bool{
		"id": true, "type": true, "name": true, "filePath": true,
		"component": true, "exportType": true, "domId": true, "componentId": true,
		"expression": true, "tagName": true, "attributes": true, "classNames": true,
		"props": true, "hooksUsed": true, "contextUsed": true,
	}

	for k, v := range rawData {
		if !knownFields[k] {
			node.Data[k] = v
		}
	}

	return node, nil
}

// parseEdgeWithFallback parses an edge with fallback to raw data
func parseEdgeWithFallback(rawEdge json.RawMessage) (KGEdge, error) {
	var edge KGEdge
	
	// First parse into the struct
	if err := json.Unmarshal(rawEdge, &edge); err != nil {
		return edge, err
	}

	// Then parse into a map to capture any additional fields
	var rawData map[string]interface{}
	if err := json.Unmarshal(rawEdge, &rawData); err != nil {
		return edge, err
	}

	// Store additional fields in Data
	edge.Data = make(map[string]interface{})
	knownFields := map[string]bool{
		"from": true, "to": true, "type": true, "intraFile": true, "fileId": true,
	}

	for k, v := range rawData {
		if !knownFields[k] {
			edge.Data[k] = v
		}
	}

	return edge, nil
}

// ImportFromFile imports the KG from a JSON file
func (ni *Neo4jImporter) ImportFromFile(filename string, batchSize int) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	decoder := json.NewDecoder(file)

	// Read opening brace
	token, err := decoder.Token()
	if err != nil {
		return fmt.Errorf("failed to read opening brace: %w", err)
	}
	if token != json.Delim('{') {
		return fmt.Errorf("expected opening brace, got %v", token)
	}

	var totalNodes, totalEdges int
	start := time.Now()

	for decoder.More() {
		// Read the key
		key, err := decoder.Token()
		if err != nil {
			return fmt.Errorf("failed to read key: %w", err)
		}

		keyStr, ok := key.(string)
		if !ok {
			return fmt.Errorf("expected string key, got %T", key)
		}

		switch keyStr {
		case "nodes":
			log.Println("Processing nodes...")
			totalNodes, err = ni.processNodesArray(decoder, batchSize)
			if err != nil {
				return fmt.Errorf("failed to process nodes: %w", err)
			}
			log.Printf("Imported %d nodes", totalNodes)

		case "edges":
			log.Println("Processing edges...")
			totalEdges, err = ni.processEdgesArray(decoder, batchSize)
			if err != nil {
				return fmt.Errorf("failed to process edges: %w", err)
			}
			log.Printf("Imported %d edges", totalEdges)

		default:
			// Skip unknown fields
			var ignored interface{}
			if err := decoder.Decode(&ignored); err != nil {
				return fmt.Errorf("failed to skip field %s: %w", keyStr, err)
			}
		}
	}

	// Read closing brace
	token, err = decoder.Token()
	if err != nil {
		return fmt.Errorf("failed to read closing brace: %w", err)
	}
	if token != json.Delim('}') {
		return fmt.Errorf("expected closing brace, got %v", token)
	}

	duration := time.Since(start)
	log.Printf("Import completed in %v: %d nodes, %d edges", duration, totalNodes, totalEdges)

	return nil
}

// processNodesArray processes the nodes array with batching
func (ni *Neo4jImporter) processNodesArray(decoder *json.Decoder, batchSize int) (int, error) {
	// Read opening bracket
	token, err := decoder.Token()
	if err != nil {
		return 0, err
	}
	if token != json.Delim('[') {
		return 0, fmt.Errorf("expected opening bracket for nodes array")
	}

	var batch []KGNode
	totalProcessed := 0

	for decoder.More() {
		var rawNode json.RawMessage
		if err := decoder.Decode(&rawNode); err != nil {
			return totalProcessed, err
		}

		node, err := parseNodeWithFallback(rawNode)
		if err != nil {
			log.Printf("Warning: Failed to parse node: %v", err)
			continue
		}

		batch = append(batch, node)

		if len(batch) >= batchSize {
			if err := ni.ImportNodes(batch); err != nil {
				return totalProcessed, err
			}
			totalProcessed += len(batch)
			log.Printf("Imported %d nodes...", totalProcessed)
			batch = batch[:0] // Reset batch
		}
	}

	// Import remaining nodes
	if len(batch) > 0 {
		if err := ni.ImportNodes(batch); err != nil {
			return totalProcessed, err
		}
		totalProcessed += len(batch)
	}

	// Read closing bracket
	token, err = decoder.Token()
	if err != nil {
		return totalProcessed, err
	}
	if token != json.Delim(']') {
		return totalProcessed, fmt.Errorf("expected closing bracket for nodes array")
	}

	return totalProcessed, nil
}

// processEdgesArray processes the edges array with batching
func (ni *Neo4jImporter) processEdgesArray(decoder *json.Decoder, batchSize int) (int, error) {
	// Read opening bracket
	token, err := decoder.Token()
	if err != nil {
		return 0, err
	}
	if token != json.Delim('[') {
		return 0, fmt.Errorf("expected opening bracket for edges array")
	}

	var batch []KGEdge
	totalProcessed := 0

	for decoder.More() {
		var rawEdge json.RawMessage
		if err := decoder.Decode(&rawEdge); err != nil {
			return totalProcessed, err
		}

		edge, err := parseEdgeWithFallback(rawEdge)
		if err != nil {
			log.Printf("Warning: Failed to parse edge: %v", err)
			continue
		}

		batch = append(batch, edge)

		if len(batch) >= batchSize {
			if err := ni.ImportEdges(batch); err != nil {
				return totalProcessed, err
			}
			totalProcessed += len(batch)
			log.Printf("Imported %d edges...", totalProcessed)
			batch = batch[:0] // Reset batch
		}
	}

	// Import remaining edges
	if len(batch) > 0 {
		if err := ni.ImportEdges(batch); err != nil {
			return totalProcessed, err
		}
		totalProcessed += len(batch)
	}

	// Read closing bracket
	token, err = decoder.Token()
	if err != nil {
		return totalProcessed, err
	}
	if token != json.Delim(']') {
		return totalProcessed, fmt.Errorf("expected closing bracket for edges array")
	}

	return totalProcessed, nil
}

func main() {
	// Configuration
	neo4jURI := getEnvOrDefault("NEO4J_URI", "bolt://localhost:7687")
	neo4jUser := getEnvOrDefault("NEO4J_USER", "neo4j")
	neo4jPassword := getEnvOrDefault("NEO4J_PASSWORD", "password")
	kgFile := getEnvOrDefault("KG_FILE", "../src/output/kg.json")
	batchSize := 1000

	log.Printf("Starting Neo4j import from %s", kgFile)
	log.Printf("Neo4j URI: %s", neo4jURI)
	log.Printf("Batch size: %d", batchSize)

	// Create importer
	importer, err := NewNeo4jImporter(neo4jURI, neo4jUser, neo4jPassword)
	if err != nil {
		log.Fatalf("Failed to create importer: %v", err)
	}
	defer importer.Close()

	// Setup constraints and indexes
	log.Println("Setting up constraints and indexes...")
	if err := importer.SetupConstraintsAndIndexes(); err != nil {
		log.Printf("Warning: Failed to setup constraints/indexes: %v", err)
	}

	// Import the KG
	if err := importer.ImportFromFile(kgFile, batchSize); err != nil {
		log.Fatalf("Import failed: %v", err)
	}

	log.Println("Import completed successfully!")
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
