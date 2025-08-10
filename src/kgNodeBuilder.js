/**
 * Aggregates and deduplicates all KG nodes across files and returns nodes and intra-file edges.
 * @param {Map<string, object[]>} kgNodesByFile - A map of KG nodes by file path.
 * @returns {{nodes: object[], edges: object[]}} The Knowledge Graph with intra-file edges.
 */
function buildKg(kgNodesByFile) {
  const allNodes = [];
  const nodeIds = new Set();

  for (const nodes of kgNodesByFile.values()) {
    nodes.forEach((node) => {
      if (!nodeIds.has(node.id)) {
        allNodes.push(node);
        nodeIds.add(node.id);
      }
    });
  }

  // Build intra-file KG edges
  const { buildKgEdges } = require('./kgEdgeBuilder');
  const edges = buildKgEdges(kgNodesByFile);

  return { nodes: allNodes, edges };
}

module.exports = {
  buildKg,
}; 