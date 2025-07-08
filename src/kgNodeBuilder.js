/**
 * Aggregates and deduplicates all KG nodes across files.
 * @param {Map<string, object[]>} kgNodesByFile - A map of KG nodes by file path.
 * @returns {{nodes: object[]}} The Knowledge Graph.
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

  return { nodes: allNodes };
}

module.exports = {
  buildKg,
}; 