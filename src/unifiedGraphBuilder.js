/**
 * Build a unified graph that connects FDG file nodes with KG nodes and adds intra-KG edges.
 * - Nodes: union of FDG file nodes and KG nodes
 * - Edges:
 *   - All FDG edges (file -> file imports)
 *   - File-to-KG containment edges (file -> kgNode)
 *   - Intra-KG edges (component -> props/hooks/context/states/handlers/jsx)
 *   - Route element edges (route -> component)
 *
 * @param {{nodes: object[], edges: object[]}} fdg
 * @param {{nodes: object[]}} kg
 * @param {Map<string, object[]>} kgNodesByFile
 * @returns {{nodes: object[], edges: object[]}}
 */
function buildUnifiedGraph(fdg, kg, kgNodesByFile) {
  const nodes = [];
  const edges = [];

  // Track existing node IDs to avoid duplicates
  const nodeIds = new Set();

  // 1) Add FDG file nodes (tag with nodeKind: 'File')
  fdg.nodes.forEach((fileNode) => {
    const id = fileNode.fileId;
    if (!nodeIds.has(id)) {
      nodes.push({
        nodeKind: 'File',
        id,
        ...fileNode,
      });
      nodeIds.add(id);
    }
  });

  // 2) Add KG nodes (tag with nodeKind: 'KG')
  kg.nodes.forEach((kgNode) => {
    const id = kgNode.id;
    if (!nodeIds.has(id)) {
      nodes.push({
        nodeKind: 'KG',
        id,
        ...kgNode,
      });
      nodeIds.add(id);
    }
  });

  // 3) Bring in all FDG edges as-is
  fdg.edges.forEach((e) => {
    edges.push({ ...e });
  });

  // 4) File -> KG containment edges
  for (const [fileId, fileKgNodes] of kgNodesByFile.entries()) {
    fileKgNodes.forEach((kgNode) => {
      edges.push({
        from: fileId,
        to: kgNode.id,
        type: 'fileContainsKgNode',
      });
    });
  }

  // Helper index by component name for quick lookup
  const componentIdByName = new Map();
  kg.nodes.forEach((n) => {
    if (n.type === 'Component' && n.name) {
      // Map name to possibly multiple component IDs; use array to keep all
      const existing = componentIdByName.get(n.name);
      if (existing) {
        if (Array.isArray(existing)) {
          existing.push(n.id);
        } else {
          componentIdByName.set(n.name, [existing, n.id]);
        }
      } else {
        componentIdByName.set(n.name, n.id);
      }
    }
  });

  // 5) Intra-KG edges
  // - Components to props/hooks/context from arrays on component nodes
  kg.nodes.forEach((n) => {
    if (n.type === 'Component') {
      const fromId = n.id;
      // props
      if (Array.isArray(n.props)) {
        n.props.forEach((propId) => {
          edges.push({ from: fromId, to: propId, type: 'componentHasProp' });
        });
      }
      // hooks
      if (Array.isArray(n.hooksUsed)) {
        n.hooksUsed.forEach((hookId) => {
          edges.push({ from: fromId, to: hookId, type: 'componentUsesHook' });
        });
      }
      // context
      if (Array.isArray(n.contextUsed)) {
        n.contextUsed.forEach((ctxId) => {
          edges.push({ from: fromId, to: ctxId, type: 'componentUsesContext' });
        });
      }
    }
  });

  // - Components to states/handlers/jsx using their respective declared fields
  kg.nodes.forEach((n) => {
    if (n.type === 'State') {
      const compId = n.declaredInComponentId || componentIdByName.get(n.declaredInComponent);
      if (Array.isArray(compId)) {
        compId.forEach((id) => edges.push({ from: id, to: n.id, type: 'componentDeclaresState' }));
      } else if (compId) {
        edges.push({ from: compId, to: n.id, type: 'componentDeclaresState' });
      }
    } else if (n.type === 'EventHandler') {
      const compId = n.definedInComponentId || componentIdByName.get(n.definedInComponent);
      if (Array.isArray(compId)) {
        compId.forEach((id) => edges.push({ from: id, to: n.id, type: 'componentDefinesEventHandler' }));
      } else if (compId) {
        edges.push({ from: compId, to: n.id, type: 'componentDefinesEventHandler' });
      }
    } else if (n.type === 'JSXElement') {
      const compId = n.componentId || componentIdByName.get(n.component);
      if (Array.isArray(compId)) {
        compId.forEach((id) => edges.push({ from: id, to: n.id, type: 'componentRendersJsx' }));
      } else if (compId) {
        edges.push({ from: compId, to: n.id, type: 'componentRendersJsx' });
      }
    }
  });

  // 6) Route element edges (Route -> Component) when names match
  kg.nodes.forEach((n) => {
    if (n.type === 'Route' && n.element) {
      const targetCompId = componentIdByName.get(n.element);
      if (targetCompId) {
        edges.push({ from: n.id, to: targetCompId, type: 'routeRendersComponent' });
      }
    }
  });

  // 7) Merge in intra-file KG edges built in KG
  if (Array.isArray(kg.edges)) {
    kg.edges.forEach((e) => edges.push({ ...e }));
  }

  return { nodes, edges };
}

module.exports = { buildUnifiedGraph };

